"""SPINN + growth-aware (adjoint/goal-oriented, kb1 113) residual weight.
Change vs parent (b03_spec, GAMMA 3->8): the full-grid PDE residual is FFT'd in x and
each mode weighted by exp(GAMMA*sigma+(k)/smax*(T-t)/T) with sigma(k)=
V2 k^2-V3 k^4 -> emphasises the exponentially-amplified unstable band
(k3-16) at early t, where residual-L2 is blind. Floor 1 keeps all modes
controlled. Everything else identical to b03_spinn.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 2
OUTPUT_DIM   = 1
INPUT_NAMES  = ("t", "x")
FIELD_NAMES  = ("u",)

# --- Domain ---
T_MIN, T_MAX = 0.0, 0.4
X_MIN, X_MAX = 0.0, 6.283185307179586   # 2*pi
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 4   # u_xxxx
N_RESIDUAL_COMPONENTS   = 1
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi")
PERIODIC_AXES  = ("x",)       # u(t, X_MIN) = u(t, X_MAX)
GEOMETRY_KIND  = "periodic_strip"

# --- Physical parameters (scaled KS: u_t + V1 u u_x + V2 u_xx + V3 u_xxxx = 0) ---
V1 = 100.0 / 16.0        # 6.25
V2 = 100.0 / 16.0**2     # 0.390625
V3 = 100.0 / 16.0**4     # ~1.5259e-3

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 300.0     # total wall-clock seconds for the WHOLE process,
                       # from _T0: imports, JIT compile, training and param
                       # save all included — eval.py kills the process at
                       # this wall; return params with margin to spare
MAX_ITER   = 1_000_000 # safety cap on the loop (not expected to bind)

# --- Eval gate: full fidelity only under task/eval.py ---
if _os.environ.get("FORGE_EVAL_TOKEN") != "task/eval.py":
    TRAIN_TIME = 10.0                                # off-eval runs are smoke:
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")   # CPU, 10 s
# ═══════════════════════════════════════════════════════════════

import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jit, random, vmap, jacfwd
import numpy as np
import optax
from flax import linen as nn


def _locate(name: str) -> Path:
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Separable sub-networks ══════════

RANK    = 128   # CP rank r: u = sum_{j=1..r} f_j(t) g_j(x)
WIDTH   = 64
DEPTH   = 3     # hidden layers per sub-net
M_MODES = 16    # periodic Fourier modes in x (KS unstable band up to k~16)


class TimeNet(nn.Module):
    """1D MLP  t -> R^r  (input scaled t/T_MAX)."""
    rank: int = RANK
    width: int = WIDTH
    depth: int = DEPTH

    @nn.compact
    def __call__(self, t):           # t: [N,1]
        kinit = jax.nn.initializers.glorot_normal()
        h = t / T_MAX
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h))
        return nn.Dense(self.rank, kernel_init=kinit)(h)   # [N, r]


class SpaceNet(nn.Module):
    """1D MLP on an exact periodic Fourier embedding of x -> R^r.
    Feature map [cos(kx), sin(kx)]_{k=1..M} makes g_j and all its
    x-derivatives exactly 2*pi-periodic (kb1 012/084)."""
    rank: int = RANK
    width: int = WIDTH
    depth: int = DEPTH
    modes: int = M_MODES

    @nn.compact
    def __call__(self, x):           # x: [N,1]
        kinit = jax.nn.initializers.glorot_normal()
        ks = jnp.arange(1, self.modes + 1, dtype=x.dtype)
        xk = x * ks                                       # [N, M]
        h = jnp.concatenate([jnp.cos(xk), jnp.sin(xk)], axis=-1)
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h))
        return nn.Dense(self.rank, kernel_init=kinit)(h)  # [N, r]


time_model  = TimeNet()
space_model = SpaceNet()


# ══════════ Data ══════════

def _build_dataset():
    nt, nx = 101, 512
    ts = np.linspace(T_MIN, T_MAX, nt)[1:].astype(np.float32)      # interior time (exclude IC line)
    xs = np.linspace(X_MIN, X_MAX, nx, endpoint=False).astype(np.float32)

    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0].astype(np.float32), ic[:, 1].astype(np.float32)

    return jnp.array(ts), jnp.array(xs), jnp.array(x_ic), jnp.array(u_ic)


# ══════════ Hyperparameters ══════════

LR        = 1e-3
WEIGHT_IC = 100.0
SAVE_MARGIN_S = 8.0
GAMMA     = 8.0    # unstable-band residual emphasis strength


# ══════════ Training ══════════

def _make_train(rng_init):
    ts, xs, x_ic, u_ic = _build_dataset()
    Nx = xs.shape[0]

    # growth-aware spectral residual weight w[t,k] (kb1 113 adjoint idea):
    # integer wavenumbers of the uniform x-grid; sigma(k)=V2 k^2 - V3 k^4.
    kfull = np.round(np.fft.fftfreq(Nx, d=(X_MAX - X_MIN) / Nx) * (X_MAX - X_MIN))
    kabs = np.abs(kfull)
    sig = V2 * kabs**2 - V3 * kabs**4
    sig_plus = np.maximum(sig, 0.0)
    smax = float(sig_plus.max())
    tnp = np.asarray(ts)
    tfac = (T_MAX - tnp) / T_MAX                       # [Nt] early t -> 1
    W = np.exp(GAMMA * (sig_plus[None, :] / smax) * tfac[:, None])  # [Nt, Nx]
    W = jnp.asarray(W.astype(np.float32))

    k1, k2, rng = random.split(rng_init, 3)
    pf = time_model.init(k1, jnp.zeros([1, 1]))
    pg = space_model.init(k2, jnp.zeros([1, 1]))
    params = {"f": pf, "g": pg}

    # scalar sub-net evaluations (input scalar -> [r])
    def f_vec(pf, t):
        return time_model.apply(pf, jnp.reshape(t, (1, 1)))[0]      # [r]

    def g_vec(pg, x):
        return space_model.apply(pg, jnp.reshape(x, (1, 1)))[0]     # [r]

    # time: value + d/dt   (forward-mode jvp, one pass)
    def f_and_dt(pf, t):
        f, df = jax.jvp(lambda tt: f_vec(pf, tt), (t,), (jnp.ones_like(t),))
        return f, df

    # space: value + x-derivatives up to 4th (nested forward-mode)
    g0 = lambda pg, x: g_vec(pg, x)
    g1 = jacfwd(g0, argnums=1)
    g2 = jacfwd(g1, argnums=1)
    g3 = jacfwd(g2, argnums=1)
    g4 = jacfwd(g3, argnums=1)

    def loss_fn(params):
        pf, pg = params["f"], params["g"]

        F, dF = vmap(lambda t: f_and_dt(pf, t))(ts)      # [Nt, r] each
        G  = vmap(lambda x: g0(pg, x))(xs)               # [Nx, r]
        G1 = vmap(lambda x: g1(pg, x))(xs)
        G2 = vmap(lambda x: g2(pg, x))(xs)
        G4 = vmap(lambda x: g4(pg, x))(xs)

        U     = F  @ G.T                                  # [Nt, Nx]
        U_t   = dF @ G.T
        U_x   = F  @ G1.T
        U_xx  = F  @ G2.T
        U_xxxx = F @ G4.T
        r = U_t + V1 * U * U_x + V2 * U_xx + V3 * U_xxxx   # [Nt, Nx]
        # growth-aware weighting in x-Fourier space; W==1 recovers mean(r^2)
        Rh = jnp.fft.fft(r, axis=1)
        Pk = Rh.real ** 2 + Rh.imag ** 2                  # [Nt, Nx]
        pde_loss = jnp.mean(jnp.sum(W * Pk, axis=1)) / (Nx ** 2)

        # IC at t=0:  u(0,x) = f(0) . g(x)
        f0 = f_vec(pf, jnp.array(0.0, dtype=ts.dtype))    # [r]
        G_ic = vmap(lambda x: g0(pg, x))(x_ic)            # [Nx_ic, r]
        u_ic_pred = G_ic @ f0                             # [Nx_ic]
        ic_loss = jnp.mean((u_ic_pred - u_ic) ** 2)

        return pde_loss + WEIGHT_IC * ic_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(params)

    @jit
    def update(params, opt_state):
        loss, grads = loss_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return params, opt_state, update, rng


# ══════════ Public contract ══════════

def predict_fn(params, X):
    """X: [BS,2] columns (t,x). u_i = sum_j f_j(t_i) g_j(x_i)."""
    t = X[:, 0:1]
    x = X[:, 1:2]
    F = time_model.apply(params["f"], t)      # [BS, r]
    G = space_model.apply(params["g"], x)     # [BS, r]
    u = jnp.sum(F * G, axis=1)                # [BS]
    return {"u": u}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, opt_state, update, rng_local = _make_train(rng)

    # warmup step (JIT compile) — a real, counted step
    params, opt_state, loss = update(params, opt_state)
    step_count = 1
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        params, opt_state, loss = update(params, opt_state)
        step_count = it + 1
        last_loss = float(loss)
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(params, it, last_loss)

    return params, step_count
