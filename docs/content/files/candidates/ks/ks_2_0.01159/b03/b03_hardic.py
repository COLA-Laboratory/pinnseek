"""SPINN with HARD initial condition: u(t,x)=IC(x)+t̂·Σ_r g_r(t)h_r(x), t̂=(t-T0)/(T1-T0).
u(0,x)=IC(x) exactly (Fourier interpolant of ks_ic.csv, analytic x-derivs), removing the
IC/PDE tension that keeps the t≈0 residual ~50× high (diag). No IC loss. Parent: b03_spinn.
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
from jax import jit, random
import jax.flatten_util
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


# ══════════ Hyperparameters ══════════

MX        = 16        # exact-periodic Fourier modes in x (network body)
KIC       = 48        # Fourier modes for the analytic IC interpolant
RANK      = 128
WIDTH     = 96
DEPTH     = 3
NT        = 64
NX        = 128
LR0       = 1.8e-3
DECAY_STEPS = 30000
EPS_SCHED = (1e-2, 1e-1, 1e0, 1e1, 1e2)
W_DELTA   = 0.99
W_FLOOR   = 0.03
SAVE_MARGIN_S = 6.0
ST        = 1.0 / (T_MAX - T_MIN)   # d(t̂)/dt


# ══════════ Analytic IC Fourier interpolant (from ks_ic.csv) ══════════

def _ic_coeffs():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    u = ic[:, 1]
    # 512 pts on [0,2π] inclusive → drop duplicate endpoint, use 511 on [0,2π)
    us = u[:-1]
    N = len(us)
    C = np.fft.rfft(us)
    a = np.zeros(KIC + 1); b = np.zeros(KIC + 1)
    a[0] = (C[0].real) / N
    for k in range(1, KIC + 1):
        a[k] = 2.0 * C[k].real / N
        b[k] = -2.0 * C[k].imag / N
    return a.astype(np.float32), b.astype(np.float32)


_A_IC, _B_IC = _ic_coeffs()
_KS_IC = np.arange(0, KIC + 1, dtype=np.float32)


def _ic_series(x, order):
    """IC^(order)(x). x: [N,1] → [N,1]. Analytic truncated Fourier series."""
    k = jnp.asarray(_KS_IC)                 # [KIC+1]
    a = jnp.asarray(_A_IC); b = jnp.asarray(_B_IC)
    kx = x * k                              # [N, KIC+1]
    c, s = jnp.cos(kx), jnp.sin(kx)
    o = order % 4
    # (d/dx)^order of [a_k cos(kx)+b_k sin(kx)] = k^order · rotate(a,b), sign by o
    km = k ** order
    if o == 0:
        val = a * c + b * s
    elif o == 1:
        val = -a * s + b * c
    elif o == 2:
        val = -a * c - b * s
    else:
        val = a * s - b * c
    return jnp.sum(km * val, axis=1, keepdims=True)   # [N,1]


# ══════════ Model: separable network body ══════════

def _embed_x(x):
    ks = jnp.arange(1, MX + 1, dtype=jnp.float32)
    return jnp.concatenate([jnp.cos(x * ks), jnp.sin(x * ks)], axis=1)


def _embed_t(t):
    return (t - T_MIN) / (T_MAX - T_MIN)


class AxisBody(nn.Module):
    embed: Callable
    width: int = WIDTH
    depth: int = DEPTH
    rank: int = RANK

    @nn.compact
    def __call__(self, z):
        kinit = jax.nn.initializers.glorot_normal()
        h0 = self.embed(z)
        U = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h0))
        V = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h0))
        H = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h0))
        for _ in range(self.depth):
            Z = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(H))
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(self.rank, kernel_init=kinit)(H)


class SPINN(nn.Module):
    def setup(self):
        self.G = AxisBody(_embed_t, name="G")
        self.H = AxisBody(_embed_x, name="H")

    def __call__(self, X):                 # pointwise [N,2] → [N] (hard IC applied)
        t = X[:, 0:1]; x = X[:, 1:2]
        P = jnp.sum(self.G(t) * self.H(x), axis=1, keepdims=True)   # [N,1]
        u = _ic_series(x, 0) + _embed_t(t) * P
        return u[:, 0]

    def g_feats(self, t):
        return self.G(t)

    def h_feats(self, x):
        return self.H(x)


def _make_train(rng_init):
    model = SPINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def g_of(pf, t_col):
        return model.apply(unravel(pf), t_col, method=SPINN.g_feats)

    def h_of(pf, x_col):
        return model.apply(unravel(pf), x_col, method=SPINN.h_feats)

    def _d(f):
        return lambda col: jax.jvp(f, (col,), (jnp.ones_like(col),))[1]

    def slice_losses(pf, t_col, x_col):
        """KS residual on [NT,NX] with hard-IC ansatz; per-slice mean-square [NT]."""
        gf = lambda tc: g_of(pf, tc)
        hf = lambda xc: h_of(pf, xc)
        g = gf(t_col); gt = _d(gf)(t_col)          # [NT,R]
        h = hf(x_col)
        hx = _d(hf); hxx = _d(hx); hxxx = _d(hxx); hxxxx = _d(hxxx)
        h1 = hx(x_col); h2 = hxx(x_col); h4 = hxxxx(x_col)   # [NX,R]
        P    = jnp.einsum('tr,xr->tx', g, h)
        P_t  = jnp.einsum('tr,xr->tx', gt, h)
        P_x  = jnp.einsum('tr,xr->tx', g, h1)
        P_xx = jnp.einsum('tr,xr->tx', g, h2)
        P_xxxx = jnp.einsum('tr,xr->tx', g, h4)
        tt = _embed_t(t_col)                        # [NT,1] = t̂
        IC0 = _ic_series(x_col, 0)[:, 0][None, :]   # [1,NX]
        IC1 = _ic_series(x_col, 1)[:, 0][None, :]
        IC2 = _ic_series(x_col, 2)[:, 0][None, :]
        IC4 = _ic_series(x_col, 4)[:, 0][None, :]
        u      = IC0 + tt * P
        u_t    = ST * P + tt * P_t
        u_x    = IC1 + tt * P_x
        u_xx   = IC2 + tt * P_xx
        u_xxxx = IC4 + tt * P_xxxx
        r = u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx
        return jnp.mean(r ** 2, axis=1)             # [NT]

    def residual_loss(pf, t_col, x_col, eps):
        Lr = slice_losses(pf, t_col, x_col)
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(Lr)[:-1]])
        w = jax.lax.stop_gradient(jnp.maximum(jnp.exp(-eps * cum), W_FLOOR))
        return jnp.mean(w * Lr), w

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR0, warmup_steps=500,
        decay_steps=DECAY_STEPS, end_value=1e-5)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(flat_params)

    loss_grad = jit(jax.value_and_grad(lambda pf, tc, xc, eps: residual_loss(pf, tc, xc, eps),
                                       has_aux=True))

    @jit
    def sample(rng, tmax):
        r1, r2 = random.split(rng, 2)
        ts = jnp.sort(random.uniform(r1, (NT, 1), minval=1e-4, maxval=tmax))
        xs = random.uniform(r2, (NX, 1), minval=X_MIN, maxval=X_MAX)
        return ts, xs

    @jit
    def step(pf, opt_state, rng, eps, tmax):
        rng, sk = random.split(rng)
        t_col, x_col = sample(sk, tmax)
        (loss, w), g = loss_grad(pf, t_col, x_col, eps)
        updates, opt_state = optimizer.update(g, opt_state)
        pf = optax.apply_updates(pf, updates)
        return pf, opt_state, rng, w, loss

    return flat_params, opt_state, step, unravel, rng


_PREDICT_MODEL = SPINN()


def predict_fn(params, X):
    return {"u": _PREDICT_MODEL.apply(params, X)}


TMAX_START = 0.15
RAMP_FRAC  = 0.55


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    pf, opt_state, step, unravel, rng = _make_train(rng)

    eps_idx = 0
    eps = jnp.asarray(EPS_SCHED[eps_idx], dtype=jnp.float32)

    def tmax_at(now):
        prog = min(1.0, (now - _T0) / (RAMP_FRAC * TRAIN_TIME))
        return jnp.asarray(TMAX_START + (T_MAX - TMAX_START) * prog, dtype=jnp.float32)

    tmax = tmax_at(time.time())
    pf, opt_state, rng, w, loss = step(pf, opt_state, rng, eps, tmax)
    step_count = 1

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        now = time.time()
        if now > deadline:
            break
        tmax = tmax_at(now)
        pf, opt_state, rng, w, loss = step(pf, opt_state, rng, eps, tmax)
        step_count = it + 1

        if it % 500 == 0:
            if float(jnp.min(w)) > W_DELTA and eps_idx < len(EPS_SCHED) - 1:
                eps_idx += 1
                eps = jnp.asarray(EPS_SCHED[eps_idx], dtype=jnp.float32)

        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(pf), it, float(loss))

    return unravel(pf), step_count
