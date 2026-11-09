"""Sequential time-marching SPINN for KS.
Change vs parent (b03_march): N_WIN 3->6 windows, each a
separate SPINN warm-started from the previous window; window w's IC is the
previous window's endpoint field (handoff). Rationale (b03 diag): the field
grows correctly until t~0.1 then decays -> short windows preserve growth;
SPINN throughput finally affords ~90k steps/window (b02 marching failed on
handoff with ~12k). Periodic x embedding kept.
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

RANK    = 128
WIDTH   = 64
DEPTH   = 3
M_MODES = 16
N_WIN   = 6          # time windows
NT_WIN  = 48         # interior collocation times per window
NX      = 512


class TimeNet(nn.Module):
    """1D MLP  s -> R^r, s in [0,1] the window-local normalised time."""
    rank: int = RANK
    width: int = WIDTH
    depth: int = DEPTH

    @nn.compact
    def __call__(self, s):           # s: [N,1] in [0,1]
        kinit = jax.nn.initializers.glorot_normal()
        h = s
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h))
        return nn.Dense(self.rank, kernel_init=kinit)(h)


class SpaceNet(nn.Module):
    """1D MLP on an exact periodic Fourier embedding of x -> R^r."""
    rank: int = RANK
    width: int = WIDTH
    depth: int = DEPTH
    modes: int = M_MODES

    @nn.compact
    def __call__(self, x):           # x: [N,1]
        kinit = jax.nn.initializers.glorot_normal()
        ks = jnp.arange(1, self.modes + 1, dtype=x.dtype)
        xk = x * ks
        h = jnp.concatenate([jnp.cos(xk), jnp.sin(xk)], axis=-1)
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h))
        return nn.Dense(self.rank, kernel_init=kinit)(h)


time_model  = TimeNet()
space_model = SpaceNet()

WEIGHT_IC = 100.0
LR        = 1e-3
SAVE_MARGIN_S = 10.0


# ══════════ per-window predict helpers ══════════

def _F(pf, s):    # s:[N,1] -> [N,r]
    return time_model.apply(pf, s)

def _G(pg, x):    # x:[N,1] -> [N,r]
    return space_model.apply(pg, x)


def _win_predict(win, s, x):
    """u for arrays s (local time in [0,1]) and x (global), same shape [N]."""
    F = _F(win["f"], s.reshape(-1, 1))
    G = _G(win["g"], x.reshape(-1, 1))
    return jnp.sum(F * G, axis=1)


# ══════════ window training ══════════

def _train_window(pf, pg, dt, xs, ic_x, ic_u, deadline, rng):
    """Train one window's SPINN (local time s in [0,1], physical span dt).
    ic_x/ic_u: IC target at s=0. Returns (pf,pg,steps)."""
    ss = jnp.asarray(np.linspace(0.0, 1.0, NT_WIN + 1)[1:].astype(np.float32))  # interior s
    params = {"f": pf, "g": pg}

    def f_vec(pf, s):
        return time_model.apply(pf, jnp.reshape(s, (1, 1)))[0]

    def g_vec(pg, x):
        return space_model.apply(pg, jnp.reshape(x, (1, 1)))[0]

    def f_and_ds(pf, s):
        f, df = jax.jvp(lambda z: f_vec(pf, z), (s,), (jnp.ones_like(s),))
        return f, df

    g0 = lambda pg, x: g_vec(pg, x)
    g1 = jacfwd(g0, argnums=1)
    g2 = jacfwd(g1, argnums=1)
    g3 = jacfwd(g2, argnums=1)
    g4 = jacfwd(g3, argnums=1)

    def loss_fn(params):
        pf, pg = params["f"], params["g"]
        F, dFds = vmap(lambda s: f_and_ds(pf, s))(ss)     # [Nt,r]
        G  = vmap(lambda x: g0(pg, x))(xs)
        G1 = vmap(lambda x: g1(pg, x))(xs)
        G2 = vmap(lambda x: g2(pg, x))(xs)
        G4 = vmap(lambda x: g4(pg, x))(xs)

        U     = F    @ G.T
        U_t   = (dFds @ G.T) / dt                          # chain rule ds/dt=1/dt
        U_x   = F    @ G1.T
        U_xx  = F    @ G2.T
        U_xxxx = F   @ G4.T
        r = U_t + V1 * U * U_x + V2 * U_xx + V3 * U_xxxx
        pde_loss = jnp.mean(r ** 2)

        f0 = f_vec(pf, jnp.array(0.0, dtype=ss.dtype))     # s=0
        G_ic = vmap(lambda x: g0(pg, x))(ic_x)
        u_ic_pred = G_ic @ f0
        ic_loss = jnp.mean((u_ic_pred - ic_u) ** 2)
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

    params, opt_state, loss = update(params, opt_state)   # warmup/compile
    steps = 1
    while time.time() < deadline:
        params, opt_state, loss = update(params, opt_state)
        steps += 1
    return params["f"], params["g"], steps


# ══════════ Public contract ══════════

def predict_fn(params, X):
    bounds = params["bounds"]        # tuple length N_WIN+1
    wins = params["win"]
    t = X[:, 0]
    x = X[:, 1]
    u = jnp.zeros_like(t)
    for w in range(len(wins)):
        t_lo = bounds[w]
        t_hi = bounds[w + 1]
        dt = t_hi - t_lo
        s = jnp.clip((t - t_lo) / dt, 0.0, 1.0)
        uw = _win_predict(wins[w], s, x)
        if w == len(wins) - 1:
            mask = (t >= t_lo)
        else:
            mask = (t >= t_lo) & (t < t_hi)
        u = jnp.where(mask, uw, u)
    return {"u": u}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    xs = jnp.asarray(np.linspace(X_MIN, X_MAX, NX, endpoint=False).astype(np.float32))

    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    ic_x0 = jnp.asarray(ic[:, 0].astype(np.float32))
    ic_u0 = jnp.asarray(ic[:, 1].astype(np.float32))

    bounds = tuple(float(b) for b in np.linspace(T_MIN, T_MAX, N_WIN + 1))

    k0, k1 = random.split(rng)
    pf = time_model.init(k0, jnp.zeros([1, 1]))
    pg = space_model.init(k1, jnp.zeros([1, 1]))

    train_budget = TRAIN_TIME - SAVE_MARGIN_S
    wins = []
    total_steps = 0
    ic_x, ic_u = ic_x0, ic_u0
    for w in range(N_WIN):
        t_lo, t_hi = bounds[w], bounds[w + 1]
        dt = t_hi - t_lo
        deadline = _T0 + train_budget * (w + 1) / N_WIN
        pf, pg, steps = _train_window(pf, pg, dt, xs, ic_x, ic_u, deadline, rng)
        total_steps += steps
        wins.append({"f": pf, "g": pg})
        # handoff: next window IC = this window's endpoint field on the x-grid
        if w < N_WIN - 1:
            u_end = _win_predict(wins[w], jnp.ones_like(xs), xs)   # s=1 -> t=t_hi
            ic_x = xs
            ic_u = jax.lax.stop_gradient(u_end)

    params = {"bounds": bounds, "win": wins}
    return params, total_steps
