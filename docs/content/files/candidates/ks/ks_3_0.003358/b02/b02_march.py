"""Time-marching periodic-embedding PINN for KS (fork: b01_pemb, rRMSE 0.276).
Change vs parent: split [0,T] into N_WINDOWS time windows, train a separate
periodic-embedding net per window in one process, warm-started + IC-anchored to
the previous window's end profile (jaxpi time-marching, kb1 084/130) to escape
the global spurious decaying-solution minimum (kb1 130 Thm 2.1). Collocation is
resampled continuously each step (kb1 130).
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
from jax import grad, jit, random, vmap
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


# ══════════ Periodic-embedding MLP (operates on prebuilt features) ══════════

M_MODES   = 16
N_WINDOWS = 4     # time windows over [T_MIN, T_MAX]
_KS = jnp.arange(1, M_MODES + 1, dtype=jnp.float32)


class MLP(nn.Module):
    """5x100 tanh MLP on the exact periodic Fourier embedding
    [tau, cos(kx), sin(kx)]_{k=1..M}, tau = (t - t_a)/dt the window-local time."""
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, feat):
        kinit = jax.nn.initializers.glorot_normal()
        h = feat
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)


_MODEL = MLP()


def _feat_batch(t, x, t_a, dt):
    tau = ((t - t_a) / dt)[:, None]
    xk = x[:, None] * _KS[None, :]
    return jnp.concatenate([tau, jnp.cos(xk), jnp.sin(xk)], axis=1)


# ══════════ Data (IC only) ══════════

def _load_ic():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0], ic[:, 1]
    return (jnp.array(x_ic.astype(np.float32)), jnp.array(u_ic.astype(np.float32)))


# ══════════ Hyperparameters ══════════

LR        = 1e-3
WEIGHT_IC = 100.0     # strong IC anchor (kb1 084)
BS_COL    = 2000
SAVE_MARGIN_S = 8.0


# ══════════ Training closures (built once, reused for every window) ══════════

def _make_closures(rng_init):
    x_ic, u_ic0 = _load_ic()
    n_ic = len(x_ic)

    dummy = jnp.zeros([1, 1 + 2 * M_MODES])
    params = _MODEL.init(random.split(rng_init)[0], dummy)
    flat0, unravel = jax.flatten_util.ravel_pytree(params)

    def u_scalar(pf, t, x, t_a, dt):
        tau = (t - t_a) / dt
        xk = x * _KS
        feat = jnp.concatenate([jnp.reshape(tau, (1,)), jnp.cos(xk), jnp.sin(xk)])
        return _MODEL.apply(unravel(pf), feat[None, :])[0, 0]

    u_t_fn    = grad(u_scalar, argnums=1)
    u_x_fn    = grad(u_scalar, argnums=2)
    u_xx_fn   = grad(u_x_fn,  argnums=2)
    u_xxx_fn  = grad(u_xx_fn, argnums=2)
    u_xxxx_fn = grad(u_xxx_fn, argnums=2)

    def resid_pt(pf, t, x, t_a, dt):
        u      = u_scalar(pf, t, x, t_a, dt)
        u_t    = u_t_fn(pf, t, x, t_a, dt)
        u_x    = u_x_fn(pf, t, x, t_a, dt)
        u_xx   = u_xx_fn(pf, t, x, t_a, dt)
        u_xxxx = u_xxxx_fn(pf, t, x, t_a, dt)
        r = u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx
        return r ** 2

    resid_batch = vmap(resid_pt, in_axes=(None, 0, 0, None, None))
    u_batch     = vmap(u_scalar, in_axes=(None, 0, 0, None, None))

    def loss_fn(pf, col_t, col_x, ic_u, t_a, dt):
        pde = jnp.mean(resid_batch(pf, col_t, col_x, t_a, dt))
        u_pred_ic = u_batch(pf, jnp.full((n_ic,), t_a), x_ic, t_a, dt)
        ic_loss = jnp.mean((u_pred_ic - ic_u) ** 2)
        return pde + WEIGHT_IC * ic_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(LR)

    @jit
    def update(pf, opt_state, key, ic_u, t_a, dt):
        k_t, k_x = random.split(key)
        col_t = t_a + dt * random.uniform(k_t, (BS_COL,))
        col_x = X_MAX * random.uniform(k_x, (BS_COL,))
        loss, grads = loss_and_grad(pf, col_t, col_x, ic_u, t_a, dt)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(pf, updates), opt_state, loss

    # batched end-of-window profile (for the next window's IC), no-grad
    def end_profile(pf, t_b, t_a, dt):
        return u_batch(pf, jnp.full((n_ic,), t_b), x_ic, t_a, dt)
    end_profile = jit(end_profile)

    return flat0, unravel, optimizer, update, end_profile, x_ic, u_ic0


# ══════════ Public contract ══════════

def predict_fn(params, X):
    """params: list of N_WINDOWS Flax param pytrees (one per time window)."""
    t = X[:, 0]
    x = X[:, 1]
    dtw = (T_MAX - T_MIN) / N_WINDOWS
    idx = jnp.clip(jnp.floor((t - T_MIN) / dtw).astype(jnp.int32), 0, N_WINDOWS - 1)
    u = jnp.zeros_like(t)
    for w in range(N_WINDOWS):
        t_a = T_MIN + w * dtw
        feat = _feat_batch(t, x, t_a, dtw)
        uw = _MODEL.apply(params[w], feat)[:, 0]
        u = jnp.where(idx == w, uw, u)
    return {"u": u}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    flat0, unravel, optimizer, update, end_profile, x_ic, u_ic0 = _make_closures(rng)
    dtw = jnp.asarray((T_MAX - T_MIN) / N_WINDOWS, jnp.float32)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S

    pf = flat0
    ic_u = u_ic0
    window_params = []
    total_steps = 0
    rng_local = random.split(rng)[0]

    for w in range(N_WINDOWS):
        t_a = jnp.asarray(T_MIN + w * (T_MAX - T_MIN) / N_WINDOWS, jnp.float32)
        opt_state = optimizer.init(pf)   # fresh momentum each window (warm-start weights)

        now = time.time()
        win_deadline = now + (deadline - now) / (N_WINDOWS - w)

        # warmup / compile (real step)
        rng_local, sub = random.split(rng_local)
        pf, opt_state, loss = update(pf, opt_state, sub, ic_u, t_a, dtw)
        total_steps += 1
        while time.time() < win_deadline:
            rng_local, sub = random.split(rng_local)
            pf, opt_state, loss = update(pf, opt_state, sub, ic_u, t_a, dtw)
            total_steps += 1
            if eval_callback is not None and (total_steps % 2000 == 0):
                eval_callback([unravel(pf)] * N_WINDOWS, total_steps, float(loss))

        window_params.append(unravel(pf))
        # next window's IC = this window's end profile at t = t_a + dtw
        if w < N_WINDOWS - 1:
            t_b = jnp.asarray(T_MIN + (w + 1) * (T_MAX - T_MIN) / N_WINDOWS, jnp.float32)
            ic_u = jax.lax.stop_gradient(end_profile(pf, t_b, t_a, dtw))

    return window_params, total_steps
