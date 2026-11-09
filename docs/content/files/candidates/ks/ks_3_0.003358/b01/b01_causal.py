"""Causal-weighted periodic-embedding PINN for KS (fork: b01_pemb).
Change vs parent: add causal temporal weighting (kb1 084) over Nt time
slices, w_i=exp(-eps*cumsum_excl L_r), with a geometric eps ramp
0.02->0.3 (residual scale from diag). Everything else as b01_pemb.
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


# ══════════ Periodic-embedding PINN ══════════

M_MODES = 16


class PINN(nn.Module):
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, tx):
        t = tx[:, 0:1] / T_MAX
        x = tx[:, 1:2]
        ks = jnp.arange(1, M_MODES + 1, dtype=tx.dtype)
        xk = x * ks
        feat = jnp.concatenate([t, jnp.cos(xk), jnp.sin(xk)], axis=-1)
        kinit = jax.nn.initializers.glorot_normal()
        h = feat
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)


# ══════════ Data ══════════

def _build_ic():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0], ic[:, 1]
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic], axis=1).astype(np.float32)
    return jnp.array(pts_ic), jnp.array(u_ic.astype(np.float32))


# ══════════ Hyperparameters ══════════

LR        = 1e-3
WEIGHT_IC = 100.0
NT_SLICES = 48       # causal time slices (fixed, ascending)
NX_COL    = 128      # random x per step, shared across slices
EPS_MIN   = 0.02     # causal-weight ramp (geometric over wall-time)
EPS_MAX   = 0.30
SAVE_MARGIN_S = 6.0


# ══════════ Training ══════════

def _make_train(rng_init):
    pts_ic, u_ic = _build_ic()
    n_ic = len(pts_ic)
    ts = jnp.asarray(np.linspace(T_MIN, T_MAX, NT_SLICES + 1)[1:].astype(np.float32))  # [Nt]

    model = PINN()
    key, rng = random.split(rng_init)
    dummy = jnp.zeros([1, INPUT_DIM])
    params = model.init(key, dummy)
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def u_scalar(params_flat, t, x):
        return model.apply(unravel(params_flat), jnp.stack([t, x])[None, :])[0, 0]

    u_t_fn    = grad(u_scalar, argnums=1)
    u_x_fn    = grad(u_scalar, argnums=2)
    u_xx_fn   = grad(u_x_fn,  argnums=2)
    u_xxx_fn  = grad(u_xx_fn, argnums=2)
    u_xxxx_fn = grad(u_xxx_fn, argnums=2)

    def residual2(params_flat, t, x):
        u      = u_scalar(params_flat, t, x)
        u_t    = u_t_fn(params_flat, t, x)
        u_x    = u_x_fn(params_flat, t, x)
        u_xx   = u_xx_fn(params_flat, t, x)
        u_xxxx = u_xxxx_fn(params_flat, t, x)
        r = u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx
        return r ** 2

    # vmap over (Nt x Nx) grid: t over slices, x over collocation
    res_over_x = vmap(residual2, in_axes=(None, None, 0))       # fixed t, many x
    res_grid   = vmap(res_over_x, in_axes=(None, 0, None))      # many t
    u_batch    = vmap(u_scalar, in_axes=(None, 0, 0))

    def loss_fn(params_flat, xs, ic_xy, ic_u, eps):
        r2 = res_grid(params_flat, ts, xs)          # [Nt, Nx]
        Lr = jnp.mean(r2, axis=1)                   # [Nt]
        cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(Lr)[:-1]])
        w = jax.lax.stop_gradient(jnp.exp(-eps * cum))   # [Nt]
        pde_loss = jnp.mean(w * Lr)
        u_pred_ic = u_batch(params_flat, ic_xy[:, 0], ic_xy[:, 1])
        ic_loss = jnp.mean((u_pred_ic - ic_u) ** 2)
        return pde_loss + WEIGHT_IC * ic_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, opt_state, key, eps):
        xs = random.uniform(key, (NX_COL,), minval=X_MIN, maxval=X_MAX)
        loss, grads = loss_and_grad(params_flat, xs, pts_ic, u_ic, eps)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params_flat, updates)
        return new_params, opt_state, loss

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params_flat, opt_state, update, unravel, rng_local = _make_train(rng)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    ramp_dur = deadline - time.time()   # wall left for the loop
    log_ratio = np.log(EPS_MAX / EPS_MIN)

    rng_local, sub = random.split(rng_local)
    params_flat, opt_state, loss = update(params_flat, opt_state, sub, jnp.float32(EPS_MIN))
    step_count = 1
    last_loss = float(loss)

    for it in range(1, MAX_ITER):
        now = time.time()
        if now > deadline:
            break
        frac = min(1.0, max(0.0, (now - (deadline - ramp_dur)) / max(ramp_dur, 1e-6)))
        eps = jnp.float32(EPS_MIN * np.exp(log_ratio * frac))
        rng_local, sub = random.split(rng_local)
        params_flat, opt_state, loss = update(params_flat, opt_state, sub, eps)
        step_count = it + 1
        last_loss = float(loss)
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(params_flat), it, last_loss)

    return unravel(params_flat), step_count
