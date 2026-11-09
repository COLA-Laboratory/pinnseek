"""Scale-PINN (sequential correction) periodic-embedding PINN for KS
(fork: b01_pemb; scale sweep f). Change vs parent: add the Scale-PINN KS sequential-
correction term S_KS to the residual (kb1 126), gamma=a2, to
escape the spurious decaying-solution minimum b01 was trapped in.
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

def _build_dataset():
    nt, nx = 101, 512
    ts = np.linspace(T_MIN, T_MAX, nt)[1:]
    xs = np.linspace(X_MIN, X_MAX, nx, endpoint=False)
    T, X = np.meshgrid(ts, xs, indexing="ij")
    pts_col = np.stack([T.ravel(), X.ravel()], axis=1).astype(np.float32)
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0], ic[:, 1]
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic], axis=1).astype(np.float32)
    return (jnp.array(pts_col), jnp.array(pts_ic), jnp.array(u_ic.astype(np.float32)))


# ══════════ Hyperparameters ══════════

LR        = 1e-3
WEIGHT_IC = 100.0
BS_COL    = 2000
BS_IC     = 256
SAVE_MARGIN_S = 6.0

# Scale-PINN sequential correction (kb1 126): gamma_KS = a2 = V2
INV_TSC = 0.3            # 1/tau_sc  (strength of the (u^k-u^{k-1}) term)
GAMMA_KS = V2             # = a2 (anti-diffusion magnitude)
INV_TA  = 0.0768           # 1/tau_alpha; B = GAMMA_KS/tau_alpha = GAMMA_KS*INV_TA


# ══════════ Training ══════════

def _make_train(rng_init):
    pts_col, pts_ic, u_ic = _build_dataset()
    n_col, n_ic = len(pts_col), len(pts_ic)
    B_SMOOTH = GAMMA_KS * INV_TA

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

    def resid_corr_point(params_flat, prev_flat, tx):
        t, x = tx[0], tx[1]
        u      = u_scalar(params_flat, t, x)
        u_t    = u_t_fn(params_flat, t, x)
        u_x    = u_x_fn(params_flat, t, x)
        u_xx   = u_xx_fn(params_flat, t, x)
        u_xxxx = u_xxxx_fn(params_flat, t, x)
        r = u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx
        # sequential correction using previous-iterate params (constants)
        u_prev  = u_scalar(prev_flat, t, x)
        uxx_prev = u_xx_fn(prev_flat, t, x)
        s = INV_TSC * (u - u_prev) - B_SMOOTH * (u_xx - uxx_prev)
        return (r + s) ** 2

    resid_batch = vmap(resid_corr_point, in_axes=(None, None, 0))
    u_batch = vmap(u_scalar, in_axes=(None, 0, 0))

    def loss_fn(params_flat, prev_flat, col, ic_xy, ic_u):
        pde_loss = jnp.mean(resid_batch(params_flat, prev_flat, col))
        u_pred_ic = u_batch(params_flat, ic_xy[:, 0], ic_xy[:, 1])
        ic_loss = jnp.mean((u_pred_ic - ic_u) ** 2)
        return pde_loss + WEIGHT_IC * ic_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, prev_flat, opt_state, keys):
        k_col, k_ic = keys
        col = pts_col[random.choice(k_col, n_col, (BS_COL,))]
        ic_idx = random.choice(k_ic, n_ic, (BS_IC,))
        loss, grads = loss_and_grad(params_flat, prev_flat, col, pts_ic[ic_idx], u_ic[ic_idx])
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
    prev_flat = params_flat        # w^{-1} = w^0

    keys = random.split(rng_local, 2)
    rng_local = random.split(rng_local)[0]
    new_flat, opt_state, loss = update(params_flat, prev_flat, opt_state, keys)
    prev_flat, params_flat = params_flat, new_flat
    step_count = 1
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        rng_local, sub = random.split(rng_local)
        keys = random.split(sub, 2)
        new_flat, opt_state, loss = update(params_flat, prev_flat, opt_state, keys)
        prev_flat, params_flat = params_flat, new_flat
        step_count = it + 1
        last_loss = float(loss)
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(params_flat), it, last_loss)

    return unravel(params_flat), step_count
