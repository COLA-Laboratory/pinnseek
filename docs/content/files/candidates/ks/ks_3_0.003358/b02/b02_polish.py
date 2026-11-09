"""Polished champion PINN for KS (fork: b01_scale_i, rRMSE 0.2721).
Change vs parent: keep pemb M=16 + b01's optimal mild sequential correction
1/tau_sc=0.6 (kb1 126) but resample collocation continuously each step (kb1 130)
instead of from a fixed grid; best-shot reproduction/refinement of the champion.
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


def _load_ic():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0], ic[:, 1]
    return (jnp.array(x_ic.astype(np.float32)), jnp.array(u_ic.astype(np.float32)))


# ══════════ Hyperparameters ══════════

LR        = 1e-3
WEIGHT_IC = 100.0
BS_COL    = 2000
SAVE_MARGIN_S = 6.0

INV_TSC  = 0.6              # b01 optimal
GAMMA_KS = V2
INV_TA   = 0.1536
B_SMOOTH = GAMMA_KS * INV_TA


def _make_train(rng_init):
    x_ic, u_ic = _load_ic()
    n_ic = len(x_ic)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def u_scalar(pf, t, x):
        return model.apply(unravel(pf), jnp.stack([t, x])[None, :])[0, 0]

    u_t_fn    = grad(u_scalar, argnums=1)
    u_x_fn    = grad(u_scalar, argnums=2)
    u_xx_fn   = grad(u_x_fn,  argnums=2)
    u_xxx_fn  = grad(u_xx_fn, argnums=2)
    u_xxxx_fn = grad(u_xxx_fn, argnums=2)

    def resid_corr_pt(pf, prev, t, x):
        u      = u_scalar(pf, t, x)
        u_t    = u_t_fn(pf, t, x)
        u_x    = u_x_fn(pf, t, x)
        u_xx   = u_xx_fn(pf, t, x)
        u_xxxx = u_xxxx_fn(pf, t, x)
        r = u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx
        u_prev   = u_scalar(prev, t, x)
        uxx_prev = u_xx_fn(prev, t, x)
        s = INV_TSC * (u - u_prev) - B_SMOOTH * (u_xx - uxx_prev)
        return (r + s) ** 2

    resid_batch = vmap(resid_corr_pt, in_axes=(None, None, 0, 0))
    u_batch = vmap(u_scalar, in_axes=(None, 0, 0))

    def loss_fn(pf, prev, col_t, col_x, ic_x, ic_u):
        pde = jnp.mean(resid_batch(pf, prev, col_t, col_x))
        u_ic_pred = u_batch(pf, jnp.zeros((n_ic,)), ic_x)
        ic_loss = jnp.mean((u_ic_pred - ic_u) ** 2)
        return pde + WEIGHT_IC * ic_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(pf, prev, opt_state, key):
        k_t, k_x = random.split(key)
        col_t = T_MAX * random.uniform(k_t, (BS_COL,))
        col_x = X_MAX * random.uniform(k_x, (BS_COL,))
        loss, grads = loss_and_grad(pf, prev, col_t, col_x, x_ic, u_ic)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(pf, updates), opt_state, loss

    return flat_params, opt_state, update, unravel, rng


_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    pf, opt_state, update, unravel, rng_local = _make_train(rng)
    prev = pf

    rng_local, sub = random.split(rng_local)
    new, opt_state, loss = update(pf, prev, opt_state, sub)
    prev, pf = pf, new
    step_count = 1

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        rng_local, sub = random.split(rng_local)
        new, opt_state, loss = update(pf, prev, opt_state, sub)
        prev, pf = pf, new
        step_count = it + 1
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(unravel(pf), it, float(loss))

    return unravel(pf), step_count
