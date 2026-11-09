"""Mixed / first-order formulation PINN for KS (fork: b01_pemb, rRMSE 0.276).
Change vs parent: network outputs (u, w) with an auxiliary field w ~= u_xx, so the
PDE residual R1 = u_t + V1 u u_x + V2 w + V3 w_xx uses only <=2nd-order autograd
(the stiff 4th-order u_xxxx term becomes w_xx) plus a coupling residual
R2 = w - u_xx. Reducing the derivative order tackles the KS 4th-order
ill-conditioning (diag: V3 u_xxxx is the largest residual term). kb1 052/087.
Exact periodic embedding kept for both fields.
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


# ══════════ Periodic-embedding net with 2 outputs (u, w) ══════════

M_MODES = 16


class MixedNet(nn.Module):
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
        return nn.Dense(2, kernel_init=kinit)(h)     # [N,2]: (u, w)


def _load_ic():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0], ic[:, 1]
    return (jnp.array(x_ic.astype(np.float32)), jnp.array(u_ic.astype(np.float32)))


# ══════════ Hyperparameters ══════════

LR        = 1e-3
WEIGHT_IC = 100.0
WEIGHT_CP = 10.0        # coupling w = u_xx
BS_COL    = 2000
SAVE_MARGIN_S = 6.0


def _make_train(rng_init):
    x_ic, u_ic = _load_ic()
    n_ic = len(x_ic)

    model = MixedNet()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def uw(pf, t, x):
        o = model.apply(unravel(pf), jnp.stack([t, x])[None, :])[0]
        return o[0], o[1]
    def u_only(pf, t, x):
        return uw(pf, t, x)[0]
    def w_only(pf, t, x):
        return uw(pf, t, x)[1]

    u_t_fn  = grad(u_only, argnums=1)
    u_x_fn  = grad(u_only, argnums=2)
    u_xx_fn = grad(u_x_fn, argnums=2)
    w_x_fn  = grad(w_only, argnums=2)
    w_xx_fn = grad(w_x_fn, argnums=2)

    def resid_pt(pf, t, x):
        u, w   = uw(pf, t, x)
        u_t    = u_t_fn(pf, t, x)
        u_x    = u_x_fn(pf, t, x)
        u_xx   = u_xx_fn(pf, t, x)
        w_xx   = w_xx_fn(pf, t, x)
        r1 = u_t + V1 * u * u_x + V2 * w + V3 * w_xx   # w stands in for u_xx
        r2 = w - u_xx                                   # coupling
        return r1 ** 2, r2 ** 2

    resid_batch = vmap(resid_pt, in_axes=(None, 0, 0))
    u_batch = vmap(u_only, in_axes=(None, 0, 0))

    def loss_fn(pf, col_t, col_x, ic_x, ic_u):
        r1sq, r2sq = resid_batch(pf, col_t, col_x)
        pde = jnp.mean(r1sq) + WEIGHT_CP * jnp.mean(r2sq)
        u_ic_pred = u_batch(pf, jnp.zeros((n_ic,)), ic_x)
        ic_loss = jnp.mean((u_ic_pred - ic_u) ** 2)
        return pde + WEIGHT_IC * ic_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(pf, opt_state, key):
        k_t, k_x = random.split(key)
        col_t = T_MAX * random.uniform(k_t, (BS_COL,))
        col_x = X_MAX * random.uniform(k_x, (BS_COL,))
        loss, grads = loss_and_grad(pf, col_t, col_x, x_ic, u_ic)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(pf, updates), opt_state, loss

    return flat_params, opt_state, update, unravel, rng


_PREDICT_MODEL = MixedNet()


def predict_fn(params, X):
    out = _PREDICT_MODEL.apply(params, X)   # [BS,2]
    return {"u": out[:, 0]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    pf, opt_state, update, unravel, rng_local = _make_train(rng)

    rng_local, sub = random.split(rng_local)
    pf, opt_state, loss = update(pf, opt_state, sub)
    step_count = 1

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        rng_local, sub = random.split(rng_local)
        pf, opt_state, loss = update(pf, opt_state, sub)
        step_count = it + 1
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(unravel(pf), it, float(loss))

    return unravel(pf), step_count
