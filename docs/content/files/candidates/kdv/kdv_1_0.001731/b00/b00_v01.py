"""Vanilla PINN baseline for the Korteweg-de Vries equation (tree root).

Contract (do not change):
  - ``predict_fn(params, X) -> {"u"}``
  - ``train(rng, eval_callback=None) -> (params, step_count)``
  - Frozen PDE-CONSTANTS header is byte-identical across all descendants.

Time budget (frozen): ``TRAIN_TIME = 100`` s of wall-clock for the
WHOLE process, anchored at ``_T0`` (process start) — imports, JIT
compilation, training and param save all included; ``eval.py`` kills
the process at this wall. ``train()`` must return params with margin
to spare (this baseline stops ``SAVE_MARGIN_S = 5`` s early).
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
T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = -1.0, 1.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 3
N_RESIDUAL_COMPONENTS   = 1   # r
HAS_SOURCE              = False  # no forcing: every term carries u

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi")
PERIODIC_AXES  = ("x",)   # u(t, X_MIN) = u(t, X_MAX)
GEOMETRY_KIND  = "periodic_box"

# --- Physical parameters ---
DELTA = 0.022    # dispersion coefficient; the PDE carries delta^2 u_xxx

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 100.0     # total wall-clock seconds for the WHOLE process,
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
from jax import jacfwd, jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Vanilla PINN: 5 × 100 tanh, output (u, v) ══════════

class PINN(nn.Module):
    """5 hidden layers × 100 nodes, tanh activation everywhere, He-uniform
    init, linear output of 1 channel (u). No feature mapping, no
    shared-trunk / split-head structure — a single MLP over (t, x)."""
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, tx):
        # tx: [N, 2]
        kinit = jax.nn.initializers.he_uniform()
        h = tx
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)  # → [N, 1]: u


# ══════════ Data: analytic collocation + periodic BC, IC from file ══════════

def _build_dataset():
    """Collocation grid + IC points (from kdv_ic.csv) + periodic-BC samples.

    Returns:
      pts_col : [Nc, 2] interior collocation coordinates (PDE residual)
      pts_ic  : [Ni, 2] (t=0, x) coordinates for the initial condition
      u_ic    : [Ni] IC targets u at t = 0
      bc_t    : [Nb] t samples at which the two x-faces must agree
    """
    # collocation: uniform grid, strict interior in time (t=0 IC line
    # excluded — the PDE residual is not enforced on the supervised
    # initial condition), x on the half-open torus [-1, 1).
    nt, nx = 201, 256
    ts = np.linspace(T_MIN, T_MAX, nt)[1:]
    xs = np.linspace(X_MIN, X_MAX, nx, endpoint=False)
    T, X = np.meshgrid(ts, xs, indexing="ij")
    pts_col = np.stack([T.ravel(), X.ravel()], axis=1).astype(np.float32)

    # initial condition (the ONLY field data we may read)
    ic = np.loadtxt(_locate("kdv_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0], ic[:, 1]
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic], axis=1).astype(np.float32)
    u_ic = u_ic.astype(np.float32)

    # periodic BC: t samples at which the two x-faces must agree
    nb = 4096
    bc_t = np.linspace(T_MIN, T_MAX, nb).astype(np.float32)

    return (
        jnp.array(pts_col),
        jnp.array(pts_ic),
        jnp.array(u_ic),
        jnp.array(bc_t),
    )


# ══════════ Training hyperparameters (intentionally minimal) ══════════

LR        = 1e-3      # fixed, no schedule
WEIGHT_IC = 1.0       # vanilla: no loss re-weighting
WEIGHT_BC = 1.0
BS_COL    = 1000      # collocation points per minibatch
BS_IC     = 128       # IC points per minibatch
BS_BC     = 128       # periodic-BC samples per minibatch
SAVE_MARGIN_S = 5.0   # stop the loop this early so device_get + pickling
                      # finish before eval.py's TRAIN_TIME wall


# ══════════ Loss: PDE residual via autograd, IC + periodic-BC L2 ══════════

def _make_train(rng_init):
    """Build the JIT'd minibatch + residual + loss + update closures."""
    pts_col, pts_ic, u_ic, bc_t = _build_dataset()
    n_col, n_ic, n_bc = len(pts_col), len(pts_ic), len(bc_t)

    model = PINN()
    key, rng = random.split(rng_init)
    dummy = jnp.zeros([1, INPUT_DIM])
    params = model.init(key, dummy)
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    # Single-point forward → returns [1]: u
    def predict_one(params_flat, tx):
        return model.apply(unravel(params_flat), tx[None, :])[0]

    # First derivatives: jacobian wrt (t, x) → shape [1, 2]
    # row = u; cols = (t, x) partials
    d1_fn = jacfwd(predict_one, argnums=1)

    # Third derivatives: shape [1, 2, 2, 2]
    # [field_idx, i, j, k] = ∂³u/(∂tx[i]∂tx[j]∂tx[k])
    d3_fn = jacfwd(jacfwd(jacfwd(predict_one, argnums=1), argnums=1), argnums=1)

    def residual_per_point(params_flat, tx):
        u = predict_one(params_flat, tx)[0]

        d1 = d1_fn(params_flat, tx)            # [1, 2]
        u_t = d1[0, 0]
        u_x = d1[0, 1]

        d3 = d3_fn(params_flat, tx)            # [1, 2, 2, 2]
        u_xxx = d3[0, 1, 1, 1]

        # Korteweg-de Vries residual
        r = u_t + u * u_x + DELTA ** 2 * u_xxx

        return r ** 2

    # Vectorize over the collocation batch
    residual_batch = vmap(residual_per_point, in_axes=(None, 0))
    predict_batch = vmap(predict_one, in_axes=(None, 0))

    def loss_fn(params_flat, col, ic_tx, ic_u, bc_batch):
        # PDE residual on interior collocation
        pde_loss = jnp.mean(residual_batch(params_flat, col))

        # initial condition: u at t = 0
        ic_u_pred = predict_batch(params_flat, ic_tx)[:, 0]
        ic_loss = jnp.mean((ic_u_pred - ic_u) ** 2)

        # periodic BC: the two x-faces agree
        tb = bc_batch
        lo = jnp.full_like(tb, X_MIN)
        hi = jnp.full_like(tb, X_MAX)
        x_lo = predict_batch(params_flat, jnp.stack([tb, lo], axis=1))
        x_hi = predict_batch(params_flat, jnp.stack([tb, hi], axis=1))
        bc_loss = jnp.mean((x_lo - x_hi) ** 2)

        return pde_loss + WEIGHT_IC * ic_loss + WEIGHT_BC * bc_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))

    optimizer = optax.adam(learning_rate=LR)   # fixed LR, no schedule
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, opt_state, key_triple):
        key1, key2, key3 = key_triple
        idx_col = random.choice(key1, n_col, (BS_COL,))
        idx_ic  = random.choice(key2, n_ic,  (BS_IC,))
        idx_bc  = random.choice(key3, n_bc,  (BS_BC,))
        loss, grads = loss_and_grad(params_flat, pts_col[idx_col],
                                    pts_ic[idx_ic], u_ic[idx_ic], bc_t[idx_bc])
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params_flat, updates)
        return new_params, opt_state, loss

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: jnp.ndarray of shape [BS, 2] with columns (t, x).
    Returns dict {"u": [BS]}.
    """
    out = _PREDICT_MODEL.apply(params, X)   # [BS, 1]
    return {"u": out[:, 0]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam training. The whole process — imports, JIT
    compilation, training, param save — must finish within TRAIN_TIME
    seconds of ``_T0`` (process start); the loop stops SAVE_MARGIN_S
    early so the caller can device_get + pickle before the wall.

    Args:
        rng: jax.random.PRNGKey — passed to ``_make_train`` for deterministic init.
        eval_callback: optional ``callback(params, step, loss)``, fires every
             1000 steps with the unraveled (pytree) params.

    Returns:
        (params, step_count) — ``params`` is a Flax pytree.
    """
    params_flat, opt_state, update, unravel, rng_local = _make_train(rng)

    # Warmup step: the first update() call JIT-compiles the whole training
    # step; float(loss) blocks until it finishes. It is a real step (counted).
    keys = random.split(rng_local, 4)
    rng_local = keys[0]
    params_flat, opt_state, loss = update(params_flat, opt_state, tuple(keys[1:]))
    step_count = 1
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        keys = random.split(rng_local, 4)
        rng_local = keys[0]
        params_flat, opt_state, loss = update(params_flat, opt_state, tuple(keys[1:]))
        step_count = it + 1
        last_loss = float(loss)
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(params_flat), it, last_loss)

    return unravel(params_flat), step_count
