"""Vanilla PINN baseline for the long-time 2D heat equation (tree root).

Contract (do not change):
  - ``predict_fn(params, X) -> {"u"}``
  - ``train(rng, eval_callback=None) -> (params, step_count)``
  - Frozen PDE-CONSTANTS header is byte-identical across all descendants.

Time budget (frozen): ``TRAIN_TIME = 180`` s of wall-clock for the
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
INPUT_DIM    = 3
OUTPUT_DIM   = 1
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u",)

# --- Domain ---
T_MIN, T_MAX = 0.0, 100.0
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 1   # r
HAS_SOURCE              = True   # and it is nonlinear in u: sin(u^2) gates it,
                                 # so u ≡ 0 solves the residual exactly

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi")
PERIODIC_AXES  = ()       # none: Dirichlet u = 0 on all four sides
GEOMETRY_KIND  = "box_dirichlet"

# --- Physical parameters ---
# Residual: u_t - D (u_xx + u_yy)
#           - SRC_AMP sin(u^2) (1 + 2 sin(pi t / T_SRC)) sin(M1 pi x) sin(M2 pi y) = 0
D        = 0.001   # diffusivity
SRC_AMP  = 5.0     # source amplitude
T_SRC    = 4.0     # source time factor 1 + 2 sin(pi t / T_SRC) — period 2*T_SRC = 8
M1, M2   = 4.0, 2.0    # source spatial wavenumbers, sin(M1 pi x) sin(M2 pi y)
IC_M1, IC_M2 = 4.0, 3.0  # u(0,x,y) = sin(IC_M1 pi x) sin(IC_M2 pi y) — IC_M2 = 3
                         # differs from the source's M2 = 2; this is not a typo

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 180.0     # total wall-clock seconds for the WHOLE process,
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


# ══════════ Vanilla PINN: 5 × 100 tanh, output u ══════════

class PINN(nn.Module):
    """5 hidden layers × 100 nodes, tanh activation everywhere, He-uniform
    init, linear output of 1 channel (u). No feature mapping, no
    shared-trunk / split-head structure — a single MLP over (t, x, y)."""
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, txy):
        # txy: [N, 3]
        kinit = jax.nn.initializers.he_uniform()
        h = txy
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)  # → [N, 1]: u


# ══════════ Source term ══════════

def _source(t, x, y, u):
    """The forcing, nonlinear in u through sin(u^2)."""
    return (SRC_AMP * jnp.sin(u ** 2)
            * (1.0 + 2.0 * jnp.sin(jnp.pi * t / T_SRC))
            * jnp.sin(M1 * jnp.pi * x) * jnp.sin(M2 * jnp.pi * y))


# ══════════ Data: analytic collocation + Dirichlet BC, IC from file ══════════

def _build_dataset():
    """Collocation grid + IC points (from heat_ic.csv) + Dirichlet-BC samples.

    Returns:
      pts_col : [Nc, 3] interior collocation coordinates (PDE residual)
      pts_ic  : [Ni, 3] (t=0, x, y) coordinates for the initial condition
      u_ic    : [Ni] IC targets at t = 0
      pts_bc  : [Nb, 3] side-face coordinates where u = 0
    """
    # collocation: uniform grid, strict interior in time (t=0 IC plane
    # excluded — the PDE residual is not enforced on the supervised
    # initial condition); x and y include the Dirichlet faces.
    nt, nx, ny = 101, 48, 48
    ts = np.linspace(T_MIN, T_MAX, nt)[1:]
    xs = np.linspace(X_MIN, X_MAX, nx)
    ys = np.linspace(Y_MIN, Y_MAX, ny)
    T, X, Y = np.meshgrid(ts, xs, ys, indexing="ij")
    pts_col = np.stack([T.ravel(), X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    # initial condition (the ONLY field data we may read)
    ic = np.loadtxt(_locate("heat_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, y_ic, u_ic = ic[:, 0], ic[:, 1], ic[:, 2]
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic, y_ic], axis=1).astype(np.float32)

    # Dirichlet BC: u = 0 on the four side faces
    nb = 64
    tb = np.linspace(T_MIN, T_MAX, nb)
    sb = np.linspace(X_MIN, X_MAX, nb)
    TB, SB = np.meshgrid(tb, sb, indexing="ij")
    tb_f, sb_f = TB.ravel(), SB.ravel()
    lo = np.full_like(sb_f, X_MIN)
    hi = np.full_like(sb_f, X_MAX)
    pts_bc = np.concatenate([
        np.stack([tb_f, lo, sb_f], axis=1), np.stack([tb_f, hi, sb_f], axis=1),
        np.stack([tb_f, sb_f, lo], axis=1), np.stack([tb_f, sb_f, hi], axis=1),
    ]).astype(np.float32)

    return (
        jnp.array(pts_col),
        jnp.array(pts_ic),
        jnp.array(u_ic.astype(np.float32)),
        jnp.array(pts_bc),
    )


# ══════════ Training hyperparameters (intentionally minimal) ══════════

LR        = 1e-3      # fixed, no schedule
WEIGHT_IC = 1.0       # vanilla: no loss re-weighting
WEIGHT_BC = 1.0
BS_COL    = 1000      # collocation points per minibatch
BS_IC     = 128       # IC points per minibatch
BS_BC     = 128       # Dirichlet-BC samples per minibatch
SAVE_MARGIN_S = 5.0   # stop the loop this early so device_get + pickling
                      # finish before eval.py's TRAIN_TIME wall


# ══════════ Loss: PDE residual via autograd, IC + Dirichlet-BC L2 ══════════

def _make_train(rng_init):
    """Build the JIT'd minibatch + residual + loss + update closures."""
    pts_col, pts_ic, u_ic, pts_bc = _build_dataset()
    n_col, n_ic, n_bc = len(pts_col), len(pts_ic), len(pts_bc)

    model = PINN()
    key, rng = random.split(rng_init)
    dummy = jnp.zeros([1, INPUT_DIM])
    params = model.init(key, dummy)
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    # Single-point forward → returns [1]: u
    def predict_one(params_flat, txy):
        return model.apply(unravel(params_flat), txy[None, :])[0]

    # First derivatives: jacobian wrt (t, x, y) → shape [1, 3]
    # row = u; cols = (t, x, y) partials
    d1_fn = jacfwd(predict_one, argnums=1)

    # Second derivatives: shape [1, 3, 3]
    # [field_idx, i, j] = ∂²u/(∂txy[i]∂txy[j])
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)

    def residual_per_point(params_flat, txy):
        u = predict_one(params_flat, txy)[0]

        d1 = d1_fn(params_flat, txy)           # [1, 3]
        u_t = d1[0, 0]

        d2 = d2_fn(params_flat, txy)           # [1, 3, 3]
        u_xx, u_yy = d2[0, 1, 1], d2[0, 2, 2]

        # heat equation with the u-nonlinear source
        r = u_t - D * (u_xx + u_yy) - _source(txy[0], txy[1], txy[2], u)

        return r ** 2

    # Vectorize over the collocation batch
    residual_batch = vmap(residual_per_point, in_axes=(None, 0))
    predict_batch = vmap(predict_one, in_axes=(None, 0))

    def loss_fn(params_flat, col, ic_txy, ic_u, bc_txy):
        # PDE residual on interior collocation
        pde_loss = jnp.mean(residual_batch(params_flat, col))

        # initial condition: u = sin(4 pi x) sin(3 pi y) on the t = 0 plane
        ic_loss = jnp.mean((predict_batch(params_flat, ic_txy)[:, 0] - ic_u) ** 2)

        # Dirichlet BC: u = 0 on the four side faces
        bc_loss = jnp.mean(predict_batch(params_flat, bc_txy)[:, 0] ** 2)

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
                                    pts_ic[idx_ic], u_ic[idx_ic], pts_bc[idx_bc])
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params_flat, updates)
        return new_params, opt_state, loss

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: jnp.ndarray of shape [BS, 3] with columns (t, x, y).
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
