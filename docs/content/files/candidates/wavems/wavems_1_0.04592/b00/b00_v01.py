"""Vanilla PINN baseline for the multiscale-medium wave equation (tree root).

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
INPUT_DIM    = 3
OUTPUT_DIM   = 1
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u",)

# --- Domain ---
T_MIN, T_MAX = 0.0, 5.0
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 2   # u_tt — the IC fixes both u and u_t at t = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 1   # r
HAS_SOURCE              = False  # no forcing: every term carries u

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi")
PERIODIC_AXES  = ()       # none: zero Neumann on all four faces
GEOMETRY_KIND  = "box"

# --- Physical parameters ---
COEF_FILE = "wavems_coef.csv"   # c(x, y) on a 256 x 256 grid over [-1, 1]^2
C_LO, C_HI = 1.0, 12.0          # the medium is two-valued; wave speed sqrt(c)
IC_X0, IC_Y0, IC_SIGMA = -0.5, 0.0, 0.3   # Gaussian pulse of the initial condition

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


# ══════════ Data: analytic collocation, IC and Neumann BC; medium from file ══════════


def _coef_grid():
    """The 256 x 256 medium c(x, y) as an array C[iy, ix] plus its axis."""
    raw = np.loadtxt(_locate(COEF_FILE), delimiter=",", skiprows=1, dtype=np.float64)
    n = int(round(np.sqrt(len(raw))))
    return raw[:, 2].reshape(n, n).astype(np.float32), n


def coef_at(C, n, x, y):
    """Bilinear interpolation of the medium at arbitrary (x, y) — numpy or jnp.
    Blocks that resample collocation points must evaluate c through this."""
    gx = (x - X_MIN) / (X_MAX - X_MIN) * (n - 1)
    gy = (y - Y_MIN) / (Y_MAX - Y_MIN) * (n - 1)
    gx = jnp.clip(gx, 0.0, n - 1 - 1e-6)
    gy = jnp.clip(gy, 0.0, n - 1 - 1e-6)
    i, j = jnp.floor(gx).astype(jnp.int32), jnp.floor(gy).astype(jnp.int32)
    fx, fy = gx - i, gy - j
    c00, c10 = C[j, i], C[j, i + 1]
    c01, c11 = C[j + 1, i], C[j + 1, i + 1]
    return (1 - fy) * ((1 - fx) * c00 + fx * c10) + fy * ((1 - fx) * c01 + fx * c11)

def _build_dataset():
    """Collocation grid (+ the medium sampled on it), analytic IC points and
    Neumann-BC samples. The medium file is the only thing read from disk; the
    initial condition is the analytic pulse of problem.md.

    Returns:
      pts_col : [Nc, 3] interior collocation coordinates (PDE residual)
      c_col   : [Nc] the medium c(x, y) at those points
      pts_ic  : [Ni, 3] (t=0, x, y) coordinates for the initial condition
      u_ic    : [Ni] IC targets u at t = 0 (u_t = 0 is imposed separately)
      bc_ts   : [Nb, 2] (t, s) samples for the four Neumann faces — s plays
                the role of y on the x-faces and of x on the y-faces
    """
    # collocation: uniform grid, strict interior in time (t=0 IC plane
    # excluded — the PDE residual is not enforced on the supervised
    # initial condition); x and y include the Neumann faces.
    nt, nx, ny = 51, 64, 64
    ts = np.linspace(T_MIN, T_MAX, nt)[1:]
    xs = np.linspace(X_MIN, X_MAX, nx)
    ys = np.linspace(Y_MIN, Y_MAX, ny)
    T, X, Y = np.meshgrid(ts, xs, ys, indexing="ij")
    pts_col = np.stack([T.ravel(), X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    # the medium (the only file read) sampled once on the fixed grid
    C, n = _coef_grid()
    c_col = np.asarray(coef_at(jnp.array(C), n,
                               jnp.array(pts_col[:, 1]),
                               jnp.array(pts_col[:, 2])), dtype=np.float32)

    # initial condition: analytic Gaussian pulse, u_t = 0
    ni = 128
    xi = np.linspace(X_MIN, X_MAX, ni)
    XI, YI = np.meshgrid(xi, xi, indexing="ij")
    x_ic, y_ic = XI.ravel(), YI.ravel()
    u_ic = np.exp(-((x_ic - IC_X0) ** 2 + (y_ic - IC_Y0) ** 2)
                  / (2.0 * IC_SIGMA ** 2))
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic, y_ic], axis=1).astype(np.float32)
    u_ic = u_ic.astype(np.float32)

    # Neumann BC: (t, s) samples at which the normal derivative must vanish
    nb = 64
    tb = np.linspace(T_MIN, T_MAX, nb)
    sb = np.linspace(X_MIN, X_MAX, nb)
    TB, SB = np.meshgrid(tb, sb, indexing="ij")
    bc_ts = np.stack([TB.ravel(), SB.ravel()], axis=1).astype(np.float32)

    return (
        jnp.array(pts_col),
        jnp.array(c_col),
        jnp.array(pts_ic),
        jnp.array(u_ic),
        jnp.array(bc_ts),
    )


# ══════════ Training hyperparameters (intentionally minimal) ══════════

LR        = 1e-3      # fixed, no schedule
WEIGHT_IC = 1.0       # vanilla: no loss re-weighting
WEIGHT_BC = 1.0
BS_COL    = 1000      # collocation points per minibatch
BS_IC     = 128       # IC points per minibatch
BS_BC     = 128       # Neumann-BC samples per minibatch
SAVE_MARGIN_S = 5.0   # stop the loop this early so device_get + pickling
                      # finish before eval.py's TRAIN_TIME wall


# ══════════ Loss: PDE residual via autograd, IC (u and u_t) + Neumann-BC L2 ══════════

def _make_train(rng_init):
    """Build the JIT'd minibatch + residual + loss + update closures."""
    pts_col, c_col, pts_ic, u_ic, bc_ts = _build_dataset()
    n_col, n_ic, n_bc = len(pts_col), len(pts_ic), len(bc_ts)

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

    def residual_per_point(params_flat, txy, c):
        d2 = d2_fn(params_flat, txy)           # [1, 3, 3]
        u_tt = d2[0, 0, 0]
        u_xx, u_yy = d2[0, 1, 1], d2[0, 2, 2]

        # wave equation in a medium of speed sqrt(c(x, y))
        r = u_tt - c * (u_xx + u_yy)

        return r ** 2

    # Vectorize over the collocation batch
    residual_batch = vmap(residual_per_point, in_axes=(None, 0, 0))
    predict_batch = vmap(predict_one, in_axes=(None, 0))
    d1_batch = vmap(d1_fn, in_axes=(None, 0))

    def loss_fn(params_flat, col, c_batch, ic_txy, ic_u, bc_batch):
        # PDE residual on interior collocation
        pde_loss = jnp.mean(residual_batch(params_flat, col, c_batch))

        # initial condition: u at t = 0, and zero initial velocity u_t
        ic_u_pred = predict_batch(params_flat, ic_txy)[:, 0]
        ic_ut_pred = d1_batch(params_flat, ic_txy)[:, 0, 0]
        ic_loss = jnp.mean((ic_u_pred - ic_u) ** 2) + jnp.mean(ic_ut_pred ** 2)

        # Neumann BC: the normal derivative vanishes on all four faces
        tb, sb = bc_batch[:, 0], bc_batch[:, 1]
        lo = jnp.full_like(sb, X_MIN)
        hi = jnp.full_like(sb, X_MAX)
        ux_lo = d1_batch(params_flat, jnp.stack([tb, lo, sb], axis=1))[:, 0, 1]
        ux_hi = d1_batch(params_flat, jnp.stack([tb, hi, sb], axis=1))[:, 0, 1]
        uy_lo = d1_batch(params_flat, jnp.stack([tb, sb, lo], axis=1))[:, 0, 2]
        uy_hi = d1_batch(params_flat, jnp.stack([tb, sb, hi], axis=1))[:, 0, 2]
        bc_loss = (jnp.mean(ux_lo ** 2) + jnp.mean(ux_hi ** 2)
                   + jnp.mean(uy_lo ** 2) + jnp.mean(uy_hi ** 2))

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
        loss, grads = loss_and_grad(params_flat, pts_col[idx_col], c_col[idx_col],
                                    pts_ic[idx_ic], u_ic[idx_ic], bc_ts[idx_bc])
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
