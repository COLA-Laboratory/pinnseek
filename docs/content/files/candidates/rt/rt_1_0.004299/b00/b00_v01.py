"""Vanilla PINN baseline for the Rayleigh-Taylor instability (tree root).

Contract (do not change):
  - ``predict_fn(params, X) -> {"u","v","p","T"}``
  - ``train(rng, eval_callback=None) -> (params, step_count)``
  - Frozen PDE-CONSTANTS header is byte-identical across all descendants.

Time budget (frozen): ``TRAIN_TIME = 480`` s of wall-clock for the
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
OUTPUT_DIM   = 4
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u", "v", "p", "T")

# --- Domain ---
T_MIN, T_MAX = 0.0, 4.75
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 2.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 4   # ru, rv, continuity, rT
HAS_SOURCE              = False  # buoyancy couples T into the y-momentum;
                                 # there is no externally prescribed forcing

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi")
PERIODIC_AXES  = ("x",)   # u(t, X_MIN, y) = u(t, X_MAX, y); the y walls are
                          # Dirichlet u = v = T = 0 (pressure is gauge-free)
GEOMETRY_KIND  = "periodic_channel"

# --- Physical parameters ---
PR     = 0.71                    # Prandtl number
RA     = 1.0e6                   # Rayleigh number
A_NU   = (PR / RA) ** 0.5        # sqrt(Pr/Ra)   = 8.426150e-04, viscous
A_KA   = 1.0 / (PR * RA) ** 0.5  # 1/sqrt(Pr Ra) = 1.186782e-03, thermal
A_BUOY = 1.0                     # buoyancy: +T on the y-momentum

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 480.0     # total wall-clock seconds for the WHOLE process,
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


# ══════════ Vanilla PINN: 5 × 100 tanh, output (u, v, p, T) ══════════

class PINN(nn.Module):
    """5 hidden layers × 100 nodes, tanh activation everywhere, He-uniform
    init, linear output of 4 channels (u, v, p, T). No feature mapping, no
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
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)  # → [N, 4]: (u, v, p, T)


# ══════════ Data: analytic collocation + BC samples, IC from file ══════════

def _build_dataset():
    """Collocation grid + IC points (from rt_ic.csv) + the two BC point sets.

    Returns:
      pts_col : [Nc, 3] interior collocation coordinates (PDE residual)
      pts_ic  : [Ni, 3] (t=0, x, y) coordinates for the initial condition
      uvT_ic  : [Ni, 3] IC targets (u, v, T) at t = 0 (pressure is gauge-free)
      per_ty  : [Np, 2] (t, y) samples at which the two x-faces must agree
      pts_dir : [Nd, 3] wall coordinates (y = Y_MIN / Y_MAX) where u = v = T = 0
    """
    # collocation: uniform grid, strict interior in time (t=0 IC plane
    # excluded — the PDE residual is not enforced on the supervised
    # initial condition); x on the half-open torus [0, 1), y spans the
    # channel including both Dirichlet walls.
    nt, nx, ny = 20, 64, 128
    ts = np.linspace(T_MIN, T_MAX, nt)[1:]
    xs = np.linspace(X_MIN, X_MAX, nx, endpoint=False)
    ys = np.linspace(Y_MIN, Y_MAX, ny)
    T, X, Y = np.meshgrid(ts, xs, ys, indexing="ij")
    pts_col = np.stack([T.ravel(), X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    # initial condition (the ONLY field data we may read)
    ic = np.loadtxt(_locate("rt_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, y_ic = ic[:, 0], ic[:, 1]
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic, y_ic], axis=1).astype(np.float32)
    uvT_ic = ic[:, 2:5].astype(np.float32)

    # periodic BC: (t, y) samples at which the x = 0 and x = 1 faces must agree
    nb = 64
    tb = np.linspace(T_MIN, T_MAX, nb)
    yb = np.linspace(Y_MIN, Y_MAX, nb)
    TB, YB = np.meshgrid(tb, yb, indexing="ij")
    per_ty = np.stack([TB.ravel(), YB.ravel()], axis=1).astype(np.float32)

    # Dirichlet walls: u = v = T = 0 on y = Y_MIN and y = Y_MAX
    xb = np.linspace(X_MIN, X_MAX, nb, endpoint=False)
    TW, XW = np.meshgrid(tb, xb, indexing="ij")
    tw, xw = TW.ravel(), XW.ravel()
    pts_dir = np.concatenate([
        np.stack([tw, xw, np.full_like(xw, Y_MIN)], axis=1),
        np.stack([tw, xw, np.full_like(xw, Y_MAX)], axis=1),
    ]).astype(np.float32)

    return (
        jnp.array(pts_col),
        jnp.array(pts_ic),
        jnp.array(uvT_ic),
        jnp.array(per_ty),
        jnp.array(pts_dir),
    )


# ══════════ Training hyperparameters (intentionally minimal) ══════════

LR        = 1e-3      # fixed, no schedule
WEIGHT_IC = 1.0       # vanilla: no loss re-weighting
WEIGHT_BC = 1.0
BS_COL    = 1000      # collocation points per minibatch
BS_IC     = 128       # IC points per minibatch
BS_BC     = 128       # BC samples per minibatch (each of the two sets)
SAVE_MARGIN_S = 5.0   # stop the loop this early so device_get + pickling
                      # finish before eval.py's TRAIN_TIME wall


# ══════════ Loss: PDE residual via autograd, IC + BC L2 ══════════

def _make_train(rng_init):
    """Build the JIT'd minibatch + residual + loss + update closures."""
    pts_col, pts_ic, uvT_ic, per_ty, pts_dir = _build_dataset()
    n_col, n_ic, n_per, n_dir = len(pts_col), len(pts_ic), len(per_ty), len(pts_dir)

    model = PINN()
    key, rng = random.split(rng_init)
    dummy = jnp.zeros([1, INPUT_DIM])
    params = model.init(key, dummy)
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    # Single-point forward → returns [4]: (u, v, p, T)
    def predict_one(params_flat, txy):
        return model.apply(unravel(params_flat), txy[None, :])[0]

    # First derivatives: jacobian wrt (t, x, y) → shape [4, 3]
    # rows = (u, v, p, T); cols = (t, x, y) partials
    d1_fn = jacfwd(predict_one, argnums=1)

    # Second derivatives: hessian-like, shape [4, 3, 3]
    # [field_idx, i, j] = ∂²(u/v/p/T)/(∂txy[i]∂txy[j])
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)

    def residual_per_point(params_flat, txy):
        uvpT = predict_one(params_flat, txy)
        u, v, temp = uvpT[0], uvpT[1], uvpT[3]

        d1 = d1_fn(params_flat, txy)           # [4, 3]
        u_t, u_x, u_y = d1[0, 0], d1[0, 1], d1[0, 2]
        v_t, v_x, v_y = d1[1, 0], d1[1, 1], d1[1, 2]
        p_x, p_y = d1[2, 1], d1[2, 2]
        T_t, T_x, T_y = d1[3, 0], d1[3, 1], d1[3, 2]

        d2 = d2_fn(params_flat, txy)           # [4, 3, 3]
        u_xx, u_yy = d2[0, 1, 1], d2[0, 2, 2]
        v_xx, v_yy = d2[1, 1, 1], d2[1, 2, 2]
        T_xx, T_yy = d2[3, 1, 1], d2[3, 2, 2]

        # Boussinesq momentum + continuity + temperature transport
        r_mx = u_t + u * u_x + v * u_y + p_x - A_NU * (u_xx + u_yy)
        r_my = v_t + u * v_x + v * v_y + p_y - A_NU * (v_xx + v_yy) - A_BUOY * temp
        r_c  = u_x + v_y
        r_T  = T_t + u * T_x + v * T_y - A_KA * (T_xx + T_yy)

        return r_mx ** 2 + r_my ** 2 + r_c ** 2 + r_T ** 2

    # Vectorize over the collocation batch
    residual_batch = vmap(residual_per_point, in_axes=(None, 0))
    predict_batch = vmap(predict_one, in_axes=(None, 0))

    def loss_fn(params_flat, col, ic_txy, ic_uvT, per_batch, dir_batch):
        # PDE residual on interior collocation
        pde_loss = jnp.mean(residual_batch(params_flat, col))

        # initial condition: (u, v, T) at t = 0 (pressure is gauge-free)
        ic_pred = predict_batch(params_flat, ic_txy)
        ic_loss = jnp.mean((ic_pred[:, 0] - ic_uvT[:, 0]) ** 2
                           + (ic_pred[:, 1] - ic_uvT[:, 1]) ** 2
                           + (ic_pred[:, 3] - ic_uvT[:, 2]) ** 2)

        # periodic BC: the two x-faces agree on all four channels
        tb, yb = per_batch[:, 0], per_batch[:, 1]
        x_lo = predict_batch(params_flat, jnp.stack(
            [tb, jnp.full_like(yb, X_MIN), yb], axis=1))
        x_hi = predict_batch(params_flat, jnp.stack(
            [tb, jnp.full_like(yb, X_MAX), yb], axis=1))
        per_loss = jnp.mean((x_lo - x_hi) ** 2)

        # Dirichlet walls: u = v = T = 0 at y = Y_MIN and y = Y_MAX
        w = predict_batch(params_flat, dir_batch)
        dir_loss = jnp.mean(w[:, 0] ** 2 + w[:, 1] ** 2 + w[:, 3] ** 2)

        return pde_loss + WEIGHT_IC * ic_loss + WEIGHT_BC * (per_loss + dir_loss)

    loss_and_grad = jit(jax.value_and_grad(loss_fn))

    optimizer = optax.adam(learning_rate=LR)   # fixed LR, no schedule
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, opt_state, key_quad):
        key1, key2, key3, key4 = key_quad
        idx_col = random.choice(key1, n_col, (BS_COL,))
        idx_ic  = random.choice(key2, n_ic,  (BS_IC,))
        idx_per = random.choice(key3, n_per, (BS_BC,))
        idx_dir = random.choice(key4, n_dir, (BS_BC,))
        loss, grads = loss_and_grad(params_flat, pts_col[idx_col],
                                    pts_ic[idx_ic], uvT_ic[idx_ic],
                                    per_ty[idx_per], pts_dir[idx_dir])
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params_flat, updates)
        return new_params, opt_state, loss

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: jnp.ndarray of shape [BS, 3] with columns (t, x, y).
    Returns dict {"u": [BS], "v": [BS], "p": [BS], "T": [BS]}.
    """
    out = _PREDICT_MODEL.apply(params, X)   # [BS, 4]
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2], "T": out[:, 3]}


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
    keys = random.split(rng_local, 5)
    rng_local = keys[0]
    params_flat, opt_state, loss = update(params_flat, opt_state, tuple(keys[1:]))
    step_count = 1
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        keys = random.split(rng_local, 5)
        rng_local = keys[0]
        params_flat, opt_state, loss = update(params_flat, opt_state, tuple(keys[1:]))
        step_count = it + 1
        last_loss = float(loss)
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(params_flat), it, last_loss)

    return unravel(params_flat), step_count
