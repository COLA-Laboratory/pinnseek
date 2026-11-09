"""Vanilla PINN baseline for 3D wing NS, Re=1000, 7 deg AoA (tree root).

Contract (do not change):
  - ``predict_fn(params, X) -> {"u","v","w","p"}``
  - ``train(rng, eval_callback=None) -> (params, step_count)``
  - Frozen PDE-CONSTANTS header is byte-identical across all descendants.

Time budget (frozen): ``TRAIN_TIME = 480`` s of wall-clock for the WHOLE
process, anchored at ``_T0`` (process start) — imports, JIT compilation,
training and param save all included; ``eval.py`` kills the process at
this wall. ``train()`` must return params with margin to spare (this
baseline stops ``SAVE_MARGIN_S = 8`` s early).

Fixed sampling (Scale-PINN convention): PDE collocation points are the
fixed fluid grid points (``phi <= 0``) read from ``task/collocation.npy``
(``x, y, z, phi``) — never resampled in space, only minibatched. Wing
no-slip points are the fixed set in ``task/wing_surface.npy``; the outer
box faces are generated analytically here. The scoring truth
``task/ref_data.npy`` is read by ``eval.py`` alone.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 4
INPUT_NAMES  = ("x", "y", "z")
FIELD_NAMES  = ("u", "v", "w", "p")

# --- Domain (farfield box; wing cut out via level set in collocation.npy) ---
X_MIN, X_MAX = -1.5, 3.5
Y_MIN, Y_MAX = -2.0, 2.0
Z_MIN, Z_MAX = 0.0, 3.5
HAS_TIME = False

# --- Equation structure ---
TIME_DERIV_ORDER        = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 4   # ru, rv, rw, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("wing", "farfield", "outlet", "symmetry")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "external_wing_half_model"

# --- Physical parameters ---
RE      = 1000.0       # 3D finite wing, Re=1000
NU      = 1.0 / RE
U_INF   = 1.0          # freestream speed (v_inf = w_inf = 0)
AOA_DEG = 7.0          # angle of attack; wing rotated -AOA about (0.5, 0)

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
    base = _os.environ.get("FORGE_TASK_DIR")
    if base:
        cand = Path(base) / name
        if cand.is_file():
            return cand
    here = Path.cwd()
    for b in [here, *here.parents]:
        cand = b / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Vanilla PINN: 5 × 100 tanh, output (u, v, w, p) ══════════

class PINN(nn.Module):
    """5 hidden layers × 100 nodes, tanh activation everywhere, He-uniform
    init, linear output of 4 channels (u, v, w, p). A single MLP over
    (x, y, z)."""
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, xyz):
        # xyz: [N, 3]
        kinit = jax.nn.initializers.he_uniform()
        h = xyz
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)  # → [N, 4]


# ══════════ Data: fixed fluid grid + wing / box boundary sets ══════════

def _build_dataset():
    """Read the collocation grid; return the fixed point sets.

    Every set is sliced out of the one shipped grid — there is no sampling
    step here and no RNG, so all descendants see identical points.

    * ``pts_int``: strict-interior fluid points — the fixed PDE-residual
      set (Scale-PINN convention).
    * ``pts_wing``: wing no-slip points (a separate fixed file).
    * ``pts_far``: inlet + lateral + top faces, freestream (U_INF, 0, 0).
    * ``pts_out``: outlet face, p = 0.
    * ``pts_sym``: symmetry plane, w = 0 and du/dz = dv/dz = 0. The wing's
      root footprint is already absent from the grid, so this is the fluid
      part of z = 0 only.

    Outlet points are excluded from the freestream set: those two faces
    meet along an edge and their conditions are different, so each point
    is claimed once.
    """
    col = np.load(_locate("collocation.npy"))          # [N, 4] x,y,z,phi
    xyz = np.ascontiguousarray(col[:, :3], dtype=np.float32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    tol = 1e-5

    on_inlet = np.abs(x - X_MIN) < tol
    on_out   = np.abs(x - X_MAX) < tol
    on_lat   = (np.abs(y - Y_MIN) < tol) | (np.abs(y - Y_MAX) < tol)
    on_top   = np.abs(z - Z_MAX) < tol
    on_sym   = np.abs(z - Z_MIN) < tol
    on_face  = on_inlet | on_out | on_lat | on_top | on_sym

    pts_int  = xyz[~on_face]
    pts_far  = xyz[(on_inlet | on_lat | on_top) & ~on_out]
    pts_out  = xyz[on_out]
    pts_sym  = xyz[on_sym]
    pts_wing = np.load(_locate("wing_surface.npy")).astype(np.float32)
    return (jnp.array(pts_int), jnp.array(pts_wing),
            jnp.array(pts_far), jnp.array(pts_out), jnp.array(pts_sym))


# ══════════ Training hyperparameters (intentionally minimal) ══════════

LR         = 1e-3     # fixed, no schedule
WEIGHT_BC  = 1.0      # default — no PDE-vs-BC re-weighting (vanilla)
BS_INT     = 1024     # interior collocation points per minibatch
BS_BC      = 256      # points per boundary set per minibatch
SAVE_MARGIN_S = 8.0   # stop the loop this early so device_get + pickling
                      # finish before eval.py's TRAIN_TIME wall


# ══════════ Loss: PDE residual via autograd, BC L2 ══════════

def _make_train(rng_init):
    pts_int, pts_wing, pts_far, pts_out, pts_sym = _build_dataset()
    n_int, n_wing = len(pts_int), len(pts_wing)
    n_far, n_out, n_sym = len(pts_far), len(pts_out), len(pts_sym)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def predict_one(params_flat, xyz):
        return model.apply(unravel(params_flat), xyz[None, :])[0]   # [4]

    d1_fn = jacfwd(predict_one, argnums=1)                          # [4, 3]
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)       # [4, 3, 3]

    def residual_per_point(params_flat, xyz):
        uvwp = predict_one(params_flat, xyz)
        u, v, w = uvwp[0], uvwp[1], uvwp[2]
        d1 = d1_fn(params_flat, xyz)
        d2 = d2_fn(params_flat, xyz)
        u_x, u_y, u_z = d1[0, 0], d1[0, 1], d1[0, 2]
        v_x, v_y, v_z = d1[1, 0], d1[1, 1], d1[1, 2]
        w_x, w_y, w_z = d1[2, 0], d1[2, 1], d1[2, 2]
        p_x, p_y, p_z = d1[3, 0], d1[3, 1], d1[3, 2]
        lap_u = d2[0, 0, 0] + d2[0, 1, 1] + d2[0, 2, 2]
        lap_v = d2[1, 0, 0] + d2[1, 1, 1] + d2[1, 2, 2]
        lap_w = d2[2, 0, 0] + d2[2, 1, 1] + d2[2, 2, 2]
        r_c  = u_x + v_y + w_z
        r_mx = u * u_x + v * u_y + w * u_z + p_x - NU * lap_u
        r_my = u * v_x + v * v_y + w * v_z + p_y - NU * lap_v
        r_mz = u * w_x + v * w_y + w * w_z + p_z - NU * lap_w
        return r_c ** 2 + r_mx ** 2 + r_my ** 2 + r_mz ** 2

    residual_batch = vmap(residual_per_point, in_axes=(None, 0))
    predict_batch = vmap(predict_one, in_axes=(None, 0))

    def sym_residual_per_point(params_flat, xyz):
        """Symmetry plane: w = 0 and du/dz = dv/dz = 0."""
        uvwp = predict_one(params_flat, xyz)
        d1 = d1_fn(params_flat, xyz)
        return uvwp[2] ** 2 + d1[0, 2] ** 2 + d1[1, 2] ** 2

    sym_batch = vmap(sym_residual_per_point, in_axes=(None, 0))

    def loss_fn(params_flat, xyz_int, xyz_wing, xyz_far, xyz_out, xyz_sym):
        pde_loss = jnp.mean(residual_batch(params_flat, xyz_int))
        wing = predict_batch(params_flat, xyz_wing)
        bc_wing = jnp.mean(wing[:, 0] ** 2 + wing[:, 1] ** 2 + wing[:, 2] ** 2)
        far = predict_batch(params_flat, xyz_far)
        bc_far = jnp.mean((far[:, 0] - U_INF) ** 2 + far[:, 1] ** 2 + far[:, 2] ** 2)
        out = predict_batch(params_flat, xyz_out)
        bc_out = jnp.mean(out[:, 3] ** 2)
        bc_sym = jnp.mean(sym_batch(params_flat, xyz_sym))
        return pde_loss + WEIGHT_BC * (bc_wing + bc_far + bc_out + bc_sym)

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, opt_state, keys):
        k0, k1, k2, k3, k4 = keys
        loss, grads = loss_and_grad(
            params_flat,
            pts_int[random.choice(k0, n_int, (BS_INT,))],
            pts_wing[random.choice(k1, n_wing, (BS_BC,))],
            pts_far[random.choice(k2, n_far, (BS_BC,))],
            pts_out[random.choice(k3, n_out, (BS_BC,))],
            pts_sym[random.choice(k4, n_sym, (BS_BC,))],
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params_flat, updates), opt_state, loss

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: [BS, 3] with columns (x, y, z). Returns {"u","v","w","p"} each [BS]."""
    out = _PREDICT_MODEL.apply(params, X)   # [BS, 4]
    return {"u": out[:, 0], "v": out[:, 1], "w": out[:, 2], "p": out[:, 3]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam training; whole process finishes within
    TRAIN_TIME s of _T0. Returns (params_pytree, step_count)."""
    params_flat, opt_state, update, unravel, rng_local = _make_train(rng)

    # Warmup step JIT-compiles the update; counted.
    keys = random.split(rng_local, 6)
    rng_local = keys[0]
    params_flat, opt_state, loss = update(params_flat, opt_state, keys[1:])
    step_count = 1
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        keys = random.split(rng_local, 6)
        rng_local = keys[0]
        params_flat, opt_state, loss = update(params_flat, opt_state, keys[1:])
        step_count = it + 1
        last_loss = float(loss)
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(params_flat), it, last_loss)

    return unravel(params_flat), step_count
