"""Vanilla PINN baseline for the 3D complex-geometry Poisson problem (tree root).

Contract (do not change):
  - ``predict_fn(params, X) -> {"u"}``
  - ``train(rng, eval_callback=None) -> (params, step_count)``
  - Frozen PDE-CONSTANTS header is byte-identical across all descendants.

Time budget (frozen): ``TRAIN_TIME = 60`` s of wall-clock for the WHOLE
process, anchored at ``_T0`` (process start) — imports, JIT compilation,
training and param save all included; ``eval.py`` kills the process at this
wall. ``train()`` must return params with margin to spare (this baseline
stops ``SAVE_MARGIN_S = 5`` s early).

Data discipline: this task ships no field data at all. The geometry, the
source and every boundary are analytic, so the baseline (and any
descendant) reads nothing. Never read ``task/ref_data.csv`` — it is the
scoring truth, used by ``task/eval.py`` for scoring alone.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 1
INPUT_NAMES  = ("x", "y", "z")
FIELD_NAMES  = ("u",)

# --- Domain ---
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.0
HAS_TIME = False

# --- Equation structure ---
TIME_DERIV_ORDER        = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 1   # r
HAS_SOURCE              = True   # and it does not vanish with u, so u ≡ 0 is
                                 # nowhere near a solution of the residual

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi",
                  "sphere_0", "sphere_1")
PERIODIC_AXES  = ()       # none: homogeneous Neumann on every boundary
GEOMETRY_KIND  = "cube_minus_spheres"

# --- Geometry (unit cube with two spherical voids) ---
SPHERES = ((0.4, 0.3, 0.6, 0.2),    # (cx, cy, cz, radius), removed from the cube
           (0.6, 0.7, 0.6, 0.2))

# --- Physical parameters ---
# Residual: -(u_xx + u_yy + u_zz) + K(z)^2 u - f(x, y, z) = 0
Z_IFACE      = 0.5      # the screening wavenumber jumps across this plane
K_LO, K_HI   = 8.0, 10.0            # K(z) = K_LO for z < Z_IFACE, else K_HI;
                                    # only u_zz jumps there, u and grad u stay
                                    # continuous — the jump is zeroth order
SRC_A1, SRC_A2 = 20.0, 100.0        # source amplitudes
SRC_M1, SRC_M2, SRC_M3 = 1.0, 10.0, 5.0   # source wavenumbers, sin(M pi ·);
                                    # M2 = 10 sets the finest scale, 5 periods
                                    # of the source across y

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 60.0      # total wall-clock seconds for the WHOLE process,
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
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


def screening(z):
    """K(z)^2 — the zeroth-order coefficient, piecewise constant in z."""
    return jnp.where(z < Z_IFACE, K_LO ** 2, K_HI ** 2)


def source(x, y, z):
    """The forcing f(x, y, z); independent of u."""
    r2 = x ** 2 + y ** 2 + z ** 2
    part1 = (jnp.exp(jnp.sin(SRC_M1 * jnp.pi * x)
                     + jnp.sin(SRC_M2 * jnp.pi * y)
                     + jnp.sin(SRC_M3 * jnp.pi * z))
             * (r2 - 1.0) / (r2 + 1.0))
    part2 = (jnp.sin(SRC_M1 * jnp.pi * x)
             * jnp.sin(SRC_M2 * jnp.pi * y)
             * jnp.sin(SRC_M3 * jnp.pi * z))
    return SRC_A1 * part1 + SRC_A2 * part2


# ══════════ Vanilla PINN: 5 × 100 tanh, output u ══════════

class PINN(nn.Module):
    """5 hidden layers × 100 nodes, tanh activation everywhere, He-uniform
    init, linear output of 1 channel (u). No feature mapping, no
    shared-trunk / split-head structure — a single MLP over (x, y, z)."""
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
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)  # → [N, 1]: u


# ══════════ Data: analytic collocation + analytic Neumann boundaries ══════════

def _build_dataset():
    """Collocation points and the two boundary sets, with outward normals.

    The geometry is two spheres cut out of a cube, so every boundary set is
    generated analytically here — there is no shipped geometry file, and no
    field data of any kind is read.

    Returns:
      pts_col  : [Nc, 3] interior collocation, both spheres removed
      pts_face : [Nf, 3] cube-face points     -> du/dn = 0
      nrm_face : [Nf, 3] unit normals of the cube faces
      pts_sph  : [Ns, 3] sphere-surface points -> du/dn = 0
      nrm_sph  : [Ns, 3] unit normals of the sphere surfaces (radial)
    """
    # ---- interior collocation: uniform grid minus the two spheres ----
    nx = ny = nz = 66
    xs = np.linspace(X_MIN, X_MAX, nx)[1:-1]      # the six cube faces are
    ys = np.linspace(Y_MIN, Y_MAX, ny)[1:-1]      # boundary planes, so the
    zs = np.linspace(Z_MIN, Z_MAX, nz)[1:-1]      # residual grid stays open
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    inside = np.zeros(X.shape, dtype=bool)
    for cx, cy, cz, r in SPHERES:
        inside |= (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= r ** 2
    pts_col = np.stack([X[~inside], Y[~inside], Z[~inside]], axis=1).astype(np.float32)

    # ---- cube faces: six planes, axis-aligned outward normals ----
    nf = 48
    a = np.linspace(0.0, 1.0, nf)
    A, B = np.meshgrid(a, a, indexing="ij")
    A, B = A.ravel(), B.ravel()
    one, zero = np.ones_like(A), np.zeros_like(A)
    faces, normals = [], []
    for axis, (lo, hi) in enumerate([(X_MIN, X_MAX), (Y_MIN, Y_MAX), (Z_MIN, Z_MAX)]):
        for value, sign in ((lo, -1.0), (hi, +1.0)):
            cols = [A, B]
            pts = [None, None, None]
            pts[axis] = np.full_like(A, value)
            for k in (i for i in range(3) if i != axis):
                pts[k] = cols.pop(0)
            faces.append(np.stack(pts, axis=1))
            nrm = [zero, zero, zero]
            nrm[axis] = sign * one
            normals.append(np.stack(nrm, axis=1))
    pts_face = np.concatenate(faces).astype(np.float32)
    nrm_face = np.concatenate(normals).astype(np.float32)

    # ---- sphere surfaces: Fibonacci lattice, radial normals ----
    ns = 1024
    idx = np.arange(ns) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / ns)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
    unit = np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)
    pts_sph = np.concatenate(
        [np.array([cx, cy, cz]) + r * unit for cx, cy, cz, r in SPHERES]).astype(np.float32)
    nrm_sph = np.concatenate([-unit for _ in SPHERES]).astype(np.float32)
    # the normal points out of the fluid domain, i.e. into the removed sphere

    return (jnp.array(pts_col),
            jnp.array(pts_face), jnp.array(nrm_face),
            jnp.array(pts_sph), jnp.array(nrm_sph))


# ══════════ Training hyperparameters (intentionally minimal) ══════════

LR        = 1e-3      # fixed, no schedule
WEIGHT_BC = 1.0       # vanilla: no loss re-weighting
BS_COL    = 1000      # collocation points per minibatch
BS_BC     = 128       # points per boundary set per minibatch
SAVE_MARGIN_S = 5.0   # stop the loop this early so device_get + pickling
                      # finish before eval.py's TRAIN_TIME wall


# ══════════ Loss: PDE residual via autograd, Neumann-BC L2 ══════════

def _make_train(rng_init):
    """Build the JIT'd minibatch + residual + loss + update closures."""
    pts_col, pts_face, nrm_face, pts_sph, nrm_sph = _build_dataset()
    n_col, n_face, n_sph = len(pts_col), len(pts_face), len(pts_sph)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def predict_one(params_flat, xyz):
        return model.apply(unravel(params_flat), xyz[None, :])[0]   # [1]

    d1_fn = jacfwd(predict_one, argnums=1)                          # [1, 3]
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)       # [1, 3, 3]

    def residual_per_point(params_flat, xyz):
        u = predict_one(params_flat, xyz)[0]
        d2 = d2_fn(params_flat, xyz)
        lap = d2[0, 0, 0] + d2[0, 1, 1] + d2[0, 2, 2]
        r = -lap + screening(xyz[2]) * u - source(xyz[0], xyz[1], xyz[2])
        return r ** 2

    def normal_deriv_per_point(params_flat, xyz, nrm):
        d1 = d1_fn(params_flat, xyz)          # [1, 3]
        return jnp.dot(d1[0], nrm) ** 2

    residual_batch = vmap(residual_per_point, in_axes=(None, 0))
    neumann_batch = vmap(normal_deriv_per_point, in_axes=(None, 0, 0))
    predict_batch = vmap(predict_one, in_axes=(None, 0))

    def loss_fn(params_flat, col, b_face, n_face_b, b_sph, n_sph_b):
        pde_loss = jnp.mean(residual_batch(params_flat, col))

        # homogeneous Neumann: du/dn = 0 on the cube faces and both spheres
        l_face = jnp.mean(neumann_batch(params_flat, b_face, n_face_b))
        l_sph = jnp.mean(neumann_batch(params_flat, b_sph, n_sph_b))

        bc_loss = l_face + l_sph
        return pde_loss + WEIGHT_BC * bc_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, opt_state, keys):
        k1, k2, k3 = keys
        i_face = random.choice(k2, n_face, (BS_BC,))
        i_sph = random.choice(k3, n_sph, (BS_BC,))
        loss, grads = loss_and_grad(
            params_flat,
            pts_col[random.choice(k1, n_col, (BS_COL,))],
            pts_face[i_face], nrm_face[i_face],
            pts_sph[i_sph], nrm_sph[i_sph],
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params_flat, updates), opt_state, loss

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: [BS, 3] with columns (x, y, z). Returns {"u"} of shape [BS]."""
    out = _PREDICT_MODEL.apply(params, X)   # [BS, 1]
    return {"u": out[:, 0]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam training; the whole process finishes within
    TRAIN_TIME s of _T0. Returns (params_pytree, step_count)."""
    params_flat, opt_state, update, unravel, rng_local = _make_train(rng)

    # Warmup step JIT-compiles the update; counted.
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
