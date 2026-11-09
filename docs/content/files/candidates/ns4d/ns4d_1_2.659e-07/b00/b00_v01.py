"""Vanilla PINN baseline for 3D unsteady Navier-Stokes (tree root).

Contract (do not change):
  - ``predict_fn(params, X) -> {"u","v","w","wx","wy","wz"}``
  - ``train(rng, eval_callback=None) -> (params, step_count)``
  - Frozen PDE-CONSTANTS header is byte-identical across all descendants.

Time budget (frozen): ``TRAIN_TIME = 300`` s of wall-clock for the
WHOLE process, anchored at ``_T0`` (process start) — imports, JIT
compilation, training and param save all included; ``eval.py`` kills
the process at this wall. ``train()`` must return params with margin
to spare (this baseline stops ``SAVE_MARGIN_S = 5`` s early).
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

import math as _math

# --- I/O shape ---
INPUT_DIM    = 4
OUTPUT_DIM   = 3          # the network emits velocity only
INPUT_NAMES  = ("t", "x", "y", "z")
FIELD_NAMES  = ("u", "v", "w", "wx", "wy", "wz")   # 3 emitted + 3 derived:
                          # (wx, wy, wz) = curl(u, v, w), by autodiff

# --- Domain ---
T_MIN, T_MAX = 0.0, 5.0
X_MIN, X_MAX = 0.0, 2.0 * _math.pi
Y_MIN, Y_MAX = 0.0, 2.0 * _math.pi
Z_MIN, Z_MAX = 0.0, 2.0 * _math.pi
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 3   # lap(curl u) is third order in the velocity
N_RESIDUAL_COMPONENTS   = 4   # three vorticity-transport components + continuity
HAS_SOURCE              = True   # the manufactured forcing f(t, x, y, z)

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
PERIODIC_AXES  = ("x", "y", "z")   # period 2*pi on each
GEOMETRY_KIND  = "periodic_box"

# --- Physical parameters ---
NU = 0.05      # viscosity
# Forcing (manufactured):
#   f_x = -FA e^{-18 NU t} sin(4y) sin(2z)
#   f_y = -FA e^{-18 NU t} sin(4x) sin(2z)
#   f_z = +FA e^{-18 NU t} sin(4x) sin(4y)
FA = 6.0

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


# ══════════ Vanilla PINN: 5 × 100 tanh, output (u, v, w) ══════════

class PINN(nn.Module):
    """5 hidden layers × 100 nodes, tanh activation everywhere, He-uniform
    init, linear output of 3 channels (u, v, w). No feature mapping, no
    shared-trunk / split-head structure — a single MLP over (t, x, y, z)."""
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, txyz):
        # txyz: [N, 4]
        kinit = jax.nn.initializers.he_uniform()
        h = txyz
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)  # → [N, 3]: (u, v, w)


# ══════════ Forcing ══════════

def _forcing(t, x, y, z):
    """The manufactured body force on the vorticity equation."""
    e = jnp.exp(-18.0 * NU * t)
    return jnp.stack([-FA * e * jnp.sin(4.0 * y) * jnp.sin(2.0 * z),
                      -FA * e * jnp.sin(4.0 * x) * jnp.sin(2.0 * z),
                      FA * e * jnp.sin(4.0 * x) * jnp.sin(4.0 * y)])


# ══════════ Data: analytic collocation + periodic BC, IC from file ══════════

def _build_dataset():
    """Collocation grid + IC points (from ns4d_ic.csv) + periodic-BC samples.

    Returns:
      pts_col : [Nc, 4] interior collocation coordinates (PDE residual)
      pts_ic  : [Ni, 4] (t=0, x, y, z) coordinates for the initial condition
      uw_ic   : [Ni, 6] IC targets (u, v, w, wx, wy, wz) at t = 0
      bc_tss  : [Nb, 3] (t, s1, s2) samples for the periodic constraints — the
                two free coordinates of whichever face pair is being matched
    """
    # collocation: uniform grid, strict interior in time (t=0 IC plane
    # excluded — the PDE residual is not enforced on the supervised
    # initial condition); x, y, z on the half-open torus [0, 2*pi).
    nt, ns = 21, 20
    ts = np.linspace(T_MIN, T_MAX, nt)[1:]
    xs = np.linspace(X_MIN, X_MAX, ns, endpoint=False)
    T, X, Y, Z = np.meshgrid(ts, xs, xs, xs, indexing="ij")
    pts_col = np.stack([T.ravel(), X.ravel(), Y.ravel(), Z.ravel()],
                       axis=1).astype(np.float32)

    # initial condition (the ONLY field data we may read)
    ic = np.loadtxt(_locate("ns4d_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, y_ic, z_ic = ic[:, 0], ic[:, 1], ic[:, 2]
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic, y_ic, z_ic], axis=1).astype(np.float32)
    uw_ic = ic[:, 3:9].astype(np.float32)

    # periodic BC: (t, s1, s2) samples at which each opposite face pair agrees
    nb = 24
    tb = np.linspace(T_MIN, T_MAX, nb)
    sb = np.linspace(X_MIN, X_MAX, nb, endpoint=False)
    TB, S1, S2 = np.meshgrid(tb, sb, sb, indexing="ij")
    bc_tss = np.stack([TB.ravel(), S1.ravel(), S2.ravel()], axis=1).astype(np.float32)

    return (
        jnp.array(pts_col),
        jnp.array(pts_ic),
        jnp.array(uw_ic),
        jnp.array(bc_tss),
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
PRED_CHUNK    = 20000 # predict_fn batches the per-point Jacobians


# ══════════ Vorticity and residual, both from autodiff ══════════

def _make_fns(unravel, model):
    """Velocity, vorticity and the per-point squared residual, all by autodiff."""

    def u_one(params_flat, q):
        """q: [4] = (t, x, y, z) → [3] velocity."""
        return model.apply(unravel(params_flat), q[None, :])[0]

    def w_one(params_flat, q):
        """Vorticity w = curl u at q → [3]. One Jacobian of the velocity."""
        J = jacfwd(u_one, argnums=1)(params_flat, q)   # [3, 4], cols = (t,x,y,z)
        return jnp.stack([J[2, 2] - J[1, 3],           # dw/dy - dv/dz
                          J[0, 3] - J[2, 1],           # du/dz - dw/dx
                          J[1, 1] - J[0, 2]])          # dv/dx - du/dy

    def residual_per_point(params_flat, q):
        u = u_one(params_flat, q)
        w = w_one(params_flat, q)

        Ju = jacfwd(u_one, argnums=1)(params_flat, q)                       # [3, 4]
        Jw = jacfwd(w_one, argnums=1)(params_flat, q)                       # [3, 4]
        Hw = jacfwd(jacfwd(w_one, argnums=1), argnums=1)(params_flat, q)    # [3, 4, 4]
        lap_w = Hw[:, 1, 1] + Hw[:, 2, 2] + Hw[:, 3, 3]

        adv     = Jw[:, 1] * u[0] + Jw[:, 2] * u[1] + Jw[:, 3] * u[2]   # (u·∇)w
        stretch = Ju[:, 1] * w[0] + Ju[:, 2] * w[1] + Ju[:, 3] * w[2]   # (w·∇)u

        # vorticity transport (3 components) + continuity
        r   = Jw[:, 0] + adv - stretch - NU * lap_w - _forcing(q[0], q[1], q[2], q[3])
        r_c = Ju[0, 1] + Ju[1, 2] + Ju[2, 3]

        return jnp.sum(r ** 2) + r_c ** 2

    return u_one, w_one, residual_per_point


# ══════════ Loss: PDE residual via autograd, IC + periodic-BC L2 ══════════

def _make_train(rng_init):
    """Build the JIT'd minibatch + residual + loss + update closures."""
    pts_col, pts_ic, uw_ic, bc_tss = _build_dataset()
    n_col, n_ic, n_bc = len(pts_col), len(pts_ic), len(bc_tss)

    model = PINN()
    key, rng = random.split(rng_init)
    dummy = jnp.zeros([1, INPUT_DIM])
    params = model.init(key, dummy)
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    u_one, w_one, residual_per_point = _make_fns(unravel, model)

    # Vectorize over the batches
    residual_batch = vmap(residual_per_point, in_axes=(None, 0))
    u_batch = vmap(u_one, in_axes=(None, 0))
    w_batch = vmap(w_one, in_axes=(None, 0))

    def loss_fn(params_flat, col, ic_q, ic_uw, bc_batch):
        # PDE residual on interior collocation
        pde_loss = jnp.mean(residual_batch(params_flat, col))

        # initial condition: velocity and vorticity at t = 0
        ic_loss = (jnp.mean((u_batch(params_flat, ic_q) - ic_uw[:, 0:3]) ** 2)
                   + jnp.mean((w_batch(params_flat, ic_q) - ic_uw[:, 3:6]) ** 2))

        # periodic BC: each opposite face pair agrees on the velocity
        tb, s1, s2 = bc_batch[:, 0], bc_batch[:, 1], bc_batch[:, 2]
        lo = jnp.full_like(s1, X_MIN)
        hi = jnp.full_like(s1, X_MAX)
        bc_loss = 0.0
        for face in range(3):
            if face == 0:      # x = 0 vs x = 2*pi
                a = jnp.stack([tb, lo, s1, s2], axis=1)
                b = jnp.stack([tb, hi, s1, s2], axis=1)
            elif face == 1:    # y = 0 vs y = 2*pi
                a = jnp.stack([tb, s1, lo, s2], axis=1)
                b = jnp.stack([tb, s1, hi, s2], axis=1)
            else:              # z = 0 vs z = 2*pi
                a = jnp.stack([tb, s1, s2, lo], axis=1)
                b = jnp.stack([tb, s1, s2, hi], axis=1)
            bc_loss = bc_loss + jnp.mean(
                (u_batch(params_flat, a) - u_batch(params_flat, b)) ** 2)

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
                                    pts_ic[idx_ic], uw_ic[idx_ic], bc_tss[idx_bc])
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params_flat, updates)
        return new_params, opt_state, loss

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: jnp.ndarray of shape [BS, 4] with columns (t, x, y, z).
    Returns dict {"u","v","w","wx","wy","wz"} — the velocity the network
    emits, plus its curl, taken by automatic differentiation.
    """
    X = jnp.asarray(X)

    def u_one(q):
        return _PREDICT_MODEL.apply(params, q[None, :])[0]

    def w_one(q):
        J = jacfwd(u_one)(q)                    # [3, 4]
        return jnp.stack([J[2, 2] - J[1, 3],
                          J[0, 3] - J[2, 1],
                          J[1, 1] - J[0, 2]])

    u_batch, w_batch = vmap(u_one), vmap(w_one)
    us, ws = [], []
    for i in range(0, X.shape[0], PRED_CHUNK):
        xb = X[i:i + PRED_CHUNK]
        us.append(u_batch(xb))
        ws.append(w_batch(xb))
    U = jnp.concatenate(us, axis=0)
    W = jnp.concatenate(ws, axis=0)
    return {"u": U[:, 0], "v": U[:, 1], "w": U[:, 2],
            "wx": W[:, 0], "wy": W[:, 1], "wz": W[:, 2]}


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
