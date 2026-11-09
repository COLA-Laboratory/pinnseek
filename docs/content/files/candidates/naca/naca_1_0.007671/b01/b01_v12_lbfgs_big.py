"""b01_v12: Adam -> L-BFGS, larger fixed subset (16384) + deeper history.
Change vs parent (b01_v9): LBFGS_INT 8192->16384 (less subset overfit),
ADAM_WALL 105->75 s, memory_size 10->30. Plain MLP, WEIGHT_BC=1 kept.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 2
OUTPUT_DIM   = 3
INPUT_NAMES  = ("x", "y")
FIELD_NAMES  = ("u", "v", "p")

# --- Domain (farfield box; airfoil cut out via level set in collocation.csv) ---
X_MIN, X_MAX = -3.0, 5.0
Y_MIN, Y_MAX = -2.0, 2.0
HAS_TIME = False

# --- Equation structure ---
TIME_DERIV_ORDER        = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 3   # ru, rv, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("airfoil", "farfield")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "external_airfoil"

# --- Physical parameters ---
RE      = 1000.0       # NS-NACA0012 Re=1000
NU      = 1.0 / RE
U_INF   = 1.0          # freestream speed (v_inf = 0)
AOA_DEG = 7.0          # angle of attack; airfoil rotated -AOA about (0.5, 0)

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
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Vanilla PINN: 5 × 100 tanh, output (u, v, p) ══════════

class PINN(nn.Module):
    """5 hidden layers × 100 nodes, tanh activation everywhere, He-uniform
    init, linear output of 3 channels (u, v, p). A single MLP over (x, y)."""
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, xy):
        # xy: [N, 2]
        kinit = jax.nn.initializers.he_uniform()
        h = xy
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)  # → [N, 3]: (u, v, p)


# ══════════ Geometry: fixed analytic NACA0012 surface ══════════

def _naca_airfoil_points(n: int = 400, thickness: float = 0.12) -> np.ndarray:
    """NACA0012 surface, cosine-clustered, rotated -AOA_DEG about (0.5, 0).
    Returns [n, 2] no-slip boundary points (u=v=0)."""
    beta = np.linspace(0.0, np.pi, n // 2)
    xc = 0.5 * (1.0 - np.cos(beta))                       # cosine spacing, clusters at LE/TE
    yt = 5 * thickness * (0.2969 * np.sqrt(xc) - 0.1260 * xc
                          - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)
    xa = np.concatenate([xc, xc])
    ya = np.concatenate([yt, -yt])                        # upper + lower surfaces
    a = np.radians(-AOA_DEG)
    xr = 0.5 + (xa - 0.5) * np.cos(a) - ya * np.sin(a)
    yr = 0.0 + (xa - 0.5) * np.sin(a) + ya * np.cos(a)
    return np.stack([xr, yr], axis=1).astype(np.float32)


# ══════════ Data: fixed fluid grid + airfoil/farfield BCs ══════════

def _build_dataset():
    """Read the reference grid; return the fixed point sets.

    * ``pts_int``: fluid collocation points (``phi <= 0``, outer edges
      excluded) — the fixed PDE-residual set (Scale-PINN convention).
    * ``pts_bc`` / ``u_bc`` / ``v_bc``: airfoil (no-slip 0,0) + farfield
      (freestream U_INF,0) boundary points and their target velocities.
    """
    ref = np.loadtxt(_locate("collocation.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    xy, phi = ref[:, 0:2], ref[:, 2]

    edge = (
        (xy[:, 0] == X_MIN) | (xy[:, 0] == X_MAX)
        | (xy[:, 1] == Y_MIN) | (xy[:, 1] == Y_MAX)
    )
    fluid = (phi <= 0.0)

    pts_int = xy[fluid & ~edge].astype(np.float32)        # fixed collocation
    pts_far = xy[edge].astype(np.float32)                 # farfield → freestream
    pts_air = _naca_airfoil_points(400)                   # airfoil → no-slip

    pts_bc = np.vstack([pts_far, pts_air])
    u_bc = np.concatenate([np.full(len(pts_far), U_INF, np.float32),
                           np.zeros(len(pts_air), np.float32)])
    v_bc = np.zeros(len(pts_bc), np.float32)

    return (jnp.array(pts_int), jnp.array(pts_bc), jnp.array(u_bc), jnp.array(v_bc))


# ══════════ Training hyperparameters ══════════

LR        = 1e-3      # Adam LR (phase 1), fixed
WEIGHT_BC = 1.0       # BC weight (empirical sweet spot for this task)
BS_INT    = 400       # interior collocation points per minibatch (Adam)
BS_BC     = 100       # boundary points per minibatch (Adam)
ADAM_WALL = 75.0      # wall-seconds of Adam before switching to L-BFGS
LBFGS_INT = 16384     # fixed interior subset for the deterministic L-BFGS loss
LBFGS_MEM = 30        # L-BFGS history size
SAVE_MARGIN_S = 5.0   # stop the loop this early so device_get + pickling
                      # finish before eval.py's TRAIN_TIME wall


# ══════════ Loss: PDE residual via autograd, BC L2 ══════════

def _make_train(rng_init):
    pts_int, pts_bc, u_bc, v_bc = _build_dataset()
    n_int, n_bc = len(pts_int), len(pts_bc)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def predict_one(params_flat, xy):
        return model.apply(unravel(params_flat), xy[None, :])[0]   # [3]

    d1_fn = jacfwd(predict_one, argnums=1)                          # [3, 2]
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)       # [3, 2, 2]

    def residual_per_point(params_flat, xy):
        uvp = predict_one(params_flat, xy)
        u, v = uvp[0], uvp[1]
        d1 = d1_fn(params_flat, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        d2 = d2_fn(params_flat, xy)
        u_xx, u_yy = d2[0, 0, 0], d2[0, 1, 1]
        v_xx, v_yy = d2[1, 0, 0], d2[1, 1, 1]
        r_c  = u_x + v_y
        r_mx = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
        r_my = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
        return r_c ** 2 + r_mx ** 2 + r_my ** 2

    residual_batch = vmap(residual_per_point, in_axes=(None, 0))

    def loss_fn(params_flat, xy_int, xy_bc_batch, u_bc_batch, v_bc_batch):
        pde_loss = jnp.mean(residual_batch(params_flat, xy_int))
        bc_uvp = vmap(predict_one, in_axes=(None, 0))(params_flat, xy_bc_batch)
        u_pred, v_pred = bc_uvp[:, 0], bc_uvp[:, 1]
        bc_loss = jnp.mean((u_pred - u_bc_batch) ** 2 + (v_pred - v_bc_batch) ** 2)
        return pde_loss + WEIGHT_BC * bc_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, opt_state, key_pair):
        key1, key2 = key_pair
        idx_int = random.choice(key1, n_int, (BS_INT,))
        idx_bc  = random.choice(key2, n_bc,  (BS_BC,))
        loss, grads = loss_and_grad(params_flat, pts_int[idx_int],
                                    pts_bc[idx_bc], u_bc[idx_bc], v_bc[idx_bc])
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params_flat, updates), opt_state, loss

    # ---- Phase-2 L-BFGS: deterministic loss on a FIXED collocation subset ----
    idx_fixed = random.choice(random.PRNGKey(12345), n_int, (LBFGS_INT,), replace=False)
    fixed_int = pts_int[idx_fixed]                     # frozen for all L-BFGS iters
    fixed_bc, fixed_ubc, fixed_vbc = pts_bc, u_bc, v_bc   # all BC points (few)

    def loss_fixed(params_flat):
        return loss_fn(params_flat, fixed_int, fixed_bc, fixed_ubc, fixed_vbc)

    lbfgs = optax.lbfgs(memory_size=LBFGS_MEM)
    value_and_grad_state = optax.value_and_grad_from_state(loss_fixed)

    @jit
    def lbfgs_step(params_flat, state):
        value, grad = value_and_grad_state(params_flat, state=state)
        updates, state = lbfgs.update(grad, state, params_flat,
                                      value=value, grad=grad, value_fn=loss_fixed)
        return optax.apply_updates(params_flat, updates), state

    return flat_params, opt_state, update, lbfgs, lbfgs_step, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: [BS, 2] with columns (x, y). Returns {"u","v","p"} each [BS]."""
    out = _PREDICT_MODEL.apply(params, X)   # [BS, 3]
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Two-phase Adam -> L-BFGS training; whole process finishes within
    TRAIN_TIME s of _T0. Returns (params_pytree, step_count)."""
    (params_flat, opt_state, update, lbfgs, lbfgs_step,
     unravel, rng_local) = _make_train(rng)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S

    # ── Phase 1: stochastic Adam (escape saddles, reach the plateau) ──
    key1, key2, rng_local = random.split(rng_local, 3)
    params_flat, opt_state, loss = update(params_flat, opt_state, (key1, key2))
    step_count = 1
    adam_deadline = min(_T0 + ADAM_WALL, deadline)
    for it in range(1, MAX_ITER):
        if time.time() > adam_deadline:
            break
        key1, key2, rng_local = random.split(rng_local, 3)
        params_flat, opt_state, loss = update(params_flat, opt_state, (key1, key2))
        step_count = it + 1

    # ── Phase 2: deterministic L-BFGS on the fixed subset (break plateau) ──
    state = lbfgs.init(params_flat)
    # warm-up compiles lbfgs_step (counted against the wall)
    params_flat, state = lbfgs_step(params_flat, state)
    step_count += 1
    while time.time() < deadline:
        params_flat, state = lbfgs_step(params_flat, state)
        step_count += 1

    return unravel(params_flat), step_count
