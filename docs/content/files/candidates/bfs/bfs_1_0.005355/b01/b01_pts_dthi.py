"""Pseudo-time-stepped PINN, DT = 2.0 (kb1/073 step-size sweep).

Change vs parent b01_pts.py: the pseudo-time step DT is 2.0 instead of
0.5 — smaller DT shifts the Jacobian spectrum further left (better
conditioning) but marches pseudo-time more slowly. Nothing else changes.
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

# --- Domain (channel downstream of the step; step face = lower half of x=X_MIN) ---
X_MIN, X_MAX = 0.0, 15.0
Y_MIN, Y_MAX = -0.5, 0.5
HAS_TIME = False

# --- Equation structure ---
TIME_DERIV_ORDER        = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 3   # ru, rv, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("inflow", "wall", "outflow")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "box"

# --- Physical parameters ---
RE     = 800.0       # NS-BFS Re=800 (channel height 1, mean inlet velocity 1)
NU     = 1.0 / RE
U_MAX  = 1.5         # peak of the parabolic inlet profile 24 y (0.5 - y)
H_STEP = 0.5         # step height = inlet height = half the channel

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
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


# ══════════ Input normalisation ══════════

_XC = 0.5 * (X_MIN + X_MAX)
_XS = 0.5 * (X_MAX - X_MIN)
_YC = 0.5 * (Y_MIN + Y_MAX)
_YS = 0.5 * (Y_MAX - Y_MIN)


def _normalize(xy):
    return jnp.stack([(xy[..., 0] - _XC) / _XS,
                      (xy[..., 1] - _YC) / _YS], axis=-1)


class PINN(nn.Module):
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, xy):
        kinit = jax.nn.initializers.he_uniform()
        h = _normalize(xy)
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)


def _inflow_u(y: np.ndarray) -> np.ndarray:
    return np.where(y > 0.0, 24.0 * y * (0.5 - y), 0.0)


def _build_dataset():
    nx, ny = 481, 65
    xs = np.linspace(X_MIN, X_MAX, nx)
    ys = np.linspace(Y_MIN, Y_MAX, ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.flatten(), Y.flatten()], axis=1)

    on_inlet  = pts[:, 0] == X_MIN
    on_outlet = pts[:, 0] == X_MAX
    on_wall   = (pts[:, 1] == Y_MIN) | (pts[:, 1] == Y_MAX)

    pts_int  = pts[~(on_inlet | on_outlet | on_wall)]
    pts_in   = pts[on_inlet]
    pts_wall = pts[on_wall & ~on_inlet]
    pts_bc = np.vstack([pts_in, pts_wall])
    u_bc = np.concatenate([_inflow_u(pts_in[:, 1]),
                           np.zeros(len(pts_wall))]).astype(np.float32)
    v_bc = np.zeros(len(pts_bc), dtype=np.float32)
    pts_out = pts[on_outlet & ~on_wall]

    return (
        jnp.array(pts_int.astype(np.float32)),
        jnp.array(pts_bc.astype(np.float32)),
        jnp.array(u_bc),
        jnp.array(v_bc),
        jnp.array(pts_out.astype(np.float32)),
    )


# ══════════ Training hyperparameters ══════════

LR0            = 1e-3
LR_DECAY_STEPS = 2000
LR_DECAY_RATE  = 0.9
WEIGHT_BC      = 100.0
WEIGHT_OUT     = 1.0
BS_INT         = 2048
CHUNK          = 100     # = inner Adam steps per pseudo-time step (kb1/073 K)
SAVE_MARGIN_S  = 6.0
DT             = 2.0   # pseudo-time step; mu = 1/DT shifts the Jacobian
                         # spectrum left (kb1/073 eq. 2).  DT -> inf recovers
                         # the vanilla steady residual loss.


def _make_train(rng_init):
    pts_int, pts_bc, u_bc, v_bc, pts_out = _build_dataset()
    n_int = len(pts_int)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def predict_one(params_flat, xy):
        return model.apply(unravel(params_flat), xy[None, :])[0]

    d1_fn = jacfwd(predict_one, argnums=1)
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)

    def residual_vec(params_flat, xy):
        """(r_mx, r_my, r_c) — steady NS operator, all derivatives by autodiff."""
        uvp = predict_one(params_flat, xy)
        u, v = uvp[0], uvp[1]
        d1 = d1_fn(params_flat, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        d2 = d2_fn(params_flat, xy)
        u_xx, u_yy = d2[0, 0, 0], d2[0, 1, 1]
        v_xx, v_yy = d2[1, 0, 0], d2[1, 1, 1]
        r_mx = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
        r_my = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
        r_c  = u_x + v_y
        return uvp, jnp.stack([r_mx, r_my, r_c])

    def pts_per_point(params_flat, params_prev, xy):
        q, r = residual_vec(params_flat, xy)
        q_n = jax.lax.stop_gradient(predict_one(params_prev, xy))
        return jnp.sum(((q - q_n) / DT + r) ** 2)

    pts_batch = vmap(pts_per_point, in_axes=(None, None, 0))
    forward_batch = vmap(predict_one, in_axes=(None, 0))

    def loss_fn(params_flat, params_prev, xy_int):
        pde_loss = jnp.mean(pts_batch(params_flat, params_prev, xy_int))
        bc_uvp = forward_batch(params_flat, pts_bc)
        bc_loss = jnp.mean((bc_uvp[:, 0] - u_bc) ** 2 + (bc_uvp[:, 1] - v_bc) ** 2)
        out_uvp = forward_batch(params_flat, pts_out)
        out_loss = jnp.mean(out_uvp[:, 2] ** 2)
        return pde_loss + WEIGHT_BC * bc_loss + WEIGHT_OUT * out_loss

    loss_and_grad = jax.value_and_grad(loss_fn)

    schedule = optax.exponential_decay(LR0, LR_DECAY_STEPS, LR_DECAY_RATE)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(flat_params)

    def one_step(params_prev, carry, key):
        params_flat, opt_state = carry
        idx = random.choice(key, n_int, (BS_INT,))
        loss, grads = loss_and_grad(params_flat, params_prev, pts_int[idx])
        updates, opt_state = optimizer.update(grads, opt_state)
        return (optax.apply_updates(params_flat, updates), opt_state), loss

    @jit
    def run_chunk(params_flat, opt_state, key):
        """One pseudo-time step: CHUNK inner Adam steps against the frozen
        snapshot q_n = q(params_flat at chunk entry)."""
        params_prev = jax.lax.stop_gradient(params_flat)
        keys = random.split(key, CHUNK)
        (p, o), losses = jax.lax.scan(lambda c, k: one_step(params_prev, c, k),
                                      (params_flat, opt_state), keys)
        return p, o, losses[-1]

    return flat_params, opt_state, run_chunk, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline pseudo-time stepping: each scan chunk is one implicit
    Euler sub-problem solved by CHUNK inner Adam steps."""
    params_flat, opt_state, run_chunk, unravel, rng_local = _make_train(rng)

    rng_local, k = random.split(rng_local)
    params_flat, opt_state, loss = run_chunk(params_flat, opt_state, k)
    step_count = CHUNK
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    while step_count < MAX_ITER:
        if time.time() > deadline:
            break
        rng_local, k = random.split(rng_local)
        params_flat, opt_state, loss = run_chunk(params_flat, opt_state, k)
        step_count += CHUNK
        last_loss = float(loss)
        if eval_callback is not None and (step_count % 10000 == 0):
            eval_callback(unravel(params_flat), step_count, last_loss)

    return unravel(params_flat), step_count
