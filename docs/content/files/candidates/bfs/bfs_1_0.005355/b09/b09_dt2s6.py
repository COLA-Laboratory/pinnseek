"""Separable PINN (kb1/070), full-lattice pseudo-time march: b09_dt2 with the
Adam decay stretched 2000 -> 6000 steps, so the LR floor is reached at 132k
instead of 44k of the 180k steps — a test of whether the schedule, not the
march, is what saturates this recipe past 50k steps."""
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
from jax import jit, random
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


# ══════════ Input normalisation ══════════

_XC = 0.5 * (X_MIN + X_MAX)
_XS = 0.5 * (X_MAX - X_MIN)
_YC = 0.5 * (Y_MIN + Y_MAX)
_YS = 0.5 * (Y_MAX - Y_MIN)


# ══════════ Separable backbone (kb1/070) ══════════
# raw_c(x, y) = RSCALE * sum_k Fx[c,k](x) Fy[c,k](y)
# One evaluation of the two 1-D bodies defines the field on the whole
# nx x ny lattice; forward-mode AD along each axis gives every derivative.

RANK      = 128
AX_WIDTH  = 128
AX_DEPTH  = 5


class AxisBody(nn.Module):
    width: int = AX_WIDTH
    depth: int = AX_DEPTH
    rank: int = RANK

    @nn.compact
    def __call__(self, z):                      # z: [N, 1] -> [N, 3, rank]
        kinit = jax.nn.initializers.he_uniform()
        h = z
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h))
        out = nn.Dense(OUTPUT_DIM * self.rank, kernel_init=kinit)(h)
        return out.reshape(out.shape[0], OUTPUT_DIM, self.rank)


class SPINN2D(nn.Module):
    rank: int = RANK

    def setup(self):
        self.ax = AxisBody(name="axis_x")
        self.ay = AxisBody(name="axis_y")

    def _fx(self, xs):
        return self.ax(((xs - _XC) / _XS)[:, None])

    def _fy(self, ys):
        return self.ay(((ys - _YC) / _YS)[:, None])

    def __call__(self, xs, ys):                 # [nx], [ny] -> [nx, ny, 3]
        sc = 1.0 / np.sqrt(self.rank)
        return sc * jnp.einsum("ick,jck->ijc", self._fx(xs), self._fy(ys))

    def pointwise(self, xy):                    # [N, 2] -> [N, 3]
        sc = 1.0 / np.sqrt(self.rank)
        return sc * jnp.einsum("ick,ick->ic", self._fx(xy[:, 0]), self._fy(xy[:, 1]))


# ══════════ Hard Dirichlet ansatz (kb1/036) — the champion's, verbatim ══════════
# u = U_in(y) (1 - s(x))  +  w(y) s(x) u_hat      v = w(y) s(x) v_hat

L_IN = 0.25


def _U_in(y):
    return jnp.where(y > 0.0, 24.0 * y * (0.5 - y), 0.0)


def _ansatz(raw, x, y):
    """raw: [..., 3]; x, y broadcastable to raw[..., 0]."""
    s = jnp.tanh(x / L_IN)
    w = 1.0 - 4.0 * y * y
    ws = w * s
    return jnp.stack([_U_in(y) * (1.0 - s) + ws * raw[..., 0],
                      ws * raw[..., 1],
                      raw[..., 2]], axis=-1)


def _build_axes():
    """The champion's collocation set, factorised: pts_int = the 481x65 grid
    minus the inlet column and both walls = xs[1:] x ys[1:-1] = 480 x 63 =
    30240 points, exactly n_int. The outlet column is xs[-1]."""
    nx, ny = 481, 65
    xs = np.linspace(X_MIN, X_MAX, nx)[1:]
    ys = np.linspace(Y_MIN, Y_MAX, ny)[1:-1]
    return jnp.array(xs.astype(np.float32)), jnp.array(ys.astype(np.float32))


# ══════════ Training hyperparameters (champion's) ══════════

LR0            = 1e-3
LR_DECAY_STEPS = 6000
LR_DECAY_RATE  = 0.9
LR_FLOOR       = 1e-4
WEIGHT_OUT     = 1.0     # p = 0 outlet gauge (the only soft term left)
CHUNK          = 100
N_CHUNKS       = 1800    # fixed stopping point (b08): 175 000 steps @ 1158 st/s
SAVE_MARGIN_S  = 6.0
EMA_DECAY      = 0.999
DT             = 2.0


def _make_train(rng_init):
    xs, ys = _build_axes()
    ones_x = jnp.ones_like(xs)
    ones_y = jnp.ones_like(ys)

    model = SPINN2D()
    key, rng = random.split(rng_init)
    params = model.init(key, xs[:2], ys[:2])
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def field(params_flat, xa, ya):
        raw = model.apply(unravel(params_flat), xa, ya)
        return _ansatz(raw, xa[:, None], ya[None, :])          # [nx, ny, 3]

    def streams(params_flat):
        """(q, dx, dy, dxx, dyy) on the lattice — all by forward-mode autodiff
        of the network (kb1/070): u[i,j] depends on xs[i] alone, so a JVP with
        an all-ones tangent on the axis vector returns the per-point partial."""
        def fx(z):
            return field(params_flat, z, ys)

        def inner_x(z):
            return jax.jvp(fx, (z,), (ones_x,))

        (q, dx), (_, dxx) = jax.jvp(inner_x, (xs,), (ones_x,))

        def fy(z):
            return field(params_flat, xs, z)

        def inner_y(z):
            return jax.jvp(fy, (z,), (ones_y,))

        (_, dy), (_, dyy) = jax.jvp(inner_y, (ys,), (ones_y,))
        return q, dx, dy, dxx, dyy

    def loss_fn(params_flat, params_prev):
        q, dx, dy, dxx, dyy = streams(params_flat)
        u, v = q[..., 0], q[..., 1]
        r_mx = u * dx[..., 0] + v * dy[..., 0] + dx[..., 2] - NU * (dxx[..., 0] + dyy[..., 0])
        r_my = u * dx[..., 1] + v * dy[..., 1] + dy[..., 2] - NU * (dxx[..., 1] + dyy[..., 1])
        r_c  = dx[..., 0] + dy[..., 1]
        r = jnp.stack([r_mx, r_my, r_c], axis=-1)
        q_n = jax.lax.stop_gradient(field(params_prev, xs, ys))
        pde_loss = jnp.mean(jnp.sum(((q - q_n) / DT + r) ** 2, axis=-1))
        out_loss = jnp.mean(q[-1, :, 2]) ** 2      # outlet column, p gauge
        return pde_loss + WEIGHT_OUT * out_loss

    loss_and_grad = jax.value_and_grad(loss_fn)

    schedule = optax.exponential_decay(LR0, LR_DECAY_STEPS, LR_DECAY_RATE,
                                       end_value=LR_FLOOR)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(flat_params)

    def one_step(params_prev, carry, _):
        params_flat, opt_state, ema = carry
        loss, grads = loss_and_grad(params_flat, params_prev)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params_flat, updates)
        ema = EMA_DECAY * ema + (1.0 - EMA_DECAY) * new_params
        return (new_params, opt_state, ema), loss

    @jit
    def run_chunk(params_flat, opt_state, ema):
        params_prev = jax.lax.stop_gradient(params_flat)
        (p, o, e), losses = jax.lax.scan(lambda c, k: one_step(params_prev, c, k),
                                         (params_flat, opt_state, ema), None,
                                         length=CHUNK)
        return p, o, e, losses[-1]

    return flat_params, opt_state, run_chunk, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = SPINN2D()


def predict_fn(params, X):
    raw = _PREDICT_MODEL.apply(params, X, method=SPINN2D.pointwise)
    out = _ansatz(raw, X[:, 0], X[:, 1])
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline pseudo-time stepping: each scan chunk is one implicit
    Euler sub-problem solved by CHUNK inner Adam steps, now full-batch over
    the whole lattice. Stops at N_CHUNKS chunks, or at the wall."""
    params_flat, opt_state, run_chunk, unravel, rng_local = _make_train(rng)
    ema = params_flat

    params_flat, opt_state, ema, loss = run_chunk(params_flat, opt_state, ema)
    step_count = CHUNK
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    n_target = CHUNK * N_CHUNKS if TRAIN_TIME > 20.0 else MAX_ITER
    while step_count < min(n_target, MAX_ITER):
        if time.time() > deadline:
            break
        params_flat, opt_state, ema, loss = run_chunk(params_flat, opt_state, ema)
        step_count += CHUNK
        last_loss = float(loss)
        if eval_callback is not None and (step_count % 10000 == 0):
            eval_callback(unravel(ema), step_count, last_loss)

    return unravel(ema), step_count
