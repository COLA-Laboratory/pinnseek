"""b10_base with the hard-BC lift decay length L_IN 0.25 -> 0.20.
Continues b10's L_IN sweep: the probes show the lift holds the exact inlet
profile ~0.09 too far downstream, displacing the shear layer and the bottom
recirculation bubble, which carries most of the squared error."""
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
from jax import jit, random, vmap
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


# ══════════ Hard Dirichlet ansatz (kb1/036) ══════════
# u = U_in(y) (1 - s(x))  +  w(y) s(x) u_hat      v = w(y) s(x) v_hat
#   s(0) = 0            -> inlet profile exact (v = 0 exact)
#   w(+-0.5) = 0, U_in(+-0.5) = 0 -> no-slip exact on both walls

L_IN = 0.20

# ══════════ b10 knobs ══════════
X_EXT   = X_MAX   # artificial outflow plane (X_MAX = no extension)
DX_TAIL = 0.25       # coarse tail spacing beyond X_MAX
MU      = 0.0     # fraction of the outlet-column mean pressure removed
                     # inside the model: 0 = champion's soft gauge, 1 = exact
                     # gauge (b05_gauge) — also removes the level drift from
                     # the artificial-compressibility pseudo-time term
DT_C    = 1.0  # pseudo-time step of the p / continuity component


def _U_in(y):
    return jnp.where(y > 0.0, 24.0 * y * (0.5 - y), 0.0)


def _ansatz(raw, xy):
    x, y = xy[..., 0], xy[..., 1]
    s = jnp.tanh(x / L_IN)
    w = 1.0 - 4.0 * y * y
    ws = w * s
    return jnp.stack([_U_in(y) * (1.0 - s) + ws * raw[..., 0],
                      ws * raw[..., 1],
                      raw[..., 2]], axis=-1)


def _build_dataset():
    nx, ny = 481, 65
    xs = np.linspace(X_MIN, X_MAX, nx)
    ys = np.linspace(Y_MIN, Y_MAX, ny)
    if X_EXT > X_MAX:
        xs = np.concatenate([xs, np.arange(X_MAX + DX_TAIL, X_EXT + 1e-9, DX_TAIL)])
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.flatten(), Y.flatten()], axis=1)

    on_inlet  = pts[:, 0] == X_MIN
    on_outlet = pts[:, 0] == xs[-1]
    on_wall   = (pts[:, 1] == Y_MIN) | (pts[:, 1] == Y_MAX)

    pts_int = pts[~(on_inlet | on_wall)]
    pts_out = pts[on_outlet & ~on_wall]
    return (jnp.array(pts_int.astype(np.float32)),
            jnp.array(pts_out.astype(np.float32)))


# ══════════ Training hyperparameters ══════════

LR0            = 1e-3
LR_DECAY_STEPS = 2000
LR_DECAY_RATE  = 0.9
LR_FLOOR       = 1e-4    # keeps the pseudo-time march alive to the end
WEIGHT_OUT     = 1.0     # p = 0 outlet gauge (the only soft term left)
BS_INT         = 2048
CHUNK          = 100
N_CHUNKS       = 1100    # fixed stopping point (b07 lead 1): 110 000 steps,
                         # ~2 s inside the wall the champion used to run to
SAVE_MARGIN_S  = 6.0
EMA_DECAY      = 0.999
DT             = 1.0
W_PPE          = 0.0     # weight of the pressure-Poisson defect (off here)

_EX = jnp.array([1.0, 0.0])
_EY = jnp.array([0.0, 1.0])
_DTV = jnp.array([DT, DT, DT_C])


def _make_train(rng_init):
    pts_int, pts_out = _build_dataset()
    n_int = len(pts_int)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def p_offset(params_flat):
        return jnp.mean(model.apply(unravel(params_flat), pts_out)[:, 2])

    def predict_one(params_flat, off, xy):
        raw = model.apply(unravel(params_flat), xy[None, :])[0]
        return _ansatz(raw, xy).at[2].add(-MU * off)

    def _dir2(params_flat, off, xy, e):
        def inner(z):
            return jax.jvp(lambda w: predict_one(params_flat, off, w), (z,), (e,))
        (q, d1), (_, d2) = jax.jvp(inner, (xy,), (e,))
        return q, d1, d2

    def residual_vec(params_flat, off, xy):
        """(r_mx, r_my, r_c) and the PPE defect — all by autodiff."""
        q, dx, dxx = _dir2(params_flat, off, xy, _EX)
        _, dy, dyy = _dir2(params_flat, off, xy, _EY)
        u, v = q[0], q[1]
        r_mx = u * dx[0] + v * dy[0] + dx[2] - NU * (dxx[0] + dyy[0])
        r_my = u * dx[1] + v * dy[1] + dy[2] - NU * (dxx[1] + dyy[1])
        r_c  = dx[0] + dy[1]
        # div of the momentum equation (kb1/039 gPINN family): needs only
        # p_xx, p_yy and first velocity derivatives — already computed.
        r_p  = dxx[2] + dyy[2] + dx[0] ** 2 + 2.0 * dy[0] * dx[1] + dy[1] ** 2
        return q, jnp.stack([r_mx, r_my, r_c]), r_p

    def pts_per_point(params_flat, params_prev, off, off_prev, xy):
        q, r, r_p = residual_vec(params_flat, off, xy)
        q_n = jax.lax.stop_gradient(predict_one(params_prev, off_prev, xy))
        return jnp.sum(((q - q_n) / _DTV + r) ** 2) + W_PPE * r_p ** 2

    pts_batch = vmap(pts_per_point, in_axes=(None, None, None, None, 0))

    def loss_fn(params_flat, params_prev, xy_int):
        off = p_offset(params_flat)
        off_prev = jax.lax.stop_gradient(p_offset(params_prev))
        pde_loss = jnp.mean(pts_batch(params_flat, params_prev, off, off_prev, xy_int))
        out_loss = ((1.0 - MU) * off) ** 2
        return pde_loss + WEIGHT_OUT * out_loss

    loss_and_grad = jax.value_and_grad(loss_fn)

    schedule = optax.exponential_decay(LR0, LR_DECAY_STEPS, LR_DECAY_RATE,
                                       end_value=LR_FLOOR)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(flat_params)

    def one_step(params_prev, carry, key):
        params_flat, opt_state, ema = carry
        idx = random.choice(key, n_int, (BS_INT,))
        loss, grads = loss_and_grad(params_flat, params_prev, pts_int[idx])
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params_flat, updates)
        ema = EMA_DECAY * ema + (1.0 - EMA_DECAY) * new_params
        return (new_params, opt_state, ema), loss

    @jit
    def run_chunk(params_flat, opt_state, ema, key):
        params_prev = jax.lax.stop_gradient(params_flat)
        keys = random.split(key, CHUNK)
        (p, o, e), losses = jax.lax.scan(lambda c, k: one_step(params_prev, c, k),
                                         (params_flat, opt_state, ema), keys)
        return p, o, e, losses[-1]

    return flat_params, opt_state, run_chunk, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


_PTS_GAUGE = None


def _gauge_pts():
    global _PTS_GAUGE
    if _PTS_GAUGE is None:
        _, po = _build_dataset()
        _PTS_GAUGE = po
    return _PTS_GAUGE


def predict_fn(params, X):
    out = _ansatz(_PREDICT_MODEL.apply(params, X), X)
    off = jnp.mean(_PREDICT_MODEL.apply(params, _gauge_pts())[:, 2])
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2] - MU * off}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline pseudo-time stepping: each scan chunk is one implicit
    Euler sub-problem solved by CHUNK inner Adam steps. Stops at N_CHUNKS
    chunks, or at the wall if the machine is slower than usual."""
    params_flat, opt_state, run_chunk, unravel, rng_local = _make_train(rng)
    ema = params_flat

    rng_local, k = random.split(rng_local)
    params_flat, opt_state, ema, loss = run_chunk(params_flat, opt_state, ema, k)
    step_count = CHUNK
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    n_target = CHUNK * N_CHUNKS if TRAIN_TIME > 20.0 else MAX_ITER
    while step_count < min(n_target, MAX_ITER):
        if time.time() > deadline:
            break
        rng_local, k = random.split(rng_local)
        params_flat, opt_state, ema, loss = run_chunk(params_flat, opt_state, ema, k)
        step_count += CHUNK
        last_loss = float(loss)
        if eval_callback is not None and (step_count % 10000 == 0):
            eval_callback(unravel(ema), step_count, last_loss)

    return unravel(ema), step_count
