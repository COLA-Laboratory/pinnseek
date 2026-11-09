"""Exact-continuity primitive PINN: outputs (psi, omega, p); u = psi_y, v = -psi_x.
Keeps the champion's momentum equations, pressure and outlet gauge (unlike the
curl form b06_psiw) and uses lap u = -omega_y, lap v = omega_x so nothing exceeds
2nd order.  omega is auxiliary: it enters only through the O(nu) viscous term."""
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


# ══════════ Input normalisation (unchanged from the parent) ══════════

_XC = 0.5 * (X_MIN + X_MAX)
_XS = 0.5 * (X_MAX - X_MIN)
_YC = 0.5 * (Y_MIN + Y_MAX)
_YS = 0.5 * (Y_MAX - Y_MIN)

N_OUT = 3   # (psi_hat, omega_hat, p)


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
        return nn.Dense(N_OUT, kernel_init=kinit)(h)


# ══════════ Stream-function ansatz (kb1/036 hard BCs, exact continuity) ══════
#
#   psi = PSI_L(x, y) + w(y)^2 sigma(x) psi_hat ,     u = psi_y ,  v = -psi_x
#   omega = OMEGA_SCALE * omega_hat
#
# w = 1 - 4y^2 vanishes on both walls, and w^2 makes psi AND psi_y vanish
# there -> psi = const on each wall (v = 0) and u = 0: no-slip exact.
# sigma(0) = sigma'(0) = 0 -> psi(0,y) = PSI_L(0,y) = G(y) (inlet profile
# exact) and v(0,y) = 0 exact.  G is the inlet stream function; its second
# derivative jumps at the step lip y = 0, so it is blended over L_IN into the
# smooth fully-developed profile PSI_FD, which already satisfies both wall
# conditions.  Continuity u_x + v_y = 0 and the channel flux int u dy = 0.5
# hold identically, by construction, everywhere.

L_IN        = 0.25    # blend length of the kinked inlet lift
# Head scales chosen so the field at initialisation sits at the physical scale:
# with he_uniform the raw heads have rms ~0.17 / 0.66, giving an initial u-noise
# of 0.46 against the lift's 0.54 and an initial vorticity of ~2 against the
# fully-developed |omega| ~ 1.7.  psi is differentiated once before it is
# scored, so an unscaled head would start 2x noisier than the signal.
PSI_SCALE   = 0.5
OMEGA_SCALE = 3.0


def _G(y):
    """Inlet stream function: int_{-1/2}^{y} u_in.  G(-.5)=0, G(.5)=.5,
    G'(+-.5)=0.  G'' jumps by 12 at the step lip y = 0."""
    return jnp.where(y > 0.0, 6.0 * y * y - 8.0 * y * y * y, 0.0)


def _PSI_FD(y):
    """Fully-developed (Poiseuille) stream function for the full channel,
    same flux 0.5, same wall values/slopes, but C-infinity."""
    return 0.25 + 0.75 * y - y * y * y


def _sigma(x):
    t = x / L_IN
    return jnp.tanh(t * t)


def _fields(raw, xy):
    """(psi, omega, p) from the raw network output."""
    x, y = xy[..., 0], xy[..., 1]
    sg = _sigma(x)
    w = 1.0 - 4.0 * y * y
    fd = _PSI_FD(y)
    lift = fd + (_G(y) - fd) * (1.0 - sg)
    psi = lift + PSI_SCALE * w * w * sg * raw[..., 0]
    om = OMEGA_SCALE * raw[..., 1]
    return jnp.stack([psi, om, raw[..., 2]], axis=-1)


def _build_dataset():
    """The champion's split exactly: residual on every point except the inlet
    column and the two wall rows; the outlet column carries the residual plus
    the mean-pressure gauge (b02's outlet finding)."""
    nx, ny = 481, 65
    xs = np.linspace(X_MIN, X_MAX, nx)
    ys = np.linspace(Y_MIN, Y_MAX, ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.flatten(), Y.flatten()], axis=1)
    on_inlet = pts[:, 0] == X_MIN
    on_outlet = pts[:, 0] == X_MAX
    on_wall = (pts[:, 1] == Y_MIN) | (pts[:, 1] == Y_MAX)
    pts_int = pts[~(on_inlet | on_wall)]
    pts_out = pts[on_outlet & ~on_wall]
    return (jnp.array(pts_int.astype(np.float32)),
            jnp.array(pts_out.astype(np.float32)))


# ══════════ Training hyperparameters (parent's, unchanged) ══════════

LR0            = 1e-3
LR_DECAY_STEPS = 2000
LR_DECAY_RATE  = 0.9
LR_FLOOR       = 1e-4
BS_INT         = 2048
CHUNK          = 100
SAVE_MARGIN_S  = 6.0
EMA_DECAY      = 0.999
DT             = 1.0     # pseudo-time step of the momentum march
DT_P           = 1.0     # pseudo-time step of the psi (coupling) relaxation
W_CPL          = 1.0     # weight of the coupling equation omega = -lap psi
WEIGHT_OUT     = 1.0     # mean-p = 0 outlet gauge (champion's, restored)

_EX = jnp.array([1.0, 0.0])
_EY = jnp.array([0.0, 1.0])
_ED = jnp.array([1.0, 1.0])     # diagonal: supplies the cross derivative psi_xy


def _make_train(rng_init):
    pts_int, pts_out = _build_dataset()
    n_int = len(pts_int)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def predict_one(params_flat, xy):
        raw = model.apply(unravel(params_flat), xy[None, :])[0]
        return _fields(raw, xy)

    def _dir2(params_flat, xy, e):
        def inner(z):
            return jax.jvp(lambda w: predict_one(params_flat, w), (z,), (e,))
        (q, d1), (_, d2) = jax.jvp(inner, (xy,), (e,))
        return q, d1, d2

    def state_and_residual(params_flat, xy):
        """Marched state (psi, u, v) and the three residuals, every derivative
        by automatic differentiation of the network.

            r_cpl = -(psi_xx + psi_yy + omega)     [ psi_t = lap psi + omega ]
            r_mx  = u u_x + v u_y + p_x + NU omega_y      (lap u  = -omega_y)
            r_my  = u v_x + v v_y + p_y - NU omega_x      (lap v  =  omega_x)

        Continuity is identically zero: u = psi_y, v = -psi_x.  r_cpl carries
        the sign that makes its damped form a *backward Euler* step of a
        diffusive relaxation; the opposite sign would be anti-diffusive."""
        q, dx, dxx = _dir2(params_flat, xy, _EX)
        _, dy, dyy = _dir2(params_flat, xy, _EY)
        _, _, ddd = _dir2(params_flat, xy, _ED)
        psi_xy = 0.5 * (ddd[0] - dxx[0] - dyy[0])
        u, v = dy[0], -dx[0]
        u_x, u_y = psi_xy, dyy[0]
        v_x, v_y = -dxx[0], -psi_xy
        om = q[1]
        r_cpl = -(dxx[0] + dyy[0] + om)
        r_mx = u * u_x + v * u_y + dx[2] + NU * dy[1]
        r_my = u * v_x + v * v_y + dy[2] - NU * dx[1]
        return jnp.stack([q[0], u, v]), jnp.stack([r_cpl, r_mx, r_my])

    def state_only(params_flat, xy):
        q, dx, _ = _dir2(params_flat, xy, _EX)
        _, dy, _ = _dir2(params_flat, xy, _EY)
        return jnp.stack([q[0], dy[0], -dx[0]])

    def pts_per_point(params_flat, params_prev, xy):
        s, r = state_and_residual(params_flat, xy)
        s_n = jax.lax.stop_gradient(state_only(params_prev, xy))
        d = s - s_n
        e_cpl = d[0] / DT_P + r[0]
        e_mx = d[1] / DT + r[1]
        e_my = d[2] / DT + r[2]
        return W_CPL * e_cpl ** 2 + e_mx ** 2 + e_my ** 2

    pts_batch = vmap(pts_per_point, in_axes=(None, None, 0))
    forward_batch = vmap(predict_one, in_axes=(None, 0))

    def loss_fn(params_flat, params_prev, xy_int):
        pde_loss = jnp.mean(pts_batch(params_flat, params_prev, xy_int))
        out_loss = jnp.mean(forward_batch(params_flat, pts_out)[:, 2]) ** 2
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


def _psi_scalar(params, xy):
    raw = _PREDICT_MODEL.apply(params, xy[None, :])[0]
    return _fields(raw, xy)[0]


@jit
def _uv(params, X):
    g = vmap(jax.grad(_psi_scalar, argnums=1), in_axes=(None, 0))(params, X)
    return g[:, 1], -g[:, 0]          # u = psi_y, v = -psi_x


def predict_fn(params, X):
    """u and v are exact derivatives of the stream function; p is the third
    network channel, gauged by mean(p) = 0 on the outlet column."""
    u, v = _uv(params, X)
    p = _PREDICT_MODEL.apply(params, X)[:, 2]
    return {"u": u, "v": v, "p": p}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline pseudo-time stepping: each scan chunk is one implicit
    Euler sub-problem solved by CHUNK inner Adam steps."""
    params_flat, opt_state, run_chunk, unravel, rng_local = _make_train(rng)
    ema = params_flat

    rng_local, k = random.split(rng_local)
    params_flat, opt_state, ema, loss = run_chunk(params_flat, opt_state, ema, k)
    step_count = CHUNK
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    while step_count < MAX_ITER:
        if time.time() > deadline:
            break
        rng_local, k = random.split(rng_local)
        params_flat, opt_state, ema, loss = run_chunk(params_flat, opt_state, ema, k)
        step_count += CHUNK
        last_loss = float(loss)
        if eval_callback is not None and (step_count % 10000 == 0):
            eval_callback(unravel(ema), step_count, last_loss)

    return unravel(ema), step_count
