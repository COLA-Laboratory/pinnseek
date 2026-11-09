"""Hard-BC pseudo-time PINN (b02_dt10 / b03_base) with SOAP replacing Adam.

Change vs parent: the inner optimiser of every pseudo-time sub-problem is
SOAP -- Adam run in the Kronecker eigenbasis of the layerwise gradient
covariance (kb1/088, kb1/108). Everything else is byte-identical in effect.
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
from jax import jit, random, vmap
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


# ══════════ Hard Dirichlet ansatz (kb1/036) — unchanged from parent ══════════

L_IN = 0.25


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
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.flatten(), Y.flatten()], axis=1)

    on_inlet  = pts[:, 0] == X_MIN
    on_outlet = pts[:, 0] == X_MAX
    on_wall   = (pts[:, 1] == Y_MIN) | (pts[:, 1] == Y_MAX)

    pts_int = pts[~(on_inlet | on_wall)]
    pts_out = pts[on_outlet & ~on_wall]
    return (jnp.array(pts_int.astype(np.float32)),
            jnp.array(pts_out.astype(np.float32)))


# ══════════ Training hyperparameters ══════════

LR0            = 1e-3
LR_DECAY_STEPS = 2000
LR_DECAY_RATE  = 0.9
LR_FLOOR       = 1e-4
WEIGHT_OUT     = 1.0
BS_INT         = 2048
CHUNK          = 100
SAVE_MARGIN_S  = 6.0
EMA_DECAY      = 0.999
DT             = 1.0

# --- SOAP (kb1/088 §"SOAP as approximate Newton preconditioner", kb1/108 §C) ---
SOAP_FREQ = 25      # steps between eigenbasis refreshes
SOAP_B1   = 0.9     # momentum (kept in the ORIGINAL basis: rotation-equivariant)
SOAP_B2   = 0.95    # second moment, kept in the rotated (eigen) basis
SOAP_B3   = 0.95    # Kronecker covariance accumulation
SOAP_EPS  = 1e-8

_EX = jnp.array([1.0, 0.0])
_EY = jnp.array([0.0, 1.0])


# ══════════ SOAP: Adam in the layerwise Kronecker eigenbasis ══════════

def _soap_init(leaves):
    st = []
    for p in leaves:
        if p.ndim == 2:
            r, c = p.shape
            st.append(dict(L=jnp.zeros((r, r)), R=jnp.zeros((c, c)),
                           Ql=jnp.eye(r), Qr=jnp.eye(c),
                           m=jnp.zeros_like(p), v=jnp.zeros_like(p)))
        else:
            st.append(dict(m=jnp.zeros_like(p), v=jnp.zeros_like(p)))
    return st


def _soap_update(gleaves, state, lr, t, do_eig):
    """One SOAP step per leaf. 2-D kernels get the Kronecker rotation; 1-D
    biases fall back to plain Adam. Bias-corrected, so the update magnitude
    matches Adam's and the parent's LR schedule carries over unchanged."""
    bc1 = 1.0 - SOAP_B1 ** (t + 1)
    bc2 = 1.0 - SOAP_B2 ** (t + 1)
    upd, new_state = [], []
    for g, s in zip(gleaves, state):
        if g.ndim == 2:
            L = SOAP_B3 * s["L"] + (1.0 - SOAP_B3) * (g @ g.T)
            R = SOAP_B3 * s["R"] + (1.0 - SOAP_B3) * (g.T @ g)
            Ql = jax.lax.cond(do_eig, lambda _: jnp.linalg.eigh(L)[1],
                              lambda _: s["Ql"], operand=None)
            Qr = jax.lax.cond(do_eig, lambda _: jnp.linalg.eigh(R)[1],
                              lambda _: s["Qr"], operand=None)
            m = SOAP_B1 * s["m"] + (1.0 - SOAP_B1) * g          # original basis
            gr = Ql.T @ g @ Qr                                   # eigenbasis
            v = SOAP_B2 * s["v"] + (1.0 - SOAP_B2) * gr * gr
            mr = Ql.T @ m @ Qr
            step_rot = (mr / bc1) / (jnp.sqrt(v / bc2) + SOAP_EPS)
            upd.append(-lr * (Ql @ step_rot @ Qr.T))
            new_state.append(dict(L=L, R=R, Ql=Ql, Qr=Qr, m=m, v=v))
        else:
            m = SOAP_B1 * s["m"] + (1.0 - SOAP_B1) * g
            v = SOAP_B2 * s["v"] + (1.0 - SOAP_B2) * g * g
            upd.append(-lr * (m / bc1) / (jnp.sqrt(v / bc2) + SOAP_EPS))
            new_state.append(dict(m=m, v=v))
    return upd, new_state


def _make_train(rng_init):
    pts_int, pts_out = _build_dataset()
    n_int = len(pts_int)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    leaves, treedef = jax.tree_util.tree_flatten(params)

    def predict_one(prm, xy):
        return _ansatz(model.apply(prm, xy[None, :])[0], xy)

    def _dir2(prm, xy, e):
        def inner(z):
            return jax.jvp(lambda w: predict_one(prm, w), (z,), (e,))
        (q, d1), (_, d2) = jax.jvp(inner, (xy,), (e,))
        return q, d1, d2

    def residual_vec(prm, xy):
        """(r_mx, r_my, r_c) — steady NS operator, all derivatives by autodiff."""
        q, dx, dxx = _dir2(prm, xy, _EX)
        _, dy, dyy = _dir2(prm, xy, _EY)
        u, v = q[0], q[1]
        r_mx = u * dx[0] + v * dy[0] + dx[2] - NU * (dxx[0] + dyy[0])
        r_my = u * dx[1] + v * dy[1] + dy[2] - NU * (dxx[1] + dyy[1])
        r_c  = dx[0] + dy[1]
        return q, jnp.stack([r_mx, r_my, r_c])

    def pts_per_point(prm, prm_prev, xy):
        q, r = residual_vec(prm, xy)
        q_n = jax.lax.stop_gradient(predict_one(prm_prev, xy))
        return jnp.sum(((q - q_n) / DT + r) ** 2)

    pts_batch = vmap(pts_per_point, in_axes=(None, None, 0))
    forward_batch = vmap(predict_one, in_axes=(None, 0))

    def loss_fn(prm, prm_prev, xy_int):
        pde_loss = jnp.mean(pts_batch(prm, prm_prev, xy_int))
        out_loss = jnp.mean(forward_batch(prm, pts_out)[:, 2]) ** 2
        return pde_loss + WEIGHT_OUT * out_loss

    loss_and_grad = jax.value_and_grad(loss_fn)

    schedule = optax.exponential_decay(LR0, LR_DECAY_STEPS, LR_DECAY_RATE,
                                       end_value=LR_FLOOR)
    state = _soap_init(leaves)

    def one_step(prm_prev, carry, key):
        prm, st, ema, t = carry
        idx = random.choice(key, n_int, (BS_INT,))
        loss, grads = loss_and_grad(prm, prm_prev, pts_int[idx])
        gl = jax.tree_util.tree_leaves(grads)
        upd, st = _soap_update(gl, st, schedule(t), t, (t % SOAP_FREQ) == 0)
        prm = jax.tree_util.tree_map(lambda p, u: p + u, prm,
                                     jax.tree_util.tree_unflatten(treedef, upd))
        ema = jax.tree_util.tree_map(
            lambda e, p: EMA_DECAY * e + (1.0 - EMA_DECAY) * p, ema, prm)
        return (prm, st, ema, t + 1), loss

    @jit
    def run_chunk(prm, st, ema, t, key):
        prm_prev = jax.lax.stop_gradient(prm)
        keys = random.split(key, CHUNK)
        (p, s, e, t2), losses = jax.lax.scan(
            lambda c, k: one_step(prm_prev, c, k), (prm, st, ema, t), keys)
        return p, s, e, t2, losses[-1]

    return params, state, run_chunk, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    out = _ansatz(_PREDICT_MODEL.apply(params, X), X)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline pseudo-time stepping: each scan chunk is one implicit
    Euler sub-problem, solved by CHUNK inner SOAP steps."""
    params, state, run_chunk, rng_local = _make_train(rng)
    ema = params
    t = jnp.int32(0)

    rng_local, k = random.split(rng_local)
    params, state, ema, t, loss = run_chunk(params, state, ema, t, k)
    step_count = CHUNK
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    while step_count < MAX_ITER:
        if time.time() > deadline:
            break
        rng_local, k = random.split(rng_local)
        params, state, ema, t, loss = run_chunk(params, state, ema, t, k)
        step_count += CHUNK
        last_loss = float(loss)
        if eval_callback is not None and (step_count % 10000 == 0):
            eval_callback(ema, step_count, last_loss)

    return ema, step_count
