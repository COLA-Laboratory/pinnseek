"""LDC Re=3200: Fourier-feature modified-MLP PINN with adaptive grad-norm
loss weighting + Reynolds curriculum. Parent b00 (vanilla MLP). kb1
050 (mod-MLP + LR-annealing), 020 (Fourier feats), 073/130 (Re ramp)."""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 2
OUTPUT_DIM   = 3
INPUT_NAMES  = ("x", "y")
FIELD_NAMES  = ("u", "v", "p")

# --- Domain ---
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
HAS_TIME = False

# --- Equation structure ---
TIME_DERIV_ORDER        = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 3   # ru, rv, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("top", "bottom", "left", "right")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "box"

# --- Physical parameters ---
RE     = 3200.0      # NS-LDC Re=3200; harder regime than Re=1000
NU     = 1.0 / RE
U_LID  = 1.0

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 150.0     # total wall-clock seconds for the WHOLE process,
                       # from _T0: imports, JIT compile, training and param
                       # save all included — eval.py kills the process at
                       # this wall; return params with margin to spare
MAX_ITER   = 1_000_000 # safety cap on the loop (not expected to bind)

# --- Eval gate: full fidelity only under task/eval.py ---
if _os.environ.get("FORGE_EVAL_TOKEN") != "task/eval.py":
    TRAIN_TIME = 10.0                                # off-eval runs are smoke:
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")   # CPU, 10 s
# ═══════════════════════════════════════════════════════════════

import time, math
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, grad, random, vmap, value_and_grad
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


# ══════════ Fixed random Fourier features (deterministic B) ══════════
_FF_N     = 48
_FF_SIGMA = 3.0
_B_FF = jnp.asarray(
    np.random.RandomState(0).normal(0.0, _FF_SIGMA, size=(_FF_N, INPUT_DIM)).astype(np.float32)
)


def _fourier(xy):
    proj = 2.0 * math.pi * (xy @ _B_FF.T)
    return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


# ══════════ Modified (gated) MLP backbone, output (u, v, p) ══════════
WIDTH = 96
DEPTH = 4


class PINN(nn.Module):
    width: int = WIDTH
    depth: int = DEPTH

    @nn.compact
    def __call__(self, xy):
        ki  = jax.nn.initializers.glorot_normal()
        act = nn.tanh
        f = _fourier(xy)
        U = act(nn.Dense(self.width, kernel_init=ki)(f))
        V = act(nn.Dense(self.width, kernel_init=ki)(f))
        H = act(nn.Dense(self.width, kernel_init=ki)(f))
        for _ in range(self.depth - 1):
            Z = act(nn.Dense(self.width, kernel_init=ki)(H))
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(OUTPUT_DIM, kernel_init=ki)(H)


_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


# ══════════ Training config ══════════
N_INT       = 512      # interior collocation points / step
N_BC_PER    = 32       # boundary points / wall / step (→ 128 total)
LR0         = 1e-3
LAM_BC_INIT = 10.0
ALPHA       = 0.9      # EMA for adaptive weight (kb1 050)
WEIGHT_EVERY = 500
RE_LO       = 100.0    # curriculum start Reynolds
RAMP_FRAC   = 0.5      # fraction of wall spent ramping Re_lo -> RE
SAVE_MARGIN_S = 6.0

_LOGLO, _LOGHI = math.log(RE_LO), math.log(RE)


def _sample_bc(key, n):
    """n points per wall (corners excluded). Top lid u=1, else 0; v=0."""
    k1, k2, k3, k4 = random.split(key, 4)
    eps = 1e-3
    def coord(k):
        return random.uniform(k, (n,), minval=eps, maxval=1.0 - eps)
    xt = coord(k1); top = jnp.stack([xt, jnp.ones(n)], 1)
    xb = coord(k2); bot = jnp.stack([xb, jnp.zeros(n)], 1)
    yl = coord(k3); lft = jnp.stack([jnp.zeros(n), yl], 1)
    yr = coord(k4); rgt = jnp.stack([jnp.ones(n),  yr], 1)
    xy = jnp.concatenate([top, bot, lft, rgt], 0)
    u  = jnp.concatenate([jnp.ones(n), jnp.zeros(3 * n)], 0)
    v  = jnp.zeros(4 * n)
    return xy, u, v


def _make_train(rng_init):
    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat, unravel = jax.flatten_util.ravel_pytree(params)

    def predict_one(fp, xy):
        return model.apply(unravel(fp), xy[None, :])[0]      # [3]

    d1_fn = jacfwd(predict_one, argnums=1)                    # [3,2]
    d2_fn = jacfwd(d1_fn, argnums=1)                          # [3,2,2]

    def res_pt(fp, xy, nu):
        uvp = predict_one(fp, xy)
        u, v = uvp[0], uvp[1]
        d1 = d1_fn(fp, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        d2 = d2_fn(fp, xy)
        u_xx, u_yy = d2[0, 0, 0], d2[0, 1, 1]
        v_xx, v_yy = d2[1, 0, 0], d2[1, 1, 1]
        rc  = u_x + v_y
        rmx = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        rmy = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        return rc, rmx, rmy

    res_batch = vmap(res_pt, in_axes=(None, 0, None))

    def pde_loss(fp, xy_int, nu):
        rc, rmx, rmy = res_batch(fp, xy_int, nu)
        return jnp.mean(rc ** 2 + rmx ** 2 + rmy ** 2)

    def bc_loss(fp, xy_bc, u_bc, v_bc):
        pred = vmap(predict_one, in_axes=(None, 0))(fp, xy_bc)
        return jnp.mean((pred[:, 0] - u_bc) ** 2 + (pred[:, 1] - v_bc) ** 2)

    def total_loss(fp, xy_int, nu, xy_bc, u_bc, v_bc, lam):
        return pde_loss(fp, xy_int, nu) + lam * bc_loss(fp, xy_bc, u_bc, v_bc)

    sched = optax.exponential_decay(LR0, transition_steps=4000,
                                    decay_rate=0.9, end_value=1e-4)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(flat)

    @jit
    def update(fp, opt_state, key, nu, lam):
        ki, kb = random.split(key)
        xy_int = random.uniform(ki, (N_INT, INPUT_DIM))
        xy_bc, u_bc, v_bc = _sample_bc(kb, N_BC_PER)
        loss, grads = value_and_grad(total_loss)(fp, xy_int, nu, xy_bc, u_bc, v_bc, lam)
        upd, opt_state = optimizer.update(grads, opt_state)
        fp = optax.apply_updates(fp, upd)
        return fp, opt_state, loss, xy_int, xy_bc, u_bc, v_bc

    @jit
    def lam_hat(fp, xy_int, nu, xy_bc, u_bc, v_bc):
        g_r = grad(lambda p: pde_loss(p, xy_int, nu))(fp)
        g_b = grad(lambda p: bc_loss(p, xy_bc, u_bc, v_bc))(fp)
        return jnp.max(jnp.abs(g_r)) / (jnp.mean(jnp.abs(g_b)) + 1e-8)

    return flat, opt_state, update, lam_hat, unravel


def _nu_now():
    p = (time.time() - _T0) / (TRAIN_TIME - SAVE_MARGIN_S)
    if p >= RAMP_FRAC:
        return 1.0 / RE
    fr = max(0.0, p) / RAMP_FRAC
    return 1.0 / math.exp(_LOGLO + fr * (_LOGHI - _LOGLO))


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    flat, opt_state, update, lam_hat, unravel = _make_train(rng)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S

    key = rng
    # warmup (JIT-compiles; counted)
    key, sk = random.split(key)
    flat, opt_state, loss, xi, xb, ub, vb = update(
        flat, opt_state, sk, jnp.float32(1.0 / RE_LO), jnp.float32(LAM_BC_INIT))
    step = 1
    lam = LAM_BC_INIT
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        key, sk = random.split(key)
        nu = jnp.float32(_nu_now())
        flat, opt_state, loss, xi, xb, ub, vb = update(
            flat, opt_state, sk, nu, jnp.float32(lam))
        step = it + 1
        if it % WEIGHT_EVERY == 0:
            lh = float(lam_hat(flat, xi, nu, xb, ub, vb))
            if math.isfinite(lh) and lh > 0.0:
                lam = min((1.0 - ALPHA) * lam + ALPHA * lh, 1.0e4)
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(unravel(flat), it, float(loss))

    return unravel(flat), step
