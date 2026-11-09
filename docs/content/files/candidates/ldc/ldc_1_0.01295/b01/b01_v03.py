"""LDC Re=3200: velocity-pressure PINN, multi-scale Fourier + mod-MLP,
wall-clustered collocation, strong fixed BC weight, Re curriculum.
Parent b01_v01. kb1 020 (multiscale FF), 050 (mod-MLP), 073/130 (Re ramp)."""
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
from jax import jacfwd, jit, random, vmap, value_and_grad
import numpy as np
import optax
from flax import linen as nn


# ══════════ Multi-scale fixed Fourier features (deterministic) ══════════
_FF_N   = 32
_SIGMAS = (1.0, 5.0, 15.0)
_B_LIST = [jnp.asarray(np.random.RandomState(100 + i)
                       .normal(0.0, s, size=(_FF_N, INPUT_DIM)).astype(np.float32))
           for i, s in enumerate(_SIGMAS)]


def _fourier(xy):
    feats = []
    for B in _B_LIST:
        proj = 2.0 * math.pi * (xy @ B.T)
        feats.append(jnp.cos(proj))
        feats.append(jnp.sin(proj))
    return jnp.concatenate(feats, axis=-1)


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
N_INT        = 700
N_BC_PER     = 48
LR0          = 1e-3
LAM_BC       = 100.0     # strong fixed BC weight — lid must drive the flow
RE_LO        = 100.0
RAMP_FRAC    = 0.4
SAVE_MARGIN_S = 6.0

_LOGLO, _LOGHI = math.log(RE_LO), math.log(RE)


def _make_train(rng_init):
    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))

    def pred1(p, xy):
        return model.apply(p, xy[None, :])[0]        # [3]

    d1_fn = jacfwd(pred1, argnums=1)                  # [3,2]
    d2_fn = jacfwd(d1_fn, argnums=1)                  # [3,2,2]

    def res_pt(p, xy, nu):
        uvp = pred1(p, xy); u, v = uvp[0], uvp[1]
        d1 = d1_fn(p, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        d2 = d2_fn(p, xy)
        u_xx, u_yy = d2[0, 0, 0], d2[0, 1, 1]
        v_xx, v_yy = d2[1, 0, 0], d2[1, 1, 1]
        rc  = u_x + v_y
        rmx = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        rmy = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        return rc, rmx, rmy

    res_batch = vmap(res_pt, in_axes=(None, 0, None))

    def pde_loss(p, xy_int, nu):
        rc, rmx, rmy = res_batch(p, xy_int, nu)
        return jnp.mean(rc ** 2 + rmx ** 2 + rmy ** 2)

    def bc_loss(p, xy_bc, u_bc, v_bc):
        pred = vmap(pred1, in_axes=(None, 0))(p, xy_bc)
        return jnp.mean((pred[:, 0] - u_bc) ** 2 + (pred[:, 1] - v_bc) ** 2)

    def total_loss(p, xy_int, nu, xy_bc, u_bc, v_bc):
        return pde_loss(p, xy_int, nu) + LAM_BC * bc_loss(p, xy_bc, u_bc, v_bc)

    sched = optax.exponential_decay(LR0, transition_steps=4000,
                                    decay_rate=0.9, end_value=1e-4)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(params)

    def _sample(key):
        k0, k1, kb = random.split(key, 3)
        m = N_INT // 2
        uni = random.uniform(k0, (N_INT - m, INPUT_DIM))
        t   = random.uniform(k1, (m, INPUT_DIM))
        clu = 0.5 - 0.5 * jnp.cos(math.pi * t)         # cluster near walls/corners
        xy_int = jnp.concatenate([uni, clu], 0)
        # boundary
        b1, b2, b3, b4 = random.split(kb, 4)
        eps = 1e-3
        c = lambda k: random.uniform(k, (N_BC_PER,), minval=eps, maxval=1.0 - eps)
        xt = c(b1); top = jnp.stack([xt, jnp.ones(N_BC_PER)], 1)
        xb = c(b2); bot = jnp.stack([xb, jnp.zeros(N_BC_PER)], 1)
        yl = c(b3); lft = jnp.stack([jnp.zeros(N_BC_PER), yl], 1)
        yr = c(b4); rgt = jnp.stack([jnp.ones(N_BC_PER), yr], 1)
        xy_bc = jnp.concatenate([top, bot, lft, rgt], 0)
        u_bc  = jnp.concatenate([jnp.ones(N_BC_PER), jnp.zeros(3 * N_BC_PER)], 0)
        v_bc  = jnp.zeros(4 * N_BC_PER)
        return xy_int, xy_bc, u_bc, v_bc

    @jit
    def update(p, opt_state, key, nu):
        xy_int, xy_bc, u_bc, v_bc = _sample(key)
        loss, grads = value_and_grad(total_loss)(p, xy_int, nu, xy_bc, u_bc, v_bc)
        upd, opt_state = optimizer.update(grads, opt_state)
        p = optax.apply_updates(p, upd)
        return p, opt_state, loss

    return params, opt_state, update


def _nu_now():
    p = (time.time() - _T0) / (TRAIN_TIME - SAVE_MARGIN_S)
    if p >= RAMP_FRAC:
        return 1.0 / RE
    fr = max(0.0, p) / RAMP_FRAC
    return 1.0 / math.exp(_LOGLO + fr * (_LOGHI - _LOGLO))


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, opt_state, update = _make_train(rng)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S

    key = rng
    key, sk = random.split(key)
    params, opt_state, loss = update(params, opt_state, sk, jnp.float32(1.0 / RE_LO))
    step = 1
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        key, sk = random.split(key)
        nu = jnp.float32(_nu_now())
        params, opt_state, loss = update(params, opt_state, sk, nu)
        step = it + 1
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(params, it, float(loss))

    return params, step
