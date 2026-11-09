"""LDC Re=3200: streamfunction-vorticity PINN, output (psi,omega), no-
penetration hard via psi=x(1-x)y(1-y)*net. Fourier+mod-MLP, adaptive bc
weight, Re curriculum. Parent b01_v01. kb1 050/020/073/130; VV form 019/093."""
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


# ══════════ Modified (gated) MLP backbone, output (psi_raw, omega) ══════════
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
        return nn.Dense(2, kernel_init=ki)(H)          # (psi_raw, omega)


_PREDICT_MODEL = PINN()


def _g(xy):
    x, y = xy[0], xy[1]
    return x * (1.0 - x) * y * (1.0 - y)               # 0 on all walls


def _psi_scalar(params, xy):
    pr = _PREDICT_MODEL.apply(params, xy[None, :])[0, 0]
    return _g(xy) * pr                                  # hard no-penetration


_grad_psi = jax.grad(_psi_scalar, argnums=1)           # (psi_x, psi_y)


def predict_fn(params, X):
    g = vmap(lambda xy: _grad_psi(params, xy))(X)       # [N,2]
    u = g[:, 1]
    v = -g[:, 0]
    return {"u": u, "v": v, "p": jnp.zeros_like(u)}


# ══════════ Training config ══════════
N_INT        = 400
N_BC_PER     = 48
LR0          = 1e-3
LAM_BC_INIT  = 10.0
ALPHA        = 0.9
WEIGHT_EVERY = 500
RE_LO        = 100.0
RAMP_FRAC    = 0.4
SAVE_MARGIN_S = 6.0
W_KIN        = 1.0        # weight on kinematic eq (omega + lap psi = 0)

_LOGLO, _LOGHI = math.log(RE_LO), math.log(RE)


def _flat_absmax(g):
    return jnp.max(jnp.abs(jnp.concatenate(
        [x.ravel() for x in jax.tree_util.tree_leaves(g)])))


def _flat_absmean(g):
    return jnp.mean(jnp.abs(jnp.concatenate(
        [x.ravel() for x in jax.tree_util.tree_leaves(g)])))


def _make_train(rng_init):
    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))

    def psi_s(p, xy):
        return _g(xy) * model.apply(p, xy[None, :])[0, 0]

    def omega_s(p, xy):
        return model.apply(p, xy[None, :])[0, 1]

    psi_g = jax.grad(psi_s, argnums=1)          # (psi_x, psi_y)
    psi_h = jacfwd(psi_g, argnums=1)            # [2,2]
    om_g  = jax.grad(omega_s, argnums=1)        # (om_x, om_y)
    om_h  = jacfwd(om_g, argnums=1)             # [2,2]

    def res_pt(p, xy, nu):
        pg = psi_g(p, xy); psi_x, psi_y = pg[0], pg[1]
        ph = psi_h(p, xy); psi_xx, psi_yy = ph[0, 0], ph[1, 1]
        om = omega_s(p, xy)
        og = om_g(p, xy); om_x, om_y = og[0], og[1]
        oh = om_h(p, xy); om_xx, om_yy = oh[0, 0], oh[1, 1]
        u, v = psi_y, -psi_x
        e_kin = om + psi_xx + psi_yy
        e_trn = u * om_x + v * om_y - nu * (om_xx + om_yy)
        return e_kin, e_trn

    res_batch = vmap(res_pt, in_axes=(None, 0, None))

    # boundary: dpsi via grad; tangential no-slip
    psi_grad_b = vmap(lambda p, xy: psi_g(p, xy), in_axes=(None, 0))

    def pde_loss(p, xy_int, nu):
        ek, et = res_batch(p, xy_int, nu)
        return W_KIN * jnp.mean(ek ** 2) + jnp.mean(et ** 2)

    def bc_loss(p, top, bot, lft, rgt):
        gt = psi_grad_b(p, top)   # psi_y = u -> 1
        gb = psi_grad_b(p, bot)   # psi_y = u -> 0
        gl = psi_grad_b(p, lft)   # psi_x = -v -> 0
        gr = psi_grad_b(p, rgt)   # psi_x = -v -> 0
        return (jnp.mean((gt[:, 1] - 1.0) ** 2)
                + jnp.mean(gb[:, 1] ** 2)
                + jnp.mean(gl[:, 0] ** 2)
                + jnp.mean(gr[:, 0] ** 2))

    def total_loss(p, xy_int, nu, top, bot, lft, rgt, lam):
        return pde_loss(p, xy_int, nu) + lam * bc_loss(p, top, bot, lft, rgt)

    sched = optax.exponential_decay(LR0, transition_steps=4000,
                                    decay_rate=0.9, end_value=1e-4)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(params)

    def _sample(key):
        k0, k1, k2, k3, k4 = random.split(key, 5)
        xy_int = random.uniform(k0, (N_INT, INPUT_DIM))
        eps = 1e-3
        c = lambda k: random.uniform(k, (N_BC_PER,), minval=eps, maxval=1.0 - eps)
        xt = c(k1); top = jnp.stack([xt, jnp.ones(N_BC_PER)], 1)
        xb = c(k2); bot = jnp.stack([xb, jnp.zeros(N_BC_PER)], 1)
        yl = c(k3); lft = jnp.stack([jnp.zeros(N_BC_PER), yl], 1)
        yr = c(k4); rgt = jnp.stack([jnp.ones(N_BC_PER), yr], 1)
        return xy_int, top, bot, lft, rgt

    @jit
    def update(p, opt_state, key, nu, lam):
        xy_int, top, bot, lft, rgt = _sample(key)
        loss, grads = value_and_grad(total_loss)(
            p, xy_int, nu, top, bot, lft, rgt, lam)
        upd, opt_state = optimizer.update(grads, opt_state)
        p = optax.apply_updates(p, upd)
        return p, opt_state, loss, (xy_int, top, bot, lft, rgt)

    @jit
    def lam_hat(p, batch, nu):
        xy_int, top, bot, lft, rgt = batch
        g_r = grad(lambda q: pde_loss(q, xy_int, nu))(p)
        g_b = grad(lambda q: bc_loss(q, top, bot, lft, rgt))(p)
        return _flat_absmax(g_r) / (_flat_absmean(g_b) + 1e-8)

    return params, opt_state, update, lam_hat


def _nu_now():
    p = (time.time() - _T0) / (TRAIN_TIME - SAVE_MARGIN_S)
    if p >= RAMP_FRAC:
        return 1.0 / RE
    fr = max(0.0, p) / RAMP_FRAC
    return 1.0 / math.exp(_LOGLO + fr * (_LOGHI - _LOGLO))


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, opt_state, update, lam_hat = _make_train(rng)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S

    key = rng
    key, sk = random.split(key)
    params, opt_state, loss, batch = update(
        params, opt_state, sk, jnp.float32(1.0 / RE_LO), jnp.float32(LAM_BC_INIT))
    step = 1
    lam = LAM_BC_INIT
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        key, sk = random.split(key)
        nu = jnp.float32(_nu_now())
        params, opt_state, loss, batch = update(
            params, opt_state, sk, nu, jnp.float32(lam))
        step = it + 1
        if it % WEIGHT_EVERY == 0:
            lh = float(lam_hat(params, batch, nu))
            if math.isfinite(lh) and lh > 0.0:
                lam = min((1.0 - ALPHA) * lam + ALPHA * lh, 1.0e4)
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(params, it, float(loss))

    return params, step
