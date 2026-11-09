"""LDC Re=3200: streamfunction-vorticity PINN (continuity EXACT via psi),
no-penetration hard (psi=x(1-x)y(1-y)*net), pseudo-time on vorticity transport
to avoid spurious solutions. Parent b01_v04. kb1 130/073, 019/093 (VV)."""
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
from jax import jacfwd, jit, random, vmap, value_and_grad, lax
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
    return _g(xy) * pr


_grad_psi = jax.grad(_psi_scalar, argnums=1)


def predict_fn(params, X):
    g = vmap(lambda xy: _grad_psi(params, xy))(X)       # (psi_x, psi_y)
    u = g[:, 1]
    v = -g[:, 0]
    return {"u": u, "v": v, "p": jnp.zeros_like(u)}


# ══════════ Training config ══════════
N_INT        = 600
N_BC_PER     = 48
LR0          = 1e-3
LAM_BC       = 100.0
W_KIN        = 1.0
SAVE_MARGIN_S = 6.0
TAU0         = 0.1
TAU_BETA     = 0.5
TAU_EVERY    = 500
TAU_MIN, TAU_MAX = 1e-3, 10.0
NU           = 1.0 / RE


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

    def parts_pt(p, xy):
        pg = psi_g(p, xy); psi_x, psi_y = pg[0], pg[1]
        ph = psi_h(p, xy); psi_xx, psi_yy = ph[0, 0], ph[1, 1]
        om = omega_s(p, xy)
        og = om_g(p, xy); om_x, om_y = og[0], og[1]
        oh = om_h(p, xy); om_xx, om_yy = oh[0, 0], oh[1, 1]
        u, v = psi_y, -psi_x
        e_kin = om + psi_xx + psi_yy
        r_trn = u * om_x + v * om_y - NU * (om_xx + om_yy)
        return e_kin, r_trn, om

    parts_b = vmap(parts_pt, in_axes=(None, 0))

    def pde_loss(p, xy_int, om_prev, tau):
        e_kin, r_trn, om = parts_b(p, xy_int)
        r_trn_pt = (om - om_prev) / tau + r_trn        # pseudo-time on vorticity
        return W_KIN * jnp.mean(e_kin ** 2) + jnp.mean(r_trn_pt ** 2)

    psi_grad_b = vmap(lambda p, xy: psi_g(p, xy), in_axes=(None, 0))

    def bc_loss(p, top, bot, lft, rgt):
        gt = psi_grad_b(p, top)   # psi_y = u -> 1
        gb = psi_grad_b(p, bot)   # psi_y = u -> 0
        gl = psi_grad_b(p, lft)   # psi_x = -v -> 0
        gr = psi_grad_b(p, rgt)   # psi_x = -v -> 0
        return (jnp.mean((gt[:, 1] - 1.0) ** 2) + jnp.mean(gb[:, 1] ** 2)
                + jnp.mean(gl[:, 0] ** 2) + jnp.mean(gr[:, 0] ** 2))

    def total_loss(p, xy_int, om_prev, tau, top, bot, lft, rgt):
        return pde_loss(p, xy_int, om_prev, tau) + LAM_BC * bc_loss(p, top, bot, lft, rgt)

    sched = optax.exponential_decay(LR0, transition_steps=4000,
                                    decay_rate=0.9, end_value=1e-4)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(params)

    def _sample(key):
        k0, k1, kb = random.split(key, 3)
        m = N_INT // 2
        uni = random.uniform(k0, (N_INT - m, INPUT_DIM))
        t   = random.uniform(k1, (m, INPUT_DIM))
        clu = 0.5 - 0.5 * jnp.cos(math.pi * t)
        xy_int = jnp.concatenate([uni, clu], 0)
        b1, b2, b3, b4 = random.split(kb, 4)
        eps = 1e-3
        c = lambda k: random.uniform(k, (N_BC_PER,), minval=eps, maxval=1.0 - eps)
        xt = c(b1); top = jnp.stack([xt, jnp.ones(N_BC_PER)], 1)
        xb = c(b2); bot = jnp.stack([xb, jnp.zeros(N_BC_PER)], 1)
        yl = c(b3); lft = jnp.stack([jnp.zeros(N_BC_PER), yl], 1)
        yr = c(b4); rgt = jnp.stack([jnp.ones(N_BC_PER), yr], 1)
        return xy_int, top, bot, lft, rgt

    @jit
    def step(p, prev, opt_state, key, tau):
        xy_int, top, bot, lft, rgt = _sample(key)
        om_prev = lax.stop_gradient(vmap(omega_s, in_axes=(None, 0))(prev, xy_int))
        loss, grads = value_and_grad(total_loss)(
            p, xy_int, om_prev, tau, top, bot, lft, rgt)
        upd, opt_state = optimizer.update(grads, opt_state)
        p = optax.apply_updates(p, upd)
        return p, opt_state, loss

    @jit
    def bb_norms(p, prev, key):
        X = random.uniform(key, (512, INPUT_DIM))
        om_now = vmap(omega_s, in_axes=(None, 0))(p, X)
        om_old = vmap(omega_s, in_axes=(None, 0))(prev, X)
        _, rt_n, _ = parts_b(p, X)
        _, rt_o, _ = parts_b(prev, X)
        du = jnp.linalg.norm(om_now - om_old)
        dr = jnp.linalg.norm(rt_n - rt_o)
        return du, dr

    return params, opt_state, step, bb_norms


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, opt_state, step, bb_norms = _make_train(rng)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    total = TRAIN_TIME - SAVE_MARGIN_S

    key = rng
    prev = params
    key, sk = random.split(key)
    params, opt_state, loss = step(params, prev, opt_state, sk, jnp.float32(TAU0))
    tau = TAU0
    step_count = 1
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        prev = params
        key, sk = random.split(key)
        params, opt_state, loss = step(params, prev, opt_state, sk, jnp.float32(tau))
        step_count = it + 1
        if it % TAU_EVERY == 0:
            key, pk = random.split(key)
            du, dr = bb_norms(params, prev, pk)
            du = float(du); dr = float(dr)
            if dr > 1e-12 and math.isfinite(du):
                frac = min(1.0, (time.time() - _T0) / total)
                gamma = 0.5 * (1.0 + math.cos(math.pi * frac))
                tau_hat = (0.2 + 0.8 * gamma) * du / dr
                tau = (1.0 - TAU_BETA) * tau + TAU_BETA * tau_hat
                tau = float(min(max(tau, TAU_MIN), TAU_MAX))
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(params, it, float(loss))

    return params, step_count
