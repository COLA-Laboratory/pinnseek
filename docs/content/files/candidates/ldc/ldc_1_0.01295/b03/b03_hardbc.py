"""LDC Re=3200: hard-BC (kb1 036) on the two-phase PTS-Adam->L-BFGS champion.
Velocity is composed as u=g_u+phi*NN, v=phi*NN so no-slip on the 3 static
walls is EXACT and the soft BC loss (residual/BC imbalance kb1 026/050) is
removed. Parent b02_v23. kb1 036 (distance hard-BC), 074, 130."""
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
        return nn.Dense(OUTPUT_DIM, kernel_init=ki)(H)


# ══════════ Hard-BC composition (kb1 036) ══════════
# phi(x,y) = 16 x(1-x) y(1-y): vanishes on all 4 walls (peak 1 at center).
# g_u(x,y) = y * w(x), w(x)=1-(2x-1)^LID_M: a regularized lid that is ~1 in
# the interior of the top edge and rounds to 0 at the two singular corners.
# u = g_u + phi*NN_u  -> u=0 exact on bottom/left/right, u=w(x) on the lid.
# v = phi*NN_v        -> v=0 exact on all four walls. p is free (gauge).
LID_M = 10


def _hard(xy_row, raw):
    x, y = xy_row[0], xy_row[1]
    phi = 16.0 * x * (1.0 - x) * y * (1.0 - y)
    w   = 1.0 - (2.0 * x - 1.0) ** LID_M
    g_u = y * w
    u = g_u + phi * raw[0]
    v = phi * raw[1]
    return jnp.array([u, v, raw[2]])


def _hard_batch(X, raw):
    x, y = X[:, 0], X[:, 1]
    phi = 16.0 * x * (1.0 - x) * y * (1.0 - y)
    w   = 1.0 - (2.0 * x - 1.0) ** LID_M
    g_u = y * w
    u = g_u + phi * raw[:, 0]
    v = phi * raw[:, 1]
    return u, v, raw[:, 2]


_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    raw = _PREDICT_MODEL.apply(params, X)
    u, v, p = _hard_batch(X, raw)
    return {"u": u, "v": v, "p": p}


# ══════════ Training config ══════════
N_INT        = 700
LR0          = 1e-3
SAVE_MARGIN_S = 8.0
TAU0         = 0.1        # pseudo-time step (implicit Euler dt)
TAU_BETA     = 0.5        # EMA for BB adaptive tau
TAU_EVERY    = 500
TAU_MIN, TAU_MAX = 1e-3, 10.0
NU           = 1.0 / RE
CONT_W       = 5.0        # continuity residual weight (b01/b02 optimum)

PHASE1_S     = 95.0       # wall seconds of PTS-Adam before the L-BFGS polish
N_LB_INT     = 4500       # dense fixed interior collocation for L-BFGS


def _make_train(rng_init):
    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))

    def pred1(p, xy):
        raw = model.apply(p, xy[None, :])[0]
        return _hard(xy, raw)

    d1_fn = jacfwd(pred1, argnums=1)
    d2_fn = jacfwd(d1_fn, argnums=1)

    def steady_pt(p, xy):
        uvp = pred1(p, xy); u, v = uvp[0], uvp[1]
        d1 = d1_fn(p, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        d2 = d2_fn(p, xy)
        u_xx, u_yy = d2[0, 0, 0], d2[0, 1, 1]
        v_xx, v_yy = d2[1, 0, 0], d2[1, 1, 1]
        rc  = u_x + v_y
        rmx = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
        rmy = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
        return rc, rmx, rmy, u, v

    steady_b = vmap(steady_pt, in_axes=(None, 0))

    def pde_loss(p, xy_int, uv_prev, tau):
        rc, rmx, rmy, u, v = steady_b(p, xy_int)
        rmx_pt = (u - uv_prev[:, 0]) / tau + rmx
        rmy_pt = (v - uv_prev[:, 1]) / tau + rmy
        return jnp.mean(CONT_W * rc ** 2 + rmx_pt ** 2 + rmy_pt ** 2)

    sched = optax.exponential_decay(LR0, transition_steps=4000,
                                    decay_rate=0.9, end_value=1e-4)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(params)

    def _sample(key):
        k0, k1 = random.split(key, 2)
        m = N_INT // 2
        uni = random.uniform(k0, (N_INT - m, INPUT_DIM))
        t   = random.uniform(k1, (m, INPUT_DIM))
        clu = 0.5 - 0.5 * jnp.cos(math.pi * t)
        xy_int = jnp.concatenate([uni, clu], 0)
        return xy_int

    @jit
    def step(p, prev, opt_state, key, tau):
        xy_int = _sample(key)
        uv_prev = lax.stop_gradient(vmap(pred1, in_axes=(None, 0))(prev, xy_int)[:, :2])
        loss, grads = value_and_grad(pde_loss)(p, xy_int, uv_prev, tau)
        upd, opt_state = optimizer.update(grads, opt_state)
        p = optax.apply_updates(p, upd)
        return p, opt_state, loss

    @jit
    def bb_norms(p, prev, key):
        X = random.uniform(key, (512, INPUT_DIM))
        uv_now = vmap(pred1, in_axes=(None, 0))(p, X)[:, :2]
        uv_old = vmap(pred1, in_axes=(None, 0))(prev, X)[:, :2]
        rc_n, rmx_n, rmy_n, _, _ = steady_b(p, X)
        rc_o, rmx_o, rmy_o, _, _ = steady_b(prev, X)
        du = jnp.linalg.norm(uv_now - uv_old)
        dr = jnp.sqrt(jnp.sum((rmx_n - rmx_o) ** 2 + (rmy_n - rmy_o) ** 2))
        return du, dr

    # ── Phase-2 dense FIXED collocation (built once, deterministic) ──
    rs = np.random.RandomState(2024)
    half = N_LB_INT // 2
    uni = rs.uniform(0.0, 1.0, size=(N_LB_INT - half, INPUT_DIM))
    t   = rs.uniform(0.0, 1.0, size=(half, INPUT_DIM))
    clu = 0.5 - 0.5 * np.cos(math.pi * t)            # wall-clustered
    xy_lb = jnp.asarray(np.concatenate([uni, clu], 0).astype(np.float32))

    def lbfgs_loss(p):
        rc, rmx, rmy, u, v = steady_b(p, xy_lb)
        return jnp.mean(CONT_W * rc ** 2 + rmx ** 2 + rmy ** 2)  # TRUE steady residual

    lb_opt = optax.lbfgs(memory_size=20)
    lb_vg  = optax.value_and_grad_from_state(lbfgs_loss)

    @jit
    def lbfgs_step(p, state):
        value, grad = lb_vg(p, state=state)
        upd, state = lb_opt.update(grad, state, p, value=value,
                                   grad=grad, value_fn=lbfgs_loss)
        p = optax.apply_updates(p, upd)
        return p, state, value

    return params, opt_state, step, bb_norms, lb_opt, lbfgs_step


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, opt_state, step, bb_norms, lb_opt, lbfgs_step = _make_train(rng)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    phase1_deadline = min(_T0 + PHASE1_S, deadline)
    total = TRAIN_TIME - SAVE_MARGIN_S

    # ── Phase 1: PTS-Adam basin finding ──
    key = rng
    prev = params
    key, sk = random.split(key)
    params, opt_state, loss = step(params, prev, opt_state, sk, jnp.float32(TAU0))
    tau = TAU0
    step_count = 1
    for it in range(1, MAX_ITER):
        if time.time() > phase1_deadline:
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
                gamma = 0.5 * (1.0 + math.cos(math.pi * frac))  # 1 -> 0
                tau_hat = (0.2 + 0.8 * gamma) * du / dr
                tau = (1.0 - TAU_BETA) * tau + TAU_BETA * tau_hat
                tau = float(min(max(tau, TAU_MIN), TAU_MAX))
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(params, it, float(loss))

    # ── Phase 2: L-BFGS polish on the dense fixed set, true steady residual ──
    lb_state = lb_opt.init(params)
    best_params = params
    params, lb_state, val = lbfgs_step(params, lb_state)   # warm compile (1 step)
    best_val = float(val)
    best_params = params
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        params, lb_state, val = lbfgs_step(params, lb_state)
        step_count += 1
        v = float(val)
        if math.isfinite(v) and v <= best_val:
            best_val = v
            best_params = params
        elif not math.isfinite(v):
            break

    return best_params, step_count
