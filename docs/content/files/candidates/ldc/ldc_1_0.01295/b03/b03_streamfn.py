"""LDC Re=3200: divergence-free stream-function ansatz (u=psi_y, v=-psi_x).
Continuity is EXACT by construction; only momentum residual + soft velocity
BC remain, so a lower residual implies lower velocity error (b02's gap fix).
Same PTS-Adam->L-BFGS pipeline. Parent b02_v23. kb1 019 (VV/stream), 074, 130."""
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
from jax import jacfwd, grad, jit, random, vmap, value_and_grad, lax
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
    """Outputs raw[0]=stream function psi, raw[1]=pressure p (raw[2] unused)."""
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


def _uvp_point(params, xy):
    """(u, v, p) at a single point from psi=raw[0], p=raw[1]."""
    def psi_scalar(a):
        return _PREDICT_MODEL.apply(params, a[None, :])[0][0]
    d = grad(psi_scalar)(xy)          # [psi_x, psi_y]
    pp = _PREDICT_MODEL.apply(params, xy[None, :])[0][1]
    return d[1], -d[0], pp            # u=psi_y, v=-psi_x, p


def predict_fn(params, X):
    u, v, p = vmap(_uvp_point, in_axes=(None, 0))(params, X)
    return {"u": u, "v": v, "p": p}


# ══════════ Training config ══════════
N_INT        = 700
N_BC_PER     = 48
LR0          = 1e-3
LAM_BC       = 100.0
SAVE_MARGIN_S = 8.0
TAU0         = 0.1
TAU_BETA     = 0.5
TAU_EVERY    = 500
TAU_MIN, TAU_MAX = 1e-3, 10.0
NU           = 1.0 / RE

PHASE1_S     = 95.0
N_LB_INT     = 4500
N_LB_BC      = 200


def _make_train(rng_init):
    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))

    def psi(p, xy):
        return model.apply(p, xy[None, :])[0][0]

    def pfield(p, xy):
        return model.apply(p, xy[None, :])[0][1]

    dpsi   = grad(psi, argnums=1)                       # [psi_x, psi_y]
    d2psi  = jacfwd(dpsi, argnums=1)                    # [2,2]
    d3psi  = jacfwd(d2psi, argnums=1)                   # [2,2,2]
    dp     = grad(pfield, argnums=1)                    # [p_x, p_y]

    def steady_pt(p, xy):
        g1 = dpsi(p, xy)
        u, v = g1[1], -g1[0]
        H2 = d2psi(p, xy)           # psi second derivs
        psi_xx, psi_xy, psi_yy = H2[0, 0], H2[0, 1], H2[1, 1]
        u_x = psi_xy;  u_y = psi_yy
        v_x = -psi_xx; v_y = -psi_xy
        H3 = d3psi(p, xy)           # psi third derivs
        u_xx = H3[1, 0, 0]          # psi_yxx
        u_yy = H3[1, 1, 1]          # psi_yyy
        v_xx = -H3[0, 0, 0]         # -psi_xxx
        v_yy = -H3[0, 1, 1]         # -psi_xyy
        gp = dp(p, xy)
        p_x, p_y = gp[0], gp[1]
        rmx = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
        rmy = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
        return rmx, rmy, u, v

    steady_b = vmap(steady_pt, in_axes=(None, 0))

    def uv_batch(p, X):
        g1 = vmap(dpsi, in_axes=(None, 0))(p, X)
        return jnp.stack([g1[:, 1], -g1[:, 0]], axis=1)   # [N,2] (u,v)

    def pde_loss(p, xy_int, uv_prev, tau):
        rmx, rmy, u, v = steady_b(p, xy_int)
        rmx_pt = (u - uv_prev[:, 0]) / tau + rmx
        rmy_pt = (v - uv_prev[:, 1]) / tau + rmy
        return jnp.mean(rmx_pt ** 2 + rmy_pt ** 2)

    def bc_loss(p, xy_bc, u_bc, v_bc):
        uv = uv_batch(p, xy_bc)
        return jnp.mean((uv[:, 0] - u_bc) ** 2 + (uv[:, 1] - v_bc) ** 2)

    def total_loss(p, xy_int, uv_prev, tau, xy_bc, u_bc, v_bc):
        return pde_loss(p, xy_int, uv_prev, tau) + LAM_BC * bc_loss(p, xy_bc, u_bc, v_bc)

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
        xy_bc = jnp.concatenate([top, bot, lft, rgt], 0)
        u_bc  = jnp.concatenate([jnp.ones(N_BC_PER), jnp.zeros(3 * N_BC_PER)], 0)
        v_bc  = jnp.zeros(4 * N_BC_PER)
        return xy_int, xy_bc, u_bc, v_bc

    @jit
    def step(p, prev, opt_state, key, tau):
        xy_int, xy_bc, u_bc, v_bc = _sample(key)
        uv_prev = lax.stop_gradient(uv_batch(prev, xy_int))
        loss, grads = value_and_grad(total_loss)(
            p, xy_int, uv_prev, tau, xy_bc, u_bc, v_bc)
        upd, opt_state = optimizer.update(grads, opt_state)
        p = optax.apply_updates(p, upd)
        return p, opt_state, loss

    @jit
    def bb_norms(p, prev, key):
        X = random.uniform(key, (512, INPUT_DIM))
        uv_now = uv_batch(p, X)
        uv_old = uv_batch(prev, X)
        rmx_n, rmy_n = steady_b(p, X)[0:2]
        rmx_o, rmy_o = steady_b(prev, X)[0:2]
        du = jnp.linalg.norm(uv_now - uv_old)
        dr = jnp.sqrt(jnp.sum((rmx_n - rmx_o) ** 2 + (rmy_n - rmy_o) ** 2))
        return du, dr

    # ── Phase-2 dense FIXED collocation ──
    rs = np.random.RandomState(2024)
    half = N_LB_INT // 2
    uni = rs.uniform(0.0, 1.0, size=(N_LB_INT - half, INPUT_DIM))
    t   = rs.uniform(0.0, 1.0, size=(half, INPUT_DIM))
    clu = 0.5 - 0.5 * np.cos(math.pi * t)
    xy_lb = jnp.asarray(np.concatenate([uni, clu], 0).astype(np.float32))
    eps = 1e-3
    edge = np.linspace(eps, 1.0 - eps, N_LB_BC).astype(np.float32)
    top = np.stack([edge, np.ones_like(edge)], 1)
    bot = np.stack([edge, np.zeros_like(edge)], 1)
    lft = np.stack([np.zeros_like(edge), edge], 1)
    rgt = np.stack([np.ones_like(edge), edge], 1)
    xy_bc_lb = jnp.asarray(np.concatenate([top, bot, lft, rgt], 0))
    u_bc_lb  = jnp.asarray(np.concatenate(
        [np.ones(N_LB_BC), np.zeros(3 * N_LB_BC)], 0).astype(np.float32))
    v_bc_lb  = jnp.zeros(4 * N_LB_BC, dtype=jnp.float32)

    def lbfgs_loss(p):
        rmx, rmy, u, v = steady_b(p, xy_lb)
        pde = jnp.mean(rmx ** 2 + rmy ** 2)                 # TRUE steady residual
        bc  = bc_loss(p, xy_bc_lb, u_bc_lb, v_bc_lb)
        return pde + LAM_BC * bc

    lb_opt = optax.lbfgs(memory_size=20)
    lb_vg  = optax.value_and_grad_from_state(lbfgs_loss)

    @jit
    def lbfgs_step(p, state):
        value, grad_ = lb_vg(p, state=state)
        upd, state = lb_opt.update(grad_, state, p, value=value,
                                   grad=grad_, value_fn=lbfgs_loss)
        p = optax.apply_updates(p, upd)
        return p, state, value

    return params, opt_state, step, bb_norms, lb_opt, lbfgs_step


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, opt_state, step, bb_norms, lb_opt, lbfgs_step = _make_train(rng)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    phase1_deadline = min(_T0 + PHASE1_S, deadline)
    total = TRAIN_TIME - SAVE_MARGIN_S

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
                gamma = 0.5 * (1.0 + math.cos(math.pi * frac))
                tau_hat = (0.2 + 0.8 * gamma) * du / dr
                tau = (1.0 - TAU_BETA) * tau + TAU_BETA * tau_hat
                tau = float(min(max(tau, TAU_MIN), TAU_MAX))
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(params, it, float(loss))

    # ── Phase 2: L-BFGS polish, true steady residual ──
    lb_state = lb_opt.init(params)
    best_params = params
    params, lb_state, val = lbfgs_step(params, lb_state)
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
