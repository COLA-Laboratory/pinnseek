"""LDC Re=3200: div-free psi-omega PINN, CONDITIONED (psiw2 vs psiw).
Fixes psiw's 2.6e9 ill-conditioning: lower FF sigma=(1,4,8), scaled vorticity
output (omega=OMEGA_SCALE*net), W_KIN=1, fewer L-BFGS pts + more headroom so the
polish can actually converge. Two-phase PTS-Adam -> L-BFGS. kb1 019, 093, 130."""
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
from jax import jacfwd, jit, random, vmap, value_and_grad, lax, grad
import numpy as np
import optax
from flax import linen as nn


# ══════════ Multi-scale fixed Fourier features (deterministic) ══════════
_FF_N   = 32
_SIGMAS = (1.0, 4.0, 8.0)
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
PSI_OUT = 2          # network outputs (psi, omega_raw) — velocity via autodiff of psi
OMEGA_SCALE = 20.0   # omega = OMEGA_SCALE * net_raw; conditions the kinematic residual


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
        return nn.Dense(PSI_OUT, kernel_init=ki)(H)


_PREDICT_MODEL = PINN()


def _psi(params, xy):                 # scalar stream function at one point
    return _PREDICT_MODEL.apply(params, xy[None, :])[0, 0]


_GRAD_PSI = jit(vmap(grad(_psi, argnums=1), in_axes=(None, 0)))


def predict_fn(params, X):
    g = _GRAD_PSI(params, X)          # (N,2): (psi_x, psi_y)
    u = g[:, 1]                        # u =  psi_y
    v = -g[:, 0]                       # v = -psi_x
    return {"u": u, "v": v, "p": jnp.zeros_like(u)}


# ══════════ Training config ══════════
N_INT        = 700
N_BC_PER     = 48
LR0          = 1e-3
LAM_BC       = 100.0
SAVE_MARGIN_S = 8.0
TAU0         = 0.1        # pseudo-time step (implicit Euler dt)
TAU_BETA     = 0.5        # EMA for BB adaptive tau
TAU_EVERY    = 500
TAU_MIN, TAU_MAX = 1e-3, 10.0
NU           = 1.0 / RE

PHASE1_S     = 100.0      # wall SAFETY cap; more L-BFGS headroom than psiw
N_PH1_STEPS  = 55000        # deterministic handoff (fits ~100s at ~570 steps/s)
TAU_FRAC_END = 0.669      # replicate wall version's tau anneal (95 s / 142 s)
N_LB_INT     = 3000       # fewer L-BFGS pts -> more iterations in the polish window
N_LB_BC      = 200        # fixed boundary points per edge for L-BFGS
W_KIN        = 1.0        # kinematic weight (psiw's 5.0 over-weighted the stiff term)


def _make_train(rng_init):
    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))

    def pred1(p, xy):                 # (psi, omega) at one point
        return model.apply(p, xy[None, :])[0]

    d1_fn = jacfwd(pred1, argnums=1)  # (2,2)
    d2_fn = jacfwd(d1_fn, argnums=1)  # (2,2,2)

    def steady_pt(p, xy):
        out = pred1(p, xy); w = OMEGA_SCALE * out[1]
        d1 = d1_fn(p, xy)
        psi_x, psi_y = d1[0, 0], d1[0, 1]
        w_x, w_y     = OMEGA_SCALE * d1[1, 0], OMEGA_SCALE * d1[1, 1]
        d2 = d2_fn(p, xy)
        psi_xx, psi_yy = d2[0, 0, 0], d2[0, 1, 1]
        w_xx, w_yy     = OMEGA_SCALE * d2[1, 0, 0], OMEGA_SCALE * d2[1, 1, 1]
        u = psi_y; v = -psi_x
        r_kin = psi_xx + psi_yy + w                       # laplacian(psi) + omega = 0
        r_trn = u * w_x + v * w_y - NU * (w_xx + w_yy)     # vorticity transport
        return r_kin, r_trn, w, u, v

    steady_b = vmap(steady_pt, in_axes=(None, 0))

    def pde_loss(p, xy_int, w_prev, tau):
        r_kin, r_trn, w, u, v = steady_b(p, xy_int)
        # implicit-Euler pseudo-time on the transported quantity (vorticity)
        r_trn_pt = (w - w_prev) / tau + r_trn
        return jnp.mean(W_KIN * r_kin ** 2 + r_trn_pt ** 2)

    def _psi_g(p, xy):                # (psi, psi_x, psi_y) at one point
        out = pred1(p, xy); d1 = d1_fn(p, xy)
        return out[0], d1[0, 0], d1[0, 1]

    psi_g_b = vmap(_psi_g, in_axes=(None, 0))

    def bc_loss(p, bc):
        xt, xb, xl, xr = bc            # points per edge
        pt, _,   pty = psi_g_b(p, xt)  # top:    psi=0, psi_y=1
        pb, _,   pby = psi_g_b(p, xb)  # bottom: psi=0, psi_y=0
        pl, plx, _   = psi_g_b(p, xl)  # left:   psi=0, psi_x=0
        pr, prx, _   = psi_g_b(p, xr)  # right:  psi=0, psi_x=0
        loss_psi = (jnp.mean(pt ** 2) + jnp.mean(pb ** 2)
                    + jnp.mean(pl ** 2) + jnp.mean(pr ** 2))
        loss_nsp = (jnp.mean((pty - 1.0) ** 2) + jnp.mean(pby ** 2)
                    + jnp.mean(plx ** 2) + jnp.mean(prx ** 2))
        return loss_psi + loss_nsp

    def total_loss(p, xy_int, w_prev, tau, bc):
        return pde_loss(p, xy_int, w_prev, tau) + LAM_BC * bc_loss(p, bc)

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
        return xy_int, (top, bot, lft, rgt)

    @jit
    def step(p, prev, opt_state, key, tau):
        xy_int, bc = _sample(key)
        w_prev = lax.stop_gradient(steady_b(prev, xy_int)[2])
        loss, grads = value_and_grad(total_loss)(p, xy_int, w_prev, tau, bc)
        upd, opt_state = optimizer.update(grads, opt_state)
        p = optax.apply_updates(p, upd)
        return p, opt_state, loss

    @jit
    def bb_norms(p, prev, key):
        X = random.uniform(key, (512, INPUT_DIM))
        _, r_now, w_now, _, _ = steady_b(p, X)
        _, r_old, w_old, _, _ = steady_b(prev, X)
        dw = jnp.linalg.norm(w_now - w_old)
        dr = jnp.linalg.norm(r_now - r_old)
        return dw, dr

    # ── Phase-2 dense FIXED collocation (built once, deterministic) ──
    rs = np.random.RandomState(2024)
    half = N_LB_INT // 2
    uni = rs.uniform(0.0, 1.0, size=(N_LB_INT - half, INPUT_DIM))
    t   = rs.uniform(0.0, 1.0, size=(half, INPUT_DIM))
    clu = 0.5 - 0.5 * np.cos(math.pi * t)            # wall-clustered
    xy_lb = jnp.asarray(np.concatenate([uni, clu], 0).astype(np.float32))
    eps = 1e-3
    edge = np.linspace(eps, 1.0 - eps, N_LB_BC).astype(np.float32)
    top = jnp.asarray(np.stack([edge, np.ones_like(edge)], 1))
    bot = jnp.asarray(np.stack([edge, np.zeros_like(edge)], 1))
    lft = jnp.asarray(np.stack([np.zeros_like(edge), edge], 1))
    rgt = jnp.asarray(np.stack([np.ones_like(edge), edge], 1))
    bc_lb = (top, bot, lft, rgt)

    def lbfgs_loss(p):
        r_kin, r_trn, w, u, v = steady_b(p, xy_lb)
        pde = jnp.mean(W_KIN * r_kin ** 2 + r_trn ** 2)   # TRUE steady residual
        bc  = bc_loss(p, bc_lb)
        return pde + LAM_BC * bc

    lb_opt = optax.lbfgs(memory_size=20)
    lb_vg  = optax.value_and_grad_from_state(lbfgs_loss)

    @jit
    def lbfgs_step(p, state):
        value, gr = lb_vg(p, state=state)
        upd, state = lb_opt.update(gr, state, p, value=value,
                                   grad=gr, value_fn=lbfgs_loss)
        p = optax.apply_updates(p, upd)
        return p, state, value

    return params, opt_state, step, bb_norms, lb_opt, lbfgs_step


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, opt_state, step, bb_norms, lb_opt, lbfgs_step = _make_train(rng)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    phase1_deadline = min(_T0 + PHASE1_S, deadline)

    # ── Phase 1: PTS-Adam basin finding on the vorticity transport ──
    key = rng
    prev = params
    key, sk = random.split(key)
    params, opt_state, loss = step(params, prev, opt_state, sk, jnp.float32(TAU0))
    tau = TAU0
    step_count = 1
    for it in range(1, MAX_ITER):
        if it >= N_PH1_STEPS or time.time() > phase1_deadline:
            break
        prev = params
        key, sk = random.split(key)
        params, opt_state, loss = step(params, prev, opt_state, sk, jnp.float32(tau))
        step_count = it + 1
        if it % TAU_EVERY == 0:
            key, pk = random.split(key)
            dw, dr = bb_norms(params, prev, pk)
            dw = float(dw); dr = float(dr)
            if dr > 1e-12 and math.isfinite(dw):
                frac = TAU_FRAC_END * (it / N_PH1_STEPS)
                gamma = 0.5 * (1.0 + math.cos(math.pi * frac))
                tau_hat = (0.2 + 0.8 * gamma) * dw / dr
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
