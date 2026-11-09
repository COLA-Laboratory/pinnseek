"""Mean-flow (RANS) PINN: Karman Reynolds stress as a marginal-stability eddy viscosity.

Parent: blocks/b01/b01_v06.py — same hard-BC lift, features, flux penalty, Adam.
Only the momentum residual changes: nu -> nu + nu_t(t,x,y), with nu_t set by the
Re_eff = Re_c rule and a logistic street-energy envelope. Reports the mean flow.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 3
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u", "v", "p")

# --- Domain ---
T_MIN, T_MAX = 0.0, 6.0
X_MIN, X_MAX = 0.0, 2.2
Y_MIN, Y_MAX = 0.0, 0.41
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 3   # ru, rv, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("inlet", "outlet", "walls", "cylinder")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "channel_with_cylinder"

# --- Geometry (DFG 2D benchmark, Schaefer & Turek) ---
CYL_X, CYL_Y = 0.2, 0.2      # cylinder centre; offset from the channel
CYL_R        = 0.05          # mid-height 0.205 to break the symmetry

# --- Physical parameters (DFG 2D-3, Schaefer & Turek: driven inflow) ---
U_PEAK  = 1.5        # peak centreline inflow; U(s) = U_PEAK*sin(pi*s/T_DRIVE)
T_DRIVE = 8.0        # driving period of the benchmark, in simulation time s
T_START = 0.0        # the task clock IS the benchmark clock: from rest
NU      = 1.0e-3     # kinematic viscosity (rho = 1)
RE      = (2.0 / 3.0) * U_PEAK * (2.0 * CYL_R) / NU   # = 100 at the drive peak
                     # Re(t) sweeps 0 -> 100 -> 71 across the window

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 480.0     # total wall-clock seconds for the WHOLE process,
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
from jax import jit, random
import numpy as np
import optax
from flax import linen as nn

# ── architecture / training knobs (b01_v06) ─────────────────────
WIDTH        = 128
DEPTH        = 5
L_IN         = 0.06
T_RAMP       = 0.5
N_UNI        = 2048
N_CYL        = 1024
N_WAKE       = 1024
W_CONT       = 1.0
W_FLUX       = 5.0
N_FLUX_TX    = 128
N_FLUX_Y     = 64
W_CYL        = 1.0
W_WAKE       = 1.0
LR_MAX       = 2e-3
LR_END       = 5e-5
WARMUP_FRAC  = 0.02
CLIP_NORM    = 10.0
SYNC_EVERY   = 100
SAVE_MARGIN_S = 8.0

# ── Karman Reynolds-stress closure ──────────────────────────────
RE_C    = 47.0     # critical Reynolds number of the circular-cylinder wake
C_G     = 0.10     # linear growth-rate constant, sigma = C_G (U/D)(Re/Re_c - 1)
T_HALF  = 4.8      # street energy reaches half saturation here (kb2 b00: the
                   # wake is steady to t ~ 4.2 and near peak by t ~ 5.5)
NUT_MAX = 1.0      # cap on nu_t / nu
NUT_SCALE = 1.0    # overall strength multiplier on the closure
W_XC    = 0.25     # eddy viscosity starts at the rear stagnation point
W_XW    = 0.10     # ramp width
W_Y0    = 0.06     # wake half-width at the cylinder
W_YS    = 0.05     # wake spreading rate d(half-width)/dx


def inlet_centreline(t):
    return U_PEAK * jnp.sin(jnp.pi * (t + T_START) / T_DRIVE)


def _ubar(t):
    return (2.0 / 3.0) * inlet_centreline(t)


# ── logistic street-energy envelope E(t) in [0, 1] ──────────────
_TG = np.linspace(T_MIN, T_MAX, 2401)
_UB = (2.0 / 3.0) * U_PEAK * np.sin(np.pi * _TG / T_DRIVE)
_REG = _UB * (2.0 * CYL_R) / NU
_SIG = C_G * (_UB / (2.0 * CYL_R)) * np.maximum(_REG / RE_C - 1.0, 0.0)
_PHI = np.concatenate([[0.0], np.cumsum(0.5 * (_SIG[1:] + _SIG[:-1])
                                        * np.diff(_TG))]) * 2.0
_PHI_H = float(np.interp(T_HALF, _TG, _PHI))
_E0 = float(np.exp(-_PHI_H))                    # seed fixed by the half time
_ENV = 1.0 / (1.0 + (1.0 / _E0 - 1.0) * np.exp(-_PHI))
_TGJ, _ENVJ = jnp.asarray(_TG), jnp.asarray(_ENV)


def _env(t):
    return jnp.interp(t, _TGJ, _ENVJ)


def _kap_sat(t):
    """nu_eff/nu - 1 at saturation, from the Re_eff = Re_c marginality rule."""
    return jnp.clip(_ubar(t) * (2.0 * CYL_R) / NU / RE_C - 1.0, 0.0, NUT_MAX)


def _swin(x, y):
    """Wake window carrying the Karman fluctuations."""
    q = jnp.clip((x - W_XC) / W_XW, 0.0, 1.0)
    sx = q * q * (3.0 - 2.0 * q)
    hw = W_Y0 + W_YS * jnp.clip(x - W_XC, 0.0, None)
    return sx * jnp.exp(-((y - CYL_Y) / hw) ** 2)


def _nut(t, x, y):
    return NU * NUT_SCALE * _kap_sat(t) * _env(t) * _swin(x, y)


# ══════════ Hard-constraint ansatz (b01_v06) ══════════

def _wall(y):
    return 4.0 * y * (Y_MAX - y) / (Y_MAX ** 2)


def _cyl(x, y):
    s = ((x - CYL_X) ** 2 + (y - CYL_Y) ** 2) / (CYL_R ** 2) - 1.0
    return -jnp.expm1(-s)


def _din(x):
    return jnp.tanh(x / L_IN)


def _ramp(t):
    return jnp.tanh(t / T_RAMP)


K_T = (1.0, 2.0, 4.0, 8.0, 16.0)
K_X = (1.0, 2.0, 4.0, 8.0, 16.0)
K_Y = (1.0, 2.0, 4.0)


def _feat(X):
    th = 2.0 * X[..., 0] / T_MAX - 1.0
    xh = 2.0 * X[..., 1] / X_MAX - 1.0
    yh = 2.0 * X[..., 2] / Y_MAX - 1.0
    f = [th, xh, yh]
    for z, ks in ((th, K_T), (xh, K_X), (yh, K_Y)):
        for k in ks:
            f.append(jnp.sin(jnp.pi * k * z))
            f.append(jnp.cos(jnp.pi * k * z))
    return jnp.stack(f, axis=-1)


class ModMLP(nn.Module):
    width: int = WIDTH
    depth: int = DEPTH
    out_dim: int = OUTPUT_DIM

    @nn.compact
    def __call__(self, z):
        a = nn.tanh
        U = a(nn.Dense(self.width, name="Wu")(z))
        V = a(nn.Dense(self.width, name="Wv")(z))
        H = a(nn.Dense(self.width, name="W0")(z))
        for i in range(self.depth - 1):
            Z = a(nn.Dense(self.width, name=f"W{i + 1}")(H))
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(self.out_dim, name="Wout",
                        kernel_init=nn.initializers.zeros,
                        bias_init=nn.initializers.zeros)(H)


_MODEL = ModMLP()


def _uvpn(params, X):
    """[N, 4]: trial field (u, v, p) plus the analytic nu_eff channel."""
    raw = _MODEL.apply(params, _feat(X))
    t, x, y = X[..., 0], X[..., 1], X[..., 2]
    w, c = _wall(y), _cyl(x, y)
    phi = w * c * _din(x) * _ramp(t)
    base = inlet_centreline(t) * w * c
    return jnp.stack([base + phi * raw[..., 0],
                      phi * raw[..., 1],
                      (1.0 - x / X_MAX) * raw[..., 2],
                      NU + _nut(t, x, y)], axis=-1)


def _uvp(params, X):
    return _uvpn(params, X)[..., 0:3]


def predict_fn(params, X):
    out = _uvp(params, jnp.asarray(X))
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


# ══════════ Residual by batched forward-mode autodiff ══════════

def _residuals(params, X):
    f = lambda Z: _uvpn(params, Z)
    ones = jnp.ones((X.shape[0],), X.dtype)
    zero = jnp.zeros_like(ones)
    Et = jnp.stack([ones, zero, zero], -1)
    Ex = jnp.stack([zero, ones, zero], -1)
    Ey = jnp.stack([zero, zero, ones], -1)

    def d12(e):
        g = lambda Z: jax.jvp(f, (Z,), (e,))
        (F, dF), (_, d2F) = jax.jvp(g, (X,), (e,))
        return F, dF, d2F

    F, Ft = jax.jvp(f, (X,), (Et,))
    _, Fx, Fxx = d12(Ex)
    _, Fy, Fyy = d12(Ey)

    u, v, ve = F[:, 0], F[:, 1], F[:, 3]
    u_t, v_t = Ft[:, 0], Ft[:, 1]
    u_x, v_x, p_x, ve_x = Fx[:, 0], Fx[:, 1], Fx[:, 2], Fx[:, 3]
    u_y, v_y, p_y, ve_y = Fy[:, 0], Fy[:, 1], Fy[:, 2], Fy[:, 3]
    u_xx, v_xx = Fxx[:, 0], Fxx[:, 1]
    u_yy, v_yy = Fyy[:, 0], Fyy[:, 1]

    # div( nu_eff (grad u + grad u^T) ) with div u = 0
    dif_x = ve * (u_xx + u_yy) + 2.0 * ve_x * u_x + ve_y * (u_y + v_x)
    dif_y = ve * (v_xx + v_yy) + ve_x * (u_y + v_x) + 2.0 * ve_y * v_y

    r_c = u_x + v_y
    r_mx = u_t + u * u_x + v * u_y + p_x - dif_x
    r_my = v_t + u * v_x + v * v_y + p_y - dif_y
    return r_c, r_mx, r_my


def _pt_loss(params, X):
    r_c, r_mx, r_my = _residuals(params, X)
    return W_CONT * r_c ** 2 + r_mx ** 2 + r_my ** 2


def _outside(X):
    return ((X[:, 1] - CYL_X) ** 2 + (X[:, 2] - CYL_Y) ** 2
            > CYL_R ** 2).astype(X.dtype)


# ══════════ Collocation sampling (resampled every step) ══════════

def _sample(key):
    k = random.split(key, 12)
    tu = random.uniform(k[0], (N_UNI,), minval=T_MIN, maxval=T_MAX)
    xu = random.uniform(k[1], (N_UNI,), minval=X_MIN, maxval=X_MAX)
    yu = random.uniform(k[2], (N_UNI,), minval=Y_MIN, maxval=Y_MAX)
    Xu = jnp.stack([tu, xu, yu], -1)

    tc = random.uniform(k[3], (N_CYL,), minval=T_MIN, maxval=T_MAX)
    th = random.uniform(k[4], (N_CYL,), minval=0.0, maxval=2.0 * jnp.pi)
    rr = CYL_R * jnp.exp(random.uniform(k[5], (N_CYL,)) * jnp.log(4.0))
    Xc = jnp.stack([tc, CYL_X + rr * jnp.cos(th),
                    CYL_Y + rr * jnp.sin(th)], -1)

    tw = random.uniform(k[6], (N_WAKE,), minval=T_MIN, maxval=T_MAX)
    xw = random.uniform(k[7], (N_WAKE,), minval=0.15, maxval=1.8)
    yw = random.uniform(k[8], (N_WAKE,), minval=0.06, maxval=0.35)
    Xw = jnp.stack([tw, xw, yw], -1)

    tq = random.uniform(k[9], (N_FLUX_TX,), minval=T_MIN, maxval=T_MAX)
    xq = random.uniform(k[10], (N_FLUX_TX,), minval=X_MIN, maxval=X_MAX)
    return Xu, Xc, Xw, tq, xq


_YQ = jnp.linspace(Y_MIN, Y_MAX, N_FLUX_Y)
_WQ = jnp.full((N_FLUX_Y,), (Y_MAX - Y_MIN) / (N_FLUX_Y - 1)).at[0].mul(0.5).at[-1].mul(0.5)
_QREF = U_PEAK * (2.0 / 3.0) * Y_MAX


def _flux_loss(params, tq, xq):
    T = jnp.repeat(tq, N_FLUX_Y)
    Xc_ = jnp.repeat(xq, N_FLUX_Y)
    Yc_ = jnp.tile(_YQ, tq.shape[0])
    P = jnp.stack([T, Xc_, Yc_], -1)
    u = _uvp(params, P)[:, 0]
    solid = ((Xc_ - CYL_X) ** 2 + (Yc_ - CYL_Y) ** 2 <= CYL_R ** 2)
    u = jnp.where(solid, 0.0, u).reshape(tq.shape[0], N_FLUX_Y)
    Q = jnp.sum(u * _WQ[None, :], axis=1)
    Q0 = inlet_centreline(tq) * (2.0 / 3.0) * Y_MAX
    return jnp.mean(((Q - Q0) / _QREF) ** 2)


def _loss_fn(params, Xu, Xc, Xw, tq, xq):
    mu, mw = _outside(Xu), _outside(Xw)
    lu = jnp.sum(_pt_loss(params, Xu) * mu) / (jnp.sum(mu) + 1e-8)
    lc = jnp.mean(_pt_loss(params, Xc))
    lw = jnp.sum(_pt_loss(params, Xw) * mw) / (jnp.sum(mw) + 1e-8)
    return lu + W_CYL * lc + W_WAKE * lw + W_FLUX * _flux_loss(params, tq, xq)


# ══════════ Training ══════════

def _lr_at(frac):
    warm = min(1.0, frac / WARMUP_FRAC) if WARMUP_FRAC > 0 else 1.0
    return LR_MAX * warm * (LR_END / LR_MAX) ** min(1.0, max(0.0, frac))


def _build(rng):
    key, rng = random.split(rng)
    params = _MODEL.init(key, _feat(jnp.zeros([1, INPUT_DIM])))
    tx = optax.chain(optax.clip_by_global_norm(CLIP_NORM), optax.scale_by_adam())
    opt_state = tx.init(params)

    @jit
    def update(params, opt_state, key, lr):
        loss, grads = jax.value_and_grad(_loss_fn)(params, *_sample(key))
        upd, opt_state = tx.update(grads, opt_state, params)
        upd = jax.tree_util.tree_map(lambda g: -lr * g, upd)
        return optax.apply_updates(params, upd), opt_state, loss

    return params, opt_state, update, rng


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, opt_state, update, rng = _build(rng)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S

    rng, k = random.split(rng)
    params, opt_state, loss = update(params, opt_state, k, jnp.float32(0.0))
    jax.block_until_ready(loss)
    t_loop = time.time()
    span = max(deadline - t_loop, 1.0)

    step_count, last_loss = 1, float(loss)
    lr = jnp.float32(_lr_at(0.0))
    for it in range(1, MAX_ITER):
        rng, k = random.split(rng)
        params, opt_state, loss = update(params, opt_state, k, lr)
        step_count = it + 1
        if it % SYNC_EVERY == 0:
            last_loss = float(loss)
            now = time.time()
            if now > deadline:
                break
            lr = jnp.float32(_lr_at((now - t_loop) / span))
            if eval_callback is not None and it % 1000 == 0:
                eval_callback(params, it, last_loss)
    return params, step_count
