"""TUCKER-contracted separable PINN: u = u0 + sum_kbc W_kbc (C_k(t)-C_k(0)) X_b(x) Y_c(y),
cosine t-features C_k (kb1 020/040) so u(0)=u0 and u_t(0)=0 hold exactly, X/Y are MLP
branches (kb1 070) and W a trainable core. Multilinear rank beats SPINN's diagonal CP
6x per FLOP here. Parent: b01_tucker.py — adds the time-marching curriculum that
already helped the CP variant (kb1 062/024/033).
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 1
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u",)

# --- Domain ---
T_MIN, T_MAX = 0.0, 5.0
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 2   # u_tt — the IC fixes both u and u_t at t = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 1   # r
HAS_SOURCE              = False  # no forcing: every term carries u

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi")
PERIODIC_AXES  = ()       # none: zero Neumann on all four faces
GEOMETRY_KIND  = "box"

# --- Physical parameters ---
COEF_FILE = "wavems_coef.csv"   # c(x, y) on a 256 x 256 grid over [-1, 1]^2
C_LO, C_HI = 1.0, 12.0          # the medium is two-valued; wave speed sqrt(c)
IC_X0, IC_Y0, IC_SIGMA = -0.5, 0.0, 0.3   # Gaussian pulse of the initial condition

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 100.0     # total wall-clock seconds for the WHOLE process,
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
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, random, vmap
import numpy as np
import optax
from flax import linen as nn


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Hyperparameters ══════════

NT, NX, NY = 128, 128, 128   # tensor-grid collocation, re-jittered every step
NCOS_T     = 48              # temporal cosine features cos(k*pi*t/T_COS)
T_COS      = 10.0            # w_k = k*pi/T_COS  (period 20 > 2*T_MAX: leaves the
                             # network free beyond t=5, so the residual can vanish)
RX, RY     = 20, 20          # spatial multilinear ranks
WIDTH      = 64              # spatial branch MLP width
DEPTH      = 4               # spatial branch MLP hidden layers
NFREQ_S    = 10              # sin/cos embedding modes for x and y
OMEGA_REF  = 12.0            # temporal feature preconditioning 1/(1+(w/OMEGA_REF)^2)
CORE_SCALE = 0.05            # init scale of the Tucker core
MARCH_T0   = 0.75            # initial collocation horizon
MARCH_FRAC = 0.45            # wall fraction at which the horizon reaches T_MAX
LR0, LR1   = 3e-3, 1e-4      # exponential LR decay across the wall budget
EPS_SCHED  = (0.0,)          # causal weighting (kb1 084) off by default
EPS_DELTA  = 0.99
SYNC_EVERY = 50
SAVE_MARGIN_S = 6.0


# ══════════ Medium c(x, y) ══════════

def _coef_grid():
    """The 256 x 256 medium c(x, y) as an array C[iy, ix] plus its axis."""
    raw = np.loadtxt(_locate(COEF_FILE), delimiter=",", skiprows=1, dtype=np.float64)
    n = int(round(np.sqrt(len(raw))))
    return raw[:, 2].reshape(n, n).astype(np.float32), n


def coef_at(C, n, x, y):
    """Bilinear interpolation of the medium at arbitrary (x, y)."""
    gx = (x - X_MIN) / (X_MAX - X_MIN) * (n - 1)
    gy = (y - Y_MIN) / (Y_MAX - Y_MIN) * (n - 1)
    gx = jnp.clip(gx, 0.0, n - 1 - 1e-6)
    gy = jnp.clip(gy, 0.0, n - 1 - 1e-6)
    i, j = jnp.floor(gx).astype(jnp.int32), jnp.floor(gy).astype(jnp.int32)
    fx, fy = gx - i, gy - j
    c00, c10 = C[j, i], C[j, i + 1]
    c01, c11 = C[j + 1, i], C[j + 1, i + 1]
    return (1 - fy) * ((1 - fx) * c00 + fx * c10) + fy * ((1 - fx) * c01 + fx * c11)


# ══════════ Analytic initial pulse (rank-1: u0 = A(x) B(y)) ══════════

def _gA(z, z0):
    return jnp.exp(-((z - z0) ** 2) / (2.0 * IC_SIGMA ** 2))


def _gA1(z, z0):
    return _gA(z, z0) * (-(z - z0) / IC_SIGMA ** 2)


def _gA2(z, z0):
    d = z - z0
    return _gA(z, z0) * (d ** 2 / IC_SIGMA ** 4 - 1.0 / IC_SIGMA ** 2)


# ══════════ Branches ══════════

class BranchS(nn.Module):
    """Spatial 1-D MLP R -> R^rank over a sin/cos embedding."""
    rank: int

    @nn.compact
    def __call__(self, feat):
        gi = nn.initializers.glorot_normal()
        h = feat
        for _ in range(DEPTH):
            h = nn.tanh(nn.Dense(WIDTH, kernel_init=gi)(h))
        return nn.Dense(self.rank, kernel_init=gi)(h)


_BX = BranchS(rank=RX)
_BY = BranchS(rank=RY)
_KS = np.arange(1, NFREQ_S + 1, dtype=np.float32)
_OM = (np.pi * np.arange(NCOS_T, dtype=np.float32) / T_COS)
_OMS = (1.0 / (1.0 + (_OM / OMEGA_REF) ** 2)).astype(np.float32)


def _embed_s(z, lo, hi):
    s = 2.0 * (z - lo) / (hi - lo) - 1.0
    ang = jnp.pi * jnp.asarray(_KS) * (s + 1.0) / 2.0
    return jnp.concatenate([jnp.reshape(s, (1,)), jnp.cos(ang), jnp.sin(ang)])


def _cos_feat(z):
    """Temporal features; every one has zero t-derivative at t = 0."""
    return jnp.asarray(_OMS) * jnp.cos(jnp.asarray(_OM) * z)


def _sfeat(mod, p, z, lo, hi, order):
    f = lambda s: mod.apply(p, _embed_s(s, lo, hi))
    for _ in range(order):
        f = jacfwd(f)
    return vmap(f)(z)


def _tfeat(z, order):
    f = _cos_feat
    for _ in range(order):
        f = jacfwd(f)
    return vmap(f)(z)


def _contract(Ct, W, Xf, Yf):
    """einsum('tk,kbc,xb,yc->txy') in the cheap order."""
    T1 = jnp.tensordot(Ct, W, axes=([1], [0]))        # [NT,RX,RY]
    T2 = jnp.einsum('tbc,xb->txc', T1, Xf)            # [NT,NX,RY]
    return jnp.einsum('txc,yc->txy', T2, Yf)          # [NT,NX,NY]


# ══════════ Loss ══════════

_TZERO = jnp.zeros((1,), dtype=jnp.float32)


def _terms(params, t, x, y, c):
    W = params["W"]
    C0 = _tfeat(t, 0)
    C2 = _tfeat(t, 2)
    Cz = _tfeat(_TZERO, 0)                            # [1, NCOS_T]
    Cd = C0 - Cz                                      # C_k(t) - C_k(0)
    X0 = _sfeat(_BX, params["x"], x, X_MIN, X_MAX, 0)
    X2 = _sfeat(_BX, params["x"], x, X_MIN, X_MAX, 2)
    Y0 = _sfeat(_BY, params["y"], y, Y_MIN, Y_MAX, 0)
    Y2 = _sfeat(_BY, params["y"], y, Y_MIN, Y_MAX, 2)

    A0, A2 = _gA(x, IC_X0), _gA2(x, IC_X0)
    B0, B2 = _gA(y, IC_Y0), _gA2(y, IC_Y0)

    u_tt = _contract(C2, W, X0, Y0)
    u_xx = (A2[:, None] * B0[None, :])[None] + _contract(Cd, W, X2, Y0)
    u_yy = (A0[:, None] * B2[None, :])[None] + _contract(Cd, W, X0, Y2)

    r = u_tt - c[None, :, :] * (u_xx + u_yy)
    L_slice = jnp.mean(r ** 2, axis=(1, 2))

    xb = jnp.array([X_MIN, X_MAX], dtype=jnp.float32)
    Xb1 = _sfeat(_BX, params["x"], xb, X_MIN, X_MAX, 1)
    Yb1 = _sfeat(_BY, params["y"], xb, Y_MIN, Y_MAX, 1)
    ux_b = ((_gA1(xb, IC_X0)[:, None] * B0[None, :])[None]
            + _contract(Cd, W, Xb1, Y0))
    uy_b = ((A0[:, None] * _gA1(xb, IC_Y0)[None, :])[None]
            + _contract(Cd, W, X0, Yb1))
    L_bc = jnp.mean(ux_b ** 2) + jnp.mean(uy_b ** 2)
    return L_slice, L_bc


def _loss(params, t, x, y, c, eps):
    L_slice, L_bc = _terms(params, t, x, y, c)
    cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(L_slice)[:-1]])
    w = jax.lax.stop_gradient(jnp.exp(-eps * cum))
    L_res = jnp.mean(w * L_slice)
    MI = jax.lax.stop_gradient(L_res)
    MB = jax.lax.stop_gradient(L_bc)
    loss = (MB * L_res + MI * L_bc) / (MI + MB + 1e-30)   # kb1 021
    return loss, (L_res, L_bc, jnp.min(w))


# ══════════ Public contract ══════════

def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u": [BS]}."""
    W = jnp.asarray(params["W"]).reshape(NCOS_T, RX * RY)
    Cz = _cos_feat(jnp.float32(0.0))
    out = []
    CH = 1 << 14
    for s in range(0, X.shape[0], CH):
        Xi = jnp.asarray(X[s:s + CH])
        t, x, y = Xi[:, 0], Xi[:, 1], Xi[:, 2]
        Cd = vmap(_cos_feat)(t) - Cz[None, :]
        xf = _sfeat(_BX, params["x"], x, X_MIN, X_MAX, 0)
        yf = _sfeat(_BY, params["y"], y, Y_MIN, Y_MAX, 0)
        P = (xf[:, :, None] * yf[:, None, :]).reshape(-1, RX * RY)
        du = jnp.sum((Cd @ W) * P, axis=-1)
        out.append(_gA(x, IC_X0) * _gA(y, IC_Y0) + du)
    return {"u": jnp.concatenate(out)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam on the Tucker-contracted cosine-basis ansatz."""
    C_np, n_grid = _coef_grid()
    C = jnp.asarray(C_np)

    k0, k1, k2, rng_local = random.split(rng, 4)
    params = {
        "W": CORE_SCALE * random.normal(k0, (NCOS_T, RX, RY)) / np.sqrt(RX * RY),
        "x": _BX.init(k1, jnp.zeros((1 + 2 * NFREQ_S,))),
        "y": _BY.init(k2, jnp.zeros((1 + 2 * NFREQ_S,))),
    }

    def sample(key, thor=T_MAX):
        ka, kb, kc = random.split(key, 3)
        it = (jnp.arange(NT) + random.uniform(ka, (NT,))) / NT
        ix = (jnp.arange(NX) + random.uniform(kb, (NX,))) / NX
        iy = (jnp.arange(NY) + random.uniform(kc, (NY,))) / NY
        t = T_MIN + (thor - T_MIN) * it
        x = X_MIN + (X_MAX - X_MIN) * ix
        y = Y_MIN + (Y_MAX - Y_MIN) * iy
        return t, x, y, coef_at(C, n_grid, x[:, None], y[None, :])

    opt = optax.scale_by_adam()
    opt_state = opt.init(params)

    @jit
    def update(params, opt_state, key, lr, eps, thor):
        t, x, y, c = sample(key, thor)
        (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
            params, t, x, y, c, eps)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p - lr * u, params, updates)
        return params, opt_state, loss, aux

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    span = max(deadline - _T0, 1e-6)

    rng_local, ks = random.split(rng_local)
    t0s, x0s, y0s, c0s = sample(ks)
    S_init = float(jnp.mean(_terms(params, t0s, x0s, y0s, c0s)[0])) + 1e-30
    eps_idx = 0
    eps = jnp.float32(EPS_SCHED[0] / S_init)

    rng_local, ks = random.split(rng_local)
    params, opt_state, loss, aux = update(params, opt_state, ks,
                                          jnp.float32(LR0), eps,
                                          jnp.float32(MARCH_T0))
    step_count = 1
    last_loss = float(loss)

    for it in range(1, MAX_ITER):
        now = time.time()
        if now > deadline:
            break
        frac = min(max((now - _T0) / span, 0.0), 1.0)
        lr = jnp.float32(LR0 * (LR1 / LR0) ** frac)
        thor = jnp.float32(MARCH_T0 + (T_MAX - MARCH_T0)
                           * min(frac / MARCH_FRAC, 1.0))
        for _ in range(SYNC_EVERY):
            rng_local, ks = random.split(rng_local)
            params, opt_state, loss, aux = update(params, opt_state, ks, lr, eps, thor)
        step_count += SYNC_EVERY
        last_loss = float(loss)
        wmin = float(aux[2])
        if wmin > EPS_DELTA and eps_idx < len(EPS_SCHED) - 1:
            eps_idx += 1
            eps = jnp.float32(EPS_SCHED[eps_idx] / S_init)
        if eval_callback is not None and (step_count % 1000 < SYNC_EVERY):
            eval_callback(params, step_count, last_loss)

    return params, step_count
