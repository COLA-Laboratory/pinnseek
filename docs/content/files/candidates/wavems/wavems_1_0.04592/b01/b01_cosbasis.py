"""SPINN (kb1 070) with a COSINE temporal branch: every t-feature is cos(w_k t),
so the gate-free ansatz u = u0 + N(t) - N(0) has u(0)=u0 and u_t(0)=0 exactly.
Fixes the 645x gate dynamic range and the t-branch spectral bias (kb1 020/040);
diagnostics show this field is spatially smooth but temporally broadband.
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

NT, NX, NY = 128, 72, 72     # tensor-grid collocation, re-jittered every step
RANK       = 128             # CP rank of the separable ansatz
WIDTH      = 64              # spatial branch MLP width
DEPTH      = 4               # spatial branch MLP hidden layers
DEPTH_T    = 0               # t-branch hidden layers (0 = pure cosine basis)
WIDTH_T    = 96
NFREQ_S    = 10              # sin/cos embedding modes for x and y
NCOS_T     = 64              # cosine modes cos(k*pi*t/T_COS) for the t branch
T_COS      = 10.0            # cosine period base: w_k = k*pi/T_COS
OMEGA_REF  = 12.0            # feature preconditioning 1/(1+(w/OMEGA_REF)^2)
LR0, LR1   = 2e-3, 1e-4      # exponential LR decay across the wall budget
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


# ══════════ Branch networks ══════════

class BranchS(nn.Module):
    """Spatial 1-D MLP R -> R^RANK over a sin/cos embedding."""
    @nn.compact
    def __call__(self, feat):
        gi = nn.initializers.glorot_normal()
        h = feat
        for _ in range(DEPTH):
            h = nn.tanh(nn.Dense(WIDTH, kernel_init=gi)(h))
        return nn.Dense(RANK, kernel_init=gi)(h)


class BranchT(nn.Module):
    """Temporal branch over cos(w_k t) features only, so every output has zero
    t-derivative at t = 0 whatever the depth — the second IC is structural."""
    @nn.compact
    def __call__(self, feat):
        gi = nn.initializers.glorot_normal()
        h = feat
        for _ in range(DEPTH_T):
            h = nn.tanh(nn.Dense(WIDTH_T, kernel_init=gi)(h))
        return nn.Dense(RANK, kernel_init=gi)(h)


_BS = BranchS()
_BT = BranchT()
_KS = np.arange(1, NFREQ_S + 1, dtype=np.float32)
_OM = (np.pi * np.arange(NCOS_T, dtype=np.float32) / T_COS)
_OMS = (1.0 / (1.0 + (_OM / OMEGA_REF) ** 2)).astype(np.float32)
_RSCALE = float(1.0 / np.sqrt(RANK))


def _embed_s(z, lo, hi):
    s = 2.0 * (z - lo) / (hi - lo) - 1.0
    ang = jnp.pi * jnp.asarray(_KS) * (s + 1.0) / 2.0
    return jnp.concatenate([jnp.reshape(s, (1,)), jnp.cos(ang), jnp.sin(ang)])


def _embed_t(z):
    return jnp.asarray(_OMS) * jnp.cos(jnp.asarray(_OM) * z)


def _sfeat(p, z, lo, hi, order):
    f = lambda s: _BS.apply(p, _embed_s(s, lo, hi))
    for _ in range(order):
        f = jacfwd(f)
    return vmap(f)(z)


def _tfeat(p, z, order):
    f = lambda s: _BT.apply(p, _embed_t(s))
    for _ in range(order):
        f = jacfwd(f)
    return vmap(f)(z)


def _outer3(a, b, c):
    ab = a[:, None, :] * b[None, :, :]
    return jnp.tensordot(ab, c, axes=([2], [1]))


# ══════════ Loss ══════════

_TZERO = jnp.zeros((1,), dtype=jnp.float32)


def _terms(params, t, x, y, c):
    ft0 = _tfeat(params["t"], t, 0)
    ft2 = _tfeat(params["t"], t, 2)
    ftz = _tfeat(params["t"], _TZERO, 0)            # [1, RANK]
    fx0 = _sfeat(params["x"], x, X_MIN, X_MAX, 0)
    fx1 = _sfeat(params["x"], x, X_MIN, X_MAX, 1)
    fx2 = _sfeat(params["x"], x, X_MIN, X_MAX, 2)
    fy0 = _sfeat(params["y"], y, Y_MIN, Y_MAX, 0)
    fy1 = _sfeat(params["y"], y, Y_MIN, Y_MAX, 1)
    fy2 = _sfeat(params["y"], y, Y_MIN, Y_MAX, 2)

    A0, A2 = _gA(x, IC_X0), _gA2(x, IC_X0)
    B0, B2 = _gA(y, IC_Y0), _gA2(y, IC_Y0)

    # u = u0 + N(t) - N(0):  u_tt = N_tt ; u_xx = u0_xx + N_xx(t) - N_xx(0)
    u_tt = _outer3(ft2, fx0, fy0) * _RSCALE
    u_xx = ((A2[:, None] * B0[None, :])[None, :, :]
            + (_outer3(ft0, fx2, fy0) - _outer3(ftz, fx2, fy0)) * _RSCALE)
    u_yy = ((A0[:, None] * B2[None, :])[None, :, :]
            + (_outer3(ft0, fx0, fy2) - _outer3(ftz, fx0, fy2)) * _RSCALE)

    r = u_tt - c[None, :, :] * (u_xx + u_yy)
    L_slice = jnp.mean(r ** 2, axis=(1, 2))

    # Neumann faces
    xb = jnp.array([X_MIN, X_MAX], dtype=jnp.float32)
    bx1 = _sfeat(params["x"], xb, X_MIN, X_MAX, 1)
    by1 = _sfeat(params["y"], xb, Y_MIN, Y_MAX, 1)
    bx0 = _sfeat(params["x"], xb, X_MIN, X_MAX, 0)
    by0 = _sfeat(params["y"], xb, Y_MIN, Y_MAX, 0)
    ux_b = ((_gA1(xb, IC_X0)[:, None] * B0[None, :])[None, :, :]
            + (_outer3(ft0, bx1, fy0) - _outer3(ftz, bx1, fy0)) * _RSCALE)
    uy_b = ((A0[:, None] * _gA1(xb, IC_Y0)[None, :])[None, :, :]
            + (_outer3(ft0, fx0, by1) - _outer3(ftz, fx0, by1)) * _RSCALE)
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
    ftz = _tfeat(params["t"], _TZERO, 0)[0]
    out = []
    CH = 1 << 16
    for s in range(0, X.shape[0], CH):
        Xi = jnp.asarray(X[s:s + CH])
        t, x, y = Xi[:, 0], Xi[:, 1], Xi[:, 2]
        ft = _tfeat(params["t"], t, 0)
        fx = _sfeat(params["x"], x, X_MIN, X_MAX, 0)
        fy = _sfeat(params["y"], y, Y_MIN, Y_MAX, 0)
        dN = jnp.sum((ft - ftz[None, :]) * fx * fy, axis=-1) * _RSCALE
        out.append(_gA(x, IC_X0) * _gA(y, IC_Y0) + dN)
    return {"u": jnp.concatenate(out)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam on the cosine-basis separable ansatz."""
    C_np, n_grid = _coef_grid()
    C = jnp.asarray(C_np)

    k0, k1, k2, rng_local = random.split(rng, 4)
    params = {
        "t": _BT.init(k0, jnp.zeros((NCOS_T,))),
        "x": _BS.init(k1, jnp.zeros((1 + 2 * NFREQ_S,))),
        "y": _BS.init(k2, jnp.zeros((1 + 2 * NFREQ_S,))),
    }

    def sample(key):
        ka, kb, kc = random.split(key, 3)
        it = (jnp.arange(NT) + random.uniform(ka, (NT,))) / NT
        ix = (jnp.arange(NX) + random.uniform(kb, (NX,))) / NX
        iy = (jnp.arange(NY) + random.uniform(kc, (NY,))) / NY
        t = T_MIN + (T_MAX - T_MIN) * it
        x = X_MIN + (X_MAX - X_MIN) * ix
        y = Y_MIN + (Y_MAX - Y_MIN) * iy
        return t, x, y, coef_at(C, n_grid, x[:, None], y[None, :])

    opt = optax.scale_by_adam()
    opt_state = opt.init(params)

    @jit
    def update(params, opt_state, key, lr, eps):
        t, x, y, c = sample(key)
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
                                          jnp.float32(LR0), eps)
    step_count = 1
    last_loss = float(loss)

    for it in range(1, MAX_ITER):
        now = time.time()
        if now > deadline:
            break
        frac = min(max((now - _T0) / span, 0.0), 1.0)
        lr = jnp.float32(LR0 * (LR1 / LR0) ** frac)
        for _ in range(SYNC_EVERY):
            rng_local, ks = random.split(rng_local)
            params, opt_state, loss, aux = update(params, opt_state, ks, lr, eps)
        step_count += SYNC_EVERY
        last_loss = float(loss)
        wmin = float(aux[2])
        if wmin > EPS_DELTA and eps_idx < len(EPS_SCHED) - 1:
            eps_idx += 1
            eps = jnp.float32(EPS_SCHED[eps_idx] / S_init)
        if eval_callback is not None and (step_count % 1000 < SYNC_EVERY):
            eval_callback(params, step_count, last_loss)

    return params, step_count
