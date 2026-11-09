"""SPINN hard-IC ansatz, 1/c residual (kb1 104), energy anchor LAM_E 0.1, LR0 1.6e-2.
Parent: b02_enlr16.py (0.3186) — adds the exact LINEAR invariant <u/c> = const as a
second anchor at pull LAM_M = 0.1 (kb1 031/127 conserved quantities). Energy is
quadratic and can be met by a shrunk field with steeper gradients; this cannot.
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

NT, NX, NY = 128, 80, 80     # tensor-grid collocation: 819k points per step
RANK       = 192             # CP rank of the separable ansatz
WIDTH      = 64              # per-axis MLP width
DEPTH      = 4               # per-axis MLP hidden layers
NFREQ_T    = 20              # sin/cos embedding modes on t (spectrum needs ~8.5
                             # full cycles over [0,5]; 8 was right at the edge)
NFREQ_S    = 10              # sin/cos embedding modes on x and y
GATE_A     = 0.25            # hard-IC gate width: g = 1 - exp(-(t/GATE_A)^2)
LAM_BC     = 0.1             # Neumann term multiplier inside the kb1 021 pairing
MARCH_T0   = 5.0             # ablation: no marching (horizon starts at T_MAX)
MARCH_FRAC = 0.45            # wall fraction at which the horizon reaches T_MAX
LR0, LR1   = 1.6e-2, 1e-4      # exponential LR decay across the wall budget
EPS_SCHED  = (0.0,)          # ablation: causal weighting off (kb1 084)
EPS_DELTA  = 0.99            # promote eps when min_i w_i exceeds this
SYNC_EVERY = 50              # host sync / schedule refresh cadence
LAM_M      = 0.1           # relative pull of the <u/c> invariant
LAM_E      = 0.1           # relative pull of the energy-conservation term
SAVE_MARGIN_S = 6.0          # stop the loop this early so device_get + pickling
                             # finish before eval.py's TRAIN_TIME wall


# ══════════ Medium c(x, y) ══════════

def _coef_grid():
    """The 256 x 256 medium c(x, y) as an array C[iy, ix] plus its axis."""
    raw = np.loadtxt(_locate(COEF_FILE), delimiter=",", skiprows=1, dtype=np.float64)
    n = int(round(np.sqrt(len(raw))))
    return raw[:, 2].reshape(n, n).astype(np.float32), n


def coef_at(C, n, x, y):
    """Bilinear interpolation of the medium at arbitrary (x, y) — numpy or jnp."""
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


# ══════════ Per-axis branch network ══════════

class Branch(nn.Module):
    """1-D modified/gated MLP (kb1 050 'improved fully-connected architecture',
    the SPINN-mod backbone of kb1 070) over a sin/cos embedding of the axis."""
    width: int = WIDTH
    depth: int = DEPTH
    rank: int = RANK

    @nn.compact
    def __call__(self, feat):                     # feat: [F]
        gi = nn.initializers.glorot_normal()
        U = nn.tanh(nn.Dense(self.width, kernel_init=gi, name="Wu")(feat))
        V = nn.tanh(nn.Dense(self.width, kernel_init=gi, name="Wv")(feat))
        h = nn.tanh(nn.Dense(self.width, kernel_init=gi, name="W0")(feat))
        for i in range(self.depth - 1):
            z = nn.tanh(nn.Dense(self.width, kernel_init=gi, name=f"W{i+1}")(h))
            h = (1.0 - z) * U + z * V
        return nn.Dense(self.rank, kernel_init=gi)(h)     # [rank]


_BRANCH = Branch()
_KT = np.arange(1, NFREQ_T + 1, dtype=np.float32)
_KS = np.arange(1, NFREQ_S + 1, dtype=np.float32)
_RSCALE = 1.0 / np.sqrt(RANK, dtype=np.float32)


def _embed(z, lo, hi, ks):
    """Scalar axis coordinate -> [1 + 2*len(ks)] features."""
    s = 2.0 * (z - lo) / (hi - lo) - 1.0                  # -> [-1, 1]
    ang = jnp.pi * jnp.asarray(ks) * (s + 1.0) / 2.0
    return jnp.concatenate([jnp.reshape(s, (1,)), jnp.cos(ang), jnp.sin(ang)])


def _feat0(p, z, lo, hi, ks):
    return vmap(lambda s: _BRANCH.apply(p, _embed(s, lo, hi, ks)))(z)


def _feat012(p, z, lo, hi, ks):
    """Branch value and its first two derivatives (forward-mode AD, kb1 070)."""
    f = lambda s: _BRANCH.apply(p, _embed(s, lo, hi, ks))
    d0 = vmap(f)(z)
    d1 = vmap(jacfwd(f))(z)
    d2 = vmap(jacfwd(jacfwd(f)))(z)
    return d0, d1, d2


def _outer3(a, b, c):
    """rank-r CP contraction: [I,r],[J,r],[K,r] -> [I,J,K]."""
    ab = a[:, None, :] * b[None, :, :]                    # [I,J,r]
    return jnp.tensordot(ab, c, axes=([2], [1]))          # [I,J,K]


# ══════════ Loss ══════════

def _terms(params, t, x, y, c):
    """Raw per-time-slice PDE residual and the four Neumann-face residuals."""
    ft0, ft1, ft2 = _feat012(params["t"], t, T_MIN, T_MAX, _KT)
    fx0, fx1, fx2 = _feat012(params["x"], x, X_MIN, X_MAX, _KS)
    fy0, fy1, fy2 = _feat012(params["y"], y, Y_MIN, Y_MAX, _KS)

    N   = _outer3(ft0, fx0, fy0) * _RSCALE
    Nt  = _outer3(ft1, fx0, fy0) * _RSCALE
    Ntt = _outer3(ft2, fx0, fy0) * _RSCALE
    Nxx = _outer3(ft0, fx2, fy0) * _RSCALE
    Nyy = _outer3(ft0, fx0, fy2) * _RSCALE
    Nx  = _outer3(ft0, fx1, fy0) * _RSCALE
    Ny  = _outer3(ft0, fx0, fy1) * _RSCALE

    # hard IC gate g(t) = 1 - exp(-(t/GATE_A)^2) : g(0)=0, g'(0)=0, g -> 1
    e = jnp.exp(-((t / GATE_A) ** 2))
    g0 = (1.0 - e)[:, None, None]
    g1 = (2.0 * t / GATE_A ** 2 * e)[:, None, None]
    g2 = ((2.0 / GATE_A ** 2) * (1.0 - 2.0 * (t / GATE_A) ** 2) * e)[:, None, None]

    A0, A2 = _gA(x, IC_X0), _gA2(x, IC_X0)
    B0, B2 = _gA(y, IC_Y0), _gA2(y, IC_Y0)

    u_tt = g2 * N + 2.0 * g1 * Nt + g0 * Ntt
    u_xx = (A2[:, None] * B0[None, :])[None, :, :] + g0 * Nxx
    u_yy = (A0[:, None] * B2[None, :])[None, :, :] + g0 * Nyy

    r = (u_tt - c[None, :, :] * (u_xx + u_yy)) / c[None, :, :]   # kb1 104: x 1/c
    L_slice = jnp.mean(r ** 2, axis=(1, 2))               # [NT]

    # E(t) = <u_t^2/c + |grad u|^2> is EXACTLY conserved here (multiply the
    # PDE by u_t/c and integrate by parts; the Neumann faces kill the flux),
    # and u_t(0)=0 fixes its value analytically at E0 = <|grad u0|^2>.
    A1, B1 = _gA1(x, IC_X0), _gA1(y, IC_Y0)
    u_t = g1 * N + g0 * Nt
    u_x = (A1[:, None] * B0[None, :])[None, :, :] + g0 * Nx
    u_y = (A0[:, None] * B1[None, :])[None, :, :] + g0 * Ny
    E_slice = jnp.mean(u_t ** 2 / c[None, :, :] + u_x ** 2 + u_y ** 2,
                       axis=(1, 2))
    E0 = jnp.mean((A1[:, None] * B0[None, :]) ** 2
                  + (A0[:, None] * B1[None, :]) ** 2)
    # LINEAR invariant: integrating u_tt/c = lap(u) over the box kills the flux
    # by Neumann, so d2/dt2 <u/c> = 0, and u_t(0)=0 makes <u/c> exactly constant
    # (verified on an independent FD solve: 2.4e-2 drift, vs 5.6e-1 for <u>).
    # Unlike the quadratic energy, a uniformly shrunk field cannot satisfy it.
    u_f = (A0[:, None] * B0[None, :])[None, :, :] + g0 * N
    M_slice = jnp.mean(u_f / c[None, :, :], axis=(1, 2))
    M0 = jnp.mean((A0[:, None] * B0[None, :]) / c)
    L_ma = jnp.mean((M_slice / M0 - 1.0) ** 2)
    L_en = jnp.mean((E_slice / E0 - 1.0) ** 2)

    # Neumann faces: u_x at x = +-1 (all t, all y) and u_y at y = +-1
    xb = jnp.array([X_MIN, X_MAX], dtype=jnp.float32)
    bx0, bx1, _ = _feat012(params["x"], xb, X_MIN, X_MAX, _KS)
    by0, by1, _ = _feat012(params["y"], xb, Y_MIN, Y_MAX, _KS)

    Nx_b = _outer3(ft0, bx1, fy0) * _RSCALE               # [NT,2,NY]
    Ny_b = _outer3(ft0, fx0, by1) * _RSCALE               # [NT,NX,2]
    ux_b = (_gA1(xb, IC_X0)[:, None] * B0[None, :])[None, :, :] + g0 * Nx_b
    uy_b = (A0[:, None] * _gA1(xb, IC_Y0)[None, :])[None, :, :] + g0 * Ny_b
    L_bc = jnp.mean(ux_b ** 2) + jnp.mean(uy_b ** 2)
    return L_slice, L_bc, L_en, L_ma


def _loss(params, t, x, y, c, eps):
    """eps is already divided by the initial residual scale, so the causal
    weights (kb1 084) relax to 1 as the absolute residual falls."""
    L_slice, L_bc, L_en, L_ma = _terms(params, t, x, y, c)
    cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(L_slice)[:-1]])
    w = jax.lax.stop_gradient(jnp.exp(-eps * cum))
    L_res = jnp.mean(w * L_slice)
    # magnitude normalisation (kb1 021): parameter-free residual/BC balance
    MI = jax.lax.stop_gradient(L_res)
    L_bc = LAM_BC * L_bc
    MB = jax.lax.stop_gradient(L_bc)
    loss = (MB * L_res + MI * L_bc) / (MI + MB + 1e-30)
    # energy anchor at a FIXED relative pull, deliberately outside the kb1 021
    # pairing: that pairing equalises the pull of every term it contains, which
    # is why b01's energy penalty outvoted the residual and won with u = u0.
    loss = loss + jax.lax.stop_gradient(loss) * (LAM_E * L_en + LAM_M * L_ma)
    return loss, (L_res, L_bc, jnp.min(w))


# ══════════ Public contract ══════════

def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u": [BS]}."""
    out = []
    CH = 1 << 16
    n = X.shape[0]
    for s in range(0, n, CH):
        Xi = jnp.asarray(X[s:s + CH])
        t, x, y = Xi[:, 0], Xi[:, 1], Xi[:, 2]
        ft = _feat0(params["t"], t, T_MIN, T_MAX, _KT)
        fx = _feat0(params["x"], x, X_MIN, X_MAX, _KS)
        fy = _feat0(params["y"], y, Y_MIN, Y_MAX, _KS)
        N = jnp.sum(ft * fx * fy, axis=-1) * _RSCALE
        u = _gA(x, IC_X0) * _gA(y, IC_Y0) + (1.0 - jnp.exp(-((t / GATE_A) ** 2))) * N
        out.append(u)
    return {"u": jnp.concatenate(out)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam on the separable hard-IC ansatz.

    Returns (params, step_count) — ``params`` is a dict of three Flax pytrees.
    """
    C_np, n_grid = _coef_grid()
    C = jnp.asarray(C_np)

    k0, k1, k2, rng_local = random.split(rng, 4)
    params = {
        "t": _BRANCH.init(k0, jnp.zeros((1 + 2 * NFREQ_T,))),
        "x": _BRANCH.init(k1, jnp.zeros((1 + 2 * NFREQ_S,))),
        "y": _BRANCH.init(k2, jnp.zeros((1 + 2 * NFREQ_S,))),
    }

    def sample(key, thor=T_MAX):
        ka, kb, kc = random.split(key, 3)
        it = (jnp.arange(NT) + random.uniform(ka, (NT,))) / NT     # sorted
        ix = (jnp.arange(NX) + random.uniform(kb, (NX,))) / NX
        iy = (jnp.arange(NY) + random.uniform(kc, (NY,))) / NY
        t = T_MIN + (thor - T_MIN) * it
        x = X_MIN + (X_MAX - X_MIN) * ix
        y = Y_MIN + (Y_MAX - Y_MIN) * iy
        c = coef_at(C, n_grid, x[:, None], y[None, :])
        return t, x, y, c

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

    # absolute residual scale at init: fixes the causal-weight units so that
    # w_i -> 1 as the true residual falls (kb1 084)
    rng_local, ks = random.split(rng_local)
    t0s, x0s, y0s, c0s = sample(ks)
    S_init = float(jnp.mean(_terms(params, t0s, x0s, y0s, c0s)[0])) + 1e-30
    eps_idx = 0
    eps = jnp.float32(EPS_SCHED[0] / S_init)

    # Warmup step: JIT-compiles the whole training step; it is a real step.
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
