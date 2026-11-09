"""Space/time-separable PINN: 1-D t-branch x a genuinely 2-D (x,y) branch.
Parent: b03_st2d.py (1.196).  Cheaper 2-D branch (W 128->64, NS 6400->3136)
to buy back steps; replaces the rank-1 CP factors X_r(x)Y_r(y)
by a single 2-D MLP Phi_r(x,y) on stratified points, keeping the hard-IC gate,
1/c residual (kb1 104), energy + mass anchors and the kb1 021 BC pairing.
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

NT         = 128             # time slices per step
NSX, NSY   = 56, 56          # stratified 2-D spatial cells (NSX*NSY points)
NB         = 48              # points per Neumann face
RANK       = 64              # space/time separation rank (reference needs ~24)
WIDTH_T    = 64              # t-branch width
DEPTH_T    = 4
WIDTH_S    = 64              # (x,y)-branch width
DEPTH_S    = 4
NFREQ_T    = 20              # sin/cos embedding modes on t
NFREQ_S    = 10              # sin/cos embedding modes on x and on y
GATE_A     = 0.35            # hard-IC gate width: g = 1 - exp(-(t/GATE_A)^2)
LAM_BC     = 0.1             # Neumann term multiplier inside the kb1 021 pairing
LR0, LR1   = 1.6e-2, 1e-4    # exponential LR decay across the wall budget
SYNC_EVERY = 50              # host sync / schedule refresh cadence
LAM_M      = 0.1             # relative pull of the <u/c> linear invariant
LAM_E      = 0.18            # relative pull of the energy-conservation anchor
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


# ══════════ Analytic initial pulse u0 = A(x) B(y) ══════════

def _gA(z, z0):
    return jnp.exp(-((z - z0) ** 2) / (2.0 * IC_SIGMA ** 2))


def _gA1(z, z0):
    return _gA(z, z0) * (-(z - z0) / IC_SIGMA ** 2)


def _gA2(z, z0):
    d = z - z0
    return _gA(z, z0) * (d ** 2 / IC_SIGMA ** 4 - 1.0 / IC_SIGMA ** 2)


# ══════════ Branch networks ══════════

class MMLP(nn.Module):
    """Modified/gated MLP (kb1 050), the SPINN-mod backbone of kb1 070."""
    width: int
    depth: int
    rank: int

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


_TNET = MMLP(WIDTH_T, DEPTH_T, RANK)
_SNET = MMLP(WIDTH_S, DEPTH_S, RANK)
_KT = np.arange(1, NFREQ_T + 1, dtype=np.float32)
_KS = np.arange(1, NFREQ_S + 1, dtype=np.float32)
_RSCALE = 1.0 / np.sqrt(RANK, dtype=np.float32)


def _emb1(z, lo, hi, ks):
    s = 2.0 * (z - lo) / (hi - lo) - 1.0
    ang = jnp.pi * jnp.asarray(ks) * (s + 1.0) / 2.0
    return jnp.concatenate([jnp.reshape(s, (1,)), jnp.cos(ang), jnp.sin(ang)])


def _emb2(xy):
    """(x, y) -> [2 + 4*NFREQ_S] features."""
    return jnp.concatenate([_emb1(xy[0], X_MIN, X_MAX, _KS),
                            _emb1(xy[1], Y_MIN, Y_MAX, _KS)])


def _t_feats(p, t):
    """t-branch value and its first two derivatives (forward-mode AD, kb1 070)."""
    f = lambda s: _TNET.apply(p, _emb1(s, T_MIN, T_MAX, _KT))
    return vmap(f)(t), vmap(jacfwd(f))(t), vmap(jacfwd(jacfwd(f)))(t)


def _s_feats(p, xy):
    """(x,y)-branch value, gradient and the two pure second derivatives."""
    f = lambda z: _SNET.apply(p, _emb2(z))
    d0 = vmap(f)(xy)                                   # [NS, R]
    d1 = vmap(jacfwd(f))(xy)                           # [NS, R, 2]
    d2 = vmap(jacfwd(jacfwd(f)))(xy)                   # [NS, R, 2, 2]
    return d0, d1[..., 0], d1[..., 1], d2[..., 0, 0], d2[..., 1, 1]


def _s_feats1(p, xy):
    """(x,y)-branch value and gradient only (boundary faces)."""
    f = lambda z: _SNET.apply(p, _emb2(z))
    d1 = vmap(jacfwd(f))(xy)
    return d1[..., 0], d1[..., 1]


def _mm(a, b):
    """[NT,R] x [NS,R] -> [NT,NS]."""
    return (a @ b.T) * _RSCALE


# ══════════ Loss ══════════

def _terms(params, t, xy, c, bx, by):
    ft0, ft1, ft2 = _t_feats(params["t"], t)
    fs0, fsx, fsy, fsxx, fsyy = _s_feats(params["s"], xy)

    x, y = xy[:, 0], xy[:, 1]
    A0, A1, A2 = _gA(x, IC_X0), _gA1(x, IC_X0), _gA2(x, IC_X0)
    B0, B1, B2 = _gA(y, IC_Y0), _gA1(y, IC_Y0), _gA2(y, IC_Y0)
    u0 = A0 * B0

    # hard IC gate g(t) = 1 - exp(-(t/GATE_A)^2): g(0)=0, g'(0)=0, g -> 1
    e = jnp.exp(-((t / GATE_A) ** 2))
    g0 = (1.0 - e)[:, None]
    g1 = (2.0 * t / GATE_A ** 2 * e)[:, None]
    g2 = ((2.0 / GATE_A ** 2) * (1.0 - 2.0 * (t / GATE_A) ** 2) * e)[:, None]

    N   = _mm(ft0, fs0)
    Nt  = _mm(ft1, fs0)
    Ntt = _mm(ft2, fs0)

    u_tt = g2 * N + 2.0 * g1 * Nt + g0 * Ntt
    u_xx = (A2 * B0)[None, :] + g0 * _mm(ft0, fsxx)
    u_yy = (A0 * B2)[None, :] + g0 * _mm(ft0, fsyy)

    r = (u_tt - c[None, :] * (u_xx + u_yy)) / c[None, :]      # kb1 104: x 1/c
    L_slice = jnp.mean(r ** 2, axis=1)                        # [NT]

    # E(t) = <u_t^2/c + |grad u|^2> is exactly conserved (multiply by u_t/c and
    # integrate by parts; Neumann kills the flux); u_t(0)=0 fixes E0 = <|grad u0|^2>.
    u_t = g1 * N + g0 * Nt
    u_x = (A1 * B0)[None, :] + g0 * _mm(ft0, fsx)
    u_y = (A0 * B1)[None, :] + g0 * _mm(ft0, fsy)
    E_slice = jnp.mean(u_t ** 2 / c[None, :] + u_x ** 2 + u_y ** 2, axis=1)
    E0 = jnp.mean((A1 * B0) ** 2 + (A0 * B1) ** 2)
    L_en = jnp.mean((E_slice / E0 - 1.0) ** 2)

    # linear invariant <u/c> = const (d2/dt2 <u/c> = 0 by Neumann; u_t(0)=0)
    u_f = u0[None, :] + g0 * N
    M_slice = jnp.mean(u_f / c[None, :], axis=1)
    M0 = jnp.mean(u0 / c)
    L_ma = jnp.mean((M_slice / M0 - 1.0) ** 2)

    # Neumann faces: u_x at x = +-1 (points bx), u_y at y = +-1 (points by)
    bxx, _ = _s_feats1(params["s"], bx)
    _, byy = _s_feats1(params["s"], by)
    ux_b = (_gA1(bx[:, 0], IC_X0) * _gA(bx[:, 1], IC_Y0))[None, :] + g0 * _mm(ft0, bxx)
    uy_b = (_gA(by[:, 0], IC_X0) * _gA1(by[:, 1], IC_Y0))[None, :] + g0 * _mm(ft0, byy)
    L_bc = jnp.mean(ux_b ** 2) + jnp.mean(uy_b ** 2)
    return L_slice, L_bc, L_en, L_ma


def _loss(params, t, xy, c, bx, by):
    L_slice, L_bc, L_en, L_ma = _terms(params, t, xy, c, bx, by)
    L_res = jnp.mean(L_slice)
    # magnitude normalisation (kb1 021): parameter-free residual/BC balance
    MI = jax.lax.stop_gradient(L_res)
    L_bc = LAM_BC * L_bc
    MB = jax.lax.stop_gradient(L_bc)
    loss = (MB * L_res + MI * L_bc) / (MI + MB + 1e-30)
    # conserved-quantity anchors at a FIXED relative pull, deliberately outside
    # the kb1 021 pairing (that pairing equalises every term it contains, and
    # u = u0 conserves energy perfectly — b02 finding 2).
    loss = loss + jax.lax.stop_gradient(loss) * (LAM_E * L_en + LAM_M * L_ma)
    return loss, (L_res, L_bc)


# ══════════ Public contract ══════════

def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u": [BS]}."""
    out = []
    CH = 1 << 15
    n = X.shape[0]
    ft_fn = jit(lambda p, t: vmap(lambda s: _TNET.apply(p, _emb1(s, T_MIN, T_MAX, _KT)))(t))
    fs_fn = jit(lambda p, z: vmap(lambda w: _SNET.apply(p, _emb2(w)))(z))
    for s in range(0, n, CH):
        Xi = jnp.asarray(X[s:s + CH])
        t, x, y = Xi[:, 0], Xi[:, 1], Xi[:, 2]
        ft = ft_fn(params["t"], t)
        fs = fs_fn(params["s"], jnp.stack([x, y], axis=1))
        N = jnp.sum(ft * fs, axis=-1) * _RSCALE
        u = _gA(x, IC_X0) * _gA(y, IC_Y0) + (1.0 - jnp.exp(-((t / GATE_A) ** 2))) * N
        out.append(u)
    return {"u": jnp.concatenate(out)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam on the space/time-separable hard-IC ansatz.

    Returns (params, step_count) — ``params`` is a dict of two Flax pytrees.
    """
    C_np, n_grid = _coef_grid()
    C = jnp.asarray(C_np)

    k0, k1, rng_local = random.split(rng, 3)
    params = {
        "t": _TNET.init(k0, jnp.zeros((1 + 2 * NFREQ_T,))),
        "s": _SNET.init(k1, jnp.zeros((2 + 4 * NFREQ_S,))),
    }

    cellx = np.repeat(np.arange(NSX, dtype=np.float32), NSY)
    celly = np.tile(np.arange(NSY, dtype=np.float32), NSX)
    cellx, celly = jnp.asarray(cellx), jnp.asarray(celly)
    NS = NSX * NSY

    def sample(key):
        ka, kb, kc, kd = random.split(key, 4)
        it = (jnp.arange(NT) + random.uniform(ka, (NT,))) / NT
        t = T_MIN + (T_MAX - T_MIN) * it
        # stratified 2-D points: one uniform draw inside each cell
        ux = (cellx + random.uniform(kb, (NS,))) / NSX
        uy = (celly + random.uniform(kc, (NS,))) / NSY
        x = X_MIN + (X_MAX - X_MIN) * ux
        y = Y_MIN + (Y_MAX - Y_MIN) * uy
        xy = jnp.stack([x, y], axis=1)
        c = coef_at(C, n_grid, x, y)
        # Neumann faces
        sb = (jnp.arange(NB) + random.uniform(kd, (NB,))) / NB
        s = X_MIN + (X_MAX - X_MIN) * sb
        bx = jnp.concatenate([jnp.stack([jnp.full((NB,), X_MIN), s], 1),
                              jnp.stack([jnp.full((NB,), X_MAX), s], 1)])
        by = jnp.concatenate([jnp.stack([s, jnp.full((NB,), Y_MIN)], 1),
                              jnp.stack([s, jnp.full((NB,), Y_MAX)], 1)])
        return t, xy, c, bx, by

    opt = optax.scale_by_adam()
    opt_state = opt.init(params)

    @jit
    def update(params, opt_state, key, lr):
        t, xy, c, bx, by = sample(key)
        (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
            params, t, xy, c, bx, by)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p - lr * u, params, updates)
        return params, opt_state, loss, aux

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    span = max(deadline - _T0, 1e-6)

    rng_local, ks = random.split(rng_local)
    params, opt_state, loss, aux = update(params, opt_state, ks, jnp.float32(LR0))
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
            params, opt_state, loss, aux = update(params, opt_state, ks, lr)
        step_count += SYNC_EVERY
        last_loss = float(loss)
        if eval_callback is not None and (step_count % 1000 < SYNC_EVERY):
            eval_callback(params, step_count, last_loss)

    return params, step_count
