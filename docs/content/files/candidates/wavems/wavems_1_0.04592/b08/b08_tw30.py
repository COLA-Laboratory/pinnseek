"""b06_aw15 + a negative-order TEMPORAL weight on the residual norm.
The norm was smoothed hard in space and (at the frequencies that matter)
not at all in time; this divides the squared residual by
(1+(Om/OMC)^2)^PT -- the temporal half of the FOSLS/H^-1 norm (kb1 087/124).
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
from jax import jit, random, vmap
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

NT, NX, NY = 512, 96, 96     # tensor-grid collocation: 2.95M points per step
RANK       = 192             # CP rank of the separable ansatz
WIDTH      = 64              # per-axis MLP width
DEPTH      = 4               # per-axis MLP hidden layers
NFREQ_T    = 20              # sin/cos embedding modes on t (spectrum needs ~8.5
                             # full cycles over [0,5]; 8 was right at the edge)
NFREQ_S    = 10              # sin/cos embedding modes on x and y
GATE_A     = 0.35            # hard-IC gate width: g = 1 - exp(-(t/GATE_A)^2)
LAM_BC     = 0.3             # Neumann term multiplier inside the kb1 021 pairing
LR0, LR1   = 3.6e-2, 1e-4      # exponential LR decay across the wall budget
SYNC_EVERY = 50              # host sync / schedule refresh cadence
LAM_M      = 0.1           # relative pull of the <u/c> invariant
LAM_E      = 0.18          # relative pull of the energy-conservation term
SIG_T      = 0.0155          # Gaussian half-width of the residual average, t
SIG_S      = 0.065         # ... and in x and y (0 disables the averaging)
M_SPEC     = 32            # Neumann cosine modes kept per spatial axis
P_NEG      = 0.0           # negative order of the spatial residual norm
LAM1       = 2.4674        # = (pi/2)^2, the first Neumann eigenvalue
AW         = 1.5           # early-time up-weight of the residual slices
OMC        = 30.0          # temporal negative-order cutoff, rad/s (<=0 disables)
PT         = 1.0           # order: squared-residual weight (1+(Om/OMC)^2)^-PT
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


def _feat01(p, z, lo, hi, ks):
    """Branch value and first derivative from a single jvp (2 fwd-equivalents)."""
    f = lambda s: _BRANCH.apply(p, _embed(s, lo, hi, ks))
    return vmap(lambda s: jax.jvp(f, (s,), (jnp.ones_like(s),)))(z)


def _feat012(p, z, lo, hi, ks):
    """Branch value and its first TWO derivatives from ONE nested jvp — 4
    forward-equivalents instead of the 1 + 2 + 4 of three separate jacfwd
    passes, and one vmapped kernel launch instead of three (kb1 070's
    forward-mode SPINN scheme, just assembled in a single sweep)."""
    f = lambda s: _BRANCH.apply(p, _embed(s, lo, hi, ks))

    def f01(s):
        return jax.jvp(f, (s,), (jnp.ones_like(s),))

    def f012(s):
        (d0, d1), (_, d2) = jax.jvp(f01, (s,), (jnp.ones_like(s),))
        return d0, d1, d2

    return vmap(f012)(z)


def _xy(b, c):
    """Outer product of two axis factor matrices, contracted-side first:
    [J,r],[K,r] -> [r, J*K].  Doing (x,y) before t keeps the intermediate
    3.3x smaller than the [NT,NX,r] of a t-first contraction."""
    r = b.shape[1]
    return (b.T[:, :, None] * c.T[:, None, :]).reshape(r, -1)


def _smooth_mat(n, lo, hi, sig):
    """Gaussian local-average matrix on the uniform grid of [lo, hi] with mirror
    (Neumann) reflection at both ends; rows sum to 1 so constants survive."""
    if sig <= 0.0:
        return None
    h = (hi - lo) / n
    i = np.arange(n)
    d0 = (i[:, None] - i[None, :]) * h
    d1 = (i[:, None] + i[None, :] + 1.0) * h
    d2 = (2 * n - 1 - i[:, None] - i[None, :]) * h
    K = (np.exp(-0.5 * (d0 / sig) ** 2) + np.exp(-0.5 * (d1 / sig) ** 2)
         + np.exp(-0.5 * (d2 / sig) ** 2))
    K /= K.sum(axis=1, keepdims=True)
    return jnp.asarray(K.astype(np.float32))


_ST = _smooth_mat(NT, T_MIN, T_MAX, SIG_T)

# Neumann cosine modes phi_j(z) = cos(j*pi*(z+1)/2), eigenvalues lam_j of -d2/dz2.
# The mirror-reflected Gaussian average of b03-b05 is exactly diagonal in this
# basis with multiplier exp(-lam*SIG_S^2/2), so squaring it is the weight
# exp(-lam*SIG_S^2); the extra (1+lam/LAM1)^-P_NEG factor makes the residual
# norm negative-order — the field error solves e_tt/c - lap(e) = r, so a
# residual at spatial mode lam feeds the metric with a gain ~ lam^-1/2 in
# amplitude (kb1 063's function-space/energy metric, kb1 124's norm-equivalent
# FOSLS weighting; kb1 027/086 set the same scale by local test functions).
_JM = np.arange(M_SPEC, dtype=np.float32)
_SJ = jnp.asarray(np.where(_JM == 0, 1.0, np.sqrt(2.0)).astype(np.float32))
_LAMJ = (np.pi * _JM / 2.0) ** 2
_LAM2 = _LAMJ[:, None] + _LAMJ[None, :]
_W2 = jnp.asarray((np.exp(-_LAM2 * SIG_S ** 2)
                   * (1.0 + _LAM2 / LAM1) ** (-P_NEG)).astype(np.float32))
_JMJ = jnp.asarray(_JM)

# Temporal half of the residual norm.  b03-b06 shaped the SPATIAL weight into
# a two-sided optimum but left the temporal one a fixed narrow Gaussian whose
# symbol exp(-Om^2 SIG_T^2/2) is still 0.93 at Om = 25 -- i.e. no temporal
# band limiting at all, while the FIELD is 99.5% below Om = 12.6.  The
# first-order-system (FOSLS) least-squares functional for this PDE is, in
# symbols, |r|^2 / (1 + (lam + Om^2/cbar^2)/beta) -- so the mixed formulation
# of kb1 087/052/124 IS a negative-order weight, and its temporal half has
# never been tried here (kb1 063's function-space metric says the same).
# Applied by a mirror-extended rFFT along t on the [NT,M,M] mode amplitudes.
_OMK = np.pi * np.arange(NT + 1) / (T_MAX - T_MIN)
_WTM = (jnp.asarray(((1.0 + (_OMK / OMC) ** 2) ** (-0.5 * PT)).astype(np.float32))
        if OMC > 0.0 else None)


def _qmat(z, lo, hi, n):
    """[n, M_SPEC] cosine quadrature matrix on the (jittered) uniform grid z:
    contracting a field against it gives mode amplitudes whose squares sum to
    the grid mean of the field's square (discrete Parseval)."""
    ang = jnp.pi * _JMJ[None, :] * (z[:, None] - lo) / (hi - lo)
    return jnp.cos(ang) * _SJ[None, :] / n


def _smooth(r, qx, qy):
    """Residual field [NT,NX,NY] -> smoothed mode amplitudes [NT,M,M]."""
    r = jnp.tensordot(r, qy, axes=(2, 0))             # [NT,NX,M]
    r = jnp.einsum("ixm,xj->ijm", r, qx)              # [NT,M,M]
    r = jnp.tensordot(_ST, r, axes=(1, 0))            # [NT,NT] . [NT,M,M]
    if _WTM is not None:
        ext = jnp.concatenate([r, r[::-1]], axis=0)   # mirror: no wrap-around
        r = jnp.fft.irfft(jnp.fft.rfft(ext, axis=0) * _WTM[:, None, None],
                          n=2 * NT, axis=0)[:NT]
    return r


# ══════════ Loss ══════════

def _terms(params, t, x, y, c):
    """Raw per-time-slice PDE residual and the four Neumann-face residuals."""
    ft0, ft1, ft2 = _feat012(params["t"], t, T_MIN, T_MAX, _KT)
    fx0, fx1, fx2 = _feat012(params["x"], x, X_MIN, X_MAX, _KS)
    fy0, fy1, fy2 = _feat012(params["y"], y, Y_MIN, Y_MAX, _KS)
    I, J, K = t.shape[0], x.shape[0], y.shape[0]

    # (x, y) factors first; the three time-derivative orders share P00's GEMM.
    P00 = _xy(fx0, fy0)
    T3 = jnp.concatenate([ft0, ft1, ft2], axis=0)              # [3I, r]
    E3 = (T3 @ P00).reshape(3, I, J, K) * _RSCALE
    N, Nt, Ntt = E3[0], E3[1], E3[2]
    Nlap = (ft0 @ (_xy(fx2, fy0) + _xy(fx0, fy2))).reshape(I, J, K) * _RSCALE

    # hard IC gate g(t) = 1 - exp(-(t/GATE_A)^2) : g(0)=0, g'(0)=0, g -> 1
    e = jnp.exp(-((t / GATE_A) ** 2))
    g0 = (1.0 - e)[:, None, None]
    g1 = (2.0 * t / GATE_A ** 2 * e)[:, None, None]
    g2 = ((2.0 / GATE_A ** 2) * (1.0 - 2.0 * (t / GATE_A) ** 2) * e)[:, None, None]

    A0, A2 = _gA(x, IC_X0), _gA2(x, IC_X0)
    B0, B2 = _gA(y, IC_Y0), _gA2(y, IC_Y0)

    u_tt = g2 * N + 2.0 * g1 * Nt + g0 * Ntt
    u_lap = ((A2[:, None] * B0[None, :] + A0[:, None] * B2[None, :])[None, :, :]
             + g0 * Nlap)

    r = (u_tt - c[None, :, :] * u_lap) / c[None, :, :]           # kb1 104: x 1/c
    qx, qy = _qmat(x, X_MIN, X_MAX, J), _qmat(y, Y_MIN, Y_MAX, K)
    L_slice = jnp.sum(_W2 * _smooth(r, qx, qy) ** 2, axis=(1, 2))   # [NT]
    wt = 1.0 + AW * (1.0 - t / T_MAX)
    L_slice = L_slice * (wt / jnp.mean(wt))

    # E(t) = <u_t^2/c + |grad u|^2> is EXACTLY conserved here (multiply the
    # PDE by u_t/c and integrate by parts; the Neumann faces kill the flux),
    # and u_t(0)=0 fixes its value analytically at E0 = <|grad u0|^2>.
    # Both anchors are quadratic/linear in a SEPARABLE field, so every integral
    # collapses to per-axis Gram matrices; only the 1/c-weighted one needs a
    # [r,r] moment of the medium.  No [NT,NX,NY] tensor is ever formed.
    A1, B1 = _gA1(x, IC_X0), _gA1(y, IC_Y0)
    rho = 1.0 / c
    ft00 = g0[:, :, 0] * ft0                                # [I, r]
    at = g1[:, :, 0] * ft0 + g0[:, :, 0] * ft1              # u_t coefficients
    Gx0, Gx1 = fx0.T @ fx0 / J, fx1.T @ fx1 / J
    Gy0, Gy1 = fy0.T @ fy0 / K, fy1.T @ fy1 / K
    KX = (fx0[:, :, None] * fx0[:, None, :]).reshape(J, -1)
    KY = (fy0[:, :, None] * fy0[:, None, :]).reshape(K, -1)
    H = jnp.sum((rho.T @ KX) * KY, axis=0).reshape(RANK, RANK) / (J * K)
    q = lambda a, G: jnp.sum((a @ G) * a, axis=1)
    vx = (fx1.T @ A1 / J) * (fy0.T @ B0 / K)
    vy = (fx0.T @ A0 / J) * (fy1.T @ B1 / K)
    cx = jnp.mean(A1 ** 2) * jnp.mean(B0 ** 2)
    cy = jnp.mean(A0 ** 2) * jnp.mean(B1 ** 2)
    E_slice = (_RSCALE ** 2 * (q(at, H) + q(ft00, Gx1 * Gy0) + q(ft00, Gx0 * Gy1))
               + 2.0 * _RSCALE * (ft00 @ (vx + vy)) + cx + cy)
    E0 = cx + cy
    # LINEAR invariant: integrating u_tt/c = lap(u) over the box kills the flux
    # by Neumann, so d2/dt2 <u/c> = 0, and u_t(0)=0 makes <u/c> exactly constant.
    # Unlike the quadratic energy, a uniformly shrunk field cannot satisfy it.
    mr = jnp.sum((fx0.T @ rho) * fy0.T, axis=1) / (J * K)
    M0 = jnp.mean((A0[:, None] * B0[None, :]) / c)
    M_slice = M0 + _RSCALE * (ft00 @ mr)
    L_ma = jnp.mean((M_slice / M0 - 1.0) ** 2)
    L_en = jnp.mean((E_slice / E0 - 1.0) ** 2)

    # Neumann faces: u_x at x = +-1 (all t, all y) and u_y at y = +-1
    xb = jnp.array([X_MIN, X_MAX], dtype=jnp.float32)
    _, bx1 = _feat01(params["x"], xb, X_MIN, X_MAX, _KS)
    _, by1 = _feat01(params["y"], xb, Y_MIN, Y_MAX, _KS)

    Nx_b = (ft0 @ _xy(bx1, fy0)).reshape(I, 2, K) * _RSCALE
    Ny_b = (ft0 @ _xy(fx0, by1)).reshape(I, J, 2) * _RSCALE
    ux_b = (_gA1(xb, IC_X0)[:, None] * B0[None, :])[None, :, :] + g0 * Nx_b
    uy_b = (A0[:, None] * _gA1(xb, IC_Y0)[None, :])[None, :, :] + g0 * Ny_b
    L_bc = jnp.mean(ux_b ** 2) + jnp.mean(uy_b ** 2)
    return L_slice, L_bc, L_en, L_ma


def _loss(params, t, x, y, c):
    L_slice, L_bc, L_en, L_ma = _terms(params, t, x, y, c)
    L_res = jnp.mean(L_slice)
    # magnitude normalisation (kb1 021): parameter-free residual/BC balance
    MI = jax.lax.stop_gradient(L_res)
    L_bc = LAM_BC * L_bc
    MB = jax.lax.stop_gradient(L_bc)
    loss = (MB * L_res + MI * L_bc) / (MI + MB + 1e-30)
    # energy anchor at a FIXED relative pull, deliberately outside the kb1 021
    # pairing: that pairing equalises the pull of every term it contains, which
    # is why b01's energy penalty outvoted the residual and won with u = u0.
    loss = loss + jax.lax.stop_gradient(loss) * (LAM_E * L_en + LAM_M * L_ma)
    return loss


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

    def sample(key):
        ka, kb, kc = random.split(key, 3)
        it = (jnp.arange(NT) + random.uniform(ka, ())) / NT        # uniform spacing
        ix = (jnp.arange(NX) + random.uniform(kb, ())) / NX
        iy = (jnp.arange(NY) + random.uniform(kc, ())) / NY
        t = T_MIN + (T_MAX - T_MIN) * it
        x = X_MIN + (X_MAX - X_MIN) * ix
        y = Y_MIN + (Y_MAX - Y_MIN) * iy
        c = coef_at(C, n_grid, x[:, None], y[None, :])
        return t, x, y, c

    opt = optax.scale_by_adam()
    opt_state = opt.init(params)

    @jit
    def update(params, opt_state, key, lr):
        t, x, y, c = sample(key)
        loss, grads = jax.value_and_grad(_loss)(params, t, x, y, c)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p - lr * u, params, updates)
        return params, opt_state, loss

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    span = max(deadline - _T0, 1e-6)

    # Warmup step: JIT-compiles the whole training step; it is a real step.
    rng_local, ks = random.split(rng_local)
    params, opt_state, loss = update(params, opt_state, ks, jnp.float32(LR0))
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
            params, opt_state, loss = update(params, opt_state, ks, lr)
        step_count += SYNC_EVERY
        last_loss = float(loss)
        if eval_callback is not None and (step_count % 1000 < SYNC_EVERY):
            eval_callback(params, step_count, last_loss)

    return params, step_count
