"""In-segment residual measure: l_pde weighted by a(t)^T_GAMMA = 0.8 on top
of the volume weighting (VW_BETA 0.2). Changed vs b08_vw02.py: the far
bracket of the same knob — a strong down-weight of the residual inside the
handoff release region.
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
T_MIN, T_MAX = 0.0, 1.2
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
T_START = 4.8        # task t = 0 is benchmark time s = T_START
NU      = 1.0e-3     # kinematic viscosity (rho = 1)
RE      = (2.0 / 3.0) * U_PEAK * (2.0 * CYL_R) / NU   # = 100 at the drive peak
                     # the window itself runs at Re(t) = 95 -> 71

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 300.0     # total wall-clock seconds for the WHOLE process,
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
from jax import jit, random
import numpy as np
import optax
from flax import linen as nn
from soap_jax import soap


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


def inlet_centreline(t):
    return U_PEAK * jnp.sin(jnp.pi * (t + T_START) / T_DRIVE)


def inlet_profile(t, y):
    return 4.0 * inlet_centreline(t) * y * (Y_MAX - y) / (Y_MAX ** 2)


# ══════════ Architecture ══════════

_LOX = jnp.array([X_MIN, Y_MIN])
_SCX = jnp.array([2.0 / (X_MAX - X_MIN), 2.0 / (Y_MAX - Y_MIN)])

WIDTH, DEPTH = 128, 4
N_SEG = 16
SEG = [T_MIN + k * (T_MAX - T_MIN) / N_SEG for k in range(N_SEG + 1)]

_M_FF = 96
_SIGMA_FF = np.array([0.0, 5.0, 3.0], dtype=np.float32)
_B_FF = jnp.array((np.random.default_rng(12345).standard_normal((_M_FF, 3))
                   * _SIGMA_FF).astype(np.float32))


# --- ADF hard-BC machinery (kb1 036) ---------------------------------
# omega_i >= 0, zero exactly on boundary i, |grad omega_i| = 1 there:
#   bottom wall y = 0, top wall y = H, cylinder r = R, inlet x = 0.
# The outlet is deliberately absent: u, v are free there.
PHI_ELL  = 0.01        # cylinder: the Re=100 separation-layer thickness
PHI_WALL = 0.04        # channel walls: a developed profile, no thin layer
PHI_IN   = 0.15        # inlet: a Dirichlet plane, no viscous layer at all


def _omegas(txy):
    x, y = txy[:, 1], txy[:, 2]
    r = jnp.sqrt((x - CYL_X) ** 2 + (y - CYL_Y) ** 2 + 1e-9)
    return (y, Y_MAX - y, jnp.maximum(r - CYL_R, 0.0), x)


def _rconj(a, b):
    """R-conjunction (smooth min): 0 iff a or b is 0, positive otherwise."""
    return a + b - jnp.sqrt(a * a + b * b + 1e-12)


def _phi(txy):
    """Product form so each boundary carries its own saturation length: the
    no-slip walls and cylinder keep the Re=100 layer thickness 0.01, the inlet
    (a Dirichlet plane with no viscous layer) is blended over PHI_IN."""
    wb, wt, wc, wi = _omegas(txy)
    return (jnp.tanh(_rconj(wb, wt) / PHI_WALL) * jnp.tanh(wc / PHI_ELL)
            * jnp.tanh(wi / PHI_IN))


def _lift_u(txy):
    """Transfinite (Rvachev) interpolant of the Dirichlet data for u: equals
    inlet_profile on x = 0 and 0 on the walls and the cylinder."""
    wb, wt, wc, wi = _omegas(txy)
    n_in = wb * wt * wc
    den = n_in + wi * wt * wc + wi * wb * wc + wi * wb * wt
    return n_in / (den + 1e-12) * inlet_profile(txy[:, 0], txy[:, 2])


class PINN(nn.Module):
    """Gated modified-MLP (kb1 050) on a fixed random Fourier embedding
    (kb1 020). Time is normalised inside the *current* segment [t_lo, t_hi],
    passed as traced scalars so all segments share one compiled kernel.
    The raw output is wrapped in the ADF trial u = g + phi*u_hat, so the
    inlet profile and the no-slip walls/cylinder hold exactly (kb1 036)."""
    width: int = WIDTH
    depth: int = DEPTH

    @nn.compact
    def __call__(self, txy, t_lo, t_hi):
        kinit = nn.initializers.glorot_normal()
        zt = 2.0 * (txy[:, :1] - t_lo) / (t_hi - t_lo) - 1.0
        zxy = (txy[:, 1:] - _LOX) * _SCX - 1.0
        z = jnp.concatenate([zt, zxy], axis=-1)
        proj = 2.0 * jnp.pi * z @ _B_FF.T
        h0 = jnp.concatenate([jnp.cos(proj), jnp.sin(proj), z], axis=-1)
        U = nn.tanh(nn.Dense(self.width, kernel_init=kinit, name="U")(h0))
        V = nn.tanh(nn.Dense(self.width, kernel_init=kinit, name="V")(h0))
        h = nn.tanh(nn.Dense(self.width, kernel_init=kinit, name="L0")(h0))
        for i in range(1, self.depth):
            z_ = nn.tanh(nn.Dense(self.width, kernel_init=kinit, name=f"L{i}")(h))
            h = (1.0 - z_) * U + z_ * V
        raw = nn.Dense(OUTPUT_DIM, kernel_init=kinit, name="out")(h)
        phi = _phi(txy)[:, None]
        u = _lift_u(txy)[:, None] + phi * raw[:, 0:1]
        v = phi * raw[:, 1:2]
        return jnp.concatenate([u, v, raw[:, 2:3]], axis=-1)


# ══════════ Sampling sizes ══════════

X_OUT = 1.80          # training outlet; scoring stops at x = 1.7
N_UNI, N_WAKE, N_ANN = 2150, 717, 717
N_IC = 1024
N_OUT = 128

W_IC, W_OUT = 1200.0, 1.0
T_GAMMA = 0.8      # in-segment residual measure: weight a(t)^T_GAMMA
TAU_FRAC = 0.38       # handoff blend length, as a fraction of a segment
LR0, LR_DECAY, LR_EVERY, LR_END = 2.5e-3, 0.82, 900, 5e-5
N_INNER = 50
SAVE_MARGIN_S = 8.0
BS_COL = N_UNI + N_WAKE + N_ANN   # collocation points per step
N_POOL = 32768        # spatial pool carrying the frozen prior's stack
REFRESH = 4           # rebuild the pool every REFRESH chunks


def _load_ic():
    ic = np.loadtxt(_locate("cyl_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    xy = np.stack([ic[:, 0], ic[:, 1]], 1).astype(np.float32)
    uv = np.stack([ic[:, 2], ic[:, 3]], 1).astype(np.float32)
    return jnp.array(xy), jnp.array(uv)


P_UNI = N_POOL * N_UNI // BS_COL
P_WAKE = N_POOL * N_WAKE // BS_COL
P_ANN = N_POOL - P_UNI - P_WAKE

# --- kb1 086 (VW-PINNs): the residual's measure --------------------------
# The collocation mixture's density varies ~30x across the box, so the plain
# mean over it is a biased estimate of the residual integral: b08_diag1 finds
# the ring d < 0.02 around the cylinder carrying 15% of l_pde on 1% of the
# volume. w = (p_far / p_mix)^VW_BETA removes that bias at VW_BETA = 0.2
# (the parent is VW_BETA = 0), leaving the enrichment as pure importance
# sampling. p_mix is analytic — no KDE needed — and is a function of space
# alone, so it rides along on the prior's pool for free.
VW_BETA = 0.2
_A_BOX = (X_OUT - X_MIN) * (Y_MAX - Y_MIN)
_WK = (0.10, 1.30, 0.03, 0.38)          # the wake box of _sample_xy
_A_WK = (_WK[1] - _WK[0]) * (_WK[3] - _WK[2])
_R_ANN = 0.15                            # annulus thickness of _sample_xy
_F_UNI, _F_WAKE, _F_ANN = (N_UNI / BS_COL, N_WAKE / BS_COL, N_ANN / BS_COL)
_P_FAR = _F_UNI / _A_BOX                 # density where only the uniform part lives


def _vw_weight(xy):
    """(p_far / p_mix)^VW_BETA for the mixture of _sample_xy: uniform on the
    box, uniform on the wake box, and r = R + 0.15 u^2 in the annulus, whose
    radial density is 1 / (2 pi r * 2 R_ANN * sqrt((r - R)/R_ANN))."""
    x, y = xy[:, 0], xy[:, 1]
    r = jnp.sqrt((x - CYL_X) ** 2 + (y - CYL_Y) ** 2 + 1e-12)
    in_wk = ((x >= _WK[0]) & (x <= _WK[1])
             & (y >= _WK[2]) & (y <= _WK[3])).astype(jnp.float32)
    s = jnp.clip((r - CYL_R) / _R_ANN, 1e-6, 1.0)
    p_ann = jnp.where(r < CYL_R + _R_ANN,
                      1.0 / (2.0 * jnp.pi * r * 2.0 * _R_ANN * jnp.sqrt(s)), 0.0)
    p = _F_UNI / _A_BOX + _F_WAKE * in_wk / _A_WK + _F_ANN * p_ann
    return (_P_FAR / p) ** VW_BETA


def _sample_xy(key):
    """The parent's collocation mixture, sampled in space only: the prior's
    stack lives at t = t_k, so a spatial pool is all that must be frozen —
    each step still draws its own times."""
    ks = random.split(key, 6)
    xu = random.uniform(ks[0], (P_UNI, 1), minval=X_MIN, maxval=X_OUT)
    yu = random.uniform(ks[1], (P_UNI, 1), minval=Y_MIN, maxval=Y_MAX)
    xw = random.uniform(ks[2], (P_WAKE, 1), minval=0.10, maxval=1.30)
    yw = random.uniform(ks[3], (P_WAKE, 1), minval=0.03, maxval=0.38)
    th = random.uniform(ks[4], (P_ANN, 1), minval=0.0, maxval=2.0 * jnp.pi)
    rr = CYL_R + 0.15 * random.uniform(ks[5], (P_ANN, 1)) ** 2
    xy = jnp.concatenate([jnp.concatenate([xu, yu], 1),
                          jnp.concatenate([xw, yw], 1),
                          jnp.concatenate([CYL_X + rr * jnp.cos(th),
                                           CYL_Y + rr * jnp.sin(th)], 1)], 0)
    m = ((xy[:, 0] - CYL_X) ** 2 + (xy[:, 1] - CYL_Y) ** 2) > CYL_R ** 2
    return xy, m.astype(jnp.float32)


def _sample_bc(key, t_lo, t_hi):
    """Only the outlet survives: p = 0 there is the one condition the ADF
    trial does not already satisfy exactly."""
    ks = random.split(key, 2)
    yout = random.uniform(ks[0], (N_OUT, 1), minval=Y_MIN, maxval=Y_MAX)
    tout = random.uniform(ks[1], (N_OUT, 1), minval=t_lo, maxval=t_hi)
    return jnp.concatenate([tout, jnp.full((N_OUT, 1), X_OUT), yout], 1)


def _make_train(rng_init):
    xy_ic, uv_ic0 = _load_ic()
    n_ic = len(xy_ic)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]), 0.0, 0.3)

    def _dirs(shape):
        return (jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), shape),
                jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), shape),
                jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), shape))

    def _fg(p, Z, lo, hi):
        """(u, v, p) of the plain ADF trial, plus the Dirichlet lift g."""
        return jnp.concatenate([model.apply(p, Z, lo, hi),
                                _lift_u(Z)[:, None]], 1)          # [N, 4]

    def stack(p, X, lo, hi):
        f = lambda Z: _fg(p, Z, lo, hi)
        ex, ey, et = _dirs(X.shape)
        fx = lambda Z: jax.jvp(f, (Z,), (ex,))
        (val, dx), (_, dxx) = jax.jvp(fx, (X,), (ex,))
        fy = lambda Z: jax.jvp(f, (Z,), (ey,))
        (_, dy), (_, dyy) = jax.jvp(fy, (X,), (ey,))
        _, dt = jax.jvp(f, (X,), (et,))
        return val, dt, dx, dy, dxx, dyy

    @jit
    def build_pool(key, pp, lo, lop, hip):
        """The frozen prior q(x, y) = [u_prev - g, v_prev](t_k, x, y) and its
        first/second spatial derivatives, on a fresh spatial pool. q has no
        t-dependence, so this is the whole per-step cost of the handoff."""
        xy, m = _sample_xy(key)
        Z = jnp.concatenate([jnp.full((N_POOL, 1), lo), xy], 1)

        def f(W):
            o = model.apply(pp, W, lop, hip)
            return jnp.stack([o[:, 0] - _lift_u(W), o[:, 1]], 1)   # [N, 2]

        ex, ey, _ = _dirs(Z.shape)
        fx = lambda W: jax.jvp(f, (W,), (ex,))
        (q, qx), (_, qxx) = jax.jvp(fx, (Z,), (ex,))
        fy = lambda W: jax.jvp(f, (W,), (ey,))
        (_, qy), (_, qyy) = jax.jvp(fy, (Z,), (ey,))
        return xy, m * _vw_weight(xy), q, qx, qy, qxx, qyy

    def loss_fn(p, key, lo, hi, w, pool, ic_uv):
        xy, m, q, qx, qy, qxx, qyy = pool
        ks, kt, kb, ki = random.split(key, 4)
        idx = random.choice(ks, N_POOL, (BS_COL,))
        tt = random.uniform(kt, (BS_COL, 1), minval=lo, maxval=hi)
        Xc = jnp.concatenate([tt, xy[idx]], 1)
        mc = m[idx]
        Q, Qx, Qy, Qxx, Qyy = q[idx], qx[idx], qy[idx], qxx[idx], qyy[idx]

        val, dt, dx, dy, dxx, dyy = stack(p, Xc, lo, hi)
        s = jnp.tanh((tt[:, 0] - lo) / (TAU_FRAC * (hi - lo)))
        a = w * s + (1.0 - w)
        b = w * (1.0 - s)
        da = w * (1.0 - s ** 2) / (TAU_FRAC * (hi - lo))          # a'(t) = -b'(t)

        # blend the ADF trial (channels 0,1; lift in channel 3) with the prior
        def bl(f_, g_, q_):
            return g_ + a * (f_ - g_) + b * q_
        u = bl(val[:, 0], val[:, 3], Q[:, 0])
        v = a * val[:, 1] + b * Q[:, 1]
        u_x = bl(dx[:, 0], dx[:, 3], Qx[:, 0])
        v_x = a * dx[:, 1] + b * Qx[:, 1]
        u_y = bl(dy[:, 0], dy[:, 3], Qy[:, 0])
        v_y = a * dy[:, 1] + b * Qy[:, 1]
        u_xx = bl(dxx[:, 0], dxx[:, 3], Qxx[:, 0])
        v_xx = a * dxx[:, 1] + b * Qxx[:, 1]
        u_yy = bl(dyy[:, 0], dyy[:, 3], Qyy[:, 0])
        v_yy = a * dyy[:, 1] + b * Qyy[:, 1]
        u_t = (dt[:, 3] + da * (val[:, 0] - val[:, 3])
               + a * (dt[:, 0] - dt[:, 3]) - da * Q[:, 0])
        v_t = da * val[:, 1] + a * dt[:, 1] - da * Q[:, 1]

        r_c = u_x + v_y
        r_mx = u_t + u * u_x + v * u_y + dx[:, 2] - NU * (u_xx + u_yy)
        r_my = v_t + u * v_x + v * v_y + dy[:, 2] - NU * (v_xx + v_yy)
        r = r_c ** 2 + r_mx ** 2 + r_my ** 2
        # the time analogue of VW_BETA: the release region carries 1.4x the
        # rms residual (b08_diag1) because the trial there is prior-dominated,
        # exactly the "dense, structurally hard" situation VW_BETA down-weights
        # in space. a(t) is a function of t alone, so this is a pure measure.
        wt = (a + 1e-3) ** T_GAMMA
        l_pde = jnp.sum(mc * wt * r) / jnp.sum(mc * wt)

        # segment 0 only (w = 0): the true initial condition. For every
        # later segment the term is multiplied by (1 - w) = 0, so the branch
        # buys back the 3.7%/step it was costing (b07_diag1).
        def _ic():
            iid = random.choice(ki, n_ic, (N_IC,))
            P = jnp.concatenate([jnp.full((N_IC, 1), lo), xy_ic[iid]], 1)
            fic = model.apply(p, P, lo, hi)
            return jnp.mean((fic[:, 0] - ic_uv[iid, 0]) ** 2
                            + (fic[:, 1] - ic_uv[iid, 1]) ** 2)

        l_ic = jax.lax.cond(w < 0.5, _ic, lambda: jnp.float32(0.0))

        fo = model.apply(p, _sample_bc(kb, lo, hi), lo, hi)
        l_out = jnp.mean(fo[:, 2] ** 2)

        return l_pde + (1.0 - w) * W_IC * l_ic + W_OUT * l_out

    sched = optax.exponential_decay(LR0, LR_EVERY, LR_DECAY, end_value=LR_END)
    optimizer = soap(learning_rate=sched, precondition_frequency=200)
    grad_fn = jax.value_and_grad(loss_fn)

    @jit
    def chunk(p, os_, key, lo, hi, w, pool, ic_uv):
        def body(carry, _):
            p, os_, k = carry
            k, sub = random.split(k)
            loss, g = grad_fn(p, sub, lo, hi, w, pool, ic_uv)
            upd, os_ = optimizer.update(g, os_)
            return (optax.apply_updates(p, upd), os_, k), loss
        (p, os_, key), losses = jax.lax.scan(
            body, (p, os_, key), None, length=N_INNER)
        return p, os_, key, losses[-1]

    return params, optimizer, chunk, build_pool, uv_ic0, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def _trial(apply_fn, p, pp, X, lo, hi, lop, hip, w):
    """u = g(t) + a(t) h(t) + b(t) h_prev(t_lo), with a(lo) = 0, b(lo) = 1 when
    w = 1: the field is then EXACTLY the previous segment's at the seam, and
    a = tanh((t-lo)/(TAU_FRAC*L)) frees the network again within a fraction
    TAU_FRAC of the segment (kb1 036's ADF idea applied to the time axis).
    w = 0 (segment 0) falls back to the parent's plain ADF trial."""
    base = apply_fn(p, X, lo, hi)
    g = _lift_u(X)
    Xk = X.at[:, 0].set(lo)
    prev = apply_fn(pp, Xk, lop, hip)
    gp = _lift_u(Xk)
    s = jnp.tanh((X[:, 0] - lo) / (TAU_FRAC * (hi - lo)))
    a = w * s + (1.0 - w)
    bb = w * (1.0 - s)
    u = g + a * (base[:, 0] - g) + bb * (prev[:, 0] - gp)
    v = a * base[:, 1] + bb * prev[:, 1]
    return jnp.stack([u, v, base[:, 2]], 1)


def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u","v","p"} each [BS].
    ``params`` is the list of per-segment parameter pytrees; each point is
    routed to the segment whose time window contains it."""
    ap = lambda q, Z, a_, b_: _PREDICT_MODEL.apply(q, Z, a_, b_)
    t = X[:, 0]
    out = _trial(ap, params[0], params[0], X, SEG[0], SEG[1],
                 SEG[0], SEG[1], 0.0)
    for k in range(1, N_SEG):
        o = _trial(ap, params[k], params[k - 1], X, SEG[k], SEG[k + 1],
                   SEG[k - 1], SEG[k], 1.0)
        sel = (t >= SEG[k]) if k == N_SEG - 1 else ((t >= SEG[k]) & (t < SEG[k + 1]))
        out = jnp.where(sel[:, None], o, out)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, optimizer, chunk, build_pool, uv_ic, key = _make_train(rng)
    opt_state = optimizer.init(params)

    # warmup / compile both kernels on segment 0's window
    key, kp = random.split(key)
    pool = build_pool(kp, params, SEG[0], SEG[0], SEG[1])
    params, opt_state, key, loss = chunk(params, opt_state, key,
                                         SEG[0], SEG[1], 0.0, pool, uv_ic)
    step_count = N_INNER
    last_loss = float(loss)

    t_now = time.time()
    total_left = _T0 + TRAIN_TIME - SAVE_MARGIN_S - t_now
    per = total_left / N_SEG
    seg_params = []
    dt = 1.0
    for k in range(N_SEG):
        lo, hi = SEG[k], SEG[k + 1]
        w = 0.0 if k == 0 else 1.0
        pp = params if k == 0 else seg_params[-1]
        lop, hip = (SEG[0], SEG[1]) if k == 0 else (SEG[k - 1], SEG[k])
        if k > 0:
            opt_state = optimizer.init(params)
        seg_deadline = t_now + (k + 1) * per
        j = 0
        while step_count < MAX_ITER and time.time() + 1.6 * dt < seg_deadline:
            t0 = time.time()
            if j % REFRESH == 0:
                key, kp = random.split(key)
                pool = build_pool(kp, pp, lo, lop, hip)
            params, opt_state, key, loss = chunk(params, opt_state, key,
                                                 lo, hi, w, pool, uv_ic)
            last_loss = float(loss)
            dt = time.time() - t0
            step_count += N_INNER
            j += 1
        seg_params.append(jax.tree_util.tree_map(lambda a: a, params))
        if eval_callback is not None:
            eval_callback(seg_params, step_count, last_loss)

    return seg_params, step_count
