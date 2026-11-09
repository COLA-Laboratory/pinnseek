"""Global solve to t=4.2, then SOAP-accurate 0.3 s marching through the wake.

Parent: blocks/b02/b02_local.py. Free scoring probes put 83% of the error in
t > 4.5, so one head owns the smooth early window and six short frozen-past
slabs, each on its own clock, own the shedding window.
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
import soap_jax
from flax import linen as nn

# ── architecture / training knobs ───────────────────────────────
WIDTH        = 128
DEPTH        = 5
L_IN         = 0.06
T_RAMP       = 0.5
TAUS         = (0.0, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6.0)   # slab edges
N_SLAB       = len(TAUS) - 1
WALL_W       = (4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)        # wall time per slab
DELTA        = 0.05     # hand-over width of the C2 blend
N_UNI        = 1536
N_CYL        = 768
N_WAKE       = 768
W_CONT       = 1.0
W_FLUX       = 5.0
N_FLUX_TX    = 96
N_FLUX_Y     = 64
W_CYL        = 1.0
W_WAKE       = 1.0
LR_MAX       = 2e-3
LR_END       = 1e-4
WARMUP_FRAC  = 0.02
CLIP_NORM    = 10.0
SYNC_EVERY   = 50
PRECOND_EVERY = 20
SAVE_MARGIN_S = 8.0
# wake-mode kick (kb2 b00: street period ~0.4 s); V0_MAX = 0 disables it
OMEGA        = 2.0 * 3.14159265358979 / 0.40
V0_MAX       = 0.0
KICK_FRAC    = 0.5      # kick annealed to zero by this point inside a slab


def inlet_centreline(t):
    return U_PEAK * jnp.sin(jnp.pi * (t + T_START) / T_DRIVE)


# ══════════ Hard-constraint ansatz ══════════

def _wall(y):
    return 4.0 * y * (Y_MAX - y) / (Y_MAX ** 2)


def _cyl(x, y):
    s = ((x - CYL_X) ** 2 + (y - CYL_Y) ** 2) / (CYL_R ** 2) - 1.0
    return -jnp.expm1(-s)


def _bump(x, y):
    s = ((x - CYL_X) ** 2 + (y - CYL_Y) ** 2) / (CYL_R ** 2) - 1.0
    return jnp.exp(-s)


def _din(x):
    return jnp.tanh(x / L_IN)


def _ramp(t):
    return jnp.tanh(t / T_RAMP)


def _sig(t, tau):
    q = jnp.clip((t - tau) / DELTA, 0.0, 1.0)
    return q * q * q * (10.0 - 15.0 * q + 6.0 * q * q)


K_T = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0)
K_X = (1.0, 2.0, 4.0, 8.0, 16.0, 24.0)
K_Y = (1.0, 2.0, 4.0, 6.0)


def _feat(X, lo, span):
    """Per-head clock: t is normalised on that head's own slab."""
    th = 2.0 * X[..., 0] / T_MAX - 1.0
    tl = 2.0 * (X[..., 0] - lo) / span - 1.0
    xh = 2.0 * X[..., 1] / X_MAX - 1.0
    yh = 2.0 * X[..., 2] / Y_MAX - 1.0
    f = [th, tl, xh, yh]
    for z, ks in ((tl, K_T), (xh, K_X), (yh, K_Y)):
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


def _wrap(raw, X, v0):
    t, x, y = X[..., 0], X[..., 1], X[..., 2]
    w, c = _wall(y), _cyl(x, y)
    phi = w * c * _din(x) * _ramp(t)
    base = inlet_centreline(t) * w * c
    return jnp.stack([base + phi * raw[..., 0],
                      _bump(x, y) * v0 * jnp.sin(OMEGA * t) + phi * raw[..., 1],
                      (1.0 - x / X_MAX) * raw[..., 2]], axis=-1)


def _uvp_stage(prev, cur, tau, pl, ps, cl, cs, X, v0=0.0):
    s = _sig(X[..., 0], tau)[..., None]
    raw = ((1.0 - s) * _MODEL.apply(prev, _feat(X, pl, ps))
           + s * _MODEL.apply(cur, _feat(X, cl, cs)))
    return _wrap(raw, X, v0)


def _head(params, k):
    return jax.tree_util.tree_map(lambda A: A[k], params)


def _span(k):
    return TAUS[k] - DELTA, (TAUS[k + 1] + DELTA) - (TAUS[k] - DELTA)


def _uvp(params, X):
    Xn = np.asarray(X, dtype=np.float32)
    idx = np.clip(np.searchsorted(np.asarray(TAUS[1:]), Xn[:, 0], side="right"),
                  0, N_SLAB - 1)
    out = np.zeros((Xn.shape[0], 3), np.float32)
    for k in range(N_SLAB):
        m = idx == k
        if not m.any():
            continue
        pl, ps = _span(max(k - 1, 0))
        cl, cs = _span(k)
        tau = -2.0 * DELTA if k == 0 else TAUS[k]
        out[m] = np.asarray(_uvp_stage(_head(params, max(k - 1, 0)),
                                       _head(params, k), tau, pl, ps, cl, cs,
                                       jnp.asarray(Xn[m])))
    return out


def predict_fn(params, X):
    out = _uvp(params, X)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


# ══════════ PDE residual by batched forward-mode autodiff ══════════

def _residuals_stage(prev, cur, tau, pl, ps, cl, cs, X, v0=0.0):
    f = lambda Z: _uvp_stage(prev, cur, tau, pl, ps, cl, cs, Z, v0)
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

    u, v = F[:, 0], F[:, 1]
    u_t, v_t = Ft[:, 0], Ft[:, 1]
    u_x, v_x, p_x = Fx[:, 0], Fx[:, 1], Fx[:, 2]
    u_y, v_y, p_y = Fy[:, 0], Fy[:, 1], Fy[:, 2]
    u_xx, v_xx = Fxx[:, 0], Fxx[:, 1]
    u_yy, v_yy = Fyy[:, 0], Fyy[:, 1]

    r_c = u_x + v_y
    r_mx = u_t + u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    r_my = v_t + u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
    return r_c, r_mx, r_my


def _pt_loss(prev, cur, g, X, v0):
    r_c, r_mx, r_my = _residuals_stage(prev, cur, *g, X, v0)
    return W_CONT * r_c ** 2 + r_mx ** 2 + r_my ** 2


def _outside(X):
    return ((X[:, 1] - CYL_X) ** 2 + (X[:, 2] - CYL_Y) ** 2
            > CYL_R ** 2).astype(X.dtype)


# ══════════ Collocation sampling inside the active slab ══════════

def _sample(key, t_lo, t_hi):
    k = random.split(key, 11)
    tu = random.uniform(k[0], (N_UNI,), minval=t_lo, maxval=t_hi)
    xu = random.uniform(k[1], (N_UNI,), minval=X_MIN, maxval=X_MAX)
    yu = random.uniform(k[2], (N_UNI,), minval=Y_MIN, maxval=Y_MAX)
    Xu = jnp.stack([tu, xu, yu], -1)

    tc = random.uniform(k[3], (N_CYL,), minval=t_lo, maxval=t_hi)
    th = random.uniform(k[4], (N_CYL,), minval=0.0, maxval=2.0 * jnp.pi)
    rr = CYL_R * jnp.exp(random.uniform(k[5], (N_CYL,)) * jnp.log(4.0))
    Xc = jnp.stack([tc, CYL_X + rr * jnp.cos(th),
                    CYL_Y + rr * jnp.sin(th)], -1)

    tw = random.uniform(k[6], (N_WAKE,), minval=t_lo, maxval=t_hi)
    xw = random.uniform(k[7], (N_WAKE,), minval=0.15, maxval=1.8)
    yw = random.uniform(k[8], (N_WAKE,), minval=0.05, maxval=0.36)
    Xw = jnp.stack([tw, xw, yw], -1)

    tq = random.uniform(k[9], (N_FLUX_TX,), minval=t_lo, maxval=t_hi)
    xq = random.uniform(k[10], (N_FLUX_TX,), minval=X_MIN, maxval=X_MAX)
    return Xu, Xc, Xw, tq, xq


_YQ = jnp.linspace(Y_MIN, Y_MAX, N_FLUX_Y)
_WQ = jnp.full((N_FLUX_Y,), (Y_MAX - Y_MIN) / (N_FLUX_Y - 1)).at[0].mul(0.5).at[-1].mul(0.5)
_QREF = U_PEAK * (2.0 / 3.0) * Y_MAX


def _flux_loss(prev, cur, g, tq, xq, v0):
    T = jnp.repeat(tq, N_FLUX_Y)
    Xc_ = jnp.repeat(xq, N_FLUX_Y)
    Yc_ = jnp.tile(_YQ, tq.shape[0])
    P = jnp.stack([T, Xc_, Yc_], -1)
    u = _uvp_stage(prev, cur, *g, P, v0)[:, 0]
    solid = ((Xc_ - CYL_X) ** 2 + (Yc_ - CYL_Y) ** 2 <= CYL_R ** 2)
    u = jnp.where(solid, 0.0, u).reshape(tq.shape[0], N_FLUX_Y)
    Q = jnp.sum(u * _WQ[None, :], axis=1)
    Q0 = inlet_centreline(tq) * (2.0 / 3.0) * Y_MAX
    return jnp.mean(((Q - Q0) / _QREF) ** 2)


def _loss_fn(cur, prev, g, v0, Xu, Xc, Xw, tq, xq):
    mu, mw = _outside(Xu), _outside(Xw)
    lu = jnp.sum(_pt_loss(prev, cur, g, Xu, v0) * mu) / (jnp.sum(mu) + 1e-8)
    lc = jnp.mean(_pt_loss(prev, cur, g, Xc, v0))
    lw = jnp.sum(_pt_loss(prev, cur, g, Xw, v0) * mw) / (jnp.sum(mw) + 1e-8)
    return (lu + W_CYL * lc + W_WAKE * lw
            + W_FLUX * _flux_loss(prev, cur, g, tq, xq, v0))


# ══════════ Training: march the slabs ══════════

def _lr_at(frac):
    warm = min(1.0, frac / WARMUP_FRAC) if WARMUP_FRAC > 0 else 1.0
    return LR_MAX * warm * (LR_END / LR_MAX) ** min(1.0, max(0.0, frac))


def _edges():
    w = np.asarray(WALL_W, dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(w) / w.sum()])


def _geom(k):
    pl, ps = _span(max(k - 1, 0))
    cl, cs = _span(k)
    tau = -2.0 * DELTA if k == 0 else TAUS[k]
    return tuple(jnp.float32(z) for z in (tau, pl, ps, cl, cs))


def _build(rng):
    key, rng = random.split(rng)
    theta = _MODEL.init(key, _feat(jnp.zeros([1, INPUT_DIM]), 0.0, 1.0))
    tx = optax.chain(optax.clip_by_global_norm(CLIP_NORM),
                     soap_jax.scale_by_soap(precondition_frequency=PRECOND_EVERY))
    opt_state = tx.init(theta)

    @jit
    def update(cur, prev, opt_state, key, lr, g, v0, t_lo, t_hi):
        loss, grads = jax.value_and_grad(_loss_fn)(
            cur, prev, g, v0, *_sample(key, t_lo, t_hi))
        upd, opt_state = tx.update(grads, opt_state, cur)
        upd = jax.tree_util.tree_map(lambda q: -lr * q, upd)
        return optax.apply_updates(cur, upd), opt_state, loss

    return theta, opt_state, update, rng


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    theta, opt_state, update, rng = _build(rng)
    prev = theta
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    edges = _edges()

    g = _geom(0)
    rng, k = random.split(rng)
    theta, opt_state, loss = update(theta, prev, opt_state, k, jnp.float32(0.0),
                                    g, jnp.float32(0.0),
                                    jnp.float32(TAUS[0]), jnp.float32(TAUS[1]))
    jax.block_until_ready(loss)
    t_loop = time.time()
    span = max(deadline - t_loop, 1.0)

    heads = [theta] * N_SLAB
    slab, step_count = 0, 1
    t_lo, t_hi = jnp.float32(TAUS[0]), jnp.float32(TAUS[1])
    lr, v0 = jnp.float32(_lr_at(0.0)), jnp.float32(0.0)
    for it in range(1, MAX_ITER):
        rng, k = random.split(rng)
        theta, opt_state, loss = update(theta, prev, opt_state, k, lr, g, v0,
                                        t_lo, t_hi)
        step_count = it + 1
        if it % SYNC_EVERY == 0:
            now = time.time()
            if now > deadline:
                break
            frac = (now - t_loop) / span
            while slab < N_SLAB - 1 and frac >= edges[slab + 1]:
                heads[slab] = theta
                prev = theta
                slab += 1
                g = _geom(slab)
            lo, hi = edges[slab], edges[slab + 1]
            sf = (frac - lo) / max(hi - lo, 1e-6)
            lr = jnp.float32(_lr_at(sf))
            v0 = jnp.float32(0.0 if slab == 0 else
                             V0_MAX * max(0.0, 1.0 - sf / KICK_FRAC))
            t_lo, t_hi = jnp.float32(TAUS[slab]), jnp.float32(TAUS[slab + 1])
            if eval_callback is not None and it % 1000 == 0:
                eval_callback(theta, it, float(loss))
    for j in range(slab, N_SLAB):
        heads[j] = theta
    return jax.tree_util.tree_map(lambda *a: jnp.stack(a), *heads), step_count
