"""Parent: b05_hmode.py.  One change: the free CP rank 32 -> 8.  The IC
modes now carry the dominant field, so the free terms are only a
correction; b04 (d) never re-examined rank after the ansatz went linear.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

import math as _math

# --- I/O shape ---
INPUT_DIM    = 4
OUTPUT_DIM   = 3          # the network emits velocity only
INPUT_NAMES  = ("t", "x", "y", "z")
FIELD_NAMES  = ("u", "v", "w", "wx", "wy", "wz")   # 3 emitted + 3 derived:
                          # (wx, wy, wz) = curl(u, v, w), by autodiff

# --- Domain ---
T_MIN, T_MAX = 0.0, 5.0
X_MIN, X_MAX = 0.0, 2.0 * _math.pi
Y_MIN, Y_MAX = 0.0, 2.0 * _math.pi
Z_MIN, Z_MAX = 0.0, 2.0 * _math.pi
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 3   # lap(curl u) is third order in the velocity
N_RESIDUAL_COMPONENTS   = 4   # three vorticity-transport components + continuity
HAS_SOURCE              = True   # the manufactured forcing f(t, x, y, z)

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
PERIODIC_AXES  = ("x", "y", "z")   # period 2*pi on each
GEOMETRY_KIND  = "periodic_box"

# --- Physical parameters ---
NU = 0.05      # viscosity
# Forcing (manufactured):
#   f_x = -FA e^{-18 NU t} sin(4y) sin(2z)
#   f_y = -FA e^{-18 NU t} sin(4x) sin(2z)
#   f_z = +FA e^{-18 NU t} sin(4x) sin(4y)
FA = 6.0

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
from jax import jit, jvp, random
import numpy as np
import optax
import soap_jax
from flax import linen as nn

_SMOKE = TRAIN_TIME < 100.0      # CPU dress rehearsal: shrink the lattice


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Hyperparameters ══════════

N_FREQ    = 4          # spatial input features cos(k s), sin(k s), k = 1..N_FREQ
BAND_INIT = 0.35       # init std of the linear trig head (matches the scale
                       # the tanh axis nets used to emit)
T_DEG     = 12         # Chebyshev degree of the (now linear) time head
TIME_INIT = 0.02       # init std of the FREE CP terms — small: the model starts
                       # at u = u_IC * (1 + a(t)), a near-correct field
RANK      = 8          # CP rank r of the free part (was 32): with the IC modes
                       # carrying the dominant field, the free terms are a
                       # correction — b04 (d) says capacity here only hurts
NT        = 24 if not _SMOKE else 6    # (unused: see N_EARLY / N_LATE)
NS        = 24 if not _SMOKE else 6    # (unused: see N_EARLY / N_LATE)
N_EARLY   = 16 if not _SMOKE else 6    # (unused)
N_LATE    = 32 if not _SMOKE else 6    # (unused)
RAMP_FRAC = 0.60       # (unused)
NS_EX     = 20 if not _SMOKE else 5    # fixed spatial grid: exact for the
                       # degree-16 squared residual (needs >= 4*N_FREQ + 1)
NT_EX     = 32 if not _SMOKE else 6    # Gauss-Legendre nodes in t: exact
                       # for the degree-4*T_DEG squared residual (needs >= 2*T_DEG+1)
LR        = 2.5e-4      # peak; warmup then cosine on the WALL CLOCK
R_IC      = 1          # CP rank of the FIXED initial-condition term
MODE_INIT = 0.05       # init std of the IC modes' time factor a(t)
WARMUP_FRAC   = 0.05   # fraction of the wall spent warming up
LR_FLOOR      = 0.001  # end LR as a fraction of the peak
SAVE_MARGIN_S = 6.0
PRED_CHUNK    = 10000


# ══════════ The fixed initial-condition factor (from task/ns4d_ic.csv) ══════════

def _ic_cp():
    """Rank-R_IC separable trig representation of u(0, x, y, z).

    The IC field is band-limited, so fit it in fp64 on {1, cos ks, sin ks}
    (k = 1..N_FREQ) per axis — this is b04's re-interpolation, but now the
    representation *is* the model's t = 0 slice, so there is no node mismatch
    left at all.  A greedy rank-1 deflation of the resulting 9x9x9 coefficient
    tensor gives per-axis kernels [9, 3*R_IC].  Uses only task/ns4d_ic.csv."""
    ic = np.loadtxt(_locate("ns4d_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    n = int(round(len(ic) ** (1.0 / 3.0)))
    g = np.unique(ic[:, 0])
    assert len(g) == n and len(ic) == n ** 3
    u = np.transpose(ic[:, 3:6].reshape(n, n, n, 3), (3, 0, 1, 2))     # [3,n,n,n]
    kk = np.arange(1, N_FREQ + 1)
    B = np.concatenate([np.ones((n, 1)), np.cos(g[:, None] * kk),
                        np.sin(g[:, None] * kk)], axis=1)              # [n, 9]
    Pi = np.linalg.pinv(B)
    C = np.einsum("ia,jb,kc,fabc->fijk", Pi, Pi, Pi, u)                # [3,9,9,9]
    m = B.shape[1]
    Kx = np.zeros((m, 3, R_IC)); Ky = np.zeros((m, 3, R_IC)); Kz = np.zeros((m, 3, R_IC))
    for f in range(3):
        Rm = C[f].copy()
        for j in range(R_IC):
            U, S, Vt = np.linalg.svd(Rm.reshape(m, m * m), full_matrices=False)
            a, s0, rest = U[:, 0], S[0], Vt[0].reshape(m, m)
            P, S2, Qt = np.linalg.svd(rest, full_matrices=False)
            b, c = P[:, 0], Qt[0]
            sc = (s0 * S2[0]) ** (1.0 / 3.0)
            sg = np.sign(s0 * S2[0]) if s0 * S2[0] != 0 else 1.0
            Kx[:, f, j] = a * sc * sg; Ky[:, f, j] = b * sc; Kz[:, f, j] = c * sc
            Rm = Rm - np.einsum("i,j,k->ijk", Kx[:, f, j], Ky[:, f, j], Kz[:, f, j])
    rel = np.linalg.norm(C - np.einsum("ifj,kfj,lfj->fikl", Kx, Ky, Kz)) / np.linalg.norm(C)
    return (Kx.reshape(m, 3 * R_IC), Ky.reshape(m, 3 * R_IC),
            Kz.reshape(m, 3 * R_IC), float(rel))


_KX, _KY, _KZ, _IC_REL = _ic_cp()
# kept as float64 numpy and cast to the caller's dtype: fp32 in training,
# fp64 in predict_fn, so the frozen IC factor never costs precision either way
_KFIX = {"x": _KX, "y": _KY, "z": _KZ}


# ══════════ Per-axis network ══════════

def _cheb_features(tau, deg):
    """[N, deg+1] Chebyshev T_0..T_deg by the three-term recurrence — a
    polynomial in tau, so autodiff of it is exact and cheap."""
    cols = [jnp.ones_like(tau), tau]
    for _ in range(deg - 1):
        cols.append(2.0 * tau * cols[-1] - cols[-2])
    return jnp.stack(cols[:deg + 1], axis=1)


class AxisNet(nn.Module):
    """R -> R^{3 x (R_IC + r)}: one axis map per input axis, LINEAR on a fixed
    basis, with the first R_IC CP slots FROZEN to carry the initial condition.

    Spatial axes: slots [0, R_IC) are the fixed IC factors (a constant kernel
    on {1, cos ks, sin ks}); the remaining r slots are a trainable Dense on
    cos(k s), sin(k s), k = 1..N_FREQ — a trig polynomial of degree N_FREQ,
    exactly 2*pi-periodic (with all its autodiff derivatives) and exactly
    band-limited.  Time axis: slots [0, R_IC) are the constant 1, and the
    trainable slots are a bias-free Dense on T_p(tau) - T_p(-1), p = 1..T_DEG
    — a polynomial of degree T_DEG that VANISHES at t = T_MIN.  Hence
    u(0, .) is the fixed IC term exactly, whatever the parameters are, and
    the residual is still a polynomial in t (Gauss-Legendre stays exact)."""
    axis: str
    rank: int = RANK

    @nn.compact
    def __call__(self, s):                       # s: [N]
        n = s.shape[0]
        if self.axis != "t":
            k = jnp.arange(1, N_FREQ + 1, dtype=s.dtype)
            ang = s[:, None] * k[None, :]
            cs = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=1)
            fix = (jnp.concatenate([jnp.ones((n, 1), s.dtype), cs], axis=1)
                   @ jnp.asarray(_KFIX[self.axis], s.dtype)
                   ).reshape(n, OUTPUT_DIM, R_IC)
            h = nn.Dense(OUTPUT_DIM * self.rank,
                         kernel_init=nn.initializers.normal(BAND_INIT))(cs)
            return jnp.concatenate([fix, h.reshape(n, OUTPUT_DIM, self.rank)], axis=2)
        tau = 2.0 * (s - T_MIN) / (T_MAX - T_MIN) - 1.0
        ch = _cheb_features(tau, T_DEG)                        # [N, T_DEG+1]
        at_lo = jnp.array([(-1.0) ** p for p in range(T_DEG + 1)], s.dtype)
        h = (ch - at_lo[None, :])[:, 1:]                       # vanishes at t = T_MIN
        a = nn.Dense(OUTPUT_DIM * R_IC, use_bias=False,
                     kernel_init=nn.initializers.normal(MODE_INIT))(h)
        a = 1.0 + a.reshape(n, OUTPUT_DIM, R_IC)   # = 1 at t = T_MIN -> hard IC
        h = nn.Dense(OUTPUT_DIM * self.rank, use_bias=False,
                     kernel_init=nn.initializers.normal(TIME_INIT))(h)
        return jnp.concatenate([a, h.reshape(n, OUTPUT_DIM, self.rank)], axis=2)


_NET_T = AxisNet(axis="t")
_NET = {a: AxisNet(axis=a) for a in ("x", "y", "z")}


def _axis_stack(net, p, s, order):
    """[order+1, N, 3, r]: the axis features and their first `order`
    derivatives, by nested forward-mode AD (one JVP per order)."""
    f = lambda ss: net.apply(p, ss)
    ones = jnp.ones_like(s)
    outs = [f(s)]
    for _ in range(order):
        f = (lambda prev: (lambda ss: jvp(prev, (ss,), (ones,))[1]))(f)
        outs.append(f(s))
    return jnp.stack(outs)


# ══════════ Which mixed partials the residual needs ══════════
# (order_t, order_x, order_y, order_z) — 22 tensors per velocity component.
IDX = (
    (0, 0, 0, 0),                                            # 0  u
    (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),                # 1  2  3  first
    (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1),                # 4  5  6  d/dt first
    (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2),                # 7  8  9  pure second
    (0, 1, 1, 0), (0, 1, 0, 1), (0, 0, 1, 1),                # 10 11 12 mixed second
    (0, 3, 0, 0), (0, 2, 1, 0), (0, 2, 0, 1),                # 13 14 15 third
    (0, 1, 2, 0), (0, 0, 3, 0), (0, 0, 2, 1),                # 16 17 18
    (0, 1, 0, 2), (0, 0, 1, 2), (0, 0, 0, 3),                # 19 20 21
)
IDX_IC = ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))

_OT = np.array([i[0] for i in IDX]);     _OX = np.array([i[1] for i in IDX])
_OY = np.array([i[2] for i in IDX]);     _OZ = np.array([i[3] for i in IDX])
_IT = np.array([i[0] for i in IDX_IC]);  _IX = np.array([i[1] for i in IDX_IC])
_IY = np.array([i[2] for i in IDX_IC]);  _IZ = np.array([i[3] for i in IDX_IC])


def _contract(Dt, Dx, Dy, Dz, ot, ox, oy, oz):
    """[K, 3, Nt, Nx, Ny, Nz]: the K requested mixed partials of the rank-r
    separable field, as outer products of the per-axis derivative features."""
    Tt, Tx = Dt[ot], Dx[ox]                              # [K, N, 3, r]
    Ty, Tz = Dy[oy], Dz[oz]
    A = jnp.einsum("kacj,kbcj->kcabj", Tt, Tx)           # [K, 3, Nt, Nx, r]
    B = jnp.einsum("kdcj,kecj->kcdej", Ty, Tz)           # [K, 3, Ny, Nz, r]
    return jnp.einsum("kcabj,kcdej->kcabde", A, B,
                      precision=jax.lax.Precision.HIGHEST)


def _forcing(ts, xs, ys, zs):
    """Manufactured body force on the lattice — separable, so broadcast."""
    e = jnp.exp(-18.0 * NU * ts)[:, None, None, None]
    s4x = jnp.sin(4.0 * xs)[None, :, None, None]
    s4y = jnp.sin(4.0 * ys)[None, None, :, None]
    s2z = jnp.sin(2.0 * zs)[None, None, None, :]
    return (-FA * e * s4y * s2z, -FA * e * s4x * s2z, FA * e * s4x * s4y)


def _pde_terms(D, ts, xs, ys, zs):
    """Vorticity-transport residual (3 components) + continuity, from the
    22 mixed partials in D (D[i, c] is partial IDX[i] of velocity comp c)."""
    u, v, w = D[0, 0], D[0, 1], D[0, 2]

    wx = D[2, 2] - D[3, 1]        # du_z/dy - du_y/dz
    wy = D[3, 0] - D[1, 2]        # du_x/dz - du_z/dx
    wz = D[1, 1] - D[2, 0]        # du_y/dx - du_x/dy

    wx_t = D[5, 2] - D[6, 1]
    wy_t = D[6, 0] - D[4, 2]
    wz_t = D[4, 1] - D[5, 0]

    wx_x, wx_y, wx_z = D[10, 2] - D[11, 1], D[8, 2] - D[12, 1], D[12, 2] - D[9, 1]
    wy_x, wy_y, wy_z = D[11, 0] - D[7, 2], D[12, 0] - D[10, 2], D[9, 0] - D[11, 2]
    wz_x, wz_y, wz_z = D[7, 1] - D[10, 0], D[10, 1] - D[8, 0], D[11, 1] - D[12, 0]

    lap_wx = (D[14, 2] + D[17, 2] + D[20, 2]) - (D[15, 1] + D[18, 1] + D[21, 1])
    lap_wy = (D[15, 0] + D[18, 0] + D[21, 0]) - (D[13, 2] + D[16, 2] + D[19, 2])
    lap_wz = (D[13, 1] + D[16, 1] + D[19, 1]) - (D[14, 0] + D[17, 0] + D[20, 0])

    adv_x = u * wx_x + v * wx_y + w * wx_z
    adv_y = u * wy_x + v * wy_y + w * wy_z
    adv_z = u * wz_x + v * wz_y + w * wz_z

    st_x = wx * D[1, 0] + wy * D[2, 0] + wz * D[3, 0]
    st_y = wx * D[1, 1] + wy * D[2, 1] + wz * D[3, 1]
    st_z = wx * D[1, 2] + wy * D[2, 2] + wz * D[3, 2]

    fx, fy, fz = _forcing(ts, xs, ys, zs)
    rx = wx_t + adv_x - st_x - NU * lap_wx - fx
    ry = wy_t + adv_y - st_y - NU * lap_wy - fy
    rz = wz_t + adv_z - st_z - NU * lap_wz - fz
    div = D[1, 0] + D[2, 1] + D[3, 2]
    return rx, ry, rz, div


# ══════════ Loss ══════════

def gauss_legendre_t(n, dtype=np.float32):
    """n-node Gauss-Legendre rule on [T_MIN, T_MAX], weights normalised to
    sum to 1 so a weighted sum is the mean over the interval.  With the
    ansatz polynomial in t the squared residual is a polynomial too, so this
    quadrature is EXACT for n >= (4*T_DEG + 2)/2 = 2*T_DEG + 1."""
    x, w = np.polynomial.legendre.leggauss(n)
    ts = 0.5 * (T_MAX - T_MIN) * (x + 1.0) + T_MIN
    return jnp.array(ts, dtype), jnp.array(w / w.sum(), dtype)


def res_fn(params, ts, tw, xs, ys, zs):
    """Weighted residual VECTOR r(theta) — PDE only, the IC being exact by
    construction.  The objective is sum(r**2), so the same function serves
    SOAP and any Gauss-Newton / natural-gradient method (kb1-063, kb1-108),
    which need J = dr/dtheta."""
    Dt = _axis_stack(_NET_T, params["t"], ts, 1)
    Dx = _axis_stack(_NET["x"], params["x"], xs, 3)
    Dy = _axis_stack(_NET["y"], params["y"], ys, 3)
    Dz = _axis_stack(_NET["z"], params["z"], zs, 3)
    D = _contract(Dt, Dx, Dy, Dz, _OT, _OX, _OY, _OZ)
    rx, ry, rz, div = _pde_terms(D, ts, xs, ys, zs)
    # quadrature weight per lattice point: GL in t, equispaced (exact) in space
    n_sp = xs.shape[0] * ys.shape[0] * zs.shape[0]
    sw = jnp.sqrt(tw / n_sp)[:, None, None, None]
    return jnp.concatenate([(sw * rx).ravel(), (sw * ry).ravel(),
                            (sw * rz).ravel(), (sw * div).ravel()])


def loss_fn(params, ts, tw, xs, ys, zs):
    pde = jnp.sum(res_fn(params, ts, tw, xs, ys, zs) ** 2)
    return pde, (pde, 0.0)


# ══════════ Public contract ══════════

def _axis_val_and_d(net, p, s):
    """Axis features and their first derivative at scattered coordinates."""
    f = lambda ss: net.apply(p, ss)
    return f(s), jvp(f, (s,), (jnp.ones_like(s),))[1]


@jax.jit
def _predict_chunk(params, xb):
    """Traced at whatever dtype it is fed: predict_fn feeds float64."""
    Ft = _NET_T.apply(params["t"], xb[:, 0])
    Fx, Gx = _axis_val_and_d(_NET["x"], params["x"], xb[:, 1])
    Fy, Gy = _axis_val_and_d(_NET["y"], params["y"], xb[:, 2])
    Fz, Gz = _axis_val_and_d(_NET["z"], params["z"], xb[:, 3])
    U = jnp.sum(Ft * Fx * Fy * Fz, axis=-1)          # [M, 3]
    Ux = jnp.sum(Ft * Gx * Fy * Fz, axis=-1)
    Uy = jnp.sum(Ft * Fx * Gy * Fz, axis=-1)
    Uz = jnp.sum(Ft * Fx * Fy * Gz, axis=-1)
    W = jnp.stack([Uy[:, 2] - Uz[:, 1],
                   Uz[:, 0] - Ux[:, 2],
                   Ux[:, 1] - Uy[:, 0]], axis=1)     # curl u, by autodiff
    return U, W


def predict_fn(params, X):
    """X: jnp.ndarray [BS, 4] with columns (t, x, y, z).  Returns the
    separable velocity field and its autodiff curl.

    Evaluated in float64.  The field is a sum of RANK CP terms; accumulating
    that sum (and the derivative sums the curl needs) in float32 costs ~1e-7
    relative — the same order as the whole solution error at this accuracy.
    x64 is switched on lazily here so training (a separate process under
    eval.py) is untouched and stays fp32."""
    jax.config.update("jax_enable_x64", True)
    params = jax.tree_util.tree_map(
        lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64), params)
    X = jnp.asarray(np.asarray(X, dtype=np.float64))
    us, ws = [], []
    for i in range(0, X.shape[0], PRED_CHUNK):
        U, W = _predict_chunk(params, X[i:i + PRED_CHUNK])
        us.append(U)
        ws.append(W)
    U = jnp.concatenate(us, axis=0)
    W = jnp.concatenate(ws, axis=0)
    return {"u": U[:, 0], "v": U[:, 1], "w": U[:, 2],
            "wx": W[:, 0], "wy": W[:, 1], "wz": W[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline SOAP on a FULLY DETERMINISTIC objective: exact
    equispaced quadrature in space, exact Gauss-Legendre in t."""
    k0, k1, k2, k3, rng_local = random.split(rng, 5)
    d0 = jnp.zeros((1,), dtype=jnp.float32)
    params = {"t": _NET_T.init(k0, d0), "x": _NET["x"].init(k1, d0),
              "y": _NET["y"].init(k2, d0), "z": _NET["z"].init(k3, d0)}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    optimizer = optax.inject_hyperparams(
        soap_jax.soap,
        static_args=("b1", "b2", "shampoo_beta", "eps", "weight_decay",
                     "correct_bias", "precondition_frequency",
                     "max_precond_dim", "precondition_1d"),
    )(learning_rate=LR, precondition_frequency=10)
    opt_state = optimizer.init(params)

    # fixed equispaced spatial grid — exact quadrature for the band-limited
    # squared residual, so no spatial sampling noise and no aliasing at all
    sgrid = jnp.array((X_MAX - X_MIN) / NS_EX * np.arange(NS_EX) + X_MIN,
                      dtype=jnp.float32)
    tgrid, tw = gauss_legendre_t(NT_EX, jnp.float32)

    @jit
    def update(params, opt_state, key, lr):
        opt_state = opt_state._replace(
            hyperparams={**opt_state.hyperparams, "learning_rate": lr})
        ts = tgrid
        (loss, aux), grads = grad_fn(params, ts, tw, sgrid, sgrid, sgrid)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    span = deadline - _T0
    ramp_at = deadline + 1.0

    def lr_at(now):
        """Warmup-then-cosine on elapsed wall time, not on step index."""
        f = min(max((now - _T0) / span, 0.0), 1.0)
        if f < WARMUP_FRAC:
            return LR * f / WARMUP_FRAC
        g = (f - WARMUP_FRAC) / (1.0 - WARMUP_FRAC)
        c = 0.5 * (1.0 + _math.cos(_math.pi * g))
        return LR * (LR_FLOOR + (1.0 - LR_FLOOR) * c)

    # Warmup step: compiles the whole training step; it is a real step.
    rng_local, k = random.split(rng_local)
    params, opt_state, loss = update(params, opt_state, k, lr_at(time.time()))
    step_count = 1
    last_loss = float(loss)

    for it in range(1, MAX_ITER):
        now = time.time()
        if now > deadline:
            break
        rng_local, k = random.split(rng_local)
        params, opt_state, loss = update(params, opt_state, k, lr_at(now))
        step_count = it + 1
        if it % 20 == 0:            # throttle the async dispatch queue so the
            last_loss = float(loss)  # wall-clock deadline stays meaningful
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(params, it, last_loss)

    return params, step_count
