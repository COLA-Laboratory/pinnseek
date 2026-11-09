"""Magnitude-normalised residual in t (kb1-021), exponent 0.5: each t
slice is weighted by its own predicted vorticity energy^-0.5, so the
exponentially decaying late-t tail — which the proxy shows owns 50-70%
of the error — stops being free.  Parent: b03_nt48.py (6.95e-7).
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
HIDDEN    = 64         # width of every axis MLP
DEPTH     = 5          # hidden layers per axis MLP
RANK      = 32         # CP rank r
NT        = 24 if not _SMOKE else 6    # (unused: see N_EARLY / N_LATE)
NS        = 24 if not _SMOKE else 6    # (unused: see N_EARLY / N_LATE)
N_EARLY   = 16 if not _SMOKE else 6    # (unused)
N_LATE    = 32 if not _SMOKE else 6    # (unused)
RAMP_FRAC = 0.60       # (unused)
NS_EX     = 20 if not _SMOKE else 5    # fixed spatial grid: exact for the
                       # degree-16 squared residual (needs >= 4*N_FREQ + 1)
NT_EX     = 48 if not _SMOKE else 5    # random t samples per step
LR        = 1e-3      # peak; warmup then cosine on the WALL CLOCK
T_POW     = 0.5      # exponent of the per-t magnitude normalisation (half)
T_WEPS    = 1e-2       # floor, as a fraction of the mean energy
W_IC      = 20.0       # weight on the (exactly known) initial condition
WARMUP_FRAC   = 0.05   # fraction of the wall spent warming up
LR_FLOOR      = 0.001  # end LR as a fraction of the peak
SAVE_MARGIN_S = 6.0
PRED_CHUNK    = 10000


# ══════════ Per-axis network ══════════

class AxisNet(nn.Module):
    """R -> R^{3 x r}: one axis map per input axis.

    Spatial axes: a single Dense on the truncated Fourier features
    cos(k s), sin(k s), k = 1..N_FREQ — i.e. the axis map is a *trig
    polynomial of degree N_FREQ*.  It is exactly 2*pi-periodic (as are all
    its derivatives, taken by autodiff of cos/sin) AND exactly band-limited,
    so no content can hide between collocation points.
    Time axis: unchanged deep tanh MLP on the normalised coordinate."""
    periodic: bool
    hidden: int = HIDDEN
    depth: int = DEPTH
    rank: int = RANK

    @nn.compact
    def __call__(self, s):                       # s: [N]
        if self.periodic:
            k = jnp.arange(1, N_FREQ + 1, dtype=s.dtype)
            ang = s[:, None] * k[None, :]
            h = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=1)
            h = nn.Dense(OUTPUT_DIM * self.rank,
                         kernel_init=nn.initializers.normal(BAND_INIT))(h)
            return h.reshape(s.shape[0], OUTPUT_DIM, self.rank)
        h = (2.0 * (s - T_MIN) / (T_MAX - T_MIN) - 1.0)[:, None]
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.hidden)(h))
        h = nn.Dense(OUTPUT_DIM * self.rank)(h)
        return h.reshape(s.shape[0], OUTPUT_DIM, self.rank)


_NET_T = AxisNet(periodic=False)
_NET_S = AxisNet(periodic=True)


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


# ══════════ Data: the initial condition (the only field data available) ══════════

def _load_ic():
    ic = np.loadtxt(_locate("ns4d_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    n = int(round(len(ic) ** (1.0 / 3.0)))
    grid = np.unique(ic[:, 0]).astype(np.float32)            # shared by x, y, z
    assert len(grid) == n and len(ic) == n ** 3
    fields = ic[:, 3:9].astype(np.float32).reshape(n, n, n, 6)   # axes: x, y, z
    uw = np.transpose(fields, (3, 0, 1, 2))                   # [6, n, n, n]
    return jnp.array(grid), jnp.array(uw)


# ══════════ Loss ══════════

def _make_loss(ic_grid, ic_uw):
    zero_t = jnp.zeros((1,), dtype=jnp.float32)

    def loss_fn(params, ts, xs, ys, zs):
        Dt = _axis_stack(_NET_T, params["t"], ts, 1)
        Dx = _axis_stack(_NET_S, params["x"], xs, 3)
        Dy = _axis_stack(_NET_S, params["y"], ys, 3)
        Dz = _axis_stack(_NET_S, params["z"], zs, 3)
        D = _contract(Dt, Dx, Dy, Dz, _OT, _OX, _OY, _OZ)
        rx, ry, rz, div = _pde_terms(D, ts, xs, ys, zs)

        # self-supervised magnitude normalisation across t (kb1-021): the
        # scored field decays by ~10x over the window, so an unweighted
        # residual leaves the late-t tail almost unconstrained.  Weight each
        # t slice by its own predicted vorticity energy (stop_gradient, so
        # this re-weights the loss, it does not create a shortcut).
        pwx = D[2, 2] - D[3, 1]
        pwy = D[3, 0] - D[1, 2]
        pwz = D[1, 1] - D[2, 0]
        mag = jnp.mean(pwx ** 2 + pwy ** 2 + pwz ** 2, axis=(1, 2, 3))   # [Nt]
        mag = jax.lax.stop_gradient(mag)
        wt = (mag + T_WEPS * jnp.mean(mag)) ** (-T_POW)
        w4 = (wt / jnp.mean(wt))[:, None, None, None]
        wm = lambda a: jnp.mean(w4 * a ** 2)
        pde = wm(rx) + wm(ry) + wm(rz) + wm(div)

        # initial condition on the supplied 32^3 grid: velocity and vorticity
        Gt = _axis_stack(_NET_T, params["t"], zero_t, 0)
        Gx = _axis_stack(_NET_S, params["x"], ic_grid, 1)
        Gy = _axis_stack(_NET_S, params["y"], ic_grid, 1)
        Gz = _axis_stack(_NET_S, params["z"], ic_grid, 1)
        G = _contract(Gt, Gx, Gy, Gz, _IT, _IX, _IY, _IZ)[:, :, 0]   # [4, 3, n,n,n]
        pw_x = G[2, 2] - G[3, 1]
        pw_y = G[3, 0] - G[1, 2]
        pw_z = G[1, 1] - G[2, 0]
        ic = (jnp.mean((G[0, 0] - ic_uw[0]) ** 2)
              + jnp.mean((G[0, 1] - ic_uw[1]) ** 2)
              + jnp.mean((G[0, 2] - ic_uw[2]) ** 2)
              + jnp.mean((pw_x - ic_uw[3]) ** 2)
              + jnp.mean((pw_y - ic_uw[4]) ** 2)
              + jnp.mean((pw_z - ic_uw[5]) ** 2))
        return pde + W_IC * ic, (pde, ic)

    return loss_fn


# ══════════ Public contract ══════════

def _axis_val_and_d(net, p, s):
    """Axis features and their first derivative at scattered coordinates."""
    f = lambda ss: net.apply(p, ss)
    return f(s), jvp(f, (s,), (jnp.ones_like(s),))[1]


@jax.jit
def _predict_chunk(params, xb):
    Ft = _NET_T.apply(params["t"], xb[:, 0])
    Fx, Gx = _axis_val_and_d(_NET_S, params["x"], xb[:, 1])
    Fy, Gy = _axis_val_and_d(_NET_S, params["y"], xb[:, 2])
    Fz, Gz = _axis_val_and_d(_NET_S, params["z"], xb[:, 3])
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
    separable velocity field and its autodiff curl."""
    X = jnp.asarray(X, dtype=jnp.float32)
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
    """Wall-deadline Adam on the SPINN losses.  Each step draws a fresh
    random per-axis sample, so the residual sees a new NT*NS^3 lattice."""
    ic_grid, ic_uw = _load_ic()

    k0, k1, k2, k3, rng_local = random.split(rng, 5)
    d0 = jnp.zeros((1,), dtype=jnp.float32)
    params = {"t": _NET_T.init(k0, d0), "x": _NET_S.init(k1, d0),
              "y": _NET_S.init(k2, d0), "z": _NET_S.init(k3, d0)}

    loss_fn = _make_loss(ic_grid, ic_uw)
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

    @jit
    def update(params, opt_state, key, lr):
        opt_state = opt_state._replace(
            hyperparams={**opt_state.hyperparams, "learning_rate": lr})
        ts = random.uniform(key, (NT_EX,), minval=T_MIN, maxval=T_MAX)
        (loss, aux), grads = grad_fn(params, ts, sgrid, sgrid, sgrid)
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
