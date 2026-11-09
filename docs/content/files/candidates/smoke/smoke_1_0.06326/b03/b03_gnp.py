"""Gauss-Newton phase with a Jacobi-preconditioned CG (kb1 108: NG + Jacobi scaling):
Hutchinson-estimated diag(J^T M J) preconditions the inner solve, damping is relative
to that diagonal and the step is full. Parent: blocks/b03/b03_gn.py (plain CG, eta 0.5,
which barely moved the loss in 659 steps)."""
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
T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 1   # r
HAS_SOURCE              = True   # the emission field f drives the whole plume

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi")
PERIODIC_AXES  = ()       # none: Dirichlet u = 0 on all four faces
GEOMETRY_KIND  = "box_dirichlet"

# --- Physical parameters ---
# Residual: u_t + a u_x + b u_y + c u - DXC u_xx - DYC u_yy - f = 0
# a, b (wind), c (wind divergence) and f (emission) are real ERA5/GFAS fields
# on a (t, y, x) grid in [0,1]^3; DXC, DYC are frozen diffusion constants.
DXC = 9.831690e-04              # K T / Lx^2, K = 2e4 m^2/s
DYC = 1.161416e-03              # K T / Ly^2
COEF_FILE = "smoke_coef.npz"    # keys: tn, xn, yn, a, b, c, f, dxc, dyc
GRID_NT, GRID_NY, GRID_NX = 73, 153, 241

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 180.0     # total wall-clock seconds for the WHOLE process,
                       # from _T0: imports, JIT compile, training and param
                       # save all included — eval.py kills the process at
                       # this wall; return params with margin to spare
MAX_ITER   = 1_000_000 # safety cap on the loop (not expected to bind)

# --- Eval gate: full fidelity only under task/eval.py ---
if _os.environ.get("FORGE_EVAL_TOKEN") != "task/eval.py":
    TRAIN_TIME = 10.0                                # off-eval runs are smoke:
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")   # CPU, 10 s
# ═══════════════════════════════════════════════════════════════

import math
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax import random
from soap_jax import soap


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Separable (CP) architecture ══════════
RANK      = 1024
WIDTH     = 128
DEPTH     = 5
OUT_SCALE = 0.05
LR_MAX    = 3e-3
LR_MIN    = 1e-5
CHUNK     = 25
SAVE_MARGIN_S = 8.0

# ── Gauss-Newton phase ──
GN_START  = 0.55      # fraction of the training span spent on SOAP first
CG_ITERS  = 10        # preconditioned-CG iterations per GN step
GN_ETA    = 1.0       # full Gauss-Newton step
GN_LAM    = 1e-3      # Levenberg damping, relative to the mean Jacobi diagonal
HUTCH     = 6         # Rademacher probes for diag(J^T M J)
LOG_PATH  = "blocks/b03/b03_gnp.log"

NFREQ = 24


class AxisNet(nn.Module):
    """R -> R^rank tanh MLP over one coordinate, on a Fourier-feature embedding."""
    width: int
    depth: int
    rank: int

    @nn.compact
    def __call__(self, z):          # z: [N, 1] coordinate in [0, 1]
        k = jnp.arange(1, NFREQ + 1, dtype=jnp.float32)
        w = 2.0 * jnp.pi * k * z    # [N, NFREQ]
        h = jnp.concatenate([2.0 * z - 1.0, jnp.cos(w), jnp.sin(w)], axis=-1)
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width)(h))
        return nn.Dense(self.rank)(h)


class SPINN3(nn.Module):
    """u(t,x,y) = sum_r ft_r(t) fy_r(y) fx_r(x); the t and boundary factors make
    u(0,.,.) = 0 and u = 0 on the four Dirichlet faces exact by construction."""
    width: int = WIDTH
    depth: int = DEPTH
    rank: int = RANK

    def setup(self):
        self.bt = AxisNet(self.width, self.depth, self.rank, name="axis_t")
        self.by = AxisNet(self.width, self.depth, self.rank, name="axis_y")
        self.bx = AxisNet(self.width, self.depth, self.rank, name="axis_x")

    def ft(self, t):
        return OUT_SCALE * t * self.bt(t)               # hard zero IC

    def fy(self, y):
        return 4.0 * y * (1.0 - y) * self.by(y)         # hard Dirichlet in y

    def fx(self, x):
        return 4.0 * x * (1.0 - x) * self.bx(x)         # hard Dirichlet in x

    def __call__(self, t, y, x):                        # scattered evaluation
        return jnp.sum(self.ft(t) * self.fy(y) * self.fx(x), axis=-1)


_PREDICT_MODEL = SPINN3()


def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u": [BS]}."""
    t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    outs, n, step = [], X.shape[0], 65536
    for i in range(0, n, step):
        outs.append(_PREDICT_MODEL.apply(params, t[i:i + step],
                                         y[i:i + step], x[i:i + step]))
    return {"u": jnp.concatenate(outs)}


# ══════════ Coefficient fields + separable trilinear interpolation ══════════

def _fields():
    dat = np.load(_locate(COEF_FILE))
    return jnp.stack([jnp.asarray(dat[k], jnp.float32) for k in ("a", "b", "c", "f")])


def _interp_axis(arr, coord, axis):
    """Linear interpolation of `arr` along `axis` at normalised coords `coord`."""
    n = arr.shape[axis]
    g = coord * (n - 1)
    i0 = jnp.clip(jnp.floor(g), 0, n - 2).astype(jnp.int32)
    w = g - i0
    shape = [1] * arr.ndim
    shape[axis] = -1
    lo = jnp.take(arr, i0, axis=axis)
    hi = jnp.take(arr, i0 + 1, axis=axis)
    return lo + (hi - lo) * w.reshape(shape)


def _coefs(FLD, tc, yc, xc):
    """a, b, c, f on the separable lattice (tc x yc x xc), trilinear."""
    K = _interp_axis(FLD, tc, 1)
    K = _interp_axis(K, yc, 2)
    K = _interp_axis(K, xc, 3)
    return K[0], K[1], K[2], K[3]


# Multi-scale residual: block sizes divide the jittered lattice (72, 152, 240)
LEVELS  = ((1, 1, 1), (1, 2, 2), (1, 4, 4), (2, 8, 8), (3, 19, 20), (4, 38, 60), (12, 76, 120), (72, 152, 240))
LEVEL_W = (1.0, 2.0, 4.0, 11.31, 33.76, 95.5, 330.82, 1620.67)


def _multiscale(res):
    """Weighted sum of mean-square block-averaged residuals over LEVELS."""
    nt, ny, nx = res.shape
    tot = 0.0
    for (bt, by, bx), w in zip(LEVELS, LEVEL_W):
        b = res if (bt, by, bx) == (1, 1, 1) else res.reshape(
            nt // bt, bt, ny // by, by, nx // bx, bx).mean(axis=(1, 3, 5))
        tot = tot + w * jnp.mean(b ** 2)
    return tot / sum(LEVEL_W)


def _d(fn, z):
    """d/dz of an elementwise-in-z map [N,1] -> [N,r], by one forward-mode JVP."""
    return jax.jvp(fn, (z,), (jnp.ones_like(z),))[1]


def _make_train(rng_init):
    FLD = _fields()
    _, NT, NY, NX = FLD.shape
    nt, ny, nx = NT - 1, NY - 1, NX - 1        # one jittered node per grid cell
    model = SPINN3()
    key, rng = random.split(rng_init)
    z0 = jnp.zeros((1, 1))
    params = model.init(key, z0, z0, z0)

    def lattice(key):
        kt, ky, kx = random.split(key, 3)
        tc = (jnp.arange(nt) + random.uniform(kt, (nt,))) / (NT - 1)
        yc = (jnp.arange(ny) + random.uniform(ky, (ny,))) / (NY - 1)
        xc = (jnp.arange(nx) + random.uniform(kx, (nx,))) / (NX - 1)
        return tc, yc, xc

    def resid_fn(p, tc, yc, xc, A, B, C, F):
        """PDE residual on the separable lattice; all derivatives by autodiff."""
        tg, yg, xg = tc[:, None], yc[:, None], xc[:, None]
        ft = lambda z: model.apply(p, z, method=SPINN3.ft)
        fy = lambda z: model.apply(p, z, method=SPINN3.fy)
        fx = lambda z: model.apply(p, z, method=SPINN3.fx)
        T0, T1 = ft(tg), _d(ft, tg)
        Y0, Y1 = fy(yg), _d(fy, yg)
        Y2 = _d(lambda z: _d(fy, z), yg)
        X0, X1 = fx(xg), _d(fx, xg)
        X2 = _d(lambda z: _d(fx, z), xg)
        r = RANK
        P = (T0[:, None, :] * Y0[None, :, :]).reshape(-1, r)
        G = (P @ jnp.concatenate([X0, X1, X2], axis=0).T).reshape(nt, ny, 3, nx)
        u, u_x, u_xx = G[:, :, 0, :], G[:, :, 1, :], G[:, :, 2, :]
        PS = jnp.concatenate([T1[:, None, :] * Y0[None, :, :],
                              T0[:, None, :] * Y1[None, :, :],
                              T0[:, None, :] * Y2[None, :, :]], axis=0)
        H = (PS.reshape(-1, r) @ X0.T).reshape(3, nt, ny, nx)
        u_t, u_y, u_yy = H[0], H[1], H[2]
        return u_t + A * u_x + B * u_y + C * u - DXC * u_xx - DYC * u_yy - F

    def loss_fn(p, tc, yc, xc, A, B, C, F):
        return _multiscale(resid_fn(p, tc, yc, xc, A, B, C, F))

    opt = optax.inject_hyperparams(
        lambda learning_rate: soap(learning_rate=learning_rate,
                                   precondition_frequency=40)
    )(learning_rate=LR_MAX)
    opt_state = opt.init(params)

    @jax.jit
    def chunk(p, s, key):
        def body(carry, k):
            p, s = carry
            tc, yc, xc = lattice(k)
            A, B, C, F = _coefs(FLD, tc, yc, xc)
            l, g = jax.value_and_grad(loss_fn)(p, tc, yc, xc, A, B, C, F)
            upd, s = opt.update(g, s, p)
            return (optax.apply_updates(p, upd), s), l
        keys = random.split(key, CHUNK)
        (p, s), ls = jax.lax.scan(body, (p, s), keys)
        return p, s, ls[-1]

    flat0, unravel = jax.flatten_util.ravel_pytree(params)

    def _gn_ops(th, key):
        """Linearised residual map at th on a fresh lattice: (r0, J., J^T., M.)."""
        tc, yc, xc = lattice(key)
        A, B, C, F = _coefs(FLD, tc, yc, xc)
        rf = lambda q: resid_fn(unravel(q), tc, yc, xc, A, B, C, F)
        r0, lin = jax.linearize(rf, th)             # J v
        linT = jax.linear_transpose(lin, th)        # J^T w
        # _multiscale is quadratic in r, so grad/2 is exactly the metric M applied to r
        Mv = lambda r: 0.5 * jax.grad(_multiscale)(r)
        return r0, lin, linT, Mv

    @jax.jit
    def gn_diag(th, key):
        """Hutchinson estimate of diag(J^T M J) — the Jacobi preconditioner."""
        r0, lin, linT, Mv = _gn_ops(th, key)

        def body(acc, k):
            v = random.rademacher(k, th.shape, dtype=th.dtype)
            return acc + v * linT(Mv(lin(v)))[0], None

        acc, _ = jax.lax.scan(body, jnp.zeros_like(th), random.split(key, HUTCH))
        return jnp.maximum(acc / HUTCH, 0.0)

    @jax.jit
    def gn_step(th, dg, key):
        """One Gauss-Newton step: Jacobi-preconditioned CG on (J^T M J + lam I)."""
        r0, lin, linT, Mv = _gn_ops(th, key)
        lam = GN_LAM * jnp.mean(dg) + 1e-30
        Av = lambda v: linT(Mv(lin(v)))[0] + lam * v
        Minv = lambda v: v / (dg + lam)
        rhs = -linT(Mv(r0))[0]
        d, _ = jax.scipy.sparse.linalg.cg(Av, rhs, M=Minv, tol=1e-10, atol=0.0,
                                          maxiter=CG_ITERS)
        return th + GN_ETA * d, _multiscale(r0)

    return params, opt_state, chunk, gn_diag, gn_step, unravel, rng


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """SOAP on the jittered-lattice pyramid loss for the first GN_START of the wall,
    then Jacobi-preconditioned Gauss-Newton (matrix-free CG) for the rest."""
    params, opt_state, chunk, gn_diag, gn_step, unravel, key = _make_train(rng)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    span = max(1e-6, TRAIN_TIME - SAVE_MARGIN_S)
    t_switch = _T0 + GN_START * span

    # compile both GN kernels up front so XLA time is not paid out of the GN phase
    th0 = jax.flatten_util.ravel_pytree(params)[0]
    key, sub = random.split(key)
    dg0 = gn_diag(th0, sub)
    _t, _l = gn_step(th0, dg0, sub)
    _l.block_until_ready()
    t_gn_ready = time.time()

    key, sub = random.split(key)
    params, opt_state, loss = chunk(params, opt_state, sub)   # warmup: JIT + steps
    soap_steps, last = CHUNK, float(loss)
    t_soap_ready = time.time()

    while time.time() < min(deadline, t_switch):
        frac = min(1.0, max(0.0, (time.time() - _T0) / (GN_START * span)))
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * frac))
        opt_state = opt_state._replace(
            hyperparams={**opt_state.hyperparams,
                         "learning_rate": jnp.asarray(lr, jnp.float32)})
        key, sub = random.split(key)
        params, opt_state, loss = chunk(params, opt_state, sub)
        soap_steps += CHUNK
        last = float(loss)
    t_soap_end, switch_loss = time.time(), last

    th = jax.flatten_util.ravel_pytree(params)[0]
    gn_steps = 0
    if time.time() < deadline:
        key, sub = random.split(key)
        dg = gn_diag(th, sub)
        while time.time() < deadline:
            key, sub = random.split(key)
            th, loss = gn_step(th, dg, sub)
            last = float(loss)
            gn_steps += 1
        params = unravel(th)

    try:
        with open(LOG_PATH, "a") as fh:
            fh.write(f"gn_compile={t_gn_ready-_T0:.1f}s soap_compile="
                     f"{t_soap_ready-t_gn_ready:.1f}s soap_steps={soap_steps} "
                     f"soap_s={t_soap_end-t_soap_ready:.1f} switch_loss={switch_loss:.6e} "
                     f"gn_steps={gn_steps} gn_s={time.time()-t_soap_end:.1f} "
                     f"last_loss={last:.6e}\n")
    except OSError:
        pass
    return params, soap_steps + gn_steps
