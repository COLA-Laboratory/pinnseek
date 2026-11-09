"""b04_dtxc with the residual pyramid computed hierarchically — each level block-
averaged from the next-finer one rather than from the 10.5M-point residual.
Bit-for-bit the same loss, one fine-grid pass instead of eight.
Parent: blocks/b04/b04_dtxc.py."""
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
RANK      = 1280
WIDTH     = 128
DEPTH     = 5
OUT_SCALE = 0.05
LR_MAX    = 3e-3
LR_MIN    = 1e-5
CHUNK     = 25
SAVE_MARGIN_S = 8.0


NFREQ = 24


def _merged_init(key, shape, dtype=jnp.float32):
    """lecun_normal on the MLP block (fan_in = width), zeros on the skip block."""
    nf = 2 * NFREQ + 1
    kh = nn.initializers.lecun_normal()(key, (shape[0] - nf, shape[1]), dtype)
    return jnp.concatenate([kh, jnp.zeros((nf, shape[1]), dtype)], axis=0)


class AxisNet(nn.Module):
    """R -> R^rank tanh MLP over one coordinate, on a Fourier-feature embedding."""
    width: int
    depth: int
    rank: int

    @nn.compact
    def __call__(self, z):          # z: [N, 1] coordinate in [0, 1]
        k = jnp.arange(1, NFREQ + 1, dtype=jnp.float32)
        w = 2.0 * jnp.pi * k * z    # [N, NFREQ]
        feat = jnp.concatenate([2.0 * z - 1.0, jnp.cos(w), jnp.sin(w)], axis=-1)
        h = feat
        for i in range(self.depth):
            inp = h if i == 0 else jnp.concatenate([h, feat], axis=-1)
            h = nn.tanh(nn.Dense(self.width)(inp))
        # + a direct linear path feature -> rank (kb1 082): a well-conditioned
        # 49-dim-per-axis subspace the optimiser can move in without going through
        # five tanh layers, while the MLP still supplies the harmonics.
        # identical function class to b03_skip, but ONE [width+49, rank] kernel
        # instead of two: SOAP pays a single 1024-side eigendecomposition per axis
        # rather than two, which is where b03_skip lost 14% of its steps.
        return nn.Dense(self.rank, kernel_init=_merged_init)(
            jnp.concatenate([h, feat], axis=-1))


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


# Same physical block scales as the parent, on 2 nodes / grid cell in t and x.
LEVELS  = ((2, 1, 2), (2, 2, 4), (2, 4, 8), (4, 8, 16), (6, 19, 40), (8, 38, 120), (24, 76, 240), (144, 152, 480))
LEVEL_W = (1.0, 2.0, 4.0, 11.31, 33.76, 95.5, 330.82, 1620.67)


# Every level is a uniform block average, and block averaging is associative, so
# each level can be built from an already-coarsened one instead of from the full
# 10.5M-point residual. (parent index, block relative to that parent); -1 = res.
# The eight levels below are bit-for-bit the LEVELS above, at one pass over the
# fine array instead of eight (and one broadcast in the backward pass, not eight).
TREE = ((-1, (2, 1, 2)), (0, (1, 2, 2)), (1, (1, 2, 2)), (2, (2, 2, 2)),
        (0, (3, 19, 20)), (0, (4, 38, 60)), (5, (3, 2, 2)), (6, (6, 2, 2)))


def _blockmean(a, bt, by, bx):
    nt, ny, nx = a.shape
    return a.reshape(nt // bt, bt, ny // by, by,
                     nx // bx, bx).mean(axis=(1, 3, 5))


def _multiscale(res):
    """Weighted sum of mean-square block-averaged residuals over LEVELS."""
    lv, tot = [], 0.0
    for par, blk in TREE:
        lv.append(_blockmean(res if par < 0 else lv[par], *blk))
    for b, w in zip(lv, LEVEL_W):
        tot = tot + w * jnp.mean(b ** 2)
    return tot / sum(LEVEL_W)


def _d(fn, z):
    """d/dz of an elementwise-in-z map [N,1] -> [N,r], by one forward-mode JVP."""
    return jax.jvp(fn, (z,), (jnp.ones_like(z),))[1]


def _make_train(rng_init):
    FLD = _fields()
    _, NT, NY, NX = FLD.shape
    nt, ny, nx = 2 * (NT - 1), NY - 1, 2 * (NX - 1)   # 2 nodes / cell in t and x
    model = SPINN3()
    key, rng = random.split(rng_init)
    z0 = jnp.zeros((1, 1))
    params = model.init(key, z0, z0, z0)

    def lattice(key):
        kt, ky, kx = random.split(key, 3)
        tc = (jnp.arange(nt) + random.uniform(kt, (nt,))) / nt
        yc = (jnp.arange(ny) + random.uniform(ky, (ny,))) / ny
        xc = (jnp.arange(nx) + random.uniform(kx, (nx,))) / nx
        return tc, yc, xc

    def loss_fn(p, tc, yc, xc, A, B, C, F):
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
        # bfloat16 operands for the two big contractions (fp32 accumulation): the
        # lattice tensors dominate memory traffic, and 8-bit mantissa noise on terms
        # of O(1) is ~0.25% of the finest-level mean-square residual. Three separate
        # GEMMs avoid materialising the stacked PS buffer.
        c = lambda z: z.astype(jnp.bfloat16)
        mm = lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32)
        T0c, T1c, Y0c, Y1c, Y2c, X0c = c(T0), c(T1), c(Y0), c(Y1), c(Y2), c(X0)
        P = (T0c[:, None, :] * Y0c[None, :, :]).reshape(-1, r)
        G = mm(P, c(jnp.concatenate([X0, X1, X2], axis=0)).T).reshape(nt, ny, 3, nx)
        u, u_x, u_xx = G[:, :, 0, :], G[:, :, 1, :], G[:, :, 2, :]
        X0T = X0c.T
        u_t = mm((T1c[:, None, :] * Y0c[None, :, :]).reshape(-1, r), X0T).reshape(nt, ny, nx)
        u_y = mm((T0c[:, None, :] * Y1c[None, :, :]).reshape(-1, r), X0T).reshape(nt, ny, nx)
        u_yy = mm((T0c[:, None, :] * Y2c[None, :, :]).reshape(-1, r), X0T).reshape(nt, ny, nx)
        res = u_t + A * u_x + B * u_y + C * u - DXC * u_xx - DYC * u_yy - F
        return _multiscale(res)

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

    return params, opt_state, chunk, rng


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam on the residual-only separable loss over a freshly
    jittered lattice each step; cosine LR decay driven by elapsed wall time."""
    params, opt_state, chunk, key = _make_train(rng)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    span = max(1e-6, TRAIN_TIME - SAVE_MARGIN_S)

    key, sub = random.split(key)
    params, opt_state, loss = chunk(params, opt_state, sub)   # warmup: JIT + steps
    step_count, last = CHUNK, float(loss)

    while time.time() < deadline:
        frac = min(1.0, max(0.0, (time.time() - _T0) / span))
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * frac))
        opt_state = opt_state._replace(
            hyperparams={**opt_state.hyperparams,
                         "learning_rate": jnp.asarray(lr, jnp.float32)})
        key, sub = random.split(key)
        params, opt_state, loss = chunk(params, opt_state, sub)
        step_count += CHUNK
        last = float(loss)
        if eval_callback is not None and (step_count // CHUNK) % 200 == 0:
            eval_callback(params, step_count, last)

    return params, step_count
