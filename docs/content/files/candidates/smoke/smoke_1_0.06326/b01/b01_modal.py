"""Partially separable "modal" PINN: u(t,x,y) = sum_r c_r(t) g_r(y,x), with c_r a
1-D time MLP and g_r a full 2-D MLP over (y,x) — general spatial modes, rank only
in time, so an advecting plume no longer needs a high CP rank. Jittered lattice.
Parent: blocks/b01/b01_big.py (rank-256 CP over all three axes)."""
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


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Modal architecture ══════════
RANK        = 128    # number of spatial modes
TIME_WIDTH  = 64
TIME_DEPTH  = 4
SPACE_WIDTH = 128
SPACE_DEPTH = 4
OUT_SCALE   = 0.05
LR_MAX      = 2e-3
LR_MIN      = 1e-5
CHUNK       = 25
SAVE_MARGIN_S = 8.0


class MLP(nn.Module):
    width: int
    depth: int
    rank: int

    @nn.compact
    def __call__(self, z):          # z: [N, d] coordinates in [0, 1]
        h = 2.0 * z - 1.0
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width)(h))
        return nn.Dense(self.rank)(h)


class ModalPINN(nn.Module):
    """u = sum_r ct_r(t) gs_r(y,x); the t factor makes u(0,.,.) = 0 and the
    x(1-x) y(1-y) factor makes u = 0 on the four Dirichlet faces, exactly."""
    rank: int = RANK

    def setup(self):
        self.tnet = MLP(TIME_WIDTH, TIME_DEPTH, self.rank, name="time")
        self.snet = MLP(SPACE_WIDTH, SPACE_DEPTH, self.rank, name="space")

    def ct(self, t):                                   # t: [N,1] -> [N,rank]
        return OUT_SCALE * t * self.tnet(t)            # hard zero IC

    def gs(self, yx):                                  # yx: [M,2] -> [M,rank]
        y, x = yx[:, 0:1], yx[:, 1:2]
        mask = 16.0 * y * (1.0 - y) * x * (1.0 - x)    # hard Dirichlet
        return mask * self.snet(yx)

    def __call__(self, t, yx):
        return jnp.sum(self.ct(t) * self.gs(yx), axis=-1)


_PREDICT_MODEL = ModalPINN()


def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u": [BS]}."""
    t = X[:, 0:1]
    yx = jnp.stack([X[:, 2], X[:, 1]], axis=1)
    outs, n, step = [], X.shape[0], 65536
    for i in range(0, n, step):
        outs.append(_PREDICT_MODEL.apply(params, t[i:i + step], yx[i:i + step]))
    return {"u": jnp.concatenate(outs)}


# ══════════ Coefficient fields + separable trilinear interpolation ══════════

def _fields():
    dat = np.load(_locate(COEF_FILE))
    return jnp.stack([jnp.asarray(dat[k], jnp.float32) for k in ("a", "b", "c", "f")])


def _interp_axis(arr, coord, axis):
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
    K = _interp_axis(FLD, tc, 1)
    K = _interp_axis(K, yc, 2)
    K = _interp_axis(K, xc, 3)
    return K[0], K[1], K[2], K[3]


def _make_train(rng_init):
    FLD = _fields()
    _, NT, NY, NX = FLD.shape
    nt, ny, nx = NT - 1, NY - 1, NX - 1
    M = ny * nx
    model = ModalPINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros((1, 1)), jnp.zeros((1, 2)))

    def lattice(key):
        kt, ky, kx = random.split(key, 3)
        tc = (jnp.arange(nt) + random.uniform(kt, (nt,))) / (NT - 1)
        yc = (jnp.arange(ny) + random.uniform(ky, (ny,))) / (NY - 1)
        xc = (jnp.arange(nx) + random.uniform(kx, (nx,))) / (NX - 1)
        return tc, yc, xc

    def loss_fn(p, tc, yc, xc, A, B, C, F):
        ct = lambda z: model.apply(p, z, method=ModalPINN.ct)
        gs = lambda z: model.apply(p, z, method=ModalPINN.gs)
        tg = tc[:, None]
        T0 = ct(tg)
        T1 = jax.jvp(ct, (tg,), (jnp.ones_like(tg),))[1]
        P = jnp.stack([jnp.repeat(yc, nx), jnp.tile(xc, ny)], axis=1)   # [M,2]
        ey = jnp.zeros_like(P).at[:, 0].set(1.0)
        ex = jnp.zeros_like(P).at[:, 1].set(1.0)
        G0, Gy = jax.jvp(gs, (P,), (ey,))
        Gyy = jax.jvp(lambda q: jax.jvp(gs, (q,), (ey,))[1], (P,), (ey,))[1]
        Gx = jax.jvp(gs, (P,), (ex,))[1]
        Gxx = jax.jvp(lambda q: jax.jvp(gs, (q,), (ex,))[1], (P,), (ex,))[1]
        S = jnp.concatenate([G0, Gx, Gxx, Gy, Gyy], axis=0)             # [5M,rank]
        Q = (T0 @ S.T).reshape(nt, 5, ny, nx)
        u, u_x, u_xx, u_y, u_yy = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3], Q[:, 4]
        u_t = (T1 @ G0.T).reshape(nt, ny, nx)
        res = u_t + A * u_x + B * u_y + C * u - DXC * u_xx - DYC * u_yy - F
        return jnp.mean(res ** 2)

    opt = optax.inject_hyperparams(optax.adam)(learning_rate=LR_MAX)
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
    """Wall-deadline Adam on the residual-only modal loss over a freshly jittered
    lattice each step; cosine LR decay driven by elapsed wall time."""
    params, opt_state, chunk, key = _make_train(rng)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    span = max(1e-6, TRAIN_TIME - SAVE_MARGIN_S)

    key, sub = random.split(key)
    params, opt_state, loss = chunk(params, opt_state, sub)
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
