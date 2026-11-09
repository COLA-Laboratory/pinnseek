"""Fourier-feature modified-MLP PINN with causal temporal weighting (kb1 084).
Changed vs b01_f1.py: collocation is stratified into 32 time bins and each
bin's residual is weighted by exp(-eps * sum of earlier bins' residuals)
(stop-gradient); eps ramps linearly to 0 over the first 75% of the wall clock.
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
from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jit, random
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


def inlet_centreline(t):
    return U_PEAK * jnp.sin(jnp.pi * (t + T_START) / T_DRIVE)


def inlet_profile(t, y):
    return 4.0 * inlet_centreline(t) * y * (Y_MAX - y) / (Y_MAX ** 2)


# ══════════ Architecture: modified MLP (kb1 050 / 083) ══════════

_LO = jnp.array([T_MIN, X_MIN, Y_MIN])
_SC = jnp.array([2.0 / (T_MAX - T_MIN), 2.0 / (X_MAX - X_MIN), 2.0 / (Y_MAX - Y_MIN)])

WIDTH = 128
DEPTH = 6

# fixed anisotropic random Fourier basis (kb1 020): sigma per (t, x, y) axis
_M_FF = 96
_SIGMA_FF = np.array([2.0, 5.0, 3.0], dtype=np.float32)
_B_FF = jnp.array((np.random.default_rng(12345).standard_normal((_M_FF, 3))
                   * _SIGMA_FF).astype(np.float32))


class PINN(nn.Module):
    """Gated modified-MLP (kb1 050) on a fixed random Fourier embedding of the
    normalised inputs (kb1 020), concatenated with the raw coordinates."""
    width: int = WIDTH
    depth: int = DEPTH

    @nn.compact
    def __call__(self, txy):
        kinit = nn.initializers.glorot_normal()
        z = (txy - _LO) * _SC - 1.0
        proj = 2.0 * jnp.pi * z @ _B_FF.T
        h0 = jnp.concatenate([jnp.cos(proj), jnp.sin(proj), z], axis=-1)
        U = nn.tanh(nn.Dense(self.width, kernel_init=kinit, name="U")(h0))
        V = nn.tanh(nn.Dense(self.width, kernel_init=kinit, name="V")(h0))
        h = nn.tanh(nn.Dense(self.width, kernel_init=kinit, name="L0")(h0))
        for i in range(1, self.depth):
            z = nn.tanh(nn.Dense(self.width, kernel_init=kinit, name=f"L{i}")(h))
            h = (1.0 - z) * U + z * V
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit, name="out")(h)


# ══════════ Sampling sizes ══════════

N_UNI, N_WAKE, N_ANN = 3072, 1024, 1024
N_BIN = 32                                     # causal time bins (kb1 084)
N_PER = (N_UNI + N_WAKE + N_ANN) // N_BIN
EPS0, EPS_OFF = 2.0, 0.75                      # causal strength, off by 75% wall
N_IC = 2048
N_IN, N_WALL, N_CYL, N_OUT = 512, 1024, 512, 256

W_IC, W_BC, W_OUT = 100.0, 100.0, 1.0
LR0, LR_DECAY, LR_EVERY, LR_END = 1e-3, 0.92, 1500, 3e-5
N_INNER = 50
SAVE_MARGIN_S = 8.0


def _load_ic():
    ic = np.loadtxt(_locate("cyl_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    pts = np.stack([np.zeros_like(ic[:, 0]), ic[:, 0], ic[:, 1]], 1).astype(np.float32)
    uv = np.stack([ic[:, 2], ic[:, 3]], 1).astype(np.float32)
    return jnp.array(pts), jnp.array(uv)


def _sample_col(key):
    """Time-stratified collocation: [N_BIN, N_PER, 3] with bin i in the i-th
    slice of [0, T_MAX]; spatial mixture uniform / wake box / cylinder annulus."""
    ks = random.split(key, 8)
    sh = (N_BIN, N_UNI // N_BIN, 1)
    shw = (N_BIN, N_WAKE // N_BIN, 1)
    sha = (N_BIN, N_ANN // N_BIN, 1)
    xu = random.uniform(ks[0], sh, minval=X_MIN, maxval=X_MAX)
    yu = random.uniform(ks[1], sh, minval=Y_MIN, maxval=Y_MAX)
    xw = random.uniform(ks[2], shw, minval=0.10, maxval=1.30)
    yw = random.uniform(ks[3], shw, minval=0.03, maxval=0.38)
    th = random.uniform(ks[4], sha, minval=0.0, maxval=2.0 * jnp.pi)
    rr = CYL_R + 0.15 * random.uniform(ks[5], sha) ** 2
    xy = jnp.concatenate([jnp.concatenate([xu, yu], 2),
                          jnp.concatenate([xw, yw], 2),
                          jnp.concatenate([CYL_X + rr * jnp.cos(th),
                                           CYL_Y + rr * jnp.sin(th)], 2)], 1)
    dt = (T_MAX - T_MIN) / N_BIN
    edges = T_MIN + dt * jnp.arange(N_BIN).reshape(N_BIN, 1, 1)
    tt = edges + dt * random.uniform(ks[6], (N_BIN, N_PER, 1))
    X = jnp.concatenate([tt, xy], 2)
    m = ((X[:, :, 1] - CYL_X) ** 2 + (X[:, :, 2] - CYL_Y) ** 2) > CYL_R ** 2
    return X.reshape(-1, 3), m.astype(jnp.float32)


def _sample_bc(key):
    ks = random.split(key, 8)
    tb = lambda k, n: random.uniform(k, (n, 1), minval=T_MIN, maxval=T_MAX)
    yin = random.uniform(ks[0], (N_IN, 1), minval=Y_MIN, maxval=Y_MAX)
    b_in = jnp.concatenate([tb(ks[1], N_IN), jnp.full((N_IN, 1), X_MIN), yin], 1)
    yout = random.uniform(ks[2], (N_OUT, 1), minval=Y_MIN, maxval=Y_MAX)
    b_out = jnp.concatenate([tb(ks[3], N_OUT), jnp.full((N_OUT, 1), X_MAX), yout], 1)
    xw = random.uniform(ks[4], (N_WALL, 1), minval=X_MIN, maxval=X_MAX)
    yw = jnp.where(random.bernoulli(ks[5], 0.5, (N_WALL, 1)), Y_MIN, Y_MAX)
    b_wall = jnp.concatenate([tb(ks[6], N_WALL), xw, yw], 1)
    th = random.uniform(ks[7], (N_CYL, 1), minval=0.0, maxval=2.0 * jnp.pi)
    b_cyl = jnp.concatenate([tb(ks[0], N_CYL), CYL_X + CYL_R * jnp.cos(th),
                             CYL_Y + CYL_R * jnp.sin(th)], 1)
    return b_in, b_out, b_wall, b_cyl


def _make_train(rng_init):
    pts_ic, uv_ic = _load_ic()
    n_ic = len(pts_ic)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))

    def net(p, X):
        return model.apply(p, X)

    def derivs(p, X):
        f = lambda Z: net(p, Z)
        ex = jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), X.shape)
        ey = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), X.shape)
        et = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), X.shape)
        fx = lambda Z: jax.jvp(f, (Z,), (ex,))
        (val, dx), (_, dxx) = jax.jvp(fx, (X,), (ex,))
        fy = lambda Z: jax.jvp(f, (Z,), (ey,))
        (_, dy), (_, dyy) = jax.jvp(fy, (X,), (ey,))
        _, dt = jax.jvp(f, (X,), (et,))
        return val, dt, dx, dy, dxx, dyy

    def pde_loss(p, X, m, eps):
        val, dt, dx, dy, dxx, dyy = derivs(p, X)
        u, v = val[:, 0], val[:, 1]
        r_c = dx[:, 0] + dy[:, 1]
        r_mx = (dt[:, 0] + u * dx[:, 0] + v * dy[:, 0] + dx[:, 2]
                - NU * (dxx[:, 0] + dyy[:, 0]))
        r_my = (dt[:, 1] + u * dx[:, 1] + v * dy[:, 1] + dy[:, 2]
                - NU * (dxx[:, 1] + dyy[:, 1]))
        r = (r_c ** 2 + r_mx ** 2 + r_my ** 2).reshape(N_BIN, N_PER)
        mb = m.reshape(N_BIN, N_PER)
        L = jnp.sum(mb * r, 1) / jnp.sum(mb, 1)             # per-bin residual
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(L)[:-1]])
        w = jax.lax.stop_gradient(jnp.exp(-eps * cum))      # causal weights
        return jnp.mean(w * L)

    def loss_fn(p, key, eps):
        kc, kb, ki = random.split(key, 3)
        Xc, mc = _sample_col(kc)
        l_pde = pde_loss(p, Xc, mc, eps)

        idx = random.choice(ki, n_ic, (N_IC,))
        fic = net(p, pts_ic[idx])
        l_ic = jnp.mean((fic[:, 0] - uv_ic[idx, 0]) ** 2
                        + (fic[:, 1] - uv_ic[idx, 1]) ** 2)

        b_in, b_out, b_wall, b_cyl = _sample_bc(kb)
        fin = net(p, b_in)
        l_in = jnp.mean((fin[:, 0] - inlet_profile(b_in[:, 0], b_in[:, 2])) ** 2
                        + fin[:, 1] ** 2)
        fw = net(p, b_wall)
        l_wall = jnp.mean(fw[:, 0] ** 2 + fw[:, 1] ** 2)
        fc = net(p, b_cyl)
        l_cyl = jnp.mean(fc[:, 0] ** 2 + fc[:, 1] ** 2)
        fo = net(p, b_out)
        l_out = jnp.mean(fo[:, 2] ** 2)

        return (l_pde + W_IC * l_ic + W_BC * (l_in + l_wall + l_cyl)
                + W_OUT * l_out)

    sched = optax.exponential_decay(LR0, LR_EVERY, LR_DECAY, end_value=LR_END)
    optimizer = optax.adam(learning_rate=sched)
    opt_state = optimizer.init(params)
    grad_fn = jax.value_and_grad(loss_fn)

    @jit
    def chunk(p, os_, key, eps):
        def body(carry, _):
            p, os_, k = carry
            k, sub = random.split(k)
            loss, g = grad_fn(p, sub, eps)
            upd, os_ = optimizer.update(g, os_)
            return (optax.apply_updates(p, upd), os_, k), loss
        (p, os_, key), losses = jax.lax.scan(
            body, (p, os_, key), None, length=N_INNER)
        return p, os_, key, losses[-1]

    return params, opt_state, chunk, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u","v","p"} each [BS]."""
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, opt_state, chunk, key = _make_train(rng)

    eps = jnp.float32(EPS0)
    params, opt_state, key, loss = chunk(params, opt_state, key, eps)
    step_count = N_INNER
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    dt = 1.0
    while step_count < MAX_ITER and time.time() + 1.6 * dt < deadline:
        t0 = time.time()
        frac = (t0 - _T0) / TRAIN_TIME
        eps = jnp.float32(EPS0 * max(0.0, 1.0 - frac / EPS_OFF))
        params, opt_state, key, loss = chunk(params, opt_state, key, eps)
        last_loss = float(loss)          # sync: JAX dispatch is async
        dt = time.time() - t0
        step_count += N_INNER
        if eval_callback is not None and (step_count % 5000 == 0):
            eval_callback(params, step_count, last_loss)

    return params, step_count
