"""Time-marching v2: per-window t-RESCALED embedding + late-biased windows (targets the diagnosed late-time mid-k failure).
Each window's net rescales t to [0,1] over its own [t_lo,t_hi], so the late,
fast-growing interval gets full temporal resolution. Late window gets more wall.
Light modified-MLP backbone (b01_combo). Parent: b01_combo (diag: error is 0 early, 84% at t=0.4).
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 2
OUTPUT_DIM   = 1
INPUT_NAMES  = ("t", "x")
FIELD_NAMES  = ("u",)

# --- Domain ---
T_MIN, T_MAX = 0.0, 0.4
X_MIN, X_MAX = 0.0, 6.283185307179586   # 2*pi
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 4   # u_xxxx
N_RESIDUAL_COMPONENTS   = 1
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi")
PERIODIC_AXES  = ("x",)       # u(t, X_MIN) = u(t, X_MAX)
GEOMETRY_KIND  = "periodic_strip"

# --- Physical parameters (scaled KS: u_t + V1 u u_x + V2 u_xx + V3 u_xxxx = 0) ---
V1 = 100.0 / 16.0        # 6.25
V2 = 100.0 / 16.0**2     # 0.390625
V3 = 100.0 / 16.0**4     # ~1.5259e-3

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
from jax import grad, jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


def _locate(name: str) -> Path:
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Hyperparameters ══════════

MX        = 16
MT        = 6
SIGMA_T   = 3.0
WIDTH     = 144
DEPTH     = 4
NT        = 32
NX        = 128
NIC       = 512
LR0       = 2.0e-3
EPS_SCHED = (1e-2, 1e-1, 1e0, 1e1, 1e2)
W_DELTA   = 0.99
LAM_IC0   = 100.0
LAM_ALPHA = 0.9
LAM_EVERY = 500
SAVE_MARGIN_S = 7.0

# late-biased windows and wall allocation (fractions of TRAIN_TIME)
BOUNDS    = (0.0, 0.22, 0.40)     # 2 windows; late one is shorter → finer resolution
WALL_FRAC = (0.42, 0.58)          # give the harder late window more wall


_BT = np.asarray(
    random.normal(random.PRNGKey(12345), (MT,)) * SIGMA_T, dtype=np.float32)


def _embed(tx, t_lo, t_hi):
    """Exact-periodic x + Fourier-time features; t RESCALED to [0,1] over
    this window's [t_lo, t_hi]."""
    t = tx[:, 0:1]
    x = tx[:, 1:2]
    ts = (t - t_lo) / (t_hi - t_lo)               # local scaled t ∈ [0,1]
    ks = jnp.arange(1, MX + 1, dtype=jnp.float32)
    xf = jnp.concatenate([jnp.cos(x * ks), jnp.sin(x * ks)], axis=1)
    bt = jnp.asarray(_BT)
    tf = jnp.concatenate([jnp.cos(2 * jnp.pi * ts * bt),
                          jnp.sin(2 * jnp.pi * ts * bt)], axis=1)
    return jnp.concatenate([ts, xf, tf], axis=1)


class ModifiedMLP(nn.Module):
    width: int = WIDTH
    depth: int = DEPTH

    @nn.compact
    def __call__(self, tx, t_lo, t_hi):
        kinit = jax.nn.initializers.glorot_normal()
        h0 = _embed(tx, t_lo, t_hi)
        U = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h0))
        V = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h0))
        H = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h0))
        for _ in range(self.depth):
            Z = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(H))
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(H)


_PREDICT_MODEL = ModifiedMLP()


def _load_ic_file():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0], ic[:, 1]
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic], axis=1).astype(np.float32)
    return jnp.array(pts_ic), jnp.array(u_ic.astype(np.float32))


def _build_window(rng_init, t_lo, t_hi):
    """Build (params, jitted step, lam_step, unravel) for one window with its
    own t-rescale baked in (one JIT compile per window)."""
    model = ModifiedMLP()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]), t_lo, t_hi)
    flat0, unravel = jax.flatten_util.ravel_pytree(params)

    def u_scalar(pf, t, x):
        return model.apply(unravel(pf), jnp.stack([t, x])[None, :], t_lo, t_hi)[0, 0]

    u_t_fn = grad(u_scalar, argnums=1)

    def _dx(f):
        return lambda pf, t, x: jax.jvp(
            lambda xx: f(pf, t, xx), (x,), (jnp.ones_like(x),))[1]

    u_x_fn    = _dx(u_scalar)
    u_xx_fn   = _dx(u_x_fn)
    u_xxx_fn  = _dx(u_xx_fn)
    u_xxxx_fn = _dx(u_xxx_fn)

    def residual_point(pf, t, x):
        u      = u_scalar(pf, t, x)
        u_t    = u_t_fn(pf, t, x)
        u_x    = u_x_fn(pf, t, x)
        u_xx   = u_xx_fn(pf, t, x)
        u_xxxx = u_xxxx_fn(pf, t, x)
        return u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx

    def slice_losses(pf, t_grid, x_grid):
        def per_slice(ti):
            r = vmap(lambda xj: residual_point(pf, ti, xj))(x_grid)
            return jnp.mean(r ** 2)
        return vmap(per_slice)(t_grid)

    u_batch = vmap(u_scalar, in_axes=(None, 0, 0))

    def ic_loss(pf, ic_xy, ic_u):
        up = u_batch(pf, ic_xy[:, 0], ic_xy[:, 1])
        return jnp.mean((up - ic_u) ** 2)

    def residual_loss(pf, t_grid, x_grid, eps):
        Lr = slice_losses(pf, t_grid, x_grid)
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(Lr)[:-1]])
        w = jax.lax.stop_gradient(jnp.exp(-eps * cum))
        return jnp.mean(w * Lr), w

    def total_loss(pf, t_grid, x_grid, ic_xy, ic_u, lam_ic, eps):
        Lr, w = residual_loss(pf, t_grid, x_grid, eps)
        Li = ic_loss(pf, ic_xy, ic_u)
        return Lr + lam_ic * Li, (w, Lr, Li)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR0, warmup_steps=300,
        decay_steps=9000, end_value=1e-5)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(flat0)

    loss_grad = jit(jax.value_and_grad(total_loss, has_aux=True))

    def _lam_update(pf, t_grid, x_grid, ic_xy, ic_u, eps):
        gr = jax.grad(lambda p: residual_loss(p, t_grid, x_grid, eps)[0])(pf)
        gi = jax.grad(lambda p: ic_loss(p, ic_xy, ic_u))(pf)
        return jnp.max(jnp.abs(gr)) / (jnp.mean(jnp.abs(gi)) + 1e-12)
    lam_update = jit(_lam_update)

    @jit
    def sample(rng, tmax):
        r1, r2 = random.split(rng)
        ts = jnp.sort(random.uniform(r1, (NT,), minval=t_lo + 1e-4, maxval=tmax))
        xs = random.uniform(r2, (NX,), minval=X_MIN, maxval=X_MAX)
        return ts, xs

    @jit
    def step(pf, opt_state, rng, lam_ic, eps, tmax, ic_xy, ic_u):
        rng, sk = random.split(rng)
        t_grid, x_grid = sample(sk, tmax)
        (loss, (w, Lr, Li)), g = loss_grad(pf, t_grid, x_grid, ic_xy, ic_u, lam_ic, eps)
        updates, opt_state = optimizer.update(g, opt_state)
        pf = optax.apply_updates(pf, updates)
        return pf, opt_state, rng, w, loss

    def lam_step(pf, rng, eps, tmax, ic_xy, ic_u):
        rng, sk = random.split(rng)
        t_grid, x_grid = sample(sk, tmax)
        lam = lam_update(pf, t_grid, x_grid, ic_xy, ic_u, eps)
        return lam, rng

    return dict(flat0=flat0, unravel=unravel, opt_state=opt_state,
                step=step, lam_step=lam_step, rng=rng)


def _run_window(W, t_lo, t_hi, ic_xy, ic_u, wall_end):
    step, lam_step = W["step"], W["lam_step"]
    pf, opt_state, rng = W["flat0"], W["opt_state"], W["rng"]
    lam_ic = jnp.asarray(LAM_IC0, dtype=jnp.float32)
    eps_idx = 0
    eps = jnp.asarray(EPS_SCHED[eps_idx], dtype=jnp.float32)
    tmax_start = t_lo + 0.05 * (t_hi - t_lo)
    win_t0 = _time.time()
    ramp_wall = 0.55 * (wall_end - win_t0)

    def horizon(now):
        prog = min(1.0, (now - win_t0) / max(ramp_wall, 1e-6))
        return jnp.asarray(tmax_start + (t_hi - tmax_start) * prog, dtype=jnp.float32)

    pf, opt_state, rng, w, loss = step(
        pf, opt_state, rng, lam_ic, eps, horizon(time.time()), ic_xy, ic_u)
    steps = 1
    for it in range(1, MAX_ITER):
        now = time.time()
        if now > wall_end:
            break
        th = horizon(now)
        pf, opt_state, rng, w, loss = step(pf, opt_state, rng, lam_ic, eps, th, ic_xy, ic_u)
        steps += 1
        if it % LAM_EVERY == 0:
            new_lam, rng = lam_step(pf, rng, eps, th, ic_xy, ic_u)
            new_lam = jnp.clip(new_lam, 1.0, 1e5)
            lam_ic = LAM_ALPHA * lam_ic + (1.0 - LAM_ALPHA) * new_lam
            if float(jnp.min(w)) > W_DELTA and eps_idx < len(EPS_SCHED) - 1:
                eps_idx += 1
                eps = jnp.asarray(EPS_SCHED[eps_idx], dtype=jnp.float32)
    return W["unravel"](pf), steps


def predict_fn(params, X):
    bounds, plist = params
    t = X[:, 0]
    u = jnp.zeros(X.shape[0], dtype=jnp.float32)
    nwin = len(plist)
    for i in range(nwin):
        t_lo, t_hi = bounds[i], bounds[i + 1]
        ui = _PREDICT_MODEL.apply(plist[i], X, t_lo, t_hi)[:, 0]
        upper = t <= t_hi if i == nwin - 1 else t < t_hi
        sel = (t >= t_lo) & upper if i > 0 else (t <= t_hi)
        u = jnp.where(sel, ui, u)
    return {"u": u}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    ic_xy, ic_u = _load_ic_file()
    plist = []
    total_steps = 0
    x_grid = jnp.linspace(X_MIN, X_MAX, NIC, endpoint=False, dtype=jnp.float32)
    nwin = len(BOUNDS) - 1
    for i in range(nwin):
        t_lo = float(BOUNDS[i]); t_hi = float(BOUNDS[i + 1])
        rng, wk = random.split(rng)
        W = _build_window(wk, t_lo, t_hi)
        wall_end = _T0 + sum(WALL_FRAC[:i + 1]) * TRAIN_TIME
        if i == nwin - 1:
            wall_end = _T0 + TRAIN_TIME - SAVE_MARGIN_S
        p, steps = _run_window(W, t_lo, t_hi, ic_xy, ic_u, wall_end)
        total_steps += steps
        plist.append(p)
        # hand off: this net's profile at t_hi becomes next window's IC
        if i < nwin - 1:
            ic_xy = jnp.stack([jnp.full((NIC,), t_hi, dtype=jnp.float32), x_grid], axis=1)
            ic_u = _PREDICT_MODEL.apply(p, ic_xy, t_lo, t_hi)[:, 0]

    return (tuple(float(b) for b in BOUNDS), plist), total_steps
