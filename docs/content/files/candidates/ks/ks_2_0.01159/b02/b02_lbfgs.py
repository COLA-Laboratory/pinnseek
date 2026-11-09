"""b01_combo champion + short L-BFGS polish (kb1 074/089) on a SMALL fixed grid with cheap backtracking line search.
Differs from b01's failed L-BFGS: MX=16 warmup-cosine Adam base, tiny 2048-pt
L-BFGS grid, backtracking (not zoom) line search. Parent: b01_combo.
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
BS_IC     = 512
LR0       = 1.8e-3
EPS_SCHED = (1e-2, 1e-1, 1e0, 1e1, 1e2)
W_DELTA   = 0.99
LAM_IC0   = 100.0
LAM_ALPHA = 0.9
LAM_EVERY = 500
SAVE_MARGIN_S = 6.0
ADAM_FRAC = 0.60      # fraction of TRAIN_TIME in the Adam phase, then L-BFGS
NT_L      = 32        # fixed time slices for the L-BFGS phase (small → cheap iters)
NX_L      = 64        # fixed spatial points per slice


_BT = np.asarray(
    random.normal(random.PRNGKey(12345), (MT,)) * SIGMA_T, dtype=np.float32)


def _embed(tx):
    t = tx[:, 0:1]
    x = tx[:, 1:2]
    ts = (t - T_MIN) / (T_MAX - T_MIN)
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
    def __call__(self, tx):
        kinit = jax.nn.initializers.glorot_normal()
        h0 = _embed(tx)
        U = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h0))
        V = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h0))
        H = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h0))
        for _ in range(self.depth):
            Z = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(H))
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(H)


def _load_ic():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0], ic[:, 1]
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic], axis=1).astype(np.float32)
    return jnp.array(pts_ic), jnp.array(u_ic.astype(np.float32))


def _make_train(rng_init):
    pts_ic, u_ic = _load_ic()
    n_ic = len(pts_ic)

    model = ModifiedMLP()
    key, rng = random.split(rng_init)
    dummy = jnp.zeros([1, INPUT_DIM])
    params = model.init(key, dummy)
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def u_scalar(pf, t, x):
        return model.apply(unravel(pf), jnp.stack([t, x])[None, :])[0, 0]

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
        init_value=0.0, peak_value=LR0, warmup_steps=500,
        decay_steps=18000, end_value=1e-5)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(flat_params)

    loss_grad = jit(jax.value_and_grad(total_loss, has_aux=True))

    def _lam_update(pf, t_grid, x_grid, ic_xy, ic_u, eps):
        gr = jax.grad(lambda p: residual_loss(p, t_grid, x_grid, eps)[0])(pf)
        gi = jax.grad(lambda p: ic_loss(p, ic_xy, ic_u))(pf)
        return jnp.max(jnp.abs(gr)) / (jnp.mean(jnp.abs(gi)) + 1e-12)
    lam_update = jit(_lam_update)

    @jit
    def sample(rng, tmax):
        r1, r2, r3 = random.split(rng, 3)
        ts = jnp.sort(random.uniform(r1, (NT,), minval=1e-4, maxval=tmax))
        xs = random.uniform(r2, (NX,), minval=X_MIN, maxval=X_MAX)
        ic_idx = random.choice(r3, n_ic, (BS_IC,), replace=False)
        return ts, xs, ic_idx

    @jit
    def step(pf, opt_state, rng, lam_ic, eps, tmax):
        rng, sk = random.split(rng)
        t_grid, x_grid, ic_idx = sample(sk, tmax)
        (loss, (w, Lr, Li)), g = loss_grad(
            pf, t_grid, x_grid, pts_ic[ic_idx], u_ic[ic_idx], lam_ic, eps)
        updates, opt_state = optimizer.update(g, opt_state)
        pf = optax.apply_updates(pf, updates)
        return pf, opt_state, rng, w, loss

    def lam_step(pf, rng, eps, tmax):
        rng, sk = random.split(rng)
        t_grid, x_grid, ic_idx = sample(sk, tmax)
        lam = lam_update(pf, t_grid, x_grid, pts_ic[ic_idx], u_ic[ic_idx], eps)
        return lam, rng

    return (flat_params, opt_state, step, lam_step, unravel, rng,
            residual_loss, ic_loss, pts_ic, u_ic)


_PREDICT_MODEL = ModifiedMLP()


def predict_fn(params, X):
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0]}


TMAX_START = 0.15
RAMP_FRAC  = 0.55


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    (pf, opt_state, step, lam_step, unravel, rng,
     residual_loss, ic_loss, pts_ic, u_ic) = _make_train(rng)

    lam_ic = jnp.asarray(LAM_IC0, dtype=jnp.float32)
    eps_idx = 0
    eps = jnp.asarray(EPS_SCHED[eps_idx], dtype=jnp.float32)

    def tmax_at(now):
        prog = min(1.0, (now - _T0) / (RAMP_FRAC * TRAIN_TIME))
        return jnp.asarray(TMAX_START + (T_MAX - TMAX_START) * prog, dtype=jnp.float32)

    # warmup (JIT compile), counts as a step
    tmax = tmax_at(time.time())
    pf, opt_state, rng, w, loss = step(pf, opt_state, rng, lam_ic, eps, tmax)
    step_count = 1

    # ── Phase 1: Adam (causal, resampled, curriculum) ──
    adam_deadline = _T0 + ADAM_FRAC * TRAIN_TIME
    for it in range(1, MAX_ITER):
        now = time.time()
        if now > adam_deadline:
            break
        tmax = tmax_at(now)
        pf, opt_state, rng, w, loss = step(pf, opt_state, rng, lam_ic, eps, tmax)
        step_count = it + 1
        if it % LAM_EVERY == 0:
            new_lam, rng = lam_step(pf, rng, eps, tmax)
            new_lam = jnp.clip(new_lam, 1.0, 1e5)
            lam_ic = LAM_ALPHA * lam_ic + (1.0 - LAM_ALPHA) * new_lam
            if float(jnp.min(w)) > W_DELTA and eps_idx < len(EPS_SCHED) - 1:
                eps_idx += 1
                eps = jnp.asarray(EPS_SCHED[eps_idx], dtype=jnp.float32)

    # ── Phase 2: full-batch L-BFGS on a small fixed grid, full horizon ──
    lam_f = float(lam_ic)
    eps_f = 1e2   # effectively uniform weighting at convergence
    t_gridL = jnp.linspace(1e-4, T_MAX, NT_L)
    x_gridL = jnp.linspace(X_MIN, X_MAX, NX_L, endpoint=False)

    def loss_L(p):
        Lr, _ = residual_loss(p, t_gridL, x_gridL, eps_f)
        Li = ic_loss(p, pts_ic, u_ic)
        return Lr + lam_f * Li

    linesearch = optax.scale_by_backtracking_linesearch(
        max_backtracking_steps=12, store_grad=True)
    solver = optax.lbfgs(learning_rate=None, memory_size=10, linesearch=linesearch)
    lstate = solver.init(pf)
    value_and_grad = optax.value_and_grad_from_state(loss_L)

    @jit
    def lbfgs_step(p, lstate):
        v, g = value_and_grad(p, state=lstate)
        updates, lstate = solver.update(
            g, lstate, p, value=v, grad=g, value_fn=loss_L)
        p = optax.apply_updates(p, updates)
        return p, lstate, v

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    pf, lstate, v = lbfgs_step(pf, lstate)   # compile
    step_count += 1
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        pf, lstate, v = lbfgs_step(pf, lstate)
        step_count += 1

    return unravel(pf), step_count
