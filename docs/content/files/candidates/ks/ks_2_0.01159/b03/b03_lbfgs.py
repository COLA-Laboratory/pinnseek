"""SPINN + Adam warmup → optax L-BFGS (zoom line search) on a LARGE FIXED grid.
SPINN's cheap 4th-order residual lets L-BFGS run many full-batch iters on a dense grid
(NT128×NX256), avoiding b02's small-grid overfit — the proper test of kb1 074/089
(quasi-Newton cracks KS). Parent: b03_spinn.
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
from jax import jit, random
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
RANK      = 128
WIDTH     = 96
DEPTH     = 3
NT        = 64        # Adam-phase grid (resampled)
NX        = 128
BS_IC     = 512
LR0       = 1.8e-3
DECAY_STEPS = 30000
EPS_SCHED = (1e-2, 1e-1, 1e0, 1e1, 1e2)
W_DELTA   = 0.99
W_FLOOR   = 0.03
LAM_IC0   = 100.0
LAM_ALPHA = 0.9
LAM_EVERY = 500
SAVE_MARGIN_S = 8.0
# ── L-BFGS phase ──
ADAM_FRAC = 0.5       # fraction of wall on Adam warmup; rest on L-BFGS
NT_L      = 128       # L-BFGS FIXED grid
NX_L      = 256
LAM_L     = 100.0     # IC weight in L-BFGS loss
LBFGS_MEM = 20


def _embed_x(x):
    ks = jnp.arange(1, MX + 1, dtype=jnp.float32)
    return jnp.concatenate([jnp.cos(x * ks), jnp.sin(x * ks)], axis=1)


def _embed_t(t):
    return (t - T_MIN) / (T_MAX - T_MIN)


class AxisBody(nn.Module):
    embed: Callable
    width: int = WIDTH
    depth: int = DEPTH
    rank: int = RANK

    @nn.compact
    def __call__(self, z):
        kinit = jax.nn.initializers.glorot_normal()
        h0 = self.embed(z)
        U = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h0))
        V = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h0))
        H = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h0))
        for _ in range(self.depth):
            Z = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(H))
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(self.rank, kernel_init=kinit)(H)


class SPINN(nn.Module):
    def setup(self):
        self.G = AxisBody(_embed_t, name="G")
        self.H = AxisBody(_embed_x, name="H")

    def __call__(self, X):
        g = self.G(X[:, 0:1]); h = self.H(X[:, 1:2])
        return jnp.sum(g * h, axis=1)

    def g_feats(self, t):
        return self.G(t)

    def h_feats(self, x):
        return self.H(x)


def _load_ic():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0], ic[:, 1]
    return (jnp.array(x_ic.astype(np.float32)).reshape(-1, 1),
            jnp.array(u_ic.astype(np.float32)))


def _make_train(rng_init):
    x_ic, u_ic = _load_ic()
    n_ic = len(x_ic)

    model = SPINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def g_of(pf, t_col):
        return model.apply(unravel(pf), t_col, method=SPINN.g_feats)

    def h_of(pf, x_col):
        return model.apply(unravel(pf), x_col, method=SPINN.h_feats)

    def _d(f):
        return lambda col: jax.jvp(f, (col,), (jnp.ones_like(col),))[1]

    def resid_field(pf, t_col, x_col):
        gf = lambda tc: g_of(pf, tc); hf = lambda xc: h_of(pf, xc)
        g = gf(t_col); gt = _d(gf)(t_col); h = hf(x_col)
        hx = _d(hf); hxx = _d(hx); hxxx = _d(hxx); hxxxx = _d(hxxx)
        h1 = hx(x_col); h2 = hxx(x_col); h4 = hxxxx(x_col)
        u    = jnp.einsum('tr,xr->tx', g, h)
        u_t  = jnp.einsum('tr,xr->tx', gt, h)
        u_x  = jnp.einsum('tr,xr->tx', g, h1)
        u_xx = jnp.einsum('tr,xr->tx', g, h2)
        u_xxxx = jnp.einsum('tr,xr->tx', g, h4)
        return u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx   # [NT,NX]

    def slice_losses(pf, t_col, x_col):
        return jnp.mean(resid_field(pf, t_col, x_col) ** 2, axis=1)

    def ic_loss(pf, x_ic_col, u_ic_v):
        g0 = g_of(pf, jnp.zeros((1, 1), jnp.float32))
        h = h_of(pf, x_ic_col)
        return jnp.mean((jnp.sum(g0 * h, axis=1) - u_ic_v) ** 2)

    def residual_loss(pf, t_col, x_col, eps):
        Lr = slice_losses(pf, t_col, x_col)
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(Lr)[:-1]])
        w = jax.lax.stop_gradient(jnp.maximum(jnp.exp(-eps * cum), W_FLOOR))
        return jnp.mean(w * Lr), w

    def total_loss(pf, t_col, x_col, x_ic_col, u_ic_v, lam_ic, eps):
        Lr, w = residual_loss(pf, t_col, x_col, eps)
        Li = ic_loss(pf, x_ic_col, u_ic_v)
        return Lr + lam_ic * Li, (w, Lr, Li)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR0, warmup_steps=500,
        decay_steps=DECAY_STEPS, end_value=1e-5)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(flat_params)
    loss_grad = jit(jax.value_and_grad(total_loss, has_aux=True))

    def _lam_update(pf, t_col, x_col, x_ic_col, u_ic_v, eps):
        gr = jax.grad(lambda p: residual_loss(p, t_col, x_col, eps)[0])(pf)
        gi = jax.grad(lambda p: ic_loss(p, x_ic_col, u_ic_v))(pf)
        return jnp.max(jnp.abs(gr)) / (jnp.mean(jnp.abs(gi)) + 1e-12)
    lam_update = jit(_lam_update)

    @jit
    def sample(rng, tmax):
        r1, r2, r3 = random.split(rng, 3)
        ts = jnp.sort(random.uniform(r1, (NT, 1), minval=1e-4, maxval=tmax))
        xs = random.uniform(r2, (NX, 1), minval=X_MIN, maxval=X_MAX)
        ic_idx = random.choice(r3, n_ic, (BS_IC,), replace=False)
        return ts, xs, ic_idx

    @jit
    def step(pf, opt_state, rng, lam_ic, eps, tmax):
        rng, sk = random.split(rng)
        t_col, x_col, ic_idx = sample(sk, tmax)
        (loss, (w, Lr, Li)), g = loss_grad(
            pf, t_col, x_col, x_ic[ic_idx], u_ic[ic_idx], lam_ic, eps)
        updates, opt_state = optimizer.update(g, opt_state)
        pf = optax.apply_updates(pf, updates)
        return pf, opt_state, rng, w, loss

    def lam_step(pf, rng, eps, tmax):
        rng, sk = random.split(rng)
        t_col, x_col, ic_idx = sample(sk, tmax)
        lam = lam_update(pf, t_col, x_col, x_ic[ic_idx], u_ic[ic_idx], eps)
        return lam, rng

    # ── L-BFGS on a FIXED dense grid ──
    tL = jnp.linspace(T_MIN + 1e-3, T_MAX, NT_L).reshape(-1, 1)
    xL = jnp.linspace(X_MIN, X_MAX, NX_L, endpoint=False).reshape(-1, 1)

    def lbfgs_loss(pf):
        Lr = jnp.mean(resid_field(pf, tL, xL) ** 2)
        Li = ic_loss(pf, x_ic, u_ic)
        return Lr + LAM_L * Li

    lbfgs = optax.lbfgs(memory_size=LBFGS_MEM)
    vg_state = optax.value_and_grad_from_state(lbfgs_loss)

    @jit
    def lbfgs_step(pf, state):
        v, g = vg_state(pf, state=state)
        updates, state = lbfgs.update(g, state, pf, value=v, grad=g, value_fn=lbfgs_loss)
        pf = optax.apply_updates(pf, updates)
        return pf, state, v

    return (flat_params, opt_state, step, lam_step, unravel, rng,
            lbfgs, lbfgs_step, lbfgs_loss)


_PREDICT_MODEL = SPINN()


def predict_fn(params, X):
    return {"u": _PREDICT_MODEL.apply(params, X)}


TMAX_START = 0.15
RAMP_FRAC  = 0.5


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    (pf, opt_state, step, lam_step, unravel, rng,
     lbfgs, lbfgs_step, lbfgs_loss) = _make_train(rng)

    lam_ic = jnp.asarray(LAM_IC0, dtype=jnp.float32)
    eps_idx = 0
    eps = jnp.asarray(EPS_SCHED[eps_idx], dtype=jnp.float32)

    def tmax_at(now):
        prog = min(1.0, (now - _T0) / (RAMP_FRAC * TRAIN_TIME))
        return jnp.asarray(TMAX_START + (T_MAX - TMAX_START) * prog, dtype=jnp.float32)

    tmax = tmax_at(time.time())
    pf, opt_state, rng, w, loss = step(pf, opt_state, rng, lam_ic, eps, tmax)
    step_count = 1

    # ── Phase 1: Adam ──
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

    # ── Phase 2: L-BFGS on fixed dense grid ──
    best_pf = pf
    state = lbfgs.init(pf)
    pf2, state, v = lbfgs_step(pf, state)   # compile
    best_val = float(lbfgs_loss(pf))
    step_count += 1
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    if np.isfinite(float(v)):
        pf = pf2
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        pf_try, state, v = lbfgs_step(pf, state)
        val = float(v)
        if not np.isfinite(val):
            break
        pf = pf_try
        if val < best_val:
            best_val = val
            best_pf = pf
        step_count += 1

    return unravel(best_pf), step_count
