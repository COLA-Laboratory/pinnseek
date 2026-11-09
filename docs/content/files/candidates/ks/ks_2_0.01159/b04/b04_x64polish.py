"""SPINN (kb1 070) + float32 Adam warmup to the plateau, then TOGGLE jax x64 and
run L-BFGS (kb1 074/089) on a dense fixed grid in float64 to push the PDE residual
below the float32 precision floor (~1e-5 MSE) that capped b03. Parent: b03_spinn.
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
# ── phase split & x64 L-BFGS phase ──
ADAM_FRAC = 0.60      # fraction of wall on float32 Adam warmup
NT_L      = 240       # x64 L-BFGS FIXED dense grid (NT_L*NX_L > #params → no overfit)
NX_L      = 480
LAM_L     = 100.0
LBFGS_MEM = 30
TMAX_START = 0.15
RAMP_FRAC  = 0.5


def _embed_x(x):
    ks = jnp.arange(1, MX + 1, dtype=x.dtype)
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


def _load_ic(dtype):
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    return (jnp.array(ic[:, 0].astype(np.float64), dtype=dtype).reshape(-1, 1),
            jnp.array(ic[:, 1].astype(np.float64), dtype=dtype))


# ── residual builders (dtype-agnostic; work in float32 or float64) ──

def _make_resid(model, unravel):
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
        return u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx

    def ic_loss(pf, x_ic_col, u_ic_v):
        g0 = g_of(pf, jnp.zeros((1, 1), x_ic_col.dtype))
        h = h_of(pf, x_ic_col)
        return jnp.mean((jnp.sum(g0 * h, axis=1) - u_ic_v) ** 2)

    return resid_field, ic_loss


_PREDICT_MODEL = SPINN()


def predict_fn(params, X):
    # params may be float64 (x64 polish) while the eval worker runs x64-off;
    # cast to X's dtype so apply is consistent regardless of precision mode.
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, X.dtype), params)
    return {"u": _PREDICT_MODEL.apply(params, X)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    # ─────────────── Phase 1: float32 Adam warmup to the plateau ───────────────
    x_ic, u_ic = _load_ic(jnp.float32)
    n_ic = len(x_ic)
    model = SPINN()
    key, rng = random.split(rng)
    params = model.init(key, jnp.zeros([1, INPUT_DIM], jnp.float32))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)
    resid_field, ic_loss = _make_resid(model, unravel)

    def slice_losses(pf, t_col, x_col):
        return jnp.mean(resid_field(pf, t_col, x_col) ** 2, axis=1)

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

    lam_ic = jnp.asarray(LAM_IC0, dtype=jnp.float32)
    eps_idx = 0
    eps = jnp.asarray(EPS_SCHED[eps_idx], dtype=jnp.float32)

    def tmax_at(now):
        prog = min(1.0, (now - _T0) / (RAMP_FRAC * TRAIN_TIME))
        return jnp.asarray(TMAX_START + (T_MAX - TMAX_START) * prog, dtype=jnp.float32)

    pf = flat_params
    tmax = tmax_at(time.time())
    pf, opt_state, rng, w, loss = step(pf, opt_state, rng, lam_ic, eps, tmax)
    step_count = 1

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

    # ─────────────── Phase 2: toggle x64, L-BFGS polish on dense grid ───────────────
    params32 = unravel(pf)                       # float32 pytree
    np_params = jax.tree_util.tree_map(np.asarray, params32)

    jax.config.update("jax_enable_x64", True)    # from here on, float64

    params64 = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), np_params)
    flat64, unravel64 = jax.flatten_util.ravel_pytree(params64)
    resid_field64, ic_loss64 = _make_resid(model, unravel64)

    x_ic64 = jnp.asarray(np.asarray(x_ic), jnp.float64)
    u_ic64 = jnp.asarray(np.asarray(u_ic), jnp.float64)

    tL = jnp.linspace(T_MIN + 1e-3, T_MAX, NT_L, dtype=jnp.float64).reshape(-1, 1)
    xL = jnp.linspace(X_MIN, X_MAX, NX_L, endpoint=False, dtype=jnp.float64).reshape(-1, 1)

    def lbfgs_loss(pf):
        Lr = jnp.mean(resid_field64(pf, tL, xL) ** 2)
        Li = ic_loss64(pf, x_ic64, u_ic64)
        return Lr + LAM_L * Li

    lbfgs = optax.lbfgs(memory_size=LBFGS_MEM)
    state = lbfgs.init(flat64)
    vg = optax.value_and_grad_from_state(lbfgs_loss)

    @jit
    def lstep(pf, state):
        v, g = vg(pf, state=state)
        up, state = lbfgs.update(g, state, pf, value=v, grad=g, value_fn=lbfgs_loss)
        return optax.apply_updates(pf, up), state, v

    best_pf = flat64
    pf2, state, v = lstep(flat64, state)         # compile (counts nothing critical)
    best_val = float(lbfgs_loss(flat64))
    if np.isfinite(float(v)):
        flat64 = pf2
        step_count += 1

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        pf_try, state, v = lstep(flat64, state)
        val = float(v)
        if not np.isfinite(val):
            break
        flat64 = pf_try
        if val < best_val:
            best_val = val
            best_pf = flat64
        step_count += 1
        if eval_callback is not None and (it % 500 == 0):
            eval_callback(unravel64(best_pf), step_count, val)

    return unravel64(best_pf), step_count
