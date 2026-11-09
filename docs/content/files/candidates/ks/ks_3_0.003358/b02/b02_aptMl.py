"""Adaptive pseudo-time (Adam) -> LBFGS PINN for KS (fork: b02_apt, rRMSE 0.2734).
Change vs parent: keep the adaptive pseudo-time + continuous-resampling Adam phase
(kb1 130) but finish with a full-batch LBFGS refinement on the plain residual
(kb1 074/089) to sharpen convergence -- M=20; combines the two best b02 levers
(apt 0.2734, lbfgs 0.2746). Exact periodic embedding + strong IC.
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


M_MODES = 20


class PINN(nn.Module):
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, tx):
        t = tx[:, 0:1] / T_MAX
        x = tx[:, 1:2]
        ks = jnp.arange(1, M_MODES + 1, dtype=tx.dtype)
        xk = x * ks
        feat = jnp.concatenate([t, jnp.cos(xk), jnp.sin(xk)], axis=-1)
        kinit = jax.nn.initializers.glorot_normal()
        h = feat
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)


def _load_ic():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0], ic[:, 1]
    return (jnp.array(x_ic.astype(np.float32)), jnp.array(u_ic.astype(np.float32)))


LR        = 1e-3
WEIGHT_IC = 100.0
BS_COL    = 2000
ADAM_FRAC = 0.6
SAVE_MARGIN_S = 8.0
NT_FULL, NX_FULL = 80, 200

TAU0     = 1.0
INV_TAU_MAX = 1.5
INV_TAU_MIN = 0.1
BETA_EMA = 0.5
M_UPDATE = 100
EPS      = 1e-8
GAMMA_MIN = 0.1
S_START, S_END = 2.0, 6.0


def _make(rng_init):
    x_ic, u_ic = _load_ic()
    n_ic = len(x_ic)

    ts = np.linspace(T_MIN, T_MAX, NT_FULL + 1)[1:]
    xs = np.linspace(X_MIN, X_MAX, NX_FULL, endpoint=False)
    Tg, Xg = np.meshgrid(ts, xs, indexing="ij")
    col_full_t = jnp.asarray(Tg.ravel().astype(np.float32))
    col_full_x = jnp.asarray(Xg.ravel().astype(np.float32))

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def u_scalar(pf, t, x):
        return model.apply(unravel(pf), jnp.stack([t, x])[None, :])[0, 0]

    u_t_fn    = grad(u_scalar, argnums=1)
    u_x_fn    = grad(u_scalar, argnums=2)
    u_xx_fn   = grad(u_x_fn,  argnums=2)
    u_xxx_fn  = grad(u_xx_fn, argnums=2)
    u_xxxx_fn = grad(u_xxx_fn, argnums=2)

    def resid_raw(pf, t, x):
        u      = u_scalar(pf, t, x)
        u_t    = u_t_fn(pf, t, x)
        u_x    = u_x_fn(pf, t, x)
        u_xx   = u_xx_fn(pf, t, x)
        u_xxxx = u_xxxx_fn(pf, t, x)
        return u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx

    resid_batch = vmap(resid_raw, in_axes=(None, 0, 0))
    u_batch = vmap(u_scalar, in_axes=(None, 0, 0))

    # pseudo-time Adam loss (relaxed residual)
    def loss_pt(pf, prev, col_t, col_x, inv_tau):
        r = resid_batch(pf, col_t, col_x)
        u_cur  = u_batch(pf, col_t, col_x)
        u_prev = u_batch(prev, col_t, col_x)
        relaxed = inv_tau * (u_cur - u_prev) + r
        pde = jnp.mean(relaxed ** 2)
        u_ic_pred = u_batch(pf, jnp.zeros((n_ic,)), x_ic)
        ic_loss = jnp.mean((u_ic_pred - u_ic) ** 2)
        return pde + WEIGHT_IC * ic_loss
    lg_pt = jit(jax.value_and_grad(loss_pt))
    adam = optax.adam(LR)

    @jit
    def adam_step(pf, prev, opt_state, key, inv_tau):
        k_t, k_x = random.split(key)
        col_t = T_MAX * random.uniform(k_t, (BS_COL,))
        col_x = X_MAX * random.uniform(k_x, (BS_COL,))
        loss, grads = lg_pt(pf, prev, col_t, col_x, inv_tau)
        updates, opt_state = adam.update(grads, opt_state)
        return optax.apply_updates(pf, updates), opt_state, loss

    @jit
    def tau_stats(pf, prev, col_t, col_x):
        du = u_batch(pf, col_t, col_x) - u_batch(prev, col_t, col_x)
        dr = resid_batch(pf, col_t, col_x) - resid_batch(prev, col_t, col_x)
        r2 = jnp.mean(resid_batch(pf, col_t, col_x) ** 2)
        return jnp.linalg.norm(du), jnp.linalg.norm(dr), r2

    @jit
    def sample(key):
        k_t, k_x = random.split(key)
        return T_MAX * random.uniform(k_t, (BS_COL,)), X_MAX * random.uniform(k_x, (BS_COL,))

    # plain-residual full-batch loss for LBFGS
    def loss_full(pf):
        pde = jnp.mean(resid_batch(pf, col_full_t, col_full_x) ** 2)
        u_ic_pred = u_batch(pf, jnp.zeros((n_ic,)), x_ic)
        ic_loss = jnp.mean((u_ic_pred - u_ic) ** 2)
        return pde + WEIGHT_IC * ic_loss
    lbfgs = optax.lbfgs()
    vg_state = optax.value_and_grad_from_state(loss_full)

    @jit
    def lbfgs_step(pf, opt_state):
        value, grads = vg_state(pf, state=opt_state)
        updates, opt_state = lbfgs.update(grads, opt_state, pf,
                                          value=value, grad=grads, value_fn=loss_full)
        return optax.apply_updates(pf, updates), opt_state, value

    return (flat_params, unravel, adam, adam_step, tau_stats, sample,
            lbfgs, lbfgs_step, rng)


_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    (pf, unravel, adam, adam_step, tau_stats, sample,
     lbfgs, lbfgs_step, rng_local) = _make(rng)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    adam_deadline = _T0 + ADAM_FRAC * TRAIN_TIME

    prev = pf
    inv_tau = float(1.0 / TAU0)
    L0 = None
    opt_state = adam.init(pf)

    rng_local, sub = random.split(rng_local)
    col_t, col_x = sample(sub)
    new, opt_state, loss = adam_step(pf, prev, opt_state, sub, jnp.asarray(inv_tau, jnp.float32))
    prev, pf = pf, new
    step_count = 1
    it = 1
    while time.time() < adam_deadline:
        rng_local, sub = random.split(rng_local)
        new, opt_state, loss = adam_step(pf, prev, opt_state, sub, jnp.asarray(inv_tau, jnp.float32))
        prev, pf = pf, new
        step_count += 1; it += 1
        if it % M_UPDATE == 0:
            col_t, col_x = sample(sub)
            du, dr, r2 = tau_stats(pf, prev, col_t, col_x)
            du = float(du); dr = float(dr); r2 = float(r2)
            if L0 is None:
                L0 = r2 + EPS
            p = (np.log10(L0 / (r2 + EPS)) - S_START) / (S_END - S_START)
            p = min(1.0, max(0.0, p))
            gamma = GAMMA_MIN + (1.0 - GAMMA_MIN) * 0.5 * (1.0 + np.cos(np.pi * p))
            tau_hat = gamma * du / (dr + EPS)
            inv_tau = (1.0 - BETA_EMA) * inv_tau + BETA_EMA * (1.0 / max(tau_hat, 1e-6))
            inv_tau = min(INV_TAU_MAX, max(INV_TAU_MIN, inv_tau))

    # LBFGS refinement on plain residual
    lstate = lbfgs.init(pf)
    while time.time() < deadline:
        pf, lstate, val = lbfgs_step(pf, lstate)
        step_count += 1
        if eval_callback is not None and (step_count % 2000 == 0):
            eval_callback(unravel(pf), step_count, float(val))

    return unravel(pf), step_count
