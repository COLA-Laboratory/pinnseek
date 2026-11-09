"""Causal-KS PINN with SOAP optimizer, lower LR 8e-4 (retry of b01_soap).
Change vs parent (b01_causal_periodic): keep exact-periodic embed + modified
MLP + causal weighting, but optimise the param *pytree* with SOAP (Kronecker
preconditioner per weight matrix) instead of flat-vector Adam.
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
from jax.experimental.jet import jet
import numpy as np
import optax
from flax import linen as nn
from soap_jax import soap


def _locate(name: str) -> Path:
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Model: exact-periodic x-embed + modified (gated) MLP ══════════

K_X    = 6
HIDDEN = 128
DEPTH  = 4


class PINN(nn.Module):
    hidden: int = HIDDEN
    depth: int = DEPTH
    k_x: int = K_X

    @nn.compact
    def __call__(self, tx):
        t = tx[..., 0:1]
        x = tx[..., 1:2]
        tau = (t - T_MIN) / (T_MAX - T_MIN)
        ks = jnp.arange(1, self.k_x + 1, dtype=tx.dtype)
        ang = x * ks
        feat = jnp.concatenate([tau, jnp.cos(ang), jnp.sin(ang)], axis=-1)
        gi = nn.initializers.glorot_normal()
        U = nn.tanh(nn.Dense(self.hidden, kernel_init=gi)(feat))
        V = nn.tanh(nn.Dense(self.hidden, kernel_init=gi)(feat))
        h = nn.tanh(nn.Dense(self.hidden, kernel_init=gi)(feat))
        for _ in range(self.depth):
            z = nn.tanh(nn.Dense(self.hidden, kernel_init=gi)(h))
            h = z * U + (1.0 - z) * V
        return nn.Dense(OUTPUT_DIM, kernel_init=gi)(h)


N_T = 32
N_X = 128


def _build_ic():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0], ic[:, 1]
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic], axis=1).astype(np.float32)
    return jnp.array(pts_ic), jnp.array(u_ic.astype(np.float32))


LR_PEAK    = 8e-4
WARMUP     = 500
DECAY_STEP = 2000
DECAY_RATE = 0.9
LAM_IC     = 100.0
EPS_SCHED  = (1e-2, 1e-1, 1.0, 1e1, 1e2)
EPS_DELTA  = 0.99
SAVE_MARGIN_S = 8.0


def _make_train(rng_init):
    pts_ic, u_ic = _build_ic()
    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))

    def u_scalar(p, t, x):
        return model.apply(p, jnp.stack([t, x])[None, :])[0, 0]

    def residual_point(p, t, x):
        gx = lambda xx: u_scalar(p, t, xx)
        one, zero = jnp.ones_like(x), jnp.zeros_like(x)
        u, (u_x, u_xx, _u_xxx, u_xxxx) = jet(gx, (x,), ((one, zero, zero, zero),))
        u_t = grad(lambda tt: u_scalar(p, tt, x))(t)
        return u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx

    res_over_x = vmap(residual_point, in_axes=(None, None, 0))
    res_grid = vmap(res_over_x, in_axes=(None, 0, None))
    u_batch = vmap(u_scalar, in_axes=(None, 0, 0))
    t_slices = jnp.linspace(T_MIN, T_MAX, N_T + 1)[1:]

    def loss_fn(p, x_col, eps):
        r = res_grid(p, t_slices, x_col)
        L_r_t = jnp.mean(r ** 2, axis=1)
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(L_r_t)[:-1]])
        w = jax.lax.stop_gradient(jnp.exp(-eps * cum))
        pde_loss = jnp.mean(w * L_r_t)
        u_ic_pred = u_batch(p, pts_ic[:, 0], pts_ic[:, 1])
        ic_loss = jnp.mean((u_ic_pred - u_ic) ** 2)
        return LAM_IC * ic_loss + pde_loss, (w, ic_loss, jnp.mean(L_r_t))

    loss_and_grad = jit(jax.value_and_grad(loss_fn, has_aux=True))

    sched = optax.warmup_exponential_decay_schedule(
        init_value=0.0, peak_value=LR_PEAK, warmup_steps=WARMUP,
        transition_steps=DECAY_STEP, decay_rate=DECAY_RATE)
    optimizer = soap(learning_rate=sched, precondition_frequency=10)
    opt_state = optimizer.init(params)

    @jit
    def update(p, opt_state, k_x, eps):
        x_col = random.uniform(k_x, (N_X,), minval=X_MIN, maxval=X_MAX)
        (loss, aux), grads = loss_and_grad(p, x_col, eps)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss, aux

    return params, opt_state, update, rng


_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, opt_state, update, rng_local = _make_train(rng)
    eps_idx = 0
    eps = EPS_SCHED[eps_idx]

    rng_local, sub = random.split(rng_local)
    params, opt_state, loss, aux = update(params, opt_state, sub, eps)
    step_count = 1

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        rng_local, sub = random.split(rng_local)
        params, opt_state, loss, aux = update(params, opt_state, sub, eps)
        step_count = it + 1
        if it % 250 == 0:
            wmin = float(aux[0].min())
            if wmin > EPS_DELTA and eps_idx < len(EPS_SCHED) - 1:
                eps_idx += 1
                eps = EPS_SCHED[eps_idx]
            if eval_callback is not None:
                eval_callback(params, it, float(loss))

    return params, step_count
