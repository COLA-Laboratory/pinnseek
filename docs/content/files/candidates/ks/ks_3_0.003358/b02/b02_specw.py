"""Spectral-weighted-residual PINN for KS (fork: b01_pemb, rRMSE 0.276).
Change vs parent: evaluate the PDE residual on a structured x-grid, FFT it, and
weight each mode's residual power by w(k) that emphasises the resolvable KS
unstable band (k~6-15, growth sigma(k)=V2 k^2-V3 k^4>0, peak k~11) and strongly
down-weights the irreducible k>16 aliasing pollution (nonlinear u u_x generates
up to 2M). Directly targets the failure to excite the unstable modes (diag:
residual is high-k dominated, solution stuck in k1-3). kb1 073/086/095.
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


# ══════════ Periodic-embedding PINN ══════════

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


# ══════════ Spectral residual weighting ══════════

NX_GRID = 64          # x-grid points for FFT residual (resolves up to k=32)
NT_BATCH = 64         # random t-slices per step
_XGRID = jnp.linspace(X_MIN, X_MAX, NX_GRID, endpoint=False)

def _build_weights():
    kk = np.arange(NX_GRID // 2 + 1)
    sig = V2 * kk**2 - V3 * kk**4           # KS linear growth rate
    spk = sig.max()
    w = 0.2 + 1.0 * np.clip(sig, 0, None) / spk   # emphasise unstable band
    w[kk > 16] = 0.05                        # down-weight irreducible k>16 pollution
    w[0] = 0.2
    return jnp.asarray(w, jnp.float32)

_WK = _build_weights()
# rfft Parseval doubling of interior modes so w=1 -> mean(r^2)
_SCALE = np.ones(NX_GRID // 2 + 1); _SCALE[1:NX_GRID // 2] = 2.0
_SCALE = jnp.asarray(_SCALE, jnp.float32)


# ══════════ Hyperparameters ══════════

LR        = 1e-3
WEIGHT_IC = 100.0
SAVE_MARGIN_S = 6.0


def _make_train(rng_init):
    x_ic, u_ic = _load_ic()
    n_ic = len(x_ic)

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

    def resid_pt(pf, t, x):
        u      = u_scalar(pf, t, x)
        u_t    = u_t_fn(pf, t, x)
        u_x    = u_x_fn(pf, t, x)
        u_xx   = u_xx_fn(pf, t, x)
        u_xxxx = u_xxxx_fn(pf, t, x)
        return u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx

    # residual over one t-slice (all NX_GRID x's)
    resid_slice = vmap(resid_pt, in_axes=(None, None, 0))     # (t scalar, xs)
    resid_grid  = vmap(resid_slice, in_axes=(None, 0, None))  # (ts, xs) -> [Nt,Nx]
    u_batch = vmap(u_scalar, in_axes=(None, 0, 0))

    def loss_fn(pf, ts, ic_x, ic_u):
        r = resid_grid(pf, ts, _XGRID)                # [Nt, Nx]
        rhat = jnp.fft.rfft(r, axis=1) / NX_GRID       # [Nt, Nx//2+1]
        power = (rhat.real**2 + rhat.imag**2)
        pde = jnp.mean(jnp.sum(_WK * _SCALE * power, axis=1))
        u_ic_pred = u_batch(pf, jnp.zeros((n_ic,)), ic_x)
        ic_loss = jnp.mean((u_ic_pred - ic_u) ** 2)
        return pde + WEIGHT_IC * ic_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(pf, opt_state, key):
        ts = T_MAX * random.uniform(key, (NT_BATCH,))
        loss, grads = loss_and_grad(pf, ts, x_ic, u_ic)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(pf, updates), opt_state, loss

    return flat_params, opt_state, update, unravel, rng


_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    pf, opt_state, update, unravel, rng_local = _make_train(rng)

    rng_local, sub = random.split(rng_local)
    pf, opt_state, loss = update(pf, opt_state, sub)
    step_count = 1

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        rng_local, sub = random.split(rng_local)
        pf, opt_state, loss = update(pf, opt_state, sub)
        step_count = it + 1
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(unravel(pf), it, float(loss))

    return unravel(pf), step_count
