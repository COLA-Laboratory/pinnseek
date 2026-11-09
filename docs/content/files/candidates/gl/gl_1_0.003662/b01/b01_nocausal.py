"""Ablation of b01_hardic: identical hard-IC / exact-periodic / modified-MLP
PINN with the causal weighting switched off (EPS0 = 0, uniform slice weights).
Parent: b01_hardic.py — isolates how much of the gain comes from causality
versus from the hard constraints alone.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 2
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u", "v")

# --- Domain ---
T_MIN, T_MAX = 0.0, 0.79
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 2   # ru, rv
HAS_SOURCE              = False  # no forcing: every term carries u or v

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi")
PERIODIC_AXES  = ("x", "y")   # u(t, X_MIN, y) = u(t, X_MAX, y), likewise in y
GEOMETRY_KIND  = "periodic_box"

# --- Physical parameters ---
EPS = 0.004      # diffusivity, identical for u and v
K   = 10.0       # reaction rate
C   = 1.5        # nonlinear phase coefficient in -(1 + i c) |A|^2 A

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
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jit, random
import jax.flatten_util
import optax
from flax import linen as nn


# ══════════ architecture / training hyper-parameters ══════════

N_HARM   = 6      # Fourier harmonics per space axis (exact 2-periodicity)
WIDTH    = 128
DEPTH    = 5      # modified-MLP gated updates = DEPTH - 1
N_SLICE  = 64     # time slices per batch (causal ordering)
N_PT     = 192    # spatial collocation points per slice
LR0      = 1e-3
DECAY    = 0.95
DECAY_EVERY = 2000
EPS0     = 0.0    # ABLATION: causal weighting disabled (uniform slice weights)
EPS_MAX  = 0.0
CHECK_EVERY = 200
SAVE_MARGIN_S = 8.0


# ══════════ network: exact-periodic embedding + modified MLP ══════════

class PeriodicModifiedMLP(nn.Module):
    """(t,x,y) -> [t/T, cos/sin(m*pi*x), cos/sin(m*pi*y)] embedding (exactly
    2-periodic in x and y, so the periodic BCs hold to machine precision)
    -> Wang et al. modified MLP with two gating encoders U, V -> 2 outputs."""
    hidden: int = WIDTH
    depth: int = DEPTH
    n_harm: int = N_HARM

    @nn.compact
    def __call__(self, X):
        kinit = jax.nn.initializers.glorot_normal()
        t = X[:, 0:1] / T_MAX
        x = X[:, 1:2]
        y = X[:, 2:3]
        w = jnp.pi * jnp.arange(1, self.n_harm + 1, dtype=X.dtype)   # period 2
        h = jnp.concatenate(
            [t,
             jnp.cos(x * w), jnp.sin(x * w),
             jnp.cos(y * w), jnp.sin(y * w)], axis=1)
        act = nn.tanh
        U = act(nn.Dense(self.hidden, kernel_init=kinit)(h))
        V = act(nn.Dense(self.hidden, kernel_init=kinit)(h))
        H = act(nn.Dense(self.hidden, kernel_init=kinit)(h))
        for _ in range(self.depth - 1):
            Z = act(nn.Dense(self.hidden, kernel_init=kinit)(H))
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(H)


def _ic_field(x, y):
    """Analytic initial amplitude A0 = 10 (x + i y) exp(-25 (x^2+y^2))."""
    e = jnp.exp(-25.0 * (x ** 2 + y ** 2))
    return jnp.concatenate([10.0 * x * e, 10.0 * y * e], axis=1)


def _ansatz(model_apply, params, X):
    """Hard initial condition: A(t,x,y) = A0(x,y) + t * N(t,x,y)."""
    return _ic_field(X[:, 1:2], X[:, 2:3]) + X[:, 0:1] * model_apply(params, X)


# ══════════ training ══════════

def _make_train(rng_init):
    model = PeriodicModifiedMLP()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    # fixed, ordered time slices (causal weighting needs a stable order)
    t_slice = jnp.linspace(T_MIN, T_MAX, N_SLICE + 1)[1:]      # (0, T]

    def field(pf, Z):
        return _ansatz(model.apply, unravel(pf), Z)

    def slice_residuals(pf, XY):
        """XY: [N_SLICE, N_PT, 2] spatial samples -> [N_SLICE] mean sq residual."""
        tt = jnp.broadcast_to(t_slice[:, None, None], XY.shape[:2] + (1,))
        B = jnp.concatenate([tt, XY], axis=2).reshape(-1, INPUT_DIM)

        f = lambda Z: field(pf, Z)
        z = jnp.zeros_like(B)
        Et = z.at[:, 0].set(1.0)
        Ex = z.at[:, 1].set(1.0)
        Ey = z.at[:, 2].set(1.0)

        uv, d_t = jax.jvp(f, (B,), (Et,))
        _, d_xx = jax.jvp(lambda Z: jax.jvp(f, (Z,), (Ex,))[1], (B,), (Ex,))
        _, d_yy = jax.jvp(lambda Z: jax.jvp(f, (Z,), (Ey,))[1], (B,), (Ey,))

        u, v = uv[:, 0], uv[:, 1]
        m2 = u ** 2 + v ** 2
        r_u = d_t[:, 0] - EPS * (d_xx[:, 0] + d_yy[:, 0]) - K * (u - u * m2 + C * v * m2)
        r_v = d_t[:, 1] - EPS * (d_xx[:, 1] + d_yy[:, 1]) - K * (v - v * m2 - C * u * m2)
        return jnp.mean((r_u ** 2 + r_v ** 2).reshape(N_SLICE, N_PT), axis=1)

    def loss_fn(pf, XY, eps):
        Ls = slice_residuals(pf, XY)
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(Ls)[:-1]])
        w = jax.lax.stop_gradient(jnp.exp(-eps * cum))
        return jnp.mean(w * Ls), w

    sched = optax.exponential_decay(LR0, DECAY_EVERY, DECAY)
    optimizer = optax.adam(learning_rate=sched)
    opt_state = optimizer.init(flat_params)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jit
    def update(pf, opt_state, key, eps):
        XY = random.uniform(key, (N_SLICE, N_PT, 2), minval=X_MIN, maxval=X_MAX)
        (loss, w), grads = grad_fn(pf, XY, eps)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(pf, updates), opt_state, loss, jnp.min(w)

    return flat_params, opt_state, update, unravel, rng


# ══════════ public contract ══════════

_PREDICT_MODEL = PeriodicModifiedMLP()


def predict_fn(params, X):
    out = _ansatz(_PREDICT_MODEL.apply, params, X)
    return {"u": out[:, 0], "v": out[:, 1]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam on the causally weighted PDE residual (the only
    loss term: IC and periodic BC are hard constraints)."""
    params_flat, opt_state, update, unravel, rng_local = _make_train(rng)
    eps = EPS0

    key, rng_local = random.split(rng_local)
    params_flat, opt_state, loss, wmin = update(params_flat, opt_state, key, eps)
    step_count = 1
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        key, rng_local = random.split(rng_local)
        params_flat, opt_state, loss, wmin = update(params_flat, opt_state, key, eps)
        step_count = it + 1
        if it % CHECK_EVERY == 0:
            last_loss = float(loss)
            if float(wmin) > 0.99 and eps < EPS_MAX:
                eps *= 10.0
            if eval_callback is not None:
                eval_callback(unravel(params_flat), it, last_loss)
            if time.time() > deadline:
                break

    return unravel(params_flat), step_count
