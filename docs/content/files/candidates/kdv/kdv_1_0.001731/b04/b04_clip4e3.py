"""Separable causal KdV PINN. Parent: b03_w20.py (0.0226). Only change: global
grad-norm clipping at 1.0 in front of Adam, with the peak LR raised 2e-3 ->
4e-3. b03 rejected every mechanism at an LR already on the stability edge; the
clip is meant to buy headroom above it.
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
T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = -1.0, 1.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 3
N_RESIDUAL_COMPONENTS   = 1   # r
HAS_SOURCE              = False  # no forcing: every term carries u

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi")
PERIODIC_AXES  = ("x",)   # u(t, X_MIN) = u(t, X_MAX)
GEOMETRY_KIND  = "periodic_box"

# --- Physical parameters ---
DELTA = 0.022    # dispersion coefficient; the PDE carries delta^2 u_xxx

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 100.0     # total wall-clock seconds for the WHOLE process,
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
from jax import jit, lax, random
import numpy as np
import optax
from flax import linen as nn


# ══════════ Architecture: two per-axis MLPs, rank-R sum ══════════

RANK     = 128        # CP rank of the separable ansatz
WIDTH    = 128        # hidden width of each axis MLP
T_DEPTH  = 3          # hidden layers in the t-net (1 JVP level: cheap)
X_DEPTH  = 1          # hidden layers in the x-net (3 nested JVPs: 4x dearer)
N_MODES  = 32         # Fourier modes k = 1..N_MODES in the x embedding

T_HI     = 1.0        # residual lattice reaches past the scored T_MAX
NT, NX   = 192, 384   # residual lattice: time slices x space points
DT_SLICE = (T_HI - T_MIN) / NT        # eps_eff = EPS_I*DT_SLICE = 6.67
INNER    = 100        # optimiser steps per jitted chunk
LR_INIT  = 4e-3
CLIP     = 1.0        # global grad-norm clip in front of Adam
DECAY_R  = 0.9
DECAY_S  = 7000
EPS_I     = 1280.0    # dt-invariant causality: EPS_I * DT_SLICE = 10 (parent)
W_MASS   = 20.0
W_ENER   = 0.0    # weight on the int u^2 conservation penalty      # weight on the mass-conservation penalty
SAVE_MARGIN_S = 6.0


class AxisNet(nn.Module):
    """1-D MLP: features -> R^RANK (tanh, He-uniform)."""
    width: int = WIDTH
    depth: int = 3
    rank: int = RANK

    @nn.compact
    def __call__(self, h):
        kinit = jax.nn.initializers.he_uniform()
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h))
        return nn.Dense(self.rank, kernel_init=kinit)(h)


class SepPINN(nn.Module):
    """u(t,x) = cos(pi x) + t * sum_j f_j(t) g_j(x); g is built on an
    exactly-periodic Fourier embedding of x (period 2), so u and all its
    x-derivatives are periodic to machine precision (kb1 012)."""
    n_modes: int = N_MODES

    def setup(self):
        self.tnet = AxisNet(depth=T_DEPTH, name="tnet")
        self.xnet = AxisNet(depth=X_DEPTH, name="xnet")

    def t_feats(self, t):                       # t: [N, 1] in [0, T_HI]
        return self.tnet(2.0 * t - 1.0)         # [N, RANK]

    def x_feats(self, x):                       # x: [N, 1] in [-1, 1]
        k = jnp.arange(1, self.n_modes + 1, dtype=x.dtype)
        ang = jnp.pi * x * k                    # [N, n_modes]
        return self.xnet(jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=-1))

    def __call__(self, t, x):
        return self.t_feats(t), self.x_feats(x)


_MODEL = SepPINN()


def _f_of_t(params, t):
    return _MODEL.apply(params, t, method=SepPINN.t_feats)


def _g_of_x(params, x):
    return _MODEL.apply(params, x, method=SepPINN.x_feats)


# ══════════ Residual on a factorised lattice (forward-mode AD) ══════════

def _residual_grid(params, t, x):
    """r(t_i, x_j) on the [NT, NX] lattice. All derivatives come from
    autodiff of the networks (nested JVPs along each axis)."""
    ones_t = jnp.ones_like(t)
    f, f_t = jax.jvp(lambda tt: _f_of_t(params, tt), (t,), (ones_t,))

    ones_x = jnp.ones_like(x)

    def g0(xx):
        return _g_of_x(params, xx)

    def g01(xx):
        return jax.jvp(g0, (xx,), (ones_x,))            # (g, g')

    def g012(xx):
        (g, g1), (_, g2) = jax.jvp(g01, (xx,), (ones_x,))
        return g, g1, g2

    (g, g1, g2), (_, _, g3) = jax.jvp(g012, (x,), (ones_x,))

    S    = f   @ g.T          # [NT, NX]
    S_t  = f_t @ g.T
    S_x  = f   @ g1.T
    S_3x = f   @ g3.T

    xr    = x.T               # [1, NX]
    u0    = jnp.cos(jnp.pi * xr)
    u0_x  = -jnp.pi * jnp.sin(jnp.pi * xr)
    u0_3x = jnp.pi ** 3 * jnp.sin(jnp.pi * xr)

    u     = u0 + t * S
    u_t   = S + t * S_t
    u_x   = u0_x + t * S_x
    u_3x  = u0_3x + t * S_3x
    # KdV invariants, referenced to the SAME lattice quadrature of the
    # initial condition so the stratified-sampling quadrature noise cancels
    # exactly: int u dx and int u^2 dx are conserved, both equal to their
    # t = 0 values.
    mass = jnp.mean(u, axis=1) - jnp.mean(u0)
    ener = jnp.mean(u ** 2, axis=1) - jnp.mean(u0 ** 2)
    return u_t + u * u_x + DELTA ** 2 * u_3x, mass, ener


def _loss_fn(params, t, x):
    r, mass, ener = _residual_grid(params, t, x)
    L = jnp.mean(r ** 2, axis=1)                        # per time slice
    cum = jnp.concatenate([jnp.zeros((1,), L.dtype), jnp.cumsum(L)[:-1]])
    # dt-invariant causal weight (kb1 084): exponent approximates the
    # integral eps * int_0^t L(s) ds, so EPS_I no longer rescales with NT.
    w = lax.stop_gradient(jnp.exp(-EPS_I * DT_SLICE * cum))
    return (jnp.mean(w * L) + W_MASS * jnp.mean(mass ** 2)
            + W_ENER * jnp.mean(ener ** 2),
            (jnp.mean(L), jnp.min(w)))


def _strat(key, n, lo, hi):
    """Stratified (jittered) 1-D sample, sorted ascending."""
    u = random.uniform(key, (n,))
    return (lo + (hi - lo) * (jnp.arange(n) + u) / n)[:, None]


# ══════════ Public contract ══════════

def predict_fn(params, X):
    """X: [BS, 2] with columns (t, x). Returns {"u": [BS]}."""
    t = X[:, 0:1]
    x = X[:, 1:2]
    f = _f_of_t(params, t)
    g = _g_of_x(params, x)
    S = jnp.sum(f * g, axis=-1)
    return {"u": jnp.cos(jnp.pi * X[:, 1]) + X[:, 0] * S}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam on the causally weighted residual loss."""
    key, sub = random.split(rng)
    params = _MODEL.init(sub, jnp.zeros((1, 1)), jnp.zeros((1, 1)))

    sched = optax.exponential_decay(LR_INIT, DECAY_S, DECAY_R)
    optimizer = optax.chain(optax.clip_by_global_norm(CLIP),
                            optax.adam(learning_rate=sched))
    opt_state = optimizer.init(params)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

    @jit
    def chunk(params, opt_state, key):
        def body(_, carry):
            params, opt_state, key, _, _ = carry
            key, k1, k2 = random.split(key, 3)
            t = _strat(k1, NT, T_MIN, T_HI)
            x = _strat(k2, NX, X_MIN, X_MAX)
            (_, (mean_L, min_w)), grads = grad_fn(params, t, x)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, key, mean_L, min_w

        return lax.fori_loop(0, INNER, body,
                             (params, opt_state, key, jnp.float32(0.0),
                              jnp.float32(0.0)))

    # Warmup: compiles the whole chunk; the steps inside are real steps.
    params, opt_state, key, mean_L, min_w = chunk(params, opt_state, key)
    step_count = INNER

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    while step_count < MAX_ITER and time.time() < deadline:
        params, opt_state, key, mean_L, min_w = chunk(params, opt_state, key)
        step_count += INNER
        if eval_callback is not None and (step_count % 1000 == 0):
            eval_callback(params, step_count, float(mean_L))

    return params, step_count
