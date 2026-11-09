"""Capacity sweep of b01_spinn: identical separable PINN with CP rank 256.
Parent: b01_spinn.py — RANK 128 -> 256, everything else unchanged; tests
whether the rank-R outer product, not the optimiser, limits accuracy.
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

RANK      = 256   # CP rank of the separable ansatz
AX_WIDTH  = 128
AX_DEPTH  = 4     # modified-MLP gated updates = AX_DEPTH - 1
N_HARM    = 8     # Fourier harmonics on x and y (exact 2-periodicity)
T_HARM    = 4     # smooth feature bank on the (non-periodic) time axis
NT        = 64    # time slices in the collocation lattice (causal order)
NXY       = 128   # random points per spatial axis, resampled every step
LR0       = 1e-3
DECAY     = 0.9
DECAY_EVERY = 5000
EPS0      = 1e-3  # causal tolerance, ramped x10 while min(w) > 0.99
EPS_MAX   = 1e4
CHECK_EVERY = 200
PRED_CHUNK  = 65536
SAVE_MARGIN_S = 8.0


# ══════════ per-axis bodies ══════════

class AxisBody(nn.Module):
    """1-D modified MLP: scalar axis coordinate -> [2, RANK] CP factors.
    Spatial axes are fed cos/sin(m*pi*s) so the product ansatz is exactly
    2-periodic in x and y (kb1 012); the time axis gets a smooth bank."""
    periodic: bool
    hidden: int = AX_WIDTH
    depth: int = AX_DEPTH

    @nn.compact
    def __call__(self, s):                      # s: [N, 1] -> [N, 2, RANK]
        kinit = jax.nn.initializers.glorot_normal()
        if self.periodic:
            w = jnp.pi * jnp.arange(1, N_HARM + 1, dtype=s.dtype)
            h = jnp.concatenate([jnp.cos(s * w), jnp.sin(s * w)], axis=1)
        else:
            tau = 2.0 * s / T_MAX - 1.0
            w = jnp.pi * jnp.arange(1, T_HARM + 1, dtype=s.dtype)
            h = jnp.concatenate([tau, jnp.cos(tau * w), jnp.sin(tau * w)], axis=1)
        act = nn.tanh
        U = act(nn.Dense(self.hidden, kernel_init=kinit)(h))
        V = act(nn.Dense(self.hidden, kernel_init=kinit)(h))
        H = act(nn.Dense(self.hidden, kernel_init=kinit)(h))
        for _ in range(self.depth - 1):
            Z = act(nn.Dense(self.hidden, kernel_init=kinit)(H))
            H = (1.0 - Z) * U + Z * V
        out = nn.Dense(OUTPUT_DIM * RANK, kernel_init=kinit)(H)
        return out.reshape(s.shape[0], OUTPUT_DIM, RANK)


class SepNet(nn.Module):
    """The three axis bodies of the separable ansatz."""
    def setup(self):
        self.body_t = AxisBody(periodic=False, name="body_t")
        self.body_x = AxisBody(periodic=True, name="body_x")
        self.body_y = AxisBody(periodic=True, name="body_y")

    def __call__(self, t, x, y):
        return self.body_t(t), self.body_x(x), self.body_y(y)


def _d1(fn, s):
    """First derivative of a 1-D axis body by forward-mode AD."""
    return jax.jvp(fn, (s,), (jnp.ones_like(s),))

def _d2(fn, s):
    """Value and second derivative of a 1-D axis body by nested JVPs."""
    g = lambda z: jax.jvp(fn, (z,), (jnp.ones_like(z),))[1]
    val = fn(s)
    return val, jax.jvp(g, (s,), (jnp.ones_like(s),))[1]


# ══════════ hard initial condition (separable, rank 1) ══════════

def _ic_axis(s):
    """p(s) = 10 s exp(-25 s^2) and q(s) = exp(-25 s^2); A0 = p(x)q(y) + i q(x)p(y)."""
    e = jnp.exp(-25.0 * s ** 2)
    return jnp.concatenate([10.0 * s * e, e], axis=1)      # [N, 2]


def _ic_tensors(x, y):
    """IC field and its x/y Laplacian parts on the lattice: each [2, Nx, Ny]."""
    px, dpx = _d2(_ic_axis, x)
    py, dpy = _d2(_ic_axis, y)
    ic = jnp.stack([px[:, 0:1] * py[:, 1], px[:, 1:2] * py[:, 0]], axis=0)
    ic_xx = jnp.stack([dpx[:, 0:1] * py[:, 1], dpx[:, 1:2] * py[:, 0]], axis=0)
    ic_yy = jnp.stack([px[:, 0:1] * dpy[:, 1], px[:, 1:2] * dpy[:, 0]], axis=0)
    return ic, ic_xx, ic_yy


# ══════════ training ══════════

def _make_train(rng_init):
    model = SepNet()
    key, rng = random.split(rng_init)
    z1 = jnp.zeros([1, 1])
    params = model.init(key, z1, z1, z1)
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    t_slice = jnp.linspace(T_MIN, T_MAX, NT + 1)[1:].reshape(-1, 1)   # (0, T]

    def slice_residuals(pf, x, y):
        p = unravel(pf)
        ft = lambda s: model.apply(p, s, z1, z1, method=lambda m, a, b, c: m.body_t(a))
        fx = lambda s: model.apply(p, z1, s, z1, method=lambda m, a, b, c: m.body_x(b))
        fy = lambda s: model.apply(p, z1, z1, s, method=lambda m, a, b, c: m.body_y(c))

        Ft, Ft_d = _d1(ft, t_slice)              # [NT, 2, R]
        Fx, Fx_dd = _d2(fx, x)                   # [NXY, 2, R]
        Fy, Fy_dd = _d2(fy, y)

        G0 = jnp.einsum("xcr,ycr->cxyr", Fx, Fy)
        Gxx = jnp.einsum("xcr,ycr->cxyr", Fx_dd, Fy)
        Gyy = jnp.einsum("xcr,ycr->cxyr", Fx, Fy_dd)
        mix = lambda F, G: jnp.einsum("tcr,cxyr->ctxy", F, G)
        Nf, Nf_t, Nf_xx, Nf_yy = mix(Ft, G0), mix(Ft_d, G0), mix(Ft, Gxx), mix(Ft, Gyy)

        ic, ic_xx, ic_yy = _ic_tensors(x, y)
        tt = t_slice.reshape(1, -1, 1, 1)
        U = ic[:, None] + tt * Nf                # hard IC: A = A0 + t N
        U_t = Nf + tt * Nf_t
        U_xx = ic_xx[:, None] + tt * Nf_xx
        U_yy = ic_yy[:, None] + tt * Nf_yy

        u, v = U[0], U[1]
        m2 = u ** 2 + v ** 2
        r_u = U_t[0] - EPS * (U_xx[0] + U_yy[0]) - K * (u - u * m2 + C * v * m2)
        r_v = U_t[1] - EPS * (U_xx[1] + U_yy[1]) - K * (v - v * m2 - C * u * m2)
        return jnp.mean(r_u ** 2 + r_v ** 2, axis=(1, 2))     # [NT]

    def loss_fn(pf, x, y, eps):
        Ls = slice_residuals(pf, x, y)
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(Ls)[:-1]])
        w = jax.lax.stop_gradient(jnp.exp(-eps * cum))
        return jnp.mean(w * Ls), w

    sched = optax.exponential_decay(LR0, DECAY_EVERY, DECAY)
    optimizer = optax.adam(learning_rate=sched)
    opt_state = optimizer.init(flat_params)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jit
    def update(pf, opt_state, key, eps):
        kx, ky = random.split(key)
        x = random.uniform(kx, (NXY, 1), minval=X_MIN, maxval=X_MAX)
        y = random.uniform(ky, (NXY, 1), minval=Y_MIN, maxval=Y_MAX)
        (loss, w), grads = grad_fn(pf, x, y, eps)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(pf, updates), opt_state, loss, jnp.min(w)

    return flat_params, opt_state, update, unravel, rng


# ══════════ public contract ══════════

_PREDICT_MODEL = SepNet()


def _predict_chunk(params, X):
    t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    Ft, Fx, Fy = _PREDICT_MODEL.apply(params, t, x, y)
    N = jnp.sum(Ft * Fx * Fy, axis=2)                    # [B, 2]
    e = jnp.exp(-25.0 * (x ** 2 + y ** 2))
    return jnp.concatenate([10.0 * x * e, 10.0 * y * e], axis=1) + t * N


def predict_fn(params, X):
    outs = [_predict_chunk(params, X[i:i + PRED_CHUNK])
            for i in range(0, X.shape[0], PRED_CHUNK)]
    out = jnp.concatenate(outs, axis=0)
    return {"u": out[:, 0], "v": out[:, 1]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam on the causally weighted PDE residual over a
    NT x NXY x NXY separable lattice (IC and periodic BC hard-constrained)."""
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
