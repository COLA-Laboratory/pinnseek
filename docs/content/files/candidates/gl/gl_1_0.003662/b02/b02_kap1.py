"""b02_phys with the residual cap tightened, KAPPA 10 -> 1.
Parent: b02_phys.py — the vortex core is unresolvable at 6 harmonics and
still dominates the raw residual; a tighter cap shifts gradient weight to
the bulk, where the score lives. Nothing else changed.
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
N_PT     = 48     # spatial collocation points per slice
LR0      = 1e-3
DECAY    = 0.6
DECAY_EVERY = 6000
EPS0     = 1e-3   # causal tolerance, held so that min(w) stays in [W_LO, W_HI]
EPS_MAX  = 1e6
EPS_MIN  = 1e-6
W_LO     = 0.3
W_HI     = 0.9
CHECK_EVERY = 200
XI       = 0.02   # vortex healing length sqrt(EPS/K); core saturation limit
KAPPA    = 1.0    # soft cap on the per-point squared residual
G_CLAMP  = 5.0    # soft bound on the log-amplitude correction
SAVE_MARGIN_S = 8.0

# --- analytic base ---
S0     = 0.02     # sigma^2 of A0's Gaussian: exp(-r^2/(2 S0)) = exp(-25 r^2)
PHI_S  = 1.0      # INITIAL values of the four trainable base scalars;
DIF    = 0.6      # dif, xi and pre are modulated by exp(tanh(.)) / tanh(.)
PRE    = 2.0      # so they stay inside a factor e / +-1 of these


# ══════════ network: exact-periodic embedding + modified MLP ══════════

class PeriodicModifiedMLP(nn.Module):
    """(t,x,y) -> [t/T, cos/sin(m*pi*x), cos/sin(m*pi*y)] embedding (exactly
    2-periodic in x and y, so the periodic BCs hold to machine precision)
    -> Wang et al. modified MLP -> (g, h), zero-initialised so the untrained
    ansatz is exactly the analytic base."""
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
        out = nn.Dense(OUTPUT_DIM, kernel_init=jax.nn.initializers.zeros)(H)
        # extra channels: the trainable scalars of the analytic base
        z = jnp.zeros((X.shape[0], 1), dtype=out.dtype)
        cols = [out] + [z + self.param(n, jax.nn.initializers.zeros, ())
                        for n in ("phi_tau", "dif_tau", "xi_tau", "pre_tau")]
        return jnp.concatenate(cols, axis=1)


# ══════════ analytic base: linear growth + diffusion, then saturation ══════════

def _base_field(t, x, y, phi, dif, xi, pre):
    """A0 = 10 (x + i y) exp(-r^2 / (2 S0)) is a degree-1 harmonic polynomial
    times a Gaussian, so the LINEAR flow A_t = eps lap A + k A keeps its shape
    exactly:  A_lin = e^{k t} 10 (x+iy) (S0/s)^2 exp(-r^2/(2 s)),
    s = S0 + 2 eps t  (in d dims a harmonic polynomial of degree n carries the
    prefactor s^-(d/2+n)).  The logistic saturation of |A|^2 with the
    healing-length ceiling sc = r^2/(r^2+xi^2) and the closed-form nonlinear
    phase -(c/2) sc ln D are then applied to it; D = 1 at t = 0, so the IC is
    exact and the zero-initialised network starts the run at this base."""
    s = S0 + 2.0 * EPS * dif * t
    g = (S0 / s) ** pre
    e = g * jnp.exp(-(x ** 2 + y ** 2) / (2.0 * s))
    ux, uy = 10.0 * x * e, 10.0 * y * e
    m_lin = (ux ** 2 + uy ** 2) * jnp.exp(2.0 * K * t)     # |A_linear|^2
    # periodic squared distance to the vortex (= x^2 + y^2 near the origin)
    rp2 = (4.0 / jnp.pi ** 2) * (jnp.sin(0.5 * jnp.pi * x) ** 2
                                 + jnp.sin(0.5 * jnp.pi * y) ** 2)
    sc = rp2 / (rp2 + xi ** 2)
    D = 1.0 + m_lin * (-jnp.expm1(-2.0 * K * t)) * (rp2 + xi ** 2) / (rp2 + 1e-20)
    amp = jnp.exp(K * t) / jnp.sqrt(D)
    p = -0.5 * C * phi * sc * jnp.log(D)
    cp, sp = jnp.cos(p), jnp.sin(p)
    return amp * (ux * cp - uy * sp), amp * (ux * sp + uy * cp)


def _ansatz(model_apply, params, X):
    """A = A_base * exp(t g + i t h), network zero-initialised."""
    t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    N = model_apply(params, X)
    uL, vL = _base_field(t, x, y,
                         PHI_S + N[:, 2:3],
                         DIF * jnp.exp(jnp.tanh(N[:, 3:4])),
                         XI * jnp.exp(jnp.tanh(N[:, 4:5])),
                         PRE + jnp.tanh(N[:, 5:6]))
    g = G_CLAMP * jnp.tanh(t * N[:, 0:1] / G_CLAMP)
    h = t * N[:, 1:2]
    eg, ch, sh = jnp.exp(g), jnp.cos(h), jnp.sin(h)
    return jnp.concatenate([eg * (uL * ch - vL * sh),
                            eg * (uL * sh + vL * ch)], axis=1)


# ══════════ training ══════════

def _make_train(rng_init):
    model = PeriodicModifiedMLP()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    t_slice = jnp.linspace(T_MIN, T_MAX, N_SLICE + 1)[1:]      # (0, T]

    def field(pf, Z):
        return _ansatz(model.apply, unravel(pf), Z)

    def slice_residuals(pf, XY):
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
        s = (r_u ** 2 + r_v ** 2).reshape(N_SLICE, N_PT)
        return jnp.mean(s / (1.0 + s / KAPPA), axis=1)

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
    """Wall-deadline Adam on the causally weighted PDE residual; IC, periodic
    BC and the local reaction-diffusion dynamics are built into the ansatz."""
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
            w_now = float(wmin)
            if w_now > W_HI and eps < EPS_MAX:
                eps *= 3.0
            elif w_now < W_LO and eps > EPS_MIN:
                eps /= 3.0
            if eval_callback is not None:
                eval_callback(unravel(params_flat), it, last_loss)
            if time.time() > deadline:
                break

    return unravel(params_flat), step_count
