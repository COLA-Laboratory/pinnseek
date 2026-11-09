"""Curriculum modified-MLP PINN for LDC Re=3200 (fork of b00 baseline).
Adds: Fourier-feature + modified-MLP backbone (kb1 020/050), grad-norm
adaptive BC weighting (kb1 050), and a log-linear Re/viscosity curriculum
(kb1 130). Velocity-pressure form; fresh uniform collocation each step.
"""
from __future__ import annotations


# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 2
OUTPUT_DIM   = 3
INPUT_NAMES  = ("x", "y")
FIELD_NAMES  = ("u", "v", "p")

# --- Domain ---
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
HAS_TIME = False

# --- Equation structure ---
TIME_DERIV_ORDER        = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 3   # ru, rv, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("top", "bottom", "left", "right")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "box"

# --- Physical parameters ---
RE     = 3200.0      # NS-LDC Re=3200; harder regime than Re=1000
NU     = 1.0 / RE
U_LID  = 1.0

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 150.0     # total wall-clock seconds for the WHOLE process,
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
from jax import jacfwd, jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


# ══════════ Hyperparameters ══════════

HIDDEN     = 128
DEPTH      = 4          # modified-MLP hidden updates
NUM_FF     = 64         # Fourier features per axis-projection (→ 2*NUM_FF dims)
FF_SCALE   = 2.0        # std of Fourier B matrix
BS_INT     = 4096       # fresh interior collocation per step
NBC_WALL   = 128        # boundary points per wall (×4)
LR0        = 1e-3
LR_END     = 2e-4
EMA_ALPHA  = 0.9        # grad-norm weight EMA (kb1 050)
LAMBDA_UPD = 250        # steps between adaptive-weight refresh
SAVE_MARGIN_S = 6.0

# Re curriculum: viscosity ramps log-linearly from NU_START up to the true
# NU = 1/3200 over the first CURR_FRAC of wall time, then holds. This tames the
# high-Re stiffness (kb1 130 lid-driven-cavity curriculum line).
NU_START   = 1.0 / 100.0
NU_TRUE    = NU        # = 1/RE from frozen header
CURR_FRAC  = 0.45

_FF_KEY    = 20240721   # fixed seed for the (frozen) Fourier B matrix


# ══════════ Model: Fourier features + modified MLP (kb1 020, 050) ══════════

class ModifiedMLP(nn.Module):
    hidden: int = HIDDEN
    depth: int = DEPTH
    num_ff: int = NUM_FF
    ff_scale: float = FF_SCALE

    @nn.compact
    def __call__(self, x):
        # x: [..., 2]. Fixed random Fourier embedding (frozen, not trained).
        B = random.normal(random.PRNGKey(_FF_KEY),
                          (INPUT_DIM, self.num_ff)) * self.ff_scale
        B = jax.lax.stop_gradient(B)
        proj = 2.0 * jnp.pi * (x @ B)                       # [..., num_ff]
        h0 = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)

        kinit = jax.nn.initializers.glorot_normal()
        U = nn.tanh(nn.Dense(self.hidden, kernel_init=kinit, name="U")(h0))
        V = nn.tanh(nn.Dense(self.hidden, kernel_init=kinit, name="V")(h0))
        H = nn.tanh(nn.Dense(self.hidden, kernel_init=kinit, name="in")(h0))
        for i in range(self.depth - 1):
            Z = nn.tanh(nn.Dense(self.hidden, kernel_init=kinit, name=f"h{i}")(H))
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit, name="out")(H)  # [...,3]


# ══════════ Training closures ══════════

def _make_train(rng_init):
    model = ModifiedMLP()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def predict_one(pf, xy):
        return model.apply(unravel(pf), xy[None, :])[0]     # [3]

    d1_fn = jacfwd(predict_one, argnums=1)                   # [3,2]
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)  # [3,2,2]

    def residual_per_point(pf, xy, nu):
        uvp = predict_one(pf, xy)
        u, v = uvp[0], uvp[1]
        d1 = d1_fn(pf, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        d2 = d2_fn(pf, xy)
        u_xx, u_yy = d2[0, 0, 0], d2[0, 1, 1]
        v_xx, v_yy = d2[1, 0, 0], d2[1, 1, 1]
        r_c  = u_x + v_y
        r_mx = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        r_my = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        return r_c ** 2 + r_mx ** 2 + r_my ** 2

    res_batch = vmap(residual_per_point, in_axes=(None, 0, None))

    def pde_loss(pf, xy_int, nu):
        return jnp.mean(res_batch(pf, xy_int, nu))

    def bc_loss(pf, xy_bc, u_bc, v_bc):
        uvp = vmap(predict_one, in_axes=(None, 0))(pf, xy_bc)
        return jnp.mean((uvp[:, 0] - u_bc) ** 2 + (uvp[:, 1] - v_bc) ** 2)

    def _sample(key):
        k1, k2 = random.split(key)
        xy_int = random.uniform(k1, (BS_INT, INPUT_DIM),
                                minval=0.0, maxval=1.0)
        t = random.uniform(k2, (NBC_WALL,))
        one = jnp.ones_like(t); zero = jnp.zeros_like(t)
        top = jnp.stack([t, one], 1)
        bot = jnp.stack([t, zero], 1)
        lft = jnp.stack([zero, t], 1)
        rgt = jnp.stack([one, t], 1)
        xy_bc = jnp.concatenate([top, bot, lft, rgt], 0)
        u_bc = jnp.concatenate([one, zero, zero, zero], 0)   # lid u=1
        v_bc = jnp.zeros(4 * NBC_WALL)
        return xy_int, xy_bc, u_bc, v_bc

    sched = optax.exponential_decay(LR0, transition_steps=8000,
                                    decay_rate=0.9, end_value=LR_END)
    optimizer = optax.adam(learning_rate=sched)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(pf, opt_state, key, nu, lam_bc):
        xy_int, xy_bc, u_bc, v_bc = _sample(key)
        def total(p):
            return pde_loss(p, xy_int, nu) + lam_bc * bc_loss(p, xy_bc, u_bc, v_bc)
        loss, grads = jax.value_and_grad(total)(pf)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(pf, updates), opt_state, loss

    @jit
    def grad_ratio(pf, key, nu):
        xy_int, xy_bc, u_bc, v_bc = _sample(key)
        g_r = jax.grad(pde_loss)(pf, xy_int, nu)
        g_b = jax.grad(bc_loss)(pf, xy_bc, u_bc, v_bc)
        return jnp.max(jnp.abs(g_r)) / (jnp.mean(jnp.abs(g_b)) + 1e-12)

    return flat_params, opt_state, update, grad_ratio, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = ModifiedMLP()


def predict_fn(params, X):
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def _nu_schedule(progress):
    """Log-linear viscosity ramp NU_START → NU_TRUE over [0, CURR_FRAC]."""
    f = min(max(progress / CURR_FRAC, 0.0), 1.0)
    log_nu = (1.0 - f) * np.log(NU_START) + f * np.log(NU_TRUE)
    return float(np.exp(log_nu))


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    pf, opt_state, update, grad_ratio, unravel, rng_local = _make_train(rng)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    lam_bc = jnp.array(100.0)   # start BC-heavy to escape the trivial field

    # warmup / JIT-compile step (counted)
    rng_local, k = random.split(rng_local)
    nu0 = jnp.array(NU_START)
    pf, opt_state, loss = update(pf, opt_state, k, nu0, lam_bc)
    step_count = 1
    last_loss = float(loss)

    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        progress = (time.time() - _T0) / TRAIN_TIME
        nu = jnp.array(_nu_schedule(progress))
        rng_local, k = random.split(rng_local)
        pf, opt_state, loss = update(pf, opt_state, k, nu, lam_bc)
        step_count = it + 1
        last_loss = float(loss)
        if it % LAMBDA_UPD == 0:
            rng_local, kg = random.split(rng_local)
            r = grad_ratio(pf, kg, nu)
            if jnp.isfinite(r):
                lam_bc = (1.0 - EMA_ALPHA) * lam_bc + EMA_ALPHA * jnp.clip(r, 1.0, 1e4)
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(unravel(pf), it, last_loss)

    return unravel(pf), step_count
