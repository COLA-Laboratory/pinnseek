"""SOAP + pseudo-time-stepping PINN for LDC Re=3200 (fork of b01_soap_curr).
Adds adaptive pseudo-time-relaxed residual (kb1 130) with fresh resampling to
destabilise the weak/spurious LDC solution the parent plateaued at (rRMSE~0.85).
Artificial-compressibility map u<-r_mx, v<-r_my, p<-continuity. SOAP + Re curr.
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
import numpy as np
import optax
from flax import linen as nn
from soap_jax import soap


# ══════════ Hyperparameters ══════════

HIDDEN     = 128
DEPTH      = 4
NUM_FF     = 64
FF_SCALE   = 1.5
BS_INT     = 2048
NBC_WALL   = 96
LR0        = 2e-3
LR_END     = 2e-4
EMA_ALPHA  = 0.9
LAMBDA_UPD = 250
LAM_MAX    = 200.0
SAVE_MARGIN_S = 6.0

NU_START   = 1.0 / 100.0
NU_TRUE    = NU
CURR_FRAC  = 0.5

# Pseudo-time stepping (kb1 130)
TAU0       = 1.0
TAU_BETA   = 0.5        # EMA on tau
TAU_UPD    = 500        # steps between tau refresh
TAU_EPS    = 1e-8

_FF_KEY    = 20240721


# ══════════ Model: Fourier features + modified MLP (kb1 020, 050) ══════════

class ModifiedMLP(nn.Module):
    hidden: int = HIDDEN
    depth: int = DEPTH
    num_ff: int = NUM_FF
    ff_scale: float = FF_SCALE

    @nn.compact
    def __call__(self, x):
        B = random.normal(random.PRNGKey(_FF_KEY),
                          (INPUT_DIM, self.num_ff)) * self.ff_scale
        B = jax.lax.stop_gradient(B)
        proj = 2.0 * jnp.pi * (x @ B)
        h0 = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        kinit = jax.nn.initializers.glorot_normal()
        U = nn.tanh(nn.Dense(self.hidden, kernel_init=kinit, name="U")(h0))
        V = nn.tanh(nn.Dense(self.hidden, kernel_init=kinit, name="V")(h0))
        H = nn.tanh(nn.Dense(self.hidden, kernel_init=kinit, name="in")(h0))
        for i in range(self.depth - 1):
            Z = nn.tanh(nn.Dense(self.hidden, kernel_init=kinit, name=f"h{i}")(H))
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit, name="out")(H)


_MODEL = ModifiedMLP()


def _tree_absflat(g):
    return jnp.concatenate([jnp.abs(x).ravel() for x in jax.tree_util.tree_leaves(g)])


# ══════════ Training closures ══════════

def _make_train(rng_init):
    key, rng = random.split(rng_init)
    params = _MODEL.init(key, jnp.zeros([1, INPUT_DIM]))

    def predict_one(params, xy):
        return _MODEL.apply(params, xy[None, :])[0]      # [3]

    d1_fn = jacfwd(predict_one, argnums=1)
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)

    def fields_res(params, xy, nu):
        """Return (uvp[3], res[3]) with res = (r_c, r_mx, r_my)."""
        uvp = predict_one(params, xy)
        u, v = uvp[0], uvp[1]
        d1 = d1_fn(params, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        d2 = d2_fn(params, xy)
        u_xx, u_yy = d2[0, 0, 0], d2[0, 1, 1]
        v_xx, v_yy = d2[1, 0, 0], d2[1, 1, 1]
        r_c  = u_x + v_y
        r_mx = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        r_my = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        return uvp, jnp.array([r_c, r_mx, r_my])

    fr_batch = vmap(fields_res, in_axes=(None, 0, None))   # -> (uvp[N,3], res[N,3])

    def interior_pts_loss(params, params_prev, xy_int, nu, tau):
        uvp_n, res_n = fr_batch(params, xy_int, nu)
        uvp_p, _ = fr_batch(params_prev, xy_int, nu)
        uvp_p = jax.lax.stop_gradient(uvp_p)
        du = uvp_n - uvp_p                                  # [N,3]
        # artificial-compressibility pseudo-time: u<-r_mx, v<-r_my, p<-continuity
        e_u = du[:, 0] / tau + res_n[:, 1]
        e_v = du[:, 1] / tau + res_n[:, 2]
        e_p = du[:, 2] / tau + res_n[:, 0]
        return jnp.mean(e_u ** 2 + e_v ** 2 + e_p ** 2)

    def bc_loss(params, xy_bc, u_bc, v_bc):
        uvp = vmap(predict_one, in_axes=(None, 0))(params, xy_bc)
        return jnp.mean((uvp[:, 0] - u_bc) ** 2 + (uvp[:, 1] - v_bc) ** 2)

    def _sample(key):
        k1, k2 = random.split(key)
        xy_int = random.uniform(k1, (BS_INT, INPUT_DIM), minval=0.0, maxval=1.0)
        t = random.uniform(k2, (NBC_WALL,))
        one = jnp.ones_like(t); zero = jnp.zeros_like(t)
        top = jnp.stack([t, one], 1)
        bot = jnp.stack([t, zero], 1)
        lft = jnp.stack([zero, t], 1)
        rgt = jnp.stack([one, t], 1)
        xy_bc = jnp.concatenate([top, bot, lft, rgt], 0)
        u_bc = jnp.concatenate([one, zero, zero, zero], 0)
        v_bc = jnp.zeros(4 * NBC_WALL)
        return xy_int, xy_bc, u_bc, v_bc

    sched = optax.exponential_decay(LR0, transition_steps=6000,
                                    decay_rate=0.9, end_value=LR_END)
    optimizer = soap(learning_rate=sched, b1=0.95, b2=0.95,
                     precondition_frequency=10)
    opt_state = optimizer.init(params)

    @jit
    def update(params, params_prev, opt_state, key, nu, tau, lam_bc):
        xy_int, xy_bc, u_bc, v_bc = _sample(key)
        def total(p):
            return (interior_pts_loss(p, params_prev, xy_int, nu, tau)
                    + lam_bc * bc_loss(p, xy_bc, u_bc, v_bc))
        loss, grads = jax.value_and_grad(total)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        # tau surrogate: change since snapshot at the same fresh points
        uvp_n, res_n = fr_batch(new_params, xy_int, nu)
        uvp_p, res_p = fr_batch(params_prev, xy_int, nu)
        du_norm = jnp.linalg.norm(uvp_n - uvp_p)
        dr_norm = jnp.linalg.norm(res_n - res_p)
        return new_params, opt_state, loss, du_norm, dr_norm

    @jit
    def grad_ratio(params, params_prev, key, nu, tau):
        xy_int, xy_bc, u_bc, v_bc = _sample(key)
        g_r = _tree_absflat(jax.grad(interior_pts_loss)(params, params_prev, xy_int, nu, tau))
        g_b = _tree_absflat(jax.grad(bc_loss)(params, xy_bc, u_bc, v_bc))
        return jnp.max(g_r) / (jnp.mean(g_b) + 1e-12)

    return params, opt_state, update, grad_ratio, rng


# ══════════ Public contract ══════════

def predict_fn(params, X):
    out = _MODEL.apply(params, X)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def _nu_schedule(progress):
    f = min(max(progress / CURR_FRAC, 0.0), 1.0)
    log_nu = (1.0 - f) * np.log(NU_START) + f * np.log(NU_TRUE)
    return float(np.exp(log_nu))


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, opt_state, update, grad_ratio, rng_local = _make_train(rng)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    lam_bc = jnp.array(10.0)
    tau = TAU0
    params_prev = params

    rng_local, k = random.split(rng_local)
    params, opt_state, loss, du, dr = update(
        params, params_prev, opt_state, k, jnp.array(NU_START), jnp.array(tau), lam_bc)
    step_count = 1
    last_loss = float(loss)

    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        progress = (time.time() - _T0) / TRAIN_TIME
        nu = jnp.array(_nu_schedule(progress))
        params_prev = params
        rng_local, k = random.split(rng_local)
        params, opt_state, loss, du, dr = update(
            params, params_prev, opt_state, k, nu, jnp.array(tau), lam_bc)
        step_count = it + 1
        last_loss = float(loss)

        if it % TAU_UPD == 0:
            gamma = max(0.05, 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0))))
            tau_hat = gamma * float(du) / (float(dr) + TAU_EPS)
            if np.isfinite(tau_hat) and tau_hat > 0:
                tau = (1.0 - TAU_BETA) * tau + TAU_BETA * tau_hat
                tau = float(np.clip(tau, 1e-3, 1e3))
        if it % LAMBDA_UPD == 0:
            rng_local, kg = random.split(rng_local)
            r = grad_ratio(params, params_prev, kg, nu, jnp.array(tau))
            if jnp.isfinite(r):
                lam_bc = (1.0 - EMA_ALPHA) * lam_bc + EMA_ALPHA * jnp.clip(r, 1.0, LAM_MAX)
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(params, it, last_loss)

    return params, step_count
