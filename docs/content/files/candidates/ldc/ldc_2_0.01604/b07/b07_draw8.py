"""Frontier noise draw #8: b06_rspol recipe verbatim (seed 0), re-run to
characterize the this-machine rRMSE band (timing-jitter step-count variance,
b04). Parent: b06_rspol."""
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


# ══════════ Scale-PINN net: sin(F*pi*.) input layer + SiLU trunk ══════════

class ScalePINN(nn.Module):
    """sin(freq*pi*Wx) on layer 1 (frequency prior), SiLU on hidden layers,
    linear output of 3 channels (u, v, p)."""
    hidden: int = 128
    depth: int = 4          # SiLU hidden layers after the sinusoidal input layer
    freq: float = 1.0

    @nn.compact
    def __call__(self, xy):
        kinit = jax.nn.initializers.he_uniform()
        h = jnp.sin(self.freq * jnp.pi * nn.Dense(self.hidden, kernel_init=kinit)(xy))
        for _ in range(self.depth):
            h = nn.silu(nn.Dense(self.hidden, kernel_init=kinit)(h))
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)   # → [N, 3]: (u, v, p)


# ══════════ Hyperparameters ══════════

HIDDEN      = 128
DEPTH       = 4
FREQ        = 1.0
TAU_SC      = 0.05         # sequential-correction pseudo-timestep (kb1 126)
TAU_ALPHA   = 8.0          # smoothing coeff = nu_eff/8 (b02 optimum)
RE_START    = 1000.0       # curriculum start Re (thicker boundary layers)
RE_WARM     = 40000        # steps to ramp Re_start -> RE=3200, then hold
ADAM_WALL   = 100.0        # seconds of Adam before switching to L-BFGS polish
POLISH_INT  = 4096         # fresh interior points per polish resample
POLISH_BC   = 2048         # fresh boundary points per polish resample
RESAMPLE_EVERY = 200       # TSONN outer step: resample points, KEEP L-BFGS memory
LBFGS_MEM   = 32
LAM_BC      = 100.0         # BC weight (kb1 126 default; NTK imbalance kb1 026/050)
BS_INT      = 1024         # fresh interior collocation points / step
BS_BC       = 256          # fresh boundary points / step (64 per edge)
PEAK_LR     = 2e-3
WARMUP      = 2000
DECAY_STEPS = 60000
END_LR      = 1e-6
SAVE_MARGIN_S = 8.0        # stop early so device_get + pickle beat the wall


# ══════════ Fresh continuous sampling (resampled every step) ══════════

def _sample_interior(key, n):
    return random.uniform(key, (n, 2), minval=0.0, maxval=1.0)


def _sample_boundary(key, n):
    """n points split across the 4 walls; u=U_LID on top, 0 elsewhere; v=0."""
    m = n // 4
    t = random.uniform(key, (4 * m,))
    top    = jnp.stack([t[0:m],       jnp.ones(m)],  axis=1)   # y=1, u=U_LID
    bottom = jnp.stack([t[m:2 * m],   jnp.zeros(m)], axis=1)   # y=0
    left   = jnp.stack([jnp.zeros(m), t[2 * m:3 * m]], axis=1) # x=0
    right  = jnp.stack([jnp.ones(m),  t[3 * m:4 * m]], axis=1) # x=1
    pts = jnp.concatenate([top, bottom, left, right], axis=0)
    u_bc = jnp.concatenate([jnp.full(m, U_LID), jnp.zeros(3 * m)])
    v_bc = jnp.zeros(4 * m)
    return pts, u_bc, v_bc


# ══════════ Loss: standard NS residual + Scale-PINN sequential correction ══════════

def _make_train(rng_init):
    model = ScalePINN(hidden=HIDDEN, depth=DEPTH, freq=FREQ)
    key, rng = random.split(rng_init)
    params0 = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat0, unravel = jax.flatten_util.ravel_pytree(params0)

    def predict_one(pf, xy):
        return model.apply(unravel(pf), xy[None, :])[0]        # [3]: (u, v, p)

    d1_fn = jacfwd(predict_one, argnums=1)                     # [3, 2]
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)  # [3, 2, 2]

    def res_per_point(pf, pf_prev, xy, nu):
        uvp = predict_one(pf, xy)
        u, v, p = uvp[0], uvp[1], uvp[2]
        d1 = d1_fn(pf, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        d2 = d2_fn(pf, xy)
        lap_u = d2[0, 0, 0] + d2[0, 1, 1]
        lap_v = d2[1, 0, 0] + d2[1, 1, 1]

        # previous-iterate fields (pf_prev independent of pf → no grad flows)
        uvp_p = predict_one(pf_prev, xy)
        u_p, v_p, p_p = uvp_p[0], uvp_p[1], uvp_p[2]
        d2p = d2_fn(pf_prev, xy)
        lap_u_p = d2p[0, 0, 0] + d2p[0, 1, 1]
        lap_v_p = d2p[1, 0, 0] + d2p[1, 1, 1]

        # steady NS residuals at the (curriculum) viscosity nu
        r_c  = u_x + v_y
        r_mx = u * u_x + v * u_y + p_x - nu * lap_u
        r_my = u * v_x + v * v_y + p_y - nu * lap_v

        # Scale-PINN sequential correction (kb1 126 §B/§C), smoothing = nu/8
        corr_u = (u - u_p) / TAU_SC - (nu / TAU_ALPHA) * (lap_u - lap_u_p)
        corr_v = (v - v_p) / TAU_SC - (nu / TAU_ALPHA) * (lap_v - lap_v_p)
        corr_c = (p - p_p) / TAU_SC   # continuity patch: pressure ~ divergence

        res_c  = r_c  + corr_c
        res_mx = r_mx + corr_u
        res_my = r_my + corr_v
        return res_c ** 2 + res_mx ** 2 + res_my ** 2

    res_batch = vmap(res_per_point, in_axes=(None, None, 0, None))
    pred_batch = vmap(predict_one, in_axes=(None, 0))

    def loss_fn(pf, pf_prev, xy_int, xy_bc, u_bc, v_bc, nu):
        pde_loss = jnp.mean(res_batch(pf, pf_prev, xy_int, nu))
        bc = pred_batch(pf, xy_bc)
        bc_loss = jnp.mean((bc[:, 0] - u_bc) ** 2 + (bc[:, 1] - v_bc) ** 2)
        return pde_loss + LAM_BC * bc_loss

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=PEAK_LR, warmup_steps=WARMUP,
        decay_steps=DECAY_STEPS, end_value=END_LR)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(flat0)
    loss_and_grad = jit(jax.value_and_grad(loss_fn))

    def _nu_eff(step):
        # ramp Re: RE_START -> RE linearly over RE_WARM steps, then hold at RE
        frac = jnp.minimum(step / RE_WARM, 1.0)
        re_eff = RE_START + (RE - RE_START) * frac
        return 1.0 / re_eff

    @jit
    def update(pf, pf_prev, opt_state, key, step):
        k1, k2 = random.split(key)
        xy_int = _sample_interior(k1, BS_INT)
        xy_bc, u_bc, v_bc = _sample_boundary(k2, BS_BC)
        nu = _nu_eff(step)
        loss, grads = loss_and_grad(pf, pf_prev, xy_int, xy_bc, u_bc, v_bc, nu)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(pf, updates), opt_state, loss

    # ---- L-BFGS polish on the TRUE steady-NS residual (no correction) ----
    def _res_true_sq(pf, xy):
        uvp = predict_one(pf, xy)
        u, v = uvp[0], uvp[1]
        d1 = d1_fn(pf, xy); d2 = d2_fn(pf, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        lap_u = d2[0, 0, 0] + d2[0, 1, 1]
        lap_v = d2[1, 0, 0] + d2[1, 1, 1]
        r_c  = u_x + v_y
        r_mx = u * u_x + v * u_y + p_x - NU * lap_u
        r_my = u * v_x + v * v_y + p_y - NU * lap_v
        return r_c ** 2 + r_mx ** 2 + r_my ** 2

    res_true_batch = vmap(_res_true_sq, in_axes=(None, 0))

    # points passed as runtime args → resampling never recompiles the step
    def polish_loss(pf, xy_int, xy_bc, u_bc, v_bc):
        pde = jnp.mean(res_true_batch(pf, xy_int))
        bc = pred_batch(pf, xy_bc)
        bc_loss = jnp.mean((bc[:, 0] - u_bc) ** 2 + (bc[:, 1] - v_bc) ** 2)
        return pde + LAM_BC * bc_loss

    opt = optax.lbfgs(memory_size=LBFGS_MEM)

    @jit
    def polish_step(pf, state, xy_int, xy_bc, u_bc, v_bc):
        value, grad = jax.value_and_grad(polish_loss)(pf, xy_int, xy_bc, u_bc, v_bc)
        value_fn = lambda p: polish_loss(p, xy_int, xy_bc, u_bc, v_bc)
        updates, state = opt.update(grad, state, pf, value=value, grad=grad,
                                    value_fn=value_fn)
        return optax.apply_updates(pf, updates), state, value

    def polish_init(pf):
        return opt.init(pf)

    return (flat0, opt_state, update, unravel, rng,
            polish_step, polish_init)


# ══════════ Public contract ══════════

_PREDICT_MODEL = ScalePINN(hidden=HIDDEN, depth=DEPTH, freq=FREQ)


def predict_fn(params, X):
    """X: [BS, 2] with columns (x, y). Returns {"u","v","p"} each [BS]."""
    out = _PREDICT_MODEL.apply(params, X)   # [BS, 3]
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam (Scale-PINN correction + Re curriculum) bulk, then a
    TSONN-style L-BFGS tail (kb1 073) that resamples points every RESAMPLE_EVERY
    steps while keeping the L-BFGS curvature history."""
    (pf, opt_state, update, unravel, rng,
     polish_step, polish_init) = _make_train(rng)
    pf_prev = pf

    # warmup step: JIT-compiles the whole training step (counted)
    key, rng = random.split(rng)
    pf_new, opt_state, loss = update(pf, pf_prev, opt_state, key, jnp.asarray(0.0))
    pf_prev, pf = pf, pf_new
    step_count = 1

    # ---- Phase 1: Adam curriculum ----
    adam_deadline = _T0 + min(ADAM_WALL, TRAIN_TIME - SAVE_MARGIN_S - 4.0)
    for it in range(1, MAX_ITER):
        if time.time() > adam_deadline:
            break
        key, rng = random.split(rng)
        pf_new, opt_state, loss = update(pf, pf_prev, opt_state, key, jnp.asarray(float(it)))
        pf_prev, pf = pf, pf_new
        step_count = it + 1
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(pf), it, float(loss))

    # ---- Phase 2: TSONN-style resampled L-BFGS tail (persistent curvature) ----
    def _fresh_points(k):
        k1, k2 = random.split(k)
        xi = _sample_interior(k1, POLISH_INT)
        xb, ub, vb = _sample_boundary(k2, POLISH_BC)
        return xi, xb, ub, vb

    key, rng = random.split(rng)
    xi, xb, ub, vb = _fresh_points(key)
    pstate = polish_init(pf)
    polish_deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    tstep = 0
    # TSONN trusts pseudo-time convergence → return the final iterate.
    # (loss is not comparable across resampled sets, so no cross-set "best".)
    while time.time() < polish_deadline:
        if tstep > 0 and (tstep % RESAMPLE_EVERY == 0):
            key, rng = random.split(rng)
            xi, xb, ub, vb = _fresh_points(key)   # KEEP pstate (curvature)
        pf, pstate, loss = polish_step(pf, pstate, xi, xb, ub, vb)
        step_count += 1
        tstep += 1

    return unravel(pf), step_count
