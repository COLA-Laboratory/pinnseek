"""Scale-PINN (kb1 126) with HARD (exact) boundary conditions via distance
functions (kb1 036): u,v are built to satisfy the walls by construction, so
the BC loss and its gradient imbalance (kb1 026/050) vanish. Parent: b01_sc_bc100."""
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
GAMMA_UV    = NU           # momentum residual-smoothing coeff (= 1/Re)
TAU_ALPHA   = 1.0
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


# ══════════ Hard (exact) boundary conditions via distance functions (kb1 036) ══════════

DELTA = 0.02   # corner-regularisation width of the lid profile (~boundary-layer scale)


def _hard_bc(xy, raw):
    """Map raw network output (u_hat, v_hat, p) to trial fields satisfying the
    LDC walls exactly. phi = 0 on every wall (a smooth ADF bubble), so the NN
    correction vanishes there; g_u carries the lid. w(x) rolls the lid off to 0
    at the side walls over width DELTA, regularising the corner discontinuity.
        u = y*w(x) + phi * u_hat      (u=U_LID*w on top, 0 on other walls)
        v = phi * v_hat               (v=0 on all walls)
        p = p_hat                     (pressure is gauge-free, no BC)
    """
    x, y = xy[0], xy[1]
    phi = 16.0 * x * (1.0 - x) * y * (1.0 - y)          # 0 on walls, peak 1 at centre
    w = jnp.tanh(x / DELTA) * jnp.tanh((1.0 - x) / DELTA)
    g_u = U_LID * y * w
    u = g_u + phi * raw[0]
    v = phi * raw[1]
    p = raw[2]
    return jnp.array([u, v, p])


# ══════════ Loss: standard NS residual + Scale-PINN sequential correction ══════════

def _make_train(rng_init):
    model = ScalePINN(hidden=HIDDEN, depth=DEPTH, freq=FREQ)
    key, rng = random.split(rng_init)
    params0 = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat0, unravel = jax.flatten_util.ravel_pytree(params0)

    def predict_one(pf, xy):
        raw = model.apply(unravel(pf), xy[None, :])[0]         # [3]: (u_hat,v_hat,p)
        return _hard_bc(xy, raw)                                # trial fields (exact BC)

    d1_fn = jacfwd(predict_one, argnums=1)                     # [3, 2]
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)  # [3, 2, 2]

    def res_per_point(pf, pf_prev, xy):
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

        # standard steady NS residuals
        r_c  = u_x + v_y
        r_mx = u * u_x + v * u_y + p_x - NU * lap_u
        r_my = u * v_x + v * v_y + p_y - NU * lap_v

        # Scale-PINN sequential correction (kb1 126 §B/§C)
        corr_u = (u - u_p) / TAU_SC - (GAMMA_UV / TAU_ALPHA) * (lap_u - lap_u_p)
        corr_v = (v - v_p) / TAU_SC - (GAMMA_UV / TAU_ALPHA) * (lap_v - lap_v_p)
        corr_c = (p - p_p) / TAU_SC   # continuity patch: pressure ~ divergence

        res_c  = r_c  + corr_c
        res_mx = r_mx + corr_u
        res_my = r_my + corr_v
        return res_c ** 2 + res_mx ** 2 + res_my ** 2

    res_batch = vmap(res_per_point, in_axes=(None, None, 0))

    def loss_fn(pf, pf_prev, xy_int):
        # BCs are exact by construction → PDE residual is the only loss term
        return jnp.mean(res_batch(pf, pf_prev, xy_int))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=PEAK_LR, warmup_steps=WARMUP,
        decay_steps=DECAY_STEPS, end_value=END_LR)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(flat0)
    loss_and_grad = jit(jax.value_and_grad(loss_fn))

    @jit
    def update(pf, pf_prev, opt_state, key):
        xy_int = _sample_interior(key, BS_INT)
        loss, grads = loss_and_grad(pf, pf_prev, xy_int)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(pf, updates), opt_state, loss

    return flat0, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = ScalePINN(hidden=HIDDEN, depth=DEPTH, freq=FREQ)


def predict_fn(params, X):
    """X: [BS, 2] with columns (x, y). Returns {"u","v","p"} each [BS], with
    the exact-BC trial transform applied so scored fields honour the walls."""
    raw = _PREDICT_MODEL.apply(params, X)                  # [BS, 3]
    out = vmap(_hard_bc)(X, raw)                            # [BS, 3] trial fields
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam with Scale-PINN sequential correction. Stores the
    previous iterate w^{k-1} to form the correction; whole process finishes
    within TRAIN_TIME s of _T0."""
    pf, opt_state, update, unravel, rng = _make_train(rng)
    pf_prev = pf

    # warmup step: JIT-compiles the whole training step (counted)
    key, rng = random.split(rng)
    pf_new, opt_state, loss = update(pf, pf_prev, opt_state, key)
    pf_prev, pf = pf, pf_new
    step_count = 1

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        key, rng = random.split(rng)
        pf_new, opt_state, loss = update(pf, pf_prev, opt_state, key)
        pf_prev, pf = pf, pf_new
        step_count = it + 1
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(pf), it, float(loss))

    return unravel(pf), step_count
