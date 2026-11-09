"""First-order velocity-vorticity (VV / FO-PINN, kb1 087/019/052) reformulation.
Outputs (u,v,p,omega) directly (strong lid Dirichlet, like primitive); replaces
nabla^2 u by -omega_y so the whole system is FIRST-ORDER (cheaper, better conditioned).
Scale-PINN pseudo-transient; DECAY 200k + LAM_BC 150 + PEAK_LR 2e-3->3e-3. Parent: b10_fovv_bc150 (0.0203)."""
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
    """sin(freq*pi*Wx) input layer, SiLU hidden layers, linear output of 4
    channels (u, v, p, omega)."""
    hidden: int = 128
    depth: int = 4
    freq: float = 1.0
    out_dim: int = 4

    @nn.compact
    def __call__(self, xy):
        kinit = jax.nn.initializers.he_uniform()
        h = jnp.sin(self.freq * jnp.pi * nn.Dense(self.hidden, kernel_init=kinit)(xy))
        for _ in range(self.depth):
            h = nn.silu(nn.Dense(self.hidden, kernel_init=kinit)(h))
        return nn.Dense(self.out_dim, kernel_init=kinit)(h)   # → [N, 4]: (u,v,p,omega)


# ══════════ Hyperparameters ══════════

HIDDEN      = 128
DEPTH       = 4
FREQ        = 1.0
TAU_SC      = 0.05
TAU_ALPHA   = 8.0
RE_START    = 1000.0
RE_WARM     = 40000
LAM_BC      = 150.0       # velocity Dirichlet BC weight
LAM_W       = 1.0          # vorticity-definition compatibility weight (kb1 087)
BS_INT      = 1024
BS_BC       = 256
PEAK_LR     = 3e-3
WARMUP      = 2000
DECAY_STEPS = 200000
END_LR      = 1e-6
ADAM_WALL   = 999.0
SAVE_MARGIN_S = 8.0


def _sample_interior(key, n):
    return random.uniform(key, (n, 2), minval=0.0, maxval=1.0)


def _sample_boundary(key, n):
    m = n // 4
    t = random.uniform(key, (4 * m,))
    top    = jnp.stack([t[0:m],       jnp.ones(m)],  axis=1)
    bottom = jnp.stack([t[m:2 * m],   jnp.zeros(m)], axis=1)
    left   = jnp.stack([jnp.zeros(m), t[2 * m:3 * m]], axis=1)
    right  = jnp.stack([jnp.ones(m),  t[3 * m:4 * m]], axis=1)
    pts = jnp.concatenate([top, bottom, left, right], axis=0)
    u_bc = jnp.concatenate([jnp.full(m, U_LID), jnp.zeros(3 * m)])
    v_bc = jnp.zeros(4 * m)
    return pts, u_bc, v_bc


# ══════════ Loss: first-order VV residuals + Scale-PINN correction ══════════

def _make_train(rng_init):
    model = ScalePINN(hidden=HIDDEN, depth=DEPTH, freq=FREQ)
    key, rng = random.split(rng_init)
    params0 = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat0, unravel = jax.flatten_util.ravel_pytree(params0)

    def predict_one(pf, xy):
        return model.apply(unravel(pf), xy[None, :])[0]        # [4]: (u,v,p,omega)

    d1_fn = jacfwd(predict_one, argnums=1)                     # [4, 2] — FIRST-ORDER only

    def res_per_point(pf, pf_prev, xy, nu):
        uvpo = predict_one(pf, xy)
        u, v, p, om = uvpo[0], uvpo[1], uvpo[2], uvpo[3]
        d1 = d1_fn(pf, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        om_x, om_y = d1[3, 0], d1[3, 1]

        # previous iterate (pseudo-transient on momentum)
        uvpo_p = predict_one(pf_prev, xy)
        u_p, v_p, p_p = uvpo_p[0], uvpo_p[1], uvpo_p[2]
        d1p = d1_fn(pf_prev, xy)
        om_x_p, om_y_p = d1p[3, 0], d1p[3, 1]

        # first-order VV residuals: nabla^2 u = -om_y, nabla^2 v = om_x
        r_c  = u_x + v_y
        r_mx = u * u_x + v * u_y + p_x + nu * om_y
        r_my = u * v_x + v * v_y + p_y - nu * om_x
        r_w  = om - (v_x - u_y)                         # vorticity definition

        # Scale-PINN correction (smoothing acts on the diffusion term nu*om_y)
        corr_u = (u - u_p) / TAU_SC + (nu / TAU_ALPHA) * (om_y - om_y_p)
        corr_v = (v - v_p) / TAU_SC - (nu / TAU_ALPHA) * (om_x - om_x_p)
        corr_c = (p - p_p) / TAU_SC

        res_c  = r_c  + corr_c
        res_mx = r_mx + corr_u
        res_my = r_my + corr_v
        return res_c ** 2 + res_mx ** 2 + res_my ** 2 + LAM_W * r_w ** 2

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

    return flat0, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = ScalePINN(hidden=HIDDEN, depth=DEPTH, freq=FREQ)


def predict_fn(params, X):
    """X: [BS, 2] (x, y). u,v,p read directly from the 4-channel output."""
    out = _PREDICT_MODEL.apply(params, X)   # [BS, 4]
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Adam (first-order VV Scale-PINN correction + Re curriculum) to the wall."""
    flat0, opt_state, update, unravel, rng = _make_train(rng)
    pf = flat0
    pf_prev = pf

    key, rng = random.split(rng)
    pf_new, opt_state, loss = update(pf, pf_prev, opt_state, key, jnp.asarray(0.0))
    pf_prev, pf = pf, pf_new
    step_count = 1

    deadline = _T0 + min(ADAM_WALL, TRAIN_TIME - SAVE_MARGIN_S)
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        key, rng = random.split(rng)
        pf_new, opt_state, loss = update(pf, pf_prev, opt_state, key, jnp.asarray(float(it)))
        pf_prev, pf = pf, pf_new
        step_count = it + 1
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(pf), it, float(loss))

    return unravel(pf), step_count
