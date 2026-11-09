"""Vorticity-streamfunction (omega-psi) reformulation with a dedicated
attractor-breaking tune. Outputs (psi_raw, omega); psi=g(x,y)*psi_raw hard-zeros
psi on all walls (exact continuity, no pressure); u=psi_y, v=-psi_x. Scale-PINN
pseudo-transient correction moved onto the vorticity-transport eq. Parent: b09_cap56000."""
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
from jax import jacfwd, grad, jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


# ══════════ Scale-PINN net: sin(F*pi*.) input layer + SiLU trunk ══════════

class ScalePINN(nn.Module):
    """sin(freq*pi*Wx) input layer, SiLU hidden layers, linear output of 2
    channels (psi_raw, omega)."""
    hidden: int = 128
    depth: int = 4
    freq: float = 1.0
    out_dim: int = 2

    @nn.compact
    def __call__(self, xy):
        kinit = jax.nn.initializers.he_uniform()
        h = jnp.sin(self.freq * jnp.pi * nn.Dense(self.hidden, kernel_init=kinit)(xy))
        for _ in range(self.depth):
            h = nn.silu(nn.Dense(self.hidden, kernel_init=kinit)(h))
        return nn.Dense(self.out_dim, kernel_init=kinit)(h)   # → [N, 2]: (psi_raw, omega)


# ══════════ Hyperparameters ══════════

HIDDEN      = 128
DEPTH       = 4
FREQ        = 1.0
TAU_SC      = 0.05         # sequential-correction pseudo-timestep (kb1 126) on omega
TAU_ALPHA   = 8.0          # smoothing coeff = nu_eff/8 (b02 optimum, on omega Laplacian)
RE_START    = 1000.0       # curriculum start Re
RE_WARM     = 40000        # steps to ramp Re_start -> RE=3200, then hold
LAM_BC      = 100.0        # velocity BC weight (u=psi_y, v=-psi_x)
LAM_KIN     = 1.0          # kinematic residual weight (lap_psi + omega = 0)
BS_INT      = 1024
BS_BC       = 256
PEAK_LR     = 2e-3
WARMUP      = 2000
DECAY_STEPS = 60000
END_LR      = 1e-6
ADAM_WALL   = 999.0        # run Adam to the wall deadline (no step cap)
SAVE_MARGIN_S = 8.0


# ══════════ Fresh continuous sampling ══════════

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


def _g(xy):
    """Wall multiplier: zero on all four walls of the unit square."""
    x, y = xy[0], xy[1]
    return x * (1.0 - x) * y * (1.0 - y)


# ══════════ Loss: omega-psi residuals + Scale-PINN correction on transport ══════════

def _make_train(rng_init):
    model = ScalePINN(hidden=HIDDEN, depth=DEPTH, freq=FREQ)
    key, rng = random.split(rng_init)
    params0 = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat0, unravel = jax.flatten_util.ravel_pytree(params0)

    def fields(pf, xy):
        """Return [psi, omega] with psi hard-zeroed on walls via g(x,y)."""
        out = model.apply(unravel(pf), xy[None, :])[0]   # [2]: (psi_raw, omega)
        return jnp.array([_g(xy) * out[0], out[1]])

    d1_fn = jacfwd(fields, argnums=1)                     # [2, 2]: rows (psi,omega); cols (x,y)
    d2_fn = jacfwd(jacfwd(fields, argnums=1), argnums=1)  # [2, 2, 2]

    def res_per_point(pf, pf_prev, xy, nu):
        d1 = d1_fn(pf, xy)
        psi_x, psi_y = d1[0, 0], d1[0, 1]
        om_x, om_y   = d1[1, 0], d1[1, 1]
        d2 = d2_fn(pf, xy)
        lap_psi = d2[0, 0, 0] + d2[0, 1, 1]
        lap_om  = d2[1, 0, 0] + d2[1, 1, 1]
        om = fields(pf, xy)[1]
        u, v = psi_y, -psi_x

        # previous iterate (omega transport pseudo-transient)
        om_p = fields(pf_prev, xy)[1]
        d2p = d2_fn(pf_prev, xy)
        lap_om_p = d2p[1, 0, 0] + d2p[1, 1, 1]

        r_kin = lap_psi + om                          # kinematic: lap(psi) = -omega
        r_tr  = u * om_x + v * om_y - nu * lap_om      # steady vorticity transport

        # Scale-PINN correction on the transport eq (pseudo-time on omega)
        corr = (om - om_p) / TAU_SC - (nu / TAU_ALPHA) * (lap_om - lap_om_p)
        res_tr = r_tr + corr
        return LAM_KIN * r_kin ** 2 + res_tr ** 2

    res_batch = vmap(res_per_point, in_axes=(None, None, 0, None))

    def vel_at(pf, xy):
        d1 = d1_fn(pf, xy)
        return jnp.array([d1[0, 1], -d1[0, 0]])       # (u, v) = (psi_y, -psi_x)

    vel_batch = vmap(vel_at, in_axes=(None, 0))

    def loss_fn(pf, pf_prev, xy_int, xy_bc, u_bc, v_bc, nu):
        pde_loss = jnp.mean(res_batch(pf, pf_prev, xy_int, nu))
        uv = vel_batch(pf, xy_bc)
        bc_loss = jnp.mean((uv[:, 0] - u_bc) ** 2 + (uv[:, 1] - v_bc) ** 2)
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
    """X: [BS, 2] (x, y). u=psi_y, v=-psi_x via autodiff; p unused (not scored)."""
    def psi_of(xy):
        out = _PREDICT_MODEL.apply(params, xy[None, :])[0]
        return _g(xy) * out[0]
    dpsi = vmap(grad(psi_of))(X)     # [BS, 2]: (psi_x, psi_y)
    u = dpsi[:, 1]
    v = -dpsi[:, 0]
    p = jnp.zeros(X.shape[0])
    return {"u": u, "v": v, "p": p}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Adam (omega-psi Scale-PINN correction + Re curriculum) to the wall."""
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
