"""Hard (exact) BC, sharper lid smoothing (CORNER_EXP=40) for LDC (kb1 036).
Change vs parent (b02_pts_bc192): u,v built as g + phi*NN so all-wall
Dirichlet BCs hold exactly (phi=16xy(1-x)(1-y); smoothed lid lift g_u);
BC loss + grad-norm balancing REMOVED. Keeps SOAP+Fourier+modMLP,
Re-continuation, pseudo-time momentum forcing (kb1 073/130), K=40 inner.
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
FF_SCALE   = 1.0
BS_INT     = 2048
LR0        = 2e-3
LR_END     = 1e-4
SAVE_MARGIN_S = 6.0

NU_START   = 1.0 / 20.0
NU_TRUE    = NU
CURR_FRAC  = 0.4         # reach true Re at 40% wall

CORNER_EXP = 40          # lid smoothing: u_top sharper corners (exp 40)

# pseudo-time (kb1 073/130)
TAU0       = 1.0
TAU_BETA   = 0.5
TAU_MIN    = 0.05
TAU_MAX    = 20.0
TAU_UPD    = 500
K_INNER    = 40
PTS_BATCH  = 2048

_FF_KEY    = 20240721


# ══════════ Hard-constraint construction (kb1 036) ══════════
# u = g_u + phi*u_hat, v = phi*v_hat, p = p_hat, with phi vanishing on all
# walls and g_u the smoothed lid lift → Dirichlet BCs exact by construction.

def _phi(x, y):
    return 16.0 * x * (1.0 - x) * y * (1.0 - y)


def _lid(x):
    # smoothed top-hat: 1 across the lid, →0 at the two top corners
    return 1.0 - (2.0 * x - 1.0) ** CORNER_EXP


def _g_u(x, y):
    return _lid(x) * y            # =lid at y=1, 0 at y=0 and x∈{0,1}


def _compose(raw, xy):
    """raw = [u_hat, v_hat, p_hat] → physical (u, v, p) with exact BCs."""
    x, y = xy[0], xy[1]
    ph = _phi(x, y)
    u = _g_u(x, y) + ph * raw[0]
    v = ph * raw[1]
    p = raw[2]
    return jnp.stack([u, v, p])


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


# ══════════ Training closures ══════════

def _make_train(rng_init):
    key, rng = random.split(rng_init)
    params = _MODEL.init(key, jnp.zeros([1, INPUT_DIM]))

    def predict_one(params, xy):
        raw = _MODEL.apply(params, xy[None, :])[0]     # [3] raw
        return _compose(raw, xy)                       # [3] physical (u,v,p)

    d1_fn = jacfwd(predict_one, argnums=1)
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)

    def raw_residual(params, xy, nu):
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
        return r_mx, r_my, r_c, u, v

    raw_batch = vmap(raw_residual, in_axes=(None, 0, None))

    def uv_batch(params, xy):
        uvp = vmap(predict_one, in_axes=(None, 0))(params, xy)
        return uvp[:, 0], uvp[:, 1]

    def _sample(key):
        return random.uniform(key, (BS_INT, INPUT_DIM), minval=0.0, maxval=1.0)

    sched = optax.exponential_decay(LR0, transition_steps=6000,
                                    decay_rate=0.85, end_value=LR_END)
    optimizer = soap(learning_rate=sched, b1=0.95, b2=0.95,
                     precondition_frequency=10)
    opt_state = optimizer.init(params)

    @jit
    def update(params, params_prev, opt_state, key, nu, tau, pts_on):
        xy_int = _sample(key)
        u_prev, v_prev = uv_batch(params_prev, xy_int)
        u_prev = jax.lax.stop_gradient(u_prev); v_prev = jax.lax.stop_gradient(v_prev)

        def total(p):
            r_mx, r_my, r_c, u, v = raw_batch(p, xy_int, nu)
            fx = r_mx + pts_on * (u - u_prev) / tau
            fy = r_my + pts_on * (v - v_prev) / tau
            return jnp.mean(fx ** 2 + fy ** 2 + r_c ** 2)

        loss, grads = jax.value_and_grad(total)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jit
    def tau_probe(params, params_prev, key, nu):
        xy = random.uniform(key, (PTS_BATCH, INPUT_DIM), minval=0.0, maxval=1.0)
        rmx_n, rmy_n, _, u_n, v_n = raw_batch(params, xy, nu)
        rmx_p, rmy_p, _, u_p, v_p = raw_batch(params_prev, xy, nu)
        du = jnp.sqrt(jnp.mean((u_n - u_p) ** 2 + (v_n - v_p) ** 2))
        dr = jnp.sqrt(jnp.mean((rmx_n - rmx_p) ** 2 + (rmy_n - rmy_p) ** 2))
        return du / (dr + 1e-8)

    return params, opt_state, update, tau_probe, rng


# ══════════ Public contract ══════════

def predict_fn(params, X):
    raw = _MODEL.apply(params, X)                       # [N, 3] raw
    ph = _phi(X[:, 0], X[:, 1])
    u = _g_u(X[:, 0], X[:, 1]) + ph * raw[:, 0]
    v = ph * raw[:, 1]
    p = raw[:, 2]
    return {"u": u, "v": v, "p": p}


def _nu_schedule(progress):
    f = min(max(progress / CURR_FRAC, 0.0), 1.0)
    log_nu = (1.0 - f) * np.log(NU_START) + f * np.log(NU_TRUE)
    return float(np.exp(log_nu))


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, opt_state, update, tau_probe, rng_local = _make_train(rng)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    tau = jnp.array(TAU0)

    rng_local, k = random.split(rng_local)
    prev = params
    params, opt_state, loss = update(params, prev, opt_state, k,
                                     jnp.array(NU_START), tau, jnp.array(0.0))
    step_count = 1
    last_loss = float(loss)

    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        progress = (time.time() - _T0) / TRAIN_TIME
        nu = jnp.array(_nu_schedule(progress))
        pts_on = jnp.array(1.0 if progress > 0.25 else 0.0)
        if (it - 1) % K_INNER == 0:
            prev = params
        rng_local, k = random.split(rng_local)
        params, opt_state, loss = update(params, prev, opt_state, k, nu, tau, pts_on)
        step_count = it + 1
        last_loss = float(loss)

        if it % TAU_UPD == 0:
            rng_local, kt = random.split(rng_local)
            th = tau_probe(params, prev, kt, nu)
            if jnp.isfinite(th):
                tau = (1.0 - TAU_BETA) * tau + TAU_BETA * jnp.clip(th, TAU_MIN, TAU_MAX)
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(params, it, last_loss)

    return params, step_count
