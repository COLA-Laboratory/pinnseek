"""Half-domain jump PINN, cosine floor alpha 0.005 (fork: b05_lr3, 2.001e-5).
Same question as the horizon, asked through the floor instead: with a 1.5x larger peak
the end-of-run LR is 1.5x larger too, and the b05_tele curve is still fluctuating at
the wall.
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
T_MIN, T_MAX = 0.0, 1.5
X_MIN, X_MAX = -1.0, 1.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 1   # u_x  (inviscid: no diffusion)
N_RESIDUAL_COMPONENTS   = 1
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi")
PERIODIC_AXES  = ("x",)       # u(t, X_MIN) = u(t, X_MAX)
GEOMETRY_KIND  = "periodic_strip"

# --- Physical parameters (inviscid Burgers: u_t + u u_x = 0) ---
# Conservation law with flux f(u) = u^2/2; no free scalar coefficient.

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
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap
import numpy as np
import optax
from flax import linen as nn
from soap_jax import soap

# TF32 (XLA's default f32 matmul on Ampere) puts ~4e-3 of round-off into a
# residual whose true rms is ~1e-3 (b03_prec): the loss floor was numerical.
jax.config.update("jax_default_matmul_precision", "highest")


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Initial condition as an analytic periodic Fourier series ══════════
# The ONLY field data read is task/ib_ic.csv (u at t=0).

K_IC = 16


def _ic_coeffs() -> np.ndarray:
    ic = np.loadtxt(_locate("ib_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x, u = ic[:, 0], ic[:, 1]
    ks = np.arange(1, K_IC + 1)
    B = np.sin(np.pi * np.outer(x, ks))      # odd basis only: u0 is odd in x
    c, *_ = np.linalg.lstsq(B, u, rcond=None)
    return c.astype(np.float32)


_IC_C = jnp.asarray(_ic_coeffs())
_KS_IC = jnp.arange(1, K_IC + 1, dtype=jnp.float32)


def _breaking_time() -> float:
    """T_B = -1/min u0'(x), from the INITIAL CONDITION alone (1/pi here)."""
    xg = np.linspace(X_MIN, X_MAX, 4001)
    ks = np.arange(1, K_IC + 1)
    du0 = (np.pi * ks * np.cos(np.pi * np.outer(xg, ks))) @ np.asarray(_ic_coeffs(), np.float64)
    return float(-1.0 / du0.min())


_TB = _breaking_time()


def u0_fn(x):
    """u(0, x); x may be a scalar or an array."""
    ang = jnp.pi * jnp.asarray(x)[..., None] * _KS_IC
    return jnp.sin(ang) @ _IC_C


# ══════════ Half-domain reduction ══════════
# u0 is odd, so u(t,-x) = -u(t,x) exactly (b01) and the shock sits at x = 0.
# Periodicity + oddness also force u(t,+-1) = 0.  Write
#       u(t, x) = sign(x) * S(t, |x|),  S(t, 1) = 0,  S(0, y) = u0(y),
# and the jump at the shock is 2*S(t,0), FREE for every t: nothing in the
# ansatz forces u continuous at x = 0 (b03/b04 needed a sawtooth + a gate to
# get this; here it is the natural form).  On y in (0,1] the characteristics
# y0 -> y0 + u0(y0) t never cross before leaving through y = 0 (the exit time
# y0/sin(pi y0) is monotone in y0), so S solves a *shock-free* advection
# problem: the classical solution exists for all t <= T_MAX, the strong-form
# residual has a unique smooth zero, no residual point is irreducible, and no
# boundary condition is needed at the outflow y = 0.  Rankine-Hugoniot
# (speed = (u_L + u_R)/2 = 0) holds identically by oddness.

# ══════════ Coordinate stretch concentrated at the shock ══════════
# phi(y) = 2 atan(tan(pi y / 2)/BETA) maps [0,1] -> [0,pi] with dphi/dy = pi/BETA
# at y = 0 and pi*BETA at y = 1: local resolution near the shock foot without the
# global high frequencies that destabilised a large K_F (b01, b02).
# Cf. kb1 012 (periodic embedding), kb1 116 (coordinate map), kb1 086 (volume wt).

BETA = 0.35


def _stretch(x):
    h = 0.5 * jnp.pi * x
    return 2.0 * jnp.arctan2(jnp.sin(h) / BETA, jnp.cos(h))


def _unstretch(phi):
    h = 0.5 * phi
    return (2.0 / jnp.pi) * jnp.arctan2(BETA * jnp.sin(h), jnp.cos(h))


def _stretch_np(x):
    return 2.0 * np.arctan2(np.sin(0.5 * np.pi * x) / BETA, np.cos(0.5 * np.pi * x))


def _dx_dphi(x):
    h = 0.5 * jnp.pi * x
    return (BETA ** 2) * jnp.cos(h) ** 2 + jnp.sin(h) ** 2   # up to 1/(pi*BETA)


# ══════════ Backbone: modified MLP on a Fourier embedding of phi(y) ══════════

WIDTH = 96
DEPTH = 4
K_F = 48


class Net(nn.Module):
    @nn.compact
    def __call__(self, ty):
        t = ty[..., 0:1]
        y = ty[..., 1:2]
        ks = jnp.arange(1, K_F + 1, dtype=ty.dtype)
        ang = _stretch(y) * ks
        f = jnp.concatenate([t / T_MAX, jnp.cos(ang), jnp.sin(ang)], axis=-1)
        kinit = nn.initializers.glorot_normal()
        U = jnp.tanh(nn.Dense(WIDTH, kernel_init=kinit)(f))
        V = jnp.tanh(nn.Dense(WIDTH, kernel_init=kinit)(f))
        h = jnp.tanh(nn.Dense(WIDTH, kernel_init=kinit)(f))
        for _ in range(DEPTH - 1):
            h = (1.0 - h) * U + h * V
            h = jnp.tanh(nn.Dense(WIDTH, kernel_init=kinit)(h))
        h = (1.0 - h) * U + h * V
        return nn.Dense(1, kernel_init=kinit)(h)


_MODEL = Net()


def _S_scalar(params, t, y):
    """S(t,y) with hard IC (S(0,y) = u0(y)) and hard S(t,1) = 0."""
    n = _MODEL.apply(params, jnp.stack([t, y])[None, :])[0, 0]
    return u0_fn(y) + (t / T_MAX) * n


def _residual(params, t, y):
    u, (u_t, u_y) = jax.value_and_grad(_S_scalar, argnums=(1, 2))(params, t, y)
    return u_t + u * u_y


_res_batch = vmap(_residual, in_axes=(None, 0, 0))


# ══════════ Training hyperparameters ══════════

DELTA = 0.002    # b03's ramped excision: the jump is exact now, but the CORNER
                 # (t = T_B, y = 0) still carries an irreducible |S_y| -> inf
_PHI_D = float(_stretch_np(DELTA))

NT = 32          # time slices (causal weighting granularity)
NX = 64          # y points per slice
CHUNK = 50
LR0 = 3e-3
COS_STEPS = 65000
EMA_D = 0.985
EPS_SCHED = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
EPS_EVERY = 6000
SAVE_MARGIN_S = 3.0
_TELE: dict = {}


def _loss_fn(params, key, eps):
    kt, kx = random.split(key)
    dt = T_MAX / NT
    ts = (jnp.arange(NT, dtype=jnp.float32) + random.uniform(kt, (NT,))) * dt
    d_t = DELTA * jnp.clip(ts / _TB, 0.0, 1.0) ** 3
    phi_lo = _stretch(d_t)
    phis = phi_lo[:, None] + (jnp.arange(NX, dtype=jnp.float32) + random.uniform(kx, ())) \
        * ((jnp.pi - phi_lo)[:, None] / NX)
    ys = _unstretch(phis)
    vw = _dx_dphi(ys)
    T = jnp.repeat(ts, NX)
    Y = ys.reshape(-1)
    r = _res_batch(params, T, Y).reshape(NT, NX)
    Lt = jnp.sum(vw * r ** 2, axis=1) / jnp.sum(vw, axis=1)
    cum = jnp.concatenate([jnp.zeros((1,), Lt.dtype), jnp.cumsum(Lt)[:-1]])
    w = lax.stop_gradient(jnp.exp(-eps * cum))
    return jnp.mean(w * Lt), (jnp.min(w), jnp.mean(Lt))


# ══════════ Public contract ══════════


def predict_fn(params, X):
    """X: [BS, 2] with columns (t, x). Returns {"u": [BS]}."""
    t, x = X[:, 0], X[:, 1]
    xw = jnp.mod(x + 1.0, 2.0) - 1.0
    y = jnp.abs(xw)
    s = jnp.sign(xw)
    n = _MODEL.apply(params, jnp.stack([t, y], axis=1))[:, 0]
    return {"u": s * (u0_fn(y) + (t / T_MAX) * n)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params = _MODEL.init(rng, jnp.zeros([1, INPUT_DIM]))
    sched = optax.cosine_decay_schedule(LR0, COS_STEPS, alpha=0.005)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), soap(learning_rate=sched))
    opt_state = optimizer.init(params)

    ema = jax.tree_util.tree_map(jnp.asarray, params)

    @jit
    def run_chunk(params, opt_state, key, eps, ema, n):
        def body(carry, _):
            p, os_, k, e, n = carry
            k, sub = random.split(k)
            (loss, aux), g = jax.value_and_grad(_loss_fn, has_aux=True)(p, sub, eps)
            upd, os_ = optimizer.update(g, os_, p)
            p = optax.apply_updates(p, upd)
            e = jax.tree_util.tree_map(lambda a, b: EMA_D * a + (1.0 - EMA_D) * b, e, p)
            return (p, os_, k, e, n + 1), (loss, aux[0], aux[1])
        (params, opt_state, key, ema, n), out = lax.scan(
            body, (params, opt_state, key, ema, n), None, length=CHUNK)
        return params, opt_state, key, out[0][-1], out[1][-1], out[2][-1], ema, n

    key = random.split(rng)[0]
    eps_i = 0
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    step_count = 0
    n = jnp.int32(0)
    params, opt_state, key, loss, wmin, lr_mean, ema, n = run_chunk(
        params, opt_state, key, jnp.float32(EPS_SCHED[eps_i]), ema, n)
    step_count += CHUNK
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        params, opt_state, key, loss, wmin, lr_mean, ema, n = run_chunk(
            params, opt_state, key, jnp.float32(EPS_SCHED[eps_i]), ema, n)
        step_count += CHUNK
        if step_count > (eps_i + 1) * EPS_EVERY and eps_i < len(EPS_SCHED) - 1:
            eps_i += 1
        _TELE.update(eps=EPS_SCHED[eps_i], wmin=float(wmin), loss=float(loss),
                     step=step_count)
        if eval_callback is not None and (it % 200 == 0):
            eval_callback(params, step_count, float(loss))
    return ema, step_count
