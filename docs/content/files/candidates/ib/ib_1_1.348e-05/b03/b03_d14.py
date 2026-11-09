"""Shock-excised SOAP PINN (fork: b02_b20, rRMSE 5.07e-5).
vs b03_hptb (3.75e-5): post-shock excision half-width DELTA 0.002 -> 0.0014.
b02 tuned DELTA against a constant-in-t excision; the ramp changes that trade.
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
from functools import partial
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
# The ONLY field data read is task/ib_ic.csv (u at t=0).  Least-squares fit of
# a period-2 trig series -> a smooth, exactly periodic, autodiff-friendly u0(x)
# that the hard-IC ansatz can use pointwise.

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
    """u(0, x); x may be a scalar or an array. Exactly 2-periodic in x."""
    ang = jnp.pi * jnp.asarray(x)[..., None] * _KS_IC
    return jnp.sin(ang) @ _IC_C


# ══════════ Periodic coordinate stretch concentrated at the shock ══════════
# phi(x) = 2 atan(tan(pi x / 2) / BETA) is a smooth, exactly 2-periodic
# reparametrisation of the ring with dphi/dx = pi/BETA at x = 0 and pi*BETA at
# x = +-1.  A K_F-mode Fourier embedding in phi therefore has ~K_F/BETA
# effective modes at the stationary shock and ~K_F*BETA at the (smooth) far
# side — local resolution without the global high-frequency modes that made
# a plain K_F = 128 embedding unstable.  Cf. kb1 012 (exact periodicity),
# kb1 116 (coordinate mapping), kb1 086 (volume weighting).

BETA = 0.20


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


# ══════════ Backbone: modified MLP on a periodic Fourier x-embedding ══════════

WIDTH = 96
DEPTH = 4
K_F = 48          # Fourier modes in x -> exact periodicity of u and all u_x


class Net(nn.Module):
    @nn.compact
    def __call__(self, tx):
        t = tx[..., 0:1]
        x = tx[..., 1:2]
        ks = jnp.arange(1, K_F + 1, dtype=tx.dtype)
        ang = _stretch(x) * ks
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


def _u_scalar(params, t, x):
    """Hard IC (u(0,x) == u0(x)) and hard odd symmetry (u(t,-x) == -u(t,x)),
    both exact for any network weights. Oddness is a symmetry of u_t + u u_x = 0
    that the odd IC hands us: it pins the shock to x=0 and forces u(t,0)=0."""
    n = _MODEL.apply(params, jnp.stack([jnp.stack([t, x]), jnp.stack([t, -x])]))
    return u0_fn(x) + (t / T_MAX) * 0.5 * (n[0, 0] - n[1, 0])


def _residual(params, t, x):
    u, (u_t, u_x) = jax.value_and_grad(_u_scalar, argnums=(1, 2))(params, t, x)
    return u_t + u * u_x


_res_batch = vmap(_residual, in_axes=(None, 0, 0))


# ══════════ Training hyperparameters ══════════

# Excision half-width. The shock sits exactly at x = 0 (the odd ansatz puts it
# there and forces u(t,0) = 0), so unlike kb1 055's compression indicator -- which
# b01 showed fires over the whole half-domain -- the shock's support is known a
# priori and can be removed from the residual measure exactly. The PDE is still
# enforced on |x| >= DELTA, and the Rankine-Hugoniot speed (u_L+u_R)/2 = 0 is
# satisfied identically by oddness, so the entropy solution is still pinned.
DELTA = 0.0014
_PHI_D = float(_stretch_np(DELTA))

NT = 32          # time slices (causal weighting granularity)
NX = 48          # x points per slice
CHUNK = 50       # optimiser steps per jitted scan
LR0 = 2e-3
COS_STEPS = 48000   # measured step horizon of this config in the 100 s wall
EMA_D = 0.999       # weight-averaging decay
EPS_SCHED = (1e-2, 1e-1, 1.0, 10.0, 100.0)
W_DELTA = 0.99   # ramp eps once every slice is unlocked
SAVE_MARGIN_S = 7.0
_TELE: dict = {}  # live telemetry for the b01_tele diagnostic probe


def _loss_fn(params, key, eps):
    kt, kx = random.split(key)
    dt = T_MAX / NT
    ts = (jnp.arange(NT, dtype=jnp.float32) + random.uniform(kt, (NT,))) * dt
    # collocation uniform in the stretched coordinate phi -> automatically
    # ~1/BETA times denser at the shock; 1/p = dx/dphi volume weights keep the
    # per-slice loss an unbiased estimate of the x-integral of r^2 (kb1 086)
    # excision half-width tracks the physical layer: before the breaking time
    # T_B the profile at x=0 is smooth and resolvable, so almost nothing need be
    # excised; as t -> T_B the layer thins to nothing and the full DELTA is
    # needed, and it stays needed once the jump exists.  T_B comes from the IC.
    d_t = DELTA * jnp.clip(ts / _TB, 0.0, 1.0) ** 3
    phi_lo = _stretch(d_t)
    phis = phi_lo[:, None] + (jnp.arange(NX, dtype=jnp.float32) + random.uniform(kx, ())) \
        * ((jnp.pi - phi_lo)[:, None] / NX)
    xs = _unstretch(phis)
    vw = _dx_dphi(xs)
    T = jnp.repeat(ts, NX)
    X = xs.reshape(-1)
    r = _res_batch(params, T, X).reshape(NT, NX)
    Lt = jnp.sum(vw * r ** 2, axis=1) / jnp.sum(vw, axis=1)
    cum = jnp.concatenate([jnp.zeros((1,), Lt.dtype), jnp.cumsum(Lt)[:-1]])
    w = lax.stop_gradient(jnp.exp(-eps * cum))
    return jnp.mean(w * Lt), (jnp.min(w), jnp.mean(Lt))


# ══════════ Public contract ══════════


def predict_fn(params, X):
    """X: [BS, 2] with columns (t, x). Returns {"u": [BS]}."""
    t, x = X[:, 0], X[:, 1]
    n = _MODEL.apply(params, X)[:, 0]
    nm = _MODEL.apply(params, jnp.stack([t, -x], axis=1))[:, 0]
    return {"u": u0_fn(x) + (t / T_MAX) * 0.5 * (n - nm)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params = _MODEL.init(rng, jnp.zeros([1, INPUT_DIM]))
    sched = optax.cosine_decay_schedule(LR0, COS_STEPS, alpha=0.02)
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
    # warmup chunk: compiles the whole scan (real steps, counted)
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
        if float(wmin) > W_DELTA and eps_i < len(EPS_SCHED) - 1:
            eps_i += 1
        _TELE.update(eps=EPS_SCHED[eps_i], wmin=float(wmin), loss=float(loss),
                     step=step_count)
        if eval_callback is not None and (it % 200 == 0):
            eval_callback(params, step_count, float(loss))
    # ema is seeded from the init weights, so no bias correction is needed:
    # after ~50k steps their residual share is EMA_D**n ~ e^-50
    return ema, step_count
