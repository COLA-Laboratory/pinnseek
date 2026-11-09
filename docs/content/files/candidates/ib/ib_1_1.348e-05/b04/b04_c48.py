"""Shock-excised SOAP PINN (fork: b04_fast, 8.16e-5 at 63050 steps).
vs b04_fast: the cosine horizon goes back to b03's 48000. b04_tele showed the longer
horizon left the LR 2.1x higher when the causal ladder jumps to eps = 1e4 at step
~37500, and the run spent 13k steps recovering from a 6x rRMSE excursion.
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
# phi(x) = 2 atan(tan(pi x / 2) / BETA): ~K_F/BETA effective modes at the
# stationary shock and ~K_F*BETA in the smooth far field (kb1 012, 116, 086).

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
# Written out by hand instead of flax so that (i) the three first-layer Dense
# that feed U, V and h are ONE (1+2K_F, 3*WIDTH) matmul and one SOAP tensor,
# and (ii) the antisymmetric twin at -x reuses cos(k phi) and only flips the
# sign of the sin block — the widest matmul is then done at N rows, not 2N.
# Mathematically identical to b03_epshi's Net (b04_bench2 checks the loss).

WIDTH = 96
DEPTH = 4
K_F = 48          # Fourier modes in x -> exact periodicity of u and all u_x
_KS_F = jnp.arange(1, K_F + 1, dtype=jnp.float32)


def _init_params(rng):
    ks = random.split(rng, DEPTH + 2)

    def gl(k, shape, fan_out):   # glorot_normal with the UNMERGED fan-out
        return random.normal(k, shape, jnp.float32) * np.sqrt(2.0 / (shape[0] + fan_out))

    p = {"W0": gl(ks[0], (1 + 2 * K_F, 3 * WIDTH), WIDTH),
         "b0": jnp.zeros((3 * WIDTH,), jnp.float32)}
    for d in range(1, DEPTH):
        p[f"W{d}"] = gl(ks[d], (WIDTH, WIDTH), WIDTH)
        p[f"b{d}"] = jnp.zeros((WIDTH,), jnp.float32)
    p["Wo"] = gl(ks[DEPTH], (WIDTH, 1), 1)
    p["bo"] = jnp.zeros((1,), jnp.float32)
    return p


def _u_batch(params, t, x):
    """Hard IC (u(0,x) == u0(x)) and hard odd symmetry (u(t,-x) == -u(t,x)),
    both exact for any weights. Oddness is a symmetry of u_t + u u_x = 0 that
    the odd IC hands us: it pins the shock to x=0 and forces u(t,0)=0."""
    ang = _stretch(x)[..., None] * _KS_F
    W0 = params["W0"]
    zc = jnp.concatenate([(t / T_MAX)[..., None], jnp.cos(ang)], -1) @ W0[:1 + K_F] \
        + params["b0"]
    zs = jnp.sin(ang) @ W0[1 + K_F:]
    U, V, h = jnp.split(jnp.tanh(jnp.concatenate([zc + zs, zc - zs], 0)), 3, axis=-1)
    for d in range(1, DEPTH):
        h = (1.0 - h) * U + h * V
        h = jnp.tanh(h @ params[f"W{d}"] + params[f"b{d}"])
    h = (1.0 - h) * U + h * V
    o = (h @ params["Wo"] + params["bo"])[..., 0]
    n = t.shape[0]
    return u0_fn(x) + (t / T_MAX) * 0.5 * (o[:n] - o[n:])


def _u_scalar(params, t, x):
    return _u_batch(params, t[None], x[None])[0]


def _residual(params, t, x):
    u, (u_t, u_x) = jax.value_and_grad(_u_scalar, argnums=(1, 2))(params, t, x)
    return u_t + u * u_x


_res_batch = vmap(_residual, in_axes=(None, 0, 0))


# ══════════ Training hyperparameters ══════════

# Excision half-width, ramped as (t/T_B)^3: the layer at x=0 is only
# irreducible once it is thinner than the network can resolve, and T_B comes
# from the IC alone (b03 finding 4). The shock sits exactly at x=0 (the odd
# ansatz puts it there), so unlike kb1 055's compression indicator the shock's
# support is known a priori and can be removed from the residual measure.
DELTA = 0.002
_PHI_D = float(_stretch_np(DELTA))

NT = 32          # time slices (causal weighting granularity)
NX = 48          # x points per slice
CHUNK = 50       # optimiser steps per jitted scan
LR0 = 2e-3
COS_STEPS = 48000   # b03's two-sided optimum; the eps ladder is step-indexed too
EMA_D = 0.999       # weight-averaging decay
EPS_SCHED = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
EPS_EVERY = 6000  # deterministic ladder: one rung per 6000 steps
SAVE_MARGIN_S = 3.0
_TELE: dict = {}  # live telemetry for the b01_tele diagnostic probe


def _loss_fn(params, key, eps):
    kt, kx = random.split(key)
    dt = T_MAX / NT
    ts = (jnp.arange(NT, dtype=jnp.float32) + random.uniform(kt, (NT,))) * dt
    # collocation uniform in the stretched coordinate phi -> automatically
    # ~1/BETA times denser at the shock; 1/p = dx/dphi volume weights keep the
    # per-slice loss an unbiased estimate of the x-integral of r^2 (kb1 086)
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
    return {"u": _u_batch(params, X[:, 0], X[:, 1])}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params = _init_params(rng)
    sched = optax.cosine_decay_schedule(LR0, COS_STEPS, alpha=0.02)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), soap(learning_rate=sched))
    opt_state = optimizer.init(params)

    ema = jax.tree_util.tree_map(jnp.asarray, params)

    @jit
    def run_chunk(params, opt_state, key, eps, ema):
        def body(carry, _):
            p, os_, k, e = carry
            k, sub = random.split(k)
            (loss, aux), g = jax.value_and_grad(_loss_fn, has_aux=True)(p, sub, eps)
            upd, os_ = optimizer.update(g, os_, p)
            p = optax.apply_updates(p, upd)
            e = jax.tree_util.tree_map(lambda a, b: EMA_D * a + (1.0 - EMA_D) * b, e, p)
            return (p, os_, k, e), (loss, aux[0])
        (params, opt_state, key, ema), out = lax.scan(
            body, (params, opt_state, key, ema), None, length=CHUNK)
        return params, opt_state, key, out[0][-1], out[1][-1], ema

    key = random.split(rng)[0]
    eps_i = 0
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    step_count = 0
    # warmup chunk: compiles the whole scan (real steps, counted)
    params, opt_state, key, loss, wmin, ema = run_chunk(
        params, opt_state, key, jnp.float32(EPS_SCHED[eps_i]), ema)
    step_count += CHUNK
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        params, opt_state, key, loss, wmin, ema = run_chunk(
            params, opt_state, key, jnp.float32(EPS_SCHED[eps_i]), ema)
        step_count += CHUNK
        if step_count > (eps_i + 1) * EPS_EVERY and eps_i < len(EPS_SCHED) - 1:
            eps_i += 1
        _TELE.update(eps=EPS_SCHED[eps_i], loss=float(loss), step=step_count)
        if eval_callback is not None and (it % 200 == 0):
            eval_callback(params, step_count, _TELE["loss"])
    # ema is seeded from the init weights, so no bias correction is needed:
    # after ~60k steps their residual share is EMA_D**n ~ e^-60
    return ema, step_count
