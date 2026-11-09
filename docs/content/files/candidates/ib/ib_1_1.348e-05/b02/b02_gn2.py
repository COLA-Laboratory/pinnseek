"""Two-stage SOAP -> Gauss-Newton PINN, anti-aliased (fork: b02_gn.py).
vs parent: GN collocation is jittered per point (not one shared x-grid), LM
damping is slower to relax, and iterates are selected on a fixed held-out
residual grid — b02_gn's GN overfitted its 64 shared x-lines.
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
from jax.flatten_util import ravel_pytree
import numpy as np
import optax
from flax import linen as nn
from soap_jax import soap

_SMOKE = TRAIN_TIME < 50.0   # CPU dress rehearsal: shrink the linear algebra


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

BETA = 0.25


def _stretch(x):
    h = 0.5 * jnp.pi * x
    return 2.0 * jnp.arctan2(jnp.sin(h) / BETA, jnp.cos(h))


def _unstretch(phi):
    h = 0.5 * phi
    return (2.0 / jnp.pi) * jnp.arctan2(BETA * jnp.sin(h), jnp.cos(h))


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

NT = 48          # time slices (causal weighting granularity)
NX = 64          # x points per slice
CHUNK = 50       # optimiser steps per jitted scan
LR0 = 2e-3
COS_STEPS = 26000   # step horizon of the (shortened) SOAP phase
EMA_D = 0.999       # weight-averaging decay
EPS_SCHED = (1e-2, 1e-1, 1.0, 10.0, 100.0)
W_DELTA = 0.99   # ramp eps once every slice is unlocked
SAVE_MARGIN_S = 7.0

# ── stage 2: low-rank Levenberg-Marquardt / Gauss-Newton ──
# The loss is a pure nonlinear least squares  L = ||r(theta)||^2  (hard IC +
# hard oddness leave no other term), and #residuals << #params, so the
# Gauss-Newton step has the cheap dual form  delta = -J^T (J J^T + lam I)^-1 r
# (kb1 108 sec. B; kb1 063 energy natural gradient).  J is assembled row-wise as
# per-point grad_theta r, so the cost is one N x P jacobian + one N x N solve.
SOAP_WALL = 55.0            # wall seconds (from _T0) spent in stage 1
GN_NT, GN_NX = 64, 64       # residual points per GN step (resampled each step)
VAL_NT, VAL_NX = 96, 192    # held-out residual grid for iterate selection
GN_LAM0 = 1e-4              # LM damping, relative to trace(J J^T)/N
GN_ETAS = (0.0, 0.1, 0.25, 0.5, 0.8, 1.0)   # line-search grid (0 = reject)
GN_VAL_EVERY = 5
if _SMOKE:
    SOAP_WALL, GN_NT, GN_NX, COS_STEPS = 0.3 * TRAIN_TIME, 12, 12, 800
    VAL_NT, VAL_NX, SAVE_MARGIN_S = 16, 24, 1.0
_TELE: dict = {}  # live telemetry for the b02_tele diagnostic probe


def _loss_fn(params, key, eps):
    kt, kx = random.split(key)
    dt = T_MAX / NT
    ts = (jnp.arange(NT, dtype=jnp.float32) + random.uniform(kt, (NT,))) * dt
    # collocation uniform in the stretched coordinate phi -> automatically
    # ~1/BETA times denser at the shock; 1/p = dx/dphi volume weights keep the
    # per-slice loss an unbiased estimate of the x-integral of r^2 (kb1 086)
    phis = (jnp.arange(NX, dtype=jnp.float32) + random.uniform(kx, ())) \
        * (jnp.pi / NX)
    xs = _unstretch(phis)
    vw = _dx_dphi(xs)
    T = jnp.repeat(ts, NX)
    X = jnp.tile(xs, NT)
    r = _res_batch(params, T, X).reshape(NT, NX)
    Lt = jnp.sum(vw[None, :] * r ** 2, axis=1) / jnp.sum(vw)
    cum = jnp.concatenate([jnp.zeros((1,), Lt.dtype), jnp.cumsum(Lt)[:-1]])
    w = lax.stop_gradient(jnp.exp(-eps * cum))
    return jnp.mean(w * Lt), (jnp.min(w), jnp.mean(Lt))


# ══════════ Stage 2: low-rank Levenberg-Marquardt (Gauss-Newton) ══════════


def _gn_grid(jit_t, jit_x, nt, nx):
    """Stratified (t, phi) sample + sqrt volume weights, sum(sw^2) = 1.
    ``jit_t``/``jit_x`` are [nt] / [nt, nx] offsets in [0, 1): a per-point x
    jitter gives nt*nx distinct x locations, so a spatial mode cannot hide
    between collocation lines (which is what b02_gn's GN exploited)."""
    ts = (jnp.arange(nt, dtype=jnp.float32) + jit_t) * (T_MAX / nt)
    phis = (jnp.arange(nx, dtype=jnp.float32)[None, :] + jit_x) * (jnp.pi / nx)
    xs = _unstretch(phis)                       # [nt, nx]
    vw = _dx_dphi(xs)
    T = jnp.repeat(ts, nx)
    sw = jnp.sqrt(vw.ravel() / jnp.sum(vw))
    return T, xs.ravel(), sw


def _gn_sample(key):
    kt, kx = random.split(key)
    return _gn_grid(random.uniform(kt, (GN_NT,)),
                    random.uniform(kx, (GN_NT, GN_NX)), GN_NT, GN_NX)


def _make_gn(params0):
    """Build the jitted GN step for the flattened parameter vector."""
    flat0, unravel = ravel_pytree(params0)

    def res_flat(flat, t, x):
        return _residual(unravel(flat), t, x)

    res_b = vmap(res_flat, in_axes=(None, 0, 0))
    jac_b = vmap(jax.grad(res_flat, argnums=0), in_axes=(None, 0, 0))
    etas = jnp.asarray(GN_ETAS, dtype=jnp.float32)

    # held-out residual grid: half-offset cells, finer than any training sample,
    # so overfitting the training collocation shows up immediately
    Tv, Xv, swv = _gn_grid(jnp.full((VAL_NT,), 0.5),
                           jnp.full((VAL_NT, VAL_NX), 0.5), VAL_NT, VAL_NX)

    @jit
    def val_loss(flat):
        return jnp.sum((swv * res_b(flat, Tv, Xv)) ** 2)

    @jit
    def gn_step(flat, key, lam):
        T, X, sw = _gn_sample(key)
        r = sw * res_b(flat, T, X)                      # [N]
        J = jac_b(flat, T, X) * sw[:, None]             # [N, P]
        M = J @ J.T
        n = M.shape[0]
        M = M + (lam * jnp.trace(M) / n + 1e-30) * jnp.eye(n, dtype=M.dtype)
        alpha = jax.scipy.linalg.solve(M, r, assume_a="pos")
        d = -(J.T @ alpha)
        d = jnp.where(jnp.isfinite(d), d, 0.0)

        def trial(eta):
            rr = sw * res_b(flat + eta * d, T, X)
            return jnp.sum(rr ** 2)

        losses = vmap(trial)(etas)
        losses = jnp.where(jnp.isfinite(losses), losses, jnp.inf)
        i = jnp.argmin(losses)
        return flat + etas[i] * d, jnp.sum(r ** 2), losses[i], etas[i]

    return flat0, unravel, gn_step, val_loss


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
    soap_end = min(_T0 + SOAP_WALL, deadline)
    step_count = 0
    # warmup chunk: compiles the whole scan (real steps, counted)
    n = jnp.int32(0)
    params, opt_state, key, loss, wmin, lr_mean, ema, n = run_chunk(
        params, opt_state, key, jnp.float32(EPS_SCHED[eps_i]), ema, n)
    step_count += CHUNK
    for it in range(1, MAX_ITER):
        if time.time() > soap_end:
            break
        params, opt_state, key, loss, wmin, lr_mean, ema, n = run_chunk(
            params, opt_state, key, jnp.float32(EPS_SCHED[eps_i]), ema, n)
        step_count += CHUNK
        if float(wmin) > W_DELTA and eps_i < len(EPS_SCHED) - 1:
            eps_i += 1
        _TELE.update(phase="soap", eps=EPS_SCHED[eps_i], wmin=float(wmin),
                     loss=float(loss), step=step_count)
        if eval_callback is not None and (it % 200 == 0):
            eval_callback(params, step_count, float(loss))

    # ── stage 2: Gauss-Newton refinement, warm-started from the EMA weights ──
    # ema is seeded from the init weights, so no bias correction is needed:
    # after ~25k steps their residual share is EMA_D**n ~ e^-25
    flat, unravel, gn_step, val_loss = _make_gn(ema)
    lam = GN_LAM0
    best_flat, best_val = flat, float(val_loss(flat))
    gn_n = 0
    while time.time() < deadline:
        flat_new, l0, l1, eta = gn_step(flat, key, jnp.float32(lam))
        key = random.split(key)[0]
        l1, eta = float(l1), float(eta)
        if not np.isfinite(l1):
            lam = min(lam * 100.0, 1e3)
            continue
        flat = flat_new
        gn_n += 1
        if eta == 0.0:
            lam = min(lam * 5.0, 1e3)          # rejected: trust the model less
        elif eta >= 0.8:
            lam = max(lam / 2.0, 1e-9)         # full step accepted: push harder
        if gn_n % GN_VAL_EVERY == 0:
            v = float(val_loss(flat))
            if np.isfinite(v) and v < best_val:
                best_val, best_flat = v, flat
            _TELE.update(phase="gn", loss=l1, val=v, gn=gn_n, lam=lam, eta=eta,
                         step=step_count)
            if eval_callback is not None and (gn_n % 50 == 0):
                eval_callback(unravel(flat), step_count + gn_n, v)
    if gn_n == 0:
        return ema, step_count
    return unravel(best_flat), step_count + gn_n
