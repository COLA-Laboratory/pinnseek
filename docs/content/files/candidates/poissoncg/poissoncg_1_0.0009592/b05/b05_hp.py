"""b05_lean with the jet matmuls in true fp32 (precision HIGHEST) instead of
the default TF32.  TF32 keeps 10 mantissa bits, and the residual is a
cancellation of O(100) terms, so its rounding may set the accuracy floor.
vs b05_lean: precision=HIGHEST on every layer dot."""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 1
INPUT_NAMES  = ("x", "y", "z")
FIELD_NAMES  = ("u",)

# --- Domain ---
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.0
HAS_TIME = False

# --- Equation structure ---
TIME_DERIV_ORDER        = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 1   # r
HAS_SOURCE              = True   # and it does not vanish with u, so u ≡ 0 is
                                 # nowhere near a solution of the residual

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi",
                  "sphere_0", "sphere_1")
PERIODIC_AXES  = ()       # none: homogeneous Neumann on every boundary
GEOMETRY_KIND  = "cube_minus_spheres"

# --- Geometry (unit cube with two spherical voids) ---
SPHERES = ((0.4, 0.3, 0.6, 0.2),    # (cx, cy, cz, radius), removed from the cube
           (0.6, 0.7, 0.6, 0.2))

# --- Physical parameters ---
# Residual: -(u_xx + u_yy + u_zz) + K(z)^2 u - f(x, y, z) = 0
Z_IFACE      = 0.5      # the screening wavenumber jumps across this plane
K_LO, K_HI   = 8.0, 10.0            # K(z) = K_LO for z < Z_IFACE, else K_HI;
                                    # only u_zz jumps there, u and grad u stay
                                    # continuous — the jump is zeroth order
SRC_A1, SRC_A2 = 20.0, 100.0        # source amplitudes
SRC_M1, SRC_M2, SRC_M3 = 1.0, 10.0, 5.0   # source wavenumbers, sin(M pi ·);
                                    # M2 = 10 sets the finest scale, 5 periods
                                    # of the source across y

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 60.0      # total wall-clock seconds for the WHOLE process,
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
from jax import jit, random
import numpy as np
import optax


def screening(z):
    """K(z)^2 — the zeroth-order coefficient, piecewise constant in z."""
    return jnp.where(z < Z_IFACE, K_LO ** 2, K_HI ** 2)


def source(x, y, z):
    """The forcing f(x, y, z); independent of u."""
    r2 = x ** 2 + y ** 2 + z ** 2
    part1 = (jnp.exp(jnp.sin(SRC_M1 * jnp.pi * x)
                     + jnp.sin(SRC_M2 * jnp.pi * y)
                     + jnp.sin(SRC_M3 * jnp.pi * z))
             * (r2 - 1.0) / (r2 + 1.0))
    part2 = (jnp.sin(SRC_M1 * jnp.pi * x)
             * jnp.sin(SRC_M2 * jnp.pi * y)
             * jnp.sin(SRC_M3 * jnp.pi * z))
    return SRC_A1 * part1 + SRC_A2 * part2


# ══════════ Architecture (b01 lineage), one dot per layer ══════════
# Inputs are only cos(k pi s)/k, k = 1..N_HARM, so du/dn == 0 on all six cube
# faces identically (kb1 036 hard BC); the ladder spans the source scales
# (kb1 020/018) and the 1/k amplitude tames the second-derivative ladder.
# The value row, the 3 first-derivative rows and the 3 second-derivative rows
# of the forward jet are stacked into ONE [N, 7, F] operand, so a layer costs a
# single dot instead of three — the 60 s step is half kernel-launch overhead.

N_HARM = 12
WIDTH = 192
DEPTH = 4
_KS = jnp.arange(1, N_HARM + 1)
NROW = 7                      # 1 value + 3 first + 3 second derivative rows


def init_params(key):
    """He-uniform dense stack, zero biases."""
    dims = [3 * N_HARM + 1] + [WIDTH] * DEPTH
    layers = []
    for i in range(DEPTH):
        key, k = random.split(key)
        lim = np.sqrt(6.0 / dims[i])
        layers.append({"W": random.uniform(k, (dims[i], dims[i + 1]),
                                           minval=-lim, maxval=lim),
                       "b": jnp.zeros(dims[i + 1])})
    key, k = random.split(key)
    lim = np.sqrt(6.0 / WIDTH)
    return {"layers": layers,
            "out": {"W": random.uniform(k, (WIDTH,), minval=-lim, maxval=lim),
                    "b": jnp.zeros(())}}


def _interface(z):
    g = jnp.where(z < Z_IFACE, -0.5 * z ** 2, 0.5 * z ** 2 - z + 0.25)
    gp = jnp.where(z < Z_IFACE, -z, z - 1.0)
    gpp = jnp.where(z < Z_IFACE, -jnp.ones_like(z), jnp.ones_like(z))
    return g, gp, gpp


def _embed(X, nrow):
    """[N, nrow, F]: the embedding stacked with its analytic derivative rows."""
    N = X.shape[0]
    a = X[:, :, None] * (_KS * jnp.pi)
    c, s = jnp.cos(a), jnp.sin(a)
    eye = jnp.eye(INPUT_DIM)[None, :, :, None]
    h0 = (c / _KS).reshape(N, 1, INPUT_DIM * N_HARM)
    h1 = (eye * (-jnp.pi * s)[:, None, :, :]).reshape(N, INPUT_DIM,
                                                      INPUT_DIM * N_HARM)
    z = X[:, 2]
    g, gp, gpp = _interface(z)
    zc = jnp.zeros_like(z)
    rows = [jnp.concatenate([h0, (8.0 * g + 1.0)[:, None, None]], -1),
            jnp.concatenate([h1, 8.0 * jnp.stack([zc, zc, gp], 1)[:, :, None]], -1)]
    if nrow == NROW:
        h2 = (eye * (-(_KS * jnp.pi ** 2) * c)[:, None, :, :]).reshape(
            N, INPUT_DIM, INPUT_DIM * N_HARM)
        rows.append(jnp.concatenate(
            [h2, 8.0 * jnp.stack([zc, zc, gpp], 1)[:, :, None]], -1))
    return jnp.concatenate(rows, 1)


def _forward(p, X, nrow):
    """Propagate the stacked jet; returns H [N, nrow, WIDTH] of the last layer."""
    H = _embed(X, nrow)
    for L in p["layers"]:
        Z = jnp.einsum("nkf,fg->nkg", H, L["W"],
                       precision=jax.lax.Precision.HIGHEST)
        t = jnp.tanh(Z[:, 0] + L["b"])
        t1 = 1.0 - t ** 2
        z1 = Z[:, 1:4]
        rows = [t[:, None], t1[:, None] * z1]
        if nrow == NROW:
            rows.append((-2.0 * t * t1)[:, None] * z1 ** 2
                        + t1[:, None] * Z[:, 4:7])
        H = jnp.concatenate(rows, 1)
    return H


def u_and_lap(p, X):
    H = _forward(p, X, NROW)
    w = p["out"]["W"]
    return H[:, 0] @ w + p["out"]["b"], (H[:, 4:7] @ w).sum(1)


def grad_u(p, X):
    return _forward(p, X, 4)[:, 1:4] @ p["out"]["W"]


# ══════════ Training hyperparameters (b04's tuned champion) ══════════

STRATA = (8, 8, 6)    # jittered-lattice strata; prod(STRATA) == BS_COL
BS_COL = 384          # fresh uniform samples of the cube per step (masked)
BS_SPH = 128          # fresh sphere-surface samples per step
W_SPH = 0.3           # weight on the sphere Neumann penalty
RES_SCALE = 1.0e4     # ~mean(f^2): puts the residual loss at O(1)
LR_MAX = 4.0e-3
LR_END = 3.0e-5
WARMUP = 200
CHUNK = 50            # steps between host syncs / LR refreshes
EMA_DECAY = 0.9997    # Polyak tail average
SAVE_MARGIN_S = 4.0


def _make_train(rng_init):
    key, rng = random.split(rng_init)
    params = init_params(key)

    _CTR = jnp.array([[c[0], c[1], c[2]] for c in SPHERES])
    _RAD = jnp.array([c[3] for c in SPHERES])
    _NSTRAT = jnp.array(STRATA, dtype=jnp.float32)
    _g = jnp.meshgrid(*[jnp.arange(n) for n in STRATA], indexing="ij")
    _LAT = jnp.stack([q.reshape(-1) for q in _g], axis=1).astype(jnp.float32)

    def sample(key):
        ka, kb, kc = random.split(key, 3)
        pts = (_LAT + random.uniform(ka, (BS_COL, INPUT_DIM))) / _NSTRAT
        keep = jnp.ones(BS_COL)
        for i in range(len(SPHERES)):
            d2 = jnp.sum((pts - _CTR[i]) ** 2, axis=1)
            keep = keep * (d2 > _RAD[i] ** 2)
        g = random.normal(kb, (BS_SPH, INPUT_DIM))
        unit = g / jnp.linalg.norm(g, axis=1, keepdims=True)
        j = random.randint(kc, (BS_SPH,), 0, len(SPHERES))
        return pts, keep, _CTR[j] + _RAD[j][:, None] * unit, -unit

    def loss_fn(p, pts, keep, sph, nrm):
        H = _forward(p, jnp.concatenate([pts, sph], 0), NROW)
        w = p["out"]["W"]
        u = H[:BS_COL, 0] @ w + p["out"]["b"]
        lap = (H[:BS_COL, 4:7] @ w).sum(1)
        r = (-lap + screening(pts[:, 2]) * u
             - source(pts[:, 0], pts[:, 1], pts[:, 2]))
        pde = jnp.sum(r ** 2 * keep) / jnp.sum(keep)
        bc = jnp.mean(jnp.sum((H[BS_COL:, 1:4] @ w) * nrm, axis=1) ** 2)
        return pde / RES_SCALE + W_SPH * bc

    optimizer = optax.adam(1.0)          # unit step; lr applied explicitly
    opt_state = optimizer.init(params)

    @jit
    def update(p, st, ema, key, lr):
        k, ks = random.split(key)
        pts, keep, sph, nrm = sample(ks)
        loss, grads = jax.value_and_grad(loss_fn)(p, pts, keep, sph, nrm)
        upd, st = optimizer.update(grads, st)
        p = optax.apply_updates(p, jax.tree_util.tree_map(lambda q: lr * q, upd))
        ema = jax.tree_util.tree_map(
            lambda e, q: EMA_DECAY * e + (1.0 - EMA_DECAY) * q, ema, p)
        return p, st, ema, loss, k

    return params, opt_state, update, rng


# ══════════ Public contract ══════════


def predict_fn(params, X):
    """X: [BS, 3] with columns (x, y, z). Returns {"u"} of shape [BS]."""
    H = _forward(params, jnp.asarray(X), 1)
    return {"u": H[:, 0] @ params["out"]["W"] + params["out"]["b"]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam with a time-based exponential LR decay."""
    params, opt_state, update, rng_local = _make_train(rng)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    ema = params
    params, opt_state, ema, loss, rng_local = update(
        params, opt_state, ema, rng_local, jnp.float32(0.0))
    jax.block_until_ready(loss)
    step_count = 1
    t_start = time.time()
    span = max(deadline - t_start, 1e-3)

    while True:
        now = time.time()
        if now > deadline:
            break
        frac = min(max((now - t_start) / span, 0.0), 1.0)
        warm = min(1.0, step_count / WARMUP)
        lr = jnp.float32(LR_MAX * warm * (LR_END / LR_MAX) ** frac)
        for _ in range(CHUNK):
            params, opt_state, ema, loss, rng_local = update(
                params, opt_state, ema, rng_local, lr)
        jax.block_until_ready(loss)
        step_count += CHUNK

    return ema, step_count
