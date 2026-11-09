"""Decaying GN step length with twice the residual rows per step.
vs b02_gneta: BS_COL 1024 -> 2048, BS_SPH 256 -> 512.
"""
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
from jax import jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


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


# ══════════ Architecture (b01's cosine-harmonic embedding) ══════════
# cos(k pi s) with 1/k amplitude makes du/dn == 0 on all six cube faces
# exact (kb1 036) and spans the source scales (kb1 020/018); one analytic
# interface feature g(z) whose g'' jumps at z = Z_IFACE.

N_HARM = 20
WIDTH = 128
DEPTH = 4


class PINN(nn.Module):
    hidden_width: int = WIDTH
    n_hidden_layers: int = DEPTH
    n_harm: int = N_HARM

    @nn.compact
    def __call__(self, xyz):
        kinit = jax.nn.initializers.he_uniform()
        ks = jnp.arange(1, self.n_harm + 1) * jnp.pi
        amp = 1.0 / jnp.arange(1, self.n_harm + 1)
        h = (jnp.cos(xyz[..., :, None] * ks) * amp).reshape(
            *xyz.shape[:-1], INPUT_DIM * self.n_harm)
        z = xyz[..., 2]
        g = jnp.where(z < Z_IFACE, -0.5 * z ** 2, 0.5 * z ** 2 - z + 0.25)
        h = jnp.concatenate([h, 8.0 * g[..., None] + 1.0], axis=-1)
        for _ in range(self.n_hidden_layers):
            h = nn.tanh(nn.Dense(self.hidden_width, kernel_init=kinit)(h))
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)


# ══════════ Gauss-Newton hyperparameters ══════════

STRATA = (8, 16, 16)
BS_COL = 2048      # interior residual rows, resampled every step
BS_SPH = 512      # sphere-Neumann residual rows
W_SPH = 0.3
RES_SCALE = 1.0e4
LAM = 1.0e-3            # Levenberg damping, relative to mean(diag)
ETA = 1.0            # GN step length at the start
ETA_END = 0.05    # ... decayed to this by the wall (stochastic-approximation
                       # noise floor scales with the step length, exactly as the
                       # Adam LR floor did for b01)
EMA_DECAY = 0.99
LOWRANK = True    # True: (J J^T + lam I) in row space; False: (J^T J + lam I)
CHUNK = 10
SAVE_MARGIN_S = 4.0


def _make_train(rng_init):
    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def u_s(p, xyz):
        return model.apply(unravel(p), xyz[None, :])[0, 0]

    grad_u = jax.grad(u_s, argnums=1)
    _E = jnp.eye(INPUT_DIM)

    def u_and_lap(p, xyz):
        """u and its Laplacian from three jvp-of-grad passes (AD only)."""
        vg = lambda x: jax.value_and_grad(u_s, argnums=1)(p, x)
        (u, _), (_, h0) = jax.jvp(vg, (xyz,), (_E[0],))
        _, (_, h1) = jax.jvp(vg, (xyz,), (_E[1],))
        _, (_, h2) = jax.jvp(vg, (xyz,), (_E[2],))
        return u, h0[0] + h1[1] + h2[2]

    # residual rows, pre-weighted so sum(r^2) reproduces b01's loss
    W_C = 1.0 / jnp.sqrt(BS_COL * RES_SCALE)
    W_S = jnp.sqrt(W_SPH / BS_SPH)

    def r_col(p, xyz, kp):
        u, lap = u_and_lap(p, xyz)
        r = -lap + screening(xyz[2]) * u - source(xyz[0], xyz[1], xyz[2])
        return W_C * jnp.sqrt(kp) * r

    def r_sph(p, xyz, nrm):
        return W_S * jnp.dot(grad_u(p, xyz), nrm)

    jac_col = vmap(jax.value_and_grad(r_col), (None, 0, 0))
    jac_sph = vmap(jax.value_and_grad(r_sph), (None, 0, 0))

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

    @jit
    def update(p, ema, key, eta):
        """One damped Gauss-Newton step (kb1 063, kb1 108 sec. B)."""
        k, ks = random.split(key)
        pts, keep, sph, nrm = sample(ks)
        rc, Jc = jac_col(p, pts, keep)
        rs, Js = jac_sph(p, sph, nrm)
        r = jnp.concatenate([rc, rs])
        J = jnp.concatenate([Jc, Js], axis=0)              # [N, P]
        if LOWRANK:
            M = J @ J.T
            M = M + LAM * (jnp.mean(jnp.diag(M)) + 1e-30) * jnp.eye(M.shape[0])
            delta = J.T @ jax.scipy.linalg.solve(M, r, assume_a="pos")
        else:
            G = J.T @ J
            G = G + LAM * (jnp.mean(jnp.diag(G)) + 1e-30) * jnp.eye(G.shape[0])
            delta = jax.scipy.linalg.solve(G, J.T @ r, assume_a="pos")
        p = p - eta * delta
        ema = EMA_DECAY * ema + (1.0 - EMA_DECAY) * p
        return p, ema, jnp.sum(r ** 2), k

    return flat_params, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: [BS, 3] with columns (x, y, z). Returns {"u"} of shape [BS]."""
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline damped Gauss-Newton (natural gradient) training."""
    params_flat, update, unravel, rng_local = _make_train(rng)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    ema = params_flat
    params_flat, ema, loss, rng_local = update(
        params_flat, ema, rng_local, jnp.float32(ETA))
    jax.block_until_ready(params_flat)
    step_count = 1
    t_start = time.time()
    span = max(deadline - t_start, 1e-3)

    while True:
        now = time.time()
        if now > deadline:
            break
        frac = min(max((now - t_start) / span, 0.0), 1.0)
        eta = jnp.float32(ETA * (ETA_END / ETA) ** frac)
        for _ in range(CHUNK):
            params_flat, ema, loss, rng_local = update(
                params_flat, ema, rng_local, eta)
        jax.block_until_ready(params_flat)
        step_count += CHUNK

    out = ema if EMA_DECAY > 0.0 else params_flat
    return unravel(out), step_count
