"""Cosine-harmonic-embedded PINN with K=5 harmonics (K sweep; sibling of b01_cos08).
on all six cube faces EXACT, leaving only the two sphere surfaces soft.
vs parent (task/baseline.py): cosine input embedding, jvp-of-grad Laplacian,
fresh masked collocation each step, time-based Adam LR decay."""
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


# ══════════ Architecture ══════════
# Every input feature is cos(k pi s) with integer k, whose derivative
# -k pi sin(k pi s) vanishes at s = 0 and s = 1.  Any smooth function of
# those features therefore has du/dn == 0 on all six cube faces by
# construction (verified to float32 roundoff) — the cube Neumann BC is a
# hard constraint, not a penalty (kb1 036 exact BC imposition), and the
# ladder k = 1..K doubles as a deterministic multi-scale Fourier embedding
# (kb1 020 / 018) matched to the source's finest scale sin(10 pi y).

N_HARM = 5
WIDTH = 128
DEPTH = 5


class PINN(nn.Module):
    hidden_width: int = WIDTH
    n_hidden_layers: int = DEPTH
    n_harm: int = N_HARM

    @nn.compact
    def __call__(self, xyz):
        kinit = jax.nn.initializers.he_uniform()
        ks = jnp.arange(1, self.n_harm + 1) * jnp.pi
        h = jnp.cos(xyz[..., :, None] * ks).reshape(*xyz.shape[:-1],
                                                    INPUT_DIM * self.n_harm)
        for _ in range(self.n_hidden_layers):
            h = nn.tanh(nn.Dense(self.hidden_width, kernel_init=kinit)(h))
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)


# ══════════ Training hyperparameters ══════════

BS_COL = 4096         # fresh uniform samples of the cube per step (masked)
BS_SPH = 512          # fresh sphere-surface samples per step
W_SPH = 1.0           # weight on the sphere Neumann penalty
RES_SCALE = 1.0e4     # ~mean(f^2): puts the residual loss at O(1)
LR_MAX = 2.0e-3
LR_END = 1.0e-4
WARMUP = 200
CHUNK = 50            # steps between host syncs / LR refreshes
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

    def res_pt(p, xyz):
        u, lap = u_and_lap(p, xyz)
        r = -lap + screening(xyz[2]) * u - source(xyz[0], xyz[1], xyz[2])
        return r ** 2

    def sph_pt(p, xyz, nrm):
        return jnp.dot(grad_u(p, xyz), nrm) ** 2

    res_b = vmap(res_pt, in_axes=(None, 0))
    sph_b = vmap(sph_pt, in_axes=(None, 0, 0))

    _CTR = jnp.array([[c[0], c[1], c[2]] for c in SPHERES])
    _RAD = jnp.array([c[3] for c in SPHERES])

    def sample(key):
        ka, kb, kc = random.split(key, 3)
        pts = random.uniform(ka, (BS_COL, INPUT_DIM))
        keep = jnp.ones(BS_COL)
        for i in range(len(SPHERES)):
            d2 = jnp.sum((pts - _CTR[i]) ** 2, axis=1)
            keep = keep * (d2 > _RAD[i] ** 2)
        g = random.normal(kb, (BS_SPH, INPUT_DIM))
        unit = g / jnp.linalg.norm(g, axis=1, keepdims=True)
        j = random.randint(kc, (BS_SPH,), 0, len(SPHERES))
        sph = _CTR[j] + _RAD[j][:, None] * unit
        return pts, keep, sph, -unit

    def loss_fn(p, pts, keep, sph, nrm):
        pde = jnp.sum(res_b(p, pts) * keep) / jnp.sum(keep)
        bc = jnp.mean(sph_b(p, sph, nrm))
        return pde / RES_SCALE + W_SPH * bc

    optimizer = optax.adam(1.0)          # unit step; lr applied explicitly
    opt_state = optimizer.init(flat_params)

    @jit
    def update(p, st, key, lr):
        k, ks_ = random.split(key)
        pts, keep, sph, nrm = sample(ks_)
        loss, grads = jax.value_and_grad(loss_fn)(p, pts, keep, sph, nrm)
        upd, st = optimizer.update(grads, st)
        p = optax.apply_updates(p, jax.tree_util.tree_map(lambda u: lr * u, upd))
        return p, st, loss, k

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: [BS, 3] with columns (x, y, z). Returns {"u"} of shape [BS]."""
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam with a time-based exponential LR decay."""
    params_flat, opt_state, update, unravel, rng_local = _make_train(rng)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    params_flat, opt_state, loss, rng_local = update(
        params_flat, opt_state, rng_local, jnp.float32(0.0))
    jax.block_until_ready(params_flat)
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
            params_flat, opt_state, loss, rng_local = update(
                params_flat, opt_state, rng_local, lr)
        jax.block_until_ready(params_flat)
        step_count += CHUNK

    return unravel(params_flat), step_count
