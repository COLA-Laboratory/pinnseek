"""b02_v5: Adam -> TSONN pseudo-time L-BFGS (kb1:073) on NACA0012.
Change vs parent (b01_v18): after Adam, replace plain L-BFGS with implicit
pseudo-time stepping — add forcing mu*(q-q_n) to the steady residual and
refresh q_n each outer step (conditions the high-Re operator). float32.
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

# --- Domain (farfield box; airfoil cut out via level set in collocation.csv) ---
X_MIN, X_MAX = -3.0, 5.0
Y_MIN, Y_MAX = -2.0, 2.0
HAS_TIME = False

# --- Equation structure ---
TIME_DERIV_ORDER        = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 3   # ru, rv, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("airfoil", "farfield")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "external_airfoil"

# --- Physical parameters ---
RE      = 1000.0       # NS-NACA0012 Re=1000
NU      = 1.0 / RE
U_INF   = 1.0          # freestream speed (v_inf = 0)
AOA_DEG = 7.0          # angle of attack; airfoil rotated -AOA about (0.5, 0)

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 180.0     # total wall-clock seconds for the WHOLE process,
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
from jax import jacfwd, jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


def _locate(name: str) -> Path:
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


class PINN(nn.Module):
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, xy):
        kinit = jax.nn.initializers.he_uniform()
        h = xy
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)


def _naca_airfoil_points(n: int = 400, thickness: float = 0.12) -> np.ndarray:
    beta = np.linspace(0.0, np.pi, n // 2)
    xc = 0.5 * (1.0 - np.cos(beta))
    yt = 5 * thickness * (0.2969 * np.sqrt(xc) - 0.1260 * xc
                          - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)
    xa = np.concatenate([xc, xc])
    ya = np.concatenate([yt, -yt])
    a = np.radians(-AOA_DEG)
    xr = 0.5 + (xa - 0.5) * np.cos(a) - ya * np.sin(a)
    yr = 0.0 + (xa - 0.5) * np.sin(a) + ya * np.cos(a)
    return np.stack([xr, yr], axis=1).astype(np.float32)


def _build_dataset():
    ref = np.loadtxt(_locate("collocation.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    xy, phi = ref[:, 0:2], ref[:, 2]
    edge = (
        (xy[:, 0] == X_MIN) | (xy[:, 0] == X_MAX)
        | (xy[:, 1] == Y_MIN) | (xy[:, 1] == Y_MAX)
    )
    fluid = (phi <= 0.0)
    pts_int = xy[fluid & ~edge].astype(np.float32)
    pts_far = xy[edge].astype(np.float32)
    pts_air = _naca_airfoil_points(400)
    pts_bc = np.vstack([pts_far, pts_air])
    u_bc = np.concatenate([np.full(len(pts_far), U_INF, np.float32),
                           np.zeros(len(pts_air), np.float32)])
    v_bc = np.zeros(len(pts_bc), np.float32)
    return (jnp.array(pts_int), jnp.array(pts_bc), jnp.array(u_bc), jnp.array(v_bc))


LR        = 1e-3
WEIGHT_BC = 1.0
BS_INT    = 400
BS_BC     = 100
ADAM_WALL = 70.0
LBFGS_INT = 6144
LBFGS_MEM = 20
MU        = 3.0      # pseudo-time forcing 1/Delta_t (kb1:073 uses Delta_t~0.3)
OUTER_S   = 3.0      # wall-seconds per pseudo-time outer step (q_n refresh)
SAVE_MARGIN_S = 6.0


def _make_train(rng_init):
    pts_int, pts_bc, u_bc, v_bc = _build_dataset()
    n_int, n_bc = len(pts_int), len(pts_bc)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM], dtype=jnp.float32))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def predict_one(pf, xy):
        return model.apply(unravel(pf), xy[None, :])[0]
    d1_fn = jacfwd(predict_one, argnums=1)
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)

    def residuals(pf, xy):
        uvp = predict_one(pf, xy)
        u, v = uvp[0], uvp[1]
        d1 = d1_fn(pf, xy); d2 = d2_fn(pf, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        u_xx, u_yy = d2[0, 0, 0], d2[0, 1, 1]
        v_xx, v_yy = d2[1, 0, 0], d2[1, 1, 1]
        r_c  = u_x + v_y
        r_mx = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
        r_my = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
        return r_c, r_mx, r_my

    def residual_sq(pf, xy):
        r_c, r_mx, r_my = residuals(pf, xy)
        return r_c ** 2 + r_mx ** 2 + r_my ** 2
    residual_batch = vmap(residual_sq, in_axes=(None, 0))

    def loss_fn(pf, xy_int, xy_bc, ub, vb):
        pde_loss = jnp.mean(residual_batch(pf, xy_int))
        bc = vmap(predict_one, in_axes=(None, 0))(pf, xy_bc)
        bc_loss = jnp.mean((bc[:, 0] - ub) ** 2 + (bc[:, 1] - vb) ** 2)
        return pde_loss + WEIGHT_BC * bc_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(pf, opt_state, key_pair):
        k1, k2 = key_pair
        i_int = random.choice(k1, n_int, (BS_INT,))
        i_bc = random.choice(k2, n_bc, (BS_BC,))
        loss, grads = loss_and_grad(pf, pts_int[i_int], pts_bc[i_bc], u_bc[i_bc], v_bc[i_bc])
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(pf, updates), opt_state, loss

    idx = random.choice(random.PRNGKey(12345), n_int, (LBFGS_INT,), replace=False)
    xi = pts_int[idx]

    # --- q_n snapshot on the fixed subset (stop-grad values) ---
    pred_batch = jit(vmap(predict_one, in_axes=(None, 0)))

    # --- TSONN forced loss + true (unforced) loss on the fixed subset ---
    def loss_forced(pf, un, vn, pn):
        def per_pt(xy, u_n, v_n, p_n):
            r_c, r_mx, r_my = residuals(pf, xy)
            uvp = predict_one(pf, xy)
            rc = r_c + MU * (uvp[2] - p_n)
            rx = r_mx + MU * (uvp[0] - u_n)
            ry = r_my + MU * (uvp[1] - v_n)
            return rc ** 2 + rx ** 2 + ry ** 2
        pde = jnp.mean(vmap(per_pt)(xi, un, vn, pn))
        bc = vmap(predict_one, in_axes=(None, 0))(pf, pts_bc)
        bc_loss = jnp.mean((bc[:, 0] - u_bc) ** 2 + (bc[:, 1] - v_bc) ** 2)
        return pde + WEIGHT_BC * bc_loss

    def true_loss(pf):
        pde = jnp.mean(residual_batch(pf, xi))
        bc = vmap(predict_one, in_axes=(None, 0))(pf, pts_bc)
        bc_loss = jnp.mean((bc[:, 0] - u_bc) ** 2 + (bc[:, 1] - v_bc) ** 2)
        return pde + WEIGHT_BC * bc_loss
    true_loss_j = jit(true_loss)

    lbfgs = optax.lbfgs(memory_size=LBFGS_MEM)
    vgs = optax.value_and_grad_from_state(loss_forced)

    @jit
    def lbfgs_step(pf, state, un, vn, pn):
        value, grad = vgs(pf, state=state, un=un, vn=vn, pn=pn)
        updates, state = lbfgs.update(grad, state, pf, value=value, grad=grad,
                                      value_fn=lambda p: loss_forced(p, un, vn, pn),
                                      un=un, vn=vn, pn=pn)
        return optax.apply_updates(pf, updates), state

    return (flat_params, opt_state, update, unravel, xi, pred_batch,
            lbfgs, lbfgs_step, true_loss_j, rng)


_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    (flat_params, opt_state, update, unravel, xi, pred_batch,
     lbfgs, lbfgs_step, true_loss_j, rng_local) = _make_train(rng)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S

    # ── Phase 1: Adam warm start (float32) ──
    key1, key2, rng_local = random.split(rng_local, 3)
    flat_params, opt_state, loss = update(flat_params, opt_state, (key1, key2))
    step_count = 1
    adam_deadline = min(_T0 + ADAM_WALL, deadline)
    for it in range(1, MAX_ITER):
        if time.time() > adam_deadline:
            break
        key1, key2, rng_local = random.split(rng_local, 3)
        flat_params, opt_state, loss = update(flat_params, opt_state, (key1, key2))
        step_count = it + 1

    # ── Phase 2: TSONN pseudo-time outer loop with inner L-BFGS ──
    best_flat = flat_params
    best_loss = float(true_loss_j(flat_params))
    # warm-up compile of lbfgs_step (counted)
    snap = pred_batch(flat_params, xi)
    un, vn, pn = snap[:, 0], snap[:, 1], snap[:, 2]
    state = lbfgs.init(flat_params)
    flat_params, state = lbfgs_step(flat_params, state, un, vn, pn)
    step_count += 1

    while time.time() < deadline:
        snap = pred_batch(flat_params, xi)         # refresh q_n (stop-grad values)
        un, vn, pn = snap[:, 0], snap[:, 1], snap[:, 2]
        state = lbfgs.init(flat_params)            # new pseudo-time subproblem
        outer_deadline = min(time.time() + OUTER_S, deadline)
        while time.time() < outer_deadline:
            flat_params, state = lbfgs_step(flat_params, state, un, vn, pn)
            step_count += 1
        tl = float(true_loss_j(flat_params))
        if np.isfinite(tl) and tl < best_loss:
            best_loss, best_flat = tl, flat_params

    return unravel(best_flat), step_count
