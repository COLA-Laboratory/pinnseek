"""b02_v1: Adam warm-start -> float64 matrix-free Gauss-Newton (natural gradient).
Change vs parent (b01_v18): replace L-BFGS tail with a damped GN phase that
solves (J^T J + lam I) psi = J^T R by matrix-free CG (kb1:063, kb1:108),
in float64. Warm start = Adam 80 s.
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
jax.config.update("jax_enable_x64", True)  # double precision for the GN phase
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


# ══════════ Vanilla PINN: 5 × 100 tanh, output (u, v, p) ══════════

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


# ══════════ Geometry: fixed analytic NACA0012 surface ══════════

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


# ══════════ Data: fixed fluid grid + airfoil/farfield BCs ══════════

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


# ══════════ Training hyperparameters ══════════

LR        = 1e-3
WEIGHT_BC = 1.0
BS_INT    = 400
BS_BC     = 100
ADAM_WALL = 80.0     # wall-seconds of Adam warm start
GN_INT    = 2048     # fixed interior subset for the GN residual (float64)
GN_LAM0   = 1e-3     # initial Levenberg-Marquardt damping
GN_CG_IT  = 40       # CG iterations per GN step
SAVE_MARGIN_S = 6.0


# ══════════ Residual / loss (Adam phase, float32) ══════════

def _make_train(rng_init):
    pts_int, pts_bc, u_bc, v_bc = _build_dataset()
    n_int, n_bc = len(pts_int), len(pts_bc)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def predict_one(params_flat, xy):
        return model.apply(unravel(params_flat), xy[None, :])[0]   # [3]

    d1_fn = jacfwd(predict_one, argnums=1)
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)

    def residual_per_point(params_flat, xy):
        uvp = predict_one(params_flat, xy)
        u, v = uvp[0], uvp[1]
        d1 = d1_fn(params_flat, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        d2 = d2_fn(params_flat, xy)
        u_xx, u_yy = d2[0, 0, 0], d2[0, 1, 1]
        v_xx, v_yy = d2[1, 0, 0], d2[1, 1, 1]
        r_c  = u_x + v_y
        r_mx = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
        r_my = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
        return r_c ** 2 + r_mx ** 2 + r_my ** 2

    residual_batch = vmap(residual_per_point, in_axes=(None, 0))

    def loss_fn(params_flat, xy_int, xy_bc_batch, u_bc_batch, v_bc_batch):
        pde_loss = jnp.mean(residual_batch(params_flat, xy_int))
        bc_uvp = vmap(predict_one, in_axes=(None, 0))(params_flat, xy_bc_batch)
        u_pred, v_pred = bc_uvp[:, 0], bc_uvp[:, 1]
        bc_loss = jnp.mean((u_pred - u_bc_batch) ** 2 + (v_pred - v_bc_batch) ** 2)
        return pde_loss + WEIGHT_BC * bc_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, opt_state, key_pair):
        key1, key2 = key_pair
        idx_int = random.choice(key1, n_int, (BS_INT,))
        idx_bc  = random.choice(key2, n_bc,  (BS_BC,))
        loss, grads = loss_and_grad(params_flat, pts_int[idx_int],
                                    pts_bc[idx_bc], u_bc[idx_bc], v_bc[idx_bc])
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params_flat, updates), opt_state, loss

    return (flat_params, opt_state, update, unravel, model,
            pts_int, pts_bc, u_bc, v_bc, n_int, rng)


# ══════════ Gauss-Newton phase (float64, matrix-free CG) ══════════

def _make_gn(model, pts_int, pts_bc, u_bc, v_bc, n_int, gn_key):
    """Build a float64 residual VECTOR R(theta) and a jitted GN step.
    R stacks: sqrt(1/N_int)*[r_c,r_mx,r_my] over a fixed subset, and
    sqrt(WEIGHT_BC/N_bc)*[u-ubc, v-vbc] over all BC points, so
    0.5||R||^2 == 0.5*(mean pde + WEIGHT_BC*mean bc) — the champion loss/2.
    """
    idx = random.choice(gn_key, n_int, (GN_INT,), replace=False)
    xi = pts_int[idx].astype(jnp.float64)
    xb = pts_bc.astype(jnp.float64)
    ubc = u_bc.astype(jnp.float64)
    vbc = v_bc.astype(jnp.float64)
    N_int = xi.shape[0]
    N_bc = xb.shape[0]
    s_pde = jnp.sqrt(1.0 / N_int)
    s_bc = jnp.sqrt(WEIGHT_BC / N_bc)

    def predict_one(flat64, unravel64, xy):
        return model.apply(unravel64(flat64), xy[None, :])[0]

    def make_R(unravel64):
        d1_fn = jacfwd(lambda f, xy: predict_one(f, unravel64, xy), argnums=1)
        d2_fn = jacfwd(jacfwd(lambda f, xy: predict_one(f, unravel64, xy), argnums=1), argnums=1)

        def res_int(flat64, xy):
            uvp = predict_one(flat64, unravel64, xy)
            u, v = uvp[0], uvp[1]
            d1 = d1_fn(flat64, xy)
            u_x, u_y = d1[0, 0], d1[0, 1]
            v_x, v_y = d1[1, 0], d1[1, 1]
            p_x, p_y = d1[2, 0], d1[2, 1]
            d2 = d2_fn(flat64, xy)
            u_xx, u_yy = d2[0, 0, 0], d2[0, 1, 1]
            v_xx, v_yy = d2[1, 0, 0], d2[1, 1, 1]
            r_c  = u_x + v_y
            r_mx = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
            r_my = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
            return jnp.array([r_c, r_mx, r_my])

        def res_bc(flat64, xy, ut, vt):
            uvp = predict_one(flat64, unravel64, xy)
            return jnp.array([uvp[0] - ut, uvp[1] - vt])

        def R(flat64):
            ri = vmap(res_int, in_axes=(None, 0))(flat64, xi).reshape(-1)
            rb = vmap(res_bc, in_axes=(None, 0, 0, 0))(flat64, xb, ubc, vbc).reshape(-1)
            return jnp.concatenate([s_pde * ri, s_bc * rb])
        return R

    etas = jnp.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0], dtype=jnp.float64)

    def build_step(unravel64):
        R = make_R(unravel64)

        @jit
        def gn_step(flat64, lam):
            Rv, vjp_fn = jax.vjp(R, flat64)
            g = vjp_fn(Rv)[0]                       # J^T R = grad(0.5||R||^2)
            old = 0.5 * jnp.dot(Rv, Rv)

            def Hv(v):
                Jv = jax.jvp(R, (flat64,), (v,))[1]
                return vjp_fn(Jv)[0] + lam * v

            psi, _ = jax.scipy.sparse.linalg.cg(Hv, g, maxiter=GN_CG_IT, tol=1e-6)

            def loss_at(eta):
                fn = flat64 - eta * psi
                r = R(fn)
                return 0.5 * jnp.dot(r, r)
            ls = vmap(loss_at)(etas)
            j = jnp.argmin(ls)
            flat_new = flat64 - etas[j] * psi
            new = ls[j]
            bad = ~jnp.isfinite(new)
            flat_new = jnp.where(bad, flat64, flat_new)
            new = jnp.where(bad, old, new)
            return flat_new, new, old
        return gn_step

    return build_step


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    (flat_params, opt_state, update, unravel, model,
     pts_int, pts_bc, u_bc, v_bc, n_int, rng_local) = _make_train(rng)

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

    # ── Phase 2: float64 Gauss-Newton ──
    p_pytree = unravel(flat_params)
    p_pytree64 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), p_pytree)
    flat64, unravel64 = jax.flatten_util.ravel_pytree(p_pytree64)

    gn_key = random.PRNGKey(777)
    build_step = _make_gn(model, pts_int, pts_bc, u_bc, v_bc, n_int, gn_key)
    gn_step = build_step(unravel64)

    lam = GN_LAM0
    # warm-up compiles gn_step (counted against the wall)
    flat64, new, old = gn_step(flat64, lam)
    step_count += 1
    best_flat = flat64
    best_loss = float(new)
    while time.time() < deadline:
        flat64, new, old = gn_step(flat64, lam)
        step_count += 1
        new_f, old_f = float(new), float(old)
        # LM adaptation on relative improvement
        if new_f < old_f * (1.0 - 1e-4):
            lam = max(lam * 0.6, 1e-8)
        else:
            lam = min(lam * 3.0, 1e2)
        if new_f < best_loss:
            best_loss = new_f
            best_flat = flat64

    return unravel64(best_flat), step_count
