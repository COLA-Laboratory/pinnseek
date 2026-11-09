"""b06_v6_streamfn: fork of b05_v3_ema99 (SC-Adam, EMA.99) with a stream-
function ansatz — net outputs (psi, p); u=psi_y, v=-psi_x makes continuity
exact by construction (no continuity residual). Momentum uses 3rd-order
autodiff of psi. Tests whether exact continuity beats the penalty form."""
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


# ══════════ Vanilla PINN: 5 × 100 tanh, output (u, v, p) ══════════

class PINN(nn.Module):
    """Scale-PINN backbone (kb1:126): first layer sin(FREQ·π·Dense(xy)),
    layers 2..L SiLU, He init, linear output of 3 channels (u, v, p)."""
    hidden_width: int = 100
    n_hidden_layers: int = 5
    freq: float = 1.0

    @nn.compact
    def __call__(self, xy):
        # xy: [N, 2]
        kinit = jax.nn.initializers.he_uniform()
        h = jnp.sin(self.freq * jnp.pi * nn.Dense(self.hidden_width, kernel_init=kinit)(xy))
        for _ in range(self.n_hidden_layers - 1):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.silu(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)  # → [N, 3]: (u, v, p)


# ══════════ Geometry: fixed analytic NACA0012 surface ══════════

def _naca_airfoil_points(n: int = 400, thickness: float = 0.12) -> np.ndarray:
    """NACA0012 surface, cosine-clustered, rotated -AOA_DEG about (0.5, 0).
    Returns [n, 2] no-slip boundary points (u=v=0)."""
    beta = np.linspace(0.0, np.pi, n // 2)
    xc = 0.5 * (1.0 - np.cos(beta))                       # cosine spacing, clusters at LE/TE
    yt = 5 * thickness * (0.2969 * np.sqrt(xc) - 0.1260 * xc
                          - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)
    xa = np.concatenate([xc, xc])
    ya = np.concatenate([yt, -yt])                        # upper + lower surfaces
    a = np.radians(-AOA_DEG)
    xr = 0.5 + (xa - 0.5) * np.cos(a) - ya * np.sin(a)
    yr = 0.0 + (xa - 0.5) * np.sin(a) + ya * np.cos(a)
    return np.stack([xr, yr], axis=1).astype(np.float32)


# ══════════ Data: fixed fluid grid + airfoil/farfield BCs ══════════

def _build_dataset():
    """Read the reference grid; return the fixed point sets.

    * ``pts_int``: fluid collocation points (``phi <= 0``, outer edges
      excluded) — the fixed PDE-residual set (Scale-PINN convention).
    * ``pts_bc`` / ``u_bc`` / ``v_bc``: airfoil (no-slip 0,0) + farfield
      (freestream U_INF,0) boundary points and their target velocities.
    """
    ref = np.loadtxt(_locate("collocation.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    xy, phi = ref[:, 0:2], ref[:, 2]

    edge = (
        (xy[:, 0] == X_MIN) | (xy[:, 0] == X_MAX)
        | (xy[:, 1] == Y_MIN) | (xy[:, 1] == Y_MAX)
    )
    fluid = (phi <= 0.0)

    pts_int = xy[fluid & ~edge].astype(np.float32)        # fixed collocation
    pts_far = xy[edge].astype(np.float32)                 # farfield → freestream
    pts_air = _naca_airfoil_points(400)                   # airfoil → no-slip

    pts_bc = np.vstack([pts_far, pts_air])
    u_bc = np.concatenate([np.full(len(pts_far), U_INF, np.float32),
                           np.zeros(len(pts_air), np.float32)])
    v_bc = np.zeros(len(pts_bc), np.float32)

    return (jnp.array(pts_int), jnp.array(pts_bc), jnp.array(u_bc), jnp.array(v_bc))


# ══════════ Training hyperparameters ══════════

LR        = 1e-3      # peak Adam LR (warmup-cosine)
WEIGHT_BC = 10.0      # BC weight (Scale-PINN lam_bc default)
BS_INT    = 1024      # interior collocation points per minibatch (Adam)
BS_BC     = 256       # boundary points per minibatch (Adam)
INV_TAU_SC = 10.0     # 1/tau_sc: sequential-correction relaxation (kb1:126)
LAP_COEF   = 0.0      # drop the SC Laplacian-smoothing term (ablation)
WARMUP_STEPS = 2000   # LR warmup steps
COS_STEPS    = 120000 # cosine-decay horizon
EMA_DECAY    = 0.99   # in-graph weight EMA (~last few hundred steps), returned at end
SAVE_MARGIN_S = 5.0   # stop the loop this early so device_get + pickling
                      # finish before eval.py's TRAIN_TIME wall


# ══════════ Loss: PDE residual via autograd, BC L2 ══════════

def _make_train(rng_init):
    pts_int, pts_bc, u_bc, v_bc = _build_dataset()
    n_int, n_bc = len(pts_int), len(pts_bc)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def psi_scalar(params_flat, xy):
        return model.apply(unravel(params_flat), xy[None, :])[0, 0]   # stream fn psi
    def p_scalar(params_flat, xy):
        return model.apply(unravel(params_flat), xy[None, :])[0, 2]   # pressure p

    gpsi   = jacfwd(psi_scalar, argnums=1)                            # [2]  (psi_x, psi_y)
    hpsi   = jacfwd(gpsi, argnums=1)                                  # [2,2]
    t3psi  = jacfwd(hpsi, argnums=1)                                  # [2,2,2]
    gp     = jacfwd(p_scalar, argnums=1)                              # [2]  (p_x, p_y)

    def uv_from_psi(params_flat, xy):
        g = gpsi(params_flat, xy)
        return g[1], -g[0]                                            # u=psi_y, v=-psi_x

    def fields_and_derivs(params_flat, xy):
        g = gpsi(params_flat, xy); H = hpsi(params_flat, xy); T = t3psi(params_flat, xy)
        gpr = gp(params_flat, xy)
        p = p_scalar(params_flat, xy)
        u, v = g[1], -g[0]
        # velocity gradients from psi Hessian (continuity u_x+v_y = H[1,0]-H[0,1] = 0 exactly)
        u_x, u_y = H[1, 0], H[1, 1]
        v_x, v_y = -H[0, 0], -H[0, 1]
        p_x, p_y = gpr[0], gpr[1]
        lap_u = T[1, 0, 0] + T[1, 1, 1]           # psi_yxx + psi_yyy
        lap_v = -(T[0, 0, 0] + T[0, 1, 1])        # -(psi_xxx + psi_xyy)
        return u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, lap_u, lap_v

    def sc_residual_per_point(params_flat, params_prev, xy):
        """Sequential-correction residual (kb1:126) in stream-function form:
        continuity is exact (dropped), so only the two momentum residuals
        remain, each carrying the SC velocity extrapolation (1/tau_sc)(f-f_prev)."""
        u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, lap_u, lap_v = fields_and_derivs(params_flat, xy)
        up, vp = uv_from_psi(params_prev, xy)
        r_mx = u * u_x + v * u_y + p_x - NU * lap_u
        r_my = u * v_x + v * v_y + p_y - NU * lap_v
        r_mx = r_mx + INV_TAU_SC * (u - up)
        r_my = r_my + INV_TAU_SC * (v - vp)
        return r_mx ** 2 + r_my ** 2

    sc_residual_batch = vmap(sc_residual_per_point, in_axes=(None, None, 0))

    def loss_fn(params_flat, params_prev, xy_int, xy_bc_batch, u_bc_batch, v_bc_batch):
        pde_loss = jnp.mean(sc_residual_batch(params_flat, params_prev, xy_int))
        uv = vmap(uv_from_psi, in_axes=(None, 0))(params_flat, xy_bc_batch)
        u_pred, v_pred = uv[0], uv[1]
        bc_loss = jnp.mean((u_pred - u_bc_batch) ** 2 + (v_pred - v_bc_batch) ** 2)
        return pde_loss + WEIGHT_BC * bc_loss

    # differentiate only w.r.t. current params; params_prev is a stop-grad constant
    loss_and_grad = jit(jax.value_and_grad(loss_fn, argnums=0))

    sched = optax.warmup_cosine_decay_schedule(
        init_value=1e-6, peak_value=LR, warmup_steps=WARMUP_STEPS,
        decay_steps=COS_STEPS, end_value=1e-6)
    optimizer = optax.adam(learning_rate=sched)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, params_prev, opt_state, ema, key_pair):
        key1, key2 = key_pair
        idx_int = random.choice(key1, n_int, (BS_INT,))
        idx_bc  = random.choice(key2, n_bc,  (BS_BC,))
        loss, grads = loss_and_grad(params_flat, params_prev, pts_int[idx_int],
                                    pts_bc[idx_bc], u_bc[idx_bc], v_bc[idx_bc])
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params_flat, updates)
        new_ema = EMA_DECAY * ema + (1.0 - EMA_DECAY) * new_params
        return new_params, opt_state, new_ema, loss

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: [BS, 2] with columns (x, y). Returns {"u","v","p"} each [BS].
    Stream-function ansatz: u=psi_y, v=-psi_x from the net's psi channel."""
    def psi_of(xy):
        return _PREDICT_MODEL.apply(params, xy[None, :])[0, 0]
    g = vmap(jacfwd(psi_of))(X)             # [BS, 2] = (psi_x, psi_y)
    out = _PREDICT_MODEL.apply(params, X)   # [BS, 3]
    return {"u": g[:, 1], "v": -g[:, 0], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Pure-Adam Scale-PINN training with a per-step previous-iterate snapshot
    driving the sequential correction; whole process finishes within TRAIN_TIME
    s of _T0. Returns (params_pytree, step_count)."""
    (params_flat, opt_state, update, unravel, rng_local) = _make_train(rng)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S

    # warmup step: prev = current (SC term is zero the first step); compiles update
    params_prev = params_flat
    ema = params_flat
    key1, key2, rng_local = random.split(rng_local, 3)
    new_params, opt_state, ema, loss = update(params_flat, params_prev, opt_state, ema, (key1, key2))
    params_prev, params_flat = params_flat, new_params
    ema = params_flat  # seed EMA at the first post-warmup iterate
    step_count = 1

    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        key1, key2, rng_local = random.split(rng_local, 3)
        new_params, opt_state, ema, loss = update(params_flat, params_prev, opt_state, ema, (key1, key2))
        params_prev, params_flat = params_flat, new_params
        step_count = it + 1

    return unravel(ema), step_count
