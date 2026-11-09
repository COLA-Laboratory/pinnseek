"""b08 anchor: exact re-run of b07 champion b07_m05c15 (RAD pool 16384, margin
b08 mean5: independent draw of the MEAN-ensemble champion (harvest).
reproduces the b07 seed-0 record (~0.019-0.021) as this block's baseline.
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

# --- Domain ---
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
HAS_TIME = False

# --- Equation structure ---
TIME_DERIV_ORDER        = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 3   # ru, rv, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("top", "bottom", "left", "right")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "box"

# --- Physical parameters ---
RE     = 3200.0      # NS-LDC Re=3200; harder regime than Re=1000
NU     = 1.0 / RE
U_LID  = 1.0

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 150.0     # total wall-clock seconds for the WHOLE process,
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
from jax import jacfwd, jit, random, vmap
import numpy as np
import optax
from flax import linen as nn
from soap_jax import soap


# ══════════ Hyperparameters ══════════

HIDDEN     = 128
DEPTH      = 4
NUM_FF     = 64
FF_SCALE   = 1.0
BS_INT     = 2048
NBC_WALL   = 192
LR0        = 2e-3
LR_END     = 1e-4
EMA_ALPHA  = 0.9
MAX_STEPS  = 16000       # nominal step budget the SCHEDULE is anchored to (b04)
LAMBDA_UPD = 250
LAM_MAX    = 30.0
SAVE_MARGIN_S = 6.0

NU_START   = 1.0 / 20.0
NU_TRUE    = NU
CURR_FRAC  = 0.4         # reach true Re at 40% wall

# pseudo-time (kb1 073/130)
TAU0       = 1.0
TAU_BETA   = 0.5         # EMA on tau
TAU_MIN    = 0.05
TAU_MAX    = 20.0
TAU_UPD    = 500         # steps between Barzilai-Borwein tau refreshes
K_INNER    = 40          # hold reference q_n fixed for K=40 steps (kb1 073)
PTS_BATCH  = 2048        # probe batch for the tau estimate
WALL_FRAC   = 0.50       # fraction of interior pts drawn near walls
WALL_MARGIN = 0.05       # push band tighter still

# Polyak/EMA parameter averaging over the late (converged, oscillating) phase
EMA_P_BETA  = 0.99       # EMA decay on params
EMA_P_START = 0.65       # begin averaging once progress exceeds this fraction

# Snapshot field ensembling: store late-phase snapshots, average their fields
SNAP_START  = 0.65       # begin collecting snapshots at this progress fraction
SNAP_EVERY  = 150        # steps between snapshots
SNAP_MAX    = 32         # keep at most this many (most recent)

# Cyclic-LR warm restarts in the tail (kb1 130 decorrelated trajectories /
# snapshot-ensemble idea). During the tail the field is converged; a moderate
# LR bump each cycle knocks params into a nearby-but-decorrelated basin, then a
# cosine anneal converges it; snapshot at the cycle minimum. Averaging the
# fields of these decorrelated snapshots cancels basin-shared error.
CYC_ON      = True
CYC_START   = 0.62       # begin cyclic phase at this progress (fixed-step anchored)
CYC_LEN     = 500        # steps per cosine restart cycle (more, shorter cycles)
CYC_PEAK    = 6e-4       # even gentler restart than cyc2 (8e-4)
CYC_LO      = 1.5e-4     # LR at cycle end (converged snapshot point)
CYC_USE_EMA = True       # keep EMA member too (more ensemble members)

# RAD residual-adaptive resampling (kb1 051)
RAD_UPD    = 200         # steps between interior-batch redraws
RAD_POOL   = 16384       # candidate pool size (finer thin-layer candidates)
RAD_K      = 1.0
RAD_C      = 0.15
RAD_START  = 0.25        # RAD only once the flow is established (else uniform)

_FF_KEY    = 20240721


# ══════════ Model: Fourier features + modified MLP (kb1 020, 050) ══════════

class ModifiedMLP(nn.Module):
    hidden: int = HIDDEN
    depth: int = DEPTH
    num_ff: int = NUM_FF
    ff_scale: float = FF_SCALE

    @nn.compact
    def __call__(self, x):
        B = random.normal(random.PRNGKey(_FF_KEY),
                          (INPUT_DIM, self.num_ff)) * self.ff_scale
        B = jax.lax.stop_gradient(B)
        proj = 2.0 * jnp.pi * (x @ B)
        h0 = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        kinit = jax.nn.initializers.glorot_normal()
        U = nn.tanh(nn.Dense(self.hidden, kernel_init=kinit, name="U")(h0))
        V = nn.tanh(nn.Dense(self.hidden, kernel_init=kinit, name="V")(h0))
        H = nn.tanh(nn.Dense(self.hidden, kernel_init=kinit, name="in")(h0))
        for i in range(self.depth - 1):
            Z = nn.tanh(nn.Dense(self.hidden, kernel_init=kinit, name=f"h{i}")(H))
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit, name="out")(H)


_MODEL = ModifiedMLP()


# ══════════ Training closures ══════════

def _tree_absflat(g):
    return jnp.concatenate([jnp.abs(x).ravel() for x in jax.tree_util.tree_leaves(g)])


def _make_train(rng_init):
    key, rng = random.split(rng_init)
    params = _MODEL.init(key, jnp.zeros([1, INPUT_DIM]))

    def predict_one(params, xy):
        return _MODEL.apply(params, xy[None, :])[0]      # [3]

    d1_fn = jacfwd(predict_one, argnums=1)
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)

    def raw_residual(params, xy, nu):
        uvp = predict_one(params, xy)
        u, v = uvp[0], uvp[1]
        d1 = d1_fn(params, xy)
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]
        d2 = d2_fn(params, xy)
        u_xx, u_yy = d2[0, 0, 0], d2[0, 1, 1]
        v_xx, v_yy = d2[1, 0, 0], d2[1, 1, 1]
        r_c  = u_x + v_y
        r_mx = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        r_my = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        return r_mx, r_my, r_c, u, v

    raw_batch = vmap(raw_residual, in_axes=(None, 0, None))

    def uv_batch(params, xy):
        uvp = vmap(predict_one, in_axes=(None, 0))(params, xy)
        return uvp[:, 0], uvp[:, 1]

    def bc_loss(params, xy_bc, u_bc, v_bc):
        uvp = vmap(predict_one, in_axes=(None, 0))(params, xy_bc)
        return jnp.mean((uvp[:, 0] - u_bc) ** 2 + (uvp[:, 1] - v_bc) ** 2)

    def _sample_pool(key):
        """Uniform + near-wall quadratic band candidate pool (kb1 086)."""
        k1, k2, k3, k4, k5 = random.split(key, 5)
        n_wall = int(RAD_POOL * WALL_FRAC)
        n_unif = RAD_POOL - n_wall
        xy_u = random.uniform(k1, (n_unif, INPUT_DIM), minval=0.0, maxval=1.0)
        tw = random.uniform(k2, (n_wall,))
        dn = WALL_MARGIN * random.uniform(k3, (n_wall,)) ** 2
        wall = random.randint(k4, (n_wall,), 0, 4)
        xw = jnp.where(wall == 2, dn, jnp.where(wall == 3, 1.0 - dn, tw))
        yw = jnp.where(wall == 0, dn, jnp.where(wall == 1, 1.0 - dn, tw))
        return jnp.concatenate([xy_u, jnp.stack([xw, yw], 1)], 0)

    def _sample_bc(key):
        t = random.uniform(key, (NBC_WALL,))
        one = jnp.ones_like(t); zero = jnp.zeros_like(t)
        top = jnp.stack([t, one], 1); bot = jnp.stack([t, zero], 1)
        lft = jnp.stack([zero, t], 1); rgt = jnp.stack([one, t], 1)
        xy_bc = jnp.concatenate([top, bot, lft, rgt], 0)
        u_bc = jnp.concatenate([one, zero, zero, zero], 0)
        v_bc = jnp.zeros(4 * NBC_WALL)
        return xy_bc, u_bc, v_bc

    @jit
    def rad_pick(key, params, nu, use_rad):
        """Draw the interior batch: uniform sub-sample of the pool, or (once
        use_rad) a RAD residual-weighted sub-sample p ~ eps^k/mean + c."""
        kp, ks = random.split(key)
        pool = _sample_pool(kp)
        r_mx, r_my, r_c, _, _ = raw_batch(params, pool, nu)
        eps = jnp.sqrt(r_mx ** 2 + r_my ** 2 + r_c ** 2)
        w_rad = (eps ** RAD_K) / (jnp.mean(eps ** RAD_K) + 1e-12) + RAD_C
        w = jnp.where(use_rad, w_rad, jnp.ones_like(eps))
        idx = random.choice(ks, RAD_POOL, (BS_INT,), replace=False, p=w / jnp.sum(w))
        return pool[idx]

    base_sched = optax.exponential_decay(LR0, transition_steps=6000,
                                         decay_rate=0.85, end_value=LR_END)
    _cyc_start_step = int(CYC_START * MAX_STEPS)

    def sched(step):
        base = base_sched(step)
        t = jnp.mod(step - _cyc_start_step, CYC_LEN)
        cos = 0.5 * (1.0 + jnp.cos(jnp.pi * (t / CYC_LEN)))
        cyclic = CYC_LO + (CYC_PEAK - CYC_LO) * cos
        use_cyc = jnp.logical_and(CYC_ON, step >= _cyc_start_step)
        return jnp.where(use_cyc, cyclic, base)

    optimizer = soap(learning_rate=sched, b1=0.95, b2=0.95,
                     precondition_frequency=10)
    opt_state = optimizer.init(params)

    @jit
    def update(params, params_prev, opt_state, xy_int, xy_bc, u_bc, v_bc,
               nu, lam_bc, tau, pts_on):
        u_prev, v_prev = uv_batch(params_prev, xy_int)
        u_prev = jax.lax.stop_gradient(u_prev); v_prev = jax.lax.stop_gradient(v_prev)

        def total(p):
            r_mx, r_my, r_c, u, v = raw_batch(p, xy_int, nu)
            fx = r_mx + pts_on * (u - u_prev) / tau
            fy = r_my + pts_on * (v - v_prev) / tau
            pde = jnp.mean(fx ** 2 + fy ** 2 + r_c ** 2)
            return pde + lam_bc * bc_loss(p, xy_bc, u_bc, v_bc)

        loss, grads = jax.value_and_grad(total)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jit
    def tau_probe(params, params_prev, key, nu):
        xy = random.uniform(key, (PTS_BATCH, INPUT_DIM), minval=0.0, maxval=1.0)
        rmx_n, rmy_n, _, u_n, v_n = raw_batch(params, xy, nu)
        rmx_p, rmy_p, _, u_p, v_p = raw_batch(params_prev, xy, nu)
        du = jnp.sqrt(jnp.mean((u_n - u_p) ** 2 + (v_n - v_p) ** 2))
        dr = jnp.sqrt(jnp.mean((rmx_n - rmx_p) ** 2 + (rmy_n - rmy_p) ** 2))
        return du / (dr + 1e-8)

    @jit
    def grad_ratio(params, xy_int, xy_bc, u_bc, v_bc, nu):
        def pde_only(p):
            r_mx, r_my, r_c, _, _ = raw_batch(p, xy_int, nu)
            return jnp.mean(r_mx ** 2 + r_my ** 2 + r_c ** 2)
        g_r = _tree_absflat(jax.grad(pde_only)(params))
        g_b = _tree_absflat(jax.grad(bc_loss)(params, xy_bc, u_bc, v_bc))
        return jnp.max(g_r) / (jnp.mean(g_b) + 1e-12)

    return params, opt_state, update, rad_pick, _sample_bc, tau_probe, grad_ratio, rng


# ══════════ Public contract ══════════

def predict_fn(params, X):
    # Ensemble structure: {"members": [pytree, ...]} -> average predicted fields.
    if isinstance(params, dict) and "members" in params:
        members = params["members"]
        acc = None
        for m in members:
            out = _MODEL.apply(m, X)
            acc = out if acc is None else acc + out
        out = acc / float(len(members))
        return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}
    out = _MODEL.apply(params, X)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def _nu_schedule(progress):
    f = min(max(progress / CURR_FRAC, 0.0), 1.0)
    log_nu = (1.0 - f) * np.log(NU_START) + f * np.log(NU_TRUE)
    return float(np.exp(log_nu))


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    (params, opt_state, update, rad_pick, sample_bc,
     tau_probe, grad_ratio, rng_local) = _make_train(rng)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    _cyc_start_step_t = int(CYC_START * MAX_STEPS)
    lam_bc = jnp.array(10.0)
    tau = jnp.array(TAU0)

    rng_local, kp, kb = random.split(rng_local, 3)
    xy_int = rad_pick(kp, params, jnp.array(NU_START), jnp.array(False))
    xy_bc, u_bc, v_bc = sample_bc(kb)

    prev = params
    params, opt_state, loss = update(params, prev, opt_state, xy_int, xy_bc, u_bc, v_bc,
                                     jnp.array(NU_START), lam_bc, tau, jnp.array(0.0))
    step_count = 1
    last_loss = float(loss)

    params_ema = params
    ema_on = False
    snaps = []                       # late-phase param snapshots (field ensemble)

    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        progress = min(it / MAX_STEPS, 1.0)
        nu = jnp.array(_nu_schedule(progress))
        pts_on = jnp.array(1.0 if progress > 0.25 else 0.0)
        if (it - 1) % K_INNER == 0:
            prev = params
        if it % RAD_UPD == 0:            # redraw interior (RAD) + BC
            rng_local, kp, kb = random.split(rng_local, 3)
            use_rad = jnp.array(progress > RAD_START)
            xy_int = rad_pick(kp, params, nu, use_rad)
            xy_bc, u_bc, v_bc = sample_bc(kb)
        params, opt_state, loss = update(params, prev, opt_state, xy_int, xy_bc, u_bc, v_bc,
                                         nu, lam_bc, tau, pts_on)
        step_count = it + 1
        last_loss = float(loss)

        if progress > EMA_P_START:
            if not ema_on:
                params_ema = params
                ema_on = True
            else:
                params_ema = jax.tree_util.tree_map(
                    lambda e, p: EMA_P_BETA * e + (1.0 - EMA_P_BETA) * p,
                    params_ema, params)

        if CYC_ON:
            # snapshot at each cycle minimum (most-converged point of the cycle)
            if it >= _cyc_start_step_t and (it - _cyc_start_step_t) % CYC_LEN == CYC_LEN - 1:
                snaps.append(params)
                if len(snaps) > SNAP_MAX:
                    snaps = snaps[-SNAP_MAX:]
        elif progress > SNAP_START and it % SNAP_EVERY == 0:
            snaps.append(params)
            if len(snaps) > SNAP_MAX:
                snaps = snaps[-SNAP_MAX:]

        if it % TAU_UPD == 0:
            rng_local, kt = random.split(rng_local)
            th = tau_probe(params, prev, kt, nu)
            if jnp.isfinite(th):
                tau = (1.0 - TAU_BETA) * tau + TAU_BETA * jnp.clip(th, TAU_MIN, TAU_MAX)
        if it % LAMBDA_UPD == 0:
            r = grad_ratio(params, xy_int, xy_bc, u_bc, v_bc, nu)
            if jnp.isfinite(r):
                lam_bc = (1.0 - EMA_ALPHA) * lam_bc + EMA_ALPHA * jnp.clip(r, 1.0, LAM_MAX)
        if eval_callback is not None and (it % 2000 == 0):
            eval_callback(params, it, last_loss)

    members = list(snaps)
    if ema_on and (CYC_USE_EMA or not CYC_ON):
        members.append(params_ema)          # EMA as one ensemble member
    if not members:
        members = [params]
    return {"members": members}, step_count
