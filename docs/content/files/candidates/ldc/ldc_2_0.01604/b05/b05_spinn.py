"""SPINN backbone (kb1 070) + Scale-PINN recipe: factorized per-axis MLPs
combined by rank-r CP product; an N x N collocation grid costs O(N) forward
passes via forward-mode AD, attacking the step-count floor all prior blocks hit.
Keeps the Scale-PINN correction + Re-curriculum + L-BFGS tail. Parent: b04_m32_int45b."""
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
from jax import jit, random
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


# ══════════ SPINN axis body: sin(F*pi*.) input + SiLU, output rank*3 ══════════

class AxisBody(nn.Module):
    """1-D MLP over a single coordinate. sin(freq*pi*.) input layer (frequency
    prior, kb1 126), SiLU hidden, linear output reshaped to [N, rank, 3] — the
    per-axis factor for u, v, p in the rank-r CP product (kb1 070)."""
    width: int = 64
    depth: int = 4
    rank: int = 128
    freq: float = 1.0

    @nn.compact
    def __call__(self, x):                    # x: [N, 1]
        kinit = jax.nn.initializers.he_uniform()
        h = jnp.sin(self.freq * jnp.pi * nn.Dense(self.width, kernel_init=kinit)(x))
        for _ in range(self.depth):
            h = nn.silu(nn.Dense(self.width, kernel_init=kinit)(h))
        h = nn.Dense(self.rank * OUTPUT_DIM, kernel_init=kinit)(h)
        return h.reshape(x.shape[0], self.rank, OUTPUT_DIM)   # [N, r, 3]


# ══════════ Hyperparameters ══════════

RANK        = 128
AX_WIDTH    = 64
AX_DEPTH    = 4
FREQ        = 1.0
NX = NY     = 128         # interior collocation grid per step (NX*NY residual pts)
N_BC        = 128         # boundary coords per wall-pair
TAU_SC      = 0.05        # sequential-correction pseudo-timestep (kb1 126)
TAU_ALPHA   = 8.0         # smoothing coeff = nu_eff/8 (b02 optimum)
RE_START    = 1000.0      # curriculum start Re (thicker boundary layers)
RE_WARM     = 40000       # steps to ramp Re_start -> RE=3200, then hold
ADAM_WALL   = 100.0       # seconds of Adam before switching to L-BFGS polish
POLISH_NX = POLISH_NY = 160   # fresh polish grid per axis
POLISH_MEM  = 32          # L-BFGS memory
LAM_BC      = 100.0       # BC weight (kb1 126 default; NTK imbalance kb1 026/050)
PEAK_LR     = 2e-3
WARMUP      = 2000
DECAY_STEPS = 120000
END_LR      = 1e-6
SAVE_MARGIN_S = 8.0       # stop early so device_get + pickle beat the wall


# ══════════ Forward-mode axis derivatives (kb1 070) ══════════

def _axis_d012(apply_axis, coords):
    """Return f0, f1, f2 = axis features and their 1st/2nd deriv wrt coords.
    coords: [N]; each output [N, r, 3]. One value call + one nested JVP."""
    fn = lambda z: apply_axis(z[:, None])
    f0 = fn(coords)
    f1fn = lambda z: jax.jvp(fn, (z,), (jnp.ones_like(z),))[1]
    f1, f2 = jax.jvp(f1fn, (coords,), (jnp.ones_like(coords),))
    return f0, f1, f2


def _grid(a, b):    # einsum CP combine: a[i,r,c], b[j,r,c] -> [i,j,c]
    return jnp.einsum("irc,jrc->ijc", a, b)


# ══════════ Training ══════════

def _make_train(rng_init):
    model = AxisBody(width=AX_WIDTH, depth=AX_DEPTH, rank=RANK, freq=FREQ)
    kx, ky, rng = random.split(rng_init, 3)
    px = model.init(kx, jnp.zeros([1, 1]))
    py = model.init(ky, jnp.zeros([1, 1]))
    params0 = {"x": px, "y": py}
    flat0, unravel = jax.flatten_util.ravel_pytree(params0)

    def _apply(pf):
        p = unravel(pf)
        ax = lambda c: model.apply(p["x"], c)   # [N,r,3]
        ay = lambda c: model.apply(p["y"], c)
        return ax, ay

    def _fields(pf, xc, yc):
        """All fields + 1st/2nd derivs on the tensor grid xc (x) X yc (y)."""
        ax, ay = _apply(pf)
        fx0, fx1, fx2 = _axis_d012(ax, xc)
        fy0, fy1, fy2 = _axis_d012(ay, yc)
        U   = _grid(fx0, fy0)     # [Nx,Ny,3]
        Ux  = _grid(fx1, fy0)
        Uy  = _grid(fx0, fy1)
        Uxx = _grid(fx2, fy0)
        Uyy = _grid(fx0, fy2)
        return U, Ux, Uy, Uxx, Uyy

    def _bc_edges(pf, xc, yc):
        """Predicted (u,v) on the four walls. xc: x-coords for top/bottom;
        yc: y-coords for left/right. Returns (uv_tb [n,2,2], uv_lr [2,n,2])."""
        ax, ay = _apply(pf)
        fx0 = ax(xc[:, None])                       # [n,r,3]
        fy0 = ay(yc[:, None])                        # [n,r,3]
        fy_tb = ay(jnp.array([[0.0], [1.0]]))        # [2,r,3] bottom,top
        fx_lr = ax(jnp.array([[0.0], [1.0]]))        # [2,r,3] left,right
        uv_tb = _grid(fx0, fy_tb)[..., :2]           # [n,2,3]->[n,2,2]
        uv_lr = _grid(fx_lr, fy0)[..., :2]           # [2,n,2]
        return uv_tb, uv_lr

    def _bc_loss(pf, xc, yc):
        uv_tb, uv_lr = _bc_edges(pf, xc, yc)
        # top/bottom: j=0 bottom (u=0,v=0), j=1 top (u=U_LID,v=0)
        u_tb_tgt = jnp.stack([jnp.zeros_like(xc), jnp.full_like(xc, U_LID)], axis=1)  # [n,2]
        loss_tb = jnp.mean((uv_tb[..., 0] - u_tb_tgt) ** 2 + uv_tb[..., 1] ** 2)
        # left/right: all u=v=0
        loss_lr = jnp.mean(uv_lr[..., 0] ** 2 + uv_lr[..., 1] ** 2)
        return 0.5 * (loss_tb + loss_lr)

    def loss_fn(pf, pf_prev, xc, yc, xbc, ybc, nu):
        U, Ux, Uy, Uxx, Uyy = _fields(pf, xc, yc)
        u, v, p = U[..., 0], U[..., 1], U[..., 2]
        ux, vx, px = Ux[..., 0], Ux[..., 1], Ux[..., 2]
        uy, vy, py = Uy[..., 0], Uy[..., 1], Uy[..., 2]
        lap_u = Uxx[..., 0] + Uyy[..., 0]
        lap_v = Uxx[..., 1] + Uyy[..., 1]

        # previous iterate (detached: pf_prev independent of pf)
        Up, _, _, Uxxp, Uyyp = _fields(pf_prev, xc, yc)
        u_p, v_p, p_p = Up[..., 0], Up[..., 1], Up[..., 2]
        lap_u_p = Uxxp[..., 0] + Uyyp[..., 0]
        lap_v_p = Uxxp[..., 1] + Uyyp[..., 1]

        r_c  = ux + vy
        r_mx = u * ux + v * uy + px - nu * lap_u
        r_my = u * vx + v * vy + py - nu * lap_v

        corr_u = (u - u_p) / TAU_SC - (nu / TAU_ALPHA) * (lap_u - lap_u_p)
        corr_v = (v - v_p) / TAU_SC - (nu / TAU_ALPHA) * (lap_v - lap_v_p)
        corr_c = (p - p_p) / TAU_SC

        res = (r_c + corr_c) ** 2 + (r_mx + corr_u) ** 2 + (r_my + corr_v) ** 2
        pde_loss = jnp.mean(res)
        bc_loss = _bc_loss(pf, xbc, ybc)
        return pde_loss + LAM_BC * bc_loss

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=PEAK_LR, warmup_steps=WARMUP,
        decay_steps=DECAY_STEPS, end_value=END_LR)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(flat0)
    loss_and_grad = jit(jax.value_and_grad(loss_fn))

    def _nu_eff(step):
        frac = jnp.minimum(step / RE_WARM, 1.0)
        re_eff = RE_START + (RE - RE_START) * frac
        return 1.0 / re_eff

    @jit
    def update(pf, pf_prev, opt_state, key, step):
        k1, k2, k3, k4 = random.split(key, 4)
        xc = random.uniform(k1, (NX,))
        yc = random.uniform(k2, (NY,))
        xbc = random.uniform(k3, (N_BC,))
        ybc = random.uniform(k4, (N_BC,))
        nu = _nu_eff(step)
        loss, grads = loss_and_grad(pf, pf_prev, xc, yc, xbc, ybc, nu)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(pf, updates), opt_state, loss

    # ---- L-BFGS tail on the TRUE steady-NS residual (no correction) ----
    def _true_res(pf, xc, yc):
        U, Ux, Uy, Uxx, Uyy = _fields(pf, xc, yc)
        u, v = U[..., 0], U[..., 1]
        ux, vx, px = Ux[..., 0], Ux[..., 1], Ux[..., 2]
        uy, vy, py = Uy[..., 0], Uy[..., 1], Uy[..., 2]
        lap_u = Uxx[..., 0] + Uyy[..., 0]
        lap_v = Uxx[..., 1] + Uyy[..., 1]
        r_c  = ux + vy
        r_mx = u * ux + v * uy + px - NU * lap_u
        r_my = u * vx + v * vy + py - NU * lap_v
        return jnp.mean(r_c ** 2 + r_mx ** 2 + r_my ** 2)

    def make_polish(pf_start, key):
        k1, k2, k3, k4 = random.split(key, 4)
        xc = random.uniform(k1, (POLISH_NX,))
        yc = random.uniform(k2, (POLISH_NY,))
        xbc = random.uniform(k3, (POLISH_NX,))
        ybc = random.uniform(k4, (POLISH_NY,))

        def polish_loss(pf):
            return _true_res(pf, xc, yc) + LAM_BC * _bc_loss(pf, xbc, ybc)

        opt = optax.lbfgs(memory_size=POLISH_MEM)
        vg = optax.value_and_grad_from_state(polish_loss)

        @jit
        def polish_step(pf, state):
            loss, grad = vg(pf, state=state)
            updates, state = opt.update(grad, state, pf, value=loss,
                                        grad=grad, value_fn=polish_loss)
            return optax.apply_updates(pf, updates), state, loss

        return polish_step, opt.init(pf_start)

    return flat0, opt_state, update, unravel, rng, make_polish


# ══════════ Public contract ══════════

_PREDICT_MODEL = AxisBody(width=AX_WIDTH, depth=AX_DEPTH, rank=RANK, freq=FREQ)


def predict_fn(params, X):
    """X: [BS, 2] with columns (x, y). Scattered points -> diagonal of the CP
    product: u(x_i,y_i)=sum_r fx[i,r,c]*fy[i,r,c]. Returns {"u","v","p"}."""
    fx = _PREDICT_MODEL.apply(params["x"], X[:, 0:1])   # [BS,r,3]
    fy = _PREDICT_MODEL.apply(params["y"], X[:, 1:2])   # [BS,r,3]
    out = jnp.einsum("irc,irc->ic", fx, fy)             # [BS,3]
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """SPINN + Scale-PINN Adam curriculum then an L-BFGS true-NS tail. Whole
    process finishes within TRAIN_TIME s of _T0."""
    pf, opt_state, update, unravel, rng, make_polish = _make_train(rng)
    pf_prev = pf

    # warmup step (JIT-compiles the training step; counted)
    key, rng = random.split(rng)
    pf_new, opt_state, loss = update(pf, pf_prev, opt_state, key, jnp.asarray(0.0))
    pf_prev, pf = pf, pf_new
    step_count = 1

    # ---- Phase 1: Adam curriculum ----
    adam_deadline = _T0 + min(ADAM_WALL, TRAIN_TIME - SAVE_MARGIN_S - 4.0)
    for it in range(1, MAX_ITER):
        if time.time() > adam_deadline:
            break
        key, rng = random.split(rng)
        pf_new, opt_state, loss = update(pf, pf_prev, opt_state, key, jnp.asarray(float(it)))
        pf_prev, pf = pf, pf_new
        step_count = it + 1
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(pf), it, float(loss))

    # ---- Phase 2: L-BFGS tail on the true steady-NS residual ----
    key, rng = random.split(rng)
    polish_step, pstate = make_polish(pf, key)
    polish_deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    best_pf, best_loss = pf, float("inf")
    while time.time() < polish_deadline:
        pf, pstate, loss = polish_step(pf, pstate)
        lval = float(loss)
        step_count += 1
        if lval < best_loss:
            best_loss, best_pf = lval, pf

    return unravel(best_pf), step_count
