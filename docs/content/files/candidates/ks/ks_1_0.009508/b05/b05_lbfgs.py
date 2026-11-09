"""Champion time-marching KS PINN (b04_rwf) + per-window Adam->L-BFGS polish.
Change vs parent (b04_rwf): each window reserves its final wall for an optax
L-BFGS finish on the uniform composite loss (kb1 074/089), started from the
Adam-EMA state and handed off — targets b04's stiff-mid-window residual wall."""
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
T_MIN, T_MAX = 0.0, 0.4
X_MIN, X_MAX = 0.0, 6.283185307179586   # 2*pi
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 4   # u_xxxx
N_RESIDUAL_COMPONENTS   = 1
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi")
PERIODIC_AXES  = ("x",)       # u(t, X_MIN) = u(t, X_MAX)
GEOMETRY_KIND  = "periodic_strip"

# --- Physical parameters (scaled KS: u_t + V1 u u_x + V2 u_xx + V3 u_xxxx = 0) ---
V1 = 100.0 / 16.0        # 6.25
V2 = 100.0 / 16.0**2     # 0.390625
V3 = 100.0 / 16.0**4     # ~1.5259e-3

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 300.0     # total wall-clock seconds for the WHOLE process,
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
from jax import grad, jit, random, vmap
from jax.experimental.jet import jet
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


K_X    = 6
HIDDEN = 128
DEPTH  = 4
RWF_MU    = 1.0     # random weight factorization scale mean (kb1 082)
RWF_SIGMA = 0.1     # scale std


class RWFDense(nn.Module):
    """Dense with Random Weight Factorization: W = exp(s) · V, per-neuron scale
    s ~ N(mu, sigma). Init reconstructs a glorot-normal W exactly."""
    features: int
    mu: float = RWF_MU
    sigma: float = RWF_SIGMA

    @nn.compact
    def __call__(self, x):
        din = x.shape[-1]
        s = self.param("s", lambda k, sh: self.mu + self.sigma * random.normal(k, sh),
                       (self.features,))
        g = jnp.exp(s)

        def v_init(k, sh):
            W0 = nn.initializers.glorot_normal()(k, sh)
            return W0 / g[None, :]
        V = self.param("V", v_init, (din, self.features))
        b = self.param("b", nn.initializers.zeros, (self.features,))
        return x @ (V * g[None, :]) + b


class PINN(nn.Module):
    """Input (tau_local, x): tau already in [0,1] within a window (no global
    re-normalisation); exact-periodic x embed + Wang-gated modified MLP,
    every Dense random-weight-factorized."""
    hidden: int = HIDDEN
    depth: int = DEPTH
    k_x: int = K_X

    @nn.compact
    def __call__(self, tx):
        tau = tx[..., 0:1]
        x = tx[..., 1:2]
        ks = jnp.arange(1, self.k_x + 1, dtype=tx.dtype)
        ang = x * ks
        feat = jnp.concatenate([tau, jnp.cos(ang), jnp.sin(ang)], axis=-1)
        U = nn.tanh(RWFDense(self.hidden)(feat))
        V = nn.tanh(RWFDense(self.hidden)(feat))
        h = nn.tanh(RWFDense(self.hidden)(feat))
        for _ in range(self.depth):
            z = nn.tanh(RWFDense(self.hidden)(h))
            h = z * U + (1.0 - z) * V
        return RWFDense(OUTPUT_DIM)(h)


_MODEL = PINN()

N_WINDOWS = 8
N_T = 24        # causal time slices per window
N_X = 128       # FIXED spatial collocation per slice (regular grid, no resample)
N_X_IC = 256    # dense x grid for anchoring the next window's IC

LR_PEAK    = 2e-3
WARMUP     = 300
DECAY_STEP = 1500
DECAY_RATE = 0.9
LAM_IC     = 100.0
EPS_SCHED  = (1e-2, 1e-1, 1.0, 1e1, 1e2)
EPS_DELTA  = 0.99
SAVE_MARGIN_S = 10.0
EMA_DECAY  = 0.999   # Polyak averaging ~last 1/(1-decay) ~ 1000 steps of a window
ADAM_FRAC  = 0.70    # fraction of each window's wall spent on Adam; rest on L-BFGS

# fixed regular collocation grid on the periodic ring [0, 2pi)
X_COL = jnp.linspace(X_MIN, X_MAX, N_X, endpoint=False)


def _ic_from_file():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    return (jnp.array(ic[:, 0].astype(np.float32)),
            jnp.array(ic[:, 1].astype(np.float32)))


def _make_window_trainer(dt_win):
    """Build the jitted causal Adam update AND an L-BFGS polish step for one
    window of width dt_win (global)."""
    def u_scalar(params, tau, x):
        return _MODEL.apply(params, jnp.stack([tau, x])[None, :])[0, 0]

    def residual_point(params, tau, x):
        gx = lambda xx: u_scalar(params, tau, xx)
        one, zero = jnp.ones_like(x), jnp.zeros_like(x)
        u, (u_x, u_xx, _u3, u_xxxx) = jet(gx, (x,), ((one, zero, zero, zero),))
        u_tau = grad(lambda tt: u_scalar(params, tt, x))(tau)
        u_t = u_tau / dt_win                       # chain rule: local->global time
        return u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx

    res_over_x = vmap(residual_point, in_axes=(None, None, 0))
    res_grid = vmap(res_over_x, in_axes=(None, 0, None))
    u_batch = vmap(u_scalar, in_axes=(None, 0, 0))
    tau_slices = jnp.linspace(0.0, 1.0, N_T + 1)[1:]

    def loss_fn(params, x_col, x_ic, u_ic, eps):
        r = res_grid(params, tau_slices, x_col)
        L_r_t = jnp.mean(r ** 2, axis=1)
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(L_r_t)[:-1]])
        w = jax.lax.stop_gradient(jnp.exp(-eps * cum))
        pde_loss = jnp.mean(w * L_r_t)
        u_pred_ic = u_batch(params, jnp.zeros_like(x_ic), x_ic)   # tau=0 boundary
        ic_loss = jnp.mean((u_pred_ic - u_ic) ** 2)
        return LAM_IC * ic_loss + pde_loss, w

    loss_and_grad = jit(jax.value_and_grad(loss_fn, has_aux=True))
    sched = optax.warmup_exponential_decay_schedule(
        init_value=0.0, peak_value=LR_PEAK, warmup_steps=WARMUP,
        transition_steps=DECAY_STEP, decay_rate=DECAY_RATE)
    opt = optax.adam(learning_rate=sched)

    @jit
    def update(params, ema, opt_state, x_ic, u_ic, eps):
        (loss, w), grads = loss_and_grad(params, X_COL, x_ic, u_ic, eps)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ema = jax.tree_util.tree_map(
            lambda e, p: EMA_DECAY * e + (1.0 - EMA_DECAY) * p, ema, params)
        return params, ema, opt_state, loss, w

    # ── L-BFGS polish: uniform (non-causal) composite loss on the fixed grid ──
    def loss_lbfgs(params, x_ic, u_ic):
        r = res_grid(params, tau_slices, X_COL)
        pde_loss = jnp.mean(r ** 2)                          # uniform over slices
        u_pred_ic = u_batch(params, jnp.zeros_like(x_ic), x_ic)
        ic_loss = jnp.mean((u_pred_ic - u_ic) ** 2)
        return LAM_IC * ic_loss + pde_loss

    lbfgs = optax.lbfgs()

    @jit
    def lbfgs_step(params, lstate, x_ic, u_ic):
        fun = lambda p: loss_lbfgs(p, x_ic, u_ic)
        val, g = jax.value_and_grad(fun)(params)
        updates, lstate = lbfgs.update(g, lstate, params, value=val, grad=g, value_fn=fun)
        params = optax.apply_updates(params, updates)
        return params, lstate, val

    return update, opt, u_batch, lbfgs, lbfgs_step


def predict_fn(params, X):
    bounds = np.asarray(params["bounds"])
    plist = params["params"]
    t = X[:, 0]
    x = X[:, 1]
    u_out = jnp.zeros((X.shape[0],))
    W = len(plist)
    for k in range(W):
        lo, hi = float(bounds[k]), float(bounds[k + 1])
        dt = hi - lo
        tau = (t - lo) / dt
        txl = jnp.stack([tau, x], axis=1)
        uk = _MODEL.apply(plist[k], txl)[:, 0]
        if k < W - 1:
            m = (t >= lo) & (t < hi)
        else:
            m = (t >= lo)
        u_out = jnp.where(m, uk, u_out)
    return {"u": u_out}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    bounds = np.linspace(T_MIN, T_MAX, N_WINDOWS + 1)
    x_ic_dense = jnp.linspace(X_MIN, X_MAX, N_X_IC, endpoint=False)

    x_ic0, u_ic0 = _ic_from_file()
    cur_x_ic, cur_u_ic = x_ic0, u_ic0

    plist = []
    total_steps = 0
    prev_params = None

    # uniform windows -> identical width -> build the jitted trainer ONCE
    dt_win = float((T_MAX - T_MIN) / N_WINDOWS)
    update, opt, u_batch, lbfgs, lbfgs_step = _make_window_trainer(dt_win)

    usable = TRAIN_TIME - SAVE_MARGIN_S
    for k in range(N_WINDOWS):
        lo, hi = bounds[k], bounds[k + 1]
        rng, ik = random.split(rng)
        if prev_params is None:
            params = _MODEL.init(ik, jnp.zeros([1, INPUT_DIM]))
        else:
            params = prev_params                       # warm start from prev handoff
        opt_state = opt.init(params)
        ema = jax.tree_util.tree_map(lambda a: a, params)   # reset EMA per window

        eps_idx = 0
        eps = EPS_SCHED[eps_idx]
        adam_deadline = _T0 + usable * (k + ADAM_FRAC) / N_WINDOWS
        win_deadline = _T0 + usable * (k + 1) / N_WINDOWS

        # ── Adam phase (causal, EMA) ──
        params, ema, opt_state, loss, w = update(params, ema, opt_state, cur_x_ic, cur_u_ic, eps)
        total_steps += 1
        it = 0
        while time.time() < adam_deadline:
            it += 1
            params, ema, opt_state, loss, w = update(params, ema, opt_state, cur_x_ic, cur_u_ic, eps)
            total_steps += 1
            if it % 250 == 0:
                if float(w.min()) > EPS_DELTA and eps_idx < len(EPS_SCHED) - 1:
                    eps_idx += 1
                    eps = EPS_SCHED[eps_idx]

        # ── L-BFGS polish from the Polyak-averaged (EMA) state ──
        pol = jax.tree_util.tree_map(lambda a: a, ema)
        lstate = lbfgs.init(pol)
        pol, lstate, lval = lbfgs_step(pol, lstate, cur_x_ic, cur_u_ic)  # warmup/compile
        total_steps += 1
        while time.time() < win_deadline:
            pol, lstate, lval = lbfgs_step(pol, lstate, cur_x_ic, cur_u_ic)
            total_steps += 1

        # hand off / warm-start / output the L-BFGS-polished state
        plist.append(jax.tree_util.tree_map(lambda a: a, pol))
        prev_params = pol
        if k < N_WINDOWS - 1:
            u_end = u_batch(pol, jnp.ones_like(x_ic_dense), x_ic_dense)
            cur_x_ic = x_ic_dense
            cur_u_ic = jax.lax.stop_gradient(u_end)
        if eval_callback is not None:
            eval_callback(pol, total_steps, float(lval))

    out = {"bounds": np.asarray(bounds, dtype=np.float64), "params": plist}
    return out, total_steps
