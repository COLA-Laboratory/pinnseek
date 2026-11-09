"""SPINN separable KS + IC warm-fit + OVERLAPPING-window residual (b08).
Change vs b07_polish: each window trains its causal residual on tau in
[0, 1+OVERLAP] (extra slices past its owned tau=1), so the hand-off point
tau=1 is interior/well-resolved rather than the causal-starved endpoint
(kb1 029/062) — cheap under SPINN. Ownership/predict unchanged."""
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
from jax import jit, random
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


K_X    = 6      # exact-periodic x embedding harmonics (same as champion)
HIDDEN = 128    # per-axis body width
DEPTH  = 4      # per-axis body depth
RANK   = 256    # separable rank r (u = sum_r f_r(tau) g_r(x))


class Body(nn.Module):
    """Per-axis MLP s -> R^RANK. If periodic, s enters through an exact-periodic
    harmonic embedding [cos ks, sin ks]_{k=1..K_X} so g_r(x) and all its
    x-derivatives are periodic by construction (no soft BC term)."""
    periodic: bool
    hidden: int = HIDDEN
    depth: int = DEPTH
    rank: int = RANK
    k_x: int = K_X

    @nn.compact
    def __call__(self, s):                 # s: [N] scalar coordinate
        s = s[..., None]                   # [N, 1]
        if self.periodic:
            ks = jnp.arange(1, self.k_x + 1, dtype=s.dtype)
            ang = s * ks
            h = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=-1)
        else:
            h = s
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.hidden)(h))
        return nn.Dense(self.rank)(h)       # [N, RANK]


_TBODY = Body(periodic=False)
_XBODY = Body(periodic=True)

N_WINDOWS = 8
N_T = 48        # causal time slices per window
N_X = 256       # FIXED spatial collocation per slice (regular grid, no resample)
N_X_IC = 256    # dense x grid for anchoring the next window's IC

LR_PEAK    = 2e-3
WARMUP     = 300
DECAY_STEP = 1500
DECAY_RATE = 0.9
LAM_IC     = 100.0
EPS_SCHED  = (1e-2, 1e-1, 1.0, 1e1, 1e2)
EPS_DELTA  = 0.99
SAVE_MARGIN_S = 10.0
EMA_DECAY  = 0.999
LR_IC      = 1e-3    # LR for the per-window pure-IC hand-off warm-fit
N_WARM     = 400     # pure-IC warm-fit steps at each hand-off window (k>=1)
OVERLAP    = 0.25    # train residual on tau in [0, 1+OVERLAP]; tau=1 interior
N_T_EXT    = round(N_T * (1.0 + OVERLAP))  # keep per-unit-tau slice density

# fixed regular collocation grid on the periodic ring [0, 2pi)
X_COL = jnp.linspace(X_MIN, X_MAX, N_X, endpoint=False)


def _ic_from_file():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    return (jnp.array(ic[:, 0].astype(np.float32)),
            jnp.array(ic[:, 1].astype(np.float32)))


def _residual_grid(p, tau_slices, x_col, dt_win):
    """KS residual on the tau x-grid via the separable ansatz. x-derivatives
    (up to 4th) from one Taylor-mode jet over the x-body; u_tau from one jet
    over the t-body. Returns r of shape [N_T, N_X]."""
    tp, xp = p["t"], p["x"]
    xf = lambda xs: _XBODY.apply(xp, xs)                       # [N_X, RANK]
    one, zero = jnp.ones_like(x_col), jnp.zeros_like(x_col)
    G0, (G1, G2, _G3, G4) = jet(xf, (x_col,), ((one, zero, zero, zero),))
    tf = lambda ts: _TBODY.apply(tp, ts)                       # [N_T, RANK]
    F0, (F1,) = jet(tf, (tau_slices,), ((jnp.ones_like(tau_slices),),))
    u      = F0 @ G0.T                                         # [N_T, N_X]
    u_t    = (F1 @ G0.T) / dt_win                              # local->global time
    u_x    = F0 @ G1.T
    u_xx   = F0 @ G2.T
    u_xxxx = F0 @ G4.T
    return u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx


def _u_at(p, tau_val, x_pts):
    """u(tau_val, x_pts) for a scalar tau (0 or 1): [N_pts]."""
    tvec = _TBODY.apply(p["t"], jnp.full((1,), tau_val))[0]    # [RANK]
    Xf = _XBODY.apply(p["x"], x_pts)                           # [N_pts, RANK]
    return Xf @ tvec


def _make_window_trainer(dt_win):
    # extend causal slices past the owned tau=1 so the hand-off point is
    # interior/well-resolved (overlap); causal cumsum still runs in tau order
    tau_slices = jnp.linspace(0.0, 1.0 + OVERLAP, N_T_EXT + 1)[1:]

    def loss_fn(p, x_ic, u_ic, eps):
        r = _residual_grid(p, tau_slices, X_COL, dt_win)
        L_r_t = jnp.mean(r ** 2, axis=1)
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(L_r_t)[:-1]])
        w = jax.lax.stop_gradient(jnp.exp(-eps * cum))
        pde_loss = jnp.mean(w * L_r_t)
        u_pred_ic = _u_at(p, 0.0, x_ic)
        ic_loss = jnp.mean((u_pred_ic - u_ic) ** 2)
        return LAM_IC * ic_loss + pde_loss, w

    loss_and_grad = jit(jax.value_and_grad(loss_fn, has_aux=True))
    sched = optax.warmup_exponential_decay_schedule(
        init_value=0.0, peak_value=LR_PEAK, warmup_steps=WARMUP,
        transition_steps=DECAY_STEP, decay_rate=DECAY_RATE)
    opt = optax.adam(learning_rate=sched)

    @jit
    def update(p, ema, opt_state, x_ic, u_ic, eps):
        (loss, w), grads = loss_and_grad(p, x_ic, u_ic, eps)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        ema = jax.tree_util.tree_map(
            lambda e, q: EMA_DECAY * e + (1.0 - EMA_DECAY) * q, ema, p)
        return p, ema, opt_state, loss, w

    # pure-IC warm-fit: pre-shift the window's tau=0 slice onto the inherited
    # hand-off target before the causal PDE loop, tightening the hand-off jump
    # (the self-consistency proxy that correlates rho=0.86 with rRMSE).
    def ic_loss_fn(p, x_ic, u_ic):
        return jnp.mean((_u_at(p, 0.0, x_ic) - u_ic) ** 2)

    ic_grad = jit(jax.value_and_grad(ic_loss_fn))
    ic_opt = optax.adam(learning_rate=LR_IC)

    @jit
    def ic_update(p, ic_state, x_ic, u_ic):
        loss, grads = ic_grad(p, x_ic, u_ic)
        updates, ic_state = ic_opt.update(grads, ic_state)
        p = optax.apply_updates(p, updates)
        return p, ic_state, loss

    return update, opt, ic_update, ic_opt


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
        Tf = _TBODY.apply(plist[k]["t"], tau)      # [N, RANK]
        Xf = _XBODY.apply(plist[k]["x"], x)        # [N, RANK]
        uk = jnp.sum(Tf * Xf, axis=1)
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

    dt_win = float((T_MAX - T_MIN) / N_WINDOWS)
    update, opt, ic_update, ic_opt = _make_window_trainer(dt_win)

    usable = TRAIN_TIME - SAVE_MARGIN_S
    for k in range(N_WINDOWS):
        rng, kt, kx = random.split(rng, 3)
        if prev_params is None:
            p = {"t": _TBODY.init(kt, jnp.zeros((1,))),
                 "x": _XBODY.init(kx, jnp.zeros((1,)))}
        else:
            p = prev_params                              # warm start from prev EMA
            # pure-IC warm-fit: lock this window's tau=0 onto the inherited
            # hand-off target before the causal PDE loop starts
            ic_state = ic_opt.init(p)
            for _w in range(N_WARM):
                p, ic_state, _icl = ic_update(p, ic_state, cur_x_ic, cur_u_ic)
                total_steps += 1
        opt_state = opt.init(p)
        ema = jax.tree_util.tree_map(lambda a: a, p)     # reset EMA per window

        eps_idx = 0
        eps = EPS_SCHED[eps_idx]
        win_deadline = _T0 + usable * (k + 1) / N_WINDOWS

        p, ema, opt_state, loss, w = update(p, ema, opt_state, cur_x_ic, cur_u_ic, eps)
        total_steps += 1
        it = 0
        while time.time() < win_deadline:
            it += 1
            p, ema, opt_state, loss, w = update(p, ema, opt_state, cur_x_ic, cur_u_ic, eps)
            total_steps += 1
            if it % 250 == 0:
                if float(w.min()) > EPS_DELTA and eps_idx < len(EPS_SCHED) - 1:
                    eps_idx += 1
                    eps = EPS_SCHED[eps_idx]

        plist.append(jax.tree_util.tree_map(lambda a: a, ema))
        prev_params = ema
        if k < N_WINDOWS - 1:
            u_end = _u_at(ema, 1.0, x_ic_dense)
            cur_x_ic = x_ic_dense
            cur_u_ic = jax.lax.stop_gradient(u_end)
        if eval_callback is not None:
            eval_callback(ema, total_steps, float(loss))

    out = {"bounds": np.asarray(bounds, dtype=np.float64), "params": plist}
    return out, total_steps
