"""SPINN separable KS + EXACT hard-IC on EVERY window incl. window 0 (fork of b09_hardic).
Window 0 also uses u = g0(x) + [F(tau)-F(0)].G(x), where g0 is a frozen periodic
net pre-fit to the data IC; its x-derivatives come from jet autodiff of that net.
So the whole chain is hard-IC (no IC loss anywhere) and every residual derivative
is autodiff of a network. Attacks window-0's sole remaining seed-variance source.
"""
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

# --- frozen IC net g0(x): fits data IC, provides exact hard-IC for window 0 ---
K_X_IC   = 16   # richer harmonics: g0 is a fixed offset (band-limited IC, modes<=~12)
HID_IC   = 64
DEPTH_IC = 3
N_IC_FIT = 3000 # Adam regression steps to fit g0 to the data IC (deterministic)


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
# scalar periodic net for the frozen IC field g0(x) (rank 1, richer harmonics)
_GBODY = Body(periodic=True, hidden=HID_IC, depth=DEPTH_IC, rank=1, k_x=K_X_IC)

N_WINDOWS = 8
N_T = 48        # causal time slices per window
N_X = 256       # FIXED spatial collocation per slice (regular grid, no resample)

LR_PEAK    = 2e-3
WARMUP     = 300
DECAY_STEP = 1500
DECAY_RATE = 0.9
EPS_SCHED  = (1e-2, 1e-1, 1.0, 1e1, 1e2)
EPS_DELTA  = 0.99
SAVE_MARGIN_S = 10.0
EMA_DECAY  = 0.999

# fixed regular collocation grid on the periodic ring [0, 2pi)
X_COL = jnp.linspace(X_MIN, X_MAX, N_X, endpoint=False)


def _ic_from_file():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    return (jnp.array(ic[:, 0].astype(np.float32)),
            jnp.array(ic[:, 1].astype(np.float32)))


def _fit_g0(x_ic, u_ic):
    """Fit the frozen scalar periodic net g0(x) to the data IC by Adam regression.
    Deterministic (fixed key) so g0 is not a seed-variance source. Returns g0 params."""
    gp = _GBODY.init(random.PRNGKey(12345), jnp.zeros((1,)))
    sched = optax.warmup_cosine_decay_schedule(
        init_value=1e-4, peak_value=5e-3, warmup_steps=200,
        decay_steps=N_IC_FIT, end_value=1e-4)
    opt = optax.adam(sched)
    st = opt.init(gp)

    def loss(gp):
        pred = _GBODY.apply(gp, x_ic)[:, 0]
        return jnp.mean((pred - u_ic) ** 2)

    lg = jit(jax.value_and_grad(loss))

    @jit
    def step(gp, st):
        l, g = lg(gp)
        upd, st = opt.update(g, st)
        return optax.apply_updates(gp, upd), st, l

    for _ in range(N_IC_FIT):
        gp, st, _l = step(gp, st)
    return gp


def _g0_derivs(gp, x_col):
    """g0 and its x/xx/xxxx derivatives on x_col, by jet autodiff of the frozen net."""
    gf = lambda xs: _GBODY.apply(gp, xs)[:, 0]      # scalar per point
    one, zero = jnp.ones_like(x_col), jnp.zeros_like(x_col)
    G0, (G1, G2, _G3, G4) = jet(gf, (x_col,), ((one, zero, zero, zero),))
    return (G0, G1, G2, G4)


# ── every window k: EXACT hard-IC. u = g(x) + [F(tau)-F(0)].G(x). g precomputed ──
def _residual_grid_hard(p, tau_slices, x_col, dt_win, gcol):
    """gcol = (g0,g1,g2,g4): the inherited field u_{k-1}(tau=1,.) and its x-,
    xx-, xxxx-derivatives, precomputed & constant on x_col. u_k(0,.) = g0 exact.
    For window 0, gcol is the frozen IC net's derivatives."""
    tp, xp = p["t"], p["x"]
    xf = lambda xs: _XBODY.apply(xp, xs)
    one, zero = jnp.ones_like(x_col), jnp.zeros_like(x_col)
    G0, (G1, G2, _G3, G4) = jet(xf, (x_col,), ((one, zero, zero, zero),))
    tf = lambda ts: _TBODY.apply(tp, ts)
    F0, (F1,) = jet(tf, (tau_slices,), ((jnp.ones_like(tau_slices),),))
    F_at0 = _TBODY.apply(tp, jnp.zeros((1,)))[0]     # F(tau=0) [RANK]
    Fd = F0 - F_at0[None, :]                          # increment vanishes at tau=0
    g0, g1, g2, g4 = gcol
    u      = g0[None, :] + Fd @ G0.T
    u_t    = (F1 @ G0.T) / dt_win                     # d/dtau[F(tau)-F(0)] = F'(tau)
    u_x    = g1[None, :] + Fd @ G1.T
    u_xx   = g2[None, :] + Fd @ G2.T
    u_xxxx = g4[None, :] + Fd @ G4.T
    return u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx


def _end_derivs(p, x_col):
    """Window output-increment at tau=1 and its x/xx/xxxx derivs on x_col.
    Increment = [F(1)-F(0)].G(x) (hard form, same for every window)."""
    F1 = _TBODY.apply(p["t"], jnp.ones((1,)))[0]
    F_at0 = _TBODY.apply(p["t"], jnp.zeros((1,)))[0]
    coef = F1 - F_at0
    xf = lambda xs: _XBODY.apply(p["x"], xs)
    one, zero = jnp.ones_like(x_col), jnp.zeros_like(x_col)
    G0, (G1, G2, _G3, G4) = jet(xf, (x_col,), ((one, zero, zero, zero),))
    return (G0 @ coef, G1 @ coef, G2 @ coef, G4 @ coef)


def _make_trainer(dt_win):
    tau_slices = jnp.linspace(0.0, 1.0, N_T + 1)[1:]

    # every window: pure causal residual (IC is exact by construction)
    def lossH(p, gcol, eps):
        r = _residual_grid_hard(p, tau_slices, X_COL, dt_win, gcol)
        L_r_t = jnp.mean(r ** 2, axis=1)
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(L_r_t)[:-1]])
        w = jax.lax.stop_gradient(jnp.exp(-eps * cum))
        return jnp.mean(w * L_r_t), w

    lgH = jit(jax.value_and_grad(lossH, has_aux=True))

    sched = optax.warmup_exponential_decay_schedule(
        init_value=0.0, peak_value=LR_PEAK, warmup_steps=WARMUP,
        transition_steps=DECAY_STEP, decay_rate=DECAY_RATE)
    opt = optax.adam(learning_rate=sched)

    @jit
    def updateH(p, ema, opt_state, gcol, eps):
        (loss, w), grads = lgH(p, gcol, eps)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        ema = jax.tree_util.tree_map(
            lambda e, q: EMA_DECAY * e + (1.0 - EMA_DECAY) * q, ema, p)
        return p, ema, opt_state, loss, w

    return updateH, opt


def predict_fn(params, X):
    bounds = np.asarray(params["bounds"])
    plist = params["params"]
    gp = params["g0"]
    t = X[:, 0]
    x = X[:, 1]
    u_out = jnp.zeros((X.shape[0],))
    W = len(plist)
    base = _GBODY.apply(gp, x)[:, 0]    # g0(x): the frozen IC field, window-0 offset
    ones1 = jnp.ones((1,))
    zero1 = jnp.zeros((1,))
    for k in range(W):
        lo, hi = float(bounds[k]), float(bounds[k + 1])
        dt = hi - lo
        tau = (t - lo) / dt
        pk = plist[k]
        Xf = _XBODY.apply(pk["x"], x)                 # [N, RANK]
        Tf = _TBODY.apply(pk["t"], tau)               # [N, RANK]
        F0v = _TBODY.apply(pk["t"], zero1)[0]             # F(0)
        uk = base + jnp.sum((Tf - F0v[None, :]) * Xf, axis=1)
        coef_end = _TBODY.apply(pk["t"], ones1)[0] - F0v  # F(1)-F(0)
        if k < W - 1:
            m = (t >= lo) & (t < hi)
        else:
            m = (t >= lo)
        u_out = jnp.where(m, uk, u_out)
        base = base + jnp.sum(coef_end[None, :] * Xf, axis=1)  # -> g_{k+1}(x)
    return {"u": u_out}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    bounds = np.linspace(T_MIN, T_MAX, N_WINDOWS + 1)
    x_ic0, u_ic0 = _ic_from_file()

    dt_win = float((T_MAX - T_MIN) / N_WINDOWS)
    updateH, opt = _make_trainer(dt_win)

    # frozen IC field g0(x): fit once, deterministically; provides window-0 hard IC
    gp = _fit_g0(x_ic0, u_ic0)
    gcol = tuple(jax.lax.stop_gradient(a) for a in _g0_derivs(gp, X_COL))

    plist = []
    total_steps = 0
    prev_params = None

    usable = TRAIN_TIME - SAVE_MARGIN_S
    for k in range(N_WINDOWS):
        rng, kt, kx = random.split(rng, 3)
        if prev_params is None:
            p = {"t": _TBODY.init(kt, jnp.zeros((1,))),
                 "x": _XBODY.init(kx, jnp.zeros((1,)))}
        else:
            p = prev_params                              # warm start from prev EMA
        opt_state = opt.init(p)
        ema = jax.tree_util.tree_map(lambda a: a, p)

        eps_idx = 0
        eps = EPS_SCHED[eps_idx]
        win_deadline = _T0 + usable * (k + 1) / N_WINDOWS

        p, ema, opt_state, loss, w = updateH(p, ema, opt_state, gcol, eps)
        total_steps += 1
        it = 0
        while time.time() < win_deadline:
            it += 1
            p, ema, opt_state, loss, w = updateH(p, ema, opt_state, gcol, eps)
            total_steps += 1
            if it % 250 == 0:
                if float(w.min()) > EPS_DELTA and eps_idx < len(EPS_SCHED) - 1:
                    eps_idx += 1
                    eps = EPS_SCHED[eps_idx]

        plist.append(jax.tree_util.tree_map(lambda a: a, ema))
        prev_params = ema
        # telescope the exact handoff: g_{k+1} = g_k + increment_k(tau=1, .)
        if k < N_WINDOWS - 1:
            incr = _end_derivs(ema, X_COL)
            gcol = tuple(jax.lax.stop_gradient(a + b) for a, b in zip(gcol, incr))
        if eval_callback is not None:
            eval_callback(ema, total_steps, float(loss))

    out = {"bounds": np.asarray(bounds, dtype=np.float64), "params": plist, "g0": gp}
    return out, total_steps
