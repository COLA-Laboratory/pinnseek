"""Hard-IC SPINN time-march (jet) + per-window LBFGS polish — parent b06_hardic.
Change vs parent: each window trains with Adam for the first ADAM_FRAC of its
wall share, then switches to full-batch LBFGS (zoom linesearch) for the rest.
SPINN's loss is the whole Nt*Nx grid (no minibatch noise), so 2nd-order LBFGS
is well-matched here (b05 rec b; kb1 063/074). Tests the well-conditioned regime.
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
from jax import jit, random, vmap
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


# ══════════ Separable sub-networks ══════════

RANK    = 128
WIDTH   = 64
DEPTH   = 3
M_MODES = 16
N_WIN   = 10
NT_WIN  = 48
NX      = 512
IC_KFIT = 16     # Fourier modes used to fit the (bandlimited) initial condition


class TimeNet(nn.Module):
    rank: int = RANK
    width: int = WIDTH
    depth: int = DEPTH

    @nn.compact
    def __call__(self, s):           # s: [N,1] in [0,1]
        kinit = jax.nn.initializers.glorot_normal()
        h = s
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h))
        return nn.Dense(self.rank, kernel_init=kinit)(h)


class SpaceNet(nn.Module):
    rank: int = RANK
    width: int = WIDTH
    depth: int = DEPTH
    modes: int = M_MODES

    @nn.compact
    def __call__(self, x):           # x: [N,1]
        kinit = jax.nn.initializers.glorot_normal()
        ks = jnp.arange(1, self.modes + 1, dtype=x.dtype)
        xk = x * ks
        h = jnp.concatenate([jnp.cos(xk), jnp.sin(xk)], axis=-1)
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width, kernel_init=kinit)(h))
        return nn.Dense(self.rank, kernel_init=kinit)(h)


time_model  = TimeNet()
space_model = SpaceNet()

LR        = 1e-3
SAVE_MARGIN_S = 10.0


def _ic_fourier_fit(kfit: int):
    """Least-squares Fourier fit of the IC (ks_ic.csv): the ONLY field data
    a candidate may read. Returns (a0, ak[kfit], bk[kfit]) so the base field
    and its exact analytic x-derivatives are available at any x."""
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x, u = ic[:, 0], ic[:, 1]
    ks = np.arange(1, kfit + 1)
    cols = [np.ones_like(x)]
    for k in ks:
        cols.append(np.cos(k * x)); cols.append(np.sin(k * x))
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, u, rcond=None)
    a0 = coef[0]
    ak = coef[1::2]
    bk = coef[2::2]
    return float(a0), ak.astype(np.float32), bk.astype(np.float32), ks.astype(np.float32)


def _ic_base_np(a0, ak, bk, ks, x, order=0):
    """Analytic base_ic and its d^order/dx^order at points x (numpy)."""
    xk = x[:, None] * ks[None, :]
    c, s = np.cos(xk), np.sin(xk)
    if order == 0:
        return a0 + (ak[None, :] * c + bk[None, :] * s).sum(1)
    kp = ks ** order
    # d/dx: cos->-sin, sin->cos ; each derivative multiplies by k and rotates.
    # order mod 4 phase: 0:(c,s) 1:(-s,c) 2:(-c,-s) 3:(s,-c)
    r = order % 4
    if r == 0:   ca, sa = c, s
    elif r == 1: ca, sa = -s, c
    elif r == 2: ca, sa = -c, -s
    else:        ca, sa = s, -c
    return ((kp[None, :] * ak[None, :]) * ca + (kp[None, :] * bk[None, :]) * sa).sum(1)


def _nn_predict(pf, pg, s, x):
    """NN_w(s,x) = sum_j F_j(s) G_j(x)."""
    F = time_model.apply(pf, s.reshape(-1, 1))
    G = space_model.apply(pg, x.reshape(-1, 1))
    return jnp.sum(F * G, axis=1)


def _ic_base_jax(coef, x):
    a0, ak, bk, ks = coef
    xk = x[:, None] * ks[None, :]
    return a0 + jnp.sum(ak[None, :] * jnp.cos(xk) + bk[None, :] * jnp.sin(xk), axis=1)


# ══════════ Public contract ══════════

def predict_fn(params, X):
    bounds = params["bounds"]
    wins = params["win"]
    coef = (params["ic_a0"], params["ic_ak"], params["ic_bk"], params["ic_ks"])
    t = X[:, 0]
    x = X[:, 1]
    u = jnp.zeros_like(t)
    base = _ic_base_jax(coef, x)      # base field at each point's x (window 0: IC)
    W = len(wins)
    for w in range(W):
        t_lo = bounds[w]
        t_hi = bounds[w + 1]
        dt = t_hi - t_lo
        s = jnp.clip((t - t_lo) / dt, 0.0, 1.0)
        pf, pg = wins[w]["f"], wins[w]["g"]
        nn_s = _nn_predict(pf, pg, s, x)                 # NN_w at query s
        nn_1 = _nn_predict(pf, pg, jnp.ones_like(x), x)  # NN_w at s=1
        uw = base + s * nn_s
        base_next = base + nn_1
        mask = (t >= t_lo) if w == W - 1 else ((t >= t_lo) & (t < t_hi))
        u = jnp.where(mask, uw, u)
        base = base_next
    return {"u": u}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    xs = jnp.asarray(np.linspace(X_MIN, X_MAX, NX, endpoint=False).astype(np.float32))
    ss = jnp.asarray(np.linspace(0.0, 1.0, NT_WIN + 1)[1:].astype(np.float32))
    s_col = ss.reshape(-1, 1)

    # --- initial-condition base field (exact analytic derivatives) ---
    a0, ak, bk, ks_fit = _ic_fourier_fit(IC_KFIT)
    xs_np = np.linspace(X_MIN, X_MAX, NX, endpoint=False).astype(np.float64)
    base0    = jnp.asarray(_ic_base_np(a0, ak, bk, ks_fit, xs_np, 0).astype(np.float32))
    base0_x  = jnp.asarray(_ic_base_np(a0, ak, bk, ks_fit, xs_np, 1).astype(np.float32))
    base0_xx = jnp.asarray(_ic_base_np(a0, ak, bk, ks_fit, xs_np, 2).astype(np.float32))
    base0_x4 = jnp.asarray(_ic_base_np(a0, ak, bk, ks_fit, xs_np, 4).astype(np.float32))

    bounds = tuple(float(b) for b in np.linspace(T_MIN, T_MAX, N_WIN + 1))

    k0, k1 = random.split(rng)
    pf = time_model.init(k0, jnp.zeros([1, 1]))
    pg = space_model.init(k1, jnp.zeros([1, 1]))
    params = {"f": pf, "g": pg}

    def f_vec(pf, s):
        return time_model.apply(pf, jnp.reshape(s, (1, 1)))[0]

    def g_vec(pg, x):
        return space_model.apply(pg, jnp.reshape(x, (1, 1)))[0]

    def f_and_ds(pf, s):
        f, df = jax.jvp(lambda z: f_vec(pf, z), (s,), (jnp.ones_like(s),))
        return f, df

    _z = lambda x: jnp.zeros_like(x)
    _o = lambda x: jnp.ones_like(x)

    def g_all(pg, x):
        y0, (y1, y2, y3, y4) = jet(lambda z: g_vec(pg, z), (x,),
                                   ((_o(x), _z(x), _z(x), _z(x)),))
        return y0, y1, y2, y4

    # ---- all windows: hard ansatz u = base + s*NN, base = prev endpoint field ----
    def lossk(params, dt, base, base_x, base_xx, base_xxxx):
        pf, pg = params["f"], params["g"]
        F, dFds = vmap(lambda s: f_and_ds(pf, s))(ss)
        G, G1, G2, G4 = vmap(lambda x: g_all(pg, x))(xs)
        NN     = F    @ G.T
        NN_s   = dFds @ G.T
        NN_x   = F    @ G1.T
        NN_xx  = F    @ G2.T
        NN_xxxx = F   @ G4.T
        U      = base[None, :]      + s_col * NN
        U_t    = (NN + s_col * NN_s) / dt
        U_x    = base_x[None, :]    + s_col * NN_x
        U_xx   = base_xx[None, :]   + s_col * NN_xx
        U_xxxx = base_xxxx[None, :] + s_col * NN_xxxx
        r = U_t + V1 * U * U_x + V2 * U_xx + V3 * U_xxxx
        return jnp.mean(r ** 2)

    lk_ag = jit(jax.value_and_grad(lossk))
    optimizer = optax.adam(learning_rate=LR)
    lbfgs_opt = optax.lbfgs()

    @jit
    def updatek(params, opt_state, dt, base, base_x, base_xx, base_xxxx):
        loss, grads = lk_ag(params, dt, base, base_x, base_xx, base_xxxx)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    @jit
    def lbfgs_step(params, state, dt, base, base_x, base_xx, base_xxxx):
        def vfn(p):
            return lossk(p, dt, base, base_x, base_xx, base_xxxx)
        value, grad = jax.value_and_grad(vfn)(params)
        updates, state = lbfgs_opt.update(grad, state, params,
                                          value=value, grad=grad, value_fn=vfn)
        return optax.apply_updates(params, updates), state

    @jit
    def endpoint_base(params, base, base_x, base_xx, base_xxxx):
        """New base arrays after a window: base + NN_w(s=1)."""
        pf, pg = params["f"], params["g"]
        fend = f_vec(pf, jnp.array(1.0, dtype=ss.dtype))     # [RANK]
        G, G1, G2, G4 = vmap(lambda x: g_all(pg, x))(xs)
        add    = G  @ fend
        add_x  = G1 @ fend
        add_xx = G2 @ fend
        add_x4 = G4 @ fend
        return (base + add, base_x + add_x, base_xx + add_xx, base_xxxx + add_x4)

    ADAM_FRAC = 0.9      # fraction of each window's wall spent on Adam before LBFGS
    train_budget = TRAIN_TIME - SAVE_MARGIN_S
    _wfrac = np.array([1.0 + 1.0 * (N_WIN - 1 - w) for w in range(N_WIN)])
    _cum = np.cumsum(_wfrac) / _wfrac.sum()
    _cum_prev = np.concatenate([[0.0], _cum[:-1]])
    wins = []
    total_steps = 0
    base, bx, bxx, bxxxx = base0, base0_x, base0_xx, base0_x4
    for w in range(N_WIN):
        dt = jnp.float32(bounds[w + 1] - bounds[w])
        opt_state = optimizer.init(params)
        t0f, t1f = float(_cum_prev[w]), float(_cum[w])
        adam_deadline = _T0 + train_budget * (t0f + ADAM_FRAC * (t1f - t0f))
        deadline = _T0 + train_budget * t1f
        # ---- Adam phase ----
        params, opt_state, aloss = updatek(params, opt_state, dt, base, bx, bxx, bxxxx)
        steps = 1
        while time.time() < adam_deadline:
            params, opt_state, aloss = updatek(params, opt_state, dt, base, bx, bxx, bxxxx)
            steps += 1
        # ---- LBFGS polish phase (full-batch, 2nd-order); guarded ----
        adam_params = params
        adam_loss = float(aloss)
        lstate = lbfgs_opt.init(params)
        params, lstate = lbfgs_step(params, lstate, dt, base, bx, bxx, bxxxx)
        steps += 1
        while time.time() < deadline:
            params, lstate = lbfgs_step(params, lstate, dt, base, bx, bxx, bxxxx)
            steps += 1
        # revert if LBFGS did not improve (or diverged) — bounds to pure Adam
        final_loss = float(lossk(params, dt, base, bx, bxx, bxxxx))
        if not np.isfinite(final_loss) or final_loss > adam_loss:
            params = adam_params
        total_steps += steps
        wins.append({"f": params["f"], "g": params["g"]})
        if w < N_WIN - 1:
            base, bx, bxx, bxxxx = endpoint_base(params, base, bx, bxx, bxxxx)
            base = jax.lax.stop_gradient(base)
            bx = jax.lax.stop_gradient(bx)
            bxx = jax.lax.stop_gradient(bxx)
            bxxxx = jax.lax.stop_gradient(bxxxx)

    out = {"bounds": bounds, "win": wins,
           "ic_a0": np.float32(a0), "ic_ak": ak, "ic_bk": bk, "ic_ks": ks_fit}
    return out, total_steps
