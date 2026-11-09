"""Sequential time-marching PINN (kb1-033 B / kb1-024 D / kb1-062): 8 windows,
one separable periodic net each, warm-started from its predecessor and soft-
anchored to it at the window edge. vs b01_sep.py: the global space-time fit
becomes 8 short anchored fits that cannot park on a stationary field."""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 2
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u", "v")

# --- Domain ---
T_MIN, T_MAX = 0.0, 0.78
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 2   # ru, rv
HAS_SOURCE              = True   # constant feed b1 in the u equation

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi")
PERIODIC_AXES  = ("x", "y")   # u(t, X_MIN, y) = u(t, X_MAX, y), likewise in y
GEOMETRY_KIND  = "periodic_box"

# --- Physical parameters ---
EPS1 = 0.2       # diffusivity of u
EPS2 = 0.1       # diffusivity of v
B1   = 40.0      # feed rate:  +b1 (1 - u)
B2   = 100.0     # kill rate:  -b2 v
C1   = 1000.0    # reaction:   -c1 u v^2
C2   = 1000.0    # reaction:   +c2 u v^2

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

import math
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jit, random
import numpy as np
import optax
from flax import linen as nn

# ══════════ architecture / training hyperparameters ══════════

NW        = 8       # time windows
EDGE_POW  = 1.5     # window edges T_MAX*(i/NW)**EDGE_POW — finer near t=0
EDGES     = (T_MAX * (np.arange(NW + 1) / NW) ** EDGE_POW).astype(np.float64)
W_SHARE   = np.array([2.0] + [1.0] * (NW - 1))       # window 0 gets a double share
W_SHARE   = np.cumsum(W_SHARE) / W_SHARE.sum()

RANK      = 64
K_SPACE   = 8       # periodic harmonics per spatial axis (period 2 -> k = n*pi)
K_TIME    = 5       # sin/cos features of the local window time
W_T, D_T  = 128, 4
W_S, D_S  = 256, 5

N_T       = 24      # time slices per residual batch (inside the window)
N_S       = 3600    # spatial collocation points per residual batch
N_IC      = 4096    # fixed spatial points carrying the window-edge anchor
R_SCALE   = 100.0   # residual normalisation
W_IC      = 20.0    # weight of the window-edge anchor loss
KAPPA     = 3.0     # causal weighting inside the window (kb1-084)
UV_REF    = np.array([0.78, 0.09], dtype=np.float32)   # output offset (field means)

LR_PEAK   = 2e-3
LR_FINAL  = 1e-4
WARMUP    = 200
CHECK_EVERY   = 50
SAVE_MARGIN_S = 8.0
PRED_CHUNK    = 32768


# ══════════ feature maps ══════════

def tfeat(tau):
    """tau: [N,1] local window time in [0,1] -> time features."""
    n = jnp.arange(1, K_TIME + 1, dtype=tau.dtype)
    ang = math.pi * tau * n
    return jnp.concatenate([2.0 * tau - 1.0, jnp.sin(ang), jnp.cos(ang)], axis=-1)


def sfeat(s):
    """s: [N,2] -> exactly-periodic embedding of (x,y), period 2 (kb1-012)."""
    n = jnp.arange(1, K_SPACE + 1, dtype=s.dtype)
    ax = math.pi * s[:, 0:1] * n
    ay = math.pi * s[:, 1:2] * n
    return jnp.concatenate([jnp.cos(ax), jnp.sin(ax), jnp.cos(ay), jnp.sin(ay)], axis=-1)


def ic_field(s):
    """Analytic t=0 initial condition at s: [N,2] -> [N,2]."""
    du = (s[:, 0] + 0.05) ** 2 + (s[:, 1] + 0.02) ** 2
    dv = (s[:, 0] - 0.05) ** 2 + (s[:, 1] - 0.02) ** 2
    return jnp.stack([1.0 - jnp.exp(-10.0 * du), jnp.exp(-10.0 * dv)], axis=-1)


# ══════════ modified-MLP trunk (kb1-050) ══════════

class Trunk(nn.Module):
    width: int
    depth: int
    out_dim: int

    @nn.compact
    def __call__(self, f):
        U = nn.tanh(nn.Dense(self.width)(f))
        V = nn.tanh(nn.Dense(self.width)(f))
        H = nn.tanh(nn.Dense(self.width)(f))
        for _ in range(self.depth - 1):
            Z = nn.tanh(nn.Dense(self.width)(H))
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(self.out_dim)(H)


TNET = Trunk(W_T, D_T, 2 * RANK)
SNET = Trunk(W_S, D_S, 2 * RANK)
CR = 1.0 / math.sqrt(RANK)
_UVR = jnp.asarray(UV_REF)


def _A(pt, tau):
    return TNET.apply(pt, tfeat(tau)).reshape(tau.shape[0], RANK, 2)


def _B(ps, s):
    return SNET.apply(ps, sfeat(s)).reshape(s.shape[0], RANK, 2)


def _field_diag(p, tau, s):
    """Field at matched (tau, s) pairs -> [N,2]."""
    return _UVR + CR * jnp.sum(_A(p["t"], tau) * _B(p["s"], s), axis=1)


# ══════════ public contract ══════════

def predict_fn(params, X):
    """X: [BS,3] with columns (t, x, y) -> {"u": [BS], "v": [BS]}."""
    Xn = np.asarray(X, dtype=np.float32)
    widx = np.clip(np.searchsorted(EDGES, Xn[:, 0], side="right") - 1, 0, NW - 1)
    out = np.zeros((Xn.shape[0], 2), dtype=np.float32)
    for w in range(NW):
        sel = np.nonzero(widx == w)[0]
        if sel.size == 0:
            continue
        ta, tb = float(EDGES[w]), float(EDGES[w + 1])
        for i in range(0, sel.size, PRED_CHUNK):
            ii = sel[i:i + PRED_CHUNK]
            xb = jnp.asarray(Xn[ii])
            tau = (xb[:, 0:1] - ta) / (tb - ta)
            out[ii] = np.asarray(_field_diag(params["nets"][w], tau, xb[:, 1:3]))
    return {"u": jnp.asarray(out[:, 0]), "v": jnp.asarray(out[:, 1])}


# ══════════ residual on the (tau) x (x,y) tensor grid ══════════

def _lap_s(fn, s):
    tot = None
    for j in (0, 1):
        e = jnp.zeros_like(s).at[:, j].set(1.0)
        d1 = lambda z: jax.jvp(fn, (z,), (e,))[1]
        d2 = jax.jvp(d1, (s,), (e,))[1]
        tot = d2 if tot is None else tot + d2
    return tot


def slice_losses(p, tau, s, hw):
    """Per-time-slice mean squared normalised residual. hw = window length."""
    pt, ps = p["t"], p["s"]
    A = _A(pt, tau)
    At = jax.jvp(lambda z: _A(pt, z), (tau,), (jnp.ones_like(tau),))[1] / hw
    B = _B(ps, s)
    Bl = _lap_s(lambda z: _B(ps, z), s)

    F = _UVR + CR * jnp.einsum("trc,src->tsc", A, B)
    Ft = CR * jnp.einsum("trc,src->tsc", At, B)
    Fl = CR * jnp.einsum("trc,src->tsc", A, Bl)

    u, v = F[..., 0], F[..., 1]
    react = C1 * u * v * v
    r_u = Ft[..., 0] - EPS1 * Fl[..., 0] - B1 * (1.0 - u) + react
    r_v = Ft[..., 1] - EPS2 * Fl[..., 1] + B2 * v - react
    return jnp.mean((r_u ** 2 + r_v ** 2) / (R_SCALE ** 2), axis=1)


def loss_fn(p, tau, s, hw, s_ic, y_ic):
    L = slice_losses(p, tau, s, hw)
    cum = jnp.concatenate([jnp.zeros((1,), L.dtype), jnp.cumsum(L)[:-1]])
    w = jax.lax.stop_gradient(jnp.exp(-KAPPA * cum / (jnp.sum(L) + 1e-12)))
    l_res = jnp.sum(w * L) / jnp.sum(w)
    f0 = _field_diag(p, jnp.zeros((s_ic.shape[0], 1), s_ic.dtype), s_ic)
    return l_res + W_IC * jnp.mean((f0 - y_ic) ** 2)


# ══════════ training ══════════

def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """March 8 anchored windows in sequence; returns ({"nets": [...]}, steps)."""
    k1, k2, rng = random.split(rng, 3)
    p0 = {"t": TNET.init(k1, jnp.zeros((1, tfeat(jnp.zeros((1, 1))).shape[-1]))),
          "s": SNET.init(k2, jnp.zeros((1, sfeat(jnp.zeros((1, 2))).shape[-1])))}

    opt = optax.inject_hyperparams(optax.adam)(learning_rate=LR_PEAK)

    @jit
    def update(p, opt_state, key, lr, ta, hw, s_ic, y_ic):
        k1, k2, k3 = random.split(key, 3)
        xi = (jnp.arange(N_T) + random.uniform(k1, (N_T,))) / N_T
        tau = xi[:, None]
        s = jnp.concatenate([random.uniform(k2, (N_S, 1), minval=X_MIN, maxval=X_MAX),
                             random.uniform(k3, (N_S, 1), minval=Y_MIN, maxval=Y_MAX)], axis=1)
        loss, grads = jax.value_and_grad(loss_fn)(p, tau, s, hw, s_ic, y_ic)
        opt_state.hyperparams["learning_rate"] = lr
        updates, opt_state = opt.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    t_start = time.time()
    span = max(deadline - t_start, 1.0)

    nets = []
    p = p0
    step = 0
    for w in range(NW):
        ta, tb = float(EDGES[w]), float(EDGES[w + 1])
        hw = tb - ta
        rng, ks = random.split(rng)
        s_ic = random.uniform(ks, (N_IC, 2), minval=X_MIN, maxval=X_MAX)
        if w == 0:
            y_ic = ic_field(s_ic)
        else:
            y_ic = jax.lax.stop_gradient(
                _field_diag(nets[w - 1], jnp.ones((N_IC, 1), s_ic.dtype), s_ic))
        opt_state = opt.init(p)
        w_end = t_start + span * float(W_SHARE[w])
        w_beg = time.time()
        w_span = max(w_end - w_beg, 1.0)
        wstep = 0
        while True:
            now = time.time()
            if now > min(w_end, deadline):
                break
            prog = min(max((now - w_beg) / w_span, 0.0), 1.0)
            lr = (LR_PEAK * (wstep + 1) / WARMUP if wstep < WARMUP else
                  LR_FINAL + 0.5 * (LR_PEAK - LR_FINAL) * (1.0 + math.cos(math.pi * prog)))
            for _ in range(CHECK_EVERY):
                rng, sk = random.split(rng)
                p, opt_state, loss = update(p, opt_state, sk, lr, ta, hw, s_ic, y_ic)
                wstep += 1
                step += 1
            loss.block_until_ready()
        nets.append(jax.tree_util.tree_map(lambda a: a, p))
        if eval_callback is not None:
            eval_callback({"nets": nets}, step, float(loss))
    while len(nets) < NW:                      # ran out of wall: reuse the last net
        nets.append(nets[-1])
    return {"nets": nets}, step
