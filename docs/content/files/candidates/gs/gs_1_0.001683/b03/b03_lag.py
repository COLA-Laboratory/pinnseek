"""Pseudo-time-stepped separable PINN (fork of b02_lr15, the champion) relaxing
against a LAGGED iterate instead of the immediately preceding one, so the
kb1-130 pseudo-time step is set by tau alone and not by the optimiser's step
size. vs b02_lr15: prev refreshed every PTS_LAG = 20 steps.
"""
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
import soap_jax
from flax import linen as nn

# ══════════ architecture / training hyperparameters ══════════

RANK      = 64      # rank of the (t) x (x,y) product ansatz
K_SPACE   = 8       # periodic harmonics per spatial axis (period 2 -> k = n*pi)
K_TIME    = 4       # sine/cosine features of the warped time coordinate
W_T, D_T  = 64, 3   # time-trunk width / depth  (modified MLP, kb1-050)
W_S, D_S  = 320, 3  # space-trunk width / depth

TG        = 0.02    # hard-IC ramp: g(t) = 1 - exp(-t/TG)
TW        = 0.02    # time-feature warp scale
N_T       = 200     # time slices per residual batch
N_S       = 1728    # spatial collocation points per residual batch
S_U       = 100.0   # residual normalisation, u equation
S_V       = 50.0    # residual normalisation, v equation (weighted 4x)
KAPPA0    = 8.0     # initial causal-weight strength (kb1-084), annealed to 0
KAPPA_END = 1.0     # causality released only at the very end of training
T_POW     = 2.0     # time sampling: t = T_MAX * xi**T_POW (denser near t=0)

# --- pseudo-time stepping (kb1-130) ---
TAU0      = 1.0     # initial pseudo-time step (large tau ~ plain residual loss)
PTS_EVERY = 100     # steps between Barzilai-Borwein tau refreshes
PTS_LAG   = 20      # steps between refreshes of the relaxation reference `prev`
PTS_SUB   = 8       # spatial subsample factor used by the tau estimate
PTS_BETA  = 0.5     # EMA weight on the new tau estimate
PTS_GAIN  = 3.0     # gain on the BB estimate (kb1-130's gamma)

LR_PEAK   = 1.5e-3
EST_STEPS = 110000
EMA_DECAY = 0.999   # Polyak averaging of the iterates   # step-count budget used by the cosine schedule
LR_FINAL  = 1e-4
WARMUP    = 400
CHECK_EVERY   = 100
SAVE_MARGIN_S = 8.0
PRED_CHUNK    = 32768


# ══════════ feature maps ══════════

_LOGW = math.log1p(T_MAX / TW)


def tfeat(t):
    """t: [N,1] -> time features (warped coordinate + its sin/cos bank)."""
    tau = jnp.log1p(t / TW) / _LOGW                      # [0,1], stretches t≈0
    n = jnp.arange(1, K_TIME + 1, dtype=t.dtype)
    ang = math.pi * tau * n
    return jnp.concatenate(
        [2.0 * t / T_MAX - 1.0, 2.0 * tau - 1.0, jnp.sin(ang), jnp.cos(ang)], axis=-1)


def sfeat(s):
    """s: [N,2] -> exactly-periodic embedding of (x,y) with period 2 (kb1-012).

    Every downstream operation is a smooth function of these features, so the
    network and all its derivatives are periodic to machine precision — the
    periodic-BC penalty of the baseline is removed entirely.
    """
    n = jnp.arange(1, K_SPACE + 1, dtype=s.dtype)
    ax = math.pi * s[:, 0:1] * n
    ay = math.pi * s[:, 1:2] * n
    return jnp.concatenate([jnp.cos(ax), jnp.sin(ax), jnp.cos(ay), jnp.sin(ay)], axis=-1)


def ic_parts(s):
    """Analytic initial condition and its Laplacian at s: [N,2] -> ([N,2],[N,2])."""
    du = (s[:, 0] + 0.05) ** 2 + (s[:, 1] + 0.02) ** 2
    dv = (s[:, 0] - 0.05) ** 2 + (s[:, 1] - 0.02) ** 2
    eu, ev = jnp.exp(-10.0 * du), jnp.exp(-10.0 * dv)
    uv0 = jnp.stack([1.0 - eu, ev], axis=-1)
    lap0 = jnp.stack([-eu * (400.0 * du - 40.0), ev * (400.0 * dv - 40.0)], axis=-1)
    return uv0, lap0


def gfun(t):
    return 1.0 - jnp.exp(-t / TG)


def gpfun(t):
    return jnp.exp(-t / TG) / TG


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


def _A(pt, t):
    return TNET.apply(pt, tfeat(t)).reshape(t.shape[0], RANK, 2)


def _B(ps, s):
    return SNET.apply(ps, sfeat(s)).reshape(s.shape[0], RANK, 2)


# ══════════ public contract ══════════

def predict_fn(params, X):
    """X: [BS,3] with columns (t, x, y) -> {"u": [BS], "v": [BS]}."""
    X = jnp.asarray(X)
    out = []
    for i in range(0, X.shape[0], PRED_CHUNK):
        xb = X[i:i + PRED_CHUNK]
        t, s = xb[:, 0:1], xb[:, 1:3]
        A = _A(params["t"], t)
        B = _B(params["s"], s)
        uv0, _ = ic_parts(s)
        out.append(uv0 + gfun(t) * (CR * jnp.sum(A * B, axis=1)))
    F = jnp.concatenate(out, axis=0)
    return {"u": F[:, 0], "v": F[:, 1]}


# ══════════ residual on the (t) x (x,y) tensor grid ══════════

def _lap_s(fn, s):
    """Laplacian of fn: [N,2]->[N,...] w.r.t. the two spatial columns of s."""
    tot = None
    for j in (0, 1):
        e = jnp.zeros_like(s).at[:, j].set(1.0)
        d1 = lambda z: jax.jvp(fn, (z,), (e,))[1]
        d2 = jax.jvp(d1, (s,), (e,))[1]
        tot = d2 if tot is None else tot + d2
    return tot


S_UV = jnp.array([S_U, S_V])


def _field(params, t, s):
    """Field values on the (t) x (x,y) tensor grid -> [Nt,Ns,2]. No derivatives."""
    A = _A(params["t"], t)
    B = _B(params["s"], s)
    uv0, _ = ic_parts(s)
    return uv0[None] + gfun(t)[:, :, None] * (CR * jnp.einsum("trc,src->tsc", A, B))


def fields_and_res(params, t, s):
    """Field values and normalised PDE residuals. t:[Nt,1] s:[Ns,2] -> 2x[Nt,Ns,2]."""
    pt, ps = params["t"], params["s"]
    A = _A(pt, t)                                            # [Nt,R,2]
    At = jax.jvp(lambda z: _A(pt, z), (t,), (jnp.ones_like(t),))[1]
    B = _B(ps, s)                                            # [Ns,R,2]
    Bl = _lap_s(lambda z: _B(ps, z), s)                      # [Ns,R,2]

    NV = CR * jnp.einsum("trc,src->tsc", A, B)
    NT = CR * jnp.einsum("trc,src->tsc", At, B)
    NL = CR * jnp.einsum("trc,src->tsc", A, Bl)

    uv0, lap0 = ic_parts(s)
    g = gfun(t)[:, :, None]                                  # [Nt,1,1]
    gp = gpfun(t)[:, :, None]

    F = uv0[None] + g * NV
    Ft = gp * NV + g * NT
    Fl = lap0[None] + g * NL

    u, v = F[..., 0], F[..., 1]
    react = C1 * u * v * v
    r_u = Ft[..., 0] - EPS1 * Fl[..., 0] - B1 * (1.0 - u) + react
    r_v = Ft[..., 1] - EPS2 * Fl[..., 1] + B2 * v - react
    return F, jnp.stack([r_u, r_v], axis=-1) / S_UV


def loss_fn(params, prev, t, s, kappa, tau):
    """Pseudo-time-relaxed residual (kb1-130): ((F - F_prev)/(tau*S) + r)^2,
    causally weighted per time slice."""
    F, r = fields_and_res(params, t, s)
    Fp = jax.lax.stop_gradient(_field(prev, t, s))
    L = jnp.mean(jnp.sum(((F - Fp) / (tau * S_UV) + r) ** 2, axis=-1), axis=1)
    cum = jnp.concatenate([jnp.zeros((1,), L.dtype), jnp.cumsum(L)[:-1]])
    w = jax.lax.stop_gradient(jnp.exp(-kappa * cum / (jnp.sum(L) + 1e-12)))
    return jnp.sum(w * L) / jnp.sum(w)


def bb_tau(params, prev, t, s):
    """Barzilai-Borwein pseudo-time estimate ||dF/S|| / ||dr|| on a subsample."""
    ss = s[::PTS_SUB]
    F1, r1 = fields_and_res(params, t, ss)
    F0, r0 = fields_and_res(prev, t, ss)
    return PTS_GAIN * (jnp.linalg.norm((F1 - F0) / S_UV)
                       / (jnp.linalg.norm(r1 - r0) + 1e-12)) / PTS_LAG


# ══════════ training ══════════

def _sample(key, nt, ns):
    k1, k2, k3 = random.split(key, 3)
    xi = (jnp.arange(nt) + random.uniform(k1, (nt,))) / nt      # stratified
    t = (T_MAX * xi ** T_POW)[:, None]
    s = jnp.concatenate([random.uniform(k2, (ns, 1), minval=X_MIN, maxval=X_MAX),
                         random.uniform(k3, (ns, 1), minval=Y_MIN, maxval=Y_MAX)], axis=1)
    return t, s


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam training of the separable PINN; returns (params, steps)."""
    k1, k2, rng = random.split(rng, 3)
    params = {"t": TNET.init(k1, jnp.zeros((1, tfeat(jnp.zeros((1, 1))).shape[-1]))),
              "s": SNET.init(k2, jnp.zeros((1, sfeat(jnp.zeros((1, 2))).shape[-1])))}

    sched = optax.warmup_cosine_decay_schedule(0.0, LR_PEAK, WARMUP, EST_STEPS, LR_FINAL)
    opt = soap_jax.soap(learning_rate=sched, precondition_frequency=10)
    opt_state = opt.init(params)

    @jit
    def update(params, prev, ema, opt_state, key, lr, kappa, tau, refresh, new_ref):
        t, s = _sample(key, N_T, N_S)
        loss, grads = jax.value_and_grad(loss_fn)(params, prev, t, s, kappa, tau)
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        ema = jax.tree_util.tree_map(lambda e, p: EMA_DECAY * e + (1.0 - EMA_DECAY) * p,
                                     ema, new_params)
        tau = jax.lax.cond(
            refresh,
            lambda: (1.0 - PTS_BETA) * tau + PTS_BETA * bb_tau(new_params, prev, t, s),
            lambda: tau)
        prev = jax.tree_util.tree_map(
            lambda a, b: jnp.where(new_ref, a, b), params, prev)
        return new_params, prev, ema, opt_state, loss, tau

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    t_start = time.time()
    span = max(deadline - t_start, 1.0)

    ema = jax.tree_util.tree_map(lambda p: p, params)
    prev = jax.tree_util.tree_map(lambda p: p, params)
    tau = jnp.asarray(TAU0)
    rng, sk = random.split(rng)
    params, prev, ema, opt_state, loss, tau = update(
        params, prev, ema, opt_state, sk, LR_PEAK, KAPPA0, tau,
        jnp.asarray(True), jnp.asarray(True))
    step = 1
    last = float(loss)

    while step < MAX_ITER:
        now = time.time()
        if now > deadline:
            break
        for _ in range(CHECK_EVERY):
            prog = min(max((now - t_start) / span, 0.0), 1.0)
            if step < WARMUP:
                lr = LR_PEAK * (step + 1) / WARMUP
            else:
                lr = LR_FINAL + 0.5 * (LR_PEAK - LR_FINAL) * (1.0 + math.cos(math.pi * prog))
            kappa = KAPPA0 * max(0.0, 1.0 - prog / KAPPA_END)
            rng, sk = random.split(rng)
            params, prev, ema, opt_state, loss, tau = update(
                params, prev, ema, opt_state, sk, lr, kappa, tau,
                jnp.asarray(step % PTS_EVERY == 0),
                jnp.asarray(step % PTS_LAG == 0))
            step += 1
        last = float(loss)
        if eval_callback is not None:
            eval_callback(params, step, last)

    return ema, step
