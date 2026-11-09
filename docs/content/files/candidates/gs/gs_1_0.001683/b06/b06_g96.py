"""Tucker-separable PINN (fork of b03_tw80): the 2-D space trunk becomes two
1-D trunks x/y plus a learned core W[r,p,q,c], so the Laplacian costs 2 1-D
passes and collocation is a random tensor grid. Rest (kb1-130 pseudo-time,
causality, hard IC) as parent. vs b06_tuck: N_X,N_Y 64 -> 96."""
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

RANK      = 64      # time rank of the (t) x (x,y) product ansatz
RX, RY    = 48, 48  # x / y mode counts of the Tucker core (SVD floor ~1e-4)
K_SPACE   = 8       # periodic harmonics per spatial axis (period 2 -> k = n*pi)
K_TIME    = 4       # sine/cosine features of the warped time coordinate
W_T, D_T  = 64, 3   # time-trunk width / depth  (modified MLP, kb1-050)
W_X, D_X  = 128, 3  # 1-D space-trunk width / depth

TG        = 0.12    # hard-IC ramp: g(t) = 1 - exp(-t/TG)
TW        = 0.80    # time-feature warp scale
N_T       = 200     # time slices per residual batch
N_X, N_Y  = 96, 96  # spatial collocation points per axis (random tensor grid)
S_U       = 100.0   # residual normalisation, u equation
S_V       = 50.0    # residual normalisation, v equation (weighted 4x)
KAPPA0    = 8.0     # initial causal-weight strength (kb1-084), annealed to 0
KAPPA_END = 1.0     # causality released only at the very end of training
T_POW     = 2.0     # time sampling: t = T_MAX * xi**T_POW (denser near t=0)

# --- pseudo-time stepping (kb1-130) ---
TAU0      = 1.0     # initial pseudo-time step (large tau ~ plain residual loss)
PTS_EVERY = 100     # steps between Barzilai-Borwein tau refreshes
PTS_SUB   = 6       # per-axis spatial subsample factor used by the tau estimate
PTS_BETA  = 0.5     # EMA weight on the new tau estimate
PTS_GAIN  = 3.0     # gain on the BB estimate (kb1-130's gamma)

LR_PEAK   = 1.5e-3
EMA_DECAY = 0.999   # Polyak averaging of the iterates
LR_FINAL  = 4e-4
WARMUP    = 400
CHECK_EVERY   = 100
SAVE_MARGIN_S = 8.0
PRED_CHUNK    = 4096
MAX_PRECOND   = 1024   # soap: skip preconditioning of dims above this


# ══════════ feature maps ══════════

_LOGW = math.log1p(T_MAX / TW)


def tfeat(t):
    """t: [N,1] -> time features (warped coordinate + its sin/cos bank)."""
    tau = jnp.log1p(t / TW) / _LOGW                      # [0,1], stretches t≈0
    n = jnp.arange(1, K_TIME + 1, dtype=t.dtype)
    ang = math.pi * tau * n
    return jnp.concatenate(
        [2.0 * t / T_MAX - 1.0, 2.0 * tau - 1.0, jnp.sin(ang), jnp.cos(ang)], axis=-1)


def afeat(a):
    """a: [N,1] -> exactly-periodic 1-D embedding with period 2 (kb1-012)."""
    n = jnp.arange(1, K_SPACE + 1, dtype=a.dtype)
    ang = math.pi * a * n
    return jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=-1)


def ic_parts(s):
    """Analytic initial condition and its Laplacian at s: [N,2] -> ([N,2],[N,2])."""
    du = (s[:, 0] + 0.05) ** 2 + (s[:, 1] + 0.02) ** 2
    dv = (s[:, 0] - 0.05) ** 2 + (s[:, 1] - 0.02) ** 2
    eu, ev = jnp.exp(-10.0 * du), jnp.exp(-10.0 * dv)
    uv0 = jnp.stack([1.0 - eu, ev], axis=-1)
    lap0 = jnp.stack([-eu * (400.0 * du - 40.0), ev * (400.0 * dv - 40.0)], axis=-1)
    return uv0, lap0


def ic_grid(x, y):
    """Same initial condition on the tensor grid: x:[Nx,1] y:[Ny,1] -> 2x[Nx,Ny,2]."""
    X = x[:, 0:1]            # [Nx,1]
    Y = y[:, 0].reshape(1, -1)  # [1,Ny]
    du = (X + 0.05) ** 2 + (Y + 0.02) ** 2
    dv = (X - 0.05) ** 2 + (Y - 0.02) ** 2
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
XNET = Trunk(W_X, D_X, 2 * RX)
YNET = Trunk(W_X, D_X, 2 * RY)
CR = 1.0 / math.sqrt(RANK)


def _A(pt, t):
    return TNET.apply(pt, tfeat(t)).reshape(t.shape[0], RANK, 2)


def _P(px, x):
    return XNET.apply(px, afeat(x)).reshape(x.shape[0], RX, 2)


def _Q(py, y):
    return YNET.apply(py, afeat(y)).reshape(y.shape[0], RY, 2)


def _basis(params, x, y):
    """Spatial basis S2[r,x,y,c] = sum_pq W[r,p,q,c] P[x,p,c] Q[y,q,c]."""
    W = params["w"]
    P, Q = _P(params["x"], x), _Q(params["y"], y)
    S1 = jnp.einsum("rpqc,yqc->rpyc", W, Q)
    return jnp.einsum("rpyc,xpc->rxyc", S1, P)


# ══════════ public contract ══════════

def predict_fn(params, X):
    """X: [BS,3] with columns (t, x, y) -> {"u": [BS], "v": [BS]}."""
    X = jnp.asarray(X)
    W = params["w"]
    out = []
    for i in range(0, X.shape[0], PRED_CHUNK):
        xb = X[i:i + PRED_CHUNK]
        t, s = xb[:, 0:1], xb[:, 1:3]
        A = _A(params["t"], t)                       # [n,R,2]
        P = _P(params["x"], s[:, 0:1])                # [n,RX,2]
        Q = _Q(params["y"], s[:, 1:2])                # [n,RY,2]
        Z = P[:, :, None, :] * Q[:, None, :, :]       # [n,RX,RY,2]
        Yv = jnp.einsum("npqc,rpqc->nrc", Z, W)       # [n,R,2]
        uv0, _ = ic_parts(s)
        out.append(uv0 + gfun(t) * (CR * jnp.sum(A * Yv, axis=1)))
    F = jnp.concatenate(out, axis=0)
    return {"u": F[:, 0], "v": F[:, 1]}


# ══════════ residual on the (t) x (x) x (y) tensor grid ══════════

def _d2(fn, a):
    """Second derivative of fn: [N,1]->[N,...] w.r.t. the single column of a."""
    e = jnp.ones_like(a)
    d1 = lambda z: jax.jvp(fn, (z,), (e,))[1]
    return jax.jvp(d1, (a,), (e,))[1]


S_UV = jnp.array([S_U, S_V])


def _field(params, t, x, y):
    """Field on the tensor grid -> [Nt,Nx,Ny,2]. No derivatives."""
    A = _A(params["t"], t)
    S2 = _basis(params, x, y)
    uv0, _ = ic_grid(x, y)
    return uv0[None] + gfun(t)[:, :, None, None] * (CR * jnp.einsum("trc,rxyc->txyc", A, S2))


def fields_and_res(params, t, x, y):
    """Field values and normalised PDE residuals on the tensor grid."""
    pt, px, py, W = params["t"], params["x"], params["y"], params["w"]
    A = _A(pt, t)
    At = jax.jvp(lambda z: _A(pt, z), (t,), (jnp.ones_like(t),))[1]

    P = _P(px, x)
    Q = _Q(py, y)
    Pxx = _d2(lambda z: _P(px, z), x)
    Qyy = _d2(lambda z: _Q(py, z), y)

    S1 = jnp.einsum("rpqc,yqc->rpyc", W, Q)
    S2 = jnp.einsum("rpyc,xpc->rxyc", S1, P)
    S2xx = jnp.einsum("rpyc,xpc->rxyc", S1, Pxx)
    S1yy = jnp.einsum("rpqc,yqc->rpyc", W, Qyy)
    S2yy = jnp.einsum("rpyc,xpc->rxyc", S1yy, P)

    NV = CR * jnp.einsum("trc,rxyc->txyc", A, S2)
    NT = CR * jnp.einsum("trc,rxyc->txyc", At, S2)
    NL = CR * jnp.einsum("trc,rxyc->txyc", A, S2xx + S2yy)

    uv0, lap0 = ic_grid(x, y)
    g = gfun(t)[:, :, None, None]
    gp = gpfun(t)[:, :, None, None]

    F = uv0[None] + g * NV
    Ft = gp * NV + g * NT
    Fl = lap0[None] + g * NL

    u, v = F[..., 0], F[..., 1]
    react = C1 * u * v * v
    r_u = Ft[..., 0] - EPS1 * Fl[..., 0] - B1 * (1.0 - u) + react
    r_v = Ft[..., 1] - EPS2 * Fl[..., 1] + B2 * v - react
    return F, jnp.stack([r_u, r_v], axis=-1) / S_UV


def loss_fn(params, prev, t, x, y, kappa, tau):
    """Pseudo-time-relaxed residual (kb1-130): ((F - F_prev)/(tau*S) + r)^2,
    causally weighted per time slice."""
    F, r = fields_and_res(params, t, x, y)
    Fp = jax.lax.stop_gradient(_field(prev, t, x, y))
    L = jnp.mean(jnp.sum(((F - Fp) / (tau * S_UV) + r) ** 2, axis=-1), axis=(1, 2))
    cum = jnp.concatenate([jnp.zeros((1,), L.dtype), jnp.cumsum(L)[:-1]])
    w = jax.lax.stop_gradient(jnp.exp(-kappa * cum / (jnp.sum(L) + 1e-12)))
    return jnp.sum(w * L) / jnp.sum(w)


def bb_tau(params, prev, t, x, y):
    """Barzilai-Borwein pseudo-time estimate ||dF/S|| / ||dr|| on a subsample."""
    xs, ys = x[::PTS_SUB], y[::PTS_SUB]
    F1, r1 = fields_and_res(params, t, xs, ys)
    F0, r0 = fields_and_res(prev, t, xs, ys)
    return PTS_GAIN * (jnp.linalg.norm((F1 - F0) / S_UV)
                       / (jnp.linalg.norm(r1 - r0) + 1e-12))


# ══════════ training ══════════

def _sample(key, nt, nx, ny):
    k1, k2, k3 = random.split(key, 3)
    xi = (jnp.arange(nt) + random.uniform(k1, (nt,))) / nt      # stratified
    t = (T_MAX * xi ** T_POW)[:, None]
    x = random.uniform(k2, (nx, 1), minval=X_MIN, maxval=X_MAX)
    y = random.uniform(k3, (ny, 1), minval=Y_MIN, maxval=Y_MAX)
    return t, x, y


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline SOAP training of the Tucker-separable PINN -> (params, steps)."""
    k1, k2, k3, k4, rng = random.split(rng, 5)
    params = {
        "t": TNET.init(k1, jnp.zeros((1, tfeat(jnp.zeros((1, 1))).shape[-1]))),
        "x": XNET.init(k2, jnp.zeros((1, afeat(jnp.zeros((1, 1))).shape[-1]))),
        "y": YNET.init(k3, jnp.zeros((1, afeat(jnp.zeros((1, 1))).shape[-1]))),
        "w": random.normal(k4, (RANK, RX, RY, 2)) / math.sqrt(RX * RY),
    }

    sched = optax.warmup_cosine_decay_schedule(0.0, LR_PEAK, WARMUP, 110000, LR_FINAL)
    opt = soap_jax.soap(learning_rate=sched, precondition_frequency=10,
                        max_precond_dim=MAX_PRECOND)
    opt_state = opt.init(params)

    @jit
    def update(params, prev, ema, opt_state, key, lr, kappa, tau, refresh):
        t, x, y = _sample(key, N_T, N_X, N_Y)
        loss, grads = jax.value_and_grad(loss_fn)(params, prev, t, x, y, kappa, tau)
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        ema = jax.tree_util.tree_map(lambda e, p: EMA_DECAY * e + (1.0 - EMA_DECAY) * p,
                                     ema, new_params)
        tau = jax.lax.cond(
            refresh,
            lambda: (1.0 - PTS_BETA) * tau + PTS_BETA * bb_tau(new_params, params, t, x, y),
            lambda: tau)
        return new_params, params, ema, opt_state, loss, tau

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    t_start = time.time()
    span = max(deadline - t_start, 1.0)

    ema = jax.tree_util.tree_map(lambda p: p, params)
    prev = jax.tree_util.tree_map(lambda p: p, params)
    tau = jnp.asarray(TAU0)
    rng, sk = random.split(rng)
    params, prev, ema, opt_state, loss, tau = update(
        params, prev, ema, opt_state, sk, LR_PEAK, KAPPA0, tau, jnp.asarray(True))
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
                jnp.asarray(step % PTS_EVERY == 0))
            step += 1
        last = float(loss)
        if eval_callback is not None:
            eval_callback(params, step, last)

    return ema, step
