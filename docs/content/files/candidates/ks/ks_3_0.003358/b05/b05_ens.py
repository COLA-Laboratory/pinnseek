"""Ensemble time-marching SPINN (jet x-derivs) — parent b04_jet10 (~0.06-0.08).
Change vs parent: N_ENS independent (f,g) trajectories vmapped over one
ensemble axis, each doing the same 10-window march; predict_fn averages
the member fields. Bet: SPINN sub-nets are tiny, so the GPU absorbs the
K-fold FLOPs at little throughput cost, and averaging cuts the large
run-to-run variance b04 found (0.06->0.30 tails) toward a lower, robust score.
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
N_ENS   = 4      # independent trajectories, averaged in predict_fn
COMBINE = "mean"  # "mean" or "median"


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

WEIGHT_IC = 100.0
LR        = 1e-3
SAVE_MARGIN_S = 10.0


def _member_predict(pf, pg, s, x):
    F = time_model.apply(pf, s.reshape(-1, 1))
    G = space_model.apply(pg, x.reshape(-1, 1))
    return jnp.sum(F * G, axis=1)


# ══════════ Public contract ══════════

def predict_fn(params, X):
    bounds = params["bounds"]
    wins = params["win"]
    combine = params.get("combine", "mean")
    t = X[:, 0]
    x = X[:, 1]
    u = jnp.zeros_like(t)
    for w in range(len(wins)):
        t_lo = bounds[w]
        t_hi = bounds[w + 1]
        dt = t_hi - t_lo
        s = jnp.clip((t - t_lo) / dt, 0.0, 1.0)
        pf, pg = wins[w]["f"], wins[w]["g"]          # stacked over ensemble axis
        uw_all = vmap(lambda a, b: _member_predict(a, b, s, x))(pf, pg)  # [E, Npts]
        if combine == "median":
            uw = jnp.median(uw_all, axis=0)
        else:
            uw = jnp.mean(uw_all, axis=0)
        mask = (t >= t_lo) if w == len(wins) - 1 else ((t >= t_lo) & (t < t_hi))
        u = jnp.where(mask, uw, u)
    return {"u": u}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    xs = jnp.asarray(np.linspace(X_MIN, X_MAX, NX, endpoint=False).astype(np.float32))
    ss = jnp.asarray(np.linspace(0.0, 1.0, NT_WIN + 1)[1:].astype(np.float32))

    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    ic_x0 = jnp.asarray(ic[:, 0].astype(np.float32))
    ic_u0 = jnp.asarray(ic[:, 1].astype(np.float32))

    bounds = tuple(float(b) for b in np.linspace(T_MIN, T_MAX, N_WIN + 1))

    # independent init per ensemble member (stacked leading axis E)
    keys = random.split(rng, 2 * N_ENS)
    kf, kg = keys[:N_ENS], keys[N_ENS:]
    pf = vmap(lambda k: time_model.init(k, jnp.zeros([1, 1])))(kf)
    pg = vmap(lambda k: space_model.init(k, jnp.zeros([1, 1])))(kg)
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

    def member_loss(pf, pg, dt, ic_x, ic_u):
        """Loss for one ensemble member (its own handoff IC)."""
        F, dFds = vmap(lambda s: f_and_ds(pf, s))(ss)
        G, G1, G2, G4 = vmap(lambda x: g_all(pg, x))(xs)
        U      = F    @ G.T
        U_t    = (dFds @ G.T) / dt
        U_x    = F    @ G1.T
        U_xx   = F    @ G2.T
        U_xxxx = F    @ G4.T
        r = U_t + V1 * U * U_x + V2 * U_xx + V3 * U_xxxx
        pde_loss = jnp.mean(r ** 2)
        f0 = f_vec(pf, jnp.array(0.0, dtype=ss.dtype))
        G_ic = vmap(lambda x: g_vec(pg, x))(ic_x)
        ic_loss = jnp.mean((G_ic @ f0 - ic_u) ** 2)
        return pde_loss + WEIGHT_IC * ic_loss

    def loss_fn(params, dt, ic_x, ic_u):
        # ic_u: [E, Nic] — each member has its own handoff IC
        losses = vmap(lambda pf, pg, icu: member_loss(pf, pg, dt, ic_x, icu))(
            params["f"], params["g"], ic_u)
        return jnp.mean(losses)

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(learning_rate=LR)

    @jit
    def update(params, opt_state, dt, ic_x, ic_u):
        loss, grads = loss_and_grad(params, dt, ic_x, ic_u)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def members_endpoint(params):
        # each member's field at s=1 (window end) -> [E, NX]
        return vmap(lambda pf, pg: _member_predict(pf, pg, jnp.ones_like(xs), xs))(
            params["f"], params["g"])

    train_budget = TRAIN_TIME - SAVE_MARGIN_S
    _wfrac = np.array([1.0 + 1.0 * (N_WIN - 1 - w) for w in range(N_WIN)])
    _cum = np.cumsum(_wfrac) / _wfrac.sum()
    wins = []
    total_steps = 0
    ic_x = ic_x0
    ic_u = jnp.broadcast_to(ic_u0, (N_ENS, ic_u0.shape[0]))   # shared true IC
    for w in range(N_WIN):
        dt = jnp.float32(bounds[w + 1] - bounds[w])
        opt_state = optimizer.init(params)               # fresh momentum per window
        deadline = _T0 + train_budget * float(_cum[w])
        params, opt_state, _ = update(params, opt_state, dt, ic_x, ic_u)  # compile on w=0
        steps = 1
        while time.time() < deadline:
            params, opt_state, _ = update(params, opt_state, dt, ic_x, ic_u)
            steps += 1
        total_steps += steps
        win = {"f": params["f"], "g": params["g"]}
        wins.append(win)
        if w < N_WIN - 1:
            u_end = members_endpoint(params)             # [E, NX]
            ic_x = xs
            ic_u = jax.lax.stop_gradient(u_end)

    return {"bounds": bounds, "win": wins, "combine": COMBINE}, total_steps
