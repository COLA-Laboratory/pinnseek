"""K=6 marching (parent b04_mtc) + layer-wise adaptive activations (L-LAAF,
kb1 016/001): tanh(n*a*x) with trainable per-layer slope a. b05 diagnostic:
window-0 delta is Adam-plateau-limited (not capacity); adaptive slopes aim to let
Adam reach lower per-window error at equal wall-time. All derivs autodiff.
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
import jax.flatten_util
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


# ══════════ Hyperparameters ══════════

K_WIN     = 6         # number of time windows
MX        = 16
RANK      = 128
WIDTH     = 96
DEPTH     = 3
NT        = 48        # collocation time samples (local tau in [0,1]) per step
NX        = 128
LR0       = 2.0e-3
IC_W      = 200.0     # IC / handoff match weight
BS_IC     = 512
SAVE_MARGIN_S = 8.0
CAUSAL_EPS   = 50.0   # within-window causal weighting strength (kb1 084)
CAUSAL_FLOOR = 0.10   # gentle floor so late tau-slices are not starved

DELTA = (T_MAX - T_MIN) / K_WIN   # window width in global time


def _embed_x(x):
    ks = jnp.arange(1, MX + 1, dtype=x.dtype)
    return jnp.concatenate([jnp.cos(x * ks), jnp.sin(x * ks)], axis=1)


def _embed_tau(tau):
    return tau                      # local time already in [0,1]


class AxisBody(nn.Module):
    embed: Callable
    width: int = WIDTH
    depth: int = DEPTH
    rank: int = RANK

    n_scale: float = 2.0    # fixed scaling factor n>=1 (kb1 001)

    @nn.compact
    def __call__(self, z):
        kinit = jax.nn.initializers.glorot_normal()
        n = self.n_scale

        def aact(x, name):
            # layer-wise adaptive slope a (init 1/n so n*a starts at 1)
            a = self.param(name, lambda k: jnp.asarray(1.0 / n, jnp.float32))
            return nn.tanh(n * a * x)

        h0 = self.embed(z)
        U = aact(nn.Dense(self.width, kernel_init=kinit)(h0), 'aU')
        V = aact(nn.Dense(self.width, kernel_init=kinit)(h0), 'aV')
        H = aact(nn.Dense(self.width, kernel_init=kinit)(h0), 'aH')
        for i in range(self.depth):
            Z = aact(nn.Dense(self.width, kernel_init=kinit)(H), f'aZ{i}')
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(self.rank, kernel_init=kinit)(H)


class SPINN(nn.Module):
    """Separable u(tau,x) = sum_r G_r(tau) H_r(x); tau is LOCAL window time [0,1]."""
    def setup(self):
        self.G = AxisBody(_embed_tau, name="G")
        self.H = AxisBody(_embed_x, name="H")

    def __call__(self, X):                 # X: [N,2] = (tau, x)
        return jnp.sum(self.G(X[:, 0:1]) * self.H(X[:, 1:2]), axis=1)

    def g_feats(self, tau):
        return self.G(tau)

    def h_feats(self, x):
        return self.H(x)


def _load_ic():
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    return (jnp.array(ic[:, 0].astype(np.float32)).reshape(-1, 1),
            jnp.array(ic[:, 1].astype(np.float32)))


# ══════════ Model handles shared across windows ══════════

_model = SPINN()
_x_ic_grid, _u_ic0 = _load_ic()


def _make_ops(unravel):
    def g_of(pf, tc):
        return _model.apply(unravel(pf), tc, method=SPINN.g_feats)

    def h_of(pf, xc):
        return _model.apply(unravel(pf), xc, method=SPINN.h_feats)

    def _d(f):
        return lambda c: jax.jvp(f, (c,), (jnp.ones_like(c),))[1]

    def resid_slab(pf, tau_col, x_col):
        gf = lambda t: g_of(pf, t); hf = lambda x: h_of(pf, x)
        g = gf(tau_col); gtau = _d(gf)(tau_col); h = hf(x_col)
        hx = _d(hf); hxx = _d(hx); hxxx = _d(hxx); hxxxx = _d(hxxx)
        h1 = hx(x_col); h2 = hxx(x_col); h4 = hxxxx(x_col)
        u    = jnp.einsum('tr,xr->tx', g, h)
        u_t  = jnp.einsum('tr,xr->tx', gtau, h) / DELTA   # global d/dt via chain rule
        u_x  = jnp.einsum('tr,xr->tx', g, h1)
        u_xx = jnp.einsum('tr,xr->tx', g, h2)
        u_xxxx = jnp.einsum('tr,xr->tx', g, h4)
        r = u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx
        return jnp.mean(r ** 2, axis=1)      # per-tau-slice residual [NT]

    def u_at_tau(pf, tau_scalar, x_col):
        g = g_of(pf, jnp.full((1, 1), tau_scalar, x_col.dtype))   # [1,R]
        h = h_of(pf, x_col)                                       # [Nx,R]
        return jnp.sum(g * h, axis=1)                             # [Nx]

    return g_of, h_of, resid_slab, u_at_tau


def predict_fn(params, X):
    """params = (stacked_flat [K,P], template_pytree). Global t -> window j,
    local tau=(t-jΔ)/Δ; evaluate window-j net. All windows share arch."""
    flats, template = params
    flats = jnp.asarray(flats, X.dtype)
    _, unravel = jax.flatten_util.ravel_pytree(
        jax.tree_util.tree_map(lambda a: jnp.asarray(a, X.dtype), template))

    t = X[:, 0]
    x = X[:, 1]
    j = jnp.clip(jnp.floor((t - T_MIN) / DELTA).astype(jnp.int32), 0, K_WIN - 1)
    tau = (t - (T_MIN + j.astype(X.dtype) * DELTA)) / DELTA
    Xloc = jnp.stack([tau, x], axis=1)

    def eval_one(pf):
        return _model.apply(unravel(pf), Xloc)     # [N]

    all_u = jax.vmap(eval_one)(flats)              # [K, N]
    u = jnp.take_along_axis(all_u, j[None, :], axis=0)[0]
    return {"u": u}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    key, rng = random.split(rng)
    template = _model.init(key, jnp.zeros([1, INPUT_DIM], jnp.float32))
    flat0, unravel = jax.flatten_util.ravel_pytree(template)
    P = flat0.size
    g_of, h_of, resid_slab, u_at_tau = _make_ops(unravel)

    def loss_fn(pf, tau_col, x_col, ic_x, ic_u):
        Lr_slices = resid_slab(pf, tau_col, x_col)          # [NT], tau ascending
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(Lr_slices)[:-1]])
        w = jax.lax.stop_gradient(jnp.maximum(jnp.exp(-CAUSAL_EPS * cum), CAUSAL_FLOOR))
        Lr = jnp.mean(w * Lr_slices)
        u0 = u_at_tau(pf, 0.0, ic_x)
        Li = jnp.mean((u0 - ic_u) ** 2)
        return Lr + IC_W * Li, (Lr, Li)

    loss_grad = jit(jax.value_and_grad(loss_fn, has_aux=True))
    optimizer = optax.adam(1e-3)

    @jit
    def step(pf, opt_state, rng, ic_x, ic_u, lr):
        rng, r1, r2 = random.split(rng, 3)
        tau = jnp.sort(random.uniform(r1, (NT, 1), minval=1e-4, maxval=1.0))
        xs = random.uniform(r2, (NX, 1), minval=X_MIN, maxval=X_MAX)
        (loss, aux), g = loss_grad(pf, tau, xs, ic_x, ic_u)
        # scale Adam update by lr via manual scaling of grads through optax? use inject
        updates, opt_state = optimizer.update(g, opt_state)
        updates = jax.tree_util.tree_map(lambda u: (lr / 1e-3) * u, updates)
        pf = optax.apply_updates(pf, updates)
        return pf, opt_state, rng, loss, aux

    @jit
    def endpoint_u(pf, x_col):
        return u_at_tau(pf, 1.0, x_col)     # field at window end (tau=1)

    def sched(frac):
        # frac in [0,1] within the window: warmup 8% then cosine to 5e-5.
        # Time-based so LR fully decays regardless of steps-per-window (any K).
        warm = 0.08
        if frac < warm:
            return LR0 * frac / warm
        p = min(1.0, (frac - warm) / (1.0 - warm))
        return 5e-5 + 0.5 * (LR0 - 5e-5) * (1 + np.cos(np.pi * p))

    # per-window wall budget
    usable = TRAIN_TIME - SAVE_MARGIN_S
    win_wall = usable / K_WIN

    flats = []
    pf = flat0
    opt_state = optimizer.init(pf)
    ic_x = _x_ic_grid
    ic_u = _u_ic0
    total_steps = 0

    for j in range(K_WIN):
        win_start = _T0 + j * win_wall
        win_deadline = win_start + win_wall
        # warm-start from previous window's converged params (keep pf), reset Adam
        opt_state = optimizer.init(pf)
        # compile / first step
        pf, opt_state, rng, loss, aux = step(pf, opt_state, rng, ic_x, ic_u, jnp.float32(1e-6))
        total_steps += 1
        while True:
            now = time.time()
            if now >= win_deadline:
                break
            lr = jnp.float32(sched((now - win_start) / win_wall))
            pf, opt_state, rng, loss, aux = step(pf, opt_state, rng, ic_x, ic_u, lr)
            total_steps += 1
        flats.append(np.asarray(pf))
        # handoff: this window's endpoint (tau=1) becomes next window's IC
        ic_u = jax.lax.stop_gradient(endpoint_u(pf, ic_x))

    stacked = jnp.stack([jnp.asarray(f) for f in flats])   # [K, P]
    template_np = jax.tree_util.tree_map(np.asarray, template)
    return (stacked, template_np), total_steps
