"""b06_d3 with a 4-channel transient bank, tau = (0.4, 1.2, 3.5, 10), between
the 3 channels it inherits and the 5 that own the run's lowest PDE error at
DEPTH 4.  b04 calls the bank a resonance rather than a trend (6 and 7 channels
both collapse), and 4 has never been measured at any depth."""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 1
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u",)

# --- Domain ---
T_MIN, T_MAX = 0.0, 100.0
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 1   # r
HAS_SOURCE              = True   # and it is nonlinear in u: sin(u^2) gates it,
                                 # so u ≡ 0 solves the residual exactly

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi")
PERIODIC_AXES  = ()       # none: Dirichlet u = 0 on all four sides
GEOMETRY_KIND  = "box_dirichlet"

# --- Physical parameters ---
# Residual: u_t - D (u_xx + u_yy)
#           - SRC_AMP sin(u^2) (1 + 2 sin(pi t / T_SRC)) sin(M1 pi x) sin(M2 pi y) = 0
D        = 0.001   # diffusivity
SRC_AMP  = 5.0     # source amplitude
T_SRC    = 4.0     # source time factor 1 + 2 sin(pi t / T_SRC) — period 2*T_SRC = 8
M1, M2   = 4.0, 2.0    # source spatial wavenumbers, sin(M1 pi x) sin(M2 pi y)
IC_M1, IC_M2 = 4.0, 3.0  # u(0,x,y) = sin(IC_M1 pi x) sin(IC_M2 pi y) — IC_M2 = 3
                         # differs from the source's M2 = 2; this is not a typo

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 180.0     # total wall-clock seconds for the WHOLE process,
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
import numpy as np
import optax
from flax import linen as nn


# ══════════ Ansatz / feature hyperparameters ══════════

P_SRC    = 2.0 * T_SRC   # forcing period = 8: the solution is an 8-periodic
                         # limit cycle once the initial transient has decayed
K_T      = 10            # harmonics of P_SRC used as the ONLY time input
TR_TAUS  = (0.4, 1.2, 3.5, 10.0)         # transient exp(-t/tau): -> 0 large t,
                             # so the net is exactly P_SRC-periodic at late time
M_S      = 12            # spatial harmonics sin/cos(m pi x), m = 1..M_S
WIDTH, DEPTH = 128, 3
TAU_IC   = 0.1           # hard-IC switch-on width
U_BND    = float(np.sqrt(np.pi))   # exact invariant |u| <= sqrt(pi): sin(u^2)=0
                                   # there, so the pointwise ODE cannot cross it
                                   # and the max principle carries it to the PDE
DELTA0   = 0.03          # initial Dirichlet wall-layer width (trainable)
LR_MAX, LR_MIN = 2e-3, 3e-4
LR_R, LR_R_MIN = 8e-4, 3e-5   # stage-2 restart
PC_FRAC = 0.45                # stage-2 start (horizon is full by 0.6)
BS_COL   = 3072
SAVE_MARGIN_S = 6.0


def _feats(txy):
    """(t, x, y) -> [2*K_T + len(TR_TAUS) + 4*M_S] embedding."""
    t = txy[..., 0:1]
    x = txy[..., 1:2]
    y = txy[..., 2:3]
    wk = jnp.arange(1, K_T + 1, dtype=txy.dtype) * (2.0 * jnp.pi / P_SRC)
    ms = jnp.arange(1, M_S + 1, dtype=txy.dtype) * jnp.pi
    tr = jnp.concatenate([jnp.exp(-t / s) for s in TR_TAUS], axis=-1)
    g = jnp.sin(M1 * jnp.pi * x) * jnp.sin(M2 * jnp.pi * y)      # source shape
    ui = jnp.sin(IC_M1 * jnp.pi * x) * jnp.sin(IC_M2 * jnp.pi * y)
    return jnp.concatenate([jnp.sin(t * wk), jnp.cos(t * wk), tr,
                            tr * ui, tr * g,          # IC memory, resolved in space
                            jnp.sin(x * ms), jnp.cos(x * ms),
                            jnp.sin(y * ms), jnp.cos(y * ms),
                            g, g * g, ui, g * ui], axis=-1)      # all smooth


class PINN(nn.Module):
    """Modified-MLP trunk (two gating encoders U, V) on the embedding, wrapped
    in a hard ansatz: u = u_ic(x,y) + s(t) B(x,y) N, with s(0)=0 (exact IC)
    and B = 0 on the four walls (exact Dirichlet)."""
    width: int = WIDTH
    depth: int = DEPTH

    @nn.compact
    def __call__(self, txy):
        f = _feats(txy)
        ki = nn.initializers.glorot_normal()
        U = nn.tanh(nn.Dense(self.width, kernel_init=ki)(f))
        V = nn.tanh(nn.Dense(self.width, kernel_init=ki)(f))
        h = nn.tanh(nn.Dense(self.width, kernel_init=ki)(f))
        for _ in range(self.depth - 1):
            z = nn.tanh(nn.Dense(self.width, kernel_init=ki)(h))
            h = (1.0 - z) * U + z * V
        n = nn.Dense(OUTPUT_DIM, kernel_init=nn.initializers.normal(1e-2))(h)[..., 0]

        ldx = self.param("log_dx", lambda k: jnp.asarray(np.log(DELTA0), jnp.float32))
        ldy = self.param("log_dy", lambda k: jnp.asarray(np.log(DELTA0), jnp.float32))
        dx, dy = jnp.exp(ldx), jnp.exp(ldy)

        t, x, y = txy[..., 0], txy[..., 1], txy[..., 2]
        bw = (jnp.tanh(x / dx) * jnp.tanh((X_MAX - x) / dx)
              * jnp.tanh(y / dy) * jnp.tanh((Y_MAX - y) / dy))
        s = 1.0 - jnp.exp(-t / TAU_IC)
        u_ic = jnp.sin(IC_M1 * jnp.pi * x) * jnp.sin(IC_M2 * jnp.pi * y)
        z = U_BND * jnp.arctanh(u_ic / U_BND) + s * bw * n   # |u_ic| <= 1 < U_BND
        return U_BND * jnp.tanh(z / U_BND)                   # g'(0) = 1, |u| < U_BND


_MODEL = PINN()


def _residual(params, X):
    """PDE residual by forward-mode autodiff of the network; returns (r, u)."""
    f = lambda Z: _MODEL.apply(params, Z)
    Et = jnp.zeros_like(X).at[:, 0].set(1.0)
    Ex = jnp.zeros_like(X).at[:, 1].set(1.0)
    Ey = jnp.zeros_like(X).at[:, 2].set(1.0)
    u, u_t = jax.jvp(f, (X,), (Et,))
    u_xx = jax.jvp(lambda Z: jax.jvp(f, (Z,), (Ex,))[1], (X,), (Ex,))[1]
    u_yy = jax.jvp(lambda Z: jax.jvp(f, (Z,), (Ey,))[1], (X,), (Ey,))[1]
    t, x, y = X[:, 0], X[:, 1], X[:, 2]
    src = (SRC_AMP * jnp.sin(u ** 2)
           * (1.0 + 2.0 * jnp.sin(jnp.pi * t / T_SRC))
           * jnp.sin(M1 * jnp.pi * x) * jnp.sin(M2 * jnp.pi * y))
    return u_t - D * (u_xx + u_yy) - src, u


CUR_T0, CUR_FRAC = 0.1, 0.6      # horizon 0.1 -> T_MAX over the first 60%
N_BINS, CAUSAL_M = 32, 3.0       # causal gate: exp(-M * (earlier loss share))


T_WIN, W_FRAC = 32.0, 0.75   # transient + 4 forcing cycles; share drawn there


M_PER_BIN = BS_COL // N_BINS   # points per loss bin — exact, not in expectation


def _sample(key, n, t_hi, pc):
    """Bin-stratified batch.  Stage 1 stratifies t/t_hi, stage 2 the forcing
    phase t mod P_SRC with the cycle index drawn under the parent's W_FRAC
    concentration on t < T_WIN; (x, y) is a Latin hypercube inside each bin."""
    m = n // N_BINS
    k1, k2, k3, k4, k5, k6 = random.split(key, 6)
    b = jnp.arange(N_BINS, dtype=jnp.float32)[:, None]
    xi = random.uniform(k1, (N_BINS, m))

    t_a = (b + xi) / N_BINS * t_hi                    # stage 1: slice on t/t_hi
    p = (b + xi) / N_BINS * P_SRC                     # stage 2: slice on phase
    hi_w = jnp.minimum(t_hi, T_WIN)
    c_w = jnp.floor(random.uniform(k2, (N_BINS, m))
                    * (jnp.floor(jnp.maximum(hi_w - p, 0.0) / P_SRC) + 1.0))
    c_f = jnp.floor(random.uniform(k3, (N_BINS, m))
                    * (jnp.floor(jnp.maximum(t_hi - p, 0.0) / P_SRC) + 1.0))
    c = jnp.where(random.uniform(k4, (N_BINS, m)) < W_FRAC, c_w, c_f)
    t_b = jnp.clip(p + P_SRC * c, T_MIN, t_hi)

    t = jnp.where(pc > 0.5, t_b, t_a).reshape(-1)
    ox = jnp.argsort(random.uniform(k5, (N_BINS, m)), axis=1)
    oy = jnp.argsort(random.uniform(k6, (N_BINS, m)), axis=1)
    jx = random.uniform(random.fold_in(k5, 1), (N_BINS, m))
    jy = random.uniform(random.fold_in(k6, 2), (N_BINS, m))
    x = ((ox + jx) / m).reshape(-1)
    y = ((oy + jy) / m).reshape(-1)
    return jnp.stack([t, x, y], axis=1)


EPS_B = 0.015    # amplitude floor of the stage-2 per-phase normaliser
EMA_FRAC, EMA_W = 0.80, 0.999   # Polyak tail: last 20% of the wall, ~1000-step window

# --- collapse insurance ---------------------------------------------------
# u == 0 satisfies the residual identically (sin(u^2) gates the source), has
# zero loss AND zero gradient, and every failed re-aiming of the loss in
# b02-b04 ended there.  The barrier is a one-sided penalty on the batch mean
# u^2 alone: floor E_B0 while the collocation horizon is still short (the
# field is then ~ the IC, mean u^2 = 0.25), rising to E_B1 once the horizon is
# full.  The saturation invariant |u| -> sqrt(pi) puts the true mean u^2 at
# O(1), so the barrier is *exactly* inactive - value and gradient identically
# zero - on any run that keeps a field, and costs one mean over the batch.
E_B0, E_B1, W_BAR = 0.15, 0.35, 1.0


def _loss_fn(params, X, t_hi, cm, pc, ef):
    """Slices cut by time (stage 1) or by forcing phase and made relative (2)."""
    r, u = _residual(params, X)
    r2 = r ** 2
    z = (1.0 - pc) * (X[:, 0] / t_hi) + pc * (jnp.mod(X[:, 0], P_SRC) / P_SRC)
    b = jnp.clip((z * N_BINS).astype(jnp.int32), 0, N_BINS - 1)
    tot = jax.ops.segment_sum(r2, b, N_BINS)
    cnt = jax.ops.segment_sum(jnp.ones_like(r2), b, N_BINS)
    ub = jax.ops.segment_sum(u ** 2, b, N_BINS) / jnp.maximum(cnt, 1.0)
    L = tot / jnp.maximum(cnt, 1.0)
    L = L / ((1.0 - pc) + pc * (jax.lax.stop_gradient(ub) + EPS_B ** 2))
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(L)[:-1]])
    gate = jax.lax.stop_gradient(jnp.exp(-cm * cum / (jnp.sum(L) + 1e-12)))
    bar = W_BAR * jnp.maximum(1.0 - jnp.mean(u ** 2) / ef, 0.0) ** 2
    return jnp.sum(gate * L) / N_BINS + bar


# ══════════ Public contract ══════════

def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u": [BS]}."""
    return {"u": _MODEL.apply(params, X)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam on the pure PDE residual (IC and BC are hard).
    Learning rate follows a cosine decay in *wall-clock* progress, since the
    step count is not known in advance.  Returns (params pytree, step_count)."""
    key, rng_local = random.split(rng)
    params = _MODEL.init(key, jnp.zeros([1, INPUT_DIM]))

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_adam())
    opt_state = opt.init(params)

    @jit
    def update(params, opt_state, ema, key, lr, t_hi, cm, pc, w, ef):
        X = _sample(key, BS_COL, t_hi, pc)
        loss, grads = jax.value_and_grad(_loss_fn)(params, X, t_hi, cm, pc, ef)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p - lr * u, params, updates)
        ema = jax.tree_util.tree_map(lambda e, p: w * e + (1.0 - w) * p, ema, params)
        return params, opt_state, ema, loss

    t_start = time.time()
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S

    # warmup step: JIT-compiles the whole training step (a real, counted step)
    rng_local, k = random.split(rng_local)
    ema = params
    params, opt_state, ema, loss = update(params, opt_state, ema, k, LR_MAX,
                                          CUR_T0, CAUSAL_M, 0.0, 0.0, E_B0)
    last_loss = float(loss)
    step_count = 1
    t_ready = time.time()
    span = max(deadline - t_ready, 1.0)

    lr, t_hi, cm, pc, w, ef = LR_MAX, CUR_T0, CAUSAL_M, 0.0, 0.0, E_B0
    for it in range(1, MAX_ITER):
        now = time.time()
        if now > deadline:
            break
        if it % 50 == 0:
            frac = min(max((now - t_ready) / span, 0.0), 1.0)
            w = EMA_W if frac > EMA_FRAC else 0.0
            ef = E_B0 + (E_B1 - E_B0) * min(1.0, frac / CUR_FRAC)
            t_hi = float(min(T_MAX, CUR_T0 * (T_MAX / CUR_T0) ** min(1.0, frac / CUR_FRAC)))
            if frac < PC_FRAC:
                q = frac / PC_FRAC
                lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + np.cos(np.pi * q))
                cm, pc = CAUSAL_M, 0.0
            else:
                q = (frac - PC_FRAC) / (1.0 - PC_FRAC)
                lr = LR_R_MIN + 0.5 * (LR_R - LR_R_MIN) * (1.0 + np.cos(np.pi * q))
                cm, pc = 0.0, 1.0
        rng_local, k = random.split(rng_local)
        params, opt_state, ema, loss = update(params, opt_state, ema, k, lr,
                                              t_hi, cm, pc, w, ef)
        step_count = it + 1
        if it % 500 == 0:
            last_loss = float(loss)
            if eval_callback is not None:
                eval_callback(ema, it, last_loss)
    return ema, step_count
