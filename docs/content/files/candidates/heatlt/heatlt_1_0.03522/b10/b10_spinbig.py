"""SPINN as b10_spinn but on the largest lattice the speed diag showed to be
free: NT_PER_BIN 2 -> 4 and NX, NY 24 -> 32, i.e. 131072 collocation points per
step against 36864, at the same ~1030 steps/s (the step is launch-bound, not
flop-bound)."""
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
TR_TAUS  = (0.5, 2.0, 8.0)               # transient exp(-t/tau): -> 0 large t,
                             # so the net is exactly P_SRC-periodic at late time
M_S      = 12            # spatial harmonics sin/cos(m pi x), m = 1..M_S
WIDTH, DEPTH = 64, 3     # per-axis body (was 128/3 for the single joint trunk)
RANK     = 128           # CP rank of the separable trunk
N_SCALE  = 1e-2          # init scale of the CP output (parent: last-layer 1e-2)
TAU_IC   = 0.1           # hard-IC switch-on width
U_BND    = float(np.sqrt(np.pi))   # exact invariant |u| <= sqrt(pi): sin(u^2)=0
                                   # there, so the pointwise ODE cannot cross it
                                   # and the max principle carries it to the PDE
DELTA0   = 0.03          # initial Dirichlet wall-layer width (trainable)
LR_S1, LR_S2 = 1.0, 2.2       # stage-1 / stage-2 step-size amplitude
LR_MAX, LR_MIN = 2e-3 * LR_S1, 3e-4 * LR_S1
LR_R, LR_R_MIN = 8e-4 * LR_S2, 3e-5 * LR_S2   # stage-2 restart
PC_FRAC = 0.45                # stage-2 start (horizon is full by 0.6)
S1E_MULT, S1E_FRAC = 1.65, 0.20   # stage-1 early step lift, ramped to 1.0
S2_HOLD = 0.40                # stage 2 holds LR_R flat for this share
SAVE_MARGIN_S = 6.0

# collocation lattice: Nt = N_BINS * NT_PER_BIN time slabs x NX x NY
NT_PER_BIN = 4
NX, NY     = 32, 32


def _ft(t):
    """t: [N,1] -> [N, 2*K_T + len(TR_TAUS)] — period-8 harmonics + transients."""
    wk = jnp.arange(1, K_T + 1, dtype=t.dtype) * (2.0 * jnp.pi / P_SRC)
    tr = jnp.concatenate([jnp.exp(-t / s) for s in TR_TAUS], axis=-1)
    return jnp.concatenate([jnp.sin(t * wk), jnp.cos(t * wk), tr], axis=-1)


def _fs(x):
    """x: [N,1] -> [N, 2*M_S] — sin/cos(m pi x), m = 1..M_S (all smooth)."""
    ms = jnp.arange(1, M_S + 1, dtype=x.dtype) * jnp.pi
    return jnp.concatenate([jnp.sin(x * ms), jnp.cos(x * ms)], axis=-1)


class Body(nn.Module):
    """One per-axis modified-MLP body: R -> R^RANK (kb1 070 + Wang's mod-MLP)."""
    feat: Callable

    @nn.compact
    def __call__(self, v):
        f = self.feat(v)
        ki = nn.initializers.glorot_normal()
        U = nn.tanh(nn.Dense(WIDTH, kernel_init=ki)(f))
        V = nn.tanh(nn.Dense(WIDTH, kernel_init=ki)(f))
        h = nn.tanh(nn.Dense(WIDTH, kernel_init=ki)(f))
        for _ in range(DEPTH - 1):
            z = nn.tanh(nn.Dense(WIDTH, kernel_init=ki)(h))
            h = (1.0 - z) * U + z * V
        return nn.Dense(RANK, kernel_init=ki)(h)


class SPINN(nn.Module):
    """Three bodies + the two trainable Dirichlet wall widths."""

    @nn.compact
    def __call__(self, tv, xv, yv):
        Ft = Body(_ft, name="bt")(tv)
        Gx = Body(_fs, name="bx")(xv)
        Hy = Body(_fs, name="by")(yv)
        ldx = self.param("log_dx", lambda k: jnp.asarray(np.log(DELTA0), jnp.float32))
        ldy = self.param("log_dy", lambda k: jnp.asarray(np.log(DELTA0), jnp.float32))
        return Ft, Gx, Hy, jnp.exp(ldx), jnp.exp(ldy)


_MODEL = SPINN()
_CPS = N_SCALE / np.sqrt(RANK)


def _wrap(n, t, x, y, dx, dy, grid):
    """Hard ansatz: u = sqrt(pi) tanh(z/sqrt(pi)), z = artanh-image of the IC
    plus s(t) B(x,y) n, so u(0)=u_ic exactly, u=0 on the walls, |u| < sqrt(pi)."""
    bx = jnp.tanh(x / dx) * jnp.tanh((X_MAX - x) / dx)
    by = jnp.tanh(y / dy) * jnp.tanh((Y_MAX - y) / dy)
    s = 1.0 - jnp.exp(-t / TAU_IC)
    if grid:
        uic = (jnp.sin(IC_M1 * jnp.pi * x)[:, None] * jnp.sin(IC_M2 * jnp.pi * y)[None, :])
        sb = s[:, None, None] * (bx[:, None] * by[None, :])[None, :, :]
        z = U_BND * jnp.arctanh(uic / U_BND)[None, :, :] + sb * n
    else:
        uic = jnp.sin(IC_M1 * jnp.pi * x) * jnp.sin(IC_M2 * jnp.pi * y)
        z = U_BND * jnp.arctanh(uic / U_BND) + s * bx * by * n
    return U_BND * jnp.tanh(z / U_BND)


def _u_grid(params, tv, xv, yv):
    """u on the tensor grid, shape [Nt, Nx, Ny]."""
    Ft, Gx, Hy, dx, dy = _MODEL.apply(params, tv, xv, yv)
    n = _CPS * jnp.einsum("ar,br,cr->abc", Ft, Gx, Hy)
    return _wrap(n, tv[:, 0], xv[:, 0], yv[:, 0], dx, dy, True)


def _residual(params, tv, xv, yv):
    """PDE residual by forward-mode autodiff of the separable network on the
    grid: one JVP chain per axis, so second derivatives cost O(Nt+Nx+Ny)."""
    ot, ox, oy = jnp.ones_like(tv), jnp.ones_like(xv), jnp.ones_like(yv)
    u, u_t = jax.jvp(lambda a: _u_grid(params, a, xv, yv), (tv,), (ot,))
    fx = lambda a: jax.jvp(lambda b: _u_grid(params, tv, b, yv), (a,), (ox,))[1]
    fy = lambda a: jax.jvp(lambda b: _u_grid(params, tv, xv, b), (a,), (oy,))[1]
    u_xx = jax.jvp(fx, (xv,), (ox,))[1]
    u_yy = jax.jvp(fy, (yv,), (oy,))[1]
    t, x, y = tv[:, 0], xv[:, 0], yv[:, 0]
    src = (SRC_AMP * jnp.sin(u ** 2)
           * (1.0 + 2.0 * jnp.sin(jnp.pi * t / T_SRC))[:, None, None]
           * jnp.sin(M1 * jnp.pi * x)[None, :, None]
           * jnp.sin(M2 * jnp.pi * y)[None, None, :])
    return u_t - D * (u_xx + u_yy) - src, u


CUR_T0, CUR_FRAC = 0.1, 0.6      # horizon 0.1 -> T_MAX over the first 60%
N_BINS, CAUSAL_M = 32, 3.0       # causal gate: exp(-M * (earlier loss share))

T_WIN, W_FRAC = 32.0, 0.75   # transient + 4 forcing cycles; share drawn there


def _sample(key, ntb, nx, ny, t_hi, pc):
    """Lattice draw.  t is stratified bin-major — N_BINS slices of the horizon
    (stage 1) or of the forcing phase t mod P_SRC (stage 2), NT_PER_BIN jittered
    sub-slices each — and x, y are independently stratified on [0,1]."""
    k1, k2, k3, k4, k5, k6 = random.split(key, 6)
    b = jnp.arange(N_BINS, dtype=jnp.float32)[:, None]
    j = jnp.arange(ntb, dtype=jnp.float32)[None, :]
    xi = random.uniform(k1, (N_BINS, ntb))
    fr = (b + (j + xi) / ntb) / N_BINS

    t_a = fr * t_hi                                   # stage 1: slice on t/t_hi
    p = fr * P_SRC                                    # stage 2: slice on phase
    hi_w = jnp.minimum(t_hi, T_WIN)
    c_w = jnp.floor(random.uniform(k2, (N_BINS, ntb))
                    * (jnp.floor(jnp.maximum(hi_w - p, 0.0) / P_SRC) + 1.0))
    c_f = jnp.floor(random.uniform(k3, (N_BINS, ntb))
                    * (jnp.floor(jnp.maximum(t_hi - p, 0.0) / P_SRC) + 1.0))
    c = jnp.where(random.uniform(k4, (N_BINS, ntb)) < W_FRAC, c_w, c_f)
    t_b = jnp.clip(p + P_SRC * c, T_MIN, t_hi)

    t = jnp.where(pc > 0.5, t_b, t_a).reshape(-1, 1)
    x = ((jnp.arange(nx, dtype=jnp.float32) + random.uniform(k5, (nx,))) / nx)[:, None]
    y = ((jnp.arange(ny, dtype=jnp.float32) + random.uniform(k6, (ny,))) / ny)[:, None]
    return t, x, y


EPS_B = 0.015    # amplitude floor of the stage-2 per-phase normaliser
EMA_FRAC, EMA_W = 0.80, 0.999   # Polyak tail: last 20% of the wall

# --- collapse insurance (b05/b08) ------------------------------------------
# u == 0 satisfies the residual identically (sin(u^2) gates the source) and has
# zero loss AND zero gradient.  The barrier is a one-sided penalty on the batch
# mean u^2 alone, its floor tracking BAR_RHO of the true mean u^2 at the current
# horizon, so it is exactly inactive on any run that keeps a field.
W_BAR, BAR_RHO = 1.0, 0.60
_BAR_TH = np.array((0.1, 0.1413, 0.1995, 0.2818, 0.3981, 0.5623, 0.7943,
                    1.122, 1.585, 2.239, 3.162, 4.467, 6.31, 8.913, 12.59,
                    17.78, 25.12, 35.48, 50.12, 70.79, 100.0))
_BAR_M2 = np.array((0.250, 0.250, 0.250, 0.324, 0.365, 0.434, 0.541,
                    0.684, 0.898, 1.100, 1.100, 1.100, 1.100, 1.100, 1.100,
                    1.100, 1.100, 1.100, 1.100, 1.100, 1.100))
E_B0 = BAR_RHO * _BAR_M2[0]
E_B1 = BAR_RHO * 1.101


def _ef(t_hi):
    """Collapse-barrier floor at horizon t_hi: BAR_RHO of the true mean u^2."""
    return BAR_RHO * float(np.interp(np.log(t_hi), np.log(_BAR_TH), _BAR_M2))


def _loss_fn(params, tv, xv, yv, t_hi, cm, pc, ef):
    """Slices cut by time (stage 1) or by forcing phase and made relative (2)."""
    r, u = _residual(params, tv, xv, yv)
    L = jnp.mean((r ** 2).reshape(N_BINS, -1), axis=1)
    u2 = (u ** 2).reshape(N_BINS, -1)
    ub = jnp.mean(u2, axis=1)
    L = L / ((1.0 - pc) + pc * (jax.lax.stop_gradient(ub) + EPS_B ** 2))
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(L)[:-1]])
    gate = jax.lax.stop_gradient(jnp.exp(-cm * cum / (jnp.sum(L) + 1e-12)))
    bar = W_BAR * jnp.maximum(1.0 - jnp.mean(u2) / ef, 0.0) ** 2
    return jnp.sum(gate * L) / N_BINS + bar


# ══════════ Public contract ══════════

def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u": [BS]}.  The CP form is
    evaluated pointwise, which is the same function as the grid form."""
    t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    Ft, Gx, Hy, dx, dy = _MODEL.apply(params, t, x, y)
    n = _CPS * jnp.sum(Ft * Gx * Hy, axis=-1)
    return {"u": _wrap(n, X[:, 0], X[:, 1], X[:, 2], dx, dy, False)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam on the pure PDE residual (IC and BC are hard).
    Learning rate follows a cosine decay in *wall-clock* progress, since the
    step count is not known in advance.  Returns (params pytree, step_count)."""
    key, rng_local = random.split(rng)
    params = _MODEL.init(key, jnp.zeros([1, 1]), jnp.zeros([1, 1]), jnp.zeros([1, 1]))

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_adam())
    opt_state = opt.init(params)

    def _make_update(ntb, nx, ny):
        @jit
        def update(params, opt_state, ema, key, lr, t_hi, cm, pc, w, ef):
            tv, xv, yv = _sample(key, ntb, nx, ny, t_hi, pc)
            loss, grads = jax.value_and_grad(_loss_fn)(params, tv, xv, yv, t_hi, cm, pc, ef)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = jax.tree_util.tree_map(lambda p, u: p - lr * u, params, updates)
            ema = jax.tree_util.tree_map(lambda e, p: w * e + (1.0 - w) * p, ema, params)
            return params, opt_state, ema, loss
        return update

    update = _make_update(NT_PER_BIN, NX, NY)

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
            t_hi = float(min(T_MAX, CUR_T0 * (T_MAX / CUR_T0) ** min(1.0, frac / CUR_FRAC)))
            ef = E_B1 if frac >= PC_FRAC else _ef(t_hi)
            if frac < PC_FRAC:
                q = frac / PC_FRAC
                lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + np.cos(np.pi * q))
                lr *= 1.0 + (S1E_MULT - 1.0) * max(0.0, 1.0 - frac / S1E_FRAC)
                cm, pc = CAUSAL_M, 0.0
            else:
                q = (frac - PC_FRAC) / (1.0 - PC_FRAC)
                q = max(0.0, (q - S2_HOLD) / (1.0 - S2_HOLD))
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
