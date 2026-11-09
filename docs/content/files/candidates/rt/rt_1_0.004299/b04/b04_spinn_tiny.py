"""Separable PINN on a 3072-point lattice — the champion's own batch size.

vs b04_spinn_nz: 16x24 -> 8x12 per step. b04_spinn went 0.462 (147k pts) ->
0.248 (12k pts); this is the third point on that axis, at the exact collocation
count the dense-MLP champion uses, isolating ansatz from batch size.
"""
from __future__ import annotations

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 4
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u", "v", "p", "T")

# --- Domain ---
T_MIN, T_MAX = 0.0, 4.75
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 2.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 4   # ru, rv, continuity, rT
HAS_SOURCE              = False  # buoyancy couples T into the y-momentum;
                                 # there is no externally prescribed forcing

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi")
PERIODIC_AXES  = ("x",)   # u(t, X_MIN, y) = u(t, X_MAX, y); the y walls are
                          # Dirichlet u = v = T = 0 (pressure is gauge-free)
GEOMETRY_KIND  = "periodic_channel"

# --- Physical parameters ---
PR     = 0.71                    # Prandtl number
RA     = 1.0e6                   # Rayleigh number
A_NU   = (PR / RA) ** 0.5        # sqrt(Pr/Ra)   = 8.426150e-04, viscous
A_KA   = 1.0 / (PR * RA) ** 0.5  # 1/sqrt(Pr Ra) = 1.186782e-03, thermal
A_BUOY = 1.0                     # buoyancy: +T on the y-momentum

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 480.0     # total wall-clock seconds for the WHOLE process,
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
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Separable architecture (kb1-070) ══════════
#  f_c(t,x,y) = OUT_SCALE * sum_r Bt[t,c,r] Bx[x,c,r] By[y,c,r]
#  * exact 1-periodicity in x: the x body sees only cos/sin(2 pi k x) (kb1-012)
#  * exact Dirichlet walls: the y body's (u,v,T) rows carry
#    phi(y) = tanh(y/DELTA) tanh((2-y)/DELTA), which factors out of the CP sum
#  * each per-axis body is a modified MLP with gated residual updates (kb1-050 B)

N_KX    = 6      # x Fourier harmonics in the x-body embedding
WIDTH   = 128
DEPTH   = 4      # hidden layers per body (1 plain + DEPTH-1 gated)
RANK    = 96     # CP rank, per output channel
DELTA   = 0.03   # wall-factor thickness (~the thermal boundary layer)
OUT_SCALE = RANK ** -0.5


class Body(nn.Module):
    """Per-axis modified MLP: features -> [N, OUTPUT_DIM, RANK]."""
    width: int = WIDTH
    depth: int = DEPTH
    rank: int = RANK

    @nn.compact
    def __call__(self, feats):
        kinit = nn.initializers.glorot_normal()
        dense = lambda n, name: nn.Dense(n, kernel_init=kinit, name=name)
        U = nn.tanh(dense(self.width, "Wu")(feats))
        V = nn.tanh(dense(self.width, "Wv")(feats))
        H = nn.tanh(dense(self.width, "W0")(feats))
        for i in range(self.depth - 1):
            Z = nn.tanh(dense(self.width, f"W{i + 1}")(H))
            H = (1.0 - Z) * U + Z * V
        out = dense(OUTPUT_DIM * self.rank, "Wout")(H)
        return out.reshape(out.shape[0], OUTPUT_DIM, self.rank)


class SPINN(nn.Module):
    """Three axis bodies; every public method returns axis factors [N, 4, RANK]."""

    def setup(self):
        self.bt = Body(name="bt")
        self.bx = Body(name="bx")
        self.by = Body(name="by")

    def tb(self, t):                                   # t: [Nt, 1]
        tn = 2.0 * (t - T_MIN) / (T_MAX - T_MIN) - 1.0
        return self.bt(tn)

    def xb(self, x):                                   # x: [Nx, 1]
        k = jnp.arange(1, N_KX + 1, dtype=x.dtype)[None, :]
        ang = 2.0 * jnp.pi * k * (x - X_MIN) / (X_MAX - X_MIN)
        return self.bx(jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=-1))

    def yb(self, y):                                   # y: [Ny, 1]
        yn = 2.0 * (y - Y_MIN) / (Y_MAX - Y_MIN) - 1.0
        out = self.by(yn)
        phi = jnp.tanh((y - Y_MIN) / DELTA) * jnp.tanh((Y_MAX - y) / DELTA)  # [Ny,1]
        mask = jnp.stack([phi[:, 0], phi[:, 0], jnp.ones_like(phi[:, 0]),
                          phi[:, 0]], axis=1)          # p is gauge-free
        return out * mask[:, :, None]

    def __call__(self, txy):                           # [N, 3] -> [N, 4]
        Bt = self.tb(txy[:, 0:1])
        Bx = self.xb(txy[:, 1:2])
        By = self.yb(txy[:, 2:3])
        return OUT_SCALE * jnp.einsum("ncr,ncr,ncr->nc", Bt, Bx, By)


# ══════════ Data: IC from file (the only field data), on its own lattice ══════════

def _load_ic():
    """IC on the 100 x 200 tensor grid: x values, y values, and (u,v,T) [3,100,200]."""
    ic = np.loadtxt(_locate("rt_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    xs = np.unique(ic[:, 0])
    ys = np.unique(ic[:, 1])
    ix = np.searchsorted(xs, ic[:, 0])
    iy = np.searchsorted(ys, ic[:, 1])
    fields = np.zeros((3, xs.size, ys.size), dtype=np.float32)
    for c in range(3):
        fields[c, ix, iy] = ic[:, 2 + c]
    return (jnp.array(xs[:, None].astype(np.float32)),
            jnp.array(ys[:, None].astype(np.float32)),
            jnp.array(fields))


# ══════════ Training hyperparameters ══════════

N_T = 32        # causal time slices spanning [T_MIN, T_MAX]
N_X = 8         # x samples per step (resampled)
N_Y = 12        # y samples per step (resampled)
LAM_IC    = 100.0    # kb1-084 recommends 1e2 - 1e4
EPS_LADDER = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)  # causality ladder (kb1-084)
EPS_DELTA  = 0.99    # promote eps once min_i w_i exceeds this
LR_PEAK   = 3e-3
LR_WARM   = 1000
LR_TRANS  = 3000
LR_DECAY  = 0.9
SYNC_EVERY = 100     # host sync / schedule check cadence
SAVE_MARGIN_S = 8.0
# pseudo-time stepping (kb1-130)
M_PTS     = 1000     # snapshot / tau refresh period
TAU_INIT  = 1.0
TAU_BETA  = 0.5      # EMA on the Barzilai-Borwein step estimate
TAU_LO, TAU_HI = 0.1, 10.0
# SOAP (kb1-108 C, kb1-130's optimiser)
SOAP_B1   = 0.9
SOAP_B2   = 0.999
SOAP_BS   = 0.99
SOAP_EPS  = 1e-8
PRECOND_EVERY = 100


# ══════════ SOAP: Adam in a Kronecker eigenbasis ══════════

class SoapState(NamedTuple):
    count: Any
    mu: Any
    nu: Any
    L: Any
    R: Any
    QL: Any
    QR: Any


def scale_by_soap():
    """optax transform: per-matrix Adam in the eigenbasis of the Kronecker
    curvature factors. Non-matrix leaves (biases) fall back to plain Adam."""

    def init(params):
        z = jax.tree_util.tree_map(jnp.zeros_like, params)
        def fac(p, axis):
            n = p.shape[axis] if p.ndim == 2 else 1
            return jnp.zeros((n, n), p.dtype)
        def eye(p, axis):
            n = p.shape[axis] if p.ndim == 2 else 1
            return jnp.eye(n, dtype=p.dtype)
        tm = jax.tree_util.tree_map
        return SoapState(
            count=jnp.zeros([], jnp.int32), mu=z, nu=z,
            L=tm(lambda p: fac(p, 0), params), R=tm(lambda p: fac(p, 1), params),
            QL=tm(lambda p: eye(p, 0), params), QR=tm(lambda p: eye(p, 1), params))

    def update(grads, state, params=None):
        del params
        count = state.count + 1
        refresh = (count % PRECOND_EVERY) == 0
        b1c = 1.0 - SOAP_B1 ** count.astype(jnp.float32)
        b2c = 1.0 - SOAP_B2 ** count.astype(jnp.float32)

        def per_leaf(g, mu, nu, L, R, QL, QR):
            mu = SOAP_B1 * mu + (1.0 - SOAP_B1) * g
            if g.ndim != 2:
                nu = SOAP_B2 * nu + (1.0 - SOAP_B2) * g ** 2
                u = (mu / b1c) / (jnp.sqrt(nu / b2c) + SOAP_EPS)
                return u, mu, nu, L, R, QL, QR
            L = SOAP_BS * L + (1.0 - SOAP_BS) * (g @ g.T)
            R = SOAP_BS * R + (1.0 - SOAP_BS) * (g.T @ g)
            eig = lambda M: jnp.linalg.eigh(
                M + 1e-6 * jnp.trace(M) / M.shape[0] * jnp.eye(M.shape[0], dtype=M.dtype))[1]
            QL = jax.lax.cond(refresh, lambda: eig(L), lambda: QL)
            QR = jax.lax.cond(refresh, lambda: eig(R), lambda: QR)
            gr = QL.T @ g @ QR
            nu = SOAP_B2 * nu + (1.0 - SOAP_B2) * gr ** 2
            mr = QL.T @ (mu / b1c) @ QR
            u = QL @ (mr / (jnp.sqrt(nu / b2c) + SOAP_EPS)) @ QR.T
            return u, mu, nu, L, R, QL, QR

        out = jax.tree_util.tree_map(per_leaf, grads, state.mu, state.nu,
                                     state.L, state.R, state.QL, state.QR)
        pick = lambda i: jax.tree_util.tree_map(
            lambda t: t[i], out, is_leaf=lambda t: isinstance(t, tuple))
        return pick(0), SoapState(count=count, mu=pick(1), nu=pick(2),
                                  L=pick(3), R=pick(4), QL=pick(5), QR=pick(6))

    return optax.GradientTransformation(init, update)


# ══════════ Loss: PDE residual via autodiff + IC L2 (no BC terms) ══════════

_EINS = "tcr,xcr,ycr->ctxy"


def _make_train(rng_init):
    x_ic, y_ic, uvT_ic = _load_ic()
    t_slices = jnp.linspace(T_MIN, T_MAX, N_T, dtype=jnp.float32)[:, None]
    t_zero = jnp.zeros((1, 1), jnp.float32)

    model = SPINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    tb = lambda pf, t: model.apply(unravel(pf), t, method=SPINN.tb)
    xb = lambda pf, x: model.apply(unravel(pf), x, method=SPINN.xb)
    yb = lambda pf, y: model.apply(unravel(pf), y, method=SPINN.yb)

    def _d1(fn, z):
        """(value, d/dz) of an axis body, forward mode."""
        return jax.jvp(fn, (z,), (jnp.ones_like(z),))

    def _d12(fn, z):
        """(value, d/dz, d2/dz2) of an axis body, forward-over-forward."""
        v = fn(z)
        d1, d2 = jax.jvp(lambda w: jax.jvp(fn, (w,), (jnp.ones_like(w),))[1],
                         (z,), (jnp.ones_like(z),))
        return v, d1, d2

    def field(pf, t, x, y):
        """Field only: [4, Nt, Nx, Ny]."""
        return OUT_SCALE * jnp.einsum(_EINS, tb(pf, t), xb(pf, x), yb(pf, y))

    def residual(pf, t, x, y):
        """Boussinesq residual on the lattice; all derivatives from autodiff.

        Only one axis body depends on each coordinate, so d/dt, d/dx, d2/dx2,
        d/dy and d2/dy2 each cost one jvp chain on a 1-D network (kb1-070).
        """
        Bt, dBt = _d1(lambda z: tb(pf, z), t)
        Bx, dBx, ddBx = _d12(lambda z: xb(pf, z), x)
        By, dBy, ddBy = _d12(lambda z: yb(pf, z), y)
        E = lambda A, B, C: OUT_SCALE * jnp.einsum(_EINS, A, B, C)
        f = E(Bt, Bx, By)
        ft = E(dBt, Bx, By)
        fx = E(Bt, dBx, By)
        fxx = E(Bt, ddBx, By)
        fy = E(Bt, Bx, dBy)
        fyy = E(Bt, Bx, ddBy)
        u, v, temp = f[0], f[1], f[3]
        r_mx = ft[0] + u * fx[0] + v * fy[0] + fx[2] - A_NU * (fxx[0] + fyy[0])
        r_my = ft[1] + u * fx[1] + v * fy[1] + fy[2] - A_NU * (fxx[1] + fyy[1]) \
            - A_BUOY * temp
        r_c = fx[0] + fy[1]
        r_T = ft[3] + u * fx[3] + v * fy[3] - A_KA * (fxx[3] + fyy[3])
        return jnp.stack([r_mx, r_my, r_c, r_T]), f

    def loss_fn(pf, pf_prev, tau, x, y, eps):
        r, f = residual(pf, t_slices, x, y)
        # pseudo-time anchor: (f - f_prev)/tau added to the matching residual
        f_old = jax.lax.stop_gradient(field(pf_prev, t_slices, x, y))
        r = r + (f - f_old) / tau
        per_slice = jnp.mean(r ** 2, axis=(0, 2, 3))              # [N_T]
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(per_slice)[:-1]])
        w = jax.lax.stop_gradient(jnp.exp(-eps * cum))            # causal weights
        pde = jnp.mean(w * per_slice)

        pred = field(pf, t_zero, x_ic, y_ic)[:, 0]                # [4, 100, 200]
        ic = jnp.mean((pred[0] - uvT_ic[0]) ** 2 + (pred[1] - uvT_ic[1]) ** 2
                      + (pred[3] - uvT_ic[2]) ** 2)
        return pde + LAM_IC * ic, (w, pde, ic)

    loss_and_grad = jit(jax.value_and_grad(loss_fn, has_aux=True))

    @jit
    def bb_tau(pf, pf_prev, x, y):
        """Barzilai-Borwein pseudo-time step: ||df|| / ||dr|| (kb1-130)."""
        r_now, f_now = residual(pf, t_slices, x, y)
        r_old, f_old = residual(pf_prev, t_slices, x, y)
        return jnp.clip(jnp.linalg.norm(f_now - f_old)
                        / (jnp.linalg.norm(r_now - r_old) + 1e-8), TAU_LO, TAU_HI)

    sched = optax.warmup_exponential_decay_schedule(
        init_value=LR_PEAK / 10, peak_value=LR_PEAK, warmup_steps=LR_WARM,
        transition_steps=LR_TRANS, decay_rate=LR_DECAY, end_value=1e-5)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), scale_by_soap(),
                            optax.scale_by_schedule(sched), optax.scale(-1.0))
    opt_state = optimizer.init(unravel(flat_params))

    @jit
    def sample_xy(key):
        k1, k2 = random.split(key)
        x = random.uniform(k1, (N_X, 1), minval=X_MIN, maxval=X_MAX)
        y = random.uniform(k2, (N_Y, 1), minval=Y_MIN, maxval=Y_MAX)
        return x, y

    @jit
    def update(pf, pf_prev, tau, opt_state, key, eps):
        x, y = sample_xy(key)
        (loss, aux), grads = loss_and_grad(pf, pf_prev, tau, x, y, eps)
        updates, opt_state = optimizer.update(unravel(grads), opt_state)
        return pf + jax.flatten_util.ravel_pytree(updates)[0], opt_state, loss, aux[0]

    return flat_params, opt_state, update, unravel, rng, bb_tau, sample_xy


# ══════════ Public contract ══════════

_PREDICT_MODEL = SPINN()


def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y) -> {"u","v","p","T"} each [BS]."""
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2], "T": out[:, 3]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline SOAP training with a causality ladder (kb1-084).

    Returns (params_pytree, step_count).
    """
    params_flat, opt_state, update, unravel, rng_local, bb_tau, sample_xy = \
        _make_train(rng)

    eps_idx = 0
    eps = jnp.float32(EPS_LADDER[0])
    params_prev = params_flat
    tau = jnp.float32(TAU_INIT)

    key, rng_local = random.split(rng_local)
    params_flat, opt_state, loss, w = update(
        params_flat, params_prev, tau, opt_state, key, eps)
    step_count = 1
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        key, rng_local = random.split(rng_local)
        params_flat, opt_state, loss, w = update(
            params_flat, params_prev, tau, opt_state, key, eps)
        step_count = it + 1
        if it % M_PTS == 0:
            key, rng_local = random.split(rng_local)
            xq, yq = sample_xy(key)
            tau_hat = bb_tau(params_flat, params_prev, xq, yq)
            tau = (1.0 - TAU_BETA) * tau + TAU_BETA * tau_hat
            params_prev = params_flat
        if it % SYNC_EVERY == 0:
            last_loss = float(loss)
            if float(w.min()) > EPS_DELTA and eps_idx < len(EPS_LADDER) - 1:
                eps_idx += 1
                eps = jnp.float32(EPS_LADDER[eps_idx])
            if not np.isfinite(last_loss):
                break
            if time.time() > deadline:
                break
            if eval_callback is not None and (it % 1000 == 0):
                eval_callback(unravel(params_flat), it, last_loss)

    return unravel(params_flat), step_count
