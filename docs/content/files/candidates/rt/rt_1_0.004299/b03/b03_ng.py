"""SOAP champion finished with stochastic Gauss-Newton natural gradient.

The last NG_S seconds solve (J J^T + lam I) a = R, theta -= eta J^T a in the
low-rank residual-space form (kb1-108 B), on a freshly resampled batch every
step (kb1-130). vs b02_soap_bs99: natural-gradient finish.
"""
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


# ══════════ Architecture ══════════
#  * exact 1-periodicity in x: the ONLY x-dependence is through
#    cos/sin(2 pi k x), k = 1..N_KX  (kb1-012)
#  * exact Dirichlet walls: u, v, T are multiplied by
#    phi(y) = tanh(y/DELTA) tanh((2-y)/DELTA), which vanishes at y = 0, 2
#    and is ~1 beyond the physical boundary layer (kb1-036 style factor)
#  * modified MLP with gated residual updates (kb1-050 §B)

N_KX      = 6      # x Fourier harmonics in the input embedding
WIDTH     = 256
DEPTH     = 5      # hidden layers (1 plain + DEPTH-1 gated)
DELTA     = 0.03   # wall-factor thickness (~the thermal boundary layer)


class PINN(nn.Module):
    width: int = WIDTH
    depth: int = DEPTH
    n_kx: int = N_KX
    delta: float = DELTA

    @nn.compact
    def __call__(self, txy):
        t = txy[:, 0:1]
        x = txy[:, 1:2]
        y = txy[:, 2:3]
        tn = 2.0 * (t - T_MIN) / (T_MAX - T_MIN) - 1.0
        yn = 2.0 * (y - Y_MIN) / (Y_MAX - Y_MIN) - 1.0
        k = jnp.arange(1, self.n_kx + 1, dtype=txy.dtype)[None, :]
        ang = 2.0 * jnp.pi * k * (x - X_MIN) / (X_MAX - X_MIN)
        feats = jnp.concatenate([tn, yn, jnp.cos(ang), jnp.sin(ang)], axis=-1)

        kinit = nn.initializers.glorot_normal()
        dense = lambda n, name: nn.Dense(n, kernel_init=kinit, name=name)
        U = nn.tanh(dense(self.width, "Wu")(feats))
        V = nn.tanh(dense(self.width, "Wv")(feats))
        H = nn.tanh(dense(self.width, "W0")(feats))
        for i in range(self.depth - 1):
            Z = nn.tanh(dense(self.width, f"W{i + 1}")(H))
            H = (1.0 - Z) * U + Z * V
        out = dense(OUTPUT_DIM, "Wout")(H)

        phi = jnp.tanh((y - Y_MIN) / self.delta) * jnp.tanh((Y_MAX - y) / self.delta)
        return jnp.concatenate(
            [phi * out[:, 0:1], phi * out[:, 1:2], out[:, 2:3], phi * out[:, 3:4]],
            axis=-1,
        )


# ══════════ Data: IC from file (the only field data) ══════════

def _load_ic():
    ic = np.loadtxt(_locate("rt_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, y_ic = ic[:, 0], ic[:, 1]
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic, y_ic], axis=1).astype(np.float32)
    uvT_ic = ic[:, 2:5].astype(np.float32)
    return jnp.array(pts_ic), jnp.array(uvT_ic)


# ══════════ Training hyperparameters ══════════

N_T_SLICE = 32       # causal time slices spanning [T_MIN, T_MAX]
N_X_SLICE = 64       # collocation points resampled per slice per step
BS_IC     = 1024     # IC points per step
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
SOAP_B1   = 0.9      # first moment (kept in the original basis)
SOAP_B2   = 0.999    # second moment, accumulated in the Kronecker eigenbasis
SOAP_BS   = 0.99     # EMA for the G G^T / G^T G preconditioner factors
SOAP_EPS  = 1e-8
PRECOND_EVERY = 100  # eigh refresh period
# Gauss-Newton natural-gradient finish (kb1-108 B, kb1-063)
NG_S      = min(100.0, 0.2 * TRAIN_TIME)  # wall seconds handed to NG
N_T_NG    = 16      # time slices in the NG batch
N_X_NG    = 24      # collocation points per slice
BS_IC_NG  = 256     # IC points in the NG batch
NG_LAM    = 1e-2    # Levenberg damping, relative to trace(J J^T)/M
NG_M      = 50      # pseudo-time anchor refresh period, in NG steps
NG_ETAS   = (1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3)  # step backtracking


# ══════════ SOAP: Adam in a Kronecker eigenbasis ══════════

class SoapState(NamedTuple):
    count: Any
    mu: Any     # first moment, original basis
    nu: Any     # second moment, rotated basis
    L: Any      # EMA of G G^T
    R: Any      # EMA of G^T G
    QL: Any     # eigenvectors of L
    QR: Any     # eigenvectors of R


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
            if g.ndim != 2:                                   # bias: plain Adam
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

_ET = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
_EX = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
_EY = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)


def _make_train(rng_init):
    pts_ic, uvT_ic = _load_ic()
    n_ic = len(pts_ic)
    t_slices = jnp.linspace(T_MIN, T_MAX, N_T_SLICE, dtype=jnp.float32)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def predict_one(pf, txy):
        return model.apply(unravel(pf), txy[None, :])[0]     # [4]

    def residual_one(pf, txy):
        """Boussinesq residual at one point; all derivatives from autodiff.

        Forward-over-forward jvps give (value, d/dt) and, per axis,
        (d/dx, d2/dx2) — the full Hessian is never formed.
        """
        f = lambda z: predict_one(pf, z)
        val, ft = jax.jvp(f, (txy,), (_ET,))
        fx, fxx = jax.jvp(lambda z: jax.jvp(f, (z,), (_EX,))[1], (txy,), (_EX,))
        fy, fyy = jax.jvp(lambda z: jax.jvp(f, (z,), (_EY,))[1], (txy,), (_EY,))
        u, v, temp = val[0], val[1], val[3]
        r_mx = ft[0] + u * fx[0] + v * fy[0] + fx[2] - A_NU * (fxx[0] + fyy[0])
        r_my = ft[1] + u * fx[1] + v * fy[1] + fy[2] - A_NU * (fxx[1] + fyy[1]) \
            - A_BUOY * temp
        r_c = fx[0] + fy[1]
        r_T = ft[3] + u * fx[3] + v * fy[3] - A_KA * (fxx[3] + fyy[3])
        return jnp.stack([r_mx, r_my, r_c, r_T])

    residual_batch = vmap(residual_one, in_axes=(None, 0))
    predict_batch = vmap(predict_one, in_axes=(None, 0))

    def loss_fn(pf, pf_prev, tau, col, ic_txy, ic_uvT, eps):
        flat_col = col.reshape(-1, INPUT_DIM)
        r = residual_batch(pf, flat_col)
        # pseudo-time anchor: (f - f_prev)/tau added to the matching residual
        f_now = predict_batch(pf, flat_col)
        f_old = jax.lax.stop_gradient(predict_batch(pf_prev, flat_col))
        r = r + (f_now - f_old) / tau
        per_slice = jnp.mean(r.reshape(N_T_SLICE, N_X_SLICE, 4) ** 2, axis=(1, 2))
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(per_slice)[:-1]])
        w = jax.lax.stop_gradient(jnp.exp(-eps * cum))       # causal weights
        pde = jnp.mean(w * per_slice)

        pred = predict_batch(pf, ic_txy)
        ic = jnp.mean((pred[:, 0] - ic_uvT[:, 0]) ** 2
                      + (pred[:, 1] - ic_uvT[:, 1]) ** 2
                      + (pred[:, 3] - ic_uvT[:, 2]) ** 2)
        return pde + LAM_IC * ic, (w, pde, ic)

    loss_and_grad = jit(jax.value_and_grad(loss_fn, has_aux=True))

    @jit
    def bb_tau(pf, pf_prev, col):
        """Barzilai-Borwein pseudo-time step: ||df|| / ||dr|| (kb1-130)."""
        flat_col = col.reshape(-1, INPUT_DIM)
        df = predict_batch(pf, flat_col) - predict_batch(pf_prev, flat_col)
        dr = residual_batch(pf, flat_col) - residual_batch(pf_prev, flat_col)
        return jnp.clip(jnp.linalg.norm(df) / (jnp.linalg.norm(dr) + 1e-8),
                        TAU_LO, TAU_HI)

    sched = optax.warmup_exponential_decay_schedule(
        init_value=LR_PEAK / 10, peak_value=LR_PEAK, warmup_steps=LR_WARM,
        transition_steps=LR_TRANS, decay_rate=LR_DECAY, end_value=1e-5)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), scale_by_soap(),
                            optax.scale_by_schedule(sched), optax.scale(-1.0))
    opt_state = optimizer.init(unravel(flat_params))

    # ---- Gauss-Newton natural gradient (kb1-108 B) ----
    n_col_ng = N_T_NG * N_X_NG
    S_COL = (1.0 / (n_col_ng * 4)) ** 0.5          # so sum(R^2) == the loss
    S_IC = (LAM_IC / (BS_IC_NG * 3)) ** 0.5
    ts_ng = jnp.linspace(T_MIN, T_MAX, N_T_NG, dtype=jnp.float32)

    def res_col_one(pf, txy, pf_prev, tau):
        """PTS-relaxed residual at one point, scaled into the LSQ vector."""
        r = residual_one(pf, txy)
        f_now = predict_one(pf, txy)
        f_old = jax.lax.stop_gradient(predict_one(pf_prev, txy))
        return (r + (f_now - f_old) / tau) * S_COL

    def res_ic_one(pf, txy, tgt):
        p = predict_one(pf, txy)
        return jnp.stack([p[0] - tgt[0], p[1] - tgt[1], p[3] - tgt[2]]) * S_IC

    def res_vec(pf, col, ic_txy, ic_uvT, pf_prev, tau):
        rc = vmap(res_col_one, in_axes=(None, 0, None, None))(
            pf, col, pf_prev, tau)
        ri = vmap(res_ic_one, in_axes=(None, 0, 0))(pf, ic_txy, ic_uvT)
        return jnp.concatenate([rc.reshape(-1), ri.reshape(-1)])

    @jit
    def sample_ng(key):
        k1, k2, k3 = random.split(key, 3)
        tg = jnp.broadcast_to(ts_ng[:, None], (N_T_NG, N_X_NG))
        xs = random.uniform(k1, (N_T_NG, N_X_NG), minval=X_MIN, maxval=X_MAX)
        ys = random.uniform(k2, (N_T_NG, N_X_NG), minval=Y_MIN, maxval=Y_MAX)
        idx = random.choice(k3, n_ic, (BS_IC_NG,))
        col = jnp.stack([tg, xs, ys], axis=-1).reshape(-1, INPUT_DIM)
        return col, pts_ic[idx], uvT_ic[idx]

    @jit
    def ng_step(pf, pf_prev, tau, col, ic_txy, ic_uvT, lam_rel=NG_LAM):
        """One damped Gauss-Newton step in the low-rank (residual-space) form."""
        R = res_vec(pf, col, ic_txy, ic_uvT, pf_prev, tau)
        L0 = jnp.sum(R ** 2)
        Jc = vmap(jax.jacrev(res_col_one, argnums=0),
                  in_axes=(None, 0, None, None))(pf, col, pf_prev, tau)
        Ji = vmap(jax.jacrev(res_ic_one, argnums=0),
                  in_axes=(None, 0, 0))(pf, ic_txy, ic_uvT)
        J = jnp.concatenate([Jc.reshape(-1, pf.shape[0]),
                             Ji.reshape(-1, pf.shape[0])], axis=0)
        M = J.shape[0]
        G = J @ J.T
        lam = lam_rel * jnp.trace(G) / M + 1e-30
        alpha = jnp.linalg.solve(G + lam * jnp.eye(M, dtype=G.dtype), R)
        delta = -(J.T @ alpha)

        def loss_at(e):
            Re = res_vec(pf + e * delta, col, ic_txy, ic_uvT, pf_prev, tau)
            return jnp.sum(Re ** 2)

        etas = jnp.array(NG_ETAS, dtype=pf.dtype)
        Ls = vmap(loss_at)(etas)
        b = jnp.argmin(Ls)
        ok = jnp.isfinite(Ls[b]) & (Ls[b] < L0)
        return (jnp.where(ok, pf + etas[b] * delta, pf), L0,
                jnp.where(ok, Ls[b], L0),
                jnp.where(ok, etas[b], 0.0), jnp.linalg.norm(delta))

    ts_grid = jnp.broadcast_to(t_slices[:, None], (N_T_SLICE, N_X_SLICE))

    @jit
    def sample_col(key):
        k1, k2 = random.split(key)
        xs = random.uniform(k1, (N_T_SLICE, N_X_SLICE), minval=X_MIN, maxval=X_MAX)
        ys = random.uniform(k2, (N_T_SLICE, N_X_SLICE), minval=Y_MIN, maxval=Y_MAX)
        return jnp.stack([ts_grid, xs, ys], axis=-1)

    @jit
    def update(pf, pf_prev, tau, opt_state, key, eps):
        k1, k2 = random.split(key)
        col = sample_col(k1)
        idx = random.choice(k2, n_ic, (BS_IC,))
        (loss, aux), grads = loss_and_grad(pf, pf_prev, tau, col,
                                           pts_ic[idx], uvT_ic[idx], eps)
        # SOAP needs the per-matrix structure, so the flat gradient is
        # unravelled, preconditioned, and re-flattened (ravel is linear).
        updates, opt_state = optimizer.update(unravel(grads), opt_state)
        return pf + jax.flatten_util.ravel_pytree(updates)[0], opt_state, loss, aux[0]

    return (flat_params, opt_state, update, unravel, rng, bb_tau, sample_col,
            sample_ng, ng_step)


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y) -> {"u","v","p","T"} each [BS]."""
    out = _PREDICT_MODEL.apply(params, X)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2], "T": out[:, 3]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam training with a causality ladder (kb1-084).

    Returns (params_pytree, step_count).
    """
    (params_flat, opt_state, update, unravel, rng_local, bb_tau, sample_col,
     sample_ng, ng_step) = _make_train(rng)

    eps_idx = 0
    eps = jnp.float32(EPS_LADDER[0])
    params_prev = params_flat
    tau = jnp.float32(TAU_INIT)

    key, rng_local = random.split(rng_local)
    params_flat, opt_state, loss, w = update(
        params_flat, params_prev, tau, opt_state, key, eps)
    step_count = 1
    last_loss = float(loss)

    # pre-compile the NG step (throwaway) so the switch costs no JIT time
    key, rng_local = random.split(rng_local)
    _c, _i, _u = sample_ng(key)
    _pf, _l0, _l1, _e, _d = ng_step(params_flat, params_flat, tau, _c, _i, _u)
    float(_l1)
    del _pf, _c, _i, _u

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    switch = deadline - NG_S
    for it in range(1, MAX_ITER):
        key, rng_local = random.split(rng_local)
        params_flat, opt_state, loss, w = update(
            params_flat, params_prev, tau, opt_state, key, eps)
        step_count = it + 1
        if it % M_PTS == 0:
            key, rng_local = random.split(rng_local)
            tau_hat = bb_tau(params_flat, params_prev, sample_col(key))
            tau = (1.0 - TAU_BETA) * tau + TAU_BETA * tau_hat
            params_prev = params_flat
        if it % SYNC_EVERY == 0:
            last_loss = float(loss)
            if float(w.min()) > EPS_DELTA and eps_idx < len(EPS_LADDER) - 1:
                eps_idx += 1
                eps = jnp.float32(EPS_LADDER[eps_idx])
            if not np.isfinite(last_loss):
                break
            if time.time() > switch:
                break
            if eval_callback is not None and (it % 1000 == 0):
                eval_callback(unravel(params_flat), it, last_loss)

    # ── Gauss-Newton natural-gradient finish (kb1-108 B) ──
    if np.isfinite(last_loss):
        anchor = params_flat
        ng_it = 0
        while True:
            key, rng_local = random.split(rng_local)
            col, ic_txy, ic_uvT = sample_ng(key)     # fresh batch (kb1-130)
            new_pf, l0, l1, _e, _d = ng_step(params_flat, anchor, tau,
                                             col, ic_txy, ic_uvT)
            if not np.isfinite(float(l1)):
                break
            params_flat = new_pf
            ng_it += 1
            step_count += 1
            last_loss = float(l1)
            if ng_it % NG_M == 0:
                anchor = params_flat
            if time.time() > deadline - 2.0:
                break

    return unravel(params_flat), step_count
