"""Hard-BC marched SOAP PINN with a 2.5x stronger handoff term.
Changed vs b02_phi01.py: W_IC 100 -> 250. The alpha-probes put b02_phi01's wake
amplitude at 0.67 of truth, i.e. fluctuation is being lost at the 11 handoffs;
with the boundary losses gone the IC term is the only data term left to raise.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 3
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u", "v", "p")

# --- Domain ---
T_MIN, T_MAX = 0.0, 1.2
X_MIN, X_MAX = 0.0, 2.2
Y_MIN, Y_MAX = 0.0, 0.41
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 3   # ru, rv, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("inlet", "outlet", "walls", "cylinder")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "channel_with_cylinder"

# --- Geometry (DFG 2D benchmark, Schaefer & Turek) ---
CYL_X, CYL_Y = 0.2, 0.2      # cylinder centre; offset from the channel
CYL_R        = 0.05          # mid-height 0.205 to break the symmetry

# --- Physical parameters (DFG 2D-3, Schaefer & Turek: driven inflow) ---
U_PEAK  = 1.5        # peak centreline inflow; U(s) = U_PEAK*sin(pi*s/T_DRIVE)
T_DRIVE = 8.0        # driving period of the benchmark, in simulation time s
T_START = 4.8        # task t = 0 is benchmark time s = T_START
NU      = 1.0e-3     # kinematic viscosity (rho = 1)
RE      = (2.0 / 3.0) * U_PEAK * (2.0 * CYL_R) / NU   # = 100 at the drive peak
                     # the window itself runs at Re(t) = 95 -> 71

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
import numpy as np
import optax
from flax import linen as nn
from soap_jax import soap


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


def inlet_centreline(t):
    return U_PEAK * jnp.sin(jnp.pi * (t + T_START) / T_DRIVE)


def inlet_profile(t, y):
    return 4.0 * inlet_centreline(t) * y * (Y_MAX - y) / (Y_MAX ** 2)


# ══════════ Architecture ══════════

_LOX = jnp.array([X_MIN, Y_MIN])
_SCX = jnp.array([2.0 / (X_MAX - X_MIN), 2.0 / (Y_MAX - Y_MIN)])

WIDTH, DEPTH = 128, 6
N_SEG = 12
SEG = [T_MIN + k * (T_MAX - T_MIN) / N_SEG for k in range(N_SEG + 1)]

_M_FF = 96
_SIGMA_FF = np.array([2.0, 5.0, 3.0], dtype=np.float32)
_B_FF = jnp.array((np.random.default_rng(12345).standard_normal((_M_FF, 3))
                   * _SIGMA_FF).astype(np.float32))


# --- ADF hard-BC machinery (kb1 036) ---------------------------------
# omega_i >= 0, zero exactly on boundary i, |grad omega_i| = 1 there:
#   bottom wall y = 0, top wall y = H, cylinder r = R, inlet x = 0.
# The outlet is deliberately absent: u, v are free there.
PHI_ELL = 0.01         # saturation length of the boundary factor


def _omegas(txy):
    x, y = txy[:, 1], txy[:, 2]
    r = jnp.sqrt((x - CYL_X) ** 2 + (y - CYL_Y) ** 2 + 1e-9)
    return (y, Y_MAX - y, jnp.maximum(r - CYL_R, 0.0), x)


def _rconj(a, b):
    """R-conjunction (smooth min): 0 iff a or b is 0, positive otherwise."""
    return a + b - jnp.sqrt(a * a + b * b + 1e-12)


def _phi(txy):
    wb, wt, wc, wi = _omegas(txy)
    return jnp.tanh(_rconj(_rconj(wb, wt), _rconj(wc, wi)) / PHI_ELL)


def _lift_u(txy):
    """Transfinite (Rvachev) interpolant of the Dirichlet data for u: equals
    inlet_profile on x = 0 and 0 on the walls and the cylinder."""
    wb, wt, wc, wi = _omegas(txy)
    n_in = wb * wt * wc
    den = n_in + wi * wt * wc + wi * wb * wc + wi * wb * wt
    return n_in / (den + 1e-12) * inlet_profile(txy[:, 0], txy[:, 2])


class PINN(nn.Module):
    """Gated modified-MLP (kb1 050) on a fixed random Fourier embedding
    (kb1 020). Time is normalised inside the *current* segment [t_lo, t_hi],
    passed as traced scalars so all segments share one compiled kernel.
    The raw output is wrapped in the ADF trial u = g + phi*u_hat, so the
    inlet profile and the no-slip walls/cylinder hold exactly (kb1 036)."""
    width: int = WIDTH
    depth: int = DEPTH

    @nn.compact
    def __call__(self, txy, t_lo, t_hi):
        kinit = nn.initializers.glorot_normal()
        zt = 2.0 * (txy[:, :1] - t_lo) / (t_hi - t_lo) - 1.0
        zxy = (txy[:, 1:] - _LOX) * _SCX - 1.0
        z = jnp.concatenate([zt, zxy], axis=-1)
        proj = 2.0 * jnp.pi * z @ _B_FF.T
        h0 = jnp.concatenate([jnp.cos(proj), jnp.sin(proj), z], axis=-1)
        U = nn.tanh(nn.Dense(self.width, kernel_init=kinit, name="U")(h0))
        V = nn.tanh(nn.Dense(self.width, kernel_init=kinit, name="V")(h0))
        h = nn.tanh(nn.Dense(self.width, kernel_init=kinit, name="L0")(h0))
        for i in range(1, self.depth):
            z_ = nn.tanh(nn.Dense(self.width, kernel_init=kinit, name=f"L{i}")(h))
            h = (1.0 - z_) * U + z_ * V
        raw = nn.Dense(OUTPUT_DIM, kernel_init=kinit, name="out")(h)
        phi = _phi(txy)[:, None]
        u = _lift_u(txy)[:, None] + phi * raw[:, 0:1]
        v = phi * raw[:, 1:2]
        return jnp.concatenate([u, v, raw[:, 2:3]], axis=-1)


# ══════════ Sampling sizes ══════════

N_UNI, N_WAKE, N_ANN = 1536, 512, 512
N_IC = 1024
N_OUT = 128

W_IC, W_OUT = 250.0, 1.0
LR0, LR_DECAY, LR_EVERY, LR_END = 2.5e-3, 0.82, 900, 5e-5
N_INNER = 50
SAVE_MARGIN_S = 8.0


def _load_ic():
    ic = np.loadtxt(_locate("cyl_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    xy = np.stack([ic[:, 0], ic[:, 1]], 1).astype(np.float32)
    uv = np.stack([ic[:, 2], ic[:, 3]], 1).astype(np.float32)
    return jnp.array(xy), jnp.array(uv)


def _sample_col(key, t_lo, t_hi):
    ks = random.split(key, 8)
    xu = random.uniform(ks[0], (N_UNI, 1), minval=X_MIN, maxval=X_MAX)
    yu = random.uniform(ks[1], (N_UNI, 1), minval=Y_MIN, maxval=Y_MAX)
    xw = random.uniform(ks[2], (N_WAKE, 1), minval=0.10, maxval=1.30)
    yw = random.uniform(ks[3], (N_WAKE, 1), minval=0.03, maxval=0.38)
    th = random.uniform(ks[4], (N_ANN, 1), minval=0.0, maxval=2.0 * jnp.pi)
    rr = CYL_R + 0.15 * random.uniform(ks[5], (N_ANN, 1)) ** 2
    xy = jnp.concatenate([jnp.concatenate([xu, yu], 1),
                          jnp.concatenate([xw, yw], 1),
                          jnp.concatenate([CYL_X + rr * jnp.cos(th),
                                           CYL_Y + rr * jnp.sin(th)], 1)], 0)
    n = N_UNI + N_WAKE + N_ANN
    tt = random.uniform(ks[6], (n, 1), minval=t_lo, maxval=t_hi)
    X = jnp.concatenate([tt, xy], 1)
    m = ((X[:, 1] - CYL_X) ** 2 + (X[:, 2] - CYL_Y) ** 2) > CYL_R ** 2
    return X, m.astype(jnp.float32)


def _sample_bc(key, t_lo, t_hi):
    """Only the outlet survives: p = 0 there is the one condition the ADF
    trial does not already satisfy exactly."""
    ks = random.split(key, 2)
    yout = random.uniform(ks[0], (N_OUT, 1), minval=Y_MIN, maxval=Y_MAX)
    tout = random.uniform(ks[1], (N_OUT, 1), minval=t_lo, maxval=t_hi)
    return jnp.concatenate([tout, jnp.full((N_OUT, 1), X_MAX), yout], 1)


def _make_train(rng_init):
    xy_ic, uv_ic0 = _load_ic()
    n_ic = len(xy_ic)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]), 0.0, 0.3)

    def net(p, X, lo, hi):
        return model.apply(p, X, lo, hi)

    def derivs(p, X, lo, hi):
        f = lambda Z: net(p, Z, lo, hi)
        ex = jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), X.shape)
        ey = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), X.shape)
        et = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), X.shape)
        fx = lambda Z: jax.jvp(f, (Z,), (ex,))
        (val, dx), (_, dxx) = jax.jvp(fx, (X,), (ex,))
        fy = lambda Z: jax.jvp(f, (Z,), (ey,))
        (_, dy), (_, dyy) = jax.jvp(fy, (X,), (ey,))
        _, dt = jax.jvp(f, (X,), (et,))
        return val, dt, dx, dy, dxx, dyy

    def loss_fn(p, key, lo, hi, ic_uv):
        kc, kb, ki = random.split(key, 3)
        Xc, mc = _sample_col(kc, lo, hi)
        val, dt, dx, dy, dxx, dyy = derivs(p, Xc, lo, hi)
        u, v = val[:, 0], val[:, 1]
        r_c = dx[:, 0] + dy[:, 1]
        r_mx = (dt[:, 0] + u * dx[:, 0] + v * dy[:, 0] + dx[:, 2]
                - NU * (dxx[:, 0] + dyy[:, 0]))
        r_my = (dt[:, 1] + u * dx[:, 1] + v * dy[:, 1] + dy[:, 2]
                - NU * (dxx[:, 1] + dyy[:, 1]))
        r = r_c ** 2 + r_mx ** 2 + r_my ** 2
        l_pde = jnp.sum(mc * r) / jnp.sum(mc)

        idx = random.choice(ki, n_ic, (N_IC,))
        P = jnp.concatenate([jnp.full((N_IC, 1), lo), xy_ic[idx]], 1)
        fic = net(p, P, lo, hi)
        l_ic = jnp.mean((fic[:, 0] - ic_uv[idx, 0]) ** 2
                        + (fic[:, 1] - ic_uv[idx, 1]) ** 2)

        b_out = _sample_bc(kb, lo, hi)
        fo = net(p, b_out, lo, hi)
        l_out = jnp.mean(fo[:, 2] ** 2)

        return l_pde + W_IC * l_ic + W_OUT * l_out

    sched = optax.exponential_decay(LR0, LR_EVERY, LR_DECAY, end_value=LR_END)
    optimizer = soap(learning_rate=sched, precondition_frequency=5)
    grad_fn = jax.value_and_grad(loss_fn)

    @jit
    def chunk(p, os_, key, lo, hi, ic_uv):
        def body(carry, _):
            p, os_, k = carry
            k, sub = random.split(k)
            loss, g = grad_fn(p, sub, lo, hi, ic_uv)
            upd, os_ = optimizer.update(g, os_)
            return (optax.apply_updates(p, upd), os_, k), loss
        (p, os_, key), losses = jax.lax.scan(
            body, (p, os_, key), None, length=N_INNER)
        return p, os_, key, losses[-1]

    @jit
    def endpoint_uv(p, lo, hi):
        P = jnp.concatenate([jnp.full((n_ic, 1), hi), xy_ic], 1)
        return net(p, P, lo, hi)[:, :2]

    return params, optimizer, chunk, endpoint_uv, uv_ic0, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u","v","p"} each [BS].
    ``params`` is the list of per-segment parameter pytrees; each point is
    routed to the segment whose time window contains it."""
    t = X[:, 0]
    out = _PREDICT_MODEL.apply(params[0], X, SEG[0], SEG[1])
    for k in range(1, N_SEG):
        o = _PREDICT_MODEL.apply(params[k], X, SEG[k], SEG[k + 1])
        sel = (t >= SEG[k]) if k == N_SEG - 1 else ((t >= SEG[k]) & (t < SEG[k + 1]))
        out = jnp.where(sel[:, None], o, out)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, optimizer, chunk, endpoint_uv, uv_ic, key = _make_train(rng)
    opt_state = optimizer.init(params)

    # warmup / compile on segment 0's window
    params, opt_state, key, loss = chunk(params, opt_state, key,
                                         SEG[0], SEG[1], uv_ic)
    step_count = N_INNER
    last_loss = float(loss)

    t_now = time.time()
    total_left = _T0 + TRAIN_TIME - SAVE_MARGIN_S - t_now
    per = total_left / N_SEG
    seg_params = []
    dt = 1.0
    for k in range(N_SEG):
        lo, hi = SEG[k], SEG[k + 1]
        if k > 0:
            uv_ic = endpoint_uv(seg_params[-1], SEG[k - 1], SEG[k])
            opt_state = optimizer.init(params)
        seg_deadline = t_now + (k + 1) * per
        while step_count < MAX_ITER and time.time() + 1.6 * dt < seg_deadline:
            t0 = time.time()
            params, opt_state, key, loss = chunk(params, opt_state, key,
                                                 lo, hi, uv_ic)
            last_loss = float(loss)
            dt = time.time() - t0
            step_count += N_INNER
        seg_params.append(jax.tree_util.tree_map(lambda a: a, params))
        if eval_callback is not None:
            eval_callback(seg_params, step_count, last_loss)

    return seg_params, step_count
