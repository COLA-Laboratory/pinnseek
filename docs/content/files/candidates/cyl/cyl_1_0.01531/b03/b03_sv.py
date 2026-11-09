"""Streamfunction-vorticity marched PINN: exactly divergence-free, pressure-free.
Changed vs b02_wic350.py: the net outputs (psi, omega) instead of (u,v,p);
u = psi_y, v = -psi_x (continuity exact) and the residuals are the vorticity
transport equation + omega = -lap(psi) (kb1 019 VV form). Hard BCs via an
order-2 ADF trial on psi with a trainable cylinder streamline value c(t).
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


def psi_pois(t, y):
    """Poiseuille streamfunction: d/dy = inlet_profile, d/dx = 0, psi(0)=0."""
    return (4.0 * inlet_centreline(t) / (Y_MAX ** 2)
            * (0.5 * Y_MAX * y ** 2 - y ** 3 / 3.0))


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


# --- ADF hard-BC machinery for the streamfunction (kb1 036) --------------
# psi must satisfy BOTH psi = const and grad psi = 0 on every no-slip wall,
# so the boundary factor vanishes to second order: PHI = tanh^2(D/PSI_ELL).
# Scale: near a wall psi ~ (omega_w/2) d^2, so PHI ~ d^2/PSI_ELL^2 with
# PSI_ELL = sqrt(2*psi_scale/omega_w) ~ sqrt(2*0.1/100) = 0.045.
PSI_ELL = 0.045        # saturation length of the boundary factor
ELL_C   = 0.04         # width of the cylinder collar carrying psi = c(t)
PSI_S   = 0.15         # output scale of the streamfunction correction
PSI_INIT = 0.02        # init shrink of the psi head (see PINN.__call__)
OM_S    = 20.0         # output scale of the vorticity channel
C_S     = 0.05         # scale of the trainable cylinder streamline offset


def _omegas(txy):
    x, y = txy[:, 1], txy[:, 2]
    r = jnp.sqrt((x - CYL_X) ** 2 + (y - CYL_Y) ** 2 + 1e-9)
    return (y, Y_MAX - y, jnp.maximum(r - CYL_R, 0.0), x)


def _rconj(a, b):
    """R-conjunction (smooth min): 0 iff a or b is 0, positive otherwise."""
    return a + b - jnp.sqrt(a * a + b * b + 1e-12)


def _dist(txy):
    wb, wt, wc, wi = _omegas(txy)
    return _rconj(_rconj(wb, wt), _rconj(wc, wi))


def _phi2(txy):
    """Order-2 boundary factor: 0 with zero gradient on walls/cylinder/inlet."""
    return jnp.tanh(_dist(txy) / PSI_ELL) ** 2


def _kappa(txy):
    """Cylinder collar: 1 with zero gradient on the cylinder, ~1e-11 at the
    channel walls, so it never disturbs them."""
    _, _, wc, _ = _omegas(txy)
    return jnp.exp(-(wc / ELL_C) ** 2)


class PINN(nn.Module):
    """Gated modified-MLP (kb1 050) on a fixed random Fourier embedding
    (kb1 020), two outputs (psi, omega).  psi is wrapped in an order-2 ADF
    trial so the inlet profile, the no-slip walls and the no-slip cylinder
    (value *and* normal derivative) hold exactly; the cylinder streamline
    value c(t) is a trainable quadratic in segment-normalised time."""
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
        # psi's velocity is a *derivative* of the output, and the Fourier
        # embedding amplifies d/dy by ~90x, so the psi head starts small:
        # the field then begins at the Poiseuille lift (kb1 082's argument
        # for a physically sensible initial state).
        small = lambda k, s, d: PSI_INIT * kinit(k, s, d)
        raw_psi = nn.Dense(1, kernel_init=small, name="out_psi")(h)[:, 0]
        raw_om = nn.Dense(1, kernel_init=kinit, name="out_om")(h)[:, 0]

        a = self.param("c_coef", nn.initializers.zeros, (3,))
        s = zt[:, 0]
        t, y = txy[:, 0], txy[:, 2]
        c = psi_pois(t, CYL_Y) + C_S * (a[0] + a[1] * s + a[2] * s * s)
        kap = _kappa(txy)
        lift = (1.0 - kap) * psi_pois(t, y) + kap * c
        psi = lift + PSI_S * _phi2(txy) * raw_psi
        om = OM_S * raw_om
        return jnp.stack([psi, om], axis=-1)


# ══════════ Sampling sizes ══════════

N_UNI, N_WAKE, N_ANN = 1536, 512, 512
N_IC = 1024

W_IC, W_ICW = 350.0, 350.0
S_KIN, S_VOR = 12.0, 100.0     # residual normalisations (omega, omega/s units)
W_KIN, W_VOR = 1.0, 1.0
LR0, LR_DECAY, LR_EVERY, LR_END = 2.5e-3, 0.82, 900, 5e-5
N_INNER = 50
SAVE_MARGIN_S = 8.0

_EX = jnp.array([0.0, 1.0, 0.0])
_EY = jnp.array([0.0, 0.0, 1.0])
_ET = jnp.array([1.0, 0.0, 0.0])


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
        ex = jnp.broadcast_to(_EX, X.shape)
        ey = jnp.broadcast_to(_EY, X.shape)
        et = jnp.broadcast_to(_ET, X.shape)
        fx = lambda Z: jax.jvp(f, (Z,), (ex,))
        (val, dx), (_, dxx) = jax.jvp(fx, (X,), (ex,))
        fy = lambda Z: jax.jvp(f, (Z,), (ey,))
        (_, dy), (_, dyy) = jax.jvp(fy, (X,), (ey,))
        _, dt = jax.jvp(f, (X,), (et,))
        return val, dt, dx, dy, dxx, dyy

    def velocity(p, X, lo, hi):
        """u = psi_y, v = -psi_x — exactly divergence-free."""
        f = lambda Z: net(p, Z, lo, hi)
        _, dx = jax.jvp(f, (X,), (jnp.broadcast_to(_EX, X.shape),))
        val, dy = jax.jvp(f, (X,), (jnp.broadcast_to(_EY, X.shape),))
        return jnp.stack([dy[:, 0], -dx[:, 0]], 1), val[:, 1]

    def loss_fn(p, key, lo, hi, ic_uv, ic_w, w_icw):
        kc, ki = random.split(key, 2)
        Xc, mc = _sample_col(kc, lo, hi)
        val, dt, dx, dy, dxx, dyy = derivs(p, Xc, lo, hi)
        u, v = dy[:, 0], -dx[:, 0]
        om = val[:, 1]
        r_kin = (om + dxx[:, 0] + dyy[:, 0]) / S_KIN
        r_vor = (dt[:, 1] + u * dx[:, 1] + v * dy[:, 1]
                 - NU * (dxx[:, 1] + dyy[:, 1])) / S_VOR
        r = W_KIN * r_kin ** 2 + W_VOR * r_vor ** 2
        l_pde = jnp.sum(mc * r) / jnp.sum(mc)

        idx = random.choice(ki, n_ic, (N_IC,))
        P = jnp.concatenate([jnp.full((N_IC, 1), lo), xy_ic[idx]], 1)
        uv, om0 = velocity(p, P, lo, hi)
        l_ic = jnp.mean((uv[:, 0] - ic_uv[idx, 0]) ** 2
                        + (uv[:, 1] - ic_uv[idx, 1]) ** 2)
        l_icw = jnp.mean(((om0 - ic_w[idx]) / OM_S) ** 2)

        return l_pde + W_IC * l_ic + w_icw * l_icw

    sched = optax.exponential_decay(LR0, LR_EVERY, LR_DECAY, end_value=LR_END)
    optimizer = soap(learning_rate=sched, precondition_frequency=5)
    grad_fn = jax.value_and_grad(loss_fn)

    @jit
    def chunk(p, os_, key, lo, hi, ic_uv, ic_w, w_icw):
        def body(carry, _):
            p, os_, k = carry
            k, sub = random.split(k)
            loss, g = grad_fn(p, sub, lo, hi, ic_uv, ic_w, w_icw)
            upd, os_ = optimizer.update(g, os_)
            return (optax.apply_updates(p, upd), os_, k), loss
        (p, os_, key), losses = jax.lax.scan(
            body, (p, os_, key), None, length=N_INNER)
        return p, os_, key, losses[-1]

    @jit
    def endpoint(p, lo, hi):
        P = jnp.concatenate([jnp.full((n_ic, 1), hi), xy_ic], 1)
        return velocity(p, P, lo, hi)

    return params, optimizer, chunk, endpoint, uv_ic0, n_ic, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u","v","p"} each [BS].
    u = psi_y, v = -psi_x from the segment owning each point; p is not part
    of this formulation (the curl of the momentum equation eliminates it)
    and is returned as zeros — it is not scored."""
    t = np.asarray(X[:, 0])
    n = X.shape[0]
    Xn = np.asarray(X)
    u = np.zeros(n, dtype=np.float32)
    v = np.zeros(n, dtype=np.float32)
    ex = jnp.array([0.0, 1.0, 0.0])
    ey = jnp.array([0.0, 0.0, 1.0])
    for k in range(N_SEG):
        lo, hi = SEG[k], SEG[k + 1]
        sel = (t >= lo) if k == N_SEG - 1 else ((t >= lo) & (t < hi))
        if k == 0:
            sel = t < SEG[1]
        idx = np.nonzero(sel)[0]
        if idx.size == 0:
            continue
        Xi = jnp.asarray(Xn[idx])
        f = lambda Z: _PREDICT_MODEL.apply(params[k], Z, lo, hi)
        _, dx = jax.jvp(f, (Xi,), (jnp.broadcast_to(ex, Xi.shape),))
        _, dy = jax.jvp(f, (Xi,), (jnp.broadcast_to(ey, Xi.shape),))
        u[idx] = np.asarray(dy[:, 0])
        v[idx] = -np.asarray(dx[:, 0])
    return {"u": jnp.asarray(u), "v": jnp.asarray(v), "p": jnp.zeros(n)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    params, optimizer, chunk, endpoint, uv_ic, n_ic, key = _make_train(rng)
    opt_state = optimizer.init(params)
    w_ic = jnp.zeros(n_ic)

    # warmup / compile on segment 0's window
    params, opt_state, key, loss = chunk(params, opt_state, key,
                                         SEG[0], SEG[1], uv_ic, w_ic, 0.0)
    step_count = N_INNER
    last_loss = float(loss)

    t_now = time.time()
    total_left = _T0 + TRAIN_TIME - SAVE_MARGIN_S - t_now
    per = total_left / N_SEG
    seg_params = []
    dt = 1.0
    w_icw = 0.0
    for k in range(N_SEG):
        lo, hi = SEG[k], SEG[k + 1]
        if k > 0:
            uv_ic, w_ic = endpoint(seg_params[-1], SEG[k - 1], SEG[k])
            w_icw = W_ICW
            opt_state = optimizer.init(params)
        seg_deadline = t_now + (k + 1) * per
        while step_count < MAX_ITER and time.time() + 1.6 * dt < seg_deadline:
            t0 = time.time()
            params, opt_state, key, loss = chunk(params, opt_state, key,
                                                 lo, hi, uv_ic, w_ic, w_icw)
            last_loss = float(loss)
            dt = time.time() - t0
            step_count += N_INNER
        seg_params.append(jax.tree_util.tree_map(lambda a: a, params))
        if eval_callback is not None:
            eval_callback(seg_params, step_count, last_loss)

    return seg_params, step_count
