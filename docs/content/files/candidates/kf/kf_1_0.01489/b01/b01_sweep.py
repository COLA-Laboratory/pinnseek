"""SPINN velocity-vorticity PINN with an adaptive causal sweeping window.
Changes vs b01_ceps.py: the NT collocation slices are packed into a moving
window [0, T_act] that advances only when every slice inside it is unlocked,
plus NT_TAIL lightly-weighted slices over the full span.
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
T_MIN, T_MAX = 0.0, 0.75
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 3   # ru, rv, continuity
HAS_SOURCE              = True   # Kolmogorov body force on the x-momentum

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi")
PERIODIC_AXES  = ("x", "y")   # u(t, X_MIN, y) = u(t, X_MAX, y), likewise in y
GEOMETRY_KIND  = "periodic_box"

# --- Physical parameters ---
RE        = 2000.0     # NS-Kolmogorov Re = 1/nu
NU        = 1.0 / RE   # 5e-4
FORCE_AMP = 2.0        # body force f(y) = FORCE_AMP * sin(FORCE_K * pi * y)
FORCE_K   = 4.0        # on the x-momentum only

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


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Design constants ══════════
K_SPACE   = 16        # exact-periodic Fourier modes per spatial axis (input embedding)
K_TIME    = 12        # half-period Fourier features on t (period 2*T_MAX -> no aliasing)
WIDTH     = 64        # per-axis body width
DEPTH     = 4         # per-axis body depth
RANK      = 128       # CP rank of the separable ansatz
NT, NX, NY = 32, 128, 128     # collocation lattice (tensor product)
NT_TAIL   = 8         # extra lightly-weighted slices spanning the whole domain
TA_MIN    = 0.08      # smallest active window
TA_GROW   = 1.5e-4    # per-step multiplicative advance of the causal front
W_OPEN, W_CLOSE = 0.5, 0.05   # advance / retreat thresholds on min causal weight
W_TAIL    = 0.05      # weight on the out-of-window slices
K_IC      = 16        # Fourier truncation of the hard initial-condition ansatz
OMEGA_SC  = 30.0      # vorticity output scale (w_rms of the IC ~ 30)

S_W, S_C, S_D = 400.0, 20.0, 30.0    # residual normalisations (transport / div / curl)
LAM_C, LAM_D  = 1.0, 1.0

LR0, LR1  = 2e-3, 2e-4        # exponential decay across the wall-clock budget
EPS0      = 50.0              # causal-weight tightness (Wang et al. 2024)
EPS_GROW  = 1.02
EPS_MAX   = 1e4
W_FLOOR   = 1e-3              # never fully ignore a time slice
CHUNK     = 100               # steps between host syncs
SAVE_MARGIN_S = 8.0


# ══════════ Hard initial condition: exact Fourier ansatz from kf_ic.csv ══════════

_IC_CACHE: dict[str, Any] = {}


def _ic_coeffs():
    """(k, Ar, Ai) with u0,v0,w0 = Re sum_ab C[c,a,b] exp(2i pi (k_a x + k_b y))."""
    if "C" not in _IC_CACHE:
        ic = np.loadtxt(_locate("kf_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
        n = int(round(np.sqrt(ic.shape[0])))
        U = ic[:, 2].reshape(n, n)          # [ix, iy]
        V = ic[:, 3].reshape(n, n)
        Uh = np.fft.fft2(U) / n ** 2
        Vh = np.fft.fft2(V) / n ** 2
        kk = np.fft.fftfreq(n, 1.0 / n)
        KX, KY = np.meshgrid(kk, kk, indexing="ij")
        Wh = 2j * np.pi * (KX * Vh - KY * Uh)          # vorticity w = v_x - u_y
        ph = np.exp(-1j * np.pi * (KX + KY) / n)       # grid is cell-centred: x_i=(i+.5)/n
        Uh, Vh, Wh = Uh * ph, Vh * ph, Wh * ph
        idx = np.concatenate([np.arange(0, K_IC + 1), np.arange(n - K_IC, n)])
        ks = np.concatenate([np.arange(0, K_IC + 1), np.arange(-K_IC, 0)]).astype(np.float32)
        C = np.stack([Uh[np.ix_(idx, idx)], Vh[np.ix_(idx, idx)], Wh[np.ix_(idx, idx)]])
        _IC_CACHE["k"] = jnp.asarray(ks)
        _IC_CACHE["Ar"] = jnp.asarray(C.real.astype(np.float32))
        _IC_CACHE["Ai"] = jnp.asarray(C.imag.astype(np.float32))
        _IC_CACHE["C"] = True
    return _IC_CACHE["k"], _IC_CACHE["Ar"], _IC_CACHE["Ai"]


def _trig(z, k):
    a = 2.0 * jnp.pi * z * k          # z: [N,1], k: [M]  ->  [N, M]
    return jnp.cos(a), jnp.sin(a)


def _ic_grid(x, y):
    """[3, NX, NY] initial field on the tensor grid (differentiable in x, y)."""
    k, Ar, Ai = _ic_coeffs()
    Cx, Sx = _trig(x, k)
    Cy, Sy = _trig(y, k)
    return (jnp.einsum("ia,cab,jb->cij", Cx, Ar, Cy)
            - jnp.einsum("ia,cab,jb->cij", Sx, Ar, Sy)
            - jnp.einsum("ia,cab,jb->cij", Sx, Ai, Cy)
            - jnp.einsum("ia,cab,jb->cij", Cx, Ai, Sy))


def _ic_scatter(x, y):
    """[3, N] initial field at scattered points."""
    k, Ar, Ai = _ic_coeffs()
    Cx, Sx = _trig(x, k)
    Cy, Sy = _trig(y, k)
    return (jnp.einsum("na,cab,nb->cn", Cx, Ar, Cy)
            - jnp.einsum("na,cab,nb->cn", Sx, Ar, Sy)
            - jnp.einsum("na,cab,nb->cn", Sx, Ai, Cy)
            - jnp.einsum("na,cab,nb->cn", Cx, Ai, Sy))


# ══════════ Separable (SPINN) backbone with exact periodic embedding ══════════

class Body(nn.Module):
    """One per-axis MLP R -> R^{3 x RANK}. Space axes get an exactly periodic
    cos/sin embedding, so every field and all its derivatives are periodic."""
    kind: str

    @nn.compact
    def __call__(self, z):                                  # z: [N, 1]
        if self.kind == "t":
            m = jnp.arange(1, K_TIME + 1, dtype=jnp.float32)
            a = jnp.pi * z * m / T_MAX                      # period 2*T_MAX
            h = jnp.concatenate([2.0 * z / T_MAX - 1.0, jnp.cos(a), jnp.sin(a)], axis=-1)
        else:
            k = jnp.arange(1, K_SPACE + 1, dtype=jnp.float32)
            a = 2.0 * jnp.pi * z * k
            h = jnp.concatenate([jnp.cos(a), jnp.sin(a)], axis=-1)
        for _ in range(DEPTH):
            h = jnp.tanh(nn.Dense(WIDTH)(h))
        return nn.Dense(3 * RANK)(h).reshape(z.shape[0], 3, RANK)


class SPINN(nn.Module):
    def setup(self):
        self.bt = Body("t", name="bt")
        self.bx = Body("x", name="bx")
        self.by = Body("y", name="by")

    def __call__(self, t, x, y):                            # tensor grid -> [3, NT, NX, NY]
        Ft, Fx, Fy = self.bt(t), self.bx(x), self.by(y)
        return jnp.einsum("icr,jcr,kcr->cijk", Ft, Fx, Fy) / RANK ** 0.5

    def scatter(self, t, x, y):                             # scattered -> [3, N]
        Ft, Fx, Fy = self.bt(t), self.bx(x), self.by(y)
        return jnp.sum(Ft * Fx * Fy, axis=-1).T / RANK ** 0.5


_MODEL = SPINN()
_SCALE = jnp.array([1.0, 1.0, OMEGA_SC])


def _trial_grid(params, t, x, y):
    """u, v, w on the lattice: hard IC + t-ramped network correction."""
    n = _MODEL.apply(params, t, x, y)                       # [3, NT, NX, NY]
    base = _ic_grid(x, y)[:, None, :, :]                    # [3, 1, NX, NY]
    ramp = (t[:, 0] / T_MAX)[None, :, None, None]
    return base + ramp * _SCALE[:, None, None, None] * n


# ══════════ Residual (velocity-vorticity form; pressure eliminated) ══════════

def _residuals(params, t, x, y):
    f_of = lambda tt, xx, yy: _trial_grid(params, tt, xx, yy)
    ot, ox, oy = jnp.ones_like(t), jnp.ones_like(x), jnp.ones_like(y)

    f, f_t = jax.jvp(lambda a: f_of(a, x, y), (t,), (ot,))

    def gx(a):
        return jax.jvp(lambda b: f_of(t, b, y), (a,), (ox,))
    (_, f_x), (_, f_xx) = jax.jvp(gx, (x,), (ox,))

    def gy(a):
        return jax.jvp(lambda b: f_of(t, x, b), (a,), (oy,))
    (_, f_y), (_, f_yy) = jax.jvp(gy, (y,), (oy,))

    u, v, w = f[0], f[1], f[2]
    curl_f = FORCE_AMP * FORCE_K * jnp.pi * jnp.cos(FORCE_K * jnp.pi * y[:, 0])
    r_w = (f_t[2] + u * f_x[2] + v * f_y[2]
           - NU * (f_xx[2] + f_yy[2]) + curl_f[None, None, :])
    r_c = f_x[0] + f_y[1]
    r_d = w - (f_x[1] - f_y[0])
    return r_w, r_c, r_d


def _slice_loss(params, t, x, y):
    r_w, r_c, r_d = _residuals(params, t, x, y)
    return (jnp.mean((r_w / S_W) ** 2, axis=(1, 2))
            + LAM_C * jnp.mean((r_c / S_C) ** 2, axis=(1, 2))
            + LAM_D * jnp.mean((r_d / S_D) ** 2, axis=(1, 2)))     # [NT]


def _loss(params, t, x, y, eps):
    Lr = _slice_loss(params, t, x, y)
    Lw, Lt = Lr[:NT], Lr[NT:]
    cum = jnp.concatenate([jnp.zeros((1,), Lr.dtype), jnp.cumsum(Lw)[:-1]])
    wgt = jax.lax.stop_gradient(jnp.maximum(jnp.exp(-eps * cum), W_FLOOR))
    loss = (jnp.sum(wgt * Lw) + W_TAIL * jnp.sum(Lt)) / (NT + NT_TAIL)
    return loss, (Lr, wgt)


def _sample(key, tact):
    k1, k2, k3, k4 = random.split(key, 4)
    tw = (jnp.arange(NT) + random.uniform(k1, (NT,))) * (tact / NT)
    tt = (jnp.arange(NT_TAIL) + random.uniform(k4, (NT_TAIL,))) * (T_MAX / NT_TAIL)
    t = jnp.concatenate([tw, tt])[:, None]
    x = ((jnp.arange(NX) + random.uniform(k2, (NX,))) / NX)[:, None]
    y = ((jnp.arange(NY) + random.uniform(k3, (NY,))) / NY)[:, None]
    return t, x, y


# ══════════ Public contract ══════════

_PREDICT_MODEL = SPINN()


def predict_fn(params, X):
    """X: [BS, 3] columns (t, x, y) -> {"u","v","p"}. The velocity-vorticity
    form carries no pressure unknown, so "p" is returned as zeros (unscored)."""
    N = X.shape[0]
    CH = 16384
    us, vs = [], []
    for s in range(0, N, CH):
        Xc = X[s:s + CH]
        t, x, y = Xc[:, 0:1], Xc[:, 1:2], Xc[:, 2:3]
        n = _PREDICT_MODEL.apply(params, t, x, y, method=SPINN.scatter)   # [3, n]
        f = _ic_scatter(x, y) + (t[:, 0] / T_MAX)[None, :] * _SCALE[:, None] * n
        us.append(f[0])
        vs.append(f[1])
    u = jnp.concatenate(us)
    return {"u": u, "v": jnp.concatenate(vs), "p": jnp.zeros_like(u)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Adam on the causally weighted residual of the velocity-vorticity NS
    system, under the hard-IC separable ansatz. Wall-deadline anchored at
    _T0; the loop stops SAVE_MARGIN_S early so params can be pickled."""
    _ic_coeffs()
    key, rng = random.split(rng)
    t0, x0, y0 = _sample(key, jnp.asarray(TA_MIN))
    params = _MODEL.init(key, t0, x0, y0)

    opt = optax.adam(learning_rate=1.0)
    opt_state = opt.init(params)

    @jit
    def update(params, opt_state, key, tact, lr):
        key, sub = random.split(key)
        t, x, y = _sample(sub, tact)
        (loss, (Lr, wgt)), grads = jax.value_and_grad(_loss, has_aux=True)(params, t, x, y, EPS0)
        upd, opt_state = opt.update(grads, opt_state)
        params = jax.tree_util.tree_map(lambda p, u: p + lr * u, params, upd)
        mw = wgt.min()
        rate = 1.0 + TA_GROW * jnp.where(mw > W_OPEN, 1.0, jnp.where(mw < W_CLOSE, -1.0, 0.0))
        tact = jnp.clip(tact * rate, TA_MIN, T_MAX)
        return params, opt_state, key, tact, loss

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    span = max(deadline - _T0, 1.0)
    tact = jnp.asarray(TA_MIN, jnp.float32)

    # warmup: compiles the whole step; it is a real step (counted)
    params, opt_state, key, tact, loss = update(params, opt_state, key, tact, jnp.asarray(LR0))
    step_count = 1
    last = float(loss)

    while time.time() < deadline and step_count < MAX_ITER:
        frac = min(max((time.time() - _T0) / span, 0.0), 1.0)
        lr = jnp.asarray(LR0 * (LR1 / LR0) ** frac, dtype=jnp.float32)
        for _ in range(CHUNK):
            params, opt_state, key, tact, loss = update(params, opt_state, key, tact, lr)
        step_count += CHUNK
        last = float(loss)
        if eval_callback is not None:
            eval_callback(params, step_count, last)

    return params, step_count
