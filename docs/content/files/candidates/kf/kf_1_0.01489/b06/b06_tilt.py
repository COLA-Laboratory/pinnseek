"""Time-marching: NWIN windows in series, each hard-initialised from the previous.
Change vs b06_k40e9.py: wall time per window tilted 1:2 toward late windows, whose
field, read off exactly as its |k|<=K coefficients.  The front diag shows the
single-window causal front stalling at t~0.5 and advancing only logarithmically.
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
from soap_jax import soap


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Design constants ══════════
K_SPACE   = 40        # band limit of the trial space AND of the test-mode box
K_TIME    = 12        # half-period Fourier features on local time tau in [0, WIN_DT]
WIDTH     = 256       # time-body width
DEPTH     = 4         # time-body depth
RANK      = 1536      # CP rank of the separable stream-function ansatz
NWIN      = 5         # time windows trained in series
T_WIN     = 0.75      # modelled/trained time window [0, T_WIN]
WIN_DT    = T_WIN / NWIN
NT, NX, NY = 64, 128, 128       # collocation lattice per window; NX >= 3*K_SPACE
K_IC      = K_SPACE   # Fourier truncation of the hard initial-condition ansatz
NG        = NX        # grid on which a window's endpoint psi is read off (>= 2K+1)
PSI_SC    = 0.05      # stream-function output scale (psi_rms of the IC ~ 0.041)

K_T       = K_SPACE   # test-mode box |kx|,|ky| <= K_T == the trial space: square system
S_M       = 8.0       # velocity-residual normalisation (|u_t| ~ 5, weak norm ~ 10)
NORM_EXPO = 1.5       # residual measured in H^-NORM_EXPO (1 = velocity/momentum norm)

LR0, LR1  = 5e-3, 3e-3        # exponential decay, restarted in every window
PRECOND_EVERY = 240           # SOAP eigenbasis refresh interval
EPS0      = 9.0e3             # causal tightness; the b03/b05 invariant is eps*NT/WIN_DT
                              # = 2.56e6, the value b05 tuned at NT 128 over [0, 0.75]
W_FLOOR   = 3e-3              # never fully ignore a time slice (1.0 = no gating)
CHUNK     = 100               # steps between host syncs
EMA       = 0.99              # iterate-averaging decay (~100-step window)
WIN_TILT  = 1.0               # later windows get more wall: share_j ~ 1 + TILT*j/(NWIN-1)
WARM      = True              # warm-start each window from the previous window's iterate
SAVE_MARGIN_S = 8.0

# spatial derivative orders of psi_0 the weak residual needs (index 0 = psi itself)
_IC_ORDERS = ((0, 0), (0, 1), (1, 0), (2, 0), (0, 2))
_SHARE = np.cumsum([1.0 + WIN_TILT * j / max(NWIN - 1, 1) for j in range(NWIN)])
_SHARE = _SHARE / _SHARE[-1]
_KS = np.concatenate([np.arange(0, K_IC + 1), np.arange(-K_IC, 0)]).astype(np.float64)


# ══════════ Hard initial condition: exact Fourier stream function ══════════

def _ic_stack(C):
    """psi_0 spectral coefficients -> the (real, imag) derivative stacks [5, M, M]."""
    kx = jnp.asarray((2j * np.pi * _KS)[:, None])
    ky = jnp.asarray((2j * np.pi * _KS)[None, :])
    D = jnp.stack([C * kx ** dx * ky ** dy for dx, dy in _IC_ORDERS])
    return jnp.real(D).astype(jnp.float32), jnp.imag(D).astype(jnp.float32)


def _ic0_coeffs():
    """Window 0's psi_0: the |k| <= K_IC coefficient box of the reference IC."""
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
    Wh = Wh * ph
    K2 = (2 * np.pi) ** 2 * (KX ** 2 + KY ** 2)    # -lap psi = w  ->  psi_h = w_h / K2
    K2[0, 0] = 1.0
    Ph = Wh / K2
    Ph[0, 0] = 0.0
    idx = np.concatenate([np.arange(0, K_IC + 1), np.arange(n - K_IC, n)])
    return jnp.asarray(Ph[np.ix_(idx, idx)].astype(np.complex64))


def _trig(z, k):
    a = 2.0 * jnp.pi * z * k          # z: [N,1], k: [M]  ->  [N, M]
    return jnp.cos(a), jnp.sin(a)


def _ic_grid(A, x, y):
    """[5, NX, NY]: the psi_0 derivatives listed in _IC_ORDERS on the lattice."""
    Ar, Ai = A
    k = jnp.asarray(_KS.astype(np.float32))
    Cx, Sx = _trig(x, k)
    Cy, Sy = _trig(y, k)
    return (jnp.einsum("ia,dab,jb->dij", Cx, Ar, Cy)
            - jnp.einsum("ia,dab,jb->dij", Sx, Ar, Sy)
            - jnp.einsum("ia,dab,jb->dij", Sx, Ai, Cy)
            - jnp.einsum("ia,dab,jb->dij", Cx, Ai, Sy))


def _ic_scatter(A, x, y):
    """[5, N]: same derivatives at scattered points (only psi_y, psi_x are used)."""
    Ar, Ai = A
    k = jnp.asarray(_KS.astype(np.float32))
    Cx, Sx = _trig(x, k)
    Cy, Sy = _trig(y, k)
    return (jnp.einsum("na,dab,nb->dn", Cx, Ar, Cy)
            - jnp.einsum("na,dab,nb->dn", Sx, Ar, Sy)
            - jnp.einsum("na,dab,nb->dn", Sx, Ai, Cy)
            - jnp.einsum("na,dab,nb->dn", Cx, Ai, Sy))


# ══════════ Separable backbone with exact periodic embedding ══════════

class Body(nn.Module):
    """One per-axis map R -> R^RANK. The time axis is a deep tanh MLP on half-period
    Fourier features of the *local* window time; the space axes are a linear map of the
    exactly periodic cos/sin features, so psi and all its derivatives are trig
    polynomials of degree <= K_SPACE to machine precision."""
    kind: str

    @nn.compact
    def __call__(self, z):                                  # z: [N, 1] -> [N, RANK]
        if self.kind == "t":
            m = jnp.arange(1, K_TIME + 1, dtype=jnp.float32)
            a = jnp.pi * z * m / WIN_DT                     # period 2*WIN_DT
            h = jnp.concatenate([2.0 * z / WIN_DT - 1.0, jnp.cos(a), jnp.sin(a)], axis=-1)
            for _ in range(DEPTH):
                h = jnp.tanh(nn.Dense(WIDTH)(h))
            return nn.Dense(RANK)(h)
        k = jnp.arange(1, K_SPACE + 1, dtype=jnp.float32)
        a = 2.0 * jnp.pi * z * k
        h = jnp.concatenate([jnp.ones_like(z), jnp.cos(a), jnp.sin(a)], axis=-1)
        return nn.Dense(RANK, use_bias=False)(h)            # exactly degree <= K_SPACE


class PSINET(nn.Module):
    def setup(self):
        self.bt = Body("t", name="bt")
        self.bx = Body("x", name="bx")
        self.by = Body("y", name="by")

    def __call__(self, t, x, y):
        return self.bt(t), self.bx(x), self.by(y)

    def one(self, z, kind):
        return {"t": self.bt, "x": self.bx, "y": self.by}[kind](z)


_MODEL = PSINET()


def _dstack(f, z, n):
    """[f(z), f'(z), ..., f^(n)(z)] for a row-wise f: [N,1] -> [N,R], by nested
    forward-mode AD (the per-axis bodies are tiny, so the 2^n cost is negligible)."""
    fs = [f]
    for _ in range(n):
        g = fs[-1]
        fs.append(lambda zz, g=g: jax.jvp(g, (zz,), (jnp.ones_like(zz),))[1])
    return [fi(z) for fi in fs]


# ══════════ Fields: u, v, w, w_t on the lattice (all by AD) ══════════

def _bodies(params):
    return lambda kind: (lambda z: _MODEL.apply(params, z, kind, method=PSINET.one))


def _fields(params, A, tau, x, y):
    b = _bodies(params)
    Ft = _dstack(b("t"), tau, 1)
    Fx = _dstack(b("x"), x, 2)
    Fy = _dstack(b("y"), y, 2)
    ic = _ic_grid(A, x, y)                                  # [5, NX, NY]
    sc = PSI_SC / RANK ** 0.5
    ramp = (tau[:, 0] / WIN_DT)[:, None, None]
    dramp = jnp.full_like(ramp, 1.0 / WIN_DT)

    def net(a, bx, cy):
        base = jnp.einsum("ir,jr,kr->ijk", Ft[0], Fx[bx], Fy[cy])
        if a == 0:
            return sc * ramp * base
        dt = jnp.einsum("ir,jr,kr->ijk", Ft[1], Fx[bx], Fy[cy])
        return sc * (dramp * base + ramp * dt)

    psi_y = ic[1][None] + net(0, 0, 1)
    psi_x = ic[2][None] + net(0, 1, 0)
    psi_xx = ic[3][None] + net(0, 2, 0)
    psi_yy = ic[4][None] + net(0, 0, 2)
    u, v = psi_y, -psi_x
    w = -(psi_xx + psi_yy)
    w_t = -(net(1, 2, 0) + net(1, 0, 2))                    # psi_0 is tau-independent
    return u, v, w, w_t


def _psi_grid(params, A, tau, x, y):
    """psi itself on the lattice — used to read a window's endpoint field."""
    b = _bodies(params)
    sc = PSI_SC / RANK ** 0.5
    ramp = (tau[:, 0] / WIN_DT)[:, None, None]
    base = jnp.einsum("ir,jr,kr->ijk", b("t")(tau), b("x")(x), b("y")(y))
    return _ic_grid(A, x, y)[0][None] + sc * ramp * base


def _endpoint_C(params, A):
    """Exact |k| <= K_IC coefficient box of psi at the end of the window.  psi is a trig
    polynomial of degree <= K_SPACE, so sampling on NG >= 2K+1 nodes and transforming is
    exact — this is what makes a restart lossless."""
    z = jnp.asarray((np.arange(NG) / NG).astype(np.float32))[:, None]
    tau = jnp.asarray(np.array([[WIN_DT]], np.float32))
    psi = _psi_grid(params, A, tau, z, z)[0]
    Ch = jnp.fft.fft2(psi) / NG ** 2
    idx = np.concatenate([np.arange(0, K_IC + 1), np.arange(NG - K_IC, NG)])
    return Ch[idx][:, idx].astype(jnp.complex64)


# ══════════ Weak (Fourier-Galerkin) vorticity residual ══════════

_SPEC: dict[str, Any] = {}


def _spec():
    """Test-mode wavenumbers, the |kx|,|ky| <= K_T *box* mask — exactly the modes the
    band-limited trial space carries, so the system is square and its zero is the
    Galerkin solution — and the 1/(2 pi |k|) weight that turns the vorticity residual
    into a velocity (momentum) residual."""
    if "kx" not in _SPEC:
        kk = np.fft.fftfreq(NX, 1.0 / NX)
        ll = np.fft.fftfreq(NY, 1.0 / NY)
        KXn, KYn = np.meshgrid(kk, ll, indexing="ij")
        K2 = KXn ** 2 + KYn ** 2
        mask = (np.abs(KXn) <= K_T + 1e-9) & (np.abs(KYn) <= K_T + 1e-9) & (K2 > 0)
        wgt = np.where(mask, 1.0 / (2 * np.pi * np.sqrt(np.where(K2 > 0, K2, 1.0))) ** NORM_EXPO, 0.0)
        _SPEC["kx"] = jnp.asarray(2 * np.pi * KXn)
        _SPEC["ky"] = jnp.asarray(2 * np.pi * KYn)
        _SPEC["k2"] = jnp.asarray(4 * np.pi ** 2 * K2)
        _SPEC["w"] = jnp.asarray(wgt.astype(np.float32))
    return _SPEC["kx"], _SPEC["ky"], _SPEC["k2"], _SPEC["w"]


def _slice_loss(params, A, tau, x, y):
    """Residual of the vorticity equation tested against exp(-2 pi i k.x), |k| <= K_T.
    Using div(u)=0, the advection term is integrated by parts as div(u w), so no
    derivative above second order of psi ever appears."""
    kx, ky, k2, wgt = _spec()
    u, v, w, w_t = _fields(params, A, tau, x, y)
    n2 = NX * NY
    F = lambda a: jnp.fft.fft2(a, axes=(1, 2)) / n2
    fy = FORCE_AMP * FORCE_K * jnp.pi * jnp.cos(FORCE_K * jnp.pi * y[:, 0])   # curl of f
    Fh = jnp.fft.fft2(jnp.broadcast_to(fy[None, None, :], (1, NX, NY)), axes=(1, 2)) / n2
    R = (F(w_t) + 1j * (kx * F(u * w) + ky * F(v * w)) + NU * k2 * F(w) + Fh)
    R2 = R.real ** 2 + R.imag ** 2          # |.|**2 directly: abs() has no gradient at 0
    return jnp.sum(R2 * (wgt / S_M) ** 2, axis=(1, 2))             # [NT]


def _loss(params, A, tau, x, y, eps):
    Lr = _slice_loss(params, A, tau, x, y)
    cum = jnp.concatenate([jnp.zeros((1,), Lr.dtype), jnp.cumsum(Lr)[:-1]])
    wgt = jax.lax.stop_gradient(jnp.maximum(jnp.exp(-eps * cum), W_FLOOR))
    return jnp.mean(wgt * Lr), (Lr, wgt)


def _sample(key):
    k1, k2, k3 = random.split(key, 3)
    tau = ((jnp.arange(NT) + random.uniform(k1, (NT,))) * (WIN_DT / NT))[:, None]
    x = (jnp.arange(NX) + random.uniform(k2, ())) / NX          # uniform grid (FFT)
    y = (jnp.arange(NY) + random.uniform(k3, ())) / NY
    return tau, x[:, None], y[:, None]


# ══════════ Public contract ══════════

def predict_fn(params, X):
    """X: [BS, 3] columns (t, x, y) -> {"u","v","p"}. u = psi_y, v = -psi_x; the
    stream-function form carries no pressure unknown, so "p" is zeros (unscored)."""
    wins = params["w"]
    N = X.shape[0]
    CH = 16384
    sc = PSI_SC / RANK ** 0.5
    us, vs = [], []
    for s in range(0, N, CH):
        Xc = X[s:s + CH]
        t, x, y = Xc[:, 0:1], Xc[:, 1:2], Xc[:, 2:3]
        j = jnp.clip(jnp.floor(t[:, 0] / WIN_DT), 0, len(wins) - 1)
        u = jnp.zeros_like(t[:, 0])
        v = jnp.zeros_like(u)
        for w_i, win in enumerate(wins):
            tau = t - w_i * WIN_DT
            A = (win["Ar"], win["Ai"])
            b = _bodies(win["p"])
            Ft = b("t")(tau)
            Fx, Fx1 = _dstack(b("x"), x, 1)
            Fy, Fy1 = _dstack(b("y"), y, 1)
            ramp = tau[:, 0] / WIN_DT
            psi_x = sc * ramp * jnp.sum(Ft * Fx1 * Fy, axis=-1)
            psi_y = sc * ramp * jnp.sum(Ft * Fx * Fy1, axis=-1)
            icd = _ic_scatter(A, x, y)
            m = (j == w_i)
            u = jnp.where(m, icd[1] + psi_y, u)
            v = jnp.where(m, -(icd[2] + psi_x), v)
        us.append(u)
        vs.append(v)
    u = jnp.concatenate(us)
    return {"u": u, "v": jnp.concatenate(vs), "p": jnp.zeros_like(u)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """SOAP on the causally weighted Fourier-Galerkin residual, window by window: each
    window is a fresh causal sweep whose hard IC is the previous window's endpoint field.
    Wall-deadline anchored at _T0; the loop stops SAVE_MARGIN_S early."""
    _spec()
    key, rng = random.split(rng)
    tau0, x0, y0 = _sample(key)
    params = _MODEL.init(key, tau0, x0, y0)
    A = _ic_stack(_ic0_coeffs())

    opt = soap(learning_rate=1.0, b2=0.99, shampoo_beta=0.95,
               precondition_frequency=PRECOND_EVERY)
    opt_state = opt.init(params)

    def update(params, A, opt_state, key, lr, pav):
        key, sub = random.split(key)
        tau, x, y = _sample(sub)
        (loss, _), grads = jax.value_and_grad(_loss, has_aux=True)(params, A, tau, x, y,
                                                                  jnp.float32(EPS0))
        upd, opt_state = opt.update(grads, opt_state)
        params = jax.tree_util.tree_map(lambda p, u: p + lr * u, params, upd)
        pav = jax.tree_util.tree_map(lambda a, p: a + (1.0 - EMA) * (p - a), pav, params)
        return params, opt_state, key, loss, pav

    @jit
    def run_chunk(params, A, opt_state, key, lr, pav):
        """CHUNK optimizer steps in one dispatch: no Python between steps."""
        def body(_, c):
            return update(c[0], A, c[1], c[2], lr, c[4])
        c = jax.lax.fori_loop(0, CHUNK, body,
                              (params, opt_state, key, jnp.float32(0.0), pav))
        return c

    end_C = jit(_endpoint_C)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    t_begin = time.time()
    span = max(deadline - t_begin, 1.0)
    step_count = 0
    wins = []
    for j in range(NWIN):
        w_end = t_begin + span * float(_SHARE[j])
        w_start = time.time()
        w_span = max(w_end - w_start, 1.0)
        pav = params
        params, opt_state, key, loss, pav = run_chunk(params, A, opt_state, key,
                                                      jnp.float32(LR0), pav)
        step_count += CHUNK
        last = float(loss)
        while time.time() < w_end and step_count < MAX_ITER:
            frac = min(max((time.time() - w_start) / w_span, 0.0), 1.0)
            lr = jnp.asarray(LR0 * (LR1 / LR0) ** frac, dtype=jnp.float32)
            params, opt_state, key, loss, pav = run_chunk(params, A, opt_state, key,
                                                          lr, pav)
            step_count += CHUNK
            last = float(loss)
            if eval_callback is not None:
                eval_callback(pav, step_count, last)
        wins.append({"p": pav, "Ar": A[0], "Ai": A[1]})
        if j < NWIN - 1:
            A = _ic_stack(end_C(pav, A))
            if WARM:
                params = pav
            else:
                key, sub = random.split(key)
                params = _MODEL.init(sub, tau0, x0, y0)
                opt_state = opt.init(params)
    return {"w": wins}, step_count
