"""Tucker (full-core) separable weak-form stream-function PINN, looser causal gate.
Change vs b04_tuck.py: EPS0 11370 -> 3000. b04_tuck stalled its causal front at
t~0.25 with a total slice loss 10x the CP parent's, so exp(-eps*cum) shut the
window; matching eps*cum to the parent's converged value needs eps ~ 1e3.
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
K_SPACE   = 16        # exact-periodic Fourier modes per spatial axis (input embedding)
K_TIME    = 12        # half-period Fourier features on t (period 2*T_WIN -> no aliasing)
WIDTH     = 64        # per-axis body width
DEPTH     = 4         # per-axis body depth
RT, RX, RY = 64, 64, 64       # Tucker ranks: per-axis feature counts; core is [RT,RX,RY]
NT, NX, NY = 64, 128, 128     # collocation lattice (tensor product)
K_IC      = 16        # Fourier truncation of the hard initial-condition ansatz
T_WIN     = 0.75      # modelled/trained time window [0, T_WIN]
PSI_SC    = 0.05      # stream-function output scale (psi_rms of the IC ~ 0.041)

K_T       = 64        # test-function cutoff: enforce the equation for |k| <= K_T
S_M       = 8.0       # velocity-residual normalisation (|u_t| ~ 5, weak norm ~ 10)
NORM_EXPO = 1.5       # residual measured in H^-NORM_EXPO (1 = velocity/momentum norm)

LR0, LR1  = 5e-3, 2e-3        # exponential decay across the wall-clock budget
PRECOND_EVERY = 20            # SOAP eigenbasis refresh interval
EPS0      = 3000.0           # causal-weight tightness (Wang et al. 2024)
EPS_GROW  = 1.02
EPS_MAX   = 1e5
W_FLOOR   = 3e-3              # never fully ignore a time slice (1.0 = no gating)
CHUNK     = 100               # steps between host syncs
SAVE_MARGIN_S = 8.0

# spatial derivative orders of psi_0 the weak residual needs
_IC_ORDERS = ((0, 1), (1, 0), (2, 0), (0, 2))


# ══════════ Hard initial condition: exact Fourier stream function ══════════

_IC: dict[str, Any] = {}


def _ic_coeffs():
    """psi_0 spectral coefficients and the spatial derivatives the residual needs."""
    if "k" not in _IC:
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
        ks = np.concatenate([np.arange(0, K_IC + 1), np.arange(-K_IC, 0)]).astype(np.float64)
        C = Ph[np.ix_(idx, idx)]
        kx = (2j * np.pi * ks)[:, None]
        ky = (2j * np.pi * ks)[None, :]
        Ar, Ai = [], []
        for dx, dy in _IC_ORDERS:
            D = C * kx ** dx * ky ** dy
            Ar.append(D.real.astype(np.float32))
            Ai.append(D.imag.astype(np.float32))
        _IC["k"] = jnp.asarray(ks.astype(np.float32))
        _IC["Ar"] = jnp.asarray(np.stack(Ar))
        _IC["Ai"] = jnp.asarray(np.stack(Ai))
    return _IC["k"], _IC["Ar"], _IC["Ai"]


def _trig(z, k):
    a = 2.0 * jnp.pi * z * k          # z: [N,1], k: [M]  ->  [N, M]
    return jnp.cos(a), jnp.sin(a)


def _ic_grid(x, y):
    """[4, NX, NY]: the psi_0 derivatives listed in _IC_ORDERS on the lattice."""
    k, Ar, Ai = _ic_coeffs()
    Cx, Sx = _trig(x, k)
    Cy, Sy = _trig(y, k)
    return (jnp.einsum("ia,dab,jb->dij", Cx, Ar, Cy)
            - jnp.einsum("ia,dab,jb->dij", Sx, Ar, Sy)
            - jnp.einsum("ia,dab,jb->dij", Sx, Ai, Cy)
            - jnp.einsum("ia,dab,jb->dij", Cx, Ai, Sy))


def _ic_scatter(x, y):
    """[4, N]: same derivatives at scattered points (only psi_y, psi_x are used)."""
    k, Ar, Ai = _ic_coeffs()
    Cx, Sx = _trig(x, k)
    Cy, Sy = _trig(y, k)
    return (jnp.einsum("na,dab,nb->dn", Cx, Ar, Cy)
            - jnp.einsum("na,dab,nb->dn", Sx, Ar, Sy)
            - jnp.einsum("na,dab,nb->dn", Sx, Ai, Cy)
            - jnp.einsum("na,dab,nb->dn", Cx, Ai, Sy))


# ══════════ Separable backbone with exact periodic embedding ══════════

class Body(nn.Module):
    """One per-axis MLP R -> R^rank. Space axes get an exactly periodic cos/sin
    embedding, so psi and all its derivatives are periodic to machine precision."""
    kind: str
    rank: int

    @nn.compact
    def __call__(self, z):                                  # z: [N, 1] -> [N, rank]
        if self.kind == "t":
            m = jnp.arange(1, K_TIME + 1, dtype=jnp.float32)
            a = jnp.pi * z * m / T_WIN                      # period 2*T_WIN
            h = jnp.concatenate([2.0 * z / T_WIN - 1.0, jnp.cos(a), jnp.sin(a)], axis=-1)
        else:
            k = jnp.arange(1, K_SPACE + 1, dtype=jnp.float32)
            a = 2.0 * jnp.pi * z * k
            h = jnp.concatenate([jnp.cos(a), jnp.sin(a)], axis=-1)
        for _ in range(DEPTH):
            h = jnp.tanh(nn.Dense(WIDTH)(h))
        return nn.Dense(self.rank)(h)


class PSINET(nn.Module):
    def setup(self):
        self.bt = Body("t", RT, name="bt")
        self.bx = Body("x", RX, name="bx")
        self.by = Body("y", RY, name="by")
        self.c = self.param("core", nn.initializers.normal(1.0), (RT, RX, RY))

    def __call__(self, t, x, y):
        return self.bt(t), self.bx(x), self.by(y), self.c

    def one(self, z, kind):
        return {"t": self.bt, "x": self.bx, "y": self.by}[kind](z)


_MODEL = PSINET()
_SC = PSI_SC / (RT * RX * RY) ** 0.5


def _core(params):
    p = params["params"] if "params" in params else params
    return p["core"]


def _dstack(f, z, n):
    """[f(z), f'(z), ..., f^(n)(z)] for a row-wise f: [N,1] -> [N,R], by nested
    forward-mode AD (the per-axis bodies are tiny, so the 2^n cost is negligible)."""
    fs = [f]
    for _ in range(n):
        g = fs[-1]
        fs.append(lambda zz, g=g: jax.jvp(g, (zz,), (jnp.ones_like(zz),))[1])
    return [fi(z) for fi in fs]


def _tuck(A, fx, fy):
    """A: [NT, RX, RY], fx: [NX, RX], fy: [NY, RY] -> [NT, NX, NY]."""
    return jnp.einsum("ijc,kc->ijk", jnp.einsum("ibc,jb->ijc", A, fx), fy)


# ══════════ Fields: u, v, w, w_t on the lattice (all by AD) ══════════

def _fields(params, t, x, y):
    b = lambda kind: (lambda z: _MODEL.apply(params, z, kind, method=PSINET.one))
    Ft = _dstack(b("t"), t, 1)
    Fx = _dstack(b("x"), x, 2)
    Fy = _dstack(b("y"), y, 2)
    C = _core(params)
    A0 = jnp.einsum("ia,abc->ibc", Ft[0], C)
    A1 = jnp.einsum("ia,abc->ibc", Ft[1], C)
    ic = _ic_grid(x, y)                                     # [4, NX, NY]
    ramp = (t[:, 0] / T_WIN)[:, None, None]
    dramp = jnp.full_like(ramp, 1.0 / T_WIN)

    psi_y = ic[0][None] + _SC * ramp * _tuck(A0, Fx[0], Fy[1])
    psi_x = ic[1][None] + _SC * ramp * _tuck(A0, Fx[1], Fy[0])
    b20, b02 = _tuck(A0, Fx[2], Fy[0]), _tuck(A0, Fx[0], Fy[2])
    d20, d02 = _tuck(A1, Fx[2], Fy[0]), _tuck(A1, Fx[0], Fy[2])
    psi_xx = ic[2][None] + _SC * ramp * b20
    psi_yy = ic[3][None] + _SC * ramp * b02
    u, v = psi_y, -psi_x
    w = -(psi_xx + psi_yy)
    w_t = -_SC * (dramp * (b20 + b02) + ramp * (d20 + d02))  # psi_0 is t-independent
    return u, v, w, w_t


# ══════════ Weak (Fourier-Galerkin) vorticity residual ══════════

_SPEC: dict[str, Any] = {}


def _spec():
    """Test-mode wavenumbers, the |k| <= K_T mask and the 1/(2 pi |k|) weight that
    turns the vorticity residual into a velocity (momentum) residual."""
    if "kx" not in _SPEC:
        kk = np.fft.fftfreq(NX, 1.0 / NX)
        ll = np.fft.fftfreq(NY, 1.0 / NY)
        KXn, KYn = np.meshgrid(kk, ll, indexing="ij")
        K2 = KXn ** 2 + KYn ** 2
        mask = (K2 <= K_T ** 2 + 1e-9) & (K2 > 0)
        wgt = np.where(mask, 1.0 / (2 * np.pi * np.sqrt(np.where(K2 > 0, K2, 1.0))) ** NORM_EXPO, 0.0)
        _SPEC["kx"] = jnp.asarray(2 * np.pi * KXn)
        _SPEC["ky"] = jnp.asarray(2 * np.pi * KYn)
        _SPEC["k2"] = jnp.asarray(4 * np.pi ** 2 * K2)
        _SPEC["w"] = jnp.asarray(wgt.astype(np.float32))
    return _SPEC["kx"], _SPEC["ky"], _SPEC["k2"], _SPEC["w"]


def _slice_loss(params, t, x, y):
    """Residual of the vorticity equation tested against exp(-2 pi i k.x), |k| <= K_T.
    Using div(u)=0, the advection term is integrated by parts as div(u w), so no
    derivative above second order of psi ever appears."""
    kx, ky, k2, wgt = _spec()
    u, v, w, w_t = _fields(params, t, x, y)
    n2 = NX * NY
    F = lambda a: jnp.fft.fft2(a, axes=(1, 2)) / n2
    fy = FORCE_AMP * FORCE_K * jnp.pi * jnp.cos(FORCE_K * jnp.pi * y[:, 0])   # curl of f
    Fh = jnp.fft.fft2(jnp.broadcast_to(fy[None, None, :], (1, NX, NY)), axes=(1, 2)) / n2
    R = (F(w_t) + 1j * (kx * F(u * w) + ky * F(v * w)) + NU * k2 * F(w) + Fh)
    R2 = R.real ** 2 + R.imag ** 2          # |.|**2 directly: abs() has no gradient at 0
    return jnp.sum(R2 * (wgt / S_M) ** 2, axis=(1, 2))             # [NT]


def _loss(params, t, x, y, eps):
    Lr = _slice_loss(params, t, x, y)
    cum = jnp.concatenate([jnp.zeros((1,), Lr.dtype), jnp.cumsum(Lr)[:-1]])
    wgt = jax.lax.stop_gradient(jnp.maximum(jnp.exp(-eps * cum), W_FLOOR))
    return jnp.mean(wgt * Lr), (Lr, wgt)


def _sample(key):
    k1, k2, k3 = random.split(key, 3)
    t = ((jnp.arange(NT) + random.uniform(k1, (NT,))) * (T_WIN / NT))[:, None]
    x = (jnp.arange(NX) + random.uniform(k2, ())) / NX          # uniform grid (FFT)
    y = (jnp.arange(NY) + random.uniform(k3, ())) / NY
    return t, x[:, None], y[:, None]


# ══════════ Public contract ══════════

def predict_fn(params, X):
    """X: [BS, 3] columns (t, x, y) -> {"u","v","p"}. u = psi_y, v = -psi_x; the
    stream-function form carries no pressure unknown, so "p" is zeros (unscored)."""
    _ic_coeffs()
    C = jnp.asarray(_core(params))
    tcol = np.asarray(X[:, 0], dtype=np.float64)
    tu, inv = np.unique(tcol, return_inverse=True)
    b = lambda kind: (lambda z: _MODEL.apply(params, z, kind, method=PSINET.one))
    u_out = np.zeros(X.shape[0], np.float64)
    v_out = np.zeros(X.shape[0], np.float64)
    groups = ([(np.where(inv == g)[0], tu[g]) for g in range(len(tu))] if len(tu) <= 128
              else [(np.arange(s, min(s + 4096, X.shape[0])), None)
                    for s in range(0, X.shape[0], 4096)])
    for idx, tval in groups:
        Xc = X[jnp.asarray(idx)]
        t, x, y = Xc[:, 0:1], Xc[:, 1:2], Xc[:, 2:3]
        Fx, Fx1 = _dstack(b("x"), x, 1)
        Fy, Fy1 = _dstack(b("y"), y, 1)
        icd = _ic_scatter(x, y)
        if tval is not None:                       # all points share one t: contract once
            Ft = b("t")(jnp.full((1, 1), tval, jnp.float32))[0]      # [RT]
            M = jnp.einsum("a,abc->bc", Ft, C)                        # [RX, RY]
            psi_x = jnp.einsum("nb,bc,nc->n", Fx1, M, Fy)
            psi_y = jnp.einsum("nb,bc,nc->n", Fx, M, Fy1)
            ramp = float(tval) / T_WIN
        else:
            Ft = b("t")(t)                                            # [n, RT]
            M = jnp.einsum("na,abc->nbc", Ft, C)
            psi_x = jnp.einsum("nb,nbc,nc->n", Fx1, M, Fy)
            psi_y = jnp.einsum("nb,nbc,nc->n", Fx, M, Fy1)
            ramp = t[:, 0] / T_WIN
        u_out[idx] = np.asarray(icd[0] + _SC * ramp * psi_y, np.float64)
        v_out[idx] = np.asarray(-(icd[1] + _SC * ramp * psi_x), np.float64)
    u = jnp.asarray(u_out, jnp.float32)
    return {"u": u, "v": jnp.asarray(v_out, jnp.float32), "p": jnp.zeros_like(u)}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """SOAP on the causally weighted Fourier-Galerkin vorticity residual of the
    hard-IC separable stream-function ansatz. Wall-deadline anchored at _T0; the
    loop stops SAVE_MARGIN_S early so params can be pickled."""
    _ic_coeffs()
    _spec()
    key, rng = random.split(rng)
    t0, x0, y0 = _sample(key)
    params = _MODEL.init(key, t0, x0, y0)

    opt = soap(learning_rate=1.0, precondition_frequency=PRECOND_EVERY)
    opt_state = opt.init(params)

    def update(params, opt_state, key, eps, lr):
        key, sub = random.split(key)
        t, x, y = _sample(sub)
        (loss, (Lr, wgt)), grads = jax.value_and_grad(_loss, has_aux=True)(params, t, x, y, eps)
        upd, opt_state = opt.update(grads, opt_state)
        params = jax.tree_util.tree_map(lambda p, u: p + lr * u, params, upd)
        eps = jnp.clip(jnp.where(wgt.min() > 0.99, eps * EPS_GROW, eps), EPS0, EPS_MAX)
        return params, opt_state, key, eps, loss

    @jit
    def run_chunk(params, opt_state, key, eps, lr):
        """CHUNK optimizer steps in one dispatch: no Python between steps."""
        def body(_, c):
            return update(c[0], c[1], c[2], c[3], lr)
        return jax.lax.fori_loop(0, CHUNK, body,
                                 (params, opt_state, key, eps, jnp.float32(0.0)))

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    span = max(deadline - _T0, 1.0)
    eps = jnp.asarray(EPS0)

    # warmup: compiles the whole chunk; its CHUNK steps are real (counted)
    params, opt_state, key, eps, loss = run_chunk(params, opt_state, key, eps,
                                                  jnp.float32(LR0))
    step_count = CHUNK
    last = float(loss)

    while time.time() < deadline and step_count < MAX_ITER:
        frac = min(max((time.time() - _T0) / span, 0.0), 1.0)
        lr = jnp.asarray(LR0 * (LR1 / LR0) ** frac, dtype=jnp.float32)
        params, opt_state, key, eps, loss = run_chunk(params, opt_state, key, eps, lr)
        step_count += CHUNK
        last = float(loss)
        if eval_callback is not None:
            eval_callback(params, step_count, last)

    return params, step_count
