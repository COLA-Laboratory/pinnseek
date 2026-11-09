"""b07_w320 with a PER-AXIS Fourier bandwidth (t 12, y 24, x 40) instead of a
uniform 24: b08's free CPU diagnostic shows the NFREQ-48 collapse is entirely a
t-axis aliasing event, and b02's empirical ceiling (grid Nyquist)/3 differs 3.3x
between t (73 grid points) and x (241). Parent: blocks/b07/b07_w320.py."""

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 1
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u",)

# --- Domain ---
T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 1   # r
HAS_SOURCE              = True   # the emission field f drives the whole plume

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi", "y_lo", "y_hi")
PERIODIC_AXES  = ()       # none: Dirichlet u = 0 on all four faces
GEOMETRY_KIND  = "box_dirichlet"

# --- Physical parameters ---
# Residual: u_t + a u_x + b u_y + c u - DXC u_xx - DYC u_yy - f = 0
# a, b (wind), c (wind divergence) and f (emission) are real ERA5/GFAS fields
# on a (t, y, x) grid in [0,1]^3; DXC, DYC are frozen diffusion constants.
DXC = 9.831690e-04              # K T / Lx^2, K = 2e4 m^2/s
DYC = 1.161416e-03              # K T / Ly^2
COEF_FILE = "smoke_coef.npz"    # keys: tn, xn, yn, a, b, c, f, dxc, dyc
GRID_NT, GRID_NY, GRID_NX = 73, 153, 241

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

import math
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax import random
from soap_jax import soap


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ Separable (CP) architecture ══════════
RANK      = 2560
WIDTH     = 320
DEPTH     = 5
OUT_SCALE = 0.05
LR_MAX    = 4.5e-3
MUON_RATIO = 1.0 / 6.0   # Muon's own optimum was 5e-4 against SOAP's 3e-3
LR_MIN    = 1e-5
CHUNK     = 25
SAVE_MARGIN_S = 8.0
MOM       = 0.95      # Muon momentum (Nesterov)
NS_STEPS  = 5         # Newton-Schulz quintic iterations


NFREQ = (12, 24, 40)          # per-axis (t, y, x) Fourier bandwidth
NKNOT = 64                    # uniform cubic-B-spline intervals per axis
NFEAT = tuple(2 * n + 1 + NKNOT + 3 for n in NFREQ)


def _bspline(z):
    """All NKNOT+3 uniform cubic B-splines on [0,1] at z: [N,1] -> [N, NKNOT+3].
    Built as the 4th difference of truncated cubics, so it is exactly C2 — the
    PDE needs a second derivative and kb1 106 shows C1 bases diverge. The
    argument is clipped to the support so nothing leaks into u_yy outside it."""
    arg = NKNOT * z - jnp.arange(NKNOT + 3, dtype=jnp.float32)[None, :] + 3.0
    q = jnp.clip(arg, 0.0, 4.0)
    p = lambda s: jnp.maximum(q - s, 0.0) ** 3
    return (p(0.0) - 4.0 * p(1.0) + 6.0 * p(2.0) - 4.0 * p(3.0) + p(4.0)) / 6.0


def _merged_init(nf):
    """lecun_normal on the MLP block (fan_in = width), zeros on the skip block."""
    def init(key, shape, dtype=jnp.float32):
        kh = nn.initializers.lecun_normal()(key, (shape[0] - nf, shape[1]), dtype)
        return jnp.concatenate([kh, jnp.zeros((nf, shape[1]), dtype)], axis=0)
    return init


class AxisNet(nn.Module):
    """R -> R^rank tanh MLP over one coordinate, on a Fourier-feature embedding."""
    width: int
    depth: int
    rank: int
    nfreq: int

    @nn.compact
    def __call__(self, z):          # z: [N, 1] coordinate in [0, 1]
        k = jnp.arange(1, self.nfreq + 1, dtype=jnp.float32)
        w = 2.0 * jnp.pi * k * z    # [N, NFREQ]
        feat = jnp.concatenate([2.0 * z - 1.0, jnp.cos(w), jnp.sin(w),
                                _bspline(z)], axis=-1)
        h = feat
        for i in range(self.depth):
            inp = h if i == 0 else jnp.concatenate([h, feat], axis=-1)
            h = nn.tanh(nn.Dense(self.width)(inp))
        # + a direct linear path feature -> rank (kb1 082): a well-conditioned
        # 49-dim-per-axis subspace the optimiser can move in without going through
        # five tanh layers, while the MLP still supplies the harmonics.
        # identical function class to b03_skip, but ONE [width+49, rank] kernel
        # instead of two: SOAP pays a single 1024-side eigendecomposition per axis
        # rather than two, which is where b03_skip lost 14% of its steps.
        return nn.Dense(self.rank, kernel_init=_merged_init(
            2 * self.nfreq + 1 + NKNOT + 3))(jnp.concatenate([h, feat], axis=-1))


class SPINN3(nn.Module):
    """u(t,x,y) = sum_r ft_r(t) fy_r(y) fx_r(x); the t and boundary factors make
    u(0,.,.) = 0 and u = 0 on the four Dirichlet faces exact by construction."""
    width: int = WIDTH
    depth: int = DEPTH
    rank: int = RANK

    def setup(self):
        self.bt = AxisNet(self.width, self.depth, self.rank, NFREQ[0], name="axis_t")
        self.by = AxisNet(self.width, self.depth, self.rank, NFREQ[1], name="axis_y")
        self.bx = AxisNet(self.width, self.depth, self.rank, NFREQ[2], name="axis_x")

    def ft(self, t):
        return OUT_SCALE * t * self.bt(t)               # hard zero IC

    def fy(self, y):
        return 4.0 * y * (1.0 - y) * self.by(y)         # hard Dirichlet in y

    def fx(self, x):
        return 4.0 * x * (1.0 - x) * self.bx(x)         # hard Dirichlet in x

    def __call__(self, t, y, x):                        # scattered evaluation
        return jnp.sum(self.ft(t) * self.fy(y) * self.fx(x), axis=-1)


_PREDICT_MODEL = SPINN3()


def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u": [BS]}."""
    t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    outs, n, step = [], X.shape[0], 65536
    for i in range(0, n, step):
        outs.append(_PREDICT_MODEL.apply(params, t[i:i + step],
                                         y[i:i + step], x[i:i + step]))
    return {"u": jnp.concatenate(outs)}


# ══════════ Coefficient fields + separable trilinear interpolation ══════════

def _fields():
    dat = np.load(_locate(COEF_FILE))
    return jnp.stack([jnp.asarray(dat[k], jnp.bfloat16) for k in ("a", "b", "c", "f")])


def _interp_axis(arr, coord, axis):
    """Linear interpolation of `arr` along `axis` at normalised coords `coord`."""
    n = arr.shape[axis]
    g = coord * (n - 1)
    i0 = jnp.clip(jnp.floor(g), 0, n - 2).astype(jnp.int32)
    w = g - i0
    shape = [1] * arr.ndim
    shape[axis] = -1
    lo = jnp.take(arr, i0, axis=axis)
    hi = jnp.take(arr, i0 + 1, axis=axis)
    return lo + (hi - lo) * w.reshape(shape).astype(arr.dtype)


def _coefs(FLD, tc, yc, xc):
    """a, b, c, f on the separable lattice (tc x yc x xc), trilinear."""
    K = _interp_axis(FLD, tc, 1)
    K = _interp_axis(K, yc, 2)
    K = _interp_axis(K, xc, 3)
    return K[0], K[1], K[2], K[3]


# Same physical block scales as the parent, now on 2 nodes / grid cell in ALL
# three axes: every x extent doubles in nodes, so level 0 is still exactly one
# grid cell (2,2,2) and every level above it covers the same physical volume.
# LEVEL_W is sqrt(block volume)/level-0, and all volumes double, so it is unchanged.
LEVELS  = ((2, 2, 2), (2, 4, 4), (2, 8, 8), (4, 16, 16), (6, 38, 40), (8, 76, 120), (24, 152, 240), (144, 304, 480))
LEVEL_W = (1.0, 2.0, 4.0, 11.31, 33.76, 95.5, 330.82, 1620.67)


# Every level is a uniform block average, and block averaging is associative, so
# each level can be built from an already-coarsened one instead of from the full
# 10.5M-point residual. (parent index, block relative to that parent); -1 = res.
# The eight levels below are bit-for-bit the LEVELS above, at one pass over the
# fine array instead of eight (and one broadcast in the backward pass, not eight).
TREE = ((-1, (2, 2, 2)), (0, (1, 2, 2)), (1, (1, 2, 2)), (2, (2, 2, 2)),
        (0, (3, 19, 20)), (0, (4, 38, 60)), (5, (3, 2, 2)), (6, (6, 2, 2)))


def _blockmean(a, bt, by, bx):
    nt, ny, nx = a.shape
    return a.reshape(nt // bt, bt, ny // by, by,
                     nx // bx, bx).mean(axis=(1, 3, 5))


def _multiscale(res):
    """Weighted sum of mean-square block-averaged residuals over LEVELS."""
    lv, tot = [], 0.0
    for par, blk in TREE:
        lv.append(_blockmean(res if par < 0 else lv[par], *blk))
    for b, w in zip(lv, LEVEL_W):
        tot = tot + w * jnp.mean(b ** 2)
    return tot / sum(LEVEL_W)


def _d(fn, z):
    """d/dz of an elementwise-in-z map [N,1] -> [N,r], by one forward-mode JVP."""
    return jax.jvp(fn, (z,), (jnp.ones_like(z),))[1]


# ══════════ Muon: orthogonalised momentum, no eigendecomposition ══════════
# kb1 108 benchmarks curvature-aware optimisers and lands on SOAP (Adam in a
# Kronecker eigenbasis) for stochastic objectives; b03 Finding 2 refuted the
# deterministic members of that family here (GN, L-BFGS) because the objective
# is a fresh random lattice every step, and b05 left one question open: SOAP's
# 2048-side `eigh` is 19% of the step and every cheaper truncation lost.
# Muon answers it — the same whitening intent (equalise the update's singular
# values) obtained by a Newton-Schulz quintic, i.e. three small GEMMs on the
# 212-side, no eigendecomposition and no preconditioner state at all.

_NS_ABC = (3.4445, -4.7750, 2.0315)   # Jordan's quintic coefficients


def _orthogonalise(G):
    """G -> ~U V^T (all singular values driven to ~1) by NS_STEPS quintic
    iterations, then rescaled so the update's RMS is 1 — the same per-element
    scale Adam/SOAP produce, so the LR is directly comparable."""
    a, b, c = _NS_ABC
    X = G.astype(jnp.float32)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.T
    X = X / (jnp.linalg.norm(X) + 1e-7)
    for _ in range(NS_STEPS):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if tr:
        X = X.T
    return X * math.sqrt(max(G.shape[0], G.shape[1]))


def _muon_orth():
    return optax.GradientTransformation(
        lambda params: optax.EmptyState(),
        lambda updates, state, params=None: (
            jax.tree_util.tree_map(_orthogonalise, updates), state),
    )


def _muon(learning_rate):
    """Muon on the [width+NFEAT, rank] output kernels (78% of the parameters and
    the only 2048-sided ones), SOAP on everything else: the remaining
    eigendecompositions are on sides of 84/128/212 and cost almost nothing."""
    muon = optax.chain(optax.trace(decay=MOM, nesterov=True), _muon_orth(),
                       optax.scale(-learning_rate * MUON_RATIO))
    rest = soap(learning_rate=learning_rate, precondition_frequency=40)
    return optax.multi_transform(
        {"muon": muon, "soap": rest},
        lambda p: jax.tree_util.tree_map(
            lambda x: "muon" if (x.ndim == 2 and max(x.shape) >= RANK)
            else "soap", p))


def _make_train(rng_init):
    FLD = _fields()
    _, NT, NY, NX = FLD.shape
    nt, ny, nx = 2 * (NT - 1), 2 * (NY - 1), 2 * (NX - 1)   # 2 nodes / cell everywhere
    model = SPINN3()
    key, rng = random.split(rng_init)
    z0 = jnp.zeros((1, 1))
    params = model.init(key, z0, z0, z0)

    def anti(key, n):
        """n/2 grid cells, 2 nodes each at cell-relative u/2 and 1 - u/2: an
        antithetic pair, so the cell average is exact for the linear part of the
        integrand and the O(h) quadrature error cancels. Measured free on CPU
        against 6 independent lattice draws of b04_hbys: the multi-scale loss'
        quadrature bias drops 1.47e-6 -> 1.05e-6 (-28%) at zero cost. At 2
        nodes/cell the pair lies INSIDE one grid cell, where a,b,c,f are smooth;
        x now qualifies too (at 1 node/cell it would have straddled a boundary
        where the fields are merely C0 and the cancellation fails)."""
        u = random.uniform(key, (n // 2,))
        return (jnp.arange(n) + jnp.stack([u, 1.0 - u], 1).reshape(-1)) / n

    def lattice(key):
        kt, ky, kx = random.split(key, 3)
        tc = anti(kt, nt)
        yc = anti(ky, ny)
        xc = anti(kx, nx)
        return tc, yc, xc

    def loss_fn(p, tc, yc, xc, A, B, C, F):
        tg, yg, xg = tc[:, None], yc[:, None], xc[:, None]
        ft = lambda z: model.apply(p, z, method=SPINN3.ft)
        fy = lambda z: model.apply(p, z, method=SPINN3.fy)
        fx = lambda z: model.apply(p, z, method=SPINN3.fx)
        T0, T1 = ft(tg), _d(ft, tg)
        Y0, Y1 = fy(yg), _d(fy, yg)
        Y2 = _d(lambda z: _d(fy, z), yg)
        X0, X1 = fx(xg), _d(fx, xg)
        X2 = _d(lambda z: _d(fx, z), xg)
        r = RANK
        # bfloat16 operands for the two big contractions (fp32 accumulation): the
        # lattice tensors dominate memory traffic, and 8-bit mantissa noise on terms
        # of O(1) is ~0.25% of the finest-level mean-square residual. Three separate
        # GEMMs avoid materialising the stacked PS buffer.
        # bfloat16 *outputs* as well as operands (fp32 accumulation inside the
        # GEMM). The six lattice tensors and their cotangents are the step's
        # dominant HBM traffic -- a --diag scan measured 22.4 -> 17.5 ms/step
        # from this alone, and forward-only is 5.4 ms, i.e. the backward pass
        # (where the cotangents are the same six tensors) is 3.1x the forward.
        # Each output is widened back to fp32 in the residual expression, which
        # XLA fuses into the elementwise kernel, so only the store is bf16:
        # the added rounding is ~3.5e-3 rms against a residual rms of 0.082,
        # i.e. 0.2% of the finest level's mean square.
        # The three X0 contractions are stacked into one GEMM (-5%).
        c = lambda z: z.astype(jnp.bfloat16)
        f32 = lambda z: z.astype(jnp.float32)
        mm = lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.bfloat16)
        T0c, T1c, Y0c, Y1c, Y2c, X0c = c(T0), c(T1), c(Y0), c(Y1), c(Y2), c(X0)
        P = (T0c[:, None, :] * Y0c[None, :, :]).reshape(-1, r)
        G = mm(P, c(jnp.concatenate([X0, X1, X2], axis=0)).T).reshape(nt, ny, 3, nx)
        u, u_x, u_xx = f32(G[:, :, 0, :]), f32(G[:, :, 1, :]), f32(G[:, :, 2, :])
        S = jnp.concatenate([(T1c[:, None, :] * Y0c[None, :, :]).reshape(-1, r),
                             (T0c[:, None, :] * Y1c[None, :, :]).reshape(-1, r),
                             (T0c[:, None, :] * Y2c[None, :, :]).reshape(-1, r)], axis=0)
        H = mm(S, X0c.T).reshape(3, nt, ny, nx)
        u_t, u_y, u_yy = f32(H[0]), f32(H[1]), f32(H[2])
        res = u_t + A * u_x + B * u_y + C * u - DXC * u_xx - DYC * u_yy - F
        return _multiscale(res)

    opt = optax.inject_hyperparams(_muon)(learning_rate=LR_MAX)
    opt_state = opt.init(params)

    @jax.jit
    def chunk(p, s, key):
        def body(carry, k):
            p, s = carry
            tc, yc, xc = lattice(k)
            A, B, C, F = _coefs(FLD, tc, yc, xc)
            l, g = jax.value_and_grad(loss_fn)(p, tc, yc, xc, A, B, C, F)
            upd, s = opt.update(g, s, p)
            return (optax.apply_updates(p, upd), s), l
        keys = random.split(key, CHUNK)
        (p, s), ls = jax.lax.scan(body, (p, s), keys)
        return p, s, ls[-1]

    return params, opt_state, chunk, rng


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam on the residual-only separable loss over a freshly
    jittered lattice each step; cosine LR decay driven by elapsed wall time."""
    params, opt_state, chunk, key = _make_train(rng)
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    span = max(1e-6, TRAIN_TIME - SAVE_MARGIN_S)

    key, sub = random.split(key)
    params, opt_state, loss = chunk(params, opt_state, sub)   # warmup: JIT + steps
    step_count, last = CHUNK, float(loss)

    while time.time() < deadline:
        frac = min(1.0, max(0.0, (time.time() - _T0) / span))
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * frac))
        opt_state = opt_state._replace(
            hyperparams={**opt_state.hyperparams,
                         "learning_rate": jnp.asarray(lr, jnp.float32)})
        key, sub = random.split(key)
        params, opt_state, loss = chunk(params, opt_state, sub)
        step_count += CHUNK
        last = float(loss)
        if eval_callback is not None and (step_count // CHUNK) % 200 == 0:
            eval_callback(params, step_count, last)

    return params, step_count
