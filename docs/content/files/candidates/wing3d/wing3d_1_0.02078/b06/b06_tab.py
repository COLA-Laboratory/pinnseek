"""Pseudo-time PINN (kb1-073) + wall-distance features (kb1-096) + kb1-021.
Vs b04_relr: the frozen mask net's 2-jet (value, d_j, d_jj) is tabulated once
on the fixed collocation grid, so each step's nested jvps run over the core
alone — exactly the same derivatives, ~1.4x more steps.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 4
INPUT_NAMES  = ("x", "y", "z")
FIELD_NAMES  = ("u", "v", "w", "p")

# --- Domain (farfield box; wing cut out via level set in collocation.npy) ---
X_MIN, X_MAX = -1.5, 3.5
Y_MIN, Y_MAX = -2.0, 2.0
Z_MIN, Z_MAX = 0.0, 3.5
HAS_TIME = False

# --- Equation structure ---
TIME_DERIV_ORDER        = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 4   # ru, rv, rw, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("wing", "farfield", "outlet", "symmetry")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "external_wing_half_model"

# --- Physical parameters ---
RE      = 1000.0       # 3D finite wing, Re=1000
NU      = 1.0 / RE
U_INF   = 1.0          # freestream speed (v_inf = w_inf = 0)
AOA_DEG = 7.0          # angle of attack; wing rotated -AOA about (0.5, 0)

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
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


def _locate(name: str) -> Path:
    base = _os.environ.get("FORGE_TASK_DIR")
    if base:
        cand = Path(base) / name
        if cand.is_file():
            return cand
    here = Path.cwd()
    for b in [here, *here.parents]:
        cand = b / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


# ══════════ hyperparameters ══════════
WIDTH, DEPTH = 128, 6
MODIFIED_MLP = True                # kb1-050 gated encoder backbone
MU = 24.0                          # 1/dt, pseudo-time forcing (kb1-073)
MU_P = 24.0                        # artificial-compressibility counterpart
OUTER_K = 50                       # inner steps between q_n snapshots
BS_A, BS_B, BS_C = 640, 512, 384   # near-wall shell / near-wing box / rest
W_A, W_B, W_C = 0.60, 0.28, 0.12   # PDE-loss tier weights
BS_FAR, BS_OUT, BS_SYM = 384, 256, 384
W_FAR, W_OUT, W_SYM = 2.0, 1.0, 2.0
LR_PEAK, LR_FINAL, WARM_FRAC = 1.5e-3, 1.5e-5, 0.03
S_REF = 0.5                        # magnitude-normalisation scale (kb1-021)
D_NEAR = 0.15                      # near-wall shell thickness
BOX = (-0.6, 2.8, 0.8, 2.2)        # enlarged near-wing box (x0, x1, |y|, z1)
SAVE_MARGIN_S = 16.0
LOG_EVERY = 500
LOG_PATH = "blocks/b06/b06_tab_log.json"
TAB_CHUNK = 131072                 # points per tabulation chunk

# mask net: 3 warped wall distances tanh(d / delta), fitted then frozen.
# Channel 0 (delta = MASK_DELTA) is the hard no-slip mask; all three are
# also handed to the core as input features (kb1-096 boundary-layer coords).
MASK_W, MASK_D, MASK_YS, MASK_DELTA, MASK_STEPS = 96, 4, 0.05, 0.05, 8000
C_FIX = 1.6                        # mask thickness MASK_DELTA / C_FIX
MASK_DELTAS = (0.05, 0.15, 0.6)
MASK_CW = (1.0, 0.5, 0.5)          # per-channel weight in the fit loss
N_MASK = len(MASK_DELTAS)

# input warp: near-wing window -> O(1); y compressed by arcsinh (thin wing)
XC, XS = 1.0, 1.2
YS = 0.2
ZC, ZS = 0.9, 0.9

N_FEAT = INPUT_DIM + N_MASK        # core input: warped coords + mask channels


# ══════════ networks ══════════

class MaskNet(nn.Module):
    """Smooth wing indicator, N_MASK channels: tanh(dist / delta_k), each 0
    on the wing surface and -> 1 away from it, at three wall scales."""
    width: int = MASK_W
    depth: int = MASK_D
    ys: float = MASK_YS

    @nn.compact
    def __call__(self, q):
        h = jnp.concatenate([(q[:, 0:1] - XC) / XS,
                             jnp.arcsinh(q[:, 1:2] / self.ys),
                             (q[:, 2:3] - ZC) / ZS], axis=-1)
        ki = nn.initializers.glorot_normal()
        for _ in range(self.depth):
            h = jnp.tanh(nn.Dense(self.width, kernel_init=ki)(h))
        return nn.Dense(N_MASK, kernel_init=ki)(h)      # [..., N_MASK]


class Core(nn.Module):
    """tanh MLP on the feature vector s = (warped x,y,z, N_MASK wall-distance
    channels); head zero-initialised so the masked field starts as uniform
    flow. Same parameters as the parent's (xyz, mfeat) core."""
    width: int = WIDTH
    depth: int = DEPTH
    modified: bool = MODIFIED_MLP

    @nn.compact
    def __call__(self, s):
        kinit = nn.initializers.glorot_normal()
        h = s
        if self.modified:
            U = jnp.tanh(nn.Dense(self.width, kernel_init=kinit)(h))
            V = jnp.tanh(nn.Dense(self.width, kernel_init=kinit)(h))
            H = jnp.tanh(nn.Dense(self.width, kernel_init=kinit)(h))
            for _ in range(self.depth - 1):
                Z = jnp.tanh(nn.Dense(self.width, kernel_init=kinit)(H))
                H = (1.0 - Z) * U + Z * V
            h = H
        else:
            for _ in range(self.depth):
                h = jnp.tanh(nn.Dense(self.width, kernel_init=kinit)(h))
        return nn.Dense(OUTPUT_DIM, kernel_init=nn.initializers.zeros,
                        bias_init=nn.initializers.zeros)(h)


_MASK = MaskNet()
_CORE = Core()
_BASE = jnp.array([U_INF, 0.0, 0.0])


# ── boundary-layer thickness control (b03) ──────────────────────────────
# The fitted channel 0 is m0 = tanh(d/MASK_DELTA), so arctanh(m0) = d/MASK_DELTA
# recovers the wall distance exactly; tanh(c * arctanh(m0)) = tanh(d/(MASK_DELTA/c))
# is the same mask with thickness MASK_DELTA/c. c = 1 reproduces the parent.

def _meff(m0, c):
    """Re-scale the hard mask to an effective wall thickness MASK_DELTA / c."""
    m = jnp.clip(m0, -0.9, 1.0 - 1e-6)
    t = 0.5 * jnp.log((1.0 + m) / (1.0 - m))
    return jnp.tanh(c * t)

def _assemble(m, o):
    """m: [...] mask, o: [..., 4] raw core output -> (u, v, w, p)."""
    return (m * (U_INF + o[..., 0]), m * o[..., 1], m * o[..., 2], o[..., 3])


_T0F = lambda m: _meff(m, C_FIX)          # scalar mask warp and its 2 derivs
_T1F = jax.grad(_T0F)
_T2F = jax.grad(_T1F)


# ══════════ feature map and its tabulated jets ══════════

def _feat(mask_p, X):
    """[N, 3] -> [N, N_FEAT]: warped coords + the frozen mask channels."""
    mf = _MASK.apply(mask_p, X)
    return jnp.concatenate([(X[:, 0:1] - XC) / XS,
                            jnp.arcsinh(X[:, 1:2] / YS),
                            (X[:, 2:3] - ZC) / ZS, mf], axis=-1)


def _tab(mask_p, X, order, chunk=TAB_CHUNK):
    """Tabulate s = feat(x) and its axis derivatives d_j s, d_jj s.

    The mask net is frozen and the collocation set is fixed, so these are
    constants of the training loop: caching them makes each residual step a
    nested jvp over the core alone, with identical derivative values (the
    2-jet of feat along each axis is all the chain rule needs)."""
    f = lambda q: _feat(mask_p, q)

    @jit
    def go(Xc):
        s0 = f(Xc)
        d1, d2 = [], []
        for j in range(INPUT_DIM):
            E = jnp.zeros_like(Xc).at[:, j].set(1.0)
            if order >= 2:
                g = lambda t: jax.jvp(f, (t,), (E,))
                (_, a), (_, b) = jax.jvp(g, (Xc,), (E,))
                d1.append(a)
                d2.append(b)
            else:
                _, a = jax.jvp(f, (Xc,), (E,))
                d1.append(a)
        out = [s0]
        if order >= 1:
            out.append(jnp.stack(d1, axis=1))           # [N, 3, N_FEAT]
        if order >= 2:
            out.append(jnp.stack(d2, axis=1))
        return tuple(out)

    parts = [go(X[i:i + chunk]) for i in range(0, len(X), chunk)]
    return tuple(jnp.concatenate([p[k] for p in parts], axis=0)
                 for k in range(len(parts[0])))


# ══════════ public contract ══════════

def predict_fn(params, X):
    """X: [BS, 3] with columns (x, y, z). Returns {"u","v","w","p"} each [BS]."""
    s = _feat(params["mask"], X)
    o = _CORE.apply(params["main"], s)
    u, v, w, p = _assemble(_meff(s[..., INPUT_DIM], C_FIX), o)
    return {"u": u, "v": v, "w": w, "p": p}


# ══════════ data ══════════

def _build_dataset():
    """Fixed fluid grid points split into a near-wall shell, a near-wing box
    and the rest (near-wing-biased minibatching), plus the mask-fit sets."""
    col = np.load(_locate("collocation.npy"))
    xyz = np.ascontiguousarray(col[:, :3], dtype=np.float32)
    d = -np.asarray(col[:, 3], dtype=np.float32)      # distance to the wing
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    tol = 1e-5
    on_inlet = np.abs(x - X_MIN) < tol
    on_out = np.abs(x - X_MAX) < tol
    on_lat = (np.abs(y - Y_MIN) < tol) | (np.abs(y - Y_MAX) < tol)
    on_top = np.abs(z - Z_MAX) < tol
    on_sym = np.abs(z - Z_MIN) < tol
    interior = ~(on_inlet | on_out | on_lat | on_top | on_sym)
    near = interior & (d < D_NEAR)
    box = (interior & ~near & (x >= BOX[0]) & (x <= BOX[1])
           & (np.abs(y) <= BOX[2]) & (z <= BOX[3]))
    rest = interior & ~near & ~box
    surf = np.load(_locate("wing_surface.npy")).astype(np.float32)
    rs = np.random.RandomState(0)
    m_hi = d >= 0.6
    sub = rs.choice(int(m_hi.sum()), 150000, replace=False)
    mask_sets = dict(surf=surf,
                     mA=xyz[d < 0.15], dA=d[d < 0.15],
                     mB=xyz[(d >= 0.15) & (d < 0.6)], dB=d[(d >= 0.15) & (d < 0.6)],
                     mC=xyz[m_hi][sub], dC=d[m_hi][sub])
    sets = dict(A=xyz[near], B=xyz[box], C=xyz[rest],
                far=xyz[(on_inlet | on_lat | on_top) & ~on_out],
                out=xyz[on_out], sym=xyz[on_sym])
    return ({k: jnp.asarray(v) for k, v in sets.items()},
            {k: jnp.asarray(v) for k, v in mask_sets.items()}, surf)


def _fit_mask(rng, M):
    """Regress the N_MASK channels tanh(dist/delta_k); surface pins them to 0."""
    p = _MASK.init(rng, jnp.zeros([1, INPUT_DIM]))
    tx = optax.adam(optax.exponential_decay(3e-3, MASK_STEPS, 0.02))
    st = tx.init(p)
    dl = jnp.asarray(MASK_DELTAS)[None, :]
    cw = jnp.asarray(MASK_CW)[None, :]
    tA, tB, tC = (jnp.tanh(M[k][:, None] / dl) for k in ("dA", "dB", "dC"))
    nS, nA, nB, nC = (len(M["surf"]), len(M["mA"]), len(M["mB"]), len(M["mC"]))

    def loss(p, k):
        k = random.split(k, 4)
        ia = random.choice(k[1], nA, (4096,))
        ib = random.choice(k[2], nB, (2048,))
        ic = random.choice(k[3], nC, (1024,))
        ls = jnp.mean(cw * _MASK.apply(
            p, M["surf"][random.choice(k[0], nS, (4096,))]) ** 2)
        lv = (jnp.sum(cw * (_MASK.apply(p, M["mA"][ia]) - tA[ia]) ** 2)
              + jnp.sum(cw * (_MASK.apply(p, M["mB"][ib]) - tB[ib]) ** 2)
              + jnp.sum(cw * (_MASK.apply(p, M["mC"][ic]) - tC[ic]) ** 2)) / 7168.0
        return 3.0 * ls + lv

    @jit
    def step(p, st, k):
        l, g = jax.value_and_grad(loss)(p, k)
        u, st = tx.update(g, st)
        return optax.apply_updates(p, u), st, l

    k = rng
    for _ in range(MASK_STEPS):
        k, sk = random.split(k)
        p, st, l = step(p, st, sk)
    return p, float(l)


def _lr_at(prog):
    if prog < WARM_FRAC:
        return LR_PEAK * (0.1 + 0.9 * prog / WARM_FRAC)
    q = (prog - WARM_FRAC) / (1.0 - WARM_FRAC)
    return LR_PEAK * (LR_FINAL / LR_PEAK) ** q


def _make_train(rng_init):
    S, M, surf_np = _build_dataset()
    sizes = {k: int(len(v)) for k, v in S.items()}
    k_mask, k_core, rng = random.split(rng_init, 3)
    t_mask = time.time()
    mask_p, mask_loss = _fit_mask(k_mask, M)
    t_mask = time.time() - t_mask

    t_tab = time.time()
    TAB = {k: _tab(mask_p, S[k], 2) for k in ("A", "B", "C")}
    TAB["sym"] = _tab(mask_p, S["sym"], 1)
    for k in ("far", "out"):
        TAB[k] = _tab(mask_p, S[k], 0)
    t_tab = time.time() - t_tab

    params = _CORE.init(k_core, jnp.zeros([1, N_FEAT]))
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    ZERO, ONE = jnp.float32(0.0), jnp.float32(1.0)

    def core_at(pf, s):
        return _CORE.apply(unravel(pf), s[None, :])[0]                 # [4]

    def field(pf, s0):
        """Assembled (u, v, w, p) at a tabulated point."""
        o = core_at(pf, s0)
        m = _T0F(s0[INPUT_DIM])
        return jnp.concatenate([m * (_BASE + o[:3]), o[3:4]])

    def pde_point(pf, pf_prev, s0, s1, s2):
        """Steady NS residual + kb1-073 pseudo-time forcing -mu*(q - q_n).

        The core's directional 2-jet along axis j is taken by autodiff along
        the tabulated path s(t) = s0 + t*d_j s + t^2/2 * d_jj s, which is the
        exact 2-jet of the composition core(feat(x)) along that axis; the
        frozen mask's own jets enter analytically through the product rule.
        Returns [rc^2, rm^2, rc_pts^2, rm_pts^2, wg*rc_pts^2, wg*rm_pts^2, wg]."""
        def axis(j):
            a, b = s1[j], s2[j]
            path = lambda t: core_at(pf, s0 + t * a + (0.5 * t * t) * b)
            g = lambda t: jax.jvp(path, (t,), (ONE,))
            (val, d1), (_, d2) = jax.jvp(g, (ZERO,), (ONE,))
            return val, d1, d2

        o, dox, doxx = axis(0)
        _, doy, doyy = axis(1)
        _, doz, dozz = axis(2)
        D1 = jnp.stack([dox, doy, doz])                # [3, 4] d_j o
        D2 = jnp.stack([doxx, doyy, dozz])             # [3, 4] d_jj o

        m0 = s0[INPUT_DIM]
        m, t1, t2 = _T0F(m0), _T1F(m0), _T2F(m0)
        g0 = s1[:, INPUT_DIM]                          # [3] d_j m0
        dm = t1 * g0                                   # [3] d_j meff
        ddm = t2 * g0 ** 2 + t1 * s2[:, INPUT_DIM]     # [3] d_jj meff

        q = _BASE + o[:3]                              # U_inf + perturbation
        val = jnp.concatenate([m * q, o[3:4]])
        duj = dm[:, None] * q[None, :] + m * D1[:, :3]       # [3, 3] d_j u_i
        gp = D1[:, 3]                                        # [3] grad p
        lap = jnp.sum(ddm[:, None] * q[None, :]
                      + 2.0 * dm[:, None] * D1[:, :3]
                      + m * D2[:, :3], axis=0)               # [3] lap u_i

        u, v, w = val[0], val[1], val[2]
        adv = u * duj[0] + v * duj[1] + w * duj[2]           # [3]
        rc = duj[0, 0] + duj[1, 1] + duj[2, 2]
        rm = adv + gp - NU * lap                             # [3]
        prev = field(pf_prev, s0)
        dq = val - prev
        rc_t = rc + MU_P * dq[3]
        rm_t = rm + MU * dq[:3]
        sc = jnp.sqrt(jnp.sum(adv ** 2) + jnp.sum(gp ** 2)
                      + NU ** 2 * jnp.sum(lap ** 2))
        wg = jax.lax.stop_gradient(S_REF / (sc + S_REF))
        rm2, rmt2 = jnp.sum(rm ** 2), jnp.sum(rm_t ** 2)
        return jnp.array([rc ** 2, rm2, rc_t ** 2, rmt2,
                          wg * (rc_t ** 2), wg * rmt2, wg])

    def sym_point(pf, s0, s1):
        """Symmetry plane: w = 0 and du/dz = dv/dz = 0."""
        a = s1[2]
        path = lambda t: core_at(pf, s0 + t * a)
        o, d1 = jax.jvp(path, (ZERO,), (ONE,))
        m0 = s0[INPUT_DIM]
        m, dm = _T0F(m0), _T1F(m0) * s1[2, INPUT_DIM]
        q = _BASE + o[:3]
        dzu = dm * q[0] + m * d1[0]
        dzv = dm * q[1] + m * d1[1]
        return (m * q[2]) ** 2 + dzu ** 2 + dzv ** 2

    pde_batch = vmap(pde_point, in_axes=(None, None, 0, 0, 0))
    sym_batch = vmap(sym_point, in_axes=(None, 0, 0))
    fwd_batch = vmap(field, in_axes=(None, 0))

    def loss_aux(pf, pf_prev, bA, bB, bC, bf, bo, bs):
        pA = jnp.mean(pde_batch(pf, pf_prev, *bA), axis=0)
        pB = jnp.mean(pde_batch(pf, pf_prev, *bB), axis=0)
        pC = jnp.mean(pde_batch(pf, pf_prev, *bC), axis=0)
        ff = fwd_batch(pf, bf)
        l_far = jnp.mean((ff[:, 0] - U_INF) ** 2 + ff[:, 1] ** 2 + ff[:, 2] ** 2)
        fo = fwd_batch(pf, bo)
        l_out = jnp.mean(fo[:, 3] ** 2)
        l_sym = jnp.mean(sym_batch(pf, *bs))
        pde = (W_A * jnp.sum(pA[4:6]) / pA[6] + W_B * jnp.sum(pB[4:6]) / pB[6]
               + W_C * jnp.sum(pC[4:6]) / pC[6])
        total = pde + W_FAR * l_far + W_OUT * l_out + W_SYM * l_sym
        aux = jnp.concatenate([pA[:4], pB[:4], pC[:4],
                               jnp.stack([l_far, l_out, l_sym])])
        return total, aux

    lg = jax.value_and_grad(loss_aux, has_aux=True)
    tx = optax.scale_by_adam()
    opt_state = tx.init(flat)

    def _take(key, name, n, bs):
        idx = random.choice(key, n, (bs,))
        return tuple(a[idx] for a in TAB[name])

    @jit
    def update(pf, pf_prev, opt_state, lr, keys):
        b = (_take(keys[0], "A", sizes["A"], BS_A),
             _take(keys[1], "B", sizes["B"], BS_B),
             _take(keys[2], "C", sizes["C"], BS_C),
             _take(keys[3], "far", sizes["far"], BS_FAR)[0],
             _take(keys[4], "out", sizes["out"], BS_OUT)[0],
             _take(keys[5], "sym", sizes["sym"], BS_SYM))
        (loss, aux), g = lg(pf, pf_prev, *b)
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        upd, opt_state = tx.update(g, opt_state)
        upd = jnp.where(jnp.isfinite(upd), upd, 0.0)
        return pf - lr * upd, opt_state, loss, aux

    @jit
    def probe(pf, mp, q):
        return vmap(field, in_axes=(None, 0))(pf, _feat(mp, q))

    info = {"mask_fit_s": round(t_mask, 1), "mask_loss": mask_loss,
            "tab_s": round(t_tab, 1), "sizes": sizes,
            "mu": MU, "outer_k": OUTER_K}
    return flat, opt_state, update, probe, unravel, mask_p, rng, info, surf_np


def _write_log(payload):
    try:
        import json
        p = Path(LOG_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam, wall-clock LR decay, kb1-073 pseudo-time outer
    loop (q_n refreshed every OUTER_K steps). Returns (params, step_count)."""
    (flat, opt_state, update, probe, unravel,
     mask_p, rng_l, info, surf_np) = _make_train(rng)

    keys = random.split(rng_l, 7)
    rng_l = keys[0]
    lr = jnp.float32(_lr_at(0.0))
    prev = flat
    flat, opt_state, loss, aux = update(flat, prev, opt_state, lr, keys[1:])
    step, hist = 1, []
    t_start = time.time()
    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    span = max(deadline - t_start, 1.0)

    while step < MAX_ITER:
        now = time.time()
        if now > deadline:
            break
        if step % 25 == 1:
            lr = jnp.float32(_lr_at(min((now - t_start) / span, 1.0)))
        if step % OUTER_K == 0:
            prev = flat                      # pseudo-time snapshot q_n
        keys = random.split(rng_l, 7)
        rng_l = keys[0]
        flat, opt_state, loss, aux = update(flat, prev, opt_state, lr, keys[1:])
        step += 1
        if step % LOG_EVERY == 0:
            hist.append([step, round(now - t_start, 1), float(lr), float(loss)]
                        + [float(a) for a in aux])
            if eval_callback is not None:
                eval_callback({"main": unravel(flat), "mask": mask_p},
                              step, float(loss))

    params = {"main": unravel(flat), "mask": mask_p}
    # ── free diagnostics: no-slip violation + a mid-span V slice ──
    try:
        gx = np.linspace(-0.3, 2.5, 71, dtype=np.float32)
        gy = np.linspace(-0.5, 0.5, 41, dtype=np.float32)
        XX, YY = np.meshgrid(gx, gy, indexing="ij")
        q = np.stack([XX.ravel(), YY.ravel(),
                      np.full(XX.size, 0.5, np.float32)], axis=1)
        o = np.asarray(probe(flat, mask_p, jnp.asarray(q)))
        Vsl = np.sqrt(o[:, 0] ** 2 + o[:, 1] ** 2 + o[:, 2] ** 2)
        ow = np.asarray(probe(flat, mask_p, jnp.asarray(surf_np[::19])))
        slip = float(np.sqrt(np.mean(ow[:, 0] ** 2 + ow[:, 1] ** 2 + ow[:, 2] ** 2)))
        info.update({"steps": step, "slip_rms": slip,
                     "hist_cols": ["step", "t", "lr", "loss",
                                   "A_c", "A_m", "A_ct", "A_mt",
                                   "B_c", "B_m", "B_ct", "B_mt",
                                   "C_c", "C_m", "C_ct", "C_mt",
                                   "far", "out", "sym"],
                     "hist": hist,
                     "slice_z0.5": {"x": gx.tolist(), "y": gy.tolist(),
                                    "V": np.round(Vsl, 4).reshape(71, 41).tolist()}})
        _write_log(info)
    except Exception:
        _write_log({"steps": step, "hist": hist, **info})
    return params, step
