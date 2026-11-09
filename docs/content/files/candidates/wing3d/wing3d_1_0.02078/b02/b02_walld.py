"""Pseudo-time PINN (kb1-073) + wall-distance input features (kb1-096).
Vs b02_wallc (96x4 mask net): the frozen mask net shrinks again to 64x3 —
its second derivatives enter the residual, and smoother won at wallc.
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
MU = 30.0                          # 1/dt, pseudo-time forcing (kb1-073)
MU_P = 30.0                        # artificial-compressibility counterpart
OUTER_K = 50                       # inner steps between q_n snapshots
BS_A, BS_B, BS_C = 640, 512, 384   # near-wall shell / near-wing box / rest
W_A, W_B, W_C = 0.45, 0.35, 0.20   # PDE-loss tier weights
BS_FAR, BS_OUT, BS_SYM = 384, 256, 384
W_FAR, W_OUT, W_SYM = 2.0, 1.0, 2.0
LR_PEAK, LR_FINAL, WARM_FRAC = 1.5e-3, 1.5e-5, 0.03
D_NEAR = 0.15                      # near-wall shell thickness
BOX = (-0.6, 2.8, 0.8, 2.2)        # enlarged near-wing box (x0, x1, |y|, z1)
SAVE_MARGIN_S = 16.0
LOG_EVERY = 500
LOG_PATH = "blocks/b02/b02_walld_log.json"

# mask net: 3 warped wall distances tanh(d / delta), fitted then frozen.
# Channel 0 (delta = MASK_DELTA) is the hard no-slip mask; all three are
# also handed to the core as input features (kb1-096 boundary-layer coords).
MASK_W, MASK_D, MASK_YS, MASK_DELTA, MASK_STEPS = 64, 3, 0.05, 0.05, 8000
MASK_DELTAS = (0.05, 0.15, 0.6)
MASK_CW = (1.0, 0.5, 0.5)          # per-channel weight in the fit loss
N_MASK = len(MASK_DELTAS)

# input warp: near-wing window -> O(1); y compressed by arcsinh (thin wing)
XC, XS = 1.0, 1.2
YS = 0.2
ZC, ZS = 0.9, 0.9


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
    """tanh MLP on warped coords + the N_MASK wall-distance channels; head
    zero-initialised so the masked field starts as uniform flow."""
    width: int = WIDTH
    depth: int = DEPTH
    modified: bool = MODIFIED_MLP

    @nn.compact
    def __call__(self, xyz, mfeat):
        h = jnp.concatenate([(xyz[:, 0:1] - XC) / XS,
                             jnp.arcsinh(xyz[:, 1:2] / YS),
                             (xyz[:, 2:3] - ZC) / ZS, mfeat], axis=-1)
        kinit = nn.initializers.glorot_normal()
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


def _assemble(m, o):
    """m: [...] mask, o: [..., 4] raw core output -> (u, v, w, p)."""
    return (m * (U_INF + o[..., 0]), m * o[..., 1], m * o[..., 2], o[..., 3])


# ══════════ public contract ══════════

def predict_fn(params, X):
    """X: [BS, 3] with columns (x, y, z). Returns {"u","v","w","p"} each [BS]."""
    mf = _MASK.apply(params["mask"], X)
    o = _CORE.apply(params["main"], X, mf)
    u, v, w, p = _assemble(mf[..., 0], o)
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

    params = _CORE.init(k_core, jnp.zeros([1, INPUT_DIM]),
                        jnp.zeros([1, N_MASK]))
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    E = jnp.eye(3)

    def f_one(pf, q):
        qq = q[None, :]
        mf = _MASK.apply(mask_p, qq)
        o = _CORE.apply(unravel(pf), qq, mf)[0]
        return jnp.stack(_assemble(mf[0, 0], o))                # [4]

    def pde_point(pf, pf_prev, q):
        """Steady NS residual + kb1-073 pseudo-time forcing -mu*(q - q_n).
        All derivatives by autodiff: one nested jvp per axis gives f, d_i f
        and d_ii f in 3 passes. Returns [rc^2, rm^2, rc_pts^2, rm_pts^2]."""
        fx = lambda t: f_one(pf, t)

        def axis(i):
            e = E[i]
            g = lambda t: jax.jvp(fx, (t,), (e,))
            (val, d1), (_, d2) = jax.jvp(g, (q,), (e,))
            return val, d1, d2

        val, dx, dxx = axis(0)
        _, dy, dyy = axis(1)
        _, dz, dzz = axis(2)
        u, v, w = val[0], val[1], val[2]
        lap = dxx + dyy + dzz
        adv = u * dx + v * dy + w * dz
        rc = dx[0] + dy[1] + dz[2]
        rmx = adv[0] + dx[3] - NU * lap[0]
        rmy = adv[1] + dy[3] - NU * lap[1]
        rmz = adv[2] + dz[3] - NU * lap[2]
        prev = f_one(pf_prev, q)
        dq = val - prev
        rc_t = rc + MU_P * dq[3]
        rmx_t = rmx + MU * dq[0]
        rmy_t = rmy + MU * dq[1]
        rmz_t = rmz + MU * dq[2]
        return jnp.array([rc ** 2, rmx ** 2 + rmy ** 2 + rmz ** 2,
                          rc_t ** 2, rmx_t ** 2 + rmy_t ** 2 + rmz_t ** 2])

    def sym_point(pf, q):
        """Symmetry plane: w = 0 and du/dz = dv/dz = 0."""
        fx = lambda t: f_one(pf, t)
        val, dz = jax.jvp(fx, (q,), (E[2],))
        return val[2] ** 2 + dz[0] ** 2 + dz[1] ** 2

    pde_batch = vmap(pde_point, in_axes=(None, None, 0))
    sym_batch = vmap(sym_point, in_axes=(None, 0))
    fwd_batch = vmap(f_one, in_axes=(None, 0))

    def loss_aux(pf, pf_prev, bA, bB, bC, bf, bo, bs):
        pA = jnp.mean(pde_batch(pf, pf_prev, bA), axis=0)
        pB = jnp.mean(pde_batch(pf, pf_prev, bB), axis=0)
        pC = jnp.mean(pde_batch(pf, pf_prev, bC), axis=0)
        ff = fwd_batch(pf, bf)
        l_far = jnp.mean((ff[:, 0] - U_INF) ** 2 + ff[:, 1] ** 2 + ff[:, 2] ** 2)
        fo = fwd_batch(pf, bo)
        l_out = jnp.mean(fo[:, 3] ** 2)
        l_sym = jnp.mean(sym_batch(pf, bs))
        pde = (W_A * jnp.sum(pA[2:]) + W_B * jnp.sum(pB[2:])
               + W_C * jnp.sum(pC[2:]))
        total = pde + W_FAR * l_far + W_OUT * l_out + W_SYM * l_sym
        aux = jnp.concatenate([pA, pB, pC, jnp.stack([l_far, l_out, l_sym])])
        return total, aux

    lg = jax.value_and_grad(loss_aux, has_aux=True)
    tx = optax.scale_by_adam()
    opt_state = tx.init(flat)

    @jit
    def update(pf, pf_prev, opt_state, lr, keys):
        b = (S["A"][random.choice(keys[0], sizes["A"], (BS_A,))],
             S["B"][random.choice(keys[1], sizes["B"], (BS_B,))],
             S["C"][random.choice(keys[2], sizes["C"], (BS_C,))],
             S["far"][random.choice(keys[3], sizes["far"], (BS_FAR,))],
             S["out"][random.choice(keys[4], sizes["out"], (BS_OUT,))],
             S["sym"][random.choice(keys[5], sizes["sym"], (BS_SYM,))])
        (loss, aux), g = lg(pf, pf_prev, *b)
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        upd, opt_state = tx.update(g, opt_state)
        upd = jnp.where(jnp.isfinite(upd), upd, 0.0)
        return pf - lr * upd, opt_state, loss, aux

    @jit
    def probe(pf, q):
        return fwd_batch(pf, q)

    info = {"mask_fit_s": round(t_mask, 1), "mask_loss": mask_loss,
            "sizes": sizes, "mu": MU, "outer_k": OUTER_K}
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
        o = np.asarray(probe(flat, jnp.asarray(q)))
        Vsl = np.sqrt(o[:, 0] ** 2 + o[:, 1] ** 2 + o[:, 2] ** 2)
        ow = np.asarray(probe(flat, jnp.asarray(surf_np[::19])))
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
