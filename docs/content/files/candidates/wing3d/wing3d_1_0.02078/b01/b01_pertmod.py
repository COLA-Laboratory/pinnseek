"""Freestream-perturbation PINN: zero-init head on uniform flow, warped inputs.
Vs baseline: tiered near-wing collocation sampling, weighted wing no-slip,
wall-clock LR decay, 6x128 kb1-050 modified MLP, Laplacian via nested jvp.
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
BS_A, BS_B, BS_C = 640, 512, 384   # near-wall shell / near-wing box / rest
W_A, W_B, W_C = 0.45, 0.35, 0.20   # PDE-loss tier weights
BS_WING, BS_FAR, BS_OUT, BS_SYM = 1024, 384, 256, 384
W_WING, W_FAR, W_OUT, W_SYM = 10.0, 1.0, 1.0, 1.0
LR_PEAK, LR_FINAL, WARM_FRAC = 1.5e-3, 1.5e-5, 0.03
D_NEAR = 0.15                      # near-wall shell thickness
BOX = (-0.6, 2.8, 0.8, 2.2)        # enlarged near-wing box (x0, x1, |y|, z1)
SAVE_MARGIN_S = 16.0
LOG_EVERY = 500
LOG_PATH = "blocks/b01/b01_pertmod_log.json"

# input warp: near-wing window -> O(1); y compressed by arcsinh (thin wing)
XC, XS = 1.0, 1.2
YS = 0.2
ZC, ZS = 0.9, 0.9


# ══════════ network: freestream + zero-init perturbation head ══════════

class PINN(nn.Module):
    """tanh MLP on warped coords; output head zero-initialised so the field
    starts as exact uniform flow (u, v, w, p) = (U_INF, 0, 0, 0)."""
    width: int = WIDTH
    depth: int = DEPTH
    modified: bool = MODIFIED_MLP

    @nn.compact
    def __call__(self, xyz):
        h = jnp.concatenate([(xyz[:, 0:1] - XC) / XS,
                             jnp.arcsinh(xyz[:, 1:2] / YS),
                             (xyz[:, 2:3] - ZC) / ZS], axis=-1)
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
        out = nn.Dense(OUTPUT_DIM, kernel_init=nn.initializers.zeros,
                       bias_init=nn.initializers.zeros)(h)
        return out + jnp.array([U_INF, 0.0, 0.0, 0.0])


# ══════════ data: fixed fluid grid, split into sampling tiers ══════════

def _build_dataset():
    """Fixed fluid grid points, partitioned into a near-wall shell
    (distance < D_NEAR), a near-wing box and the rest, so minibatches can be
    drawn with a near-wing bias — the score lives in the near-wing window."""
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
    sets = dict(A=xyz[near], B=xyz[box], C=xyz[rest],
                wing=np.load(_locate("wing_surface.npy")).astype(np.float32),
                far=xyz[(on_inlet | on_lat | on_top) & ~on_out],
                out=xyz[on_out], sym=xyz[on_sym])
    return {k: jnp.asarray(v) for k, v in sets.items()}


# ══════════ public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: [BS, 3] with columns (x, y, z). Returns {"u","v","w","p"} each [BS]."""
    out = _PREDICT_MODEL.apply(params, X)   # → [BS, 4]
    return {"u": out[:, 0], "v": out[:, 1], "w": out[:, 2], "p": out[:, 3]}


def _lr_at(prog):
    if prog < WARM_FRAC:
        return LR_PEAK * (0.1 + 0.9 * prog / WARM_FRAC)
    q = (prog - WARM_FRAC) / (1.0 - WARM_FRAC)
    return LR_PEAK * (LR_FINAL / LR_PEAK) ** q


def _make_train(rng_init):
    S = _build_dataset()
    sizes = {k: int(len(v)) for k, v in S.items()}
    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    E = jnp.eye(3)

    def f_one(pf, q):
        return model.apply(unravel(pf), q[None, :])[0]          # [4]

    def pde_point(pf, q):
        """Continuity + momentum residuals, all derivatives by autodiff.
        One nested jvp per axis yields f, d_i f and d_ii f in 3 passes."""
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
        return jnp.array([rc ** 2, rmx ** 2 + rmy ** 2 + rmz ** 2])

    def sym_point(pf, q):
        """Symmetry plane: w = 0 and du/dz = dv/dz = 0."""
        fx = lambda t: f_one(pf, t)
        val, dz = jax.jvp(fx, (q,), (E[2],))
        return val[2] ** 2 + dz[0] ** 2 + dz[1] ** 2

    pde_batch = vmap(pde_point, in_axes=(None, 0))
    sym_batch = vmap(sym_point, in_axes=(None, 0))
    fwd_batch = vmap(f_one, in_axes=(None, 0))

    def loss_aux(pf, bA, bB, bC, bw, bf, bo, bs):
        pA = jnp.mean(pde_batch(pf, bA), axis=0)
        pB = jnp.mean(pde_batch(pf, bB), axis=0)
        pC = jnp.mean(pde_batch(pf, bC), axis=0)
        fw = fwd_batch(pf, bw)
        l_wing = jnp.mean(fw[:, 0] ** 2 + fw[:, 1] ** 2 + fw[:, 2] ** 2)
        ff = fwd_batch(pf, bf)
        l_far = jnp.mean((ff[:, 0] - U_INF) ** 2 + ff[:, 1] ** 2 + ff[:, 2] ** 2)
        fo = fwd_batch(pf, bo)
        l_out = jnp.mean(fo[:, 3] ** 2)
        l_sym = jnp.mean(sym_batch(pf, bs))
        pde = W_A * jnp.sum(pA) + W_B * jnp.sum(pB) + W_C * jnp.sum(pC)
        total = (pde + W_WING * l_wing + W_FAR * l_far
                 + W_OUT * l_out + W_SYM * l_sym)
        aux = jnp.concatenate([pA, pB, pC,
                               jnp.stack([l_wing, l_far, l_out, l_sym])])
        return total, aux

    lg = jax.value_and_grad(loss_aux, has_aux=True)
    tx = optax.scale_by_adam()
    opt_state = tx.init(flat)

    @jit
    def update(pf, opt_state, lr, keys):
        b = (S["A"][random.choice(keys[0], sizes["A"], (BS_A,))],
             S["B"][random.choice(keys[1], sizes["B"], (BS_B,))],
             S["C"][random.choice(keys[2], sizes["C"], (BS_C,))],
             S["wing"][random.choice(keys[3], sizes["wing"], (BS_WING,))],
             S["far"][random.choice(keys[4], sizes["far"], (BS_FAR,))],
             S["out"][random.choice(keys[5], sizes["out"], (BS_OUT,))],
             S["sym"][random.choice(keys[6], sizes["sym"], (BS_SYM,))])
        (loss, aux), g = lg(pf, *b)
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        upd, opt_state = tx.update(g, opt_state)
        upd = jnp.where(jnp.isfinite(upd), upd, 0.0)
        return pf - lr * upd, opt_state, loss, aux

    @jit
    def probe(pf, q):
        return fwd_batch(pf, q)

    return flat, opt_state, update, probe, unravel, rng, sizes


def _write_log(payload):
    try:
        import json
        p = Path(LOG_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam with a wall-clock LR decay; the whole process
    finishes within TRAIN_TIME s of _T0. Returns (params_pytree, step_count)."""
    flat, opt_state, update, probe, unravel, rng_l, sizes = _make_train(rng)

    keys = random.split(rng_l, 8)
    rng_l = keys[0]
    lr = jnp.float32(_lr_at(0.0))
    flat, opt_state, loss, aux = update(flat, opt_state, lr, keys[1:])
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
        keys = random.split(rng_l, 8)
        rng_l = keys[0]
        flat, opt_state, loss, aux = update(flat, opt_state, lr, keys[1:])
        step += 1
        if step % LOG_EVERY == 0:
            hist.append([step, round(now - t_start, 1), float(lr), float(loss)]
                        + [float(a) for a in aux])
            if eval_callback is not None:
                eval_callback(unravel(flat), step, float(loss))

    params = unravel(flat)
    # ── free diagnostics: no-slip violation + a mid-span V slice ──
    try:
        gx = np.linspace(-0.3, 2.5, 71, dtype=np.float32)
        gy = np.linspace(-0.5, 0.5, 41, dtype=np.float32)
        XX, YY = np.meshgrid(gx, gy, indexing="ij")
        q = np.stack([XX.ravel(), YY.ravel(),
                      np.full(XX.size, 0.5, np.float32)], axis=1)
        o = np.asarray(probe(flat, jnp.asarray(q)))
        Vsl = np.sqrt(o[:, 0] ** 2 + o[:, 1] ** 2 + o[:, 2] ** 2)
        wing = np.load(_locate("wing_surface.npy")).astype(np.float32)[::19]
        ow = np.asarray(probe(flat, jnp.asarray(wing)))
        slip = float(np.sqrt(np.mean(ow[:, 0] ** 2 + ow[:, 1] ** 2 + ow[:, 2] ** 2)))
        _write_log({"steps": step, "sizes": sizes, "slip_rms": slip,
                    "hist_cols": ["step", "t", "lr", "loss", "A_c", "A_m",
                                  "B_c", "B_m", "C_c", "C_m",
                                  "wing", "far", "out", "sym"],
                    "hist": hist,
                    "slice_z0.5": {"x": gx.tolist(), "y": gy.tolist(),
                                   "V": np.round(Vsl, 4).reshape(71, 41).tolist()}})
    except Exception:
        _write_log({"steps": step, "hist": hist})
    return params, step
