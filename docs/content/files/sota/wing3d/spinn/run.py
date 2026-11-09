"""SPINN on steady 3D flow over a finite wing — their helmholtz3d recipe, retargeted.

SPINN publishes no Navier-Stokes in three space dimensions and nothing around a
body. Its only **3D steady** example is `helmholtz3d.py`, which is where the
hyperparameters and the training loop come from; the multi-output mechanism is
`SPINN3d`'s own (`out_dim` rank blocks). NATIVE = False.

Taken unmodified from reference/SPINN:

  * the `SPINN3d` body (modified-MLP per axis, rank-r outer product, out_dim
    block split) — reproduced in `SPINN3dDiag` below with one addition, the
    diagonal contraction described under "the wing" below,
  * `hvp_fwdfwd` from networks/hessian_vector_products.py and `jvp`, used as
    helmholtz3d.py uses them,
  * `update_model`, Adam at a constant rate, per-axis collocation, and
    helmholtz3d.py's unweighted `residual + boundary` sum.

Hyperparameters are helmholtz3d.py's own defaults: modified_mlp, n_layers=3,
features=128, r=128, Adam(1e-3), 50000 epochs. `out_dim = 4` because this
problem carries (u, v, w, p). Adam runs at a constant learning rate with no
schedule tied to the step count, so extra epochs are simply more training and
filling the budget is the right mapping (GUIDE.md rule 5).

**The task's collocation file is already a tensor grid**, which is what makes
this port possible at all: `task/collocation.npy` is 144 x 115 x 101 minus the
1480 points inside the wing (1671080 of 1672560). SPINN evaluates on exactly
such an outer product, so the axes are taken straight from the file — every
other 3D port in this comparison had to invent its own grid — and the residual
is averaged over an in-fluid mask built from the missing points. The axes are
taken at stride 2, giving 72 x 58 x 51, which puts the per-axis count near the
`nc = 64` their own 3D examples use.

What this problem needs that a Helmholtz box does not have:

  * **four boundary kinds instead of one.** helmholtz3d matches `u = 0` on the
    six faces. Here the inlet, the lateral faces and the top carry the
    freestream `u = 1, v = w = 0`; the outlet carries `p = 0`; the symmetry
    plane `z = 0` carries `w = 0` and `du/dz = dv/dz = 0`; and the wing surface
    is no-slip. All but the last are faces of the grid, so they stay separable.

  * **the wing surface is scattered.** A separable network has no notion of a
    scattered point: its output is the outer product of three axis factors. So
    the module gains a `diag=True` path contracting the three factors
    pointwise, `einsum('fn,fn,fn->n')`, which is **algebraically identical** to
    the diagonal of their `einsum('fxy,fz->xyz')` at `N` cost instead of `N^3`
    (checked to 4.7e-10 on the poissoncg port, where the same path is used).
    `diag` is a call-time keyword, so the parameter tree is `SPINN3d`'s exactly.
    A random 4096-point subset of the 77256 surface points is drawn each time
    the loss is built, which keeps that term the same size as the others.

**Coordinates are rescaled to [-1,1] on all three axes, and the residual
carries the chain-rule factors.** The box is
`[-1.5, 3.5] x [-2, 2] x [0, 3.5]`, which is neither their `[-1, 1]^3` nor
anything like it. The map is `X = 2(x - x_lo)/(x_hi - x_lo) - 1` per axis, so
`d/dx = SX d/dX` with `SX = 2/5`, `SY = 2/4`, `SZ = 2/3.5`; second derivatives
carry the square. They are applied explicitly in the residual and nowhere else,
and `predict` maps the scoring points the same way. This is a change of
variables, not a change of method — the same one the bfs and cyl ports make.

Reading the score: predicting zero gives exactly 1.0 and the freestream
`u = 1, v = w = 0` everywhere gives 0.2116 — that is the anchor to beat.
`problem.md` states an **error floor of 0.0071**, so a score below about 0.01 is
fitting discretisation error rather than physics.

**Published settings vs. the file's argparse defaults (2026-08-26 SPINN review).**
The numbers above are `helmholtz3d.py`'s *argparse defaults*. They are not what the
authors ran: `scripts/helmholtz3d_spinn.sh` and the paper's appendix ("three body networks of 4 hidden
layers with 64/32 hidden feature/output size each", Adam 0.001, 50,000
iterations) specifies
`--n_layers=4 --features=64 --r=32 --nc=16`. So the published body network is 4 layers of 64 with rank 32 on a 16-point
axis, against the 3 layers of 128 with rank 128 on 64 points used here.
The larger, default-flag configuration is kept, for two reasons that are worth
stating rather than assuming. It is still the repository's own setting, and it
is the one that flatters the method here — measured at the same wall clock,
seed 1: poissoncg scores 0.0205 on the defaults (22300 epochs, 59.0 s) and 0.0554 on the published flags (20000 epochs, 55.5 s); naca 0.4821 vs about 0.55 interpolated to its budget; kdv, where only `n_layers` differs, 0.6736 vs 0.6779 — inside seed noise. Switching to the published
flags would lower this baseline's score, i.e. manufacture a hero win, which
GUIDE.md's rule 4 exists to prevent. The published values are recorded in
`info` as `published_flags`.
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).parent
METHOD_LIBS = "spinn"
NATIVE = False
SOURCE = "reference/SPINN (SPINN3d + helmholtz3d.py training loop), their only 3D steady example"
SPINN_ROOT = "~/reference/SPINN"
TASK = "~/wing3d/task"

STRIDE = 2
FEATURES = 128
N_LAYERS = 3
RANK = 128
OUT_DIM = 4
LR = 1e-3
MLP = "modified_mlp"
PUBLISHED_EPOCHS = 50000

NU = 1e-3
U_INF = 1.0
XL, XU = -1.5, 3.5
YL, YU = -2.0, 2.0
ZL, ZU = 0.0, 3.5
N_WING = 4096              # surface points per loss build

SX, SY, SZ = 2.0 / (XU - XL), 2.0 / (YU - YL), 2.0 / (ZU - ZL)


def _to_net(v, lo, hi):
    return 2.0 * (v - lo) / (hi - lo) - 1.0


def _grid():
    """(x_axis, y_axis, z_axis, mask) from the shipped collocation file.

    The file is a tensor grid with the wing's interior removed, so the axes are
    its own unique values and the mask marks which grid points it contains.
    """
    col = np.load(f"{TASK}/collocation.npy")
    xs, ix = np.unique(col[:, 0], return_inverse=True)
    ys, iy = np.unique(col[:, 1], return_inverse=True)
    zs, iz = np.unique(col[:, 2], return_inverse=True)
    full = np.zeros((len(xs), len(ys), len(zs)), dtype=np.float32)
    full[ix, iy, iz] = 1.0
    s = slice(None, None, STRIDE)
    return xs[s], ys[s], zs[s], full[s, s, s]


def train(*, steps: int, seed: int, gpu: str) -> dict:
    if SPINN_ROOT not in sys.path:
        sys.path.insert(0, SPINN_ROOT)

    import jax
    import jax.numpy as jnp
    import optax
    import flax.linen as nn
    from jax import jvp
    from networks.hessian_vector_products import hvp_fwdfwd
    from utils.training_utils import update_model

    class SPINN3dDiag(nn.Module):
        """SPINN3d verbatim, plus a pointwise (diagonal) contraction path.

        One difference from `SPINN3d` that matters only if someone changes a
        constant: their class applies the `pos_enc` sin/cos lift to the second
        and third arguments when `pos_enc != 0`, and that branch is **not**
        reproduced here. This case passes `pos_enc = 0`, so the two are
        identical as called; the assert below keeps it that way.
        """
        features: Sequence[int]
        r: int
        out_dim: int
        pos_enc: int
        mlp: str

        @nn.compact
        def __call__(self, x, y, z, diag: bool = False):
            assert self.pos_enc == 0, "pos_enc is not implemented in this subclass"
            inputs, outputs, xy, pred = [x, y, z], [], [], []
            init = nn.initializers.glorot_normal()
            if self.mlp == 'mlp':
                for X in inputs:
                    for fs in self.features[:-1]:
                        X = nn.Dense(fs, kernel_init=init)(X)
                        X = nn.activation.tanh(X)
                    X = nn.Dense(self.r * self.out_dim, kernel_init=init)(X)
                    outputs += [jnp.transpose(X, (1, 0))]
            else:
                for X in inputs:
                    U = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                    V = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                    Hh = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                    for fs in self.features[:-1]:
                        Z = nn.Dense(fs, kernel_init=init)(Hh)
                        Z = nn.activation.tanh(Z)
                        Hh = (jnp.ones_like(Z) - Z) * U + Z * V
                    Hh = nn.Dense(self.r * self.out_dim, kernel_init=init)(Hh)
                    outputs += [jnp.transpose(Hh, (1, 0))]
            for i in range(self.out_dim):
                a = outputs[0][self.r * i:self.r * (i + 1)]
                b = outputs[1][self.r * i:self.r * (i + 1)]
                c = outputs[2][self.r * i:self.r * (i + 1)]
                if diag:                       # [sota patch] scattered points
                    pred += [jnp.einsum('fn, fn, fn->n', a, b, c)]
                else:
                    xy += [jnp.einsum('fx, fy->fxy', a, b)]
                    pred += [jnp.einsum('fxy, fz->xyz', xy[i], c)]
            return pred[0] if len(pred) == 1 else pred

    gx, gy, gz, gmask = _grid()
    x_ax = jnp.asarray(_to_net(gx, XL, XU)[:, None].astype(np.float32))
    y_ax = jnp.asarray(_to_net(gy, YL, YU)[:, None].astype(np.float32))
    z_ax = jnp.asarray(_to_net(gz, ZL, ZU)[:, None].astype(np.float32))
    mask = jnp.asarray(gmask)
    n_kept = float(np.sum(gmask))

    wing_np = np.load(f"{TASK}/wing_surface.npy").astype(np.float32)

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key, 2)
    model = SPINN3dDiag(tuple([FEATURES] * N_LAYERS), RANK, OUT_DIM, 0, MLP)
    params = model.init(subkey, x_ax, y_ax, z_ax)
    apply_fn = jax.jit(model.apply, static_argnames=("diag",))

    lo, hi = jnp.array([[-1.0]]), jnp.array([[1.0]])

    def wing_batch(key):
        idx = np.asarray(jax.random.choice(key, len(wing_np), (N_WING,), replace=False))
        w = wing_np[idx]
        return (jnp.asarray(_to_net(w[:, 0], XL, XU)[:, None]),
                jnp.asarray(_to_net(w[:, 1], YL, YU)[:, None]),
                jnp.asarray(_to_net(w[:, 2], ZL, ZU)[:, None]))

    @partial(jax.jit, static_argnums=(0,))
    def apply_model_spinn(apply_fn, params, wx, wy, wz):
        def residual_loss(params, x, y, z):
            vx, vy, vz = jnp.ones(x.shape), jnp.ones(y.shape), jnp.ones(z.shape)
            u, v, w, p = apply_fn(params, x, y, z)

            def d1(i, ax, vec):
                return jvp(lambda q: apply_fn(params,
                                              q if ax == 0 else x,
                                              q if ax == 1 else y,
                                              q if ax == 2 else z)[i],
                           ((x, y, z)[ax],), (vec,))[1]

            def d2(i, ax, vec):
                return hvp_fwdfwd(lambda q: apply_fn(params,
                                                     q if ax == 0 else x,
                                                     q if ax == 1 else y,
                                                     q if ax == 2 else z)[i],
                                  ((x, y, z)[ax],), (vec,))

            S = (SX, SY, SZ)
            vs = (vx, vy, vz)
            g = [[S[a] * d1(i, a, vs[a]) for a in range(3)] for i in range(4)]
            lap = [sum(S[a] ** 2 * d2(i, a, vs[a]) for a in range(3)) for i in range(3)]

            adv = lambda gi: u * gi[0] + v * gi[1] + w * gi[2]
            ru = adv(g[0]) + g[3][0] - NU * lap[0]
            rv = adv(g[1]) + g[3][1] - NU * lap[1]
            rw = adv(g[2]) + g[3][2] - NU * lap[2]
            rc = g[0][0] + g[1][1] + g[2][2]
            r2 = ru ** 2 + rv ** 2 + rw ** 2 + rc ** 2
            # [sota patch] the wing's interior is dropped from the average
            return jnp.sum(r2 * mask) / n_kept

        def boundary_loss(params, x, y, z, wx, wy, wz):
            """[sota patch] theirs is `mean(u**2)` on six Dirichlet faces; this
            task has four different conditions on five faces plus a body."""
            loss = 0.
            # freestream: inlet x = XL, lateral y = YL/YU, top z = ZU
            for args in ((lo, y, z), (x, lo, z), (x, hi, z), (x, y, hi)):
                u, v, w, _ = apply_fn(params, *args)
                loss += (jnp.mean((u - U_INF) ** 2) + jnp.mean(v ** 2)
                         + jnp.mean(w ** 2))
            # outlet x = XU: p = 0
            _, _, _, p = apply_fn(params, hi, y, z)
            loss += jnp.mean(p ** 2)
            # symmetry plane z = ZL: w = 0 and du/dz = dv/dz = 0
            u_s, v_s, w_s, _ = apply_fn(params, x, y, lo)
            dz = jvp(lambda q: apply_fn(params, x, y, q), (lo,), (jnp.ones(lo.shape),))[1]
            loss += (jnp.mean(w_s ** 2) + jnp.mean((SZ * dz[0]) ** 2)
                     + jnp.mean((SZ * dz[1]) ** 2))
            # wing surface: scattered, so the diagonal path
            u_w, v_w, w_w, _ = apply_fn(params, wx, wy, wz, diag=True)
            loss += (jnp.mean(u_w ** 2) + jnp.mean(v_w ** 2) + jnp.mean(w_w ** 2))
            return loss

        loss_fn = lambda params: (residual_loss(params, x_ax, y_ax, z_ax)
                                  + boundary_loss(params, x_ax, y_ax, z_ax, wx, wy, wz))
        return jax.value_and_grad(loss_fn)(params)

    optim = optax.adam(learning_rate=LR)
    state = optim.init(params)
    key, subkey = jax.random.split(key)
    wb = wing_batch(subkey)

    for e in range(1, int(steps) + 1):
        if e % 100 == 0:                       # theirs: resample every 100
            key, subkey = jax.random.split(key)
            wb = wing_batch(subkey)
        loss, gradient = apply_model_spinn(apply_fn, params, *wb)
        params, state = update_model(optim, gradient, params, state)
        if e % 10000 == 0:
            print(f'Epoch: {e}/{steps} --> total loss: {loss:.8f}')

    def predict(prm, X):
        """X is (N,3) as (x, y, z). The scored set is a scattered subset of the
        reference cloud, so it goes through the diagonal path in chunks."""
        Xn = np.asarray(X, dtype=np.float32)
        out = {"u": [], "v": [], "w": []}
        for k in range(0, len(Xn), 16384):
            b = Xn[k:k + 16384]
            u, v, w, _ = apply_fn(prm,
                                  jnp.asarray(_to_net(b[:, 0:1], XL, XU)),
                                  jnp.asarray(_to_net(b[:, 1:2], YL, YU)),
                                  jnp.asarray(_to_net(b[:, 2:3], ZL, ZU)),
                                  diag=True)
            out["u"].append(u); out["v"].append(v); out["w"].append(w)
        return {k: jnp.concatenate(v) for k, v in out.items()}

    return {"params": params, "predict": predict,
            "info": {"mlp": MLP, "n_layers": N_LAYERS, "features": FEATURES,
                     "r": RANK, "out_dim": OUT_DIM, "lr": LR,
                     "epochs": int(steps), "published_epochs": PUBLISHED_EPOCHS,
                     "flags_used": "argparse defaults of the donor .py",
                     "published_flags": "helmholtz3d_spinn.sh: n_layers=4 features=64 r=32 nc=16",
                     "template": "helmholtz3d.py (their only 3D steady example)",
                     "steady": True, "nu": NU, "Re": 1000.0,
                     "grid": [int(len(gx)), int(len(gy)), int(len(gz))],
                     "grid_stride": STRIDE,
                     "grid_source": "task/collocation.npy is itself a 144 x 115 x 101 tensor grid",
                     "residual_mask": "the wing's interior, from the points the file omits",
                     "n_wing_per_step": N_WING,
                     "bc": ("farfield u=1 on inlet/lateral/top, wing no-slip, "
                            "outlet p=0, symmetry plane w=0 and du/dz=dv/dz=0"),
                     "scattered_eval": "diagonal contraction einsum('fn,fn,fn->n')",
                     "input_rescale": "x,y,z mapped to [-1,1]; residual carries SX, SY, SZ",
                     "error_floor": 0.0071}}
