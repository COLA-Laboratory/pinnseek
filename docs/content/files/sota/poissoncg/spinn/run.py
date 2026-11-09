"""SPINN on the 3D screened Poisson problem — their helmholtz3d recipe, retargeted.

The donor is unusually close. `helmholtz3d.py` is SPINN's own 3D steady
screened-Laplacian example, and its residual is written as

    (uzz + uyy + uxx + lda * u) - source_term

with `lda = 1`. This task's equation is

    -(u_xx + u_yy + u_zz) + K(z)^2 u - f(x, y, z) = 0,

which is that same expression multiplied by -1 with **lda = -K(z)^2** and
**source_term = -f**. So the residual line survives verbatim; only what is fed
into `lda` and `source_term` changes. `K` is piecewise constant (8 below
z = 0.5, 10 above), so `lda` becomes a function of z rather than a scalar —
that is the whole of the equation change.

Taken unmodified from reference/SPINN:

  * the `SPINN3d` body (modified-MLP per axis, rank-r outer product) from
    networks/physics_informed_neural_networks.py,
  * `hvp_fwdfwd` from networks/hessian_vector_products.py, used exactly as
    helmholtz3d.py uses it for the three second derivatives,
  * `update_model`, Adam at a constant rate, per-axis collocation of nc points
    resampled every 100 epochs, and helmholtz3d.py's unweighted
    `residual + boundary` sum.

Hyperparameters are helmholtz3d.py's own defaults: modified_mlp, n_layers=3,
features=128, r=128, nc=64, Adam(1e-3), 50000 epochs, out_dim=1. Adam runs at a
constant learning rate with no schedule tied to the step count, so extra epochs
are simply more training and filling the budget is the right mapping
(GUIDE.md rule 5).

NATIVE = False. Three things this problem has that helmholtz3d does not:

  * **the domain is not a box.** Two spheres of radius 0.2 are cut out of the
    cube. SPINN evaluates on the outer product of its per-axis samples, so the
    residual is computed on the whole `nc x nc x nc` grid and then averaged
    over an in-domain mask — the same trick the naca port uses for its
    immersed airfoil. Roughly 3.3% of the grid falls inside a sphere.

  * **the boundaries are Neumann, not Dirichlet.** helmholtz3d matches u to
    zero on the six faces; here every boundary is `du/dn = 0`. On the cube
    faces that is a single `jvp` along the pinned axis, the same primitive
    `hvp_fwdfwd` is built from. `[sota patch]` in `boundary_loss`.

  * **two of the eight boundaries are not axis-aligned.** The sphere surfaces
    are scattered points, and a separable network has no notion of one: its
    output is the outer product of three axis factors. Feeding N points as all
    three axes and taking the diagonal is exact but costs `N^3` memory. So the
    module gains a `diag=True` path that contracts the three axis factors
    pointwise, `einsum('fn,fn,fn->n')` — **algebraically identical** to the
    diagonal of their `einsum('fxy,fz->xyz')`, at `N` cost instead of `N^3`.
    The same path scores the 65202 unstructured reference points, which no
    amount of axis-unique-ing can turn into a tensor grid (they have 59185
    distinct x values among 65202 points).

    The parameter tree is untouched by this: `diag` is a call-time keyword,
    the layers are created in the same order with the same shapes, so the
    module is `SPINN3d` in every respect that training sees.

Coordinates are rescaled from the task's `[0, 1]^3` to the donor's `[-1, 1]^3`
so the network sees the input range helmholtz3d was tuned on. The residual
carries the chain factor: `SCALE = 2` per axis, so first derivatives pick up
`SCALE` and second derivatives `SCALE^2`. Nothing else in the equation moves,
because the source and the screening are evaluated at physical coordinates.

Reading the score: predicting zero gives exactly 1.0 and the best constant
0.929 — the field has a nonzero mean, so the two floors differ.

Pure Neumann is well posed here: the `K^2 u` term keeps the constant out of the
kernel, so no mean constraint is needed and none is added.

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
SOURCE = "reference/SPINN (SPINN3d + helmholtz3d.py training loop)"
SPINN_ROOT = "~/reference/SPINN"

NC = 64
NB = 48                # per-axis samples on a cube face; theirs is nc
N_SPHERE = 1024        # scattered points per sphere, as task/baseline.py uses
FEATURES = 128
N_LAYERS = 3
RANK = 128
OUT_DIM = 1
LR = 1e-3
MLP = "modified_mlp"
PUBLISHED_EPOCHS = 50000

# task/baseline.py frozen header, verbatim
SPHERES = ((0.4, 0.3, 0.6, 0.2), (0.6, 0.7, 0.6, 0.2))
Z_IFACE = 0.5
K_LO, K_HI = 8.0, 10.0
SRC_A1, SRC_A2 = 20.0, 100.0
SRC_M1, SRC_M2, SRC_M3 = 1.0, 10.0, 5.0

SCALE = 2.0            # physical [0,1] -> network [-1,1]; d/dx = SCALE d/dxh


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

        `diag=False` is their `einsum('fx,fy->fxy')` then `('fxy,fz->xyz')`.
        `diag=True` returns the diagonal of that tensor directly, which is what
        a scattered point set needs. Same parameters, same layer order.

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
                    H = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                    for fs in self.features[:-1]:
                        Z = nn.Dense(fs, kernel_init=init)(H)
                        Z = nn.activation.tanh(Z)
                        H = (jnp.ones_like(Z) - Z) * U + Z * V
                    H = nn.Dense(self.r * self.out_dim, kernel_init=init)(H)
                    outputs += [jnp.transpose(H, (1, 0))]
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

    # ── the task's own coefficient and source, at physical coordinates ────
    def screening(z):
        """K(z)^2 — task/baseline.py:screening, verbatim."""
        return jnp.where(z < Z_IFACE, K_LO ** 2, K_HI ** 2)

    def source(x, y, z):
        """f(x, y, z) — task/baseline.py:source, verbatim."""
        r2 = x ** 2 + y ** 2 + z ** 2
        part1 = (jnp.exp(jnp.sin(SRC_M1 * jnp.pi * x)
                         + jnp.sin(SRC_M2 * jnp.pi * y)
                         + jnp.sin(SRC_M3 * jnp.pi * z))
                 * (r2 - 1.0) / (r2 + 1.0))
        part2 = (jnp.sin(SRC_M1 * jnp.pi * x)
                 * jnp.sin(SRC_M2 * jnp.pi * y)
                 * jnp.sin(SRC_M3 * jnp.pi * z))
        return SRC_A1 * part1 + SRC_A2 * part2

    def to_net(p):
        """physical [0,1] -> network [-1,1]."""
        return 2.0 * p - 1.0

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    model = SPINN3dDiag(tuple([FEATURES] * N_LAYERS), RANK, OUT_DIM, 0, MLP)
    params = model.init(subkey, jnp.ones((NC, 1)), jnp.ones((NC, 1)), jnp.ones((NC, 1)))
    apply_fn = jax.jit(model.apply, static_argnames=("diag",))

    sph_c = jnp.array([[cx, cy, cz] for cx, cy, cz, _ in SPHERES])
    sph_r = jnp.array([r for *_, r in SPHERES])

    def generate_train_data(key):
        """helmholtz3d's `_spinn_train_generator_helmholtz3d`, retargeted.

        Theirs draws nc points per axis on [-1, 1] and builds the source on the
        meshgrid; this draws them on the physical [0, 1], builds `lda` and the
        source there, adds the in-domain mask, and hands the network its own
        rescaled copy. The six faces keep their separable form; the two sphere
        surfaces are scattered and go through the diagonal path.
        """
        k = jax.random.split(key, 6)
        xp = jax.random.uniform(k[0], (NC, 1), minval=0.0, maxval=1.0)
        yp = jax.random.uniform(k[1], (NC, 1), minval=0.0, maxval=1.0)
        zp = jax.random.uniform(k[2], (NC, 1), minval=0.0, maxval=1.0)
        Xm, Ym, Zm = jnp.meshgrid(xp[:, 0], yp[:, 0], zp[:, 0], indexing='ij')
        src = -source(Xm, Ym, Zm)                 # donor's `source_term`
        lda = -screening(zp[:, 0])[None, None, :]  # donor's `lda`, now z-dependent
        mask = jnp.ones_like(Xm)
        for i in range(len(SPHERES)):
            d2 = ((Xm - sph_c[i, 0]) ** 2 + (Ym - sph_c[i, 1]) ** 2
                  + (Zm - sph_c[i, 2]) ** 2)
            mask = mask * (d2 > sph_r[i] ** 2)

        # cube faces: one axis pinned, the other two sampled — separable
        fa = jax.random.uniform(k[3], (NB, 1), minval=0.0, maxval=1.0)
        fb = jax.random.uniform(k[4], (NB, 1), minval=0.0, maxval=1.0)
        lo, hi = jnp.zeros((1, 1)), jnp.ones((1, 1))
        # grouped by pinned axis so the axis stays a Python int through the
        # jit -- carrying it inside the tuple would make it a tracer
        faces = (((lo, fa, fb), (hi, fa, fb)),
                 ((fa, lo, fb), (fa, hi, fb)),
                 ((fa, fb, lo), (fa, fb, hi)))

        # sphere surfaces: scattered, area-uniform directions
        v = jax.random.normal(k[5], (N_SPHERE * len(SPHERES), 3))
        v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
        which = jnp.repeat(jnp.arange(len(SPHERES)), N_SPHERE)
        sp = sph_c[which] + v * sph_r[which][:, None]
        # the normal that leaves the fluid domain points into the sphere
        snrm = -v
        faces = tuple(tuple(tuple(to_net(v) for v in tri) for tri in pair)
                      for pair in faces)
        return (to_net(xp), to_net(yp), to_net(zp), src, lda, mask,
                faces, to_net(sp), snrm)

    @partial(jax.jit, static_argnums=(0,))
    def apply_model_spinn(apply_fn, params, xc, yc, zc, src, lda, mask,
                          faces, sp, snrm):
        def residual_loss(params, x, y, z, source_term, lda, mask):
            v_x, v_y, v_z = jnp.ones(x.shape), jnp.ones(y.shape), jnp.ones(z.shape)
            uxx = hvp_fwdfwd(lambda x: apply_fn(params, x, y, z), (x,), (v_x,))
            uyy = hvp_fwdfwd(lambda y: apply_fn(params, x, y, z), (y,), (v_y,))
            uzz = hvp_fwdfwd(lambda z: apply_fn(params, x, y, z), (z,), (v_z,))
            u = apply_fn(params, x, y, z)
            # [sota patch] SCALE**2 is the chain factor of the [0,1]->[-1,1] map;
            # the mask drops the grid points that fall inside a sphere
            r = (SCALE ** 2 * (uzz + uyy + uxx) + lda * u) - source_term
            return jnp.sum((r ** 2) * mask) / jnp.sum(mask)

        def boundary_loss(params, faces, sp, snrm):
            """[sota patch] theirs is `mean(u**2)` on six Dirichlet faces; this
            task is `du/dn = 0` on those six plus the two sphere surfaces."""
            loss = 0.
            for ax, pair in enumerate(faces):          # ax is a Python int
                for (a, b, c) in pair:
                    one = (jnp.ones(a.shape), jnp.ones(b.shape), jnp.ones(c.shape))[ax]
                    if ax == 0:
                        du = jvp(lambda a: apply_fn(params, a, b, c), (a,), (one,))[1]
                    elif ax == 1:
                        du = jvp(lambda b: apply_fn(params, a, b, c), (b,), (one,))[1]
                    else:
                        du = jvp(lambda c: apply_fn(params, a, b, c), (c,), (one,))[1]
                    loss += jnp.mean((SCALE * du) ** 2)
            # spheres: one jvp along the (nx, ny, nz) direction, on the diagonal
            sx, sy, sz = sp[:, 0:1], sp[:, 1:2], sp[:, 2:3]
            tx, ty, tz = snrm[:, 0:1], snrm[:, 1:2], snrm[:, 2:3]
            dn = jvp(lambda a, b, c: apply_fn(params, a, b, c, diag=True),
                     (sx, sy, sz), (tx, ty, tz))[1]
            loss += jnp.mean((SCALE * dn) ** 2)
            return loss

        loss_fn = lambda params: (residual_loss(params, xc, yc, zc, src, lda, mask)
                                  + boundary_loss(params, faces, sp, snrm))
        return jax.value_and_grad(loss_fn)(params)

    optim = optax.adam(learning_rate=LR)
    state = optim.init(params)
    key, subkey = jax.random.split(key)
    train_data = generate_train_data(subkey)

    for e in range(1, int(steps) + 1):
        if e % 100 == 0:                       # theirs: resample every 100
            key, subkey = jax.random.split(key)
            train_data = generate_train_data(subkey)
        loss, gradient = apply_model_spinn(apply_fn, params, *train_data)
        params, state = update_model(optim, gradient, params, state)
        if e % 10000 == 0:
            print(f'Epoch: {e}/{steps} --> total loss: {loss:.8f}')

    def predict(prm, X):
        """X is (N, 3) as (x, y, z). The reference is an unstructured cloud, so
        it goes through the diagonal path in chunks."""
        Xn = jnp.asarray(np.asarray(X, dtype=np.float32))
        out = []
        for k in range(0, Xn.shape[0], 16384):
            b = Xn[k:k + 16384]
            out.append(apply_fn(prm, to_net(b[:, 0:1]), to_net(b[:, 1:2]),
                                to_net(b[:, 2:3]), diag=True))
        return {"u": jnp.concatenate(out, axis=0)}

    return {"params": params, "predict": predict,
            "info": {"mlp": MLP, "n_layers": N_LAYERS, "features": FEATURES,
                     "r": RANK, "out_dim": OUT_DIM, "nc": NC, "nb": NB,
                     "n_sphere": N_SPHERE, "lr": LR, "epochs": int(steps),
                     "published_epochs": PUBLISHED_EPOCHS,
                     "flags_used": "argparse defaults of the donor .py",
                     "published_flags": "helmholtz3d_spinn.sh: n_layers=4 features=64 r=32 nc=16",
                     "template": "helmholtz3d.py (lda -> -K(z)^2, source -> -f)",
                     "steady": True, "K_lo": K_LO, "K_hi": K_HI,
                     "z_interface": Z_IFACE,
                     "spheres": [list(s) for s in SPHERES],
                     "bc": "homogeneous Neumann on 6 faces + 2 sphere surfaces",
                     "residual_mask": "analytic: grid points inside a sphere dropped",
                     "scattered_eval": "diagonal contraction einsum('fn,fn,fn->n')",
                     "input_rescale": "[0,1] -> [-1,1]; residual carries SCALE and SCALE^2"}}
