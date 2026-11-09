"""SPINN on the backward-facing step — their architecture, ported to this problem.

SPINN publishes no Navier-Stokes-in-a-channel case. Its only **2D steady**
example is `poisson2d.py`, which is where every hyperparameter below comes
from; the two-dimensional network class `SPINN2d` is single-output, so the
multi-output mechanism is taken from `SPINN3d`/`SPINN4d`. NATIVE = False.

Taken unmodified from reference/SPINN:

  * the body of `SPINN2d` (modified-MLP per axis, rank-r outer product) —
    reproduced in `SPINN2dMulti` below with exactly one structural change,
    which is itself copied from `SPINN3d`: the last Dense produces
    `r * out_dim` features and the contraction runs over `out_dim` separate
    rank blocks. That is their own way of giving a SPINN several outputs;
    the 2D class simply never needed it,
  * `hvp_fwdfwd` from networks/hessian_vector_products.py, used as
    poisson2d.py uses it (`return_primals=True`, first and second derivative
    from one nesting),
  * `update_model`, Adam, per-axis collocation resampled every 100 epochs,
    and poisson2d.py's loss shape: residual + `1000 *` boundary.

Hyperparameters (poisson2d.py defaults): modified_mlp, n_layers=3,
features=128, r=128, nc=64, Adam(1e-3), 50000 epochs, boundary weight 1000.
`out_dim = 3` because this problem has (u, v, p).

**Coordinates are rescaled to [-1,1] on both axes, and the residual carries
the chain-rule factors.** The channel is [0,15] x [-0.5,0.5]: an aspect ratio
of 15 and an x range no SPINN example ever faces (poisson2d is the unit
square, the 3D cases are [0,2*pi] or [0,1]). Feeding raw x into a tanh axis
network is precisely the failure the task's own notes identify — the
PINNSeek baseline sits at 0.9975 with raw inputs and reaches 0.348 with the
inputs rescaled to [-1,1], the same architecture otherwise. The map is
    X = 2x/15 - 1,  Y = 2y
so d/dx = (2/15) d/dX and d/dy = 2 d/dY, applied explicitly below and nowhere
else; the PDE being solved is unchanged, and `predict` maps the scoring grid
the same way. This is a change of variables, not a change of method — but it
is a change their code does not make, so it is called out here rather than
buried.

Boundary conditions are the task's table: inflow u = 24y(0.5-y) for y>0 else
0 and v = 0 at x = 0; u = v = 0 on y = +-0.5; p = 0 at x = 15. They enter as
a weighted least-squares term, the only mechanism poisson2d.py has (SPINN
publishes no hard-constraint machinery for this shape).

Reading the score: predicting zero gives exactly 1.0 and a per-field constant
0.667, so anything near 1.0 means the run produced nothing.

**Published settings vs. the file's argparse defaults (2026-08-26 SPINN review).**
The numbers above are `poisson2d.py`'s *argparse defaults*. They are not what the
authors ran: `scripts/poisson2d_spinn.sh` (the paper's L-shaped-Poisson appendix gives no
hyperparameters, so the script is the only record) specifies
`--n_layers=4 --features=64 --r=128 --nc=256 --epochs=5000`. So the published run is 4 layers of 64 on a 256-point axis for 5000 epochs,
against 3 layers of 128 on 64 points here.
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
SOURCE = "reference/SPINN (SPINN2d body + SPINN3d out_dim; poisson2d.py hyperparameters)"
SPINN_ROOT = "~/reference/SPINN"

NC = 64
FEATURES = 128
N_LAYERS = 3
RANK = 128
OUT_DIM = 3            # u, v, p
LR = 1e-3
MLP = "modified_mlp"
BC_WEIGHT = 1000       # poisson2d.py
PUBLISHED_EPOCHS = 50000

NU = 1.0 / 800.0
XL, XU = 0.0, 15.0
YL, YU = -0.5, 0.5
SX = 2.0 / (XU - XL)   # dX/dx
SY = 2.0 / (YU - YL)   # dY/dy


def train(*, steps: int, seed: int, gpu: str) -> dict:
    if SPINN_ROOT not in sys.path:
        sys.path.insert(0, SPINN_ROOT)

    import jax
    import jax.numpy as jnp
    import optax
    import flax.linen as nn
    from networks.hessian_vector_products import hvp_fwdfwd
    from utils.training_utils import update_model
    from tqdm import trange

    class SPINN2dMulti(nn.Module):
        """SPINN2d's body with SPINN3d's out_dim block split."""
        features: Sequence[int]
        r: int
        out_dim: int
        mlp: str

        @nn.compact
        def __call__(self, x, y):
            inputs, outputs, pred = [x, y], [], []
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
                pred += [jnp.einsum('fx, fy->xy',
                                    outputs[0][self.r * i:self.r * (i + 1)],
                                    outputs[1][self.r * i:self.r * (i + 1)])]
            return pred

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    model = SPINN2dMulti(tuple([FEATURES] * N_LAYERS), RANK, OUT_DIM, MLP)
    params = model.init(subkey, jnp.ones((NC, 1)), jnp.ones((NC, 1)))
    apply_fn = jax.jit(model.apply)

    # boundary axes, in scaled coordinates
    xb_in = jnp.array([[-1.0]])                       # x = 0
    xb_out = jnp.array([[1.0]])                       # x = 15
    yb_wall = jnp.array([[-1.0], [1.0]])              # y = -+0.5

    def generate_train_data(key):
        kx, ky = jax.random.split(key)
        xc = jax.random.uniform(kx, (NC, 1), minval=-1., maxval=1.)
        yc = jax.random.uniform(ky, (NC, 1), minval=-1., maxval=1.)
        # inflow target on the sampled y axis (physical y = Y / SY)
        yphys = yc[:, 0] / SY
        u_in = jnp.where(yphys > 0, 24. * yphys * (0.5 - yphys), 0.)
        return xc, yc, u_in

    @partial(jax.jit, static_argnums=(0,))
    def apply_model_spinn(apply_fn, params, xc, yc, u_in):
        def residual_loss(params, x, y):
            u, v, p = apply_fn(params, x, y)
            v_x = jnp.ones(x.shape)
            v_y = jnp.ones(y.shape)
            fx = lambda x: apply_fn(params, x, y)
            fy = lambda y: apply_fn(params, x, y)
            (ux, vx, px), (uxx, vxx, _) = hvp_fwdfwd(fx, (x,), (v_x,), True)
            (uy, vy, py), (uyy, vyy, _) = hvp_fwdfwd(fy, (y,), (v_y,), True)
            # physical derivatives
            ux, vx, px, uxx, vxx = SX * ux, SX * vx, SX * px, SX**2 * uxx, SX**2 * vxx
            uy, vy, py, uyy, vyy = SY * uy, SY * vy, SY * py, SY**2 * uyy, SY**2 * vyy
            ru = u * ux + v * uy + px - NU * (uxx + uyy)
            rv = u * vx + v * vy + py - NU * (vxx + vyy)
            rc = ux + vy
            return jnp.mean(ru ** 2) + jnp.mean(rv ** 2) + jnp.mean(rc ** 2)

        def boundary_loss(params, xc, yc, u_in):
            u_i, v_i, _ = apply_fn(params, xb_in, yc)        # (1, nc) each
            loss = jnp.mean((u_i[0] - u_in) ** 2) + jnp.mean(v_i[0] ** 2)
            u_w, v_w, _ = apply_fn(params, xc, yb_wall)      # (nc, 2)
            loss += jnp.mean(u_w ** 2) + jnp.mean(v_w ** 2)
            _, _, p_o = apply_fn(params, xb_out, yc)         # (1, nc)
            loss += jnp.mean(p_o ** 2)
            return loss

        loss_fn = lambda params: (residual_loss(params, xc, yc)
                                  + BC_WEIGHT * boundary_loss(params, xc, yc, u_in))
        return jax.value_and_grad(loss_fn)(params)

    optim = optax.adam(learning_rate=LR)
    state = optim.init(params)
    key, subkey = jax.random.split(key)
    train_data = generate_train_data(subkey)

    for e in trange(1, int(steps) + 1):
        if e % 100 == 0:
            key, subkey = jax.random.split(key)
            train_data = generate_train_data(subkey)
        loss, gradient = apply_model_spinn(apply_fn, params, *train_data)
        params, state = update_model(optim, gradient, params, state)
        if e % 10000 == 0:
            print(f'Epoch: {e}/{steps} --> total loss: {loss:.8f}')

    def predict(prm, X):
        """X is (N, 2) as (x, y). The reference is an unstructured vtu mesh, so
        the factorised grid is built from its distinct axis values and the
        points are gathered out of it."""
        Xn = np.asarray(X)
        x_vals, x_inv = np.unique(Xn[:, 0], return_inverse=True)
        y_vals, y_inv = np.unique(Xn[:, 1], return_inverse=True)
        xs = jnp.asarray(((x_vals - XL) * SX - 1.0)[:, None].astype(np.float32))
        ys = jnp.asarray((y_vals * SY)[:, None].astype(np.float32))
        u_g, v_g, _ = apply_fn(prm, xs, ys)
        return {"u": jnp.asarray(np.asarray(u_g)[x_inv, y_inv]),
                "v": jnp.asarray(np.asarray(v_g)[x_inv, y_inv])}

    return {"params": params, "predict": predict,
            "info": {"mlp": MLP, "n_layers": N_LAYERS, "features": FEATURES,
                     "r": RANK, "out_dim": OUT_DIM, "nc": NC, "lr": LR,
                     "bc_weight": BC_WEIGHT, "epochs": int(steps),
                     "published_epochs": PUBLISHED_EPOCHS,
                     "flags_used": "argparse defaults of the donor .py",
                     "published_flags": "poisson2d_spinn.sh: n_layers=4 features=64 r=128 nc=256 epochs=5000",
                     "template": "poisson2d.py (out_dim mechanism from SPINN3d)",
                     "nu": NU, "Re": 800, "steady": True,
                     "input_rescale": "x,y mapped to [-1,1]; residual carries the factors"}}
