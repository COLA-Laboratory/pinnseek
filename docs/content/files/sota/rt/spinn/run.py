"""SPINN on Rayleigh-Taylor — their unsteady Navier-Stokes recipe, in primitive variables.

SPINN publishes no buoyancy-driven flow. Its only unsteady Navier-Stokes case is
`navier_stokes3d.py`, which is where every hyperparameter below comes from.
NATIVE = False.

**Why primitive variables rather than their vorticity form.** Their script
solves 2D NS in vorticity transport, with `pos_enc=5` making the network
exactly periodic in *both* spatial directions so no boundary loss is needed —
that is what the kf case here uses. This problem is periodic in x with period 1
but has **no-slip, no-flux walls at y = 0 and y = 2**. `SPINN3d` applies
`pos_enc` to both spatial axes or neither, and the encoding only buys exact
periodicity when the axis has been scaled so its span is one 2*pi period --
which is what the kf port does to x. Here the walled y axis forbids that, and
enabling `pos_enc` on the raw axes would buy Fourier features without the
exact-periodicity property that is the point of it. The port therefore keeps
their network, their derivative machinery, their optimiser and their loss
shape, but solves for (u, v, p, T) directly with `out_dim = 4` and
`pos_enc = 0`, adding the boundary terms the walls require. This is the one
place rt departs from navier_stokes3d.py's flags, and it is recorded in `info`. Their
`lbda_c`-weighted divergence term carries over unchanged.

Taken unmodified from reference/SPINN:

  * `SPINN3d` (modified-MLP per axis, rank-r outer product, out_dim block
    split),
  * `hvp_fwdfwd` and the `jvp` nesting, used as navier_stokes3d.py uses them,
  * `update_model`, Adam, per-axis collocation resampled every 100 epochs, and
    the `residual + lbda_c * divergence + lbda_ic * initial` loss shape.

Hyperparameters are navier_stokes3d.py's own values: modified_mlp,
n_layers=3, features=128, r=128, nc=64, Adam(2e-3), 100000 epochs,
`lbda_c=5000`, `lbda_ic=10000`. These agree with `scripts/navier_stokes3d.sh`
and with the paper's appendix ("three body networks (modified MLP) of 3 hidden
layers with 128/256 hidden feature/output sizes", Adam 0.002, 100,000
iterations per time window), so this donor is one of the two where the
argparse defaults and the published run agree. Adam runs at a constant learning rate with
nothing tied to the step count, so filling the budget is the right mapping.

The equations are the task's four, Boussinesq, with
`NU = sqrt(Pr/Ra) = 8.426150e-4` and `KAPPA = 1/sqrt(Pr*Ra) = 1.186782e-3` for
Pr = 0.71, Ra = 1e6 — the values `problem.md` gives.

The initial condition is the task's own `rt_ic.csv` on the full 100 x 200
grid. The initial term is evaluated on the *collocation* axes (as
klein_gordon3d.py does: `xi = xc`), which are random rather than on that
lattice, so the table is looked up by nearest neighbour on its own stretched
axes. Only T is scored, so `predict` returns the key "T"; u, v and p are
trained because the buoyancy term couples T into the momentum equation.

Reading the score: predicting zero gives exactly 1.0 and the best possible
constant 0.740.
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
METHOD_LIBS = "spinn"
NATIVE = False
SOURCE = "reference/SPINN (SPINN3d + navier_stokes3d.py training loop, primitive variables)"
SPINN_ROOT = "~/reference/SPINN"
TASK = "~/rt/task"

NC = 64
FEATURES = 128
N_LAYERS = 3
RANK = 128
OUT_DIM = 4            # u, v, p, T
LR = 2e-3
MLP = "modified_mlp"
LBDA_C = 5000          # their incompressibility weight
LBDA_IC = 10000        # navier_stokes3d.py's own value (both the argparse
                       # default and scripts/navier_stokes3d.sh). It was 100
                       # here until a review caught the disagreement; measured
                       # seed 1, 248000 steps: 0.46727 at 100, 0.46297 at 10000
PUBLISHED_EPOCHS = 100000

XL, XU = 0.0, 1.0
YL, YU = 0.0, 2.0
T_MAX = 4.75
PR, RA = 0.71, 1e6
NU = float(np.sqrt(PR / RA))
KAPPA = float(1.0 / np.sqrt(PR * RA))


def _ic_tables():
    ic = np.loadtxt(f"{TASK}/rt_ic.csv", delimiter=",", skiprows=1, dtype=np.float64)
    xs, ys = np.unique(ic[:, 0]), np.unique(ic[:, 1])
    o = np.lexsort((ic[:, 1], ic[:, 0]))
    U = ic[o, 2].reshape(len(xs), len(ys))
    V = ic[o, 3].reshape(len(xs), len(ys))
    T = ic[o, 4].reshape(len(xs), len(ys))
    assert np.allclose(ic[o, 0].reshape(U.shape), xs[:, None])
    return xs, ys, U, V, T


def train(*, steps: int, seed: int, gpu: str) -> dict:
    if SPINN_ROOT not in sys.path:
        sys.path.insert(0, SPINN_ROOT)

    import jax
    import jax.numpy as jnp
    import optax
    from jax import jvp
    from networks.physics_informed_neural_networks import SPINN3d
    from networks.hessian_vector_products import hvp_fwdfwd
    from utils.training_utils import update_model
    from tqdm import trange

    icx, icy, ICU, ICV, ICT = _ic_tables()
    icx_j, icy_j = jnp.asarray(icx), jnp.asarray(icy)
    ICU_j, ICV_j, ICT_j = (jnp.asarray(a, dtype=jnp.float32) for a in (ICU, ICV, ICT))

    def ic_on(x, y):
        """The task's t=0 fields on the outer product of the sampled axes."""
        def _nn(a, v):
            # nearest, not `searchsorted` alone: that returns the first index at
            # or above v, i.e. a ceiling. The grid is stretched (dx runs 0.0032
            # to 0.0166), and the collocation axes this is evaluated on are
            # random, so the bias is real: against bilinear interpolation of the
            # same table the ceiling is 0.0191 in relative L2 and this is 0.0094.
            j = jnp.clip(jnp.searchsorted(a, v), 1, a.shape[0] - 1)
            return jnp.where(jnp.abs(v - a[j - 1]) <= jnp.abs(v - a[j]), j - 1, j)
        ix = _nn(icx_j, x[:, 0])
        iy = _nn(icy_j, y[:, 0])
        return (ICU_j[ix][:, iy], ICV_j[ix][:, iy], ICT_j[ix][:, iy])

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    model = SPINN3d(tuple([FEATURES] * N_LAYERS), RANK, OUT_DIM, 0, MLP)
    params = model.init(subkey, jnp.ones((NC, 1)), jnp.ones((NC, 1)), jnp.ones((NC, 1)))
    apply_fn = jax.jit(model.apply)

    def generate_train_data(key):
        kt, kx, ky = jax.random.split(key, 3)
        tc = jax.random.uniform(kt, (NC, 1), minval=0., maxval=T_MAX)
        xc = jax.random.uniform(kx, (NC, 1), minval=XL, maxval=XU)
        yc = jax.random.uniform(ky, (NC, 1), minval=YL, maxval=YU)
        ti = jnp.zeros((1, 1))
        u0, v0, T0 = ic_on(xc, yc)
        yb = jnp.array([[YL], [YU]])                 # the two walls
        xlo = jnp.array([[XL]])                      # periodic pair, x and x+1
        xhi = jnp.array([[XU]])
        return tc, xc, yc, ti, u0, v0, T0, yb, xlo, xhi

    @partial(jax.jit, static_argnums=(0,))
    def apply_model_spinn(apply_fn, params, tc, xc, yc, ti, u0, v0, T0, yb, xlo, xhi):
        def residual_loss(params, t, x, y):
            u, v, p, T = apply_fn(params, t, x, y)
            vt, vx, vy = jnp.ones(t.shape), jnp.ones(x.shape), jnp.ones(y.shape)
            ft = lambda t: apply_fn(params, t, x, y)
            fx = lambda x: apply_fn(params, t, x, y)
            fy = lambda y: apply_fn(params, t, x, y)
            ut, vt_, _pt, Tt = jvp(ft, (t,), (vt,))[1]
            (ux, vx_, px, Tx), (uxx, vxx, _pxx, Txx) = hvp_fwdfwd(fx, (x,), (vx,), True)
            (uy, vy_, py, Ty), (uyy, vyy, _pyy, Tyy) = hvp_fwdfwd(fy, (y,), (vy,), True)
            adv = lambda f_x, f_y: u * f_x + v * f_y
            r_u = ut + adv(ux, uy) + px - NU * (uxx + uyy)
            r_v = vt_ + adv(vx_, vy_) + py - NU * (vxx + vyy) - T
            r_T = Tt + adv(Tx, Ty) - KAPPA * (Txx + Tyy)
            div = ux + vy_
            return (jnp.mean(r_u ** 2) + jnp.mean(r_v ** 2) + jnp.mean(r_T ** 2),
                    jnp.mean(div ** 2))

        def initial_loss(params, ti, x, y, u0, v0, T0):
            u, v, _p, T = apply_fn(params, ti, x, y)
            return (jnp.mean((jnp.squeeze(u) - u0) ** 2)
                    + jnp.mean((jnp.squeeze(v) - v0) ** 2)
                    + jnp.mean((jnp.squeeze(T) - T0) ** 2))

        def boundary_loss(params, t, x, y, yb, xlo, xhi):
            # walls: u = v = T = 0 at y = 0 and y = 2
            uw, vw, _pw, Tw = apply_fn(params, t, x, yb)
            wall = jnp.mean(uw ** 2) + jnp.mean(vw ** 2) + jnp.mean(Tw ** 2)
            # periodic in x with period 1: match the two faces, all four fields
            lo = apply_fn(params, t, xlo, y)
            hi = apply_fn(params, t, xhi, y)
            per = sum(jnp.mean((h - l) ** 2) for l, h in zip(lo, hi))
            return wall + per

        def loss_fn(params):
            res, div = residual_loss(params, tc, xc, yc)
            return (res + LBDA_C * div
                    + LBDA_IC * initial_loss(params, ti, xc, yc, u0, v0, T0)
                    + boundary_loss(params, tc, xc, yc, yb, xlo, xhi))

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
        """X is (N,3) as (t, x, y). Only T is scored, so only T is returned."""
        Xn = np.asarray(X)
        t_vals, t_inv = np.unique(Xn[:, 0], return_inverse=True)
        x_vals, x_inv = np.unique(Xn[:, 1], return_inverse=True)
        y_vals, y_inv = np.unique(Xn[:, 2], return_inverse=True)
        out = apply_fn(prm, jnp.asarray(t_vals[:, None].astype(np.float32)),
                       jnp.asarray(x_vals[:, None].astype(np.float32)),
                       jnp.asarray(y_vals[:, None].astype(np.float32)))
        T = np.asarray(out[3])
        return {"T": jnp.asarray(T[t_inv, x_inv, y_inv])}

    return {"params": params, "predict": predict,
            "info": {"mlp": MLP, "n_layers": N_LAYERS, "features": FEATURES,
                     "r": RANK, "out_dim": OUT_DIM, "nc": NC, "lr": LR,
                     "lbda_c": LBDA_C, "lbda_ic": LBDA_IC,
                     "epochs": int(steps), "published_epochs": PUBLISHED_EPOCHS,
                     "template": "navier_stokes3d.py, primitive variables (pos_enc off)",
                     "NU": NU, "KAPPA": KAPPA, "Pr": PR, "Ra": RA,
                     "ic": "task rt_ic.csv, nearest-neighbour on its stretched axes",
                     "pos_enc": "0 -- their 5 needs a 2*pi-scaled axis, which the walls forbid",
                     "bc": "walls u=v=T=0 at y=0,2; periodic in x with period 1",
                     "scored_field": "T only"}}
