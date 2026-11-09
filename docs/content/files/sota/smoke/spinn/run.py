"""SPINN on wildfire-smoke transport — their diffusion3d recipe, retargeted.

SPINN publishes no advection-diffusion case with variable coefficients, so this
is a port: NATIVE = False. The donor is unusually close, though.
`diffusion3d.py` is 2D + time, its residual already assembles exactly the
derivatives this task needs (`ut`, `ux`, `uxx`, `uy`, `uyy`), and its
`boundary_loss` is already **zero Dirichlet on the four sides** — which is this
task's boundary condition, not something that had to be added.

Taken unmodified from reference/SPINN:

  * `SPINN3d` (modified-MLP per axis, rank-r outer product) from
    networks/physics_informed_neural_networks.py,
  * `hvp_fwdfwd` and the `jvp` nesting, used exactly as diffusion3d.py uses
    them for the first and second derivatives,
  * `update_model`, Adam, per-axis collocation of nc=64 points resampled every
    100 epochs, and diffusion3d.py's unweighted
    `residual + initial + boundary` sum,
  * `boundary_loss`, which is already the squared network value on the four
    faces.

Hyperparameters are diffusion3d.py's own defaults: modified_mlp, n_layers=3,
features=128, r=128, nc=64, Adam(1e-3), 50000 epochs, out_dim=1. Adam runs at a
constant learning rate with nothing tied to the step count, so extra epochs are
simply more training and the budget is filled (GUIDE.md rule 5).

What this problem needs that diffusion3d does not have:

  * **the equation.** Theirs is a nonlinear diffusion,
    `ut - alpha*(ux^2 + u*uxx + uy^2 + u*uyy)`. This task's is
    `u_t + a u_x + b u_y + c u - DXC u_xx - DYC u_yy - f`, with a, b, c, f the
    ERA5/GFAS fields.
  * **the coefficient lookup.** SPINN evaluates on the outer product of its
    per-axis samples, so the four fields are interpolated onto that
    `nc x nc x nc` grid each time the collocation is resampled. The lookup is
    trilinear as `problem.md` specifies, and uses the `(t, y, x)` storage order
    the task warns about while the coordinates are `(t, x, y)` — see
    `cases/smoke/methods/verify_coef.py`, which checks the same construction
    against the task's own function (max |diff| 0.000; the transposed order
    differs by up to 57).
  * **the initial condition is zero.** u(0,x,y) = 0 everywhere, so the initial
    term is the squared network value rather than an error against a profile.

Coordinates are used as-is: the task normalises all three to [0,1], which is
inside the range diffusion3d samples.

Reading the score: predicting zero gives exactly 1.0 and the best possible
constant 0.972 — the field is 98.9% empty sky, so anything at or above ~0.97
means the plume was never found.

**Published settings vs. the file's argparse defaults (2026-08-26 SPINN review).**
The numbers above are `diffusion3d.py`'s *argparse defaults*. They are not what the
authors ran: `scripts/diffusion3d_spinn.sh` and the paper's appendix ("three body networks of 4 hidden
layers with 64/32 hidden feature/output size each", Adam 0.001, 50,000
iterations) specifies
`--n_layers=4 --features=64 --r=32 --nc=64`. So the published body network is 4 layers of 64 with rank 32, against the
3 layers of 128 with rank 128 used here.
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

import numpy as np

HERE = Path(__file__).parent
METHOD_LIBS = "spinn"
NATIVE = False
SOURCE = "reference/SPINN (SPINN3d + diffusion3d.py training loop)"
SPINN_ROOT = "~/reference/SPINN"
TASK = "~/smoke/task"

NC = 64
FEATURES = 128
N_LAYERS = 3
RANK = 128
OUT_DIM = 1
LR = 1e-3
MLP = "modified_mlp"
PUBLISHED_EPOCHS = 50000
GRID_NT, GRID_NY, GRID_NX = 73, 153, 241


def train(*, steps: int, seed: int, gpu: str) -> dict:
    if SPINN_ROOT not in sys.path:
        sys.path.insert(0, SPINN_ROOT)

    import jax
    import jax.numpy as jnp
    import optax
    from jax import jvp
    from jax.scipy.ndimage import map_coordinates
    from networks.physics_informed_neural_networks import SPINN3d
    from networks.hessian_vector_products import hvp_fwdfwd
    from utils.training_utils import update_model
    from tqdm import trange

    z = np.load(f"{TASK}/smoke_coef.npz")
    A, B, C, F = (jnp.asarray(z[k], dtype=jnp.float32) for k in ("a", "b", "c", "f"))
    DXC, DYC = float(z["dxc"]), float(z["dyc"])
    assert A.shape == (GRID_NT, GRID_NY, GRID_NX), A.shape

    def coef_grid(field, t, x, y):
        """The field on the outer product of the sampled axes, (nt, nx, ny).

        task/baseline.py:coef_at evaluated on a grid: the stored layout is
        (t, y, x), so the coordinate order fed to map_coordinates is
        (t, y, x) while ours are (t, x, y).
        """
        T = (t[:, 0] * (GRID_NT - 1))[:, None, None]
        X = (x[:, 0] * (GRID_NX - 1))[None, :, None]
        Y = (y[:, 0] * (GRID_NY - 1))[None, None, :]
        T, X, Y = jnp.broadcast_arrays(T, X, Y)
        return map_coordinates(field, jnp.stack([T, Y, X]), order=1, mode="nearest")

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    model = SPINN3d(tuple([FEATURES] * N_LAYERS), RANK, OUT_DIM, 0, MLP)
    params = model.init(subkey, jnp.ones((NC, 1)), jnp.ones((NC, 1)), jnp.ones((NC, 1)))
    apply_fn = jax.jit(model.apply)

    def generate_train_data(key):
        kt, kx, ky = jax.random.split(key, 3)
        tc = jax.random.uniform(kt, (NC, 1), minval=0., maxval=1.)
        xc = jax.random.uniform(kx, (NC, 1), minval=0., maxval=1.)
        yc = jax.random.uniform(ky, (NC, 1), minval=0., maxval=1.)
        a = coef_grid(A, tc, xc, yc)
        b = coef_grid(B, tc, xc, yc)
        c = coef_grid(C, tc, xc, yc)
        f = coef_grid(F, tc, xc, yc)
        ti = jnp.zeros((1, 1))                       # u(0, x, y) = 0
        tb = tc
        xb = jnp.array([[0.], [1.]])
        yb = jnp.array([[0.], [1.]])
        return tc, xc, yc, a, b, c, f, ti, tb, xb, yb

    @partial(jax.jit, static_argnums=(0,))
    def apply_model_spinn(apply_fn, params, tc, xc, yc, a, b, c, f, ti, tb, xb, yb):
        def residual_loss(params, t, x, y, a, b, c, f):
            u = apply_fn(params, t, x, y)
            v_t, v_x, v_y = jnp.ones(t.shape), jnp.ones(x.shape), jnp.ones(y.shape)
            ut = jvp(lambda t: apply_fn(params, t, x, y), (t,), (v_t,))[1]
            ux, uxx = hvp_fwdfwd(lambda x: apply_fn(params, t, x, y), (x,), (v_x,), True)
            uy, uyy = hvp_fwdfwd(lambda y: apply_fn(params, t, x, y), (y,), (v_y,), True)
            r = ut + a * ux + b * uy + c * u - DXC * uxx - DYC * uyy - f
            return jnp.mean(r ** 2)

        def initial_loss(params, ti, x, y):
            return jnp.mean(apply_fn(params, ti, x, y) ** 2)      # u(0,.) = 0

        def boundary_loss(params, tb, xc, yc, xb, yb):
            # zero Dirichlet on the four sides, as diffusion3d.py already does
            return (jnp.mean(apply_fn(params, tb, xb, yc) ** 2)
                    + jnp.mean(apply_fn(params, tb, xc, yb) ** 2))

        loss_fn = lambda params: (residual_loss(params, tc, xc, yc, a, b, c, f)
                                  + initial_loss(params, ti, xc, yc)
                                  + boundary_loss(params, tb, xc, yc, xb, yb))
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
        """X is (N,3) as (t, x, y); SPINN evaluates the factorised axis grid."""
        Xn = np.asarray(X)
        t_vals, t_inv = np.unique(Xn[:, 0], return_inverse=True)
        x_vals, x_inv = np.unique(Xn[:, 1], return_inverse=True)
        y_vals, y_inv = np.unique(Xn[:, 2], return_inverse=True)
        u_g = apply_fn(prm, jnp.asarray(t_vals[:, None].astype(np.float32)),
                       jnp.asarray(x_vals[:, None].astype(np.float32)),
                       jnp.asarray(y_vals[:, None].astype(np.float32)))
        u_g = np.asarray(u_g[0] if isinstance(u_g, (list, tuple)) else u_g)
        return {"u": jnp.asarray(u_g[t_inv, x_inv, y_inv])}

    return {"params": params, "predict": predict,
            "info": {"mlp": MLP, "n_layers": N_LAYERS, "features": FEATURES,
                     "r": RANK, "out_dim": OUT_DIM, "nc": NC, "lr": LR,
                     "epochs": int(steps), "published_epochs": PUBLISHED_EPOCHS,
                     "flags_used": "argparse defaults of the donor .py",
                     "published_flags": "diffusion3d_spinn.sh: n_layers=4 features=64 r=32 nc=64",
                     "template": "diffusion3d.py (2D+time, Dirichlet sides)",
                     "dxc": DXC, "dyc": DYC,
                     "coefficients": "trilinear, (t,y,x) storage as the task stores them",
                     "ic": "u(0,.) = 0", "bc": "zero Dirichlet on all four sides"}}
