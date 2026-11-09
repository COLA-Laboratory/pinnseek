"""SPINN on the long-time nonlinear heat equation — their diffusion3d recipe, retargeted.

SPINN publishes no case with a solution-dependent source, so this is a port:
NATIVE = False. The donor is close, though. `diffusion3d.py` is 2D + time, its
residual already assembles `ut`, `uxx` and `uyy`, and its `boundary_loss` is
already **zero Dirichlet on the four sides** — this task's boundary condition,
not something that had to be added.

Taken unmodified from reference/SPINN:

  * `SPINN3d` (modified-MLP per axis, rank-r outer product),
  * `hvp_fwdfwd` and the `jvp` nesting, used as diffusion3d.py uses them,
  * `update_model`, Adam, per-axis collocation of nc=64 points resampled every
    100 epochs, and diffusion3d.py's unweighted
    `residual + initial + boundary` sum,
  * `boundary_loss`, already the squared network value on the four faces.

Hyperparameters are diffusion3d.py's own defaults: modified_mlp, n_layers=3,
features=128, r=128, nc=64, Adam(1e-3), 50000 epochs, out_dim=1. Adam runs at a
constant learning rate with nothing tied to the step count, so filling the
budget is the right mapping (GUIDE.md rule 5).

What this problem needs that diffusion3d does not have:

  * **the equation.** Theirs is a nonlinear diffusion in the *gradient*; this is
        u_t - D (u_xx + u_yy) - A sin(u^2) (1 + 2 sin(pi t / 4)) sin(4 pi x) sin(2 pi y)
    with D = 0.001 and A = 5, all analytic — no table to interpolate.
  * **the initial condition** `sin(4 pi x) sin(3 pi y)`, which is separable, so
    it is built directly on the sampled axes as an outer product rather than
    looked up. (The task also ships it as `heat_ic.csv` on a 128 x 128 grid;
    the analytic form is what `problem.md` states and the file reproduces it.)
  * **a long window.** t runs to 100, which is 12.5 periods of the forcing
    (period 8). The axis is fed as-is: diffusion3d samples t in [0,1], so this
    is far outside the range its features were tuned for, and that is part of
    what the case measures. Nothing is rescaled, and the docstring says so
    rather than the code hiding it.

**The trap: u == 0 satisfies the PDE and all four sides exactly**, because the
source carries `sin(u^2)` and vanishes with u. Only the initial term rules it
out, and diffusion3d.py weights all three terms equally — that is the published
recipe and where it lands is the measurement.

Reading the score: predicting zero gives exactly 1.0, and so does the best
possible constant (the field is a zero-mean oscillation), so anything at or
above 1.0 means the run produced nothing.

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

NC = 64
FEATURES = 128
N_LAYERS = 3
RANK = 128
OUT_DIM = 1
LR = 1e-3
MLP = "modified_mlp"
PUBLISHED_EPOCHS = 50000

D_COEF = 0.001
A_COEF = 5.0
T_MAX = 100.0


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

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    model = SPINN3d(tuple([FEATURES] * N_LAYERS), RANK, OUT_DIM, 0, MLP)
    params = model.init(subkey, jnp.ones((NC, 1)), jnp.ones((NC, 1)), jnp.ones((NC, 1)))
    apply_fn = jax.jit(model.apply)

    def generate_train_data(key):
        kt, kx, ky = jax.random.split(key, 3)
        tc = jax.random.uniform(kt, (NC, 1), minval=0., maxval=T_MAX)
        xc = jax.random.uniform(kx, (NC, 1), minval=0., maxval=1.)
        yc = jax.random.uniform(ky, (NC, 1), minval=0., maxval=1.)
        # the source's separable spatial and temporal factors, on the axes
        st = (1.0 + 2.0 * jnp.sin(jnp.pi * tc[:, 0] / 4.0))          # (nc,)
        sx = jnp.sin(4.0 * jnp.pi * xc[:, 0])                        # (nc,)
        sy = jnp.sin(2.0 * jnp.pi * yc[:, 0])                        # (nc,)
        src = A_COEF * st[:, None, None] * sx[None, :, None] * sy[None, None, :]
        ti = jnp.zeros((1, 1))
        ui = (jnp.sin(4.0 * jnp.pi * xc[:, 0])[None, :, None]
              * jnp.sin(3.0 * jnp.pi * yc[:, 0])[None, None, :])     # (1, nc, nc)
        xb = jnp.array([[0.], [1.]])
        yb = jnp.array([[0.], [1.]])
        return tc, xc, yc, src, ti, ui, xb, yb

    @partial(jax.jit, static_argnums=(0,))
    def apply_model_spinn(apply_fn, params, tc, xc, yc, src, ti, ui, xb, yb):
        def residual_loss(params, t, x, y, src):
            u = apply_fn(params, t, x, y)
            v_t, v_x, v_y = jnp.ones(t.shape), jnp.ones(x.shape), jnp.ones(y.shape)
            ut = jvp(lambda t: apply_fn(params, t, x, y), (t,), (v_t,))[1]
            _ux, uxx = hvp_fwdfwd(lambda x: apply_fn(params, t, x, y), (x,), (v_x,), True)
            _uy, uyy = hvp_fwdfwd(lambda y: apply_fn(params, t, x, y), (y,), (v_y,), True)
            r = ut - D_COEF * (uxx + uyy) - jnp.sin(u ** 2) * src
            return jnp.mean(r ** 2)

        def initial_loss(params, ti, x, y, ui):
            return jnp.mean((apply_fn(params, ti, x, y) - ui) ** 2)

        def boundary_loss(params, t, x, y, xb, yb):
            # zero Dirichlet on the four sides, as diffusion3d.py already does
            return (jnp.mean(apply_fn(params, t, xb, y) ** 2)
                    + jnp.mean(apply_fn(params, t, x, yb) ** 2))

        loss_fn = lambda params: (residual_loss(params, tc, xc, yc, src)
                                  + initial_loss(params, ti, xc, yc, ui)
                                  + boundary_loss(params, tc, xc, yc, xb, yb))
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
        """X is (N,3) as (t, x, y).

        The reference is a COMSOL mesh of 16 x 12 nodes replicated over 501
        snapshots, so the distinct coordinate values are few and the factorised
        grid is small; the points are gathered out of it.
        """
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
                     "D": D_COEF, "A": A_COEF, "t_max": T_MAX,
                     "forcing_period": 8.0, "forcing_cycles": T_MAX / 8.0,
                     "ic": "sin(4 pi x) sin(3 pi y), separable outer product",
                     "bc": "zero Dirichlet on all four sides",
                     "no_rescaling": "t fed as-is over [0,100]"}}
