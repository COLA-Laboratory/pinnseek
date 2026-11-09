"""SPINN on KdV — the authors' architecture, ported to this problem.

SPINN publishes no KdV example (its smallest cases are 3D), so this is a
port, not their own benchmark: NATIVE = False. Everything that is the
method comes unmodified from reference/SPINN —

  * `SPINN2d` (modified-MLP body, rank-r outer product) from
    networks/physics_informed_neural_networks.py,
  * forward-mode factorised derivatives in the style of
    networks/hessian_vector_products.py (the third x-derivative the KdV
    residual needs is the same jvp nesting taken one level deeper),
  * `setup`/`update_model` conventions, Adam(1e-3), per-axis collocation
    of nc=64 points resampled every 100 epochs, and the unweighted
    residual + IC + BC loss sum, all following klein_gordon3d.py.

Their published scripts default to 50000 epochs; `steps` is the number of
epochs, sized by bench to the wall-clock budget. Two problem-specific
choices had to be made where their examples offer no template: the IC is
matched at the sampled x axis points against the task's analytic
u(0,x) = cos(pi x), and the periodic boundary is enforced the way the
classic continuous-time KdV PINN does it — matching u and u_x at
x = -1 and x = 1 (their examples are all Dirichlet).

Config: modified_mlp, 3 layers x 64 features, r=32 — klein_gordon3d.py
defaults (`features` default is per-script; the flagship time-dependent
example uses 64).

**Published settings vs. the file's argparse defaults (2026-08-26 SPINN review).**
The numbers above are `klein_gordon3d.py`'s *argparse defaults*. They are not what the
authors ran: `scripts/klein_gordon3d_spinn.sh` and the paper's appendix ("three body networks of 4 hidden
layers with 64/32 hidden feature/output size each", Adam 0.001, 50,000
iterations) specifies
`--n_layers=4 --features=64 --r=32 --nc=64`. Only `n_layers` differs here — 4 published against the 3 used — since the
argparse defaults already carry features=64 and r=32.
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
SOURCE = "reference/SPINN (SPINN2d + klein_gordon3d.py training loop)"
SPINN_ROOT = "~/reference/SPINN"

NC = 64                # collocation points per axis, their default
FEATURES = 64          # klein_gordon3d.py default
N_LAYERS = 3
RANK = 32
LR = 1e-3
MLP = "modified_mlp"
PUBLISHED_EPOCHS = 50000
DELTA2 = 0.022 ** 2    # dispersion, problem.md


def train(*, steps: int, seed: int, gpu: str) -> dict:
    if SPINN_ROOT not in sys.path:
        sys.path.insert(0, SPINN_ROOT)

    import jax
    import jax.numpy as jnp
    import optax
    from jax import jvp
    from networks.physics_informed_neural_networks import SPINN2d
    from utils.training_utils import update_model
    from tqdm import trange

    # ── model, their init convention ──────────────────────────────────────
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    model = SPINN2d(tuple([FEATURES] * N_LAYERS), RANK, MLP)
    params = model.init(subkey, jnp.ones((NC, 1)), jnp.ones((NC, 1)))
    apply_fn = jax.jit(model.apply)

    # ── data, klein_gordon3d generator style: random per-axis points ──────
    def generate_train_data(key):
        kt, kx = jax.random.split(key)
        tc = jax.random.uniform(kt, (NC, 1), minval=0., maxval=1.)
        xc = jax.random.uniform(kx, (NC, 1), minval=-1., maxval=1.)
        # initial points: t = 0 line over the sampled x axis
        ti = jnp.zeros((1, 1))
        ui = jnp.cos(jnp.pi * xc).T          # (1, nc): u(0,x) = cos(pi x)
        # boundary points: the two spatial ends over the sampled t axis
        xb = jnp.array([[-1.], [1.]])
        return tc, xc, ti, ui, xb

    # ── loss, apply_model_spinn style with forward-mode derivatives ───────
    @partial(jax.jit, static_argnums=(0,))
    def apply_model_spinn(apply_fn, params, tc, xc, ti, ui, xb):
        def residual_loss(params, t, x):
            u = apply_fn(params, t, x)
            v_t = jnp.ones(t.shape)
            v_x = jnp.ones(x.shape)
            ut = jvp(lambda t: apply_fn(params, t, x), (t,), (v_t,))[1]
            f = lambda x: apply_fn(params, t, x)
            ux = jvp(f, (x,), (v_x,))[1]
            # third derivative: hvp_fwdfwd's jvp nesting, one level deeper
            g1 = lambda x: jvp(f, (x,), (v_x,))[1]
            g2 = lambda x: jvp(g1, (x,), (v_x,))[1]
            uxxx = jvp(g2, (x,), (v_x,))[1]
            return jnp.mean((ut + u * ux + DELTA2 * uxxx) ** 2)

        def initial_loss(params, ti, xc, ui):
            return jnp.mean((apply_fn(params, ti, xc) - ui) ** 2)

        def boundary_loss(params, tc, xb):
            ub = apply_fn(params, tc, xb)                                # (nc, 2)
            ubx = jvp(lambda x: apply_fn(params, tc, x),
                      (xb,), (jnp.ones(xb.shape),))[1]                   # (nc, 2)
            return (jnp.mean((ub[:, 0] - ub[:, 1]) ** 2)
                    + jnp.mean((ubx[:, 0] - ubx[:, 1]) ** 2))

        loss_fn = lambda params: (residual_loss(params, tc, xc)
                                  + initial_loss(params, ti, xc, ui)
                                  + boundary_loss(params, tc, xb))
        loss, gradient = jax.value_and_grad(loss_fn)(params)
        return loss, gradient

    # ── training loop, klein_gordon3d.py structure ────────────────────────
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

    # ── prediction on the scoring grid ────────────────────────────────────
    def predict(prm, X):
        """X is (N, 2) as (t, x); SPINN evaluates the factorised axis grid."""
        Xn = np.asarray(X)
        t_vals, t_inv = np.unique(Xn[:, 0], return_inverse=True)
        x_vals, x_inv = np.unique(Xn[:, 1], return_inverse=True)
        u_grid = apply_fn(prm, jnp.asarray(t_vals[:, None]),
                          jnp.asarray(x_vals[:, None]))
        u = np.asarray(u_grid)[t_inv, x_inv]
        return {"u": jnp.asarray(u)}

    return {"params": params, "predict": predict,
            "info": {"mlp": MLP, "n_layers": N_LAYERS, "features": FEATURES,
                     "r": RANK, "nc": NC, "lr": LR, "epochs": int(steps),
                     "published_epochs": PUBLISHED_EPOCHS,
                     "flags_used": "argparse defaults of the donor .py",
                     "published_flags": "klein_gordon3d_spinn.sh: n_layers=4 features=64 r=32 nc=64",
                     "resample_every": 100,
                     "bc": "periodic u and u_x matched at x=+-1"}}
