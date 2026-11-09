"""SPINN on KS — the authors' architecture, ported to this problem.

SPINN publishes no KS example (its repository's cases are 3D/4D: diffusion,
Helmholtz, Klein-Gordon, Navier-Stokes, flow-mixing, plus poisson2d), so this
is a port, not their own benchmark: NATIVE = False. Everything that is the
method comes unmodified from reference/SPINN —

  * `SPINN2d` (modified-MLP body, rank-r outer product) from
    networks/physics_informed_neural_networks.py,
  * forward-mode factorised derivatives in the style of
    networks/hessian_vector_products.py: `hvp_fwdfwd` is one `jvp` nested in
    another, and the fourth x-derivative the KS residual needs is that same
    nesting taken two levels deeper. This is cheap here exactly because the
    representation is separable — the x-axis derivative of a rank-r outer
    product only differentiates the x factor,
  * `setup`/`update_model` conventions, Adam(1e-3), per-axis collocation of
    nc=64 points resampled every 100 epochs, and the unweighted residual +
    IC + BC loss sum, all following klein_gordon3d.py.

Their published scripts default to 50000 epochs; `steps` is the number of
epochs, sized by bench to the wall-clock budget.

Problem-specific choices, where their examples offer no template:

  * the residual is the task's, u_t + nu1 u u_x + nu2 u_xx + nu3 u_xxxx with
    nu1 = 100/16, nu2 = 100/16**2, nu3 = 100/16**4 — the same three constants
    Scale-PINN's and APTS's KS code use;
  * the IC is matched at the sampled x points against u(0,x) = cos(x)(1+sin(x)).
    That analytic profile is the task's `ks_ic.csv` to 1e-9 and the reference
    solution's own first frame to 2e-15, and it is what the reference data was
    generated from (`gen_ks_chaotic.m`: `S.init = cos(x).*(1+sin(x))`). Using
    it rather than the 512-point csv lets the IC be evaluated at the freshly
    sampled x axis, which is what klein_gordon3d.py does with its analytic
    initial data;
  * the periodic boundary is enforced the way the classic continuous-time PINN
    does it — matching the trace at x = 0 and x = 2*pi (their own examples are
    all Dirichlet). KS is fourth order in x, so u, u_x, u_xx and u_xxx are all
    matched: those are the derivatives that have to agree for the periodic
    extension to satisfy the equation across the seam. (The kdv port in this
    repository matches u and u_x, the classic choice for that third-order
    operator.)

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

NU1 = 100.0 / 16.0            # problem.md nu_1 = 6.25
NU2 = 100.0 / 16.0 ** 2       # nu_2 = 0.390625
NU3 = 100.0 / 16.0 ** 4       # nu_3 = 1.52587890625e-3
T_MAX = 0.4
X_MIN, X_MAX = 0.0, 2.0 * np.pi


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
        tc = jax.random.uniform(kt, (NC, 1), minval=0., maxval=T_MAX)
        xc = jax.random.uniform(kx, (NC, 1), minval=X_MIN, maxval=X_MAX)
        # initial points: t = 0 line over the sampled x axis
        ti = jnp.zeros((1, 1))
        ui = (jnp.cos(xc) * (1. + jnp.sin(xc))).T     # (1, nc): u(0,x)
        # boundary points: the two spatial ends over the sampled t axis
        xb = jnp.array([[X_MIN], [X_MAX]])
        return tc, xc, ti, ui, xb

    # ── loss, apply_model_spinn style with forward-mode derivatives ───────
    @partial(jax.jit, static_argnums=(0,))
    def apply_model_spinn(apply_fn, params, tc, xc, ti, ui, xb):
        def dx_chain(params, t, x, order):
            """u and its first `order` x-derivatives, hvp_fwdfwd nesting."""
            v_x = jnp.ones(x.shape)
            f = lambda x: apply_fn(params, t, x)
            outs = [f(x)]
            g = f
            for _ in range(order):
                g_prev = g
                g = lambda x, g_prev=g_prev: jvp(g_prev, (x,), (v_x,))[1]
                outs.append(g(x))
            return outs

        def residual_loss(params, t, x):
            u, ux, uxx, uxxx, uxxxx = dx_chain(params, t, x, 4)
            v_t = jnp.ones(t.shape)
            ut = jvp(lambda t: apply_fn(params, t, x), (t,), (v_t,))[1]
            return jnp.mean((ut + NU1 * u * ux + NU2 * uxx + NU3 * uxxxx) ** 2)

        def initial_loss(params, ti, xc, ui):
            return jnp.mean((apply_fn(params, ti, xc) - ui) ** 2)

        def boundary_loss(params, tc, xb):
            # KS is 4th order in x: u, u_x, u_xx and u_xxx must match at the seam
            traces = dx_chain(params, tc, xb, 3)                 # each (nc, 2)
            return sum(jnp.mean((d[:, 0] - d[:, 1]) ** 2) for d in traces)

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
                     "nu": [NU1, NU2, NU3],
                     "bc": "periodic u, u_x, u_xx, u_xxx matched at x=0 and 2*pi"}}
