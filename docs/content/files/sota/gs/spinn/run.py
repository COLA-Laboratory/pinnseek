"""SPINN on Grey-Scott — the authors' architecture, ported to this problem.

SPINN publishes no Grey-Scott example, so this is a port, not their own
benchmark: NATIVE = False. Everything that is the method comes unmodified from
reference/SPINN —

  * `SPINN3d` (modified-MLP body, rank-r outer product, `out_dim` separate
    rank blocks) from networks/physics_informed_neural_networks.py — the same
    class navier_stokes3d.py uses with out_dim=2 for its two fields,
  * forward-mode factorised derivatives via `hvp_fwdfwd` from
    networks/hessian_vector_products.py, used exactly as diffusion3d.py uses
    it (`return_primals=True` to get u_x and u_xx from one nesting),
  * `update_model` from utils/training_utils.py, Adam, per-axis collocation of
    nc=64 points resampled every 100 epochs, and the unweighted residual +
    IC + BC loss sum, all following diffusion3d.py.

Hyperparameters come from **diffusion3d.py**, their closest published
example — a 3D (t,x,y) time-dependent diffusion problem, the same shape as
this one: modified_mlp, n_layers=3, features=128, r=128, pos_enc=0, nc=64,
Adam(1e-3), 50000 epochs. The single departure is `out_dim = 2`, because
Grey-Scott has two fields; that is navier_stokes3d.py's value, the repository's
own setting for a two-field problem. `steps` is the number of epochs, sized by
bench to the wall-clock budget.

Problem-specific choices, where their examples offer no template:

  * the residual is the task's pair,
        u_t - eps1(u_xx+u_yy) - b1(1-u) + c1 u v^2
        v_t - eps2(v_xx+v_yy) + b2 v - c2 u v^2
    with eps1,eps2 = 0.2,0.1; b1,b2 = 40,100; c1=c2=1000 — the same six
    constants Scale-PINN hard-codes and APTS reads out of `gray_scott.mat`;
  * the IC is matched on the sampled (x,y) axes against the task's analytic
    u0 = 1-exp(-10((x+0.05)^2+(y+0.02)^2)), v0 = exp(-10((x-0.05)^2+(y-0.02)^2)).
    Both are separable in x and y, so they are formed as an outer product on
    the axis samples — exactly the representation SPINN wants. That analytic
    pair is the task's ref_data first frame to 3e-7;
  * the periodic boundary is enforced the way the classic continuous-time PINN
    does it — matching the traces at x = -1 vs x = +1 and y = -1 vs y = +1
    (their own examples are all Dirichlet). Grey-Scott is second order in
    space, so u, v and their first normal derivatives are matched: the
    derivatives up to order N-1 that have to agree for the periodic extension
    to satisfy the equation across the seam.

The coordinates here are the task's own (t, x, y). Unlike Scale-PINN and APTS,
nothing is read out of `gray_scott.mat`, so the [t,y,x] storage order that
makes those two need a swap at predict time does not arise.

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

NC = 64                # collocation points per axis, diffusion3d.py default
FEATURES = 128         # diffusion3d.py default
N_LAYERS = 3
RANK = 128
OUT_DIM = 2            # navier_stokes3d.py's value for a two-field problem
POS_ENC = 0
LR = 1e-3
MLP = "modified_mlp"
PUBLISHED_EPOCHS = 50000

EPS1, EPS2 = 0.2, 0.1
B1, B2 = 40.0, 100.0
C1, C2 = 1000.0, 1000.0
T_MAX = 0.78
XL, XU = -1.0, 1.0
YL, YU = -1.0, 1.0


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

    # ── model, their init convention ──────────────────────────────────────
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    model = SPINN3d(tuple([FEATURES] * N_LAYERS), RANK, OUT_DIM, POS_ENC, MLP)
    params = model.init(subkey, jnp.ones((NC, 1)), jnp.ones((NC, 1)), jnp.ones((NC, 1)))
    apply_fn = jax.jit(model.apply)

    def ic_grid(xc, yc):
        """u0, v0 on the sampled axes — both separable, so an outer product."""
        gx = jnp.exp(-10. * (xc[:, 0] + 0.05) ** 2)
        gy = jnp.exp(-10. * (yc[:, 0] + 0.02) ** 2)
        hx = jnp.exp(-10. * (xc[:, 0] - 0.05) ** 2)
        hy = jnp.exp(-10. * (yc[:, 0] - 0.02) ** 2)
        u0 = 1. - gx[None, :, None] * gy[None, None, :]
        v0 = hx[None, :, None] * hy[None, None, :]
        return u0, v0                              # (1, nc, nc) each

    # ── data, diffusion3d generator style: random per-axis points ─────────
    def generate_train_data(key):
        kt, kx, ky = jax.random.split(key, 3)
        tc = jax.random.uniform(kt, (NC, 1), minval=0., maxval=T_MAX)
        xc = jax.random.uniform(kx, (NC, 1), minval=XL, maxval=XU)
        yc = jax.random.uniform(ky, (NC, 1), minval=YL, maxval=YU)
        ti = jnp.zeros((1, 1))
        u0, v0 = ic_grid(xc, yc)
        xb = jnp.array([[XL], [XU]])
        yb = jnp.array([[YL], [YU]])
        return tc, xc, yc, ti, u0, v0, xb, yb

    # ── loss, diffusion3d's apply_model_spinn with forward-mode derivatives ─
    @partial(jax.jit, static_argnums=(0,))
    def apply_model_spinn(apply_fn, params, tc, xc, yc, ti, u0, v0, xb, yb):
        def residual_loss(params, t, x, y):
            u, v = apply_fn(params, t, x, y)
            v_t = jnp.ones(t.shape)
            v_x = jnp.ones(x.shape)
            v_y = jnp.ones(y.shape)
            ut, vt = jvp(lambda t: apply_fn(params, t, x, y), (t,), (v_t,))[1]
            _, (uxx, vxx) = hvp_fwdfwd(lambda x: apply_fn(params, t, x, y),
                                       (x,), (v_x,), True)
            _, (uyy, vyy) = hvp_fwdfwd(lambda y: apply_fn(params, t, x, y),
                                       (y,), (v_y,), True)
            ru = ut - EPS1 * (uxx + uyy) - B1 * (1. - u) + C1 * u * v ** 2
            rv = vt - EPS2 * (vxx + vyy) + B2 * v - C2 * u * v ** 2
            return jnp.mean(ru ** 2) + jnp.mean(rv ** 2)

        def initial_loss(params, ti, xc, yc, u0, v0):
            u, v = apply_fn(params, ti, xc, yc)
            return jnp.mean((u - u0) ** 2) + jnp.mean((v - v0) ** 2)

        def boundary_loss(params, tc, xc, yc, xb, yb):
            # periodic in x: u, v and u_x, v_x agree at x = -1 and x = +1
            vb = jnp.ones(xb.shape)
            (ux_u, ux_v) = jvp(lambda x: apply_fn(params, tc, x, yc), (xb,), (vb,))[1]
            ub, vb_ = apply_fn(params, tc, xb, yc)               # (nt, 2, ny)
            lx = (jnp.mean((ub[:, 0] - ub[:, 1]) ** 2)
                  + jnp.mean((vb_[:, 0] - vb_[:, 1]) ** 2)
                  + jnp.mean((ux_u[:, 0] - ux_u[:, 1]) ** 2)
                  + jnp.mean((ux_v[:, 0] - ux_v[:, 1]) ** 2))
            # periodic in y
            wb = jnp.ones(yb.shape)
            (uy_u, uy_v) = jvp(lambda y: apply_fn(params, tc, xc, y), (yb,), (wb,))[1]
            ub2, vb2 = apply_fn(params, tc, xc, yb)              # (nt, nx, 2)
            ly = (jnp.mean((ub2[..., 0] - ub2[..., 1]) ** 2)
                  + jnp.mean((vb2[..., 0] - vb2[..., 1]) ** 2)
                  + jnp.mean((uy_u[..., 0] - uy_u[..., 1]) ** 2)
                  + jnp.mean((uy_v[..., 0] - uy_v[..., 1]) ** 2))
            return lx + ly

        loss_fn = lambda params: (residual_loss(params, tc, xc, yc)
                                  + initial_loss(params, ti, xc, yc, u0, v0)
                                  + boundary_loss(params, tc, xc, yc, xb, yb))
        loss, gradient = jax.value_and_grad(loss_fn)(params)
        return loss, gradient

    # ── training loop, diffusion3d.py structure ───────────────────────────
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
        """X is (N, 3) as (t, x, y); SPINN evaluates the factorised axis grid."""
        Xn = np.asarray(X)
        t_vals, t_inv = np.unique(Xn[:, 0], return_inverse=True)
        x_vals, x_inv = np.unique(Xn[:, 1], return_inverse=True)
        y_vals, y_inv = np.unique(Xn[:, 2], return_inverse=True)
        u_g, v_g = apply_fn(prm, jnp.asarray(t_vals[:, None]),
                            jnp.asarray(x_vals[:, None]),
                            jnp.asarray(y_vals[:, None]))
        u = np.asarray(u_g)[t_inv, x_inv, y_inv]
        v = np.asarray(v_g)[t_inv, x_inv, y_inv]
        return {"u": jnp.asarray(u), "v": jnp.asarray(v)}

    return {"params": params, "predict": predict,
            "info": {"mlp": MLP, "n_layers": N_LAYERS, "features": FEATURES,
                     "r": RANK, "out_dim": OUT_DIM, "pos_enc": POS_ENC,
                     "nc": NC, "lr": LR, "epochs": int(steps),
                     "published_epochs": PUBLISHED_EPOCHS,
                     "flags_used": "argparse defaults of the donor .py",
                     "published_flags": "diffusion3d_spinn.sh: n_layers=4 features=64 r=32 nc=64",
                     "template": "diffusion3d.py (out_dim from navier_stokes3d.py)",
                     "resample_every": 100,
                     "coeffs": [EPS1, EPS2, B1, B2, C1, C2],
                     "bc": "periodic u,v and first normal derivatives at x,y = -1 and +1"}}
