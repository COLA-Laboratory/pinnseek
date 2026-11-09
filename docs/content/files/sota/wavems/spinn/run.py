"""SPINN on the multiscale wave equation — their klein_gordon3d recipe, retargeted.

SPINN publishes no variable-coefficient wave case, so this is a port:
NATIVE = False. But the donor is unusually close. `klein_gordon3d.py` is a
2D+time, **second order in time** equation whose residual their own code writes
as `utt - uxx - uyy + u**2 - source`; this task is `utt - c(x,y)(uxx + uyy)`.
The derivative machinery, the network, the sampling and the loss shape all
carry over; what changes is the coefficient, the initial condition and the
boundary type.

Taken unmodified from reference/SPINN:

  * `SPINN3d` (modified-MLP per axis, rank-r outer product, out_dim block
    split) from networks/physics_informed_neural_networks.py,
  * `hvp_fwdfwd` from networks/hessian_vector_products.py, used exactly as
    klein_gordon3d.py uses it for the three second derivatives,
  * `update_model`, Adam, per-axis collocation of nc=64 points resampled every
    100 epochs, and klein_gordon3d.py's unweighted
    `residual + initial + boundary` sum.

Hyperparameters are klein_gordon3d.py's own defaults: modified_mlp,
n_layers=3, features=64, r=32, nc=64, Adam(1e-3), 50000 epochs, out_dim=1.
Adam runs at a constant learning rate with no schedule tied to the step count,
so extra epochs are simply more training and filling the budget is the right
mapping (GUIDE.md rule 5).

What this problem needs that klein_gordon3d does not have:

  * **A variable coefficient.** `c(x, y)` takes the values 1 and 12 on a
    256x256 grid and `problem.md` specifies bilinear interpolation between grid
    points. SPINN evaluates on the outer product of its per-axis samples, so
    the coefficient is interpolated onto that `nc x nc` grid each time the
    collocation is resampled — bilinearly, as specified, not snapped.
  * **A second initial condition.** The task gives u(0,·) and u_t(0,·) = 0, so
    there are two initial terms rather than one. `u_t` at t = 0 comes from a
    single forward-mode `jvp` along the t axis, the same primitive
    `hvp_fwdfwd` is built from.
  * **Neumann boundaries, not Dirichlet.** klein_gordon3d matches u against an
    exact solution on the four sides. Here u_x = 0 at x = +-1 and u_y = 0 at
    y = +-1, so the boundary term is the squared normal derivative, again by
    `jvp` along the axis in question.

**The trap in this case is that u == 0 satisfies the equation and both Neumann
conditions exactly.** Only the initial condition rules it out, so a run whose
IC term is under-weighted collapses to zero and scores exactly 1.0. The loss is
left unweighted, as klein_gordon3d.py leaves it — this is what the published
recipe does, and where it lands is the measurement.

Coordinates are used as-is: t in [0,5] and x,y in [-1,1] are within the range
klein_gordon3d samples ([0,10] x [-1,1]^2), so no rescaling is applied.

Reading the score: predicting zero gives exactly 1.0, the best constant 0.781,
and freezing the initial condition 1.274.

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
SOURCE = "reference/SPINN (SPINN3d + klein_gordon3d.py training loop)"
SPINN_ROOT = "~/reference/SPINN"
TASK = "~/wavems/task"

NC = 64
FEATURES = 64
N_LAYERS = 3
RANK = 32
OUT_DIM = 1
LR = 1e-3
MLP = "modified_mlp"
PUBLISHED_EPOCHS = 50000

T_MAX = 5.0
XL, XU = -1.0, 1.0
YL, YU = -1.0, 1.0
SIGMA = 0.3


def _medium():
    """(x_axis, y_axis, c_grid) — the task's own 256x256 wave-speed field."""
    csv = np.loadtxt(f"{TASK}/wavems_coef.csv", delimiter=",", skiprows=1, dtype=np.float64)
    xs, ys = np.unique(csv[:, 0]), np.unique(csv[:, 1])
    order = np.lexsort((csv[:, 1], csv[:, 0]))
    c = csv[order, 2].reshape(len(xs), len(ys))
    assert np.allclose(csv[order, 0].reshape(len(xs), len(ys)), xs[:, None])
    return xs, ys, c


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

    gx, gy, gc = _medium()
    gx_j, gy_j, gc_j = jnp.asarray(gx), jnp.asarray(gy), jnp.asarray(gc)

    def c_bilinear(x, y):
        """c on the outer product of the sampled axes, bilinear as problem.md says."""
        fx = jnp.clip((x[:, 0] - gx_j[0]) / (gx_j[1] - gx_j[0]), 0.0, gx_j.shape[0] - 1.001)
        fy = jnp.clip((y[:, 0] - gy_j[0]) / (gy_j[1] - gy_j[0]), 0.0, gy_j.shape[0] - 1.001)
        i0, j0 = jnp.floor(fx).astype(int), jnp.floor(fy).astype(int)
        wx, wy = (fx - i0)[:, None], (fy - j0)[None, :]
        c00 = gc_j[i0][:, j0]
        c10 = gc_j[i0 + 1][:, j0]
        c01 = gc_j[i0][:, j0 + 1]
        c11 = gc_j[i0 + 1][:, j0 + 1]
        return ((1 - wx) * (1 - wy) * c00 + wx * (1 - wy) * c10
                + (1 - wx) * wy * c01 + wx * wy * c11)

    def u0_fn(x, y):
        """u(0,x,y) = exp(-((x+0.5)^2 + y^2) / (2 sigma^2)), outer product form."""
        return jnp.exp(-((x[:, 0] + 0.5) ** 2)[:, None] / (2 * SIGMA ** 2)) * \
               jnp.exp(-(y[:, 0] ** 2)[None, :] / (2 * SIGMA ** 2))

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    model = SPINN3d(tuple([FEATURES] * N_LAYERS), RANK, OUT_DIM, 0, MLP)   # (features, r, out_dim, pos_enc, mlp)
    params = model.init(subkey, jnp.ones((NC, 1)), jnp.ones((NC, 1)), jnp.ones((NC, 1)))
    apply_fn = jax.jit(model.apply)

    def generate_train_data(key):
        kt, kx, ky = jax.random.split(key, 3)
        tc = jax.random.uniform(kt, (NC, 1), minval=0., maxval=T_MAX)
        xc = jax.random.uniform(kx, (NC, 1), minval=XL, maxval=XU)
        yc = jax.random.uniform(ky, (NC, 1), minval=YL, maxval=YU)
        cc = c_bilinear(xc, yc)                     # (nc, nc) on the sampled grid
        ti = jnp.zeros((1, 1))
        ui = u0_fn(xc, yc)[None, :, :]              # (1, nc, nc)
        xb = jnp.array([[XL], [XU]])
        yb = jnp.array([[YL], [YU]])
        return tc, xc, yc, cc, ti, ui, xb, yb

    @partial(jax.jit, static_argnums=(0,))
    def apply_model_spinn(apply_fn, params, tc, xc, yc, cc, ti, ui, xb, yb):
        def residual_loss(params, t, x, y, c):
            v_t, v_x, v_y = jnp.ones(t.shape), jnp.ones(x.shape), jnp.ones(y.shape)
            utt = hvp_fwdfwd(lambda t: apply_fn(params, t, x, y), (t,), (v_t,))
            uxx = hvp_fwdfwd(lambda x: apply_fn(params, t, x, y), (x,), (v_x,))
            uyy = hvp_fwdfwd(lambda y: apply_fn(params, t, x, y), (y,), (v_y,))
            return jnp.mean((utt - c[None, :, :] * (uxx + uyy)) ** 2)

        def initial_loss(params, ti, x, y, ui):
            u = apply_fn(params, ti, x, y)
            ut = jvp(lambda t: apply_fn(params, t, x, y), (ti,), (jnp.ones(ti.shape),))[1]
            return jnp.mean((u - ui) ** 2) + jnp.mean(ut ** 2)   # u(0,.) and u_t(0,.)=0

        def boundary_loss(params, t, x, y, xb, yb):
            ux_b = jvp(lambda x: apply_fn(params, t, x, y), (xb,), (jnp.ones(xb.shape),))[1]
            uy_b = jvp(lambda y: apply_fn(params, t, x, y), (yb,), (jnp.ones(yb.shape),))[1]
            return jnp.mean(ux_b ** 2) + jnp.mean(uy_b ** 2)     # zero Neumann

        loss_fn = lambda params: (residual_loss(params, tc, xc, yc, cc)
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
                     "published_flags": "klein_gordon3d_spinn.sh: n_layers=4 features=64 r=32 nc=64",
                     "template": "klein_gordon3d.py (2D+time, second order in t)",
                     "t_max": T_MAX, "sigma": SIGMA,
                     "coefficient": "task 256x256 c(x,y), bilinear as problem.md specifies",
                     "bc": "zero Neumann on all four sides",
                     "ic": "u(0,.) and u_t(0,.)=0, both terms unweighted"}}
