"""SPINN on Kolmogorov flow — their own 2D unsteady Navier-Stokes machinery.

SPINN publishes `navier_stokes3d.py`: two-dimensional unsteady incompressible
Navier-Stokes on a periodic square, solved in the velocity-vorticity form with
a divergence penalty. That is this task's equation apart from three problem
constants, so this is their closest possible donor — but it is still a port,
not their benchmark: NATIVE = False.

Taken unmodified from reference/SPINN:

  * `SPINN3d` with `out_dim=2` and `pos_enc=5` — their exact-periodicity
    construction (the positional encoding is sin/cos of integer multiples of
    the spatial input, so the network is periodic by construction and there is
    no boundary loss, exactly as their script says),
  * `velocity_to_vorticity_fwd` from utils/vorticity.py, and `hvp_fwdfwd`
    from networks/hessian_vector_products.py,
  * the loss of `apply_model_spinn` in navier_stokes3d.py: vorticity-transport
    residual + `lbda_c` * divergence + `lbda_ic` * (vorticity + velocity at
    t=0), unweighted otherwise,
  * their multi-offset collocation scheme: `offset_num=8` jittered copies of
    an evenly spaced grid, cycled every `offset_iter=100` epochs,
  * `update_model`, Adam(2e-3), and their published hyperparameters
    lbda_c=5000, lbda_ic=10000, modified_mlp, 3 layers x 128 features, r=128.

Three problem constants change, and one coordinate convention:

  1. `nu` is the task's 5e-4; their script hard-codes 0.01;
  2. the task has Kolmogorov forcing f = 2 sin(4 pi y) on the x-momentum. In
     the vorticity form that enters as the curl of the force,
     d(f_y)/dx - d(f_x)/dy = -8 pi cos(4 pi y), so the residual is
     w_t + u w_x + v w_y - nu (w_xx + w_yy) + 8 pi cos(4 pi y);
  3. their spatial domain is [0, 2*pi]^2 and the task's is [0, 1]^2. Rather
     than break their exact-periodicity encoding (which has period 2*pi
     built in), the network is fed the scaled coordinates X = 2*pi*x,
     Y = 2*pi*y, so its periodicity is exactly the task's period 1. Every
     spatial derivative therefore carries a factor S = 2*pi, applied
     explicitly in the residual below and nowhere else. `predict` scales the
     scoring grid the same way;
  4. `t` runs over the task's [0, 0.75].

Time marching: their script solves the window in `marching_steps` chunks
launched as separate processes, each seeded by the previous chunk's
prediction. Under a single wall-clock budget there is no published way to
split the budget across chunks, so this runs one window over the whole
interval — their own `step_idx=0, marching_steps=0` path (`generate_train_data`
reads `marching_steps != 0` as the switch). That is recorded here because it
is a real departure: their published result for this problem is a marched one.

The initial condition is the task's `kf_ic.csv` velocity field on the full
256x256 grid, with the initial vorticity obtained from it spectrally (the
same quantity their generator ships as `w0`). `steps` is the number of
epochs, sized by bench to the wall-clock budget; their default is 100000.
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
METHOD_LIBS = "spinn"
NATIVE = False
SOURCE = "reference/SPINN/navier_stokes3d.py (nu, Kolmogorov forcing, unit-square scaling)"
SPINN_ROOT = "~/reference/SPINN"
TASK = Path("~/kf/task")

FEATURES = 128         # navier_stokes3d.py defaults
N_LAYERS = 3
RANK = 128
OUT_DIM = 2
POS_ENC = 5
LR = 2e-3
MLP = "modified_mlp"
LBDA_C = 5000
LBDA_IC = 10000
OFFSET_NUM = 8
OFFSET_ITER = 100
PUBLISHED_EPOCHS = 100000

NT = 32                # collocation points on the time axis
NXY = 64               # per spatial axis
NU = 5e-4
T_MAX = 0.75
S = 2.0 * np.pi        # task x in [0,1] -> network X in [0,2pi]


def _ic():
    """(x, y, u0, v0, w0) from the task's own initial condition file.

    w0 = v_x - u_y by FFT on the periodic 256x256 grid — the same quantity
    SPINN's data generator ships alongside u0/v0 for this equation.
    """
    csv = np.loadtxt(TASK / "kf_ic.csv", delimiter=",", skiprows=1, dtype=np.float64)
    x = np.unique(csv[:, 0]); y = np.unique(csv[:, 1])
    nx, ny = len(x), len(y)
    # rows are ordered with y fastest, matching the reference grid
    u = csv[:, 2].reshape(nx, ny)
    v = csv[:, 3].reshape(nx, ny)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=1.0 / nx)      # period 1 in x
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=1.0 / ny)
    KX = kx[:, None]; KY = ky[None, :]
    v_x = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(v)))
    u_y = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(u)))
    return x, y, u, v, v_x - u_y


def train(*, steps: int, seed: int, gpu: str) -> dict:
    if SPINN_ROOT not in sys.path:
        sys.path.insert(0, SPINN_ROOT)

    import jax
    import jax.numpy as jnp
    import optax
    from jax import jvp
    from networks.physics_informed_neural_networks import SPINN3d
    from networks.hessian_vector_products import hvp_fwdfwd
    from utils.vorticity import velocity_to_vorticity_fwd
    from utils.training_utils import update_model
    from tqdm import trange

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    model = SPINN3d(tuple([FEATURES] * N_LAYERS), RANK, OUT_DIM, POS_ENC, MLP)
    params = model.init(subkey, jnp.ones((NT, 1)), jnp.ones((NXY, 1)), jnp.ones((NXY, 1)))
    apply_fn = jax.jit(model.apply)

    xg, yg, u0g, v0g, w0g = _ic()
    ti = jnp.zeros((1, 1))
    xi = jnp.asarray(S * xg[:, None])                     # scaled axes
    yi = jnp.asarray(S * yg[:, None])
    u0 = jnp.asarray(u0g[None, :, :]); v0 = jnp.asarray(v0g[None, :, :])
    w0 = jnp.asarray(w0g[None, :, :])

    # collocation grid, their generator's construction (evenly spaced,
    # endpoint excluded) plus the offset_num jittered copies
    def make_grids(key):
        kt, kxy = jax.random.split(key)
        tc = jnp.linspace(0., T_MAX, NT, endpoint=False)[:, None]
        xc = jnp.linspace(0., S, NXY, endpoint=False)[:, None]
        dt = tc[1, 0] - tc[0, 0]; dxy = xc[1, 0] - xc[0, 0]
        ot = jax.random.uniform(kt, (OFFSET_NUM - 1,), minval=0., maxval=dt)
        oxy = jax.random.uniform(kxy, (OFFSET_NUM - 1,), minval=0., maxval=dxy)
        tcm = jnp.concatenate([tc[None], jnp.stack([tc + o for o in ot])])
        xcm = jnp.concatenate([xc[None], jnp.stack([xc + o for o in oxy])])
        return tcm, xcm, xcm

    key, subkey = jax.random.split(key)
    tc_mult, xc_mult, yc_mult = make_grids(subkey)

    @partial(jax.jit, static_argnums=(0,))
    def apply_model_spinn(apply_fn, params, tc, xc, yc, ti, xi, yi, w0, u0, v0):
        def residual_loss(params, t, x, y):
            uv = apply_fn(params, t, x, y)
            vec_t = jnp.ones(t.shape); vec_x = jnp.ones(x.shape); vec_y = jnp.ones(y.shape)
            wfn = lambda t, x, y: velocity_to_vorticity_fwd(apply_fn, params, t, x, y)
            w_t = jvp(lambda t: wfn(t, x, y), (t,), (vec_t,))[1]
            w_x = jvp(lambda x: wfn(t, x, y), (x,), (vec_x,))[1]
            w_y = jvp(lambda y: wfn(t, x, y), (y,), (vec_y,))[1]
            w_xx = hvp_fwdfwd(lambda x: wfn(t, x, y), (x,), (vec_x,))
            w_yy = hvp_fwdfwd(lambda y: wfn(t, x, y), (y,), (vec_y,))
            # the network's spatial argument is X = S*x, so d/dx = S * d/dX and
            # the vorticity itself already carries one factor of S
            R_w = (S * w_t + S * S * (uv[0] * w_x + uv[1] * w_y)
                   - NU * S ** 3 * (w_xx + w_yy)
                   + 8. * jnp.pi * jnp.cos(4. * jnp.pi * y / S))
            u_x = jvp(lambda x: apply_fn(params, t, x, y)[0], (x,), (vec_x,))[1]
            v_y = jvp(lambda y: apply_fn(params, t, x, y)[1], (y,), (vec_y,))[1]
            R_c = S * (u_x + v_y)
            return jnp.mean(R_w ** 2) + LBDA_C * jnp.mean(R_c ** 2)

        def initial_loss(params, ti, xi, yi, w0, u0, v0):
            w = velocity_to_vorticity_fwd(apply_fn, params, ti, xi, yi) * S
            u, v = apply_fn(params, ti, xi, yi)
            return (jnp.mean((jnp.squeeze(w) - jnp.squeeze(w0)) ** 2)
                    + jnp.mean((jnp.squeeze(u) - jnp.squeeze(u0)) ** 2)
                    + jnp.mean((jnp.squeeze(v) - jnp.squeeze(v0)) ** 2))

        loss_fn = lambda params: (residual_loss(params, tc, xc, yc)
                                  + LBDA_IC * initial_loss(params, ti, xi, yi, w0, u0, v0))
        return jax.value_and_grad(loss_fn)(params)

    optim = optax.adam(learning_rate=LR)
    state = optim.init(params)
    tc, xc, yc = tc_mult[0], xc_mult[0], yc_mult[0]

    for e in trange(1, int(steps) + 1):
        if e % OFFSET_ITER == 0:
            i = (e // OFFSET_ITER) % OFFSET_NUM
            tc, xc, yc = tc_mult[i], xc_mult[i], yc_mult[i]
        loss, gradient = apply_model_spinn(apply_fn, params, tc, xc, yc,
                                           ti, xi, yi, w0, u0, v0)
        params, state = update_model(optim, gradient, params, state)
        if e % 20000 == 0:
            print(f'Epoch: {e}/{steps} --> total loss: {loss:.8f}')

    def predict(prm, X):
        """X is (N, 3) as (t, x, y); the network takes the scaled spatial axes."""
        Xn = np.asarray(X)
        t_vals, t_inv = np.unique(Xn[:, 0], return_inverse=True)
        x_vals, x_inv = np.unique(Xn[:, 1], return_inverse=True)
        y_vals, y_inv = np.unique(Xn[:, 2], return_inverse=True)
        u_g, v_g = apply_fn(prm, jnp.asarray(t_vals[:, None]),
                            jnp.asarray(S * x_vals[:, None]),
                            jnp.asarray(S * y_vals[:, None]))
        return {"u": jnp.asarray(np.asarray(u_g)[t_inv, x_inv, y_inv]),
                "v": jnp.asarray(np.asarray(v_g)[t_inv, x_inv, y_inv])}

    return {"params": params, "predict": predict,
            "info": {"mlp": MLP, "n_layers": N_LAYERS, "features": FEATURES,
                     "r": RANK, "out_dim": OUT_DIM, "pos_enc": POS_ENC,
                     "nt": NT, "nxy": NXY, "lr": LR,
                     "lbda_c": LBDA_C, "lbda_ic": LBDA_IC,
                     "offset_num": OFFSET_NUM, "offset_iter": OFFSET_ITER,
                     "epochs": int(steps), "published_epochs": PUBLISHED_EPOCHS,
                     "nu": NU, "forcing_curl": "-8 pi cos(4 pi y)",
                     "space_scaling": "network sees 2*pi*x so its period is the task's 1",
                     "time_marching": "off (single window over [0, 0.75])"}}
