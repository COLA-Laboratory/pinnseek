"""SPINN on the vortex street grown from rest — their navier_stokes3d recipe, retargeted.

**This is the `cyl` port with four constants changed**, and it is written that
way on purpose: cyl2 is the same channel, the same cylinder, the same
`nu = 1e-3` and the same equations, started from rest instead of from a
developed cycle and run five times as long. The four changes are
`U(t) = sin(pi t / 8)` in place of `sin(pi (t + 4.8) / 8)`, `T_MAX = 6`, an
initial condition of `u = v = 0` that is analytic rather than read from a
file, and the budget. Everything else -- the donor, the hyperparameters, the
rescaling, the masking, the diagonal path -- is identical, so the two rows can
be read against each other.

`reference/SPINN/navier_stokes3d.py` is SPINN's own **2D + time incompressible
Navier-Stokes**, written in vorticity form:

    w_t + u w_x + v w_y - nu (w_xx + w_yy) = 0,   u_x + v_y = 0,   w = v_x - u_y

with the network emitting (u, v) and the vorticity taken as their autodiff
curl (`utils/vorticity.py:velocity_to_vorticity_fwd`). That is exactly this
task's equation with the pressure eliminated. The residual, the two loss
weights, the network and the training loop all carry over. NATIVE = False,
because their case is a periodic box with no obstacle and no boundary loss at
all, and this one is a channel with a cylinder cut out of it.

Taken unmodified from reference/SPINN:

  * the `SPINN3d` body (modified-MLP per axis, rank-r outer product, out_dim
    block split) — reproduced in `SPINN3dDiag` below with one addition, the
    diagonal contraction described under "the cylinder" below,
  * `hvp_fwdfwd` from networks/hessian_vector_products.py and `jvp`, used as
    navier_stokes3d.py uses them,
  * `update_model`, Adam, and their loss shape
    `residual + lbda_c * divergence + lbda_ic * initial`.

Hyperparameters are navier_stokes3d.py's own defaults: modified_mlp,
n_layers=3, features=128, r=128, out_dim=2, pos_enc=5, Adam(2e-3),
lbda_c=5000, lbda_ic=10000, 100000 epochs. Two of its flags have no default in
the repository (`--nt` and `--nxy` are `None`), so the per-axis count is taken
from every other 3D example in the same repository, which uses `nc = 64`.
Adam runs at a constant learning rate with no schedule tied to the step count,
so extra epochs are simply more training and filling the budget is the right
mapping (GUIDE.md rule 5). Time marching is off (`marching_steps = 0` is their
own no-marching setting) — one window over the whole `t in [0, 6]`.

**One flag of theirs is not carried over: the multi-offset collocation.**
navier_stokes3d.py jitters its evenly spaced grid into `offset_num = 8` copies
and cycles them every `offset_iter = 100` epochs. Here the grid is the plain
`linspace` one and is not jittered, so the collocation points are fixed for the
whole run. The kf port, from the same donor, does implement the offsets.

That inconsistency was measured on `cyl`, this port's parent, rather than
left as a question. Adding the jitter there — eight copies, offsets drawn in `[0, d)` per axis, cycled every
100 epochs, the mask and the inlet target rebuilt for each copy — scores
**1.4690** at seed 1 and 86000 epochs against **0.5360** for the fixed grid,
at the same wall clock. The scheme is built for their doubly-periodic box,
where a shifted grid is still the same grid; on a channel it walks the
collocation points off the inlet plane at `x = XL` for seven copies out of
eight, and the boundary terms are what tie this problem down. So the fixed
grid stays, and this is the measurement that says why.

What this problem has that their periodic box does not:

  * **an obstacle.** The residual is computed on the whole `nt x nx x ny` outer
    product and then averaged over an in-fluid mask; the 0.9% of the grid
    inside the cylinder is dropped (the disc is pi*0.05^2 / (2.2*0.41) = 0.87%
    of the channel, and 0.90% of this grid's points at nc = 64). Same trick as
    the naca and poissoncg ports.

  * **four boundary conditions where they have none.** Their case is periodic
    and says so ("no boundary loss since we're using exact periodic b.c").
    Here the inlet carries the task's driven profile
    `u = 6 U(t) y (H - y) / H^2` with `U(t) = sin(pi (t + 4.8) / 8)`, the two
    channel walls and the cylinder surface are no-slip, and all three are
    velocity Dirichlet, which this formulation expresses directly.

  * **the outlet condition cannot be expressed at all.** The task sets `p = 0`
    at `x = 2.2`, and there is no pressure in the vorticity form — it is
    exactly what taking the curl removes. It is therefore absent from the loss,
    and that is a limitation of their formulation on this problem, not an
    omission: the inflow plus incompressibility still fix the mass flux. It is
    recorded in `info` as `outlet`.

  * **the cylinder surface is scattered.** A separable network has no notion of
    a scattered point: its output is the outer product of three axis factors.
    So the module gains a `diag=True` path contracting the three factors
    pointwise, `einsum('fn,fn,fn->n')`, which is **algebraically identical** to
    the diagonal of their `einsum('fxy,fz->xyz')` at `N` cost instead of `N^3`
    (checked to 4.7e-10 on the poissoncg port, where the same path is used).
    `diag` is a call-time keyword, so the parameter tree is `SPINN3d`'s exactly.

**Coordinates are rescaled to [-1,1] on all three axes, and the residual
carries the chain-rule factors.** The channel is `[0, 2.2] x [0, 0.41]`, an
aspect ratio of 5.4 with a y range of 0.41 — nothing like the `[0, 2 pi]^2`
their case samples. The map is

    T = 2t/1.2 - 1,   X = 2x/2.2 - 1,   Y = 2y/0.41 - 1

so `d/dt = ST d/dT` with `ST = 2/1.2`, and likewise `SX = 2/2.2`,
`SY = 2/0.41`; second derivatives carry the square. They are applied explicitly
in the residual and nowhere else, and `predict` maps the scoring grid the same
way. This is a change of variables, not a change of method — the same one the
bfs port makes, and for the same reason.

Reading the score: the fluid starts at rest, so freezing the initial condition
*is* predicting zero and both score exactly 1.0. The best constant gives 0.585
and the inlet profile imposed everywhere -- plug flow that ignores the cylinder
-- gives 0.430. The last is the anchor to beat: a method that only transports
the inlet without shedding lands there.
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
SOURCE = "reference/SPINN/navier_stokes3d.py (their own 2D+time Navier-Stokes, vorticity form)"
SPINN_ROOT = "~/reference/SPINN"
TASK = "~/cyl2/task"

NT = NXY = 64
FEATURES = 128
N_LAYERS = 3
RANK = 128
OUT_DIM = 2
POS_ENC = 5
LR = 2e-3
MLP = "modified_mlp"
LBDA_C = 5000
LBDA_IC = 10000
PUBLISHED_EPOCHS = 100000

NU = 1e-3
H = 0.41
XL, XU = 0.0, 2.2
YL, YU = 0.0, 0.41
T_MAX = 6.0
CX, CY, CR = 0.2, 0.2, 0.05
N_CYL = 128                # points around the cylinder, per time sample
N_CYL_T = 16               # time samples for the cylinder condition

ST, SX, SY = 2.0 / T_MAX, 2.0 / (XU - XL), 2.0 / (YU - YL)


def _to_net(v, lo, hi):
    return 2.0 * (v - lo) / (hi - lo) - 1.0


def _ic():
    """(x_axis, y_axis, u_grid, v_grid, mask) — the fluid starts at rest.

    cyl ships a `cyl_ic.csv` with a developed shedding cycle; cyl2's initial
    condition is `u = v = 0` and analytic, so nothing is read. The grid is
    built at the same 0.005 spacing the cyl file uses, and the mask drops the
    cylinder's interior, where there is no fluid.
    """
    xs = np.arange(0.0, 2.2 + 1e-9, 0.005)
    ys = np.arange(0.0, 0.41 + 1e-9, 0.005)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    m = (((Xg - CX) ** 2 + (Yg - CY) ** 2) > CR ** 2).astype(np.float64)
    z = np.zeros_like(m)
    return xs, ys, z, z, m


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
        """SPINN3d verbatim, plus a pointwise (diagonal) contraction path."""
        features: Sequence[int]
        r: int
        out_dim: int
        pos_enc: int
        mlp: str

        @nn.compact
        def __call__(self, x, y, z, diag: bool = False):
            if self.pos_enc != 0:
                # positional encoding only to spatial coordinates (theirs)
                freq = jnp.expand_dims(jnp.arange(1, self.pos_enc + 1, 1), 0)
                y = jnp.concatenate((jnp.ones((y.shape[0], 1)), jnp.sin(y @ freq),
                                     jnp.cos(y @ freq)), 1)
                z = jnp.concatenate((jnp.ones((z.shape[0], 1)), jnp.sin(z @ freq),
                                     jnp.cos(z @ freq)), 1)
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
                    Hh = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                    for fs in self.features[:-1]:
                        Z = nn.Dense(fs, kernel_init=init)(Hh)
                        Z = nn.activation.tanh(Z)
                        Hh = (jnp.ones_like(Z) - Z) * U + Z * V
                    Hh = nn.Dense(self.r * self.out_dim, kernel_init=init)(Hh)
                    outputs += [jnp.transpose(Hh, (1, 0))]
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

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key, 2)
    model = SPINN3dDiag(tuple([FEATURES] * N_LAYERS), RANK, OUT_DIM, POS_ENC, MLP)
    params = model.init(subkey, jnp.ones((NT, 1)), jnp.ones((NXY, 1)), jnp.ones((NXY, 1)))
    apply_fn = jax.jit(model.apply, static_argnames=("diag",))

    # ── the initial condition, on the task's own grid ────────────────────
    icx, icy, icu, icv, icm = _ic()
    ic_x = jnp.asarray(_to_net(icx, XL, XU)[:, None].astype(np.float32))
    ic_y = jnp.asarray(_to_net(icy, YL, YU)[:, None].astype(np.float32))
    ic_t = jnp.asarray(np.array([[_to_net(0.0, 0.0, T_MAX)]], dtype=np.float32))
    ic_u = jnp.asarray(icu[None, :, :].astype(np.float32))
    ic_v = jnp.asarray(icv[None, :, :].astype(np.float32))
    ic_m = jnp.asarray(icm[None, :, :].astype(np.float32))

    # ── the cylinder surface, scattered, at N_CYL_T time samples ─────────
    ang = np.linspace(0.0, 2.0 * np.pi, N_CYL, endpoint=False)
    cyl_x = CX + CR * np.cos(ang)
    cyl_y = CY + CR * np.sin(ang)
    ct = np.repeat(np.linspace(0.0, T_MAX, N_CYL_T), N_CYL)
    cx_ = np.tile(cyl_x, N_CYL_T)
    cy_ = np.tile(cyl_y, N_CYL_T)
    cyl_t = jnp.asarray(_to_net(ct, 0.0, T_MAX)[:, None].astype(np.float32))
    cyl_xn = jnp.asarray(_to_net(cx_, XL, XU)[:, None].astype(np.float32))
    cyl_yn = jnp.asarray(_to_net(cy_, YL, YU)[:, None].astype(np.float32))

    def generate_train_data(key):
        """navier_stokes3d's own linspace collocation, on this channel.

        Theirs is `linspace(0, 2 pi, nxy, endpoint=False)` per spatial axis and
        `linspace(0, t_end, nt, endpoint=False)` in time; this is the same,
        over the task's box, plus the in-fluid mask and the two Dirichlet
        boundary sets that their periodic case does not have. The `key` is
        accepted for signature compatibility and, as in theirs, the grid is
        deterministic.
        """
        tc = np.linspace(0.0, T_MAX, NT, endpoint=False)
        xc = np.linspace(XL, XU, NXY, endpoint=False)
        yc = np.linspace(YL, YU, NXY, endpoint=False)
        Xg, Yg = np.meshgrid(xc, yc, indexing='ij')
        mask = (((Xg - CX) ** 2 + (Yg - CY) ** 2) > CR ** 2).astype(np.float32)
        # inlet: x = 0, all t and y; target from the task's driven profile
        Tg, Yi = np.meshgrid(tc, yc, indexing='ij')
        # [sota patch] cyl2 ramps from rest: U(t) = sin(pi t / 8)
        u_in = (6.0 * np.sin(np.pi * Tg / 8.0) * Yi * (H - Yi) / H ** 2)
        return (jnp.asarray(_to_net(tc, 0.0, T_MAX)[:, None].astype(np.float32)),
                jnp.asarray(_to_net(xc, XL, XU)[:, None].astype(np.float32)),
                jnp.asarray(_to_net(yc, YL, YU)[:, None].astype(np.float32)),
                jnp.asarray(mask[None, :, :]),
                jnp.asarray(u_in[:, None, :].astype(np.float32)))

    @partial(jax.jit, static_argnums=(0,))
    def apply_model_spinn(apply_fn, params, tc, xc, yc, mask, u_in):
        def vort(params, t, x, y):
            """[sota patch] their velocity_to_vorticity_fwd, with the chain
            factors of the [-1,1] map: w = SX v_X - SY u_Y."""
            v_x = jvp(lambda x: apply_fn(params, t, x, y)[1], (x,), (jnp.ones(x.shape),))[1]
            u_y = jvp(lambda y: apply_fn(params, t, x, y)[0], (y,), (jnp.ones(y.shape),))[1]
            return SX * v_x - SY * u_y

        def residual_loss(params, t, x, y, mask):
            uv = apply_fn(params, t, x, y)
            vec_t, vec_x, vec_y = jnp.ones(t.shape), jnp.ones(x.shape), jnp.ones(y.shape)
            w_t = ST * jvp(lambda t: vort(params, t, x, y), (t,), (vec_t,))[1]
            w_x = SX * jvp(lambda x: vort(params, t, x, y), (x,), (vec_x,))[1]
            w_y = SY * jvp(lambda y: vort(params, t, x, y), (y,), (vec_y,))[1]
            w_xx = SX ** 2 * hvp_fwdfwd(lambda x: vort(params, t, x, y), (x,), (vec_x,))
            w_yy = SY ** 2 * hvp_fwdfwd(lambda y: vort(params, t, x, y), (y,), (vec_y,))
            R_w = w_t + uv[0] * w_x + uv[1] * w_y - NU * (w_xx + w_yy)

            u_x = SX * jvp(lambda x: apply_fn(params, t, x, y)[0], (x,), (vec_x,))[1]
            v_y = SY * jvp(lambda y: apply_fn(params, t, x, y)[1], (y,), (vec_y,))[1]
            R_c = u_x + v_y
            # [sota patch] the cylinder is cut out of the residual average
            n = jnp.sum(mask) * t.shape[0] / mask.shape[0]
            return (jnp.sum((R_w ** 2) * mask) / n
                    + LBDA_C * jnp.sum((R_c ** 2) * mask) / n)

        def initial_loss(params):
            """Their initial_loss, on the task's t = 0 velocity. The vorticity
            term they also carry is dropped: the task ships u and v only, and
            the curl of a matched velocity is matched anyway."""
            u0, v0 = apply_fn(params, ic_t, ic_x, ic_y)
            n = jnp.sum(ic_m)
            return (jnp.sum(((u0 - ic_u) ** 2) * ic_m) / n
                    + jnp.sum(((v0 - ic_v) ** 2) * ic_m) / n)

        def boundary_loss(params, t, x, y, u_in):
            """[sota patch] the three velocity-Dirichlet boundaries. Their case
            is periodic and has no boundary loss at all."""
            lo_x = x[0:1] * 0.0 - 1.0                     # x = XL in net coords
            lo_y = y[0:1] * 0.0 - 1.0
            hi_y = y[0:1] * 0.0 + 1.0
            u_i, v_i = apply_fn(params, t, lo_x, y)       # inlet
            loss = jnp.mean((u_i - u_in) ** 2) + jnp.mean(v_i ** 2)
            for yy in (lo_y, hi_y):                       # channel walls
                u_w, v_w = apply_fn(params, t, x, yy)
                loss += jnp.mean(u_w ** 2) + jnp.mean(v_w ** 2)
            # cylinder: scattered, so the diagonal path
            u_c, v_c = apply_fn(params, cyl_t, cyl_xn, cyl_yn, diag=True)
            loss += jnp.mean(u_c ** 2) + jnp.mean(v_c ** 2)
            return loss

        loss_fn = lambda params: (residual_loss(params, tc, xc, yc, mask)
                                  + LBDA_IC * initial_loss(params)
                                  + LBDA_IC * boundary_loss(params, tc, xc, yc, u_in))
        return jax.value_and_grad(loss_fn)(params)

    optim = optax.adam(learning_rate=LR)
    state = optim.init(params)
    key, subkey = jax.random.split(key)
    train_data = generate_train_data(subkey)

    for e in range(1, int(steps) + 1):
        loss, gradient = apply_model_spinn(apply_fn, params, *train_data)
        params, state = update_model(optim, gradient, params, state)
        if e % 10000 == 0:
            print(f'Epoch: {e}/{steps} --> total loss: {loss:.8f}')

    def predict(prm, X):
        """X is (N,3) as (t, x, y). The scored set is a scattered subset of a
        tensor grid, so the factorised prediction is built on its distinct axis
        values and the points are gathered out of it."""
        Xn = np.asarray(X)
        tv, ti = np.unique(Xn[:, 0], return_inverse=True)
        xv, xi = np.unique(Xn[:, 1], return_inverse=True)
        yv, yi = np.unique(Xn[:, 2], return_inverse=True)
        t = jnp.asarray(_to_net(tv, 0.0, T_MAX)[:, None].astype(np.float32))
        x = jnp.asarray(_to_net(xv, XL, XU)[:, None].astype(np.float32))
        y = jnp.asarray(_to_net(yv, YL, YU)[:, None].astype(np.float32))
        u_g, v_g = apply_fn(prm, t, x, y)
        return {"u": jnp.asarray(np.asarray(u_g)[ti, xi, yi]),
                "v": jnp.asarray(np.asarray(v_g)[ti, xi, yi])}

    return {"params": params, "predict": predict,
            "info": {"mlp": MLP, "n_layers": N_LAYERS, "features": FEATURES,
                     "r": RANK, "out_dim": OUT_DIM, "pos_enc": POS_ENC,
                     "nt": NT, "nxy": NXY, "lr": LR, "epochs": int(steps),
                     "published_epochs": PUBLISHED_EPOCHS,
                     "lbda_c": LBDA_C, "lbda_ic": LBDA_IC,
                     "template": "navier_stokes3d.py (vorticity form), no time marching",
                     "nu": NU, "t_max": T_MAX,
                     "cylinder": [CX, CY, CR],
                     "residual_mask": "analytic: grid points inside the cylinder dropped",
                     "derived_from": "cases/cyl/methods/spinn/run.py; only U(t), T_MAX and the IC differ",
                     "ic": "u = v = 0 (analytic: the fluid starts at rest)",
                     "bc": "inlet profile + two walls + cylinder, all velocity Dirichlet",
                     "outlet": ("p = 0 at x = 2.2 has no expression in the vorticity "
                                "form -- taking the curl is what removes the pressure "
                                "-- so it is absent from the loss"),
                     "scattered_eval": "diagonal contraction einsum('fn,fn,fn->n')",
                     "input_rescale": "t,x,y mapped to [-1,1]; residual carries ST, SX, SY"}}
