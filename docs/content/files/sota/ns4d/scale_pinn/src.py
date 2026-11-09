"""Scale-PINN on 3D unsteady Navier-Stokes — their Gray-Scott recipe, widened to 4D.

Scale-PINN publishes nothing 3-dimensional in space and nothing in
vorticity-velocity form. Its only 2D + time example is `gs`, and that is the
donor — the same one the `rt` and `heatlt` ports here use: same sin-feature
network with the skip-concat structure, same minibatch composition, same ER
correction, same warmup-cosine schedule. `src.py` is that notebook extracted by
tools/extract_scale_pinn.py and then changed in the places marked
`[sota patch]`. NATIVE = False.

What changed:

  * **four inputs, three fields.** Gray-Scott is `(t, x, y) -> (u, v)`; this is
    `(t, x, y, z) -> (ux, uy, uz)`. The feature map's
    `hstack([t - t_l, x, y, t - t_u])` gains the z column.

  * **the equation, in vorticity-velocity form.**

        w_t + (u . grad) w - (w . grad) u - nu lap(w) - f = 0,   div u = 0

    with `w = curl u` and `nu = 0.05`. The vorticity is **not** a network
    output, so every `w` above is a first derivative of the velocity and
    `lap(w)` is a *third* derivative of it. That is the one structural cost of
    this task: the donor's residual needed only second derivatives.

  * **the forcing** is `problem.md`'s three-component `f`, identical to SPINN's
    `navier_stokes4d_forcing_term`, which is where the task's own reference
    comes from.

  * **three periodic directions instead of two.** The donor matched `x_u` to
    `x_l` and `y_u` to `y_l`; here z joins them. The period is exactly `2 pi`
    and the domain endpoints are 0 and `2 pi`, so the matched pair is the
    periodicity itself — no cell-centred correction is needed (the `rt` and
    `ib` cases needed one; this grid does not).

  * **the ER correction is applied per evolution equation** — one for each
    vorticity component, each with `nu` in the high-order half. The divergence
    constraint carries no time derivative and so gets no proximal term, the
    same reason the donor's pressure does not.

  * the segment-stitching machinery and the results-file writing are removed;
    this task is solved in one window.

**Training reads only the initial condition.** `task/ns4d_ic.csv` supplies
`(u, v, w)` on a 32^3 grid at t = 0; the collocation points are drawn here from
the analytic box, and `task/ref_data.csv` is never opened. The donor drew its
collocation from its data file, which would have meant touching the scoring
truth's coordinates; it does not here.

Hyperparameters are the donor's own, from the paper's Table A1 row for
Gray-Scott and the notebook's live call:

    ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000, max_lr=2e-3, exponent=1

with 300000 published iterations. `max_iter` is also `decay_steps` of the
cosine schedule, so the step count is compressed to what fits the budget rather
than the schedule being cut off part-way.
"""

import jax
import flax
import optax
from jax import lax, random, numpy as jnp
from jax import random, grad, vmap, hessian, jacfwd, jit
from jax import config
from flax import linen as nn
from evojax.util import get_params_format_fn

import os
import time
import numpy as np
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
import argparse

# choose GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

# [sota patch] problem.md's constants
NU = 0.05
L = 2.0 * np.pi
T_MAX = 5.0
IC_CSV = '~/ns4d/task/ns4d_ic.csv'


def forcing(t, x, y, z):
    """[sota patch] problem.md's f — SPINN's navier_stokes4d_forcing_term."""
    e = jnp.exp(-18.0 * NU * t)
    return (-6.0 * e * jnp.sin(4.0 * y) * jnp.sin(2.0 * z),
            -6.0 * e * jnp.sin(4.0 * x) * jnp.sin(2.0 * z),
            6.0 * e * jnp.sin(4.0 * x) * jnp.sin(4.0 * y))


"""DNN / PINN   """


class PINN(nn.Module):
    """PINNs"""
    n_nodes: int
    t_l: float
    t_u: float

    def setup(self):
        self.layers = [nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       jnp.sin,
                       nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       nn.silu]
        # [sota patch] three velocity components; there is no pressure in the
        # vorticity-velocity form.
        self.last_layer_u = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
        self.last_layer_v = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
        self.last_layer_w = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

    def _trunk(self, t, x, y, z):
        # [sota patch] the donor's hstack([t - t_l, x, y, t - t_u]) gains z
        f = jnp.hstack([t - self.t_l, x, y, z, t - self.t_u])
        fs = [f]
        for i, lyr in enumerate(self.layers):
            f = lyr(f)
            if (i == 0):
                f = 2 * jnp.pi * f
            if (i % 2 != 0):
                fs.append(f)
        return jnp.hstack(fs)

    @nn.compact
    def __call__(self, inputs):
        t, x, y, z = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3], inputs[:,3:4]

        def get_f(t, x, y, z):
            f = self._trunk(t, x, y, z)
            return (self.last_layer_u(f), self.last_layer_v(f), self.last_layer_w(f))

        # [sota patch] the vorticity, as the autodiff curl of the velocity.
        # Written as its own function so the residual's w_t, grad w and lap w
        # are one, two and three derivatives of the network respectively.
        def get_w(t, x, y, z):
            d_x = jacfwd(get_f, argnums=1)(t, x, y, z)
            d_y = jacfwd(get_f, argnums=2)(t, x, y, z)
            d_z = jacfwd(get_f, argnums=3)(t, x, y, z)
            ux_y, ux_z = d_y[0][:,0], d_z[0][:,0]
            uy_x, uy_z = d_x[1][:,0], d_z[1][:,0]
            uz_x, uz_y = d_x[2][:,0], d_y[2][:,0]
            return (uz_y - uy_z, ux_z - uz_x, uy_x - ux_y)

        # the primal fields on the whole batch, as the donor does; only the
        # derivatives go through the vmap
        ux, uy, uz = get_f(t, x, y, z)                   # (N,1) each

        def get_dir(t, x, y, z):
            w = get_w(t, x, y, z)
            du_x = jacfwd(get_f, argnums=1)(t, x, y, z)
            du_y = jacfwd(get_f, argnums=2)(t, x, y, z)
            du_z = jacfwd(get_f, argnums=3)(t, x, y, z)
            dw_t = jacfwd(get_w)(t, x, y, z)
            dw_x = jacfwd(get_w, argnums=1)(t, x, y, z)
            dw_y = jacfwd(get_w, argnums=2)(t, x, y, z)
            dw_z = jacfwd(get_w, argnums=3)(t, x, y, z)
            wxx = jacfwd(jacfwd(get_w, argnums=1), argnums=1)(t, x, y, z)
            wyy = jacfwd(jacfwd(get_w, argnums=2), argnums=2)(t, x, y, z)
            wzz = jacfwd(jacfwd(get_w, argnums=3), argnums=3)(t, x, y, z)
            return w, du_x, du_y, du_z, dw_t, dw_x, dw_y, dw_z, wxx, wyy, wzz

        (ww, du_x, du_y, du_z, dw_t, dw_x, dw_y, dw_z,
         wxx, wyy, wzz) = vmap(get_dir, in_axes=(0, 0, 0, 0))(t, x, y, z)

        wx, wy, wz = ww                                  # (N,1) each
        ux_x, uy_x, uz_x = (a[:,:,0] for a in du_x)
        ux_y, uy_y, uz_y = (a[:,:,0] for a in du_y)
        ux_z, uy_z, uz_z = (a[:,:,0] for a in du_z)
        wx_t, wy_t, wz_t = (a[:,:,0] for a in dw_t)
        wx_x, wy_x, wz_x = (a[:,:,0] for a in dw_x)
        wx_y, wy_y, wz_y = (a[:,:,0] for a in dw_y)
        wx_z, wy_z, wz_z = (a[:,:,0] for a in dw_z)
        wx_xx, wy_xx, wz_xx = (a[:,:,0,0] for a in wxx)
        wx_yy, wy_yy, wz_yy = (a[:,:,0,0] for a in wyy)
        wx_zz, wy_zz, wz_zz = (a[:,:,0,0] for a in wzz)

        lwx = wx_xx + wx_yy + wx_zz
        lwy = wy_xx + wy_yy + wy_zz
        lwz = wz_xx + wz_yy + wz_zz

        fx, fy, fz = forcing(t, x, y, z)

        # [sota patch] the task's equations. adv = (u.grad)w, str = (w.grad)u.
        adv = lambda a, b, c: ux*a + uy*b + uz*c
        pde_x = (wx_t + adv(wx_x, wx_y, wx_z)
                 - (wx*ux_x + wy*ux_y + wz*ux_z) - NU*lwx - fx)
        pde_y = (wy_t + adv(wy_x, wy_y, wy_z)
                 - (wx*uy_x + wy*uy_y + wz*uy_z) - NU*lwy - fy)
        pde_z = (wz_t + adv(wz_x, wz_y, wz_z)
                 - (wx*uz_x + wy*uz_y + wz*uz_z) - NU*lwz - fz)
        pde_c = ux_x + uy_y + uz_z

        # obtain IC/BC indices
        ic = (t == t_l)
        bcx, bcy, bcz = (x == x_l), (y == y_l), (z == z_l)

        # [sota patch] periodic in x, y and z with period 2 pi; the endpoints
        # are 0 and 2 pi, so the matched pair is the periodicity itself
        one = jnp.ones_like(t)
        ax, ay, az = get_f(t, one*x_u, y, z), get_f(t, x, one*y_u, z), get_f(t, x, y, one*z_u)
        pbx_u, pbx_v, pbx_w = ax[0] - ux, ax[1] - uy, ax[2] - uz
        pby_u, pby_v, pby_w = ay[0] - ux, ay[1] - uy, ay[2] - uz
        pbz_u, pbz_v, pbz_w = az[0] - ux, az[1] - uy, az[2] - uz

        outputs = jnp.hstack([ux, uy, uz, wx, wy, wz, lwx, lwy, lwz,
                              pde_x, pde_y, pde_z, pde_c,
                              pbx_u, pbx_v, pbx_w, pby_u, pby_v, pby_w,
                              pbz_u, pbz_v, pbz_w, ic, bcx, bcy, bcz])
        return outputs


class DNN(nn.Module):
    """DNNs"""
    n_nodes: int
    t_l: float
    t_u: float

    def setup(self):
        self.layers = [nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       jnp.sin,
                       nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       nn.silu]
        self.last_layer_u = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
        self.last_layer_v = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
        self.last_layer_w = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

    def _trunk(self, t, x, y, z):
        f = jnp.hstack([t - self.t_l, x, y, z, t - self.t_u])
        fs = [f]
        for i, lyr in enumerate(self.layers):
            f = lyr(f)
            if (i == 0):
                f = 2 * jnp.pi * f
            if (i % 2 != 0):
                fs.append(f)
        return jnp.hstack(fs)

    @nn.compact
    def __call__(self, inputs):
        t, x, y, z = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3], inputs[:,3:4]

        def get_f(t, x, y, z):
            f = self._trunk(t, x, y, z)
            return (self.last_layer_u(f), self.last_layer_v(f), self.last_layer_w(f))

        def get_w(t, x, y, z):
            d_x = jacfwd(get_f, argnums=1)(t, x, y, z)
            d_y = jacfwd(get_f, argnums=2)(t, x, y, z)
            d_z = jacfwd(get_f, argnums=3)(t, x, y, z)
            ux_y, ux_z = d_y[0][:,0], d_z[0][:,0]
            uy_x, uy_z = d_x[1][:,0], d_z[1][:,0]
            uz_x, uz_y = d_x[2][:,0], d_y[2][:,0]
            return (uz_y - uy_z, ux_z - uz_x, uy_x - ux_y)

        ux, uy, uz = get_f(t, x, y, z)

        def get_dir(t, x, y, z):
            w = get_w(t, x, y, z)
            wxx = jacfwd(jacfwd(get_w, argnums=1), argnums=1)(t, x, y, z)
            wyy = jacfwd(jacfwd(get_w, argnums=2), argnums=2)(t, x, y, z)
            wzz = jacfwd(jacfwd(get_w, argnums=3), argnums=3)(t, x, y, z)
            return w, wxx, wyy, wzz

        ww, wxx, wyy, wzz = vmap(get_dir, in_axes=(0, 0, 0, 0))(t, x, y, z)
        wx, wy, wz = ww
        wx_xx, wy_xx, wz_xx = (a[:,:,0,0] for a in wxx)
        wx_yy, wy_yy, wz_yy = (a[:,:,0,0] for a in wyy)
        wx_zz, wy_zz, wz_zz = (a[:,:,0,0] for a in wzz)
        lwx = wx_xx + wx_yy + wx_zz
        lwy = wy_xx + wy_yy + wy_zz
        lwz = wz_xx + wz_yy + wz_zz

        outputs = jnp.hstack([ux, uy, uz, wx, wy, wz, lwx, lwy, lwz])
        return outputs


def main(ER, ER_xx, weight_ic, weight_bc, max_lr, exponent, max_iter, seed, gpu):

    global t_l, t_u, x_l, x_u, y_l, y_u, z_l, z_u

    # [sota patch] the task's own initial condition, and nothing else. The
    # collocation points are drawn from the analytic box below, so
    # task/ref_data.csv -- the scoring truth -- is never opened.
    _ic = np.loadtxt(IC_CSV, delimiter=',', skiprows=1, dtype=np.float64)
    n_ic_pts = len(_ic)
    data_X_IC = np.concatenate([np.zeros((n_ic_pts, 1)), _ic[:, 0:3]], axis=1)
    data_Y_IC = _ic[:, 3:6]                       # u, v, w at t = 0

    t_l, t_u = 0.0, T_MAX
    x_l, x_u = 0.0, float(L)
    y_l, y_u = 0.0, float(L)
    z_l, z_u = 0.0, float(L)

    data_X_IC, data_Y_IC = jnp.array(data_X_IC), jnp.array(data_Y_IC)

    # choose seed
    seed = seed
    key, rng = random.split(random.PRNGKey(seed))

    # dummy input
    a = random.normal(key, [1,4])                 # [sota patch] four inputs

    # initialization call
    n_nodes = 128
    model, model_0 = PINN(n_nodes, t_l, t_u), DNN(n_nodes, t_l, t_u)
    params = model.init(key, a)
    num_params, format_params_fn = get_params_format_fn(params)

    # flatten initial params
    params = jax.flatten_util.ravel_pytree(params)[0]

    # update u_0
    params_0 = params

    # minibatch (set #sample)
    BS_ALL = 1000
    BS_IC = 50
    BS_BC = 50

    n_ic = len(data_X_IC)

    @jit
    def minibatch(key):
        k1, k2, k3, k4, k5, k6 = key
        # [sota patch] the interior points are drawn from the box; the donor
        # took them from its data file, which here would be the scoring truth
        n_int = BS_ALL - BS_IC - BS_BC
        batch_int = jnp.hstack([
            random.uniform(k1, (n_int, 1), minval=t_l, maxval=t_u),
            random.uniform(k2, (n_int, 3), minval=x_l, maxval=x_u)])
        batch_ic = data_X_IC[random.choice(k3, n_ic, (BS_IC,))]
        # boundary rows sit on the lower face of one axis, chosen in thirds
        m = BS_BC // 3
        bc_raw = jnp.hstack([
            random.uniform(k4, (3 * m, 1), minval=t_l, maxval=t_u),
            random.uniform(k5, (3 * m, 3), minval=x_l, maxval=x_u)])
        bc_raw = bc_raw.at[0:m, 1].set(x_l)
        bc_raw = bc_raw.at[m:2*m, 2].set(y_l)
        bc_raw = bc_raw.at[2*m:3*m, 3].set(z_l)
        batch_R = jnp.hstack([
            random.uniform(k6, (BS_ALL, 1), minval=t_l, maxval=t_u),
            random.uniform(k6, (BS_ALL, 3), minval=x_l, maxval=x_u)])
        batch_X = jnp.vstack([batch_int, batch_ic, bc_raw, batch_R])
        batch_Y = data_Y_IC[random.choice(k3, n_ic, (BS_IC,))]   # same key: same rows
        return (batch_X, batch_Y)

    # loss function
    def eval_loss(params, params_0, inputs, labels, n_int):
        pred = model.apply(format_params_fn(params), inputs)
        (ux, uy, uz, wx, wy, wz, lwx, lwy, lwz,
         pde_x, pde_y, pde_z, pde_c,
         pbx_u, pbx_v, pbx_w, pby_u, pby_v, pby_w,
         pbz_u, pbz_v, pbz_w, ic, bcx, bcy, bcz) = jnp.split(pred, 26, axis=1)
        # stable evolution
        pred_0 = model_0.apply(format_params_fn(params_0), inputs)
        (ux0, uy0, uz0, wx0, wy0, wz0, lwx0, lwy0, lwz0) = jnp.split(pred_0, 9, axis=1)
        if (ER > 0):
            # [sota patch] their ER correction, one term per evolution
            # equation -- here the three vorticity components -- with nu in the
            # high-order half. The divergence constraint has no time derivative
            # and so gets no proximal term, as their pressure does not.
            pde_x = pde_x + (wx - wx0) / ER - NU*(lwx - lwx0) / ER_xx
            pde_y = pde_y + (wy - wy0) / ER - NU*(lwy - lwy0) / ER_xx
            pde_z = pde_z + (wz - wz0) / ER - NU*(lwz - lwz0) / ER_xx
        pde_loss = (jnp.mean(jnp.square(pde_x)) + jnp.mean(jnp.square(pde_y))
                    + jnp.mean(jnp.square(pde_z)) + jnp.mean(jnp.square(pde_c)))

        # IC: the task's own t = 0 velocity, on the rows that carry it
        lo, hi = n_int, n_int + BS_IC
        pred_ic = jnp.hstack([ux[lo:hi], uy[lo:hi], uz[lo:hi]])
        ic_loss = jnp.mean(jnp.square(pred_ic - labels))

        # [sota patch] three periodic directions instead of two
        per = (jnp.sum(jnp.square(jnp.hstack([pbx_u, pbx_v, pbx_w]) * bcx)) / jnp.maximum(bcx.sum(), 1.0)
               + jnp.sum(jnp.square(jnp.hstack([pby_u, pby_v, pby_w]) * bcy)) / jnp.maximum(bcy.sum(), 1.0)
               + jnp.sum(jnp.square(jnp.hstack([pbz_u, pbz_v, pbz_w]) * bcz)) / jnp.maximum(bcz.sum(), 1.0))
        bc_loss = per

        # [sota patch] no labels away from t = 0, so the progress numbers are
        # the initial-condition error and the raw residual
        mse = ic_loss
        rl2 = pde_loss
        loss = pde_loss + weight_ic* ic_loss + weight_bc* bc_loss
        return loss, (mse, rl2)

    loss_grad = jax.jit(jax.value_and_grad(eval_loss, has_aux=True), static_argnums=(4,))

    # weights update
    @jit
    def update(params, params_0, opt_state, key):
        batch_X, batch_Y = minibatch(key)
        (loss, (mse, rl2)), grad = loss_grad(params, params_0, batch_X, batch_Y,
                                             BS_ALL - BS_IC - BS_BC)
        updates, opt_state = optimizer.update(grad, opt_state)
        params_0 = params # update u_0
        params = optax.apply_updates(params, updates)
        return params, params_0, opt_state, loss, mse, rl2

    # optimizer
    max_iters = max_iter
    max_lr = max_lr
    lr_scheduler = optax.warmup_cosine_decay_schedule(init_value=max_lr, peak_value=max_lr, warmup_steps=0,
                                                    decay_steps=max_iters, end_value=1e-10, exponent=exponent)
    optimizer = optax.adam(learning_rate=lr_scheduler) # Choose the method
    opt_state = optimizer.init(params)

    """Training"""

    runtime = 0
    train_iters = 0

    while (train_iters <= max_iters):    # [sota patch] the wall-clock guard is
                                         # the benchmark's job, not the notebook's
        start = time.time()
        key1, key2, key3, key4, key5, key6, rng = random.split(rng, 7)
        params, params_0, opt_state, loss, mse, rl2 = update(
            params, params_0, opt_state, (key1, key2, key3, key4, key5, key6))
        end = time.time()
        runtime += (end-start)
        if (train_iters % 5000 == 0):
            print ('iter. = %05d,  time = %03ds,  loss = %.2e  |  ic = %.2e,  pde = %.2e'%(train_iters, runtime, loss, mse, rl2))
        train_iters += 1

    return format_params_fn(params), train_iters
