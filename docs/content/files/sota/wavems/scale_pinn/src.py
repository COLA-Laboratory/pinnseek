"""Generated from plot_benchmark_gs.ipynb by tools/extract_scale_pinn.py.

The authors' code, unmodified: their definitions up to the first top-level
main(...) call. The experiment scripting after it is not included.
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
        # [sota patch] one field, not two: this task has a single u.
        self.last_layer_u = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

    @nn.compact
    def __call__(self, inputs):
        t, x, y = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
        def get_u(t, x, y):
            # [sota patch] the donor's t window is 0.78 long, this task's is 5.
            # Their first layer multiplies by 2*pi, so a raw `t - t_l` up to 5
            # puts the sin feature at ~31 rad and the run diverges (measured:
            # 2.03 at 3000 iters, 2.7e5 at 30000). The two time features are
            # scaled to [-1,1]; derivatives are still taken w.r.t. physical t,
            # so no chain-rule factor is needed and the PDE is unchanged.
            _st = 2.0 / (self.t_u - self.t_l)
            f = jnp.hstack([_st * (t - self.t_l) - 1.0, x, y,
                            _st * (t - self.t_u) + 1.0])
            fs = [f]
            for i, lyr in enumerate(self.layers):
                f = lyr(f)
                if (i == 0):
                    f = 2 *jnp.pi *f
                if (i%2 != 0):
                    fs.append(f)
            f = jnp.hstack(fs)
            return self.last_layer_u(f)

        u = get_u(t, x, y)

        # [sota patch] the donor is first order in time (u_t); this equation is
        # second order, so the t derivative is taken twice, exactly the way the
        # spatial ones already are.
        def get_u_dir(get_u, t, x, y):
            u_tt = jacfwd(jacfwd(get_u), argnums=0)(t, x, y)
            u_xx = jacfwd(jacfwd(get_u, argnums=1), argnums=1)(t, x, y)
            u_yy = jacfwd(jacfwd(get_u, argnums=2), argnums=2)(t, x, y)
            u_t  = jacfwd(get_u)(t, x, y)
            return u_tt, u_xx, u_yy, u_t

        f_dir_vmap = vmap(get_u_dir, in_axes=(None, 0, 0, 0))
        u_tt, u_xx, u_yy, u_t = f_dir_vmap(get_u, t, x, y)
        u_tt, u_xx, u_yy = u_tt[:,:,0,0], u_xx[:,:,0,0], u_yy[:,:,0,0]
        u_t = u_t[:,:,0]

        diff_u = u_xx + u_yy

        # [sota patch] PDE: u_tt - c(x,y)*(u_xx + u_yy) = 0. c is the task's
        # 256x256 table, interpolated bilinearly as problem.md specifies.
        c = c_at(x, y)
        pde_u = u_tt - c * diff_u

        # obtain BC/IC indices
        ic  = (t == t_l)
        bcx = (x == x_l) | (x == x_u)
        bcy = (y == y_l) | (y == y_u)

        # [sota patch] zero Neumann on the four sides, where the donor matched
        # opposite faces for a periodic domain. The normal derivative comes from
        # the same jacfwd the residual uses.
        def get_u_n(get_u, t, x, y):
            return jacfwd(get_u, argnums=1)(t, x, y), jacfwd(get_u, argnums=2)(t, x, y)
        n_vmap = vmap(get_u_n, in_axes=(None, 0, 0, 0))
        u_x, u_y = n_vmap(get_u, t, x, y)
        u_x, u_y = u_x[:,:,0], u_y[:,:,0]

        outputs = jnp.hstack([u, u_t, diff_u, pde_u, u_x, u_y, ic, bcx, bcy])
        return outputs


class DNN(nn.Module):
    """DNNs — [sota patch] one field, matching the PINN class above."""
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

    @nn.compact
    def __call__(self, inputs):
        t, x, y = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
        def get_u(t, x, y):
            # [sota patch] the donor's t window is 0.78 long, this task's is 5.
            # Their first layer multiplies by 2*pi, so a raw `t - t_l` up to 5
            # puts the sin feature at ~31 rad and the run diverges (measured:
            # 2.03 at 3000 iters, 2.7e5 at 30000). The two time features are
            # scaled to [-1,1]; derivatives are still taken w.r.t. physical t,
            # so no chain-rule factor is needed and the PDE is unchanged.
            _st = 2.0 / (self.t_u - self.t_l)
            f = jnp.hstack([_st * (t - self.t_l) - 1.0, x, y,
                            _st * (t - self.t_u) + 1.0])
            fs = [f]
            for i, lyr in enumerate(self.layers):
                f = lyr(f)
                if (i == 0):
                    f = 2 *jnp.pi *f
                if (i%2 != 0):
                    fs.append(f)
            f = jnp.hstack(fs)
            return self.last_layer_u(f)

        u = get_u(t, x, y)

        def get_u_dir(get_u, t, x, y):
            u_xx = jacfwd(jacfwd(get_u, argnums=1), argnums=1)(t, x, y)
            u_yy = jacfwd(jacfwd(get_u, argnums=2), argnums=2)(t, x, y)
            return u_xx, u_yy

        f_dir_vmap = vmap(get_u_dir, in_axes=(None, 0, 0, 0))
        u_xx, u_yy = f_dir_vmap(get_u, t, x, y)
        u_xx, u_yy = u_xx[:,:,0,0], u_yy[:,:,0,0]

        outputs = jnp.hstack([u, u_xx + u_yy])
        return outputs


def main(ER, ER_xx, weight_ic, weight_bc, max_lr, exponent, max_iter, seed, gpu):

    global c_at

    # [sota patch] the donor reads grey_scott.mat and carries six reaction
    # coefficients; this task's physics is one spatially varying coefficient,
    # read from the task's own 256x256 table. problem.md specifies bilinear
    # interpolation between grid points, so c_at interpolates rather than snaps.
    _coef = np.loadtxt('~/wavems/task/wavems_coef.csv',
                       delimiter=',', skiprows=1, dtype=np.float64)
    _cx, _cy = np.unique(_coef[:, 0]), np.unique(_coef[:, 1])
    _co = np.lexsort((_coef[:, 1], _coef[:, 0]))
    _cg = _coef[_co, 2].reshape(len(_cx), len(_cy))
    assert np.allclose(_coef[_co, 0].reshape(_cg.shape), _cx[:, None])
    cx_j, cy_j, cg_j = jnp.array(_cx), jnp.array(_cy), jnp.array(_cg)

    def c_at(x, y):
        fx = jnp.clip((x - cx_j[0]) / (cx_j[1] - cx_j[0]), 0.0, cx_j.shape[0] - 1.001)
        fy = jnp.clip((y - cy_j[0]) / (cy_j[1] - cy_j[0]), 0.0, cy_j.shape[0] - 1.001)
        i0, j0 = jnp.floor(fx).astype(jnp.int32), jnp.floor(fy).astype(jnp.int32)
        wx, wy = fx - i0, fy - j0
        return ((1 - wx) * (1 - wy) * cg_j[i0, j0] + wx * (1 - wy) * cg_j[i0 + 1, j0]
                + (1 - wx) * wy * cg_j[i0, j0 + 1] + wx * wy * cg_j[i0 + 1, j0 + 1])

    # [sota patch] the task's reference grid in place of theirs. Same role:
    # collocation, the IC labels and the progress print.
    _ref = np.loadtxt('~/wavems/task/ref_data.csv',
                      delimiter=',', skiprows=1, dtype=np.float64)
    data_X, data_Y = _ref[:, 0:3], _ref[:, 3:4]

    global t_l, t_u, x_l, x_u, y_l, y_u

    t_l, t_u = np.min(data_X[:,0]), np.max(data_X[:,0])
    x_l, x_u, y_l, y_u = np.min(data_X[:,1]), np.max(data_X[:,1]), np.min(data_X[:,2]), np.max(data_X[:,2])

    # [sota patch] the donor stitches time segments and can take its IC from a
    # previous segment's PINN. This task is solved in one window, so that path
    # is gone and the IC is the task's own analytic field, which the reference's
    # t = 0 slice reproduces to 5e-11.
    ic  = (data_X[:,0] == t_l)
    bcx = (data_X[:,1] == x_l) | (data_X[:,1] == x_u)
    bcy = (data_X[:,2] == y_l) | (data_X[:,2] == y_u)
    data_X_IC, data_Y_IC = data_X[ic], data_Y[ic]
    data_X_BC, data_Y_BC = data_X[bcx | bcy], data_Y[bcx | bcy]

    # convert to jnp
    data_X, data_Y = jnp.array(data_X), jnp.array(data_Y)
    data_X_IC, data_Y_IC, data_X_BC, data_Y_BC = jnp.array(data_X_IC), jnp.array(data_Y_IC), jnp.array(data_X_BC), jnp.array(data_Y_BC)
    
    
    # choose seed
    seed = seed
    key, rng = random.split(random.PRNGKey(seed))

    # dummy input
    a = random.normal(key, [1,3])

    # initialization call
    n_nodes = 128
    model, model_0 = PINN(n_nodes, t_l, t_u), DNN(n_nodes, t_l, t_u)
    params = model.init(key, a)
    num_params, format_params_fn = get_params_format_fn(params)
    # print (num_params)

    # flatten initial params
    params = jax.flatten_util.ravel_pytree(params)[0]

    # update u_0
    params_0 = params

    # minibatch (set #sample)
    BS_ALL = 1000
    BS_IC = 50
    BS_BC = 50
    
    n_all, n_ic, n_bc = len(data_X), len(data_X_IC), len(data_X_BC)

    @jit
    def minibatch(key):
        key1, key2, key3, key4, key5, key6 = key
        batch_all = random.choice(key1, n_all, (BS_ALL - BS_IC - BS_BC,))
        batch_ic = random.choice(key2, n_ic, (BS_IC,)) 
        batch_bc = random.choice(key3, n_bc, (BS_BC,)) 
        batch_R = jnp.hstack([random.uniform(key4, shape=(BS_ALL, 1), minval=t_l, maxval=t_u),
                              random.uniform(key5, shape=(BS_ALL, 1), minval=x_l, maxval=x_u),
                              random.uniform(key6, shape=(BS_ALL, 1), minval=y_l, maxval=y_u)])
        batch_X = jnp.vstack([data_X[batch_all], data_X_IC[batch_ic], data_X_BC[batch_bc], batch_R])
        batch_Y = jnp.vstack([data_Y[batch_all], data_Y_IC[batch_ic], data_Y_BC[batch_bc]])
        return (batch_X, batch_Y)   

    # loss function
    def eval_loss(params, params_0, inputs, labels):
        pred = model.apply(format_params_fn(params), inputs)
        u, u_t, diff_u, pde_u, u_x, u_y, ic, bcx, bcy = jnp.split(pred, 9, axis=1)
        # stable evolution
        pred_0 = model_0.apply(format_params_fn(params_0), inputs)
        u_0, diff_u_0 = jnp.split(pred_0, 2, axis=1)
        if (ER > 0):
            # [sota patch] their ER correction, with c(x,y) where the donor has
            # its constant diffusivity: the second term is the high-order part
            # of the operator, which here is c times the Laplacian.
            c = c_at(inputs[:, 1:2], inputs[:, 2:3])
            pde_u = pde_u + (u - u_0) / ER - c*(diff_u - diff_u_0) / ER_xx  # ER @PDE
        pde_loss = jnp.mean(jnp.square(pde_u))
        # remove random uniform sample
        u, u_t, ic = u[:BS_ALL], u_t[:BS_ALL], ic[:BS_ALL]
        # [sota patch] two initial terms, not one: the task prescribes u(0,.)
        # and u_t(0,.) = 0. The donor is first order in time and has only the
        # first. Both are carried at the same weight, as the donor carries its
        # single one.
        ic_e = (labels - u) * ic
        ic_t_e = u_t * ic
        ic_loss = (jnp.sum(jnp.square(ic_e)) + jnp.sum(jnp.square(ic_t_e))) / ic.sum()
        # [sota patch] zero Neumann in place of the donor's periodic matching.
        bcx_e = u_x * bcx
        bcy_e = u_y * bcy
        bc_loss = jnp.sum(jnp.square(bcx_e)) / bcx.sum() + jnp.sum(jnp.square(bcy_e)) / bcy.sum()
        # mse
        mse = jnp.mean(jnp.square(labels - u))
        rl2 = jnp.linalg.norm(labels - u) / jnp.linalg.norm(labels)
        loss = pde_loss + weight_ic* ic_loss + weight_bc* bc_loss
        return loss, (mse, rl2)

    loss_grad = jax.jit(jax.value_and_grad(eval_loss, has_aux=True))

    # weights update
    @jit
    def update(params, params_0, opt_state, key):
        batch_X, batch_Y = minibatch(key)
        (loss, (mse, rl2)), grad = loss_grad(params, params_0, batch_X, batch_Y)
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

    # training iteration
    runtime = 0
    train_iters = 0

    #store = []
    store_nnws = []
    store_nnws.append([0, format_params_fn(params)])

    while (train_iters <= max_iters) and (runtime < 10000):
        # mini-batch update
        start = time.time()
        key1, key2, key3, key4, key5, key6, rng = random.split(rng, 7) # update random generator
        params, params_0, opt_state, loss, mse, rl2 = update(params, params_0, opt_state, (key1, key2, key3, key4, key5, key6))
        end = time.time()
        runtime += (end-start)
        # append weights
        if (train_iters % 3000 == 0):
            print ('iter. = %05d,  time = %03ds,  loss = %.2e  |  mse = %.2e,  rl2 = %.2e'%(train_iters, runtime, loss, mse, rl2))
            # store.append([train_iters, runtime, loss, mse, rl2])
            store_nnws.append([runtime, format_params_fn(params)])
        train_iters += 1

    #store = jnp.array(store)

    """PINN solution"""

    # [sota patch] the donor reports on its last time slice and writes a results
    # file for its segment-stitching loop; this case is scored from outside on
    # the whole window, so the report is over the whole reference grid and the
    # file writing is gone.
    pred = model_0.apply(format_params_fn(params), data_X)
    u, _ = jnp.split(pred, 2, axis=1)
    mse = jnp.mean(jnp.square(data_Y - u))
    rl2 = jnp.linalg.norm(data_Y - u) / jnp.linalg.norm(data_Y)
    print ('[T = %s - %s] :  MSE = %.2e  RL2 = %.2e'%(t_l, t_u, mse, rl2))

    return format_params_fn(params), store_nnws
