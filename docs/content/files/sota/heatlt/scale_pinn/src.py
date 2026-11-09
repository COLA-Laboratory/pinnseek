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

# [sota patch] the task's constants: D = 0.001 diffusivity, A = 5 source
# amplitude, from problem.md.
D_COEF = 0.001
A_COEF = 5.0


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
            # [sota patch] the donor's t window is 0.78 long, this task's is
            # 100 -- 128x. Their first layer multiplies by 2*pi, so a raw
            # `t - t_l` would put the sin feature at ~628 rad. The two time
            # features are scaled to [-1,1]; derivatives are still taken w.r.t.
            # physical t, so no chain-rule factor is needed and the PDE is
            # unchanged. (The same rescale is used in the wavems port, where
            # the raw form was measured to diverge; here it is the 128x window
            # ratio that motivates it, and no divergence was measured.)
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

        # [sota patch] first order in time, like the donor -- the wavems port
        # took the t derivative twice, this task takes it once.
        def get_u_dir(get_u, t, x, y):
            u_t  = jacfwd(get_u)(t, x, y)
            u_xx = jacfwd(jacfwd(get_u, argnums=1), argnums=1)(t, x, y)
            u_yy = jacfwd(jacfwd(get_u, argnums=2), argnums=2)(t, x, y)
            return u_t, u_xx, u_yy

        f_dir_vmap = vmap(get_u_dir, in_axes=(None, 0, 0, 0))
        u_t, u_xx, u_yy = f_dir_vmap(get_u, t, x, y)
        u_t = u_t[:,:,0]
        u_xx, u_yy = u_xx[:,:,0,0], u_yy[:,:,0,0]

        diff_u = u_xx + u_yy

        # [sota patch] PDE: u_t - D lap(u) - A sin(u^2)(1+2 sin(pi t/4))
        #                                    sin(4 pi x) sin(2 pi y) = 0.
        # Note the source depends on u, so u == 0 satisfies the whole equation
        # and every boundary exactly; only the initial condition rules it out.
        src = (A_COEF * jnp.sin(u ** 2)
               * (1.0 + 2.0 * jnp.sin(jnp.pi * t / 4.0))
               * jnp.sin(4.0 * jnp.pi * x) * jnp.sin(2.0 * jnp.pi * y))
        pde_u = u_t - D_COEF * diff_u - src

        # obtain BC/IC indices
        ic  = (t == t_l)
        bcx = (x == x_l) | (x == x_u)
        bcy = (y == y_l) | (y == y_u)

        outputs = jnp.hstack([u, diff_u, pde_u, ic, bcx, bcy])
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
            # [sota patch] the donor's t window is 0.78 long, this task's is
            # 100 -- 128x. Their first layer multiplies by 2*pi, so a raw
            # `t - t_l` would put the sin feature at ~628 rad. The two time
            # features are scaled to [-1,1]; derivatives are still taken w.r.t.
            # physical t, so no chain-rule factor is needed and the PDE is
            # unchanged. (The same rescale is used in the wavems port, where
            # the raw form was measured to diverge; here it is the 128x window
            # ratio that motivates it, and no divergence was measured.)
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

    # [sota patch] the task's reference grid in place of the donor's .mat.
    # It is a COMSOL mesh (501 snapshots on 16 x 12 = 192 nodes), not a tensor
    # grid, which is fine: their minibatch draws rows, it never reshapes.
    _ref = np.loadtxt('~/heatlt/task/ref_data.csv',
                      delimiter=',', skiprows=1, dtype=np.float64)
    data_X, data_Y = _ref[:, 0:3], _ref[:, 3:4]

    global t_l, t_u, x_l, x_u, y_l, y_u

    t_l, t_u = np.min(data_X[:,0]), np.max(data_X[:,0])
    # [sota patch] the donor took its domain from its data file, whose grid runs
    # to the domain edge. This task's reference is a COMSOL mesh whose 16 x 12
    # nodes are ALL INTERIOR (x in 0.03125..0.96875, y in 0.041667..0.958333),
    # so taking min/max would put the zero-Dirichlet trace on two interior lines
    # -- where the true field has rms 0.85 against a global rms of 1.066, on 27%
    # of the scored points -- and would shrink the collocation box to that same
    # rectangle. problem.md's domain is [0,1]^2 and that is what is used; the
    # boundary collocation is built analytically on the four real faces, which
    # is also where the hero and the other three baselines put it.
    x_l, x_u, y_l, y_u = 0.0, 1.0, 0.0, 1.0

    # [sota patch] one window, no segment stitching. The IC is the task's own
    # analytic sin(4 pi x) sin(3 pi y) and the four sides are u = 0.
    ic  = (data_X[:,0] == t_l)
    data_X_IC = data_X[ic]
    data_Y_IC = (np.sin(4*np.pi*data_X_IC[:,1]) * np.sin(3*np.pi*data_X_IC[:,2]))[:, None]
    _tv = np.unique(data_X[:,0])
    _s  = np.linspace(0.0, 1.0, 33)
    _T, _S = np.meshgrid(_tv, _s, indexing='ij')
    _T, _S = _T.ravel(), _S.ravel()
    data_X_BC = np.concatenate(
        [np.stack([_T, np.full_like(_T, v), _S], axis=1) for v in (x_l, x_u)]
        + [np.stack([_T, _S, np.full_like(_T, v)], axis=1) for v in (y_l, y_u)])
    data_Y_BC = np.zeros((len(data_X_BC), 1))

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
        u, diff_u, pde_u, ic, bcx, bcy = jnp.split(pred, 6, axis=1)
        pred_0 = model_0.apply(format_params_fn(params_0), inputs)
        u_0, diff_u_0 = jnp.split(pred_0, 2, axis=1)
        if (ER > 0):
            pde_u = pde_u + (u - u_0) / ER - D_COEF*(diff_u - diff_u_0) / ER_xx  # ER @PDE
        pde_loss = jnp.mean(jnp.square(pde_u))
        # remove random uniform sample; the masks pair with labels so they are
        # cut to BS_ALL too (the minibatch is labelled points then random ones)
        u, ic = u[:BS_ALL], ic[:BS_ALL]
        bcx, bcy = bcx[:BS_ALL], bcy[:BS_ALL]
        ic_e = (labels - u) * ic
        ic_loss = jnp.sum(jnp.square(ic_e)) / ic.sum()
        # [sota patch] Dirichlet u = 0 on the four sides, where the donor
        # matched opposite faces for a periodic domain
        bc_e = u * (bcx + bcy)
        bc_loss = jnp.sum(jnp.square(bc_e)) / jnp.maximum((bcx + bcy).sum(), 1.0)
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
