"""Generated from plot_benchmark_ks.ipynb by tools/extract_scale_pinn.py.

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
    def setup(self):
        self.layers = [nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       jnp.sin,
                       nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       nn.silu]
        self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

    @nn.compact
    def __call__(self, inputs):
        # split the two variables, probably just by slicing
        t, x = inputs[:,0:1], inputs[:,1:2]
        def get_u(t, x):
            f = jnp.hstack([t, x / jnp.pi / 2., t - t_u])
            fs = [f]
            for i, lyr in enumerate(self.layers):
                f = lyr(f)
                if (i == 0):
                    f = 4 *jnp.pi *f
                if (i%2 != 0):
                    fs.append(f)
            f = jnp.hstack(fs)
            u = self.last_layer(f)
            return u

        u = get_u(t, x)

        # obtain u_t, u_x, u_xx, u_xxxx
        def get_u_dir(get_u, t, x):
            u_t, u_x = jacfwd(get_u)(t, x), jacfwd(get_u, argnums=1)(t, x)
            u_xx = jacfwd(jacfwd(get_u, argnums=1), argnums=1)(t, x)
            u_xxxx = jacfwd(jacfwd(jacfwd(jacfwd(get_u, argnums=1), argnums=1), argnums=1), argnums=1)(t, x)
            return u_t, u_x, u_xx, u_xxxx

        f_dir_vmap = vmap(get_u_dir, in_axes=(None, 0, 0))
        u_t, u_x, u_xx, u_xxxx = f_dir_vmap(get_u, t, x)
        u_t, u_x, u_xx, u_xxxx = u_t[:,:,0], u_x[:,:,0], u_xx[:,:,0,0], u_xxxx[:,:,0,0,0,0]

        # PDE: u_t + v1*u*u_x + v2*u_xx + v3*u_xxxx = 0
        pde = u_t + v1*u*u_x + v2*u_xx + v3*u_xxxx

        # obtain BC/IC indices
        ic, bc = (t == t_l), (x == x_l)  # lower x
        pbc = get_u(t, jnp.ones_like(t)*x_u) - u

        outputs = jnp.hstack([u, u_xx, pde, pbc, ic, bc])
        return outputs


class DNN(nn.Module):
    """DNNs"""
    n_nodes: int
    def setup(self):

        self.layers = [nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       jnp.sin,
                       nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                       nn.silu]
        self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

    @nn.compact
    def __call__(self, inputs):
        # split the two variables, probably just by slicing
        t, x = inputs[:,0:1], inputs[:,1:2]
        def get_u(t, x):
            f = jnp.hstack([t, x / jnp.pi / 2., t - t_u])
            fs = [f]
            for i, lyr in enumerate(self.layers):
                f = lyr(f)
                if (i == 0):
                    f = 4 *jnp.pi *f
                if (i%2 != 0):
                    fs.append(f)
            f = jnp.hstack(fs)
            u = self.last_layer(f)
            return u

        u = get_u(t, x)

        # obtain u_t, u_x, u_xx, u_xxxx
        def get_u_dir(get_u, t, x):
            u_xx = jacfwd(jacfwd(get_u, argnums=1), argnums=1)(t, x)
            return u_xx

        f_dir_vmap = vmap(get_u_dir, in_axes=(None, 0, 0))
        u_xx = f_dir_vmap(get_u, t, x)[:,:,0,0]

        outputs = jnp.hstack([u, u_xx])
        return outputs

def main(ER, ER_xx, weight_ic, weight_bc, max_lr, exponent, max_iter, seed, gpu):
    
    
    global v1, v2, v3
    
    # parameter
    v1 = 100 / (16)
    v2 = 100 / (16**2)
    v3 = 100 / (16**4)
    
    # read data
    mat_data = io.loadmat('ks_chaotic.mat')
    
    # spatial resolution
    dx = jnp.pi * 2. / 511
    dt = 1. / 250
    
    sim_t = np.repeat(mat_data['t'], 512).reshape(-1,1)
    sim_x = np.tile(mat_data['x'], 251).reshape(-1,1)
    sim_u = mat_data['usol'].reshape(-1,1)
    
    # subset segment
    T = 0.4
    data_X, data_Y = np.hstack([sim_t, sim_x]), sim_u
    data_X, data_Y = data_X[data_X[:,0] <= T], data_Y[data_X[:,0] <= T]
    
    global t_l, t_u, x_l, x_u
    
    t_l, t_u, x_l, x_u = np.min(data_X[:,0]), np.max(data_X[:,0]), np.min(data_X[:,1]), np.max(data_X[:,1])
    
    # split into BC / IC data
    bic = (data_X[:,0] == 0) | (data_X[:,1] == x_l)
    data_X_BIC, data_Y_BIC = data_X[bic], data_Y[bic]
    
    # convert to jnp
    data_X, data_Y, data_X_BIC, data_Y_BIC = jnp.array(data_X), jnp.array(data_Y), jnp.array(data_X_BIC), jnp.array(data_Y_BIC)
    
    
    # choose seed
    seed = seed
    key, rng = random.split(random.PRNGKey(seed))

    # dummy input
    a = random.normal(key, [1,2])

    # initialization call
    n_nodes = 128
    model, model_0 = PINN(n_nodes), DNN(n_nodes)
    params = model.init(key, a)
    num_params, format_params_fn = get_params_format_fn(params)
    # print (num_params)

    # flatten initial params
    params = jax.flatten_util.ravel_pytree(params)[0]

    # update u_0
    params_0 = params

    # minibatch (set #sample)
    BS_ALL = 1000
    BS_BIC = 50

    n_all, n_bic = len(data_X), len(data_X_BIC)

    @jit
    def minibatch(key):
        key1, key2 = key
        batch_all = random.choice(key1, n_all, (BS_ALL - BS_BIC,))
        batch_bic = random.choice(key2, n_bic, (BS_BIC,))
        batch_X = jnp.vstack([data_X[batch_all], data_X_BIC[batch_bic]])
        batch_Y = jnp.vstack([data_Y[batch_all], data_Y_BIC[batch_bic]])
        return (batch_X, batch_Y)

    # loss function
    def eval_loss(params, params_0, inputs, labels):
        pred = model.apply(format_params_fn(params), inputs)
        u, u_xx, pde, pbc, ic, bc = jnp.split(pred, 6, axis=1)
        # stable evolution
        pred0 = model_0.apply(format_params_fn(params_0), inputs)
        u_0, u0_xx = jnp.split(pred0, 2, axis=1)
        if (ER > 0):
            pde = pde + (u - u_0) / ER - v2*(u_xx - u0_xx) / ER_xx  # ER @PDE
        pde_loss = jnp.mean(jnp.square(pde))
        # IC
        ic_e = (labels - u) * ic
        ic_loss = jnp.sum(jnp.square(ic_e)) / ic.sum()
        # BC (periodic)
        bc_e = pbc * bc
        bc_loss = jnp.sum(jnp.square(bc_e)) / bc.sum()
        # stable evolution
        #er_loss = jnp.mean(jnp.square((u - u_0))) / .2
        # mse
        mse = jnp.mean(jnp.square(labels - u))
        rl2 = jnp.linalg.norm(labels - u) / jnp.linalg.norm(labels)
        loss = pde_loss + weight_ic* ic_loss + weight_bc* bc_loss #+ er_loss
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
        key1, key2, rng = random.split(rng, 3) # update random generator
        params, params_0, opt_state, loss, mse, rl2 = update(params, params_0, opt_state, (key1, key2))
        end = time.time()
        runtime += (end-start)
        # append weights
        if (train_iters % 2000 == 0):
            print ('iter. = %05d,  time = %03ds,  loss = %.2e  |  mse = %.2e,  rl2 = %.2e'%(train_iters, runtime, loss, mse, rl2))
            # store.append([train_iters, runtime, loss, mse, rl2])
            store_nnws.append([runtime, format_params_fn(params)])
        train_iters += 1

    #store = jnp.array(store)

    """PINN solution"""

    # solution
    inputs, labels = data_X, data_Y
    pred = model.apply(format_params_fn(params), inputs)
    u, _, _, _, _, _ = jnp.split(pred, 6, axis=1)
    mse = jnp.mean(jnp.square(labels - u))
    rl2 = jnp.linalg.norm(labels - u) / jnp.linalg.norm(labels)
    print ('[v1=%.1e  v2=%.1e  v3=%.1e] :  MSE = %.2e  RL2 = %.2e'%(v1, v2, v3, mse, rl2))
    
    return format_params_fn(params), store_nnws