"""Scale-PINN's Grey-Scott code, re-pointed at Ginzburg-Landau.

Scale-PINN publishes ac / gs / kdv / ks / ns-fps / ns-ldc / ns-naca / ns-rb and
no Ginzburg-Landau case, so this is a port: everything that is the *method* is
their `gs/plot_benchmark_gs.ipynb` verbatim (the file this was generated from,
via tools/extract_scale_pinn.py — the sinusoidal-feature PINN body, the
concatenative skip connection, the He init, the minibatch construction, the
sequential-corrected loss with its ER / ER_xx terms, the warm-up cosine decay
schedule and the training loop). Grey-Scott is their closest published problem
to this one: 2D + time, two coupled fields, reaction-diffusion, periodic on
[-1, 1]^2 — the same shape, so the architecture transfers without a decision.

Changes from that file, each marked `[sota patch]` where it occurs:

  1. the coefficient globals `b1 b2 c1 c2 ep1 ep2` become GL's `eps k c`;
  2. `pde_u` / `pde_v` in `PINN.__call__` become the task's residual pair,
     u_t - eps lap u - k(u - u|A|^2 + c v|A|^2) and
     v_t - eps lap v - k(v - v|A|^2 - c u|A|^2);
  3. the ER_xx (residual-smoothing) term multiplies `eps` for both components
     instead of `ep1`/`ep2` — the paper sets that gamma to the problem's own
     diffusion coefficient (gamma_GSu = eps1, gamma_GSv = eps2), and GL has one
     diffusivity for both;
  4. the data file is `ginzburg_landau_square.mat` and the number of frames is
     read from it rather than hard-coded at 101 (it is also 101 here, but the
     hard-coded literal was a Grey-Scott fact).

Nothing else is touched: same layer stack, same 2*pi frequency annealing, same
BS_ALL/BS_IC/BS_BC, same loss weights and optimiser wiring.
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
        self.last_layer_u = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
        self.last_layer_v = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

    @nn.compact
    def __call__(self, inputs):
        # split the two variables, probably just by slicing
        t, x, y = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
        def get_uv(t, x, y):
            f = jnp.hstack([t - self.t_l, x, y, t - self.t_u])
            fs = [f]
            for i, lyr in enumerate(self.layers):
                f = lyr(f)
                if (i == 0):
                    f = 2 *jnp.pi *f
                if (i%2 != 0):
                    fs.append(f)
            f = jnp.hstack(fs)
            u, v = self.last_layer_u(f), self.last_layer_v(f)
            return (u, v)

        u, v = get_uv(t, x, y)
        
        # obtain u_t, u_xx, u_yy, v_t, v_xx, v_yy
        def get_uv_dir(get_uv, t, x, y):
            u_t, v_t = jacfwd(get_uv)(t, x, y)
            u_xx, v_xx = jacfwd(jacfwd(get_uv, argnums=1), argnums=1)(t, x, y)
            u_yy, v_yy = jacfwd(jacfwd(get_uv, argnums=2), argnums=2)(t, x, y)
            return u_t, u_xx, u_yy, v_t, v_xx, v_yy
        
        f_dir_vmap = vmap(get_uv_dir, in_axes=(None, 0, 0, 0))
        u_t, u_xx, u_yy, v_t, v_xx, v_yy = f_dir_vmap(get_uv, t, x, y)   
        u_t, u_xx, u_yy = u_t[:,:,0], u_xx[:,:,0,0], u_yy[:,:,0,0]
        v_t, v_xx, v_yy = v_t[:,:,0], v_xx[:,:,0,0], v_yy[:,:,0,0]

        diff_u = u_xx + u_yy
        diff_v = v_xx + v_yy
        
        # [sota patch] Ginzburg-Landau residual in place of Grey-Scott's:
        # PDE: u_t - eps*(u_xx + u_yy) - k*(u - u*|A|^2 + c*v*|A|^2) = 0
        # PDE: v_t - eps*(v_xx + v_yy) - k*(v - v*|A|^2 - c*u*|A|^2) = 0
        a2 = u*u + v*v
        pde_u = u_t - eps*(u_xx + u_yy) - k*(u - u*a2 + c*v*a2)
        pde_v = v_t - eps*(v_xx + v_yy) - k*(v - v*a2 - c*u*a2)
        
        # obtain BC/IC indices
        ic, bcx, bcy = (t == t_l), (x == x_l), (y == y_l)  # lower x, lower y
        u_xu, v_xu = get_uv(t, jnp.ones_like(t)*x_u, y)
        u_yu, v_yu = get_uv(t, x, jnp.ones_like(t)*y_u)
        pbcx_u, pbcx_v = u_xu - u, v_xu - v
        pbcy_u, pbcy_v = u_yu - u, v_yu - v

        outputs = jnp.hstack([u, v, diff_u, diff_v, pde_u, pde_v, pbcx_u, pbcx_v, pbcy_u, pbcy_v, ic, bcx, bcy])
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

    @nn.compact
    def __call__(self, inputs):
        # split the two variables, probably just by slicing
        t, x, y = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
        def get_uv(t, x, y):
            f = jnp.hstack([t - self.t_l, x, y, t - self.t_u])
            fs = [f]
            for i, lyr in enumerate(self.layers):
                f = lyr(f)
                if (i == 0):
                    f = 2 *jnp.pi *f
                if (i%2 != 0):
                    fs.append(f)
            f = jnp.hstack(fs)
            u, v = self.last_layer_u(f), self.last_layer_v(f)
            return (u, v)

        u, v = get_uv(t, x, y)

        # obtain u_t, u_xx, u_yy, v_t, v_xx, v_yy
        def get_uv_dir(get_uv, t, x, y):
            u_xx, v_xx = jacfwd(jacfwd(get_uv, argnums=1), argnums=1)(t, x, y)
            u_yy, v_yy = jacfwd(jacfwd(get_uv, argnums=2), argnums=2)(t, x, y)
            return u_xx, u_yy, v_xx, v_yy
        
        f_dir_vmap = vmap(get_uv_dir, in_axes=(None, 0, 0, 0))
        u_xx, u_yy, v_xx, v_yy = f_dir_vmap(get_uv, t, x, y)   
        u_xx, u_yy = u_xx[:,:,0,0], u_yy[:,:,0,0]
        v_xx, v_yy = v_xx[:,:,0,0], v_yy[:,:,0,0]

        diff_u = u_xx + u_yy
        diff_v = v_xx + v_yy

        outputs = jnp.hstack([u, v, diff_u, diff_v])
        return outputs    

def main(ER, ER_xx, weight_ic, weight_bc, max_lr, exponent, max_iter, seed, gpu, T_0, T_1, pred_T):
    
    
    # [sota patch] GL coefficients in place of Grey-Scott's six
    global eps, k, c
    
    # parameter
    eps, k, c = 0.004, 10.0, 1.5
    
    # read data
    # [sota patch] GL's reference file
    mat_data = io.loadmat('ginzburg_landau_square.mat')
    n_frames = mat_data['t'].size
    
    # spatial resolution
    sim_t = np.repeat(mat_data['t'],200*200).reshape(-1,1)
    sim_x1d, sim_y1d = mat_data['x'].reshape(-1,1), mat_data['y'].reshape(-1,1)
    sim_x, sim_y = np.meshgrid(sim_x1d, sim_y1d, indexing='ij')
    sim_x, sim_y = sim_x.reshape(1,-1), sim_y.reshape(1,-1)
    # [sota patch] frame count from the file instead of the literal 101
    sim_x, sim_y = np.tile(sim_x,n_frames).reshape(-1,1), np.tile(sim_y,n_frames).reshape(-1,1)
    sim_u, sim_v = mat_data['usol'].reshape(-1,1), mat_data['vsol'].reshape(-1,1)
    
    # subset segment
    data_X, data_Y = np.hstack([sim_t, sim_x, sim_y]), np.hstack([sim_u, sim_v])
    data_X, data_Y = data_X[(data_X[:,0] >= T_0) & (data_X[:,0] <= T_1)], data_Y[(data_X[:,0] >= T_0) & (data_X[:,0] <= T_1)]
    
    global t_l, t_u, x_l, x_u, y_l, y_u

    t_l, t_u = np.min(data_X[:,0]), np.max(data_X[:,0])
    x_l, x_u, y_l, y_u = np.min(data_X[:,1]), np.max(data_X[:,1]), np.min(data_X[:,2]), np.max(data_X[:,2])

    # IC from previous solution
    if len(pred_T) > 0:
        _t = (data_X[:,0] == t_l)
        data_Y[_t] = pred_T
        print ('Use previous PINN solution as IC.')
        
    # split into BC / IC data
    ic = (data_X[:,0] == t_l)
    bc = (data_X[:,1] == x_l) | (data_X[:,2] == y_l)
    data_X_IC, data_Y_IC, data_X_BC, data_Y_BC = data_X[ic], data_Y[ic], data_X[bc], data_Y[bc]
    
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
        u, v, diff_u, diff_v, pde_u, pde_v, pbcx_u, pbcx_v, pbcy_u, pbcy_v, ic, bcx, bcy = jnp.split(pred, 13, axis=1)
        # stable evolution
        pred_0 = model_0.apply(format_params_fn(params_0), inputs)
        u_0, v_0, diff_u_0, diff_v_0 = jnp.split(pred_0, 4, axis=1)
        if (ER > 0):
            # [sota patch] gamma = the problem's diffusivity, one for both fields
            pde_u = pde_u + (u - u_0) / ER - eps*(diff_u - diff_u_0) / ER_xx  # ER @PDE
            pde_v = pde_v + (v - v_0) / ER - eps*(diff_v - diff_v_0) / ER_xx  # ER @PDE
        pde_loss = jnp.mean(jnp.square(pde_u)) + jnp.mean(jnp.square(pde_v))
        # remove random uniform sample
        u, v, ic = u[:BS_ALL], v[:BS_ALL], ic[:BS_ALL]
        # IC
        ic_e = (labels - jnp.hstack([u, v])) * ic
        ic_loss = jnp.sum(jnp.square(ic_e)) / ic.sum()
        # BC (periodic)
        bcx_e = jnp.hstack([pbcx_u, pbcx_v]) * bcx
        bcy_e = jnp.hstack([pbcy_u, pbcy_v]) * bcy
        bc_loss = jnp.sum(jnp.square(bcx_e)) / bcx.sum() + jnp.sum(jnp.square(bcy_e)) / bcy.sum()
        # mse
        mse = jnp.mean(jnp.square(labels - jnp.hstack([u, v])))
        rl2 = jnp.linalg.norm(labels - jnp.hstack([u, v])) / jnp.linalg.norm(labels)
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

    # solution
    _t = (data_X[:,0] == t_u)
    inputs, labels = data_X[_t], data_Y[_t]
    pred = model_0.apply(format_params_fn(params), inputs)
    u, v, _, _ = jnp.split(pred, 4, axis=1)
    mse = jnp.mean(jnp.square(labels - jnp.hstack([u, v])))
    rl2 = jnp.linalg.norm(labels - jnp.hstack([u, v])) / jnp.linalg.norm(labels)
    print ('[T = %s - %s] :  MSE = %.2e  RL2 = %.2e'%(t_l, t_u, mse, rl2))

    # save outcome
    save_results = [ER, ER_xx, weight_ic, weight_bc, max_lr, exponent, max_iter, seed, round(runtime)]
    file_path = 'gs_T=0-2s_batch_run_%s.txt'%(gpu)
    with open(file_path, 'a') as f:
        for value in save_results:
            f.write(f"{value},")
        f.write(f"{mse:.2e},")
        f.write(f"{rl2:.2e}\n")

    # info. to pass down to next time window..
    params_T = format_params_fn(params)
    pred_T = jnp.hstack([u, v])
    
    return params_T, pred_T, store_nnws

# # T = 0 - 0.5
# params_T0, pred_T0 = main(ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000,
#                           max_lr=2e-3, exponent=1, max_iter=300000, seed=20, gpu=0,
#                           T_0=0.0, T_1=0.5, pred_T=[])