"""Generated from plot_benchmark_pinn_ldc3200.ipynb by tools/extract_scale_pinn.py.

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
    n_nodes:int
    def setup(self):
        # initialization
        kinit = jax.nn.initializers.he_uniform()
        # feature mapping later
        self.feats = nn.Dense(self.n_nodes * 4, kernel_init = kinit)
        # hidden layers
        self.layers = [nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu]
        # split layers
        self.splitu = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layeru = [nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitv = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layerv = [nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitp = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layerp = [nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]

    @nn.compact
    def __call__(self, inputs):
        # split the two variables, probably just by slicing
        x, y = inputs[:,0:1], inputs[:,1:2]
        def get_uvp(x, y):
            inputs = jnp.hstack([x, y, x - 1., y - 1.])
            # feature mapping
            hidden = self.feats(inputs)
            hidden = jnp.sin(2*jnp.pi*hidden)
            # share hidden layer
            for i, lyr in enumerate(self.layers):
                hidden = lyr(hidden)
            # split hidden layer
            u = self.splitu(hidden)
            for i, lyr in enumerate(self.layeru):
                u = lyr(u)
            v = self.splitv(hidden)
            for i, lyr in enumerate(self.layerv):
                v = lyr(v)
            p = self.splitp(hidden)
            for i, lyr in enumerate(self.layerp):
                p = lyr(p)
            return (u, v, p)

        u, v, p = get_uvp(x, y)

        # obtain u_x, u_y, v_x, v_y, p_x, p_y
        def get_uvp_xy(get_uvp, x, y):
            u_x, v_x, p_x = jacfwd(get_uvp)(x, y)
            u_y, v_y, p_y = jacfwd(get_uvp, argnums=1)(x, y)
            return u_x, u_y, v_x, v_y, p_x , p_y

        get_uvp_xy_vmap = vmap(get_uvp_xy, in_axes=(None, 0, 0))
        u_x, u_y, v_x, v_y, p_x, p_y = get_uvp_xy_vmap(get_uvp, x, y)
        u_x, u_y, v_x, v_y, p_x, p_y = u_x[:,:,0], u_y[:,:,0], v_x[:,:,0], v_y[:,:,0], p_x[:,:,0] , p_y[:,:,0]

       # obtain u_xx, u_yy, v_xx, v_yy, p_xx, p_yy
        def get_uvp_xxyy(get_uvp, x, y):
            u_xx, v_xx, p_xx = jacfwd(jacfwd(get_uvp))(x, y)
            u_yy, v_yy, p_yy = jacfwd(jacfwd(get_uvp, argnums=1), argnums=1)(x, y)
            return u_xx, u_yy, v_xx, v_yy, p_xx , p_yy

        get_uvp_xxyy_vmap = vmap(get_uvp_xxyy, in_axes=(None, 0, 0))
        u_xx, u_yy, v_xx, v_yy, p_xx, p_yy = get_uvp_xxyy_vmap(get_uvp, x, y)
        u_xx, u_yy, v_xx, v_yy, p_xx, p_yy = u_xx[:,:,0,0], u_yy[:,:,0,0], v_xx[:,:,0,0], v_yy[:,:,0,0], p_xx[:,:,0,0] , p_yy[:,:,0,0]

        # obtain BC indices
        bc  = (x == x_l) | (x == x_u) | (y == y_l) | (y == y_u)
        nbc = (~bc)

        # compute PDE residual : N-S equations
        # continuity = u_x + v_y
        # momentum_1 = u*u_x + v*u_y + p_x - 1.0/Re*(u_xx + u_yy)
        # momentum_2 = u*v_x + v*v_y + p_y - 1.0/Re*(v_xx + v_yy)
        residuals_continuity = u_x + v_y
        residuals_momentum_1 = u*u_x + v*u_y + p_x - 1.0/Re*(u_xx + u_yy)
        residuals_momentum_2 = u*v_x + v*v_y + p_y - 1.0/Re*(v_xx + v_yy)

        mom1 = -1.0/Re*(u_xx + u_yy)
        mom2 = -1.0/Re*(v_xx + v_yy)

        outputs = jnp.hstack([u, v, p, residuals_continuity, residuals_momentum_1, residuals_momentum_2, bc, nbc, mom1, mom2])
        return outputs


class DNN(nn.Module):
    """DNNs"""
    n_nodes:int
    def setup(self):
        # initialization
        kinit = jax.nn.initializers.he_uniform()
        # feature mapping later
        self.feats = nn.Dense(self.n_nodes * 4, kernel_init = kinit)
        # hidden layers
        self.layers = [nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu]
        # split layers
        self.splitu = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layeru = [nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitv = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layerv = [nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitp = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layerp = [nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]

    @nn.compact
    def __call__(self, inputs):
        # split the two variables, probably just by slicing
        x, y = inputs[:,0:1], inputs[:,1:2]
        def get_uvp(x, y):
            inputs = jnp.hstack([x, y, x - 1., y - 1.])
            # feature mapping
            hidden = self.feats(inputs)
            hidden = jnp.sin(2*jnp.pi*hidden)
            # share hidden layer
            for i, lyr in enumerate(self.layers):
                hidden = lyr(hidden)
            # split hidden layer
            u = self.splitu(hidden)
            for i, lyr in enumerate(self.layeru):
                u = lyr(u)
            v = self.splitv(hidden)
            for i, lyr in enumerate(self.layerv):
                v = lyr(v)
            p = self.splitp(hidden)
            for i, lyr in enumerate(self.layerp):
                p = lyr(p)
            return (u, v, p)

        u, v, p = get_uvp(x, y)

       # obtain u_xx, u_yy, v_xx, v_yy, p_xx, p_yy
        def get_uvp_xxyy(get_uvp, x, y):
            u_xx, v_xx, _ = jacfwd(jacfwd(get_uvp))(x, y)
            u_yy, v_yy, _ = jacfwd(jacfwd(get_uvp, argnums=1), argnums=1)(x, y)
            return u_xx, u_yy, v_xx, v_yy

        get_uvp_xxyy_vmap = vmap(get_uvp_xxyy, in_axes=(None, 0, 0))
        u_xx, u_yy, v_xx, v_yy = get_uvp_xxyy_vmap(get_uvp, x, y)
        u_xx, u_yy, v_xx, v_yy = u_xx[:,:,0,0], u_yy[:,:,0,0], v_xx[:,:,0,0], v_yy[:,:,0,0]

        mom1 = -1.0/Re*(u_xx + u_yy)
        mom2 = -1.0/Re*(v_xx + v_yy)

        #shift p_ref
        x_pref, y_pref = jnp.ones_like(x)*x_ref, jnp.ones_like(y)*y_ref
        _, _, pref = get_uvp(x_pref, y_pref)
        pout = p - pref

        outputs = jnp.hstack([u, v, pout, mom1, mom2])
        return outputs

def main(ER, ER_xx, weight_bc, max_lr, exponent, max_iters, seed, gpu):

    global  Re, x_l, x_u, y_l, y_u, x_ref, y_ref, dx, dy

    Re = 3200
    
    sim = pd.read_csv('~/reference/scale-pinn/ns-ldc/LDC_RE3200_100x100_subsample200x200.csv')  # [sota patch] absolute path to their own file
    data_X, data_Y = sim[['x', 'y']].values, sim[['u', 'v', 'p']].values
    
    # dx & dy
    dx = dy = 1 / (101-1)

    # exclude corner points
    x_l, x_u, y_l, y_u = np.min(data_X[:,0]), np.max(data_X[:,0]), np.min(data_X[:,1]), np.max(data_X[:,1])
    corners = (sim.x == x_l) & (sim.y == y_u) | (sim.x == x_u) & (sim.y == y_u) | (sim.x == x_l) & (sim.y == y_l) | (sim.x == x_u) & (sim.y == y_l)
    data_X, data_Y = data_X[~corners], data_Y[~corners]

    # pressure ref
    x_ref, y_ref = np.unique(sim.x), np.unique(sim.y)
    x_ref, y_ref = np.take(x_ref, x_ref.size //2), np.take(y_ref, y_ref.size //2)

    # split into BC data
    bc = (data_X[:,0] == x_l) | (data_X[:,0] == x_u) | (data_X[:,1] == y_l) | (data_X[:,1] == y_u)
    data_X_BC, data_Y_BC = data_X[bc], data_Y[bc]

    # convert to jnp
    data_X, data_Y, data_X_BC, data_Y_BC = jnp.array(data_X), jnp.array(data_Y), jnp.array(data_X_BC), jnp.array(data_Y_BC)

    
    # choose seed
    seed = seed
    key, rng = random.split(random.PRNGKey(seed))

    # dummy input
    a = random.normal(key, [1,2])

    # initialization call
    n_nodes = 32
    model, model_0 = PINN(n_nodes), DNN(n_nodes)
    params = model.init(key, a)
    num_params, format_params_fn = get_params_format_fn(params)

    # flatten initial params
    params = jax.flatten_util.ravel_pytree(params)[0]

    # update u_0
    params_0 = params

    # minibatch (set #sample)
    BS_ALL = 400
    BS_BC = 50

    n_all, n_bc = len(data_X), len(data_X_BC)

    @jit
    def minibatch(key):
        key1, key2 = key
        batch_all = random.choice(key1, n_all, (BS_ALL - BS_BC,))
        batch_bc = random.choice(key2, n_bc, (BS_BC,))
        batch_X = jnp.vstack([data_X[batch_all], data_X_BC[batch_bc]])
        batch_Y = jnp.vstack([data_Y[batch_all], data_Y_BC[batch_bc]])
        return (batch_X, batch_Y)

    # loss function
    def eval_loss(params, params_0, inputs, labels):
        pred = model.apply(format_params_fn(params), inputs)
        u, v, p, residuals_continuity, residuals_momentum_1, residuals_momentum_2, bc, nbc, m_1, m_2 = jnp.split(pred, 10, axis=1)
        gt_u, gt_v, gt_p = jnp.split(labels, 3, axis=1)
        # PDE loss unadjusted
        pde_uvp_uadj  = jnp.square(residuals_continuity) + jnp.square(residuals_momentum_1) + jnp.square(residuals_momentum_2)
        pde_loss_uadj = jnp.sum(pde_uvp_uadj *nbc) / nbc.sum()
        # stable evolution
        pred0 = model_0.apply(format_params_fn(params_0), inputs)
        u_0, v_0, p_0, m0_1, m0_2 = jnp.split(pred0, 5, axis=1)
        # PDE
        if (ER > 0):
            residuals_continuity = residuals_continuity + (p - p_0) / ER # ER @PDE
            residuals_momentum_1 = residuals_momentum_1 + (u - u_0) / ER + (m_1 - m0_1) / ER_xx # ER @PDE
            residuals_momentum_2 = residuals_momentum_2 + (v - v_0) / ER + (m_2 - m0_2) / ER_xx # ER @PDE
        pde_uvp  = jnp.square(residuals_continuity) + jnp.square(residuals_momentum_1) + jnp.square(residuals_momentum_2)
        pde_loss = jnp.sum(pde_uvp *nbc) / nbc.sum()
        # BC
        bc_loss = jnp.sum(jnp.square(u - gt_u) *bc) / bc.sum() + jnp.sum(jnp.square(v - gt_v) *bc) / bc.sum()
        # mse
        uv, gt_uv = jnp.hstack([u, v]), jnp.hstack([gt_u, gt_v])
        mse = jnp.mean(jnp.square(uv - gt_uv))
        rl2 = jnp.linalg.norm(uv - gt_uv) / jnp.linalg.norm(gt_uv)
        loss = pde_loss + weight_bc* bc_loss
        return loss, (mse, rl2, pde_loss, pde_loss_uadj)

    loss_grad = jax.jit(jax.value_and_grad(eval_loss, has_aux=True))

    @jit
    def eval_ghia(params, x_ref_ghia, y_ref_ghia, v_ref_ghia, u_ref_ghia):
        # prediction
        u_pinn_ghia = model_0.apply(format_params_fn(params), y_ref_ghia)[:,0]
        v_pinn_ghia = model_0.apply(format_params_fn(params), x_ref_ghia)[:,1]
        ghia_uv, pinn_uv = jnp.vstack([u_ref_ghia, v_ref_ghia]).T, jnp.vstack([u_pinn_ghia, v_pinn_ghia]).T
        ghia_mse = jnp.mean(jnp.square(ghia_uv - pinn_uv))
        ghia_rl2 = jnp.linalg.norm(ghia_uv - pinn_uv) / jnp.linalg.norm(ghia_uv)
        return ghia_mse, ghia_rl2, pinn_uv

    # weights update
    @jit
    def update(params, params_0, opt_state, key):
        batch_X, batch_Y = minibatch(key)
        (loss, (mse, rl2, pde_loss, pde_loss_uadj)), grad = loss_grad(params, params_0, batch_X, batch_Y)
        updates, opt_state = optimizer.update(grad, opt_state)
        params_0 = params # update u_0
        params = optax.apply_updates(params, updates)
        return params, params_0, opt_state, loss, mse, rl2, pde_loss, pde_loss_uadj

    # optimizer
    max_iters = max_iters
    max_lr = max_lr
    lr_scheduler = optax.warmup_cosine_decay_schedule(init_value=max_lr, peak_value=max_lr, warmup_steps=0,
                                                      decay_steps=max_iters, end_value=1e-10 ,exponent=exponent)
    optimizer = optax.adam(learning_rate=lr_scheduler) # Choose the method
    opt_state = optimizer.init(params)

    """Training"""

    # training iteration
    runtime = 0
    train_iters = 0

    # store = []
    # store_ghia = []
    # store_loss = []
    store_nnws = []
    store_nnws.append([0, format_params_fn(params)])
    
    # CFD @ Ghia cut lines
    # [sota patch] absolute paths to their own Ghia reference cuts. The
    # separator is also rewritten from '\\s+' to r'\s+' -- the same Python
    # string either way, so this is cosmetic and changes nothing.
    u_ghia, v_ghia = (pd.read_csv('~/reference/scale-pinn/ns-ldc/LDC_512x512_y_u.dat', sep=r'\s+'),
                      pd.read_csv('~/reference/scale-pinn/ns-ldc/LDC_512x512_x_v.dat', sep=r'\s+'))
    x_ghia, y_ghia = v_ghia.x.values, u_ghia.y.values
    x_ref_ghia, y_ref_ghia = jnp.vstack([x_ghia, jnp.ones_like(x_ghia) *0.5]).T, jnp.vstack([jnp.ones_like(y_ghia) *0.5, y_ghia]).T
    v_ref_ghia, u_ref_ghia = v_ghia.v.values, u_ghia.u.values  

    while (train_iters <= max_iters) and (runtime < 300):
        # mini-batch update
        start = time.time()
        key1, key2, rng = random.split(rng, 3) # update random generator
        params, params_0, opt_state, loss, mse, rl2, pde_loss, pde_loss_uadj = update(params, params_0, opt_state, (key1, key2))
        end = time.time()
        runtime += (end-start)
        # store_loss.append([train_iters, loss, mse, rl2, pde_loss, pde_loss_uadj])
        # append weights
        if (train_iters % 500 == 0):
            #print ('iter. = %05d,  time = %03ds,  loss = %.2e  |  mse = %.2e,  rl2 = %.2e'%(train_iters, runtime, loss, mse, rl2))
            #store.append([train_iters, runtime, loss, mse, rl2])
            ghia_mse, ghia_rl2, pinn_uv = eval_ghia(params, x_ref_ghia, y_ref_ghia, v_ref_ghia, u_ref_ghia)
            print ('iter. = %05d,  time = %03ds,  loss = %.2e  |  mse = %.2e,  rl2 = %.2e  |  Ghia mse = %.2e,  Ghia rl2 = %.2e'%(train_iters, runtime, 
                                                                                                                                  loss, mse, rl2,
                                                                                                                                  ghia_mse, ghia_rl2))
            # store.append([train_iters, runtime, loss, mse, rl2, ghia_mse, ghia_rl2])
            # store_ghia.append(pinn_uv)
            store_nnws.append([runtime, format_params_fn(params)])
        train_iters += 1

    # store, store_ghia, store_loss = jnp.array(store), np.array(store_ghia), np.array(store_loss)

    """PINN solution"""
    inputs, labels = data_X, data_Y
    prediction = model_0.apply(format_params_fn(params), inputs)
    u, v, _, _, _ = jnp.split(prediction, 5, axis=-1)
    gt_u, gt_v, gt_p = jnp.split(labels, 3, axis=-1)
    # prediction errors
    uv, gt_uv = jnp.hstack([u, v]), jnp.hstack([gt_u, gt_v])
    mse = jnp.mean(jnp.square(uv - gt_uv))
    rl2 = jnp.linalg.norm(uv - gt_uv) / jnp.linalg.norm(gt_uv)
    print ('[In-house] :  MSE = %.2e  RL2 = %.2e'%(mse, rl2))

    """prediction @ Ghia"""
    ghia_mse, ghia_rl2, _ = eval_ghia(params, x_ref_ghia, y_ref_ghia, v_ref_ghia, u_ref_ghia)
    print ('[Ghia]     :  MSE = %.2e  RL2 = %.2e'%(ghia_mse, ghia_rl2))

    # """prediction @ JAX-PI"""
    # from scipy import io
    # mat_data = io.loadmat('ldc_Re3200.mat')
    # xx, yy = mat_data['x'].reshape(-1,1), mat_data['y'].reshape(-1,1)
    # x_256, y_256 = jnp.meshgrid(xx[:,0], yy[:,0], indexing='ij')
    # u_256, v_256 = mat_data['u'], mat_data['v']
    # xy_256 = jnp.hstack([x_256.reshape(-1,1), y_256.reshape(-1,1)])
    # uv_256 = jnp.hstack([u_256.reshape(-1,1), v_256.reshape(-1,1)])
    # # prediction errors
    # uv_pinn = model_0.apply(format_params_fn(params), xy_256)[:,:2]
    # jaxpi_mse = jnp.mean(jnp.square(uv_256 - uv_pinn))
    # jaxpi_rl2 = jnp.linalg.norm(uv_256 - uv_pinn) / jnp.linalg.norm(uv_256)
    # print ('[JAX-PI]   :  MSE = %.2e  RL2 = %.2e'%(jaxpi_mse, jaxpi_rl2))
    
    return format_params_fn(params), store_nnws