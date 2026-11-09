"""Generated from plot_naca_1_re1000.ipynb by tools/extract_scale_pinn.py.

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

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# choose GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

class PINN(nn.Module):
    global nn_acf
    nn_acf = nn.silu
    
    """PINNs"""
    n_nodes: int    
    def setup(self):
        # initialization
        kinit = jax.nn.initializers.he_uniform()
        # feature mapping later
        self.feats = nn.Dense(self.n_nodes * 2, kernel_init = kinit)
        # hidden layers
        self.layers = [nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf]
        # split layers
        self.splitu = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layeru = [nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,                       
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitv = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layerv = [nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,                       
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitp = nn.Dense(self.n_nodes, kernel_init = kinit)      
        self.layerp = [nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,                       
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]     

    @nn.compact
    def __call__(self, inputs):
        # split the two variables, probably just by slicing
        x, y = inputs[:,0:1], inputs[:,1:2]
        def get_uvp(x, y):
            inputs = jnp.hstack([x / 5., y / 2., (x - 1.) / 5.])
            # feature mapping
            hidden = self.feats(inputs)
            hidden = jnp.sin(jnp.pi*hidden)
            # hidden = jnp.sin(hidden)

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

        def get_uvp_xy(get_uvp, x, y):
            u_x, v_x, p_x = jacfwd(get_uvp)(x, y)
            u_y, v_y, p_y = jacfwd(get_uvp, argnums=1)(x, y)

            return u_x, u_y, v_x, v_y, p_x , p_y
            
        get_uvp_xy_vmap = vmap(get_uvp_xy, in_axes=(None, 0, 0))
        u_x, u_y, v_x, v_y, p_x , p_y = get_uvp_xy_vmap(get_uvp, x, y)
        u_x, u_y, v_x, v_y, p_x , p_y = u_x[:,:,0], u_y[:,:,0], v_x[:,:,0], v_y[:,:,0], p_x[:,:,0] , p_y[:,:,0]

       # obtain u_xx, u_yy, v_xx, v_yy, p_xx, p_yy
        def get_uvp_xxyy(get_uvp, x, y):
            u_xx, v_xx, p_xx = jacfwd(jacfwd(get_uvp))(x, y)
            u_yy, v_yy, p_yy = jacfwd(jacfwd(get_uvp, argnums=1), argnums=1)(x, y)

            return u_xx, u_yy, v_xx, v_yy, p_xx , p_yy
        
        get_uvp_xxyy_vmap = vmap(get_uvp_xxyy, in_axes=(None, 0, 0))
        u_xx, u_yy, v_xx, v_yy, p_xx , p_yy = get_uvp_xxyy_vmap(get_uvp, x, y)
        u_xx, u_yy, v_xx, v_yy, p_xx , p_yy = u_xx[:,:,0,0], u_yy[:,:,0,0], v_xx[:,:,0,0], v_yy[:,:,0,0], p_xx[:,:,0,0] , p_yy[:,:,0,0]

        # obtain BC indices
        #nbc = (x != x_l) | (x != x_u) | (y != y_l) | (y != y_u)
        bc  = (x == x_l) | (x == x_u) | (y == y_l) | (y == y_u)
        nbc = (~bc)
        uv_bc = (x == x_l) | (y == y_l) | (y == y_u)
        p_bc = (x == x_u)

        # compute PDE residual : N-S equations
        # continuity = u_x + v_y
        # momentum_1 = u*u_x + v*u_y + p_x - 1.0/Re*(u_xx + u_yy)
        # momentum_2 = u*v_x + v*v_y + p_y - 1.0/Re*(v_xx + v_yy)
        residuals_continuity = u_x + v_y
        residuals_momentum_1 = u*u_x + v*u_y + p_x - 1.0/Re*(u_xx + u_yy)
        residuals_momentum_2 = u*v_x + v*v_y + p_y - 1.0/Re*(v_xx + v_yy)

        mom1 = -1.0/Re*(u_xx + u_yy)
        mom2 = -1.0/Re*(v_xx + v_yy)
        
        outputs = jnp.hstack([u, v, p, residuals_continuity, residuals_momentum_1, residuals_momentum_2, u_y, uv_bc, p_bc, nbc, mom1, mom2])
        return outputs 


class DNN(nn.Module):
    global nn_acf
    nn_acf = nn.silu    
    """DNNs"""
    n_nodes: int
    def setup(self):
        # initialization
        kinit = jax.nn.initializers.he_uniform()
        # feature mapping later
        self.feats = nn.Dense(self.n_nodes * 2, kernel_init = kinit)
        # hidden layers
        self.layers = [nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf]
        # split layers
        self.splitu = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layeru = [nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,                       
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitv = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layerv = [nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,                       
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitp = nn.Dense(self.n_nodes, kernel_init = kinit)      
        self.layerp = [nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,                       
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]    

    @nn.compact
    def __call__(self, inputs):
        # split the two variables, probably just by slicing
        x, y = inputs[:,0:1], inputs[:,1:2]
        def get_uvp(x, y):
            inputs = jnp.hstack([x / 5., y / 2., (x - 1.) / 5.])
            # feature mapping
            hidden = self.feats(inputs)
            # hidden = jnp.sin(hidden)
            hidden = jnp.sin(jnp.pi*hidden)

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

        def get_uvp_xxyy(get_uvp, x, y):
            u_xx, v_xx, _ = jacfwd(jacfwd(get_uvp))(x, y)
            u_yy, v_yy, _ = jacfwd(jacfwd(get_uvp, argnums=1), argnums=1)(x, y)

            return u_xx, u_yy, v_xx, v_yy
        
        get_uvp_xxyy_vmap = vmap(get_uvp_xxyy, in_axes=(None, 0, 0))
        u_xx, u_yy, v_xx, v_yy, = get_uvp_xxyy_vmap(get_uvp, x, y)
        u_xx, u_yy, v_xx, v_yy = u_xx[:,:,0,0], u_yy[:,:,0,0], v_xx[:,:,0,0], v_yy[:,:,0,0]

        mom1 = -1.0/Re*(u_xx + u_yy)
        mom2 = -1.0/Re*(v_xx + v_yy)

        outputs = jnp.hstack([u, v, p, mom1, mom2])
        return outputs 


# compute thickness
def naca0012_thickness(x, thickness=0.12):
    return 5 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 
                            0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

# sample points on airfoil
def sample_points_on_airfoil(num_points=100, thickness=0.12, equal_spacing=True):

    x_dense = np.linspace(0, 1, 1000)
    y_dense = naca0012_thickness(x_dense, thickness)
    
    # upper & lower
    x_upper, y_upper = x_dense[::-1], y_dense[::-1]
    x_lower, y_lower = x_dense[1:], -y_dense[1:]
    # both
    x_all, y_all = np.concatenate([x_upper, x_lower]), np.concatenate([y_upper, y_lower])
    
    # equal spacing
    if equal_spacing:
        ds = np.sqrt(np.diff(x_all)**2 + np.diff(y_all)**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        total_length = s[-1]
        s_uniform = np.linspace(0, total_length, num_points)
        fx, fy = interp1d(s, x_all), interp1d(s, y_all)
        x_sampled, y_sampled = fx(s_uniform), fy(s_uniform)
    else:
        half_points = num_points // 2
        x_upper_sampled = np.linspace(0, 1, half_points)
        y_upper_sampled = naca0012_thickness(x_upper_sampled, thickness)
        x_lower_sampled, y_lower_sampled = x_upper_sampled, -y_upper_sampled
        x_sampled, y_sampled = np.concatenate([x_upper_sampled, x_lower_sampled]), np.concatenate([y_upper_sampled, y_lower_sampled])
    
    # rotate anti-clockwise
    theta = np.deg2rad(-7)
    x_center, y_center = 0.5, 0
    x_trans = x_sampled - x_center
    y_trans = y_sampled - y_center
    x_rot = x_trans * np.cos(theta) - y_trans * np.sin(theta)
    y_rot = x_trans * np.sin(theta) + y_trans * np.cos(theta)
    x_sampled = x_rot + x_center
    y_sampled = y_rot + y_center
    
    return x_sampled, y_sampled

def main(ER, ER_xx, weight_bc, batch, max_lr, exponent, max_iter, seed, gpu):

    global Re, x_l, x_u, y_l, y_u
    
    # parameter
    Re = 1000

    # [sota patch] absolute path to their own file. The separator is also
    # rewritten from the notebook's '\\s+' to r'\s+' -- these are the SAME
    # Python string (both are backslash-s-plus), so this is cosmetic and
    # changes nothing.
    sim = pd.read_csv('~/reference/scale-pinn/ns-naca/naca0012_atk7_Re1000_801x401_ls.csv', sep=r'\s+')
    sim_x, sim_y, sim_ls = sim['x'].values.reshape(-1,1), sim['y'].values.reshape(-1,1), sim['phi'].values.reshape(-1,1) 
    sim_u, sim_v, sim_p = sim['u'].values.reshape(-1,1), sim['v'].values.reshape(-1,1), sim['p'].values.reshape(-1,1) 
    x_l, x_u, y_l, y_u = np.min(sim_x), np.max(sim_x), np.min(sim_y), np.max(sim_y)
    #ext = [x_l, x_u, y_l, y_u]
    #print (x_l, x_u, y_l, y_u)

    data_X, data_Y = np.hstack([sim_x, sim_y]), np.hstack([sim_u, sim_v, sim_p])

    dx = (np.unique(data_X[:,0])[1] - np.unique(data_X[:,0])[0])
    _inner = (sim_ls[:,0] <= 0.0) | ( abs(sim_ls[:,0] ) <= 0.1*dx)
    data_X, data_Y = data_X[_inner], data_Y[_inner]

    airfoil_x, airfoil_y = sample_points_on_airfoil(num_points=400, thickness=0.12, equal_spacing=True)
    airfoil_x, airfoil_y = airfoil_x.reshape(-1,1), airfoil_y.reshape(-1,1)
    airfoil_shape = np.hstack([airfoil_x, airfoil_y])
    airfoil_shape_value = np.zeros((len(airfoil_shape[:,0]),3))
    
    # print('the shape of airfoil shape is: ', airfoil_shape.shape)
    # print('data_X shape:',data_X.shape,'data_Y shape:',data_Y.shape)
    
    l_bc, r_bc, tb_bc = (data_X[:,0] == x_l), (data_X[:,0] == x_u), (data_X[:,1] == y_l) | (data_X[:,1] == y_u)
    data_X_l_BC, data_Y_l_BC = data_X[l_bc], data_Y[l_bc]
    data_X_r_BC, data_Y_r_BC = data_X[r_bc], data_Y[r_bc]
    data_X_tb_BC, data_Y_tb_BC = data_X[tb_bc], data_Y[tb_bc]
    data_X_airfoil_BC, data_Y_airfoil_BC = airfoil_shape, airfoil_shape_value
    
    # print('data_X_l_BC shape:',data_X_l_BC.shape,'data_Y_l_BC shape:',data_Y_l_BC.shape)
    # print('data_X_r_BC shape:',data_X_r_BC.shape,'data_Y_r_BC shape:',data_Y_r_BC.shape)
    # print('data_X_tb_BC shape:',data_X_tb_BC.shape,'data_Y_tb_BC shape:',data_Y_tb_BC.shape)
    # print('data_X_naca_BC shape:',data_X_airfoil_BC.shape,'data_Y_naca_BC shape:',data_Y_airfoil_BC.shape)

    # convert to jnp
    data_X, data_Y = jnp.array(data_X), jnp.array(data_Y) # flow domain
    data_X_l_BC, data_Y_l_BC = jnp.array(data_X_l_BC), jnp.array(data_Y_l_BC)
    data_X_r_BC, data_Y_r_BC = jnp.array(data_X_r_BC), jnp.array(data_Y_r_BC)
    data_X_tb_BC, data_Y_tb_BC = jnp.array(data_X_tb_BC), jnp.array(data_Y_tb_BC)
    data_X_airfoil_BC, data_Y_airfoil_BC = jnp.array(data_X_airfoil_BC), jnp.array(data_Y_airfoil_BC)

    BS_ALL = batch
    BS_l_BC, BS_r_BC, BS_tb_BC = int(len(data_X_l_BC)*0.1), int(len(data_X_r_BC)*0.1), int(len(data_X_tb_BC)*0.1)
    BS_airfoil_BC = int(len(data_X_airfoil_BC)*0.1)

    BS_PDE = BS_ALL - BS_l_BC - BS_r_BC - BS_tb_BC - BS_airfoil_BC  #- BS_interpolation # - BS_BC
    n_all, n_l_BC, n_r_BC, n_tb_BC, n_airfoil_BC  = len(data_X), len(data_X_l_BC), len(data_X_r_BC), len(data_X_tb_BC), len(data_X_airfoil_BC)
    print('The total number of samples:' ,n_all)
    print('BS_ALL:', BS_ALL, ' BS_PDE:',BS_PDE, ' BS_l_BC:',BS_l_BC, ' BS_r_BC:',BS_r_BC, ' BS_tb_BC:',BS_tb_BC, ' BS_airfoil_BC:',BS_airfoil_BC)

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
    #print (num_params)
    
    # flatten initial params
    params = jax.flatten_util.ravel_pytree(params)[0]  
    
    params_0 = params 
    params_interp = params

    @jit
    def minibatch(key):
        key1, key2, key3, key4, key5, key6 = key
        batch_all = random.choice(key1, n_all , (BS_PDE,))
        batch_bc_l = random.choice(key2, n_l_BC, (BS_l_BC,))   
        batch_bc_r = random.choice(key3, n_r_BC, (BS_r_BC,))   
        batch_bc_tb = random.choice(key4, n_tb_BC, (BS_tb_BC,))   
        batch_bc_airfoil = random.choice(key5, n_airfoil_BC, (BS_airfoil_BC,))
        batch_X = jnp.vstack([data_X[batch_all], 
                              data_X_l_BC[batch_bc_l], 
                              data_X_r_BC[batch_bc_r],
                              data_X_tb_BC[batch_bc_tb],
                              data_X_airfoil_BC[batch_bc_airfoil]])
        batch_Y = jnp.vstack([data_Y[batch_all], 
                              data_Y_l_BC[batch_bc_l],
                              data_Y_r_BC[batch_bc_r], 
                              data_Y_tb_BC[batch_bc_tb],
                              data_Y_airfoil_BC[batch_bc_airfoil]])
        return (batch_X, batch_Y)

    def eval_loss(params,params_0, inputs, labels):
        pred = model.apply(format_params_fn(params), inputs)
        u, v, p, residuals_continuity, residuals_momentum_1, residuals_momentum_2, u_y, uv_bc, p_bc, nbc, m_1, m_2 = jnp.split(pred, 12, axis=1)
        gt_u, gt_v, gt_p = jnp.split(labels, 3, axis=1)
        x, y = jnp.split(inputs, 2 , axis=1)
        # stable evolution
        pred0 = model_0.apply(format_params_fn(params_0), inputs)
        u_0, v_0, p_0, m0_1, m0_2 = jnp.split(pred0, 5, axis=1) 
        # PDE
        residuals_continuity = residuals_continuity + (p - p_0) / ER # ER @PDE
        residuals_momentum_1 = residuals_momentum_1 + (u - u_0) / ER + (m_1 - m0_1) / ER_xx # ER @PDE
        residuals_momentum_2 = residuals_momentum_2 + (v - v_0) / ER + (m_2 - m0_2) / ER_xx # ER @PDE
        pde_uvp  = jnp.square(residuals_continuity) + jnp.square(residuals_momentum_1) + jnp.square(residuals_momentum_2)
        pde_uvp, nbc = pde_uvp[:-BS_airfoil_BC], nbc[:-BS_airfoil_BC]  # remove airfoil points
        pde_loss = jnp.sum(pde_uvp*nbc) / nbc.sum()
    
        bc_u, bc_v = (u - gt_u), (v - gt_v)
        _tb = (y == y_l) | (y == y_u)
        bc_u = jnp.where(_tb, u_y, bc_u)
        bc_uv = jnp.square(bc_u) + jnp.square(bc_v)
        bc_uv_loss = jnp.sum(bc_uv*uv_bc) / uv_bc.sum()
        
        bc_p = (p)  # p_bc is the mask for p boundary condition
        bc_p_loss =  jnp.sum(jnp.square(bc_p)*p_bc) / p_bc.sum()  # p_bc is the mask for p boundary condition
        
        bc_u_airfoil, bc_v_airfoil = u[-BS_airfoil_BC:], v[-BS_airfoil_BC:]  # airfoil_BC is the mask for airfoil boundary condition
        bc_airfoil = jnp.mean(jnp.square(bc_u_airfoil)) + jnp.mean(jnp.square(bc_v_airfoil))  # airfoil_BC is the mask for airfoil boundary condition
        bc_loss = bc_uv_loss + bc_p_loss + bc_airfoil  # airfoil_BC is the mask for airfoil boundary condition
        
        loss = pde_loss + bc_loss*weight_bc
        gt_V = jnp.sqrt(gt_u**2 + gt_v**2)  # ground truth velocity magnitude
        V = jnp.sqrt(u**2 + v**2)  # predicted velocity magnitude
        mse_u = jnp.mean(jnp.square(u - gt_u)) 
        mse_v = jnp.mean(jnp.square(v - gt_v)) 
        mse_V = jnp.mean(jnp.square(V - gt_V))  # mean squared error of velocity magnitude
        l2_u = jnp.linalg.norm(u - gt_u) / jnp.linalg.norm(gt_u)
        l2_v = jnp.linalg.norm(v - gt_v) / jnp.linalg.norm(gt_v)
        l2_V = jnp.linalg.norm(V - gt_V) / jnp.linalg.norm(gt_V)  # relative l2 error of velocity magnitude
        
        return loss, (mse_u, mse_v, mse_V, l2_u, l2_v, l2_V, pde_loss, bc_loss)
    
    loss_grad = jax.jit(jax.value_and_grad(eval_loss, has_aux=True))    

    # weights update  
    @jit
    def update(params, params_0,opt_state, key):
        batch_X, batch_Y = minibatch(key)
        (loss, (mse_u, mse_v, mse_V, l2_u, l2_v, l2_V, pde_loss, bc_loss)), grad = loss_grad(params,params_0, batch_X, batch_Y)
        updates, opt_state = optimizer.update(grad, opt_state)
        params_0 = params # update u_0
        params = optax.apply_updates(params, updates)
        return params, params_0, opt_state, loss,mse_u, mse_v, mse_V, l2_u, l2_v, l2_V, pde_loss, bc_loss

    
    # optimizer
    max_iters = max_iter
    max_lr = max_lr
    lr_scheduler = optax.warmup_cosine_decay_schedule(init_value=max_lr, peak_value=max_lr, warmup_steps=int(0.0*max_iters),  
                                                      decay_steps=max_iters, end_value=1e-10,exponent=exponent)
    optimizer = optax.adam(learning_rate=lr_scheduler) # Choose the method
    opt_state = optimizer.init(params)

    # training iteration
    runtime = 0
    train_iters = 0
    
    store = []
    while (train_iters <= max_iters):
        # mini-batch update
        start = time.time()
        key1, key2, key3, key4, key5, key6, rng = random.split(rng, 7) # update random generator
        params, params_0, opt_state, loss, mse_u, mse_v, mse_V, l2_u, l2_v, l2_V, pde_loss, bc_loss = update(params,params_0, opt_state, (key1, key2,key3,key4,key5,key6))
        end = time.time()
        runtime += (end-start)    
        # append weights
        if (train_iters % 5000 == 0):
            print ('iter. = %05d,  time = %03ds,  loss = %.2e  |  mse_u = %.2e,  mse_v = %.2e,  rl2_u = %.2e,  rl2_v = %.2e'%(train_iters, runtime, loss, mse_u, mse_v, l2_u, l2_v))
            store.append([train_iters, runtime, loss, mse_u, mse_v, mse_V, l2_u, l2_v, l2_V, pde_loss, bc_loss])
        train_iters += 1
    
    store = jnp.array(store)

    """PINN solution"""
    # solution
    inputs, labels = data_X, data_Y
    uvp = model.apply(format_params_fn(params), inputs)
    u, v, p, residuals_continuity, residuals_momentum_1, residuals_momentum_2, uv_bc,p_bc, nbc,res_u,res_v,fx,fy,mom_x_solid,mom_y_solild,res_fx,res_fy = uvp[:,0:1], uvp[:,1:2], uvp[:,2:3], uvp[:,3:4], uvp[:,4:5], uvp[:,5:6], uvp[:,6:7], uvp[:,7:8], uvp[:,8:9], uvp[:,9:10], uvp[:,10:11], uvp[:,11:12], uvp[:,12:13], uvp[:,13:14], uvp[:,14:15],uvp[:,15:16],uvp[:,16:17]
    gt_u, gt_v,gt_p = jnp.split(labels, 3, axis=-1)
    gt_V = jnp.sqrt(gt_u**2 + gt_v**2)  # ground truth velocity magnitude
    V = jnp.sqrt(u**2 + v**2)  # predicted velocity magnitude
    # errors
    mse_u = jnp.mean(jnp.square(u - gt_u)) 
    mse_v = jnp.mean(jnp.square(v - gt_v)) 
    mse_V = jnp.mean(jnp.square(V - gt_V)) 
    mse_p = jnp.mean(jnp.square(p - gt_p))  # mean squared error of pressure
    l2_u = jnp.linalg.norm(u - gt_u) / jnp.linalg.norm(gt_u)
    l2_v = jnp.linalg.norm(v - gt_v) / jnp.linalg.norm(gt_v)
    l2_V = jnp.linalg.norm(V - gt_V) / jnp.linalg.norm(gt_V)  # relative l2 error of velocity magnitude
    l2_p = jnp.linalg.norm(p - gt_p) / jnp.linalg.norm(gt_p)  # relative l2 error of pressure
    print ('[FULLDOM] : mse_V = %.2e, rl2_V = %.2e'%(mse_V, l2_V))

    # inner domain
    _small = (data_X[:,0] <= 1.05) & (data_X[:,0] >= -0.05) & (data_X[:,1] <= 0.2) & (data_X[:,1] >= -0.2)
    data_X, data_Y = data_X[_small], data_Y[_small]
    inputs, labels = data_X, data_Y
    uvp = model.apply(format_params_fn(params), inputs)
    u, v, p, residuals_continuity, residuals_momentum_1, residuals_momentum_2, uv_bc,p_bc, nbc,res_u,res_v,fx,fy,mom_x_solid,mom_y_solild,res_fx,res_fy = uvp[:,0:1], uvp[:,1:2], uvp[:,2:3], uvp[:,3:4], uvp[:,4:5], uvp[:,5:6], uvp[:,6:7], uvp[:,7:8], uvp[:,8:9], uvp[:,9:10], uvp[:,10:11], uvp[:,11:12], uvp[:,12:13], uvp[:,13:14], uvp[:,14:15],uvp[:,15:16],uvp[:,16:17]
    gt_u, gt_v,gt_p = jnp.split(labels, 3, axis=-1)
    gt_V = jnp.sqrt(gt_u**2 + gt_v**2)  # ground truth velocity magnitude
    V = jnp.sqrt(u**2 + v**2)  # predicted velocity magnitude
    # errors
    mse_u = jnp.mean(jnp.square(u - gt_u)) 
    mse_v = jnp.mean(jnp.square(v - gt_v)) 
    mse_V = jnp.mean(jnp.square(V - gt_V)) 
    mse_p = jnp.mean(jnp.square(p - gt_p))  # mean squared error of pressure
    l2_u = jnp.linalg.norm(u - gt_u) / jnp.linalg.norm(gt_u)
    l2_v = jnp.linalg.norm(v - gt_v) / jnp.linalg.norm(gt_v)
    l2_V = jnp.linalg.norm(V - gt_V) / jnp.linalg.norm(gt_V)  # relative l2 error of velocity magnitude
    l2_p = jnp.linalg.norm(p - gt_p) / jnp.linalg.norm(gt_p)  # relative l2 error of pressure
    print ('[COMPACT] : mse_V = %.2e, rl2_V = %.2e'%(mse_V, l2_V))
    
    params_ = format_params_fn(params[:])
    
    return params_, u, v, p
