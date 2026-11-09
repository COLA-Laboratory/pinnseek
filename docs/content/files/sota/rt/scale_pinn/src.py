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

# [sota patch] the task's Boussinesq coefficients: Pr = 0.71, Ra = 1e6, so
# NU = sqrt(Pr/Ra) = 8.426150e-4 and KAPPA = 1/sqrt(Pr*Ra) = 1.186782e-3,
# the values problem.md states.
PR, RA = 0.71, 1e6
NU = float(np.sqrt(PR / RA))
KAPPA = float(1.0 / np.sqrt(PR * RA))


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
        # [sota patch] four fields, not two: this problem carries (u, v, p, T).
        self.last_layer_u = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
        self.last_layer_v = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
        self.last_layer_p = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
        self.last_layer_T = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

    @nn.compact
    def __call__(self, inputs):
        t, x, y = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
        def get_f(t, x, y):
            f = jnp.hstack([t - self.t_l, x, y, t - self.t_u])
            fs = [f]
            for i, lyr in enumerate(self.layers):
                f = lyr(f)
                if (i == 0):
                    f = 2 *jnp.pi *f
                if (i%2 != 0):
                    fs.append(f)
            f = jnp.hstack(fs)
            return (self.last_layer_u(f), self.last_layer_v(f),
                    self.last_layer_p(f), self.last_layer_T(f))

        u, v, p, Tf = get_f(t, x, y)

        # [sota patch] the Boussinesq system needs first derivatives of every
        # field and second derivatives of u, v and T (the pressure enters only
        # through its gradient).
        def get_dir(get_f, t, x, y):
            d_t = jacfwd(get_f)(t, x, y)
            d_x = jacfwd(get_f, argnums=1)(t, x, y)
            d_y = jacfwd(get_f, argnums=2)(t, x, y)
            dxx = jacfwd(jacfwd(get_f, argnums=1), argnums=1)(t, x, y)
            dyy = jacfwd(jacfwd(get_f, argnums=2), argnums=2)(t, x, y)
            return d_t, d_x, d_y, dxx, dyy

        f_dir_vmap = vmap(get_dir, in_axes=(None, 0, 0, 0))
        d_t, d_x, d_y, dxx, dyy = f_dir_vmap(get_f, t, x, y)
        u_t, v_t, _p_t, T_t = (a[:,:,0] for a in d_t)
        u_x, v_x, p_x, T_x   = (a[:,:,0] for a in d_x)
        u_y, v_y, p_y, T_y   = (a[:,:,0] for a in d_y)
        u_xx, v_xx, _pxx, T_xx = (a[:,:,0,0] for a in dxx)
        u_yy, v_yy, _pyy, T_yy = (a[:,:,0,0] for a in dyy)

        lap_u, lap_v, lap_T = u_xx + u_yy, v_xx + v_yy, T_xx + T_yy
        adv = lambda fx, fy: u*fx + v*fy

        # [sota patch] the task's four equations, Boussinesq at Pr=0.71, Ra=1e6:
        #   u_t + u u_x + v u_y + p_x - NU lap(u)      = 0
        #   v_t + u v_x + v v_y + p_y - NU lap(v) - T  = 0
        #   u_x + v_y                                   = 0
        #   T_t + u T_x + v T_y - KAPPA lap(T)          = 0
        pde_u = u_t + adv(u_x, u_y) + p_x - NU*lap_u
        pde_v = v_t + adv(v_x, v_y) + p_y - NU*lap_v - Tf
        pde_c = u_x + v_y
        pde_T = T_t + adv(T_x, T_y) - KAPPA*lap_T

        # obtain BC/IC indices
        ic  = (t == t_l)
        bcy = (y == y_l) | (y == y_u)          # the no-slip, no-flux walls
        bcx = (x == x_l)                        # periodic pair, matched below

        # [sota patch] periodic in x with period 1 (theirs matched a square);
        # the walls are Dirichlet, so their trace is used directly.
        u_xu, v_xu, p_xu, T_xu = get_f(t, jnp.ones_like(t)*x_u, y)
        pbc_u, pbc_v = u_xu - u, v_xu - v
        pbc_p, pbc_T = p_xu - p, T_xu - Tf

        outputs = jnp.hstack([u, v, p, Tf, lap_u, lap_v, lap_T,
                              pde_u, pde_v, pde_c, pde_T,
                              pbc_u, pbc_v, pbc_p, pbc_T, ic, bcx, bcy])
        return outputs


class DNN(nn.Module):
    """DNNs — [sota patch] four fields, matching the PINN class above."""
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
        self.last_layer_p = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
        self.last_layer_T = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

    @nn.compact
    def __call__(self, inputs):
        t, x, y = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
        def get_f(t, x, y):
            f = jnp.hstack([t - self.t_l, x, y, t - self.t_u])
            fs = [f]
            for i, lyr in enumerate(self.layers):
                f = lyr(f)
                if (i == 0):
                    f = 2 *jnp.pi *f
                if (i%2 != 0):
                    fs.append(f)
            f = jnp.hstack(fs)
            return (self.last_layer_u(f), self.last_layer_v(f),
                    self.last_layer_p(f), self.last_layer_T(f))

        u, v, p, Tf = get_f(t, x, y)

        def get_dir(get_f, t, x, y):
            dxx = jacfwd(jacfwd(get_f, argnums=1), argnums=1)(t, x, y)
            dyy = jacfwd(jacfwd(get_f, argnums=2), argnums=2)(t, x, y)
            return dxx, dyy

        f_dir_vmap = vmap(get_dir, in_axes=(None, 0, 0, 0))
        dxx, dyy = f_dir_vmap(get_f, t, x, y)
        u_xx, v_xx, _pxx, T_xx = (a[:,:,0,0] for a in dxx)
        u_yy, v_yy, _pyy, T_yy = (a[:,:,0,0] for a in dyy)

        outputs = jnp.hstack([u, v, p, Tf, u_xx + u_yy, v_xx + v_yy, T_xx + T_yy])
        return outputs


def main(ER, ER_xx, weight_ic, weight_bc, max_lr, exponent, max_iter, seed, gpu):

    # [sota patch] the task's reference grid in place of the donor's .mat.
    # Columns are t, x, y, u, v, T; the pressure is not supplied and is not
    # scored, so the IC pins u, v and T only.
    _ref = np.loadtxt('~/rt/task/ref_data.csv',
                      delimiter=',', skiprows=1, dtype=np.float64)
    data_X = _ref[:, 0:3]
    data_Y = np.stack([_ref[:, 3], _ref[:, 4], _ref[:, 5]], axis=1)   # u, v, T

    global t_l, t_u, x_l, x_u, y_l, y_u

    t_l, t_u = np.min(data_X[:,0]), np.max(data_X[:,0])
    # [sota patch] the donor took its domain from its data file, whose grid ran
    # to the domain edge. This grid is CELL-CENTRED and STRETCHED: x runs
    # 0.0015386..0.9984614 (span 0.99692, one cell short of the period 1) and y
    # runs 0.0015178..1.9984822, so neither wall nor either periodic face is
    # actually sampled. problem.md's domain is used instead --
    # x in [0,1] periodic, y in [0,2] with no-slip walls -- and both boundary
    # sets are built analytically below. Two consequences:
    #   * the periodic term becomes u(t, x_l+1, y) - u(t, x_l, y), i.e. the
    #     periodicity itself rather than a pairing one cell short of it
    #     (same trap and same patch as the ib and kf cases);
    #   * the Dirichlet walls land on y = 0 and y = 2 rather than on the
    #     near-wall cell centres, where the true field still reaches |u| 0.047
    #     and |T| 0.028. That is where spinn and dsgnar put them too.
    x_l, x_u = 0.0, 1.0
    y_l, y_u = 0.0, 2.0

    # [sota patch] one window, no segment stitching. The IC is the task's own
    # t = 0 fields (u, v, T); the walls are y = 0 and y = 2 where u = v = T = 0;
    # x is periodic with period 1 and is matched face to face in the network.
    ic  = (data_X[:,0] == t_l)
    data_X_IC, data_Y_IC = data_X[ic], data_Y[ic]
    _tv = np.unique(data_X[:,0])
    _n  = 33
    _T, _S = np.meshgrid(_tv, np.linspace(0.0, 1.0, _n), indexing='ij')
    _T, _S = _T.ravel(), _S.ravel()
    _walls = np.concatenate([np.stack([_T, _S, np.full_like(_T, v)], axis=1)
                             for v in (y_l, y_u)])                    # y = 0, 2
    _T2, _S2 = np.meshgrid(_tv, np.linspace(0.0, 2.0, 2*_n), indexing='ij')
    _T2, _S2 = _T2.ravel(), _S2.ravel()
    _per = np.stack([_T2, np.full_like(_T2, x_l), _S2], axis=1)       # x = 0
    data_X_BC = np.concatenate([_walls, _per])
    data_Y_BC = np.zeros((len(data_X_BC), 3))

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
        (u, v, p, Tf, lap_u, lap_v, lap_T,
         pde_u, pde_v, pde_c, pde_T,
         pbc_u, pbc_v, pbc_p, pbc_T, ic, bcx, bcy) = jnp.split(pred, 18, axis=1)
        # stable evolution
        pred_0 = model_0.apply(format_params_fn(params_0), inputs)
        u_0, v_0, p_0, T_0, lap_u0, lap_v0, lap_T0 = jnp.split(pred_0, 7, axis=1)
        if (ER > 0):
            # [sota patch] their ER correction, applied to each evolution
            # equation with that equation's own diffusivity in the high-order
            # half (the paper's S_Mu / S_Mv, with gamma the problem's own
            # diffusivity), plus the paper's S_Cn = (p^k - p^(k-1))/tau_sc on
            # the continuity residual -- the term the paper introduces
            # specifically "to explicitly provide a relation between pressure
            # and the continuity equation". Both donors (ns-fps, ns-ldc)
            # implement it, as do the kf, cyl, cyl2, wing3d and bfs ports here.
            pde_u = pde_u + (u - u_0) / ER - NU*(lap_u - lap_u0) / ER_xx
            pde_v = pde_v + (v - v_0) / ER - NU*(lap_v - lap_v0) / ER_xx
            pde_c = pde_c + (p - p_0) / ER
            pde_T = pde_T + (Tf - T_0) / ER - KAPPA*(lap_T - lap_T0) / ER_xx
        pde_loss = (jnp.mean(jnp.square(pde_u)) + jnp.mean(jnp.square(pde_v))
                    + jnp.mean(jnp.square(pde_c)) + jnp.mean(jnp.square(pde_T)))
        # remove random uniform sample
        u, v, Tf, ic = u[:BS_ALL], v[:BS_ALL], Tf[:BS_ALL], ic[:BS_ALL]
        # IC: the task's own t = 0 u, v and T (the pressure is not supplied)
        gt_u, gt_v, gt_T = jnp.split(labels, 3, axis=1)
        ic_e = (jnp.hstack([gt_u, gt_v, gt_T]) - jnp.hstack([u, v, Tf])) * ic
        ic_loss = jnp.sum(jnp.square(ic_e)) / ic.sum()
        # [sota patch] two different boundary kinds: Dirichlet walls in y where
        # u = v = T = 0, and a periodic pair in x matched for all four fields.
        wall_e = jnp.hstack([u, v, Tf]) * bcy[:BS_ALL]
        wall_loss = jnp.sum(jnp.square(wall_e)) / jnp.maximum(bcy[:BS_ALL].sum(), 1.0)
        per_e = jnp.hstack([pbc_u, pbc_v, pbc_p, pbc_T]) * bcx
        per_loss = jnp.sum(jnp.square(per_e)) / jnp.maximum(bcx.sum(), 1.0)
        bc_loss = wall_loss + per_loss
        # mse — only T is scored, so the report tracks T
        mse = jnp.mean(jnp.square(gt_T - Tf))
        rl2 = jnp.linalg.norm(gt_T - Tf) / jnp.linalg.norm(gt_T)
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

    # [sota patch] report over the whole window on T, the only scored field;
    # the donor reported on its last time slice and wrote a results file for
    # its segment loop, neither of which applies here.
    pred = model_0.apply(format_params_fn(params), data_X)
    _u, _v, _p, Tf, _lu, _lv, _lT = jnp.split(pred, 7, axis=1)
    gt_T = data_Y[:, 2:3]
    mse = jnp.mean(jnp.square(gt_T - Tf))
    rl2 = jnp.linalg.norm(gt_T - Tf) / jnp.linalg.norm(gt_T)
    print ('[T = %s - %s] :  MSE = %.2e  RL2(T) = %.2e'%(t_l, t_u, mse, rl2))

    return format_params_fn(params), store_nnws
