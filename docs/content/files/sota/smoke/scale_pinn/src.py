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

# ── [sota patch] the task's ERA5/GFAS coefficient fields ──────────────────────
# task/baseline.py:coef_at reproduced exactly: trilinear (order=1, as
# problem.md specifies) with the (t, y, x) storage order the task warns about,
# while our coordinates are (t, x, y). verify_coef.py checks this construction
# against the task's own function: max |diff| 0.000, versus up to 57 for the
# transposed order.
from jax.scipy.ndimage import map_coordinates as _map_coordinates
_SMOKE_TASK = "~/smoke/task"
_z = np.load(f"{_SMOKE_TASK}/smoke_coef.npz")
A_TAB, B_TAB, C_TAB, F_TAB = (jnp.asarray(_z[k], dtype=jnp.float32)
                              for k in ("a", "b", "c", "f"))
DXC, DYC = float(_z["dxc"]), float(_z["dyc"])
GRID_NT, GRID_NY, GRID_NX = 73, 153, 241
assert A_TAB.shape == (GRID_NT, GRID_NY, GRID_NX), A_TAB.shape


def coef_at(field, t, x, y):
    """task/baseline.py:coef_at. Fields are stored (t, y, x)."""
    coords = jnp.stack([t * (GRID_NT - 1), y * (GRID_NY - 1), x * (GRID_NX - 1)])
    return _map_coordinates(field, coords, order=1, mode="nearest")


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
            # the donor's own time framing, unchanged: this task's window is
            # [0,1] against their [0,0.78], so the sin feature sees the same
            # O(1) range they tuned it on and there is nothing to rescale.
            f = jnp.hstack([t - self.t_l, x, y, t - self.t_u])
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

        # [sota patch] this task is FIRST order in time (advection-diffusion),
        # so the t derivative is taken once -- the wavems port took it twice.
        def get_u_dir(get_u, t, x, y):
            u_t  = jacfwd(get_u)(t, x, y)
            u_x  = jacfwd(get_u, argnums=1)(t, x, y)
            u_y  = jacfwd(get_u, argnums=2)(t, x, y)
            u_xx = jacfwd(jacfwd(get_u, argnums=1), argnums=1)(t, x, y)
            u_yy = jacfwd(jacfwd(get_u, argnums=2), argnums=2)(t, x, y)
            return u_t, u_x, u_y, u_xx, u_yy

        f_dir_vmap = vmap(get_u_dir, in_axes=(None, 0, 0, 0))
        u_t, u_x, u_y, u_xx, u_yy = f_dir_vmap(get_u, t, x, y)
        u_t, u_x, u_y = u_t[:,:,0], u_x[:,:,0], u_y[:,:,0]
        u_xx, u_yy = u_xx[:,:,0,0], u_yy[:,:,0,0]

        diff_u = DXC * u_xx + DYC * u_yy

        # [sota patch] PDE: u_t + a u_x + b u_y + c u - DXC u_xx - DYC u_yy - f.
        # a, b, c, f are the task's ERA5/GFAS fields, looked up exactly as
        # task/baseline.py:coef_at does it (trilinear, fields stored (t,y,x)).
        a = coef_at(A_TAB, t, x, y)
        b = coef_at(B_TAB, t, x, y)
        c = coef_at(C_TAB, t, x, y)
        f = coef_at(F_TAB, t, x, y)
        pde_u = u_t + a*u_x + b*u_y + c*u - diff_u - f

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
            # the donor's own time framing, unchanged: this task's window is
            # [0,1] against their [0,0.78], so the sin feature sees the same
            # O(1) range they tuned it on and there is nothing to rescale.
            f = jnp.hstack([t - self.t_l, x, y, t - self.t_u])
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

        outputs = jnp.hstack([u, DXC * u_xx + DYC * u_yy])   # [sota patch] weighted, matching the PDE
        return outputs


def main(ER, ER_xx, weight_ic, weight_bc, max_lr, exponent, max_iter, seed, gpu):

    # [sota patch] the task's reference grid in place of the donor's .mat.
    # Same role: the collocation grid, the IC/BC labels and the progress print.
    # The coefficient fields are module-level (see coef_at above).
    _ref = np.loadtxt('~/smoke/task/ref_data.csv',
                      delimiter=',', skiprows=1, dtype=np.float64)
    data_X, data_Y = _ref[:, 0:3], _ref[:, 3:4]

    global t_l, t_u, x_l, x_u, y_l, y_u

    t_l, t_u = np.min(data_X[:,0]), np.max(data_X[:,0])
    x_l, x_u, y_l, y_u = np.min(data_X[:,1]), np.max(data_X[:,1]), np.min(data_X[:,2]), np.max(data_X[:,2])

    # [sota patch] the donor stitches time segments and can take its IC from a
    # previous segment's PINN. This task is solved in one window, and both the
    # initial condition and the four sides are u = 0, so the labels are zeros
    # rather than a supplied profile -- the source term f builds the whole plume.
    ic  = (data_X[:,0] == t_l)
    bcx = (data_X[:,1] == x_l) | (data_X[:,1] == x_u)
    bcy = (data_X[:,2] == y_l) | (data_X[:,2] == y_u)
    data_X_IC, data_Y_IC = data_X[ic], np.zeros_like(data_Y[ic])
    data_X_BC, data_Y_BC = data_X[bcx | bcy], np.zeros_like(data_Y[bcx | bcy])

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
        # stable evolution
        pred_0 = model_0.apply(format_params_fn(params_0), inputs)
        u_0, diff_u_0 = jnp.split(pred_0, 2, axis=1)
        if (ER > 0):
            # their ER correction; the high-order half uses the same weighted
            # diffusion term that appears in the residual
            pde_u = pde_u + (u - u_0) / ER - (diff_u - diff_u_0) / ER_xx  # ER @PDE
        pde_loss = jnp.mean(jnp.square(pde_u))
        # remove random uniform sample. The minibatch is the labelled points
        # followed by an equal number of unlabelled random ones (their design),
        # so every term that pairs with a label has to be cut to BS_ALL --
        # the boundary masks included, or the shapes do not line up.
        u, ic = u[:BS_ALL], ic[:BS_ALL]
        bcx, bcy = bcx[:BS_ALL], bcy[:BS_ALL]
        # [sota patch] IC is u = 0 (labels are zeros), same expression as theirs
        ic_e = (labels - u) * ic
        ic_loss = jnp.sum(jnp.square(ic_e)) / ic.sum()
        # [sota patch] Dirichlet u = 0 on the four sides, where the donor
        # matched opposite faces for a periodic domain
        bc_e = u * (bcx + bcy)
        bc_loss = jnp.sum(jnp.square(bc_e)) / jnp.maximum((bcx + bcy).sum(), 1.0)
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
