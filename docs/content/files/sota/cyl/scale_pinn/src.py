"""Scale-PINN on the von Karman vortex street — their flow-past-squares recipe, given a clock.

Two donors, both theirs:

  * `ns-fps/plot_pfs_3_re25-n32.ipynb` — Scale-PINN's own **flow past obstacles
    in a channel**: an inlet, a `p = 0` outlet, top and bottom walls, and three
    squares cut out of the domain. Everything about how a boundary is written
    here comes from it: the `uv_bc` / `p_bc` masks, the obstacle points
    appended at the end of the batch so the residual can drop them, the
    obstacle-relative feature map, and the loss's shape.
  * `gs/plot_benchmark_gs.ipynb` — their only 2D + time example, and the source
    of the time framing (`t - t_l`, `t - t_u` in the feature map) and of the
    initial-condition term, which a steady case does not have.

`src.py` is the ns-fps notebook extracted by tools/extract_scale_pinn.py and
then changed in the places marked `[sota patch]`. NATIVE = False: their case is
steady and this one is not.

What changed:

  * **a time axis.** `(x, y) -> (t, x, y)`, so the momentum equations gain
    `u_t` and `v_t`, and an initial-condition term is added against the task's
    `cyl_ic.csv`.
  * **the obstacle is a circle, not three squares**, and there is one of it.
  * **the walls are no-slip.** Theirs writes `u_y = 0` on the top and bottom
    (a symmetry condition, `bc_u = jnp.where(_tb, u_y, bc_u)`); this task's
    channel walls are `u = v = 0`, so the velocity itself is matched there.
  * **the inlet is driven.** `u = 6 U(t) y (H - y) / H^2` with
    `U(t) = sin(pi (t + 4.8) / 8)`. Theirs is a fixed profile read from its
    data file; this one is analytic and time-dependent, so it is computed
    inside the network rather than carried as a label.
  * **the collocation set is analytic.** Theirs came from its simulation file;
    this task ships no field data except the initial condition and the scoring
    truth, and the scoring truth is never opened.

Hyperparameters are the ns-fps notebook's own live call (cell 6):

    ER=0.035, ER_xx=15, weight_bc=5, batch=4000, max_lr=5e-3, exponent=1.3

with 80000 published iterations. The one weight neither donor supplies for this
combination is `weight_ic`: ns-fps is steady and has no initial condition, so it
is set to 5x `weight_bc`, which is the ratio the gs notebook uses
(weight_ic=5000 against weight_bc=1000).
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
NU = 1e-3
H = 0.41
XL, XU = 0.0, 2.2
YL, YU = 0.0, 0.41
T_MAX = 1.2
CX, CY, CR = 0.2, 0.2, 0.05
IC_CSV = '~/cyl/task/cyl_ic.csv'


def inlet_u(t, y):
    """[sota patch] the task's driven inlet profile."""
    return 6.0 * jnp.sin(jnp.pi * (t + 4.8) / 8.0) * y * (H - y) / H ** 2


"""DNN / PINN   """


class PINN(nn.Module):
    global nn_acf
    nn_acf = nn.silu

    """PINNs"""
    n_nodes: int

    def setup(self):
        kinit = jax.nn.initializers.he_uniform()
        self.feats = nn.Dense(self.n_nodes * 2, kernel_init = kinit)
        self.layers = [nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf]
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

    def _uvp(self, t, x, y):
        # [sota patch] their obstacle-relative feature map, on this geometry
        # and with the gs notebook's time framing added
        inputs = jnp.hstack([t / T_MAX, x / XU, y / YU,
                             (x - CX) / XU, (y - CY) / YU,
                             (t - T_MAX) / T_MAX])
        hidden = self.feats(inputs)
        hidden = jnp.sin(jnp.pi * hidden)
        for i, lyr in enumerate(self.layers):
            hidden = lyr(hidden)
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

    @nn.compact
    def __call__(self, inputs):
        t, x, y = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]

        def get_uvp(t, x, y):
            return self._uvp(t, x, y)

        u, v, p = get_uvp(t, x, y)

        # [sota patch] the time derivative is what their steady case lacks
        def get_d1(get_uvp, t, x, y):
            u_t, v_t, _ = jacfwd(get_uvp)(t, x, y)
            u_x, v_x, p_x = jacfwd(get_uvp, argnums=1)(t, x, y)
            u_y, v_y, p_y = jacfwd(get_uvp, argnums=2)(t, x, y)
            return u_t, v_t, u_x, u_y, v_x, v_y, p_x, p_y

        d1 = vmap(get_d1, in_axes=(None, 0, 0, 0))(get_uvp, t, x, y)
        u_t, v_t, u_x, u_y, v_x, v_y, p_x, p_y = (a[:,:,0] for a in d1)

        def get_d2(get_uvp, t, x, y):
            u_xx, v_xx, _ = jacfwd(jacfwd(get_uvp, argnums=1), argnums=1)(t, x, y)
            u_yy, v_yy, _ = jacfwd(jacfwd(get_uvp, argnums=2), argnums=2)(t, x, y)
            return u_xx, u_yy, v_xx, v_yy

        d2 = vmap(get_d2, in_axes=(None, 0, 0, 0))(get_uvp, t, x, y)
        u_xx, u_yy, v_xx, v_yy = (a[:,:,0,0] for a in d2)

        # obtain BC indices
        bc = (x == XL) | (x == XU) | (y == YL) | (y == YU)
        nbc = (~bc)
        uv_bc = (x == XL) | (y == YL) | (y == YU)   # velocity Dirichlet
        p_bc = (x == XU)                            # the outlet
        ic = (t == 0.0)

        # compute PDE residual : unsteady N-S equations
        residuals_continuity = u_x + v_y
        residuals_momentum_1 = u_t + u*u_x + v*u_y + p_x - NU*(u_xx + u_yy)
        residuals_momentum_2 = v_t + u*v_x + v*v_y + p_y - NU*(v_xx + v_yy)

        mom1 = -NU*(u_xx + u_yy)
        mom2 = -NU*(v_xx + v_yy)

        # [sota patch] the inlet target is analytic and time-dependent, so the
        # velocity error is formed here rather than against a label column.
        # The walls are no-slip; theirs writes u_y = 0 there instead.
        u_target = jnp.where(x == XL, inlet_u(t, y), 0.0)
        bcu_err = u - u_target

        outputs = jnp.hstack([u, v, p, residuals_continuity,
                              residuals_momentum_1, residuals_momentum_2,
                              bcu_err, uv_bc, p_bc, nbc, ic, mom1, mom2])
        return outputs


class DNN(nn.Module):
    """DNNs"""
    n_nodes: int

    def setup(self):
        kinit = jax.nn.initializers.he_uniform()
        self.feats = nn.Dense(self.n_nodes * 2, kernel_init = kinit)
        self.layers = [nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf]
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

    def _uvp(self, t, x, y):
        inputs = jnp.hstack([t / T_MAX, x / XU, y / YU,
                             (x - CX) / XU, (y - CY) / YU,
                             (t - T_MAX) / T_MAX])
        hidden = self.feats(inputs)
        hidden = jnp.sin(jnp.pi * hidden)
        for i, lyr in enumerate(self.layers):
            hidden = lyr(hidden)
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

    @nn.compact
    def __call__(self, inputs):
        t, x, y = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]

        def get_uvp(t, x, y):
            return self._uvp(t, x, y)

        u, v, p = get_uvp(t, x, y)

        def get_d2(get_uvp, t, x, y):
            u_xx, v_xx, _ = jacfwd(jacfwd(get_uvp, argnums=1), argnums=1)(t, x, y)
            u_yy, v_yy, _ = jacfwd(jacfwd(get_uvp, argnums=2), argnums=2)(t, x, y)
            return u_xx, u_yy, v_xx, v_yy

        d2 = vmap(get_d2, in_axes=(None, 0, 0, 0))(get_uvp, t, x, y)
        u_xx, u_yy, v_xx, v_yy = (a[:,:,0,0] for a in d2)
        mom1 = -NU*(u_xx + u_yy)
        mom2 = -NU*(v_xx + v_yy)

        outputs = jnp.hstack([u, v, p, mom1, mom2])
        return outputs


def main(ER, ER_xx, weight_bc, weight_ic, batch, max_lr, exponent, max_iter, seed, gpu):

    # [sota patch] the task's own initial condition, and nothing else; the
    # collocation points are drawn from the analytic box below, so
    # task/ref_data.csv -- the scoring truth -- is never opened.
    _ic = np.loadtxt(IC_CSV, delimiter=',', skiprows=1, dtype=np.float64)
    data_X_IC = np.concatenate([np.zeros((len(_ic), 1)), _ic[:, 0:2]], axis=1)
    data_Y_IC = _ic[:, 2:4]
    data_X_IC, data_Y_IC = jnp.array(data_X_IC), jnp.array(data_Y_IC)
    n_ic = len(data_X_IC)

    # choose seed
    seed = seed
    key, rng = random.split(random.PRNGKey(seed))

    # dummy input
    a = random.normal(key, [1,3])                  # [sota patch] three inputs

    # initialization call
    n_nodes = 32
    model, model_0 = PINN(n_nodes), DNN(n_nodes)
    params = model.init(key, a)
    num_params, format_params_fn = get_params_format_fn(params)

    params = jax.flatten_util.ravel_pytree(params)[0]
    params_0 = params

    # [sota patch] batch composition follows theirs -- interior, then each
    # boundary set, then the obstacle points LAST so the residual can drop them
    BS_ALL = batch
    BS_IN = BS_OUT = BS_WALL = int(BS_ALL * 0.05)
    BS_IC = int(BS_ALL * 0.10)
    BS_CYL = int(BS_ALL * 0.10)
    BS_PDE = BS_ALL - BS_IN - BS_OUT - BS_WALL - BS_IC - BS_CYL

    @jit
    def minibatch(key):
        k1, k2, k3, k4, k5, k6, k7 = key
        # interior, cylinder cut out by pushing points to its surface
        s = jnp.hstack([random.uniform(k1, (BS_PDE, 1), minval=0.0, maxval=T_MAX),
                        random.uniform(k2, (BS_PDE, 1), minval=XL, maxval=XU),
                        random.uniform(k3, (BS_PDE, 1), minval=YL, maxval=YU)])
        c = jnp.array([CX, CY])
        v = s[:, 1:3] - c
        d = jnp.linalg.norm(v, axis=1, keepdims=True)
        safe = jnp.where(d > 1e-12, d, 1e-12)
        s = s.at[:, 1:3].set(jnp.where(d < CR, c + v / safe * CR * 1.0001, s[:, 1:3]))

        tb = lambda k, n: random.uniform(k, (n, 1), minval=0.0, maxval=T_MAX)
        yb = lambda k, n: random.uniform(k, (n, 1), minval=YL, maxval=YU)
        xb = lambda k, n: random.uniform(k, (n, 1), minval=XL, maxval=XU)
        inlet = jnp.hstack([tb(k4, BS_IN), jnp.full((BS_IN, 1), XL), yb(k5, BS_IN)])
        outlet = jnp.hstack([tb(k5, BS_OUT), jnp.full((BS_OUT, 1), XU), yb(k4, BS_OUT)])
        half = BS_WALL // 2
        wall = jnp.vstack([
            jnp.hstack([tb(k6, half), xb(k7, half), jnp.full((half, 1), YL)]),
            jnp.hstack([tb(k7, BS_WALL - half), xb(k6, BS_WALL - half),
                        jnp.full((BS_WALL - half, 1), YU)])])
        idx = random.choice(k6, n_ic, (BS_IC,))
        ic_X, ic_Y = data_X_IC[idx], data_Y_IC[idx]
        ang = random.uniform(k7, (BS_CYL, 1), minval=0.0, maxval=2.0 * jnp.pi)
        cyl = jnp.hstack([tb(k1, BS_CYL), CX + CR * jnp.cos(ang), CY + CR * jnp.sin(ang)])

        batch_X = jnp.vstack([s, inlet, outlet, wall, ic_X, cyl])
        return (batch_X, ic_Y)

    def eval_loss(params, params_0, inputs, ic_labels):
        pred = model.apply(format_params_fn(params), inputs)
        (u, v, p, res_c, res_m1, res_m2, bcu_err,
         uv_bc, p_bc, nbc, ic, m_1, m_2) = jnp.split(pred, 13, axis=1)
        # stable evolution
        pred0 = model_0.apply(format_params_fn(params_0), inputs)
        u_0, v_0, p_0, m0_1, m0_2 = jnp.split(pred0, 5, axis=1)
        # PDE
        res_c = res_c + (p - p_0) / ER
        res_m1 = res_m1 + (u - u_0) / ER + (m_1 - m0_1) / ER_xx
        res_m2 = res_m2 + (v - v_0) / ER + (m_2 - m0_2) / ER_xx
        pde_uvp = jnp.square(res_c) + jnp.square(res_m1) + jnp.square(res_m2)
        # [sota patch] the cylinder rows sit at the end, as their square rows do
        pde_uvp, nbc_i = pde_uvp[:-BS_CYL], nbc[:-BS_CYL]
        pde_loss = jnp.sum(pde_uvp * nbc_i) / jnp.maximum(nbc_i.sum(), 1.0)

        # [sota patch] no-slip walls and a driven inlet, both matched on u and v
        bc_uv = jnp.square(bcu_err) + jnp.square(v)
        bc_uv_loss = jnp.sum(bc_uv * uv_bc) / jnp.maximum(uv_bc.sum(), 1.0)
        bc_p_loss = jnp.sum(jnp.square(p) * p_bc) / jnp.maximum(p_bc.sum(), 1.0)
        bc_cyl = jnp.mean(jnp.square(u[-BS_CYL:]) + jnp.square(v[-BS_CYL:]))
        bc_loss = bc_uv_loss + bc_p_loss + bc_cyl

        # [sota patch] the initial condition, which their steady case has not
        lo = BS_PDE + BS_IN + BS_OUT + BS_WALL
        gt_u, gt_v = ic_labels[:, 0:1], ic_labels[:, 1:2]
        ic_loss = (jnp.mean(jnp.square(u[lo:lo + BS_IC] - gt_u))
                   + jnp.mean(jnp.square(v[lo:lo + BS_IC] - gt_v)))

        loss = pde_loss + weight_bc * bc_loss + weight_ic * ic_loss
        return loss, (ic_loss, bc_loss, pde_loss)

    loss_grad = jax.jit(jax.value_and_grad(eval_loss, has_aux=True))

    @jit
    def update(params, params_0, opt_state, key):
        batch_X, ic_Y = minibatch(key)
        (loss, aux), grad = loss_grad(params, params_0, batch_X, ic_Y)
        updates, opt_state = optimizer.update(grad, opt_state)
        params_0 = params
        params = optax.apply_updates(params, updates)
        return params, params_0, opt_state, loss, aux[0], aux[1], aux[2]

    max_iters = max_iter
    lr_scheduler = optax.warmup_cosine_decay_schedule(init_value=max_lr, peak_value=max_lr, warmup_steps=0,
                                                      decay_steps=max_iters, end_value=1e-10, exponent=exponent)
    optimizer = optax.adam(learning_rate=lr_scheduler)
    opt_state = optimizer.init(params)

    """Training"""
    runtime = 0
    train_iters = 0

    while (train_iters <= max_iters):    # [sota patch] the wall-clock guard is
                                         # the benchmark's job, not the notebook's
        start = time.time()
        keys = random.split(rng, 8)
        rng = keys[0]
        params, params_0, opt_state, loss, ic_l, bc_l, pde_l = update(
            params, params_0, opt_state, tuple(keys[1:]))
        end = time.time()
        runtime += (end - start)
        if (train_iters % 5000 == 0):
            print('iter. = %05d,  time = %03ds,  loss = %.2e  |  ic = %.2e,  bc = %.2e,  pde = %.2e'
                  % (train_iters, runtime, loss, ic_l, bc_l, pde_l))
        train_iters += 1

    return format_params_fn(params), train_iters
