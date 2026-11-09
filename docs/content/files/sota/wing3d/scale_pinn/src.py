"""Scale-PINN on steady 3D flow over a finite wing — their flow-past-squares recipe in 3D.

`ns-fps/plot_pfs_3_re25-n32.ipynb` is Scale-PINN's own **steady incompressible
flow past bodies immersed in a channel**, and it is an unusually close donor:
it reads a signed-distance level set (`phi`) from its data file, selects the
fluid with `sim_ls <= 0`, appends the body-surface points at the end of the
minibatch so the residual can drop them, and carries an inlet, a `p = 0`
outlet, and walls. This task uses **the same level-set convention** —
`collocation.npy` is `(x, y, z, phi)` with `phi <= 0` in the fluid — and the
same four kinds of boundary. `src.py` is that notebook extracted by
tools/extract_scale_pinn.py and then changed in the places marked
`[sota patch]`. NATIVE = False.

What changed:

  * **three spatial dimensions.** `(x, y) -> (x, y, z)` and
    `(u, v, p) -> (u, v, w, p)`, so the momentum equations gain their z
    component and the divergence its `w_z`.
  * **the immersed body is a wing, given by a file.** Theirs are three squares
    with their own coordinate files; here the surface is
    `task/wing_surface.npy` and the fluid mask is already applied to
    `task/collocation.npy` (every shipped point has `phi <= 0`).
  * **a symmetry plane.** `z = 0` carries `w = 0` and `du/dz = dv/dz = 0`.
    Theirs writes `u_y = 0` on its top and bottom, which is the same kind of
    condition — a normal-derivative match — so the machinery is theirs; the
    plane and the components are the task's.
  * **the lateral and top faces are freestream**, `u = 1, v = w = 0`, where
    theirs has channel walls.
  * `Re = 1000` in place of their 25.

Their feature map is built from body-relative offsets (`(x - 7)/30` for each
square). The analogue here is offsets from the wing's own extent, which is
what the map below uses.

Hyperparameters are the ns-fps notebook's own live call (cell 6):

    ER=0.035, ER_xx=15, weight_bc=5, batch=4000, max_lr=5e-3, exponent=1.3

with 80000 published iterations. `max_iter` is also `decay_steps` of the cosine
schedule, so the step count is compressed to what fits the budget rather than
the schedule being cut off part-way.

Training reads the two geometry files and nothing else; `ref_data.npy`, the
scoring truth, is never opened.
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
RE = 1000.0
NU = 1.0 / RE
U_INF = 1.0
XL, XU = -1.5, 3.5
YL, YU = -2.0, 2.0
ZL, ZU = 0.0, 3.5
TOL = 1e-5
TASK = '~/wing3d/task'
# the wing's own extent, for the body-relative features
WX, WY, WZ = 1.0, 0.0, 0.75


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
        # [sota patch] a fourth head: this problem carries (u, v, w, p)
        self.splitu = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layeru = [nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitv = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layerv = [nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitw = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layerw = [nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitp = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layerp = [nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(1, kernel_init = kinit, use_bias=False)]

    def _uvwp(self, x, y, z):
        # [sota patch] their body-relative feature map, on this geometry:
        # normalised coordinates plus offsets from the wing's own extent
        inputs = jnp.hstack([x / (XU - XL), y / (YU - YL), z / (ZU - ZL),
                             (x - WX) / (XU - XL), (y - WY) / (YU - YL),
                             (z - WZ) / (ZU - ZL)])
        hidden = self.feats(inputs)
        hidden = jnp.sin(jnp.pi * hidden)
        for i, lyr in enumerate(self.layers):
            hidden = lyr(hidden)
        u = self.splitu(hidden)
        for lyr in self.layeru:
            u = lyr(u)
        v = self.splitv(hidden)
        for lyr in self.layerv:
            v = lyr(v)
        w = self.splitw(hidden)
        for lyr in self.layerw:
            w = lyr(w)
        p = self.splitp(hidden)
        for lyr in self.layerp:
            p = lyr(p)
        return (u, v, w, p)

    @nn.compact
    def __call__(self, inputs):
        x, y, z = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]

        def get_uvwp(x, y, z):
            return self._uvwp(x, y, z)

        u, v, w, p = get_uvwp(x, y, z)

        def get_d1(get_uvwp, x, y, z):
            d_x = jacfwd(get_uvwp)(x, y, z)
            d_y = jacfwd(get_uvwp, argnums=1)(x, y, z)
            d_z = jacfwd(get_uvwp, argnums=2)(x, y, z)
            return d_x, d_y, d_z

        d_x, d_y, d_z = vmap(get_d1, in_axes=(None, 0, 0, 0))(get_uvwp, x, y, z)
        u_x, v_x, w_x, p_x = (a[:,:,0] for a in d_x)
        u_y, v_y, w_y, p_y = (a[:,:,0] for a in d_y)
        u_z, v_z, w_z, p_z = (a[:,:,0] for a in d_z)

        def get_d2(get_uvwp, x, y, z):
            dxx = jacfwd(jacfwd(get_uvwp), argnums=0)(x, y, z)
            dyy = jacfwd(jacfwd(get_uvwp, argnums=1), argnums=1)(x, y, z)
            dzz = jacfwd(jacfwd(get_uvwp, argnums=2), argnums=2)(x, y, z)
            return dxx, dyy, dzz

        dxx, dyy, dzz = vmap(get_d2, in_axes=(None, 0, 0, 0))(get_uvwp, x, y, z)
        u_xx, v_xx, w_xx, _ = (a[:,:,0,0] for a in dxx)
        u_yy, v_yy, w_yy, _ = (a[:,:,0,0] for a in dyy)
        u_zz, v_zz, w_zz, _ = (a[:,:,0,0] for a in dzz)

        lap_u = u_xx + u_yy + u_zz
        lap_v = v_xx + v_yy + v_zz
        lap_w = w_xx + w_yy + w_zz
        adv = lambda fx, fy, fz: u*fx + v*fy + w*fz

        # [sota patch] the task's four steady equations in three dimensions
        res_c = u_x + v_y + w_z
        res_mx = adv(u_x, u_y, u_z) + p_x - NU*lap_u
        res_my = adv(v_x, v_y, v_z) + p_y - NU*lap_v
        res_mz = adv(w_x, w_y, w_z) + p_z - NU*lap_w

        mom_u, mom_v, mom_w = -NU*lap_u, -NU*lap_v, -NU*lap_w

        # [sota patch] the boundary masks: freestream faces, the outlet, and
        # the symmetry plane, where theirs has an inlet, an outlet and walls
        far_bc = ((jnp.abs(x - XL) < TOL) | (jnp.abs(y - YL) < TOL)
                  | (jnp.abs(y - YU) < TOL) | (jnp.abs(z - ZU) < TOL))
        p_bc = (jnp.abs(x - XU) < TOL)
        sym_bc = (jnp.abs(z - ZL) < TOL)
        far_bc = far_bc & (~p_bc)             # the outlet edge is claimed once
        nbc = (~far_bc) & (~p_bc) & (~sym_bc)

        outputs = jnp.hstack([u, v, w, p, res_c, res_mx, res_my, res_mz,
                              u_z, v_z, far_bc, p_bc, sym_bc, nbc,
                              mom_u, mom_v, mom_w])
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
        self.layeru = [nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitv = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layerv = [nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitw = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layerw = [nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(1, kernel_init = kinit, use_bias=False)]
        self.splitp = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layerp = [nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn_acf, nn.Dense(1, kernel_init = kinit, use_bias=False)]

    def _uvwp(self, x, y, z):
        inputs = jnp.hstack([x / (XU - XL), y / (YU - YL), z / (ZU - ZL),
                             (x - WX) / (XU - XL), (y - WY) / (YU - YL),
                             (z - WZ) / (ZU - ZL)])
        hidden = self.feats(inputs)
        hidden = jnp.sin(jnp.pi * hidden)
        for i, lyr in enumerate(self.layers):
            hidden = lyr(hidden)
        u = self.splitu(hidden)
        for lyr in self.layeru:
            u = lyr(u)
        v = self.splitv(hidden)
        for lyr in self.layerv:
            v = lyr(v)
        w = self.splitw(hidden)
        for lyr in self.layerw:
            w = lyr(w)
        p = self.splitp(hidden)
        for lyr in self.layerp:
            p = lyr(p)
        return (u, v, w, p)

    @nn.compact
    def __call__(self, inputs):
        x, y, z = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]

        def get_uvwp(x, y, z):
            return self._uvwp(x, y, z)

        u, v, w, p = get_uvwp(x, y, z)

        def get_d2(get_uvwp, x, y, z):
            dxx = jacfwd(jacfwd(get_uvwp), argnums=0)(x, y, z)
            dyy = jacfwd(jacfwd(get_uvwp, argnums=1), argnums=1)(x, y, z)
            dzz = jacfwd(jacfwd(get_uvwp, argnums=2), argnums=2)(x, y, z)
            return dxx, dyy, dzz

        dxx, dyy, dzz = vmap(get_d2, in_axes=(None, 0, 0, 0))(get_uvwp, x, y, z)
        u_xx, v_xx, w_xx, _ = (a[:,:,0,0] for a in dxx)
        u_yy, v_yy, w_yy, _ = (a[:,:,0,0] for a in dyy)
        u_zz, v_zz, w_zz, _ = (a[:,:,0,0] for a in dzz)
        mom_u = -NU*(u_xx + u_yy + u_zz)
        mom_v = -NU*(v_xx + v_yy + v_zz)
        mom_w = -NU*(w_xx + w_yy + w_zz)

        outputs = jnp.hstack([u, v, w, p, mom_u, mom_v, mom_w])
        return outputs


def _sets():
    """[sota patch] the shipped geometry, sliced the way task/baseline.py
    slices it. Theirs read a csv with a `phi` column and selected `phi <= 0`;
    `collocation.npy` is already filtered to `phi <= 0`, so only the faces have
    to be separated out."""
    col = np.load(f'{TASK}/collocation.npy')
    xyz = np.ascontiguousarray(col[:, :3], dtype=np.float32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    on_inlet = np.abs(x - XL) < TOL
    on_out = np.abs(x - XU) < TOL
    on_lat = (np.abs(y - YL) < TOL) | (np.abs(y - YU) < TOL)
    on_top = np.abs(z - ZU) < TOL
    on_sym = np.abs(z - ZL) < TOL
    on_face = on_inlet | on_out | on_lat | on_top | on_sym
    wing = np.load(f'{TASK}/wing_surface.npy').astype(np.float32)
    return (xyz[~on_face], xyz[(on_inlet | on_lat | on_top) & ~on_out],
            xyz[on_out], xyz[on_sym], wing)


def main(ER, ER_xx, weight_bc, batch, max_lr, exponent, max_iter, seed, gpu):

    data_int, data_far, data_out, data_sym, data_wing = _sets()
    data_int = jnp.array(data_int); data_far = jnp.array(data_far)
    data_out = jnp.array(data_out); data_sym = jnp.array(data_sym)
    data_wing = jnp.array(data_wing)
    n_int, n_far, n_out, n_sym, n_wing = (len(a) for a in
                                          (data_int, data_far, data_out, data_sym, data_wing))

    # choose seed
    seed = seed
    key, rng = random.split(random.PRNGKey(seed))

    # dummy input
    a = random.normal(key, [1,3])

    # initialization call
    n_nodes = 32
    model, model_0 = PINN(n_nodes), DNN(n_nodes)
    params = model.init(key, a)
    num_params, format_params_fn = get_params_format_fn(params)
    params = jax.flatten_util.ravel_pytree(params)[0]
    params_0 = params

    # [sota patch] batch composition follows theirs -- interior, then each
    # boundary set, then the BODY points LAST so the residual can drop them
    BS_ALL = batch
    BS_FAR = int(BS_ALL * 0.10)
    BS_OUT = int(BS_ALL * 0.05)
    BS_SYM = int(BS_ALL * 0.05)
    BS_WING = int(BS_ALL * 0.20)
    BS_PDE = BS_ALL - BS_FAR - BS_OUT - BS_SYM - BS_WING

    @jit
    def minibatch(key):
        k1, k2, k3, k4, k5 = key
        return jnp.vstack([data_int[random.choice(k1, n_int, (BS_PDE,))],
                           data_far[random.choice(k2, n_far, (BS_FAR,))],
                           data_out[random.choice(k3, n_out, (BS_OUT,))],
                           data_sym[random.choice(k4, n_sym, (BS_SYM,))],
                           data_wing[random.choice(k5, n_wing, (BS_WING,))]])

    def eval_loss(params, params_0, inputs):
        pred = model.apply(format_params_fn(params), inputs)
        (u, v, w, p, res_c, res_mx, res_my, res_mz, u_z, v_z,
         far_bc, p_bc, sym_bc, nbc, m_u, m_v, m_w) = jnp.split(pred, 17, axis=1)
        # stable evolution
        pred0 = model_0.apply(format_params_fn(params_0), inputs)
        u_0, v_0, w_0, p_0, m0_u, m0_v, m0_w = jnp.split(pred0, 7, axis=1)
        # PDE
        res_c = res_c + (p - p_0) / ER
        res_mx = res_mx + (u - u_0) / ER + (m_u - m0_u) / ER_xx
        res_my = res_my + (v - v_0) / ER + (m_v - m0_v) / ER_xx
        res_mz = res_mz + (w - w_0) / ER + (m_w - m0_w) / ER_xx
        pde = (jnp.square(res_c) + jnp.square(res_mx)
               + jnp.square(res_my) + jnp.square(res_mz))
        # [sota patch] the wing rows sit at the end, as their square rows do
        pde, nbc_i = pde[:-BS_WING], nbc[:-BS_WING]
        pde_loss = jnp.sum(pde * nbc_i) / jnp.maximum(nbc_i.sum(), 1.0)

        # [sota patch] freestream faces, outlet, symmetry plane, wing
        far_e = jnp.square(u - U_INF) + jnp.square(v) + jnp.square(w)
        far_loss = jnp.sum(far_e * far_bc) / jnp.maximum(far_bc.sum(), 1.0)
        p_loss = jnp.sum(jnp.square(p) * p_bc) / jnp.maximum(p_bc.sum(), 1.0)
        sym_e = jnp.square(w) + jnp.square(u_z) + jnp.square(v_z)
        sym_loss = jnp.sum(sym_e * sym_bc) / jnp.maximum(sym_bc.sum(), 1.0)
        wing_loss = jnp.mean(jnp.square(u[-BS_WING:]) + jnp.square(v[-BS_WING:])
                             + jnp.square(w[-BS_WING:]))
        bc_loss = far_loss + p_loss + sym_loss + wing_loss

        loss = pde_loss + weight_bc * bc_loss
        return loss, (wing_loss, bc_loss, pde_loss)

    loss_grad = jax.jit(jax.value_and_grad(eval_loss, has_aux=True))

    @jit
    def update(params, params_0, opt_state, key):
        batch_X = minibatch(key)
        (loss, aux), grad = loss_grad(params, params_0, batch_X)
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
        keys = random.split(rng, 6)
        rng = keys[0]
        params, params_0, opt_state, loss, wing_l, bc_l, pde_l = update(
            params, params_0, opt_state, tuple(keys[1:]))
        end = time.time()
        runtime += (end - start)
        if (train_iters % 5000 == 0):
            print('iter. = %05d,  time = %03ds,  loss = %.2e  |  wing = %.2e,  bc = %.2e,  pde = %.2e'
                  % (train_iters, runtime, loss, wing_l, bc_l, pde_l))
        train_iters += 1

    return format_params_fn(params), train_iters
