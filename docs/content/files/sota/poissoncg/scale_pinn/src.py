"""Scale-PINN on the 3D screened Poisson problem — their ns-ldc notebook, retargeted.

Generated from `ns-ldc/plot_benchmark_pinn_ldc3200.ipynb` the same way every
other Scale-PINN port here is (tools/extract_scale_pinn.py takes the authors'
definitions up to the first top-level `main(...)` call), then changed in the
places marked `[sota patch]`.

Why that notebook is the donor: it is Scale-PINN's only **steady** case, so it
is the only one whose loop has no time axis to remove. Their other 2D examples
(`gs`, `ns-rb`, `ns-fps`) all carry one. Everything structural survives — the
`sin(2 pi * Dense(...))` feature map, the shared trunk with a per-field split
head, the `he_uniform` init, the `BS_ALL`/`BS_BC` minibatch composition, the
`(u - u_0)/ER` pseudo-time correction with its separate `ER_xx` term on the
diffusion part, the warmup-cosine schedule, and the paired `PINN`/`DNN` pattern
where the second net is the plain forward evaluator.

What changed:

  * **three inputs, one field.** Their `(x, y) -> (u, v, p)` becomes
    `(x, y, z) -> u`, so `splitv`/`splitp` and their head stacks are gone and
    the feature map's `hstack([x, y, x-1, y-1])` gains the z pair.
  * **the equation.** Theirs is steady Navier-Stokes; this task is
        -(u_xx + u_yy + u_zz) + K(z)^2 u - f(x, y, z) = 0
    with `K = 8` below z = 0.5 and 10 above. `screening` and `source` are
    `task/baseline.py`'s own functions, reproduced there term for term.
  * **the boundary condition is Neumann, and on eight surfaces.** Theirs is
    Dirichlet against csv values on the four sides of a square. Here every
    boundary is `du/dn = 0`, on the six cube faces and the two sphere
    surfaces, so `bc_loss` is the squared normal derivative and there are no
    boundary labels at all. The normal cannot be recovered from a sphere
    point's coordinates the way a square's can, so it is carried alongside it:
    the input array is `[x, y, z, nx, ny, nz, is_bc]` and the network reads
    only the first three columns.
  * **there is no data file.** This task ships nothing but the scoring truth.
    Their collocation set was their csv's grid; here it is a uniform lattice
    with the two spheres cut out, plus the analytic boundary sets. Their
    progress print reported `rl2` against csv labels, so it now reports the
    residual instead — there is no label to compare against.
  * **the pressure-reference shift is gone.** It existed because pressure is
    defined up to a constant; `K^2 u` pins this problem's level, so there is
    no gauge freedom to remove.

The ER correction is kept at the notebook's published value. This is a steady
problem, the setting their `(u - u_0)/ER` pseudo-time term was formulated for,
and `ER_xx` keeps its role on the diffusion part `m = -(u_xx + u_yy + u_zz)`.

Hyperparameters are the notebook's own live call (cell 4, the five-seed loop):
ER=0.06, ER_xx=1, weight_bc=15, max_lr=5e-4, exponent=0.8, n_nodes=32.
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

# [sota patch] task/baseline.py frozen header, verbatim
SPHERES = ((0.4, 0.3, 0.6, 0.2), (0.6, 0.7, 0.6, 0.2))
Z_IFACE = 0.5
K_LO, K_HI = 8.0, 10.0
SRC_A1, SRC_A2 = 20.0, 100.0
SRC_M1, SRC_M2, SRC_M3 = 1.0, 10.0, 5.0


def screening(z):
    """[sota patch] K(z)^2 — task/baseline.py:screening, verbatim."""
    return jnp.where(z < Z_IFACE, K_LO ** 2, K_HI ** 2)


def source(x, y, z):
    """[sota patch] f(x, y, z) — task/baseline.py:source, verbatim."""
    r2 = x ** 2 + y ** 2 + z ** 2
    part1 = (jnp.exp(jnp.sin(SRC_M1 * jnp.pi * x)
                     + jnp.sin(SRC_M2 * jnp.pi * y)
                     + jnp.sin(SRC_M3 * jnp.pi * z))
             * (r2 - 1.0) / (r2 + 1.0))
    part2 = (jnp.sin(SRC_M1 * jnp.pi * x)
             * jnp.sin(SRC_M2 * jnp.pi * y)
             * jnp.sin(SRC_M3 * jnp.pi * z))
    return SRC_A1 * part1 + SRC_A2 * part2


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
        # [sota patch] one field, so only the u head survives
        self.splitu = nn.Dense(self.n_nodes, kernel_init = kinit)
        self.layeru = [nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(self.n_nodes, kernel_init = kinit),
                       nn.silu,
                       nn.Dense(1, kernel_init = kinit, use_bias=False)]

    @nn.compact
    def __call__(self, inputs):
        # [sota patch] columns are [x, y, z, nx, ny, nz, is_bc]; the network
        # reads the first three, the rest ride along for the loss
        x, y, z = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
        nx, ny, nz = inputs[:,3:4], inputs[:,4:5], inputs[:,5:6]
        bc = inputs[:,6:7]
        nbc = 1.0 - bc

        def get_u(x, y, z):
            # [sota patch] their hstack([x, y, x-1., y-1.]) gains the z pair
            inputs = jnp.hstack([x, y, z, x - 1., y - 1., z - 1.])
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
            return (u,)

        (u,) = get_u(x, y, z)

        # obtain u_x, u_y, u_z
        def get_u_xyz(get_u, x, y, z):
            (u_x,) = jacfwd(get_u)(x, y, z)
            (u_y,) = jacfwd(get_u, argnums=1)(x, y, z)
            (u_z,) = jacfwd(get_u, argnums=2)(x, y, z)
            return u_x, u_y, u_z

        get_u_xyz_vmap = vmap(get_u_xyz, in_axes=(None, 0, 0, 0))
        u_x, u_y, u_z = get_u_xyz_vmap(get_u, x, y, z)
        u_x, u_y, u_z = u_x[:,:,0], u_y[:,:,0], u_z[:,:,0]

        # obtain u_xx, u_yy, u_zz
        def get_u_xxyyzz(get_u, x, y, z):
            (u_xx,) = jacfwd(jacfwd(get_u))(x, y, z)
            (u_yy,) = jacfwd(jacfwd(get_u, argnums=1), argnums=1)(x, y, z)
            (u_zz,) = jacfwd(jacfwd(get_u, argnums=2), argnums=2)(x, y, z)
            return u_xx, u_yy, u_zz

        get_u_xxyyzz_vmap = vmap(get_u_xxyyzz, in_axes=(None, 0, 0, 0))
        u_xx, u_yy, u_zz = get_u_xxyyzz_vmap(get_u, x, y, z)
        u_xx, u_yy, u_zz = u_xx[:,:,0,0], u_yy[:,:,0,0], u_zz[:,:,0,0]

        # [sota patch] compute PDE residual : screened Poisson
        # residual = -(u_xx + u_yy + u_zz) + K(z)^2 u - f(x, y, z)
        m = -(u_xx + u_yy + u_zz)                      # their `mom` diffusion part
        residuals_pde = m + screening(z) * u - source(x, y, z)

        # [sota patch] du/dn, the Neumann residual; zero away from a boundary
        u_n = (u_x * nx + u_y * ny + u_z * nz)

        outputs = jnp.hstack([u, residuals_pde, u_n, bc, nbc, m])
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

    @nn.compact
    def __call__(self, inputs):
        x, y, z = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
        def get_u(x, y, z):
            inputs = jnp.hstack([x, y, z, x - 1., y - 1., z - 1.])
            hidden = self.feats(inputs)
            hidden = jnp.sin(2*jnp.pi*hidden)
            for i, lyr in enumerate(self.layers):
                hidden = lyr(hidden)
            u = self.splitu(hidden)
            for i, lyr in enumerate(self.layeru):
                u = lyr(u)
            return (u,)

        (u,) = get_u(x, y, z)

        def get_u_xxyyzz(get_u, x, y, z):
            (u_xx,) = jacfwd(jacfwd(get_u))(x, y, z)
            (u_yy,) = jacfwd(jacfwd(get_u, argnums=1), argnums=1)(x, y, z)
            (u_zz,) = jacfwd(jacfwd(get_u, argnums=2), argnums=2)(x, y, z)
            return u_xx, u_yy, u_zz

        get_u_xxyyzz_vmap = vmap(get_u_xxyyzz, in_axes=(None, 0, 0, 0))
        u_xx, u_yy, u_zz = get_u_xxyyzz_vmap(get_u, x, y, z)
        u_xx, u_yy, u_zz = u_xx[:,:,0,0], u_yy[:,:,0,0], u_zz[:,:,0,0]

        m = -(u_xx + u_yy + u_zz)
        # [sota patch] no pressure reference to subtract: K^2 u pins the level

        outputs = jnp.hstack([u, m])
        return outputs


def _build_sets():
    """[sota patch] collocation and boundary sets, all analytic.

    Their notebook read a csv and used its grid. This task ships no data, so
    the interior lattice is generated here on the same open-grid convention
    `task/baseline.py` uses (`linspace(...)[1:-1]`, since the cube faces are
    boundary planes), the two spheres are cut out, and the eight boundaries are
    built with their outward normals — axis-aligned on the faces, radially
    inward on the spheres, as `task/baseline.py:_build_dataset` has them.
    """
    n = 48
    a = np.linspace(0.0, 1.0, n)[1:-1]
    X, Y, Z = np.meshgrid(a, a, a, indexing='ij')
    inside = np.zeros(X.shape, dtype=bool)
    for cx, cy, cz, r in SPHERES:
        inside |= (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= r ** 2
    pts_col = np.stack([X[~inside], Y[~inside], Z[~inside]], axis=1)
    nrm_col = np.zeros_like(pts_col)

    nf = 32
    b = np.linspace(0.0, 1.0, nf)
    A, B = np.meshgrid(b, b, indexing='ij')
    A, B = A.ravel(), B.ravel()
    one, zero = np.ones_like(A), np.zeros_like(A)
    faces, normals = [], []
    for axis in range(3):
        for value, sign in ((0.0, -1.0), (1.0, 1.0)):
            cols = [A, B]
            p = [None, None, None]
            p[axis] = np.full_like(A, value)
            for k in (i for i in range(3) if i != axis):
                p[k] = cols.pop(0)
            faces.append(np.stack(p, axis=1))
            nn_ = [zero, zero, zero]
            nn_[axis] = sign * one
            normals.append(np.stack(nn_, axis=1))
    pts_face = np.concatenate(faces)
    nrm_face = np.concatenate(normals)

    ns = 1024
    idx = np.arange(ns) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / ns)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
    unit = np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)
    pts_sph = np.concatenate([np.array([cx, cy, cz]) + r * unit
                              for cx, cy, cz, r in SPHERES])
    nrm_sph = np.concatenate([-unit for _ in SPHERES])

    def pack(p, nrm, is_bc):
        return np.concatenate([p, nrm, np.full((len(p), 1), is_bc)], axis=1)

    data_X = pack(pts_col, nrm_col, 0.0).astype(np.float32)
    data_X_BC = np.concatenate([pack(pts_face, nrm_face, 1.0),
                                pack(pts_sph, nrm_sph, 1.0)]).astype(np.float32)
    return data_X, data_X_BC


def main(ER, ER_xx, weight_bc, max_lr, exponent, max_iters, seed, gpu):

    global x_l, x_u, y_l, y_u, dx, dy       # [sota patch] no Re, no p reference

    # [sota patch] analytic sets in place of their csv
    data_X_np, data_X_BC_np = _build_sets()

    x_l, x_u = 0.0, 1.0
    y_l, y_u = 0.0, 1.0
    dx = dy = 1 / (48 - 1)

    data_X = jnp.array(data_X_np)
    data_X_BC = jnp.array(data_X_BC_np)

    # choose seed
    seed = seed
    key, rng = random.split(random.PRNGKey(seed))

    # dummy input
    a = random.normal(key, [1,7])           # [sota patch] 7 columns

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
        return batch_X

    # loss function
    def eval_loss(params, params_0, inputs):
        pred = model.apply(format_params_fn(params), inputs)
        u, residuals_pde, u_n, bc, nbc, m = jnp.split(pred, 6, axis=1)
        # PDE loss unadjusted
        pde_loss_uadj = jnp.sum(jnp.square(residuals_pde) *nbc) / nbc.sum()
        # stable evolution
        pred0 = model_0.apply(format_params_fn(params_0), inputs)
        u_0, m0 = jnp.split(pred0, 2, axis=1)
        # PDE
        if (ER > 0):
            residuals_pde = residuals_pde + (u - u_0) / ER + (m - m0) / ER_xx # ER @PDE
        pde_loss = jnp.sum(jnp.square(residuals_pde) *nbc) / nbc.sum()
        # [sota patch] BC: homogeneous Neumann, so the residual is du/dn itself
        # and there is no label to subtract
        bc_loss = jnp.sum(jnp.square(u_n) *bc) / bc.sum()
        loss = pde_loss + weight_bc* bc_loss
        # [sota patch] no labels anywhere, so the progress numbers are the two
        # residuals rather than an error against data
        return loss, (pde_loss_uadj, bc_loss, pde_loss, pde_loss_uadj)

    loss_grad = jax.jit(jax.value_and_grad(eval_loss, has_aux=True))

    # weights update
    @jit
    def update(params, params_0, opt_state, key):
        batch_X = minibatch(key)
        (loss, (mse, rl2, pde_loss, pde_loss_uadj)), grad = loss_grad(params, params_0, batch_X)
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

    while (train_iters <= max_iters):    # [sota patch] the 300 s guard is the
                                         # benchmark's job, not the notebook's
        # mini-batch update
        start = time.time()
        key1, key2, rng = random.split(rng, 3) # update random generator
        params, params_0, opt_state, loss, mse, rl2, pde_loss, pde_loss_uadj = update(params, params_0, opt_state, (key1, key2))
        end = time.time()
        runtime += (end-start)
        if (train_iters % 5000 == 0):
            print ('iter. = %05d,  time = %03ds,  loss = %.2e  |  pde = %.2e,  bc = %.2e'%(train_iters, runtime, loss, pde_loss_uadj, rl2))
        train_iters += 1

    """PINN solution"""
    return format_params_fn(params), train_iters
