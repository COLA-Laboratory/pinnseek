from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator

# [sota patch] their `sample_points_on_square_boundary` becomes one sampler per
# boundary shape, plus the task's own coefficient, source and domain mask
from utils import (sample_points_on_cube_faces, sample_points_on_spheres,
                   screening, source, outside_spheres)


class PoissonCG3D(ForwardBVP):
    """[sota patch] their steady 2D cavity model, moved onto this 3D BVP.

    Unchanged: the class it derives from (`ForwardBVP`, the steady base — this
    task has no time coordinate either), the `losses` shape (named boundary
    terms plus whatever `compute_residual_losses` returns, which is where the
    pseudo-time stepping the paper contributes lives), and the way each term is
    written. What changed:

      * `neural_net` takes (x, y, z) instead of (x, y), and returns one field
        instead of three;
      * `r_net` is the task's equation,
            -(u_xx + u_yy + u_zz) + K(z)^2 u - f(x, y, z) = 0,
        with `screening` and `source` reproduced from `task/baseline.py`;
      * **the residual is masked.** Their cavity fills its square; here two
        spheres of radius 0.2 are cut out of the cube, so the 6.7% of each
        uniform batch that lands inside one (2 * (4/3) pi 0.2^3 = 0.0670 of the
        unit cube; 6.72% by Monte Carlo over 2e6 points) is dropped from the
        mean. `ForwardBVP` computes that mean itself, so the mask is applied by
        zeroing the residual on those points and dividing by the kept fraction
        — which is exactly the masked mean, and leaves their code path
        untouched. One caveat, measured rather than assumed: when
        `pseudo_time` is on, `compute_residual_losses` adds its proximal term
        `pts_weights * (sols_pred - sols_prev)` *after* `r_net` has masked, so
        that term survives inside the spheres. Masking it too changes the
        score by 0.8% (seed 1, 1290 steps: 0.021164 unmasked, 0.021329 masked),
        so it is left as their code path has it;
      * **the boundary terms are Neumann, not Dirichlet.** Their lid matched
        `u = 1` and the walls `u = v = 0`; every boundary here is `du/dn = 0`,
        on the six cube faces (`bc_face`) and the two sphere surfaces
        (`bc_sph`). Both are the same expression — the normal derivative
        squared — and are split only because they are sampled differently.

    Pure Neumann is well posed here: `K(z)^2 u` keeps the constant out of the
    kernel, so the level is pinned by the equation and no gauge condition is
    added.
    """

    def __init__(self, config, lr, tx, arch, state):
        super().__init__(config, lr, tx, arch, state)

        num_face, num_sph = 48, 1024          # theirs is 128 per side
        self.x_face, self.n_face = sample_points_on_cube_faces(num_face)
        self.x_sph, self.n_sph = sample_points_on_spheres(num_sph)

        self.u_pred_fn = vmap(self.u_net, (None, 0, 0, 0))
        self.dn_pred_fn = vmap(self.dn_net, (None, 0, 0, 0, 0, 0, 0))

    def neural_net(self, params, x, y, z):
        w = jnp.stack([x, y, z])
        outputs = self.state.apply_fn(params, w)
        return outputs[0]

    def u_net(self, params, x, y, z):
        return self.neural_net(params, x, y, z)

    def dn_net(self, params, x, y, z, nx, ny, nz):
        """[sota patch] the outward normal derivative — the Neumann residual."""
        u_x, u_y, u_z = grad(self.u_net, argnums=(1, 2, 3))(params, x, y, z)
        return u_x * nx + u_y * ny + u_z * nz

    def r_net(self, params, x, y, z):
        u = self.u_net(params, x, y, z)
        u_hessian = hessian(self.u_net, argnums=(1, 2, 3))(params, x, y, z)
        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]
        u_zz = u_hessian[2][2]
        r = -(u_xx + u_yy + u_zz) + screening(z) * u - source(x, y, z)
        # [sota patch] zero the residual inside the removed spheres; the
        # 1/kept rescale below turns ForwardBVP's plain mean into the masked one
        keep = outside_spheres(jnp.stack([x, y, z])[None, :])[0]
        return r * keep

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # [sota patch] Neumann on the six cube faces and the two spheres
        dn_face = self.dn_pred_fn(params,
                                  self.x_face[:, 0], self.x_face[:, 1], self.x_face[:, 2],
                                  self.n_face[:, 0], self.n_face[:, 1], self.n_face[:, 2])
        dn_sph = self.dn_pred_fn(params,
                                 self.x_sph[:, 0], self.x_sph[:, 1], self.x_sph[:, 2],
                                 self.n_sph[:, 0], self.n_sph[:, 1], self.n_sph[:, 2])
        bc_face_loss = jnp.mean(dn_face ** 2)
        bc_sph_loss = jnp.mean(dn_sph ** 2)

        res_losses = self.compute_residual_losses(
            params, state, batch, pseudo_time=self.config.pseudo_time.enabled)
        # [sota patch] undo the dilution the mask causes in their plain mean
        kept = jnp.mean(outside_spheres(batch))
        res_losses = {k: v / kept for k, v in res_losses.items()}

        return {"bc_face": bc_face_loss, "bc_sph": bc_sph_loss, **res_losses}

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, coords, u_test):
        u_pred = self.u_pred_fn(params, coords[:, 0], coords[:, 1], coords[:, 2])
        return jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)


class PoissonEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, coords, u_ref):
        self.log_dict["l2_error"] = model.compute_l2_error(params, coords, u_ref)

    def __call__(self, model, state, loss_dict, batch, coords, u_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, coords, u_ref)

        if self.config.logging.log_raw_losses:
            self.log_raw_losses(model, state.params, state, batch)

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
