from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator


class Wave2DMS(ForwardIVP):
    """[sota patch] their Wave1D, extended to two spatial dimensions with a
    variable medium and Neumann walls.

    What is unchanged: the class it derives from, the loss *shape*
    (`u0` + `u_t0` + `bcs` + the residual losses their `compute_residual_losses`
    builds, with causal weighting), and the way each term is written. What
    changed, and only this:

      * `neural_net` takes (t, x, y) instead of (t, x);
      * `r_net` is `u_tt - c(x,y) * (u_xx + u_yy)`. Theirs is
        `u_tt - c**2 * u_xx` with a scalar c; this task supplies c(x,y) as a
        256x256 table and `problem.md` asks for bilinear interpolation between
        grid points, so `_c_at` interpolates;
      * the initial terms are evaluated on the (x, y) lattice rather than a
        line;
      * `bcs_loss` is the squared **normal derivative** on the four sides
        (zero Neumann), where theirs is the Dirichlet trace at the two ends.

    u == 0 satisfies the equation and both Neumann conditions exactly, so the
    initial terms are the only thing pinning the solution. The published loss
    weights (all 1.0) are left alone.
    """

    def __init__(self, config, lr, tx, arch, state, u0, t_star, x_star, y_star,
                 cx, cy, c_grid):
        super().__init__(config, lr, tx, arch, state)

        self.u0 = u0
        self.t_star = t_star
        self.x_star = x_star
        self.y_star = y_star
        self.cx, self.cy, self.c_grid = cx, cy, c_grid

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

    def _c_at(self, x, y):
        """Bilinear interpolation of c(x, y), as problem.md specifies."""
        fx = jnp.clip((x - self.cx[0]) / (self.cx[1] - self.cx[0]),
                      0.0, self.cx.shape[0] - 1.001)
        fy = jnp.clip((y - self.cy[0]) / (self.cy[1] - self.cy[0]),
                      0.0, self.cy.shape[0] - 1.001)
        i0 = jnp.floor(fx).astype(jnp.int32)
        j0 = jnp.floor(fy).astype(jnp.int32)
        wx, wy = fx - i0, fy - j0
        return ((1 - wx) * (1 - wy) * self.c_grid[i0, j0]
                + wx * (1 - wy) * self.c_grid[i0 + 1, j0]
                + (1 - wx) * wy * self.c_grid[i0, j0 + 1]
                + wx * wy * self.c_grid[i0 + 1, j0 + 1])

    def neural_net(self, params, t, x, y):
        z = jnp.stack([t, x, y])
        u = self.state.apply_fn(params, z)
        return u[0]

    def u_t_net(self, params, t, x, y):
        return grad(self.neural_net, argnums=1)(params, t, x, y)

    def u_x_net(self, params, t, x, y):
        return grad(self.neural_net, argnums=2)(params, t, x, y)

    def u_y_net(self, params, t, x, y):
        return grad(self.neural_net, argnums=3)(params, t, x, y)

    def r_net(self, params, t, x, y):
        u_tt = grad(grad(self.neural_net, argnums=1), argnums=1)(params, t, x, y)
        u_xx = grad(grad(self.neural_net, argnums=2), argnums=2)(params, t, x, y)
        u_yy = grad(grad(self.neural_net, argnums=3), argnums=3)(params, t, x, y)
        return u_tt - self._c_at(x, y) * (u_xx + u_yy)

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # Initial condition loss, on the (x, y) lattice
        u0_pred = vmap(vmap(self.neural_net, (None, None, None, 0)),
                       (None, None, 0, None))(params, self.t0, self.x_star, self.y_star)
        u0_loss = jnp.mean((self.u0 - u0_pred) ** 2)

        u_t0_pred = vmap(vmap(self.u_t_net, (None, None, None, 0)),
                         (None, None, 0, None))(params, self.t0, self.x_star, self.y_star)
        u_t0_loss = jnp.mean((0 - u_t0_pred) ** 2)

        # Zero-Neumann boundary loss on the four sides
        ux_lo = vmap(vmap(self.u_x_net, (None, None, None, 0)),
                     (None, 0, None, None))(params, self.t_star, self.x_star[0], self.y_star)
        ux_hi = vmap(vmap(self.u_x_net, (None, None, None, 0)),
                     (None, 0, None, None))(params, self.t_star, self.x_star[-1], self.y_star)
        uy_lo = vmap(vmap(self.u_y_net, (None, None, 0, None)),
                     (None, 0, None, None))(params, self.t_star, self.x_star, self.y_star[0])
        uy_hi = vmap(vmap(self.u_y_net, (None, None, 0, None)),
                     (None, 0, None, None))(params, self.t_star, self.x_star, self.y_star[-1])
        bcs_loss = (jnp.mean(ux_lo ** 2) + jnp.mean(ux_hi ** 2)
                    + jnp.mean(uy_lo ** 2) + jnp.mean(uy_hi ** 2))

        res_losses = self.compute_residual_losses(params, state, batch,
                                                  pseudo_time=self.config.pseudo_time.enabled,
                                                  causal=self.config.causal.enabled)
        loss_dict = {"u0": u0_loss, "u_t0": u_t0_loss, "bcs": bcs_loss, **res_losses}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = vmap(vmap(vmap(self.neural_net, (None, None, None, 0)),
                           (None, None, 0, None)),
                      (None, 0, None, None))(params, self.t_star, self.x_star, self.y_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class Wave1DEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, u_ref):
        l2_error = model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def __call__(self, model, state, loss_dict, batch, u_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, u_ref)

        if self.config.causal.enabled and self.config.logging.log_causal_weights:
            causal_weight = model.compute_causal_weights(state, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
