from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator

from utils import coef_at   # [sota patch] task/baseline.py's own lookup


class Smoke2D(ForwardIVP):
    """[sota patch] their 1D constant-speed advection model, extended to this task.

    Unchanged: the class it derives from, the loss *shape* (`ics` plus the
    causally-weighted residual losses their `compute_residual_losses` builds),
    and the way each term is written. What changed:

      * `neural_net` takes (t, x, y) instead of (t, x);
      * `r_net` is the task's equation,
            u_t + a u_x + b u_y + c u - DXC u_xx - DYC u_yy - f = 0,
        with a, b, c, f the ERA5/GFAS fields looked up by `coef_at` --
        `task/baseline.py:coef_at` reproduced exactly, trilinear as problem.md
        specifies, with the (t, y, x) storage order the task warns about;
      * the initial condition is u = 0 on the whole t = 0 plane (the atmosphere
        starts clean; the source f builds the plume), so the IC term is the
        squared network value rather than an error against a supplied profile;
      * **a Dirichlet boundary term is added.** The donor is periodic in x and
        needs no boundary loss; this task is u = 0 on all four sides.
    """

    def __init__(self, config, lr, tx, arch, state, d):
        super().__init__(config, lr, tx, arch, state)
        self.A, self.B, self.C, self.F = d["A"], d["B"], d["C"], d["F"]
        self.dxc, self.dyc = d["dxc"], d["dyc"]
        self.t_star, self.x_star, self.y_star = d["t_star"], d["x_star"], d["y_star"]
        self.t0 = self.t_star[0]
        self.t1 = self.t_star[-1]

        n = 64                                  # IC plane, u(0,x,y) = 0
        g = jnp.linspace(0.0, 1.0, n)
        gx, gy = jnp.meshgrid(g, g, indexing="ij")
        self.ic_x, self.ic_y = gx.reshape(-1), gy.reshape(-1)

        m = 48                                  # Dirichlet faces, u = 0
        s_ = jnp.linspace(0.0, 1.0, m)
        tt, ss = jnp.meshgrid(s_, s_, indexing="ij")
        tt, ss = tt.reshape(-1), ss.reshape(-1)
        zero, one = jnp.zeros_like(ss), jnp.ones_like(ss)
        self.bc_t = jnp.concatenate([tt, tt, tt, tt])
        self.bc_x = jnp.concatenate([zero, one, ss, ss])
        self.bc_y = jnp.concatenate([ss, ss, zero, one])

    def neural_net(self, params, t, x, y):
        z = jnp.stack([t, x, y])
        u = self.state.apply_fn(params, z)
        return u[0]

    def r_net(self, params, t, x, y):
        u = self.neural_net(params, t, x, y)
        u_t = grad(self.neural_net, argnums=1)(params, t, x, y)
        u_x = grad(self.neural_net, argnums=2)(params, t, x, y)
        u_y = grad(self.neural_net, argnums=3)(params, t, x, y)
        u_xx = grad(grad(self.neural_net, argnums=2), argnums=2)(params, t, x, y)
        u_yy = grad(grad(self.neural_net, argnums=3), argnums=3)(params, t, x, y)
        a = coef_at(self.A, t, x, y)
        b = coef_at(self.B, t, x, y)
        c = coef_at(self.C, t, x, y)
        f = coef_at(self.F, t, x, y)
        return (u_t + a * u_x + b * u_y + c * u
                - self.dxc * u_xx - self.dyc * u_yy - f)

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        u_ic = vmap(self.neural_net, (None, None, 0, 0))(
            params, self.t0, self.ic_x, self.ic_y)
        ics_loss = jnp.mean(u_ic ** 2)

        # [sota patch] Dirichlet sides, which the periodic donor does not have
        u_bc = vmap(self.neural_net, (None, 0, 0, 0))(
            params, self.bc_t, self.bc_x, self.bc_y)
        bcs_loss = jnp.mean(u_bc ** 2)

        res_losses = self.compute_residual_losses(params, state, batch,
                                                  pseudo_time=self.config.pseudo_time.enabled,
                                                  causal=self.config.causal.enabled)
        loss_dict = {"ics": ics_loss, "bcs": bcs_loss, **res_losses}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = vmap(vmap(vmap(self.neural_net, (None, None, None, 0)),
                           (None, None, 0, None)),
                      (None, 0, None, None))(params, self.t_star, self.x_star, self.y_star)
        return jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)


class Advection1DEvaluator(BaseEvaluator):
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
