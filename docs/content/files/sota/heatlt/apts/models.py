from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator

from utils import D_COEF, A_COEF, T_MAX   # [sota patch] the task's constants


class HeatLT2D(ForwardIVP):
    """[sota patch] their 1D constant-speed advection model, retargeted.

    Unchanged: the class it derives from, the loss *shape* (`ics` plus the
    causally-weighted residual losses their `compute_residual_losses` builds),
    and the way each term is written. What changed:

      * `neural_net` takes (t, x, y) instead of (t, x);
      * `r_net` is the task's equation,
            u_t - D (u_xx + u_yy)
                - A sin(u^2) (1 + 2 sin(pi t / 4)) sin(4 pi x) sin(2 pi y)
        with D = 0.001 and A = 5. Everything is analytic; nothing is
        interpolated;
      * the initial term is `sin(4 pi x) sin(3 pi y)` on an (x, y) lattice
        rather than a supplied profile on a line;
      * **a Dirichlet boundary term is added.** The donor is periodic in x and
        needs no boundary loss; this task is u = 0 on all four sides, and the
        matching `bcs` weight is added to `configs/base.py` because jaxpi
        requires the weight dict to match the loss dict key for key.

    u == 0 satisfies the PDE and all four sides exactly, because the source
    carries `sin(u^2)` and vanishes with u; only the initial term rules it out.
    The published weights start every term at 1.0 and the dynamic weighting
    moves them from there, which is left alone.
    """

    def __init__(self, config, lr, tx, arch, state, d):
        super().__init__(config, lr, tx, arch, state)
        self.t_star, self.x_star, self.y_star = d["t_star"], d["x_star"], d["y_star"]
        self.ref_coords = d["coords"]
        self.t0 = 0.0
        self.t1 = T_MAX

        n = 64                                   # IC lattice
        g = jnp.linspace(0.0, 1.0, n)
        gx, gy = jnp.meshgrid(g, g, indexing="ij")
        self.ic_x, self.ic_y = gx.reshape(-1), gy.reshape(-1)
        self.ic_u = jnp.sin(4.0 * jnp.pi * self.ic_x) * jnp.sin(3.0 * jnp.pi * self.ic_y)

        m = 48                                   # Dirichlet faces
        s_ = jnp.linspace(0.0, 1.0, m)
        tt, ss = jnp.meshgrid(jnp.linspace(0.0, T_MAX, m), s_, indexing="ij")
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
        u_xx = grad(grad(self.neural_net, argnums=2), argnums=2)(params, t, x, y)
        u_yy = grad(grad(self.neural_net, argnums=3), argnums=3)(params, t, x, y)
        src = (A_COEF * jnp.sin(u ** 2)
               * (1.0 + 2.0 * jnp.sin(jnp.pi * t / 4.0))
               * jnp.sin(4.0 * jnp.pi * x) * jnp.sin(2.0 * jnp.pi * y))
        return u_t - D_COEF * (u_xx + u_yy) - src

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        u_ic = vmap(self.neural_net, (None, None, 0, 0))(
            params, self.t0, self.ic_x, self.ic_y)
        ics_loss = jnp.mean((u_ic - self.ic_u) ** 2)

        # [sota patch] Dirichlet sides, which the periodic donor does not have
        u_bc = vmap(self.neural_net, (None, 0, 0, 0))(
            params, self.bc_t, self.bc_x, self.bc_y)
        bcs_loss = jnp.mean(u_bc ** 2)

        res_losses = self.compute_residual_losses(params, state, batch,
                                                  pseudo_time=self.config.pseudo_time.enabled,
                                                  causal=self.config.causal.enabled)
        return {"ics": ics_loss, "bcs": bcs_loss, **res_losses}

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        # [sota patch] their evaluator calls this as (params, u_test), so the
        # reference coordinates are held on the object rather than passed in.
        # The reference is a COMSOL mesh, not a tensor grid, so the points are
        # evaluated directly instead of as a nested vmap over axes.
        u_pred = vmap(self.neural_net, (None, 0, 0, 0))(
            params, self.ref_coords[:, 0], self.ref_coords[:, 1], self.ref_coords[:, 2])
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
