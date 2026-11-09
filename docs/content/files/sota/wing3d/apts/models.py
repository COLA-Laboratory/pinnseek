from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator

from utils import U_INF   # [sota patch] the task's freestream


class NavierStokes3D(ForwardBVP):
    """[sota patch] their steady 2D backward-facing step, lifted to this 3D wing.

    Unchanged: the class it derives from (`ForwardBVP`, the steady base -- this
    task has no time coordinate either), the `losses` shape (one named term per
    boundary set plus whatever `compute_residual_losses` returns, which is
    where the pseudo-time stepping the paper contributes lives), and the way
    each term is written. What changed:

      * `neural_net` takes (x, y, z) instead of (x, y) and returns
        (u, v, w, p) instead of (u, v, p);
      * `r_net` gains its z-momentum equation and the `w_z` term in the
        divergence -- their two-dimensional three lines become four;
      * **the inlet is freestream, not a profile.** Their `u_in` is a parabola
        read from a coordinate array; here the inlet, the two lateral faces and
        the top all carry `u = 1, v = w = 0`, so the three become one `far`
        term;
      * **a symmetry plane is added.** `z = 0` carries `w = 0` and
        `du/dz = dv/dz = 0`. That is the only condition here that is not a
        plain value match, and nothing in the donor resembles it -- the
        normal-derivative terms are written out below.

    The wing's no-slip surface plays the role of their channel walls, and the
    outlet's `p = 0` is theirs unchanged.
    """

    def __init__(self, config, lr, tx, arch, state,
                 far_coords, wing_coords, out_coords, sym_coords, nu):
        super().__init__(config, lr, tx, arch, state)
        self.nu = nu
        self.far_coords = far_coords
        self.wing_coords = wing_coords
        self.out_coords = out_coords
        self.sym_coords = sym_coords

        self.uvwp_fn = vmap(self.neural_net, (None, 0, 0, 0))

    def neural_net(self, params, x, y, z):
        w_ = jnp.stack([x, y, z])
        outputs = self.state.apply_fn(params, w_)
        return outputs[0], outputs[1], outputs[2], outputs[3]

    def u_net(self, params, x, y, z):
        u, _, _, _ = self.neural_net(params, x, y, z)
        return u

    def v_net(self, params, x, y, z):
        _, v, _, _ = self.neural_net(params, x, y, z)
        return v

    def w_net(self, params, x, y, z):
        _, _, w, _ = self.neural_net(params, x, y, z)
        return w

    def r_net(self, params, x, y, z):
        u, v, w, p = self.neural_net(params, x, y, z)
        (du, dv, dw, dp) = jacrev(self.neural_net, argnums=(1, 2, 3))(params, x, y, z)
        u_x, u_y, u_z = du
        v_x, v_y, v_z = dv
        w_x, w_y, w_z = dw
        p_x, p_y, p_z = dp

        u_h = hessian(self.u_net, argnums=(1, 2, 3))(params, x, y, z)
        v_h = hessian(self.v_net, argnums=(1, 2, 3))(params, x, y, z)
        w_h = hessian(self.w_net, argnums=(1, 2, 3))(params, x, y, z)
        lap_u = u_h[0][0] + u_h[1][1] + u_h[2][2]
        lap_v = v_h[0][0] + v_h[1][1] + v_h[2][2]
        lap_w = w_h[0][0] + w_h[1][1] + w_h[2][2]

        adv = lambda a, b, c: u * a + v * b + w * c
        ru = adv(u_x, u_y, u_z) + p_x - self.nu * lap_u
        rv = adv(v_x, v_y, v_z) + p_y - self.nu * lap_v
        rw = adv(w_x, w_y, w_z) + p_z - self.nu * lap_w
        rc = u_x + v_y + w_z

        return ru, rv, rw, rc

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # [sota patch] freestream on the inlet, the lateral faces and the top
        c = self.far_coords
        u_f, v_f, w_f, _ = self.uvwp_fn(params, c[:, 0], c[:, 1], c[:, 2])
        far_loss = (jnp.mean((u_f - U_INF) ** 2) + jnp.mean(v_f ** 2)
                    + jnp.mean(w_f ** 2))

        # No-slip on the wing surface (their channel walls)
        c = self.wing_coords
        u_n, v_n, w_n, _ = self.uvwp_fn(params, c[:, 0], c[:, 1], c[:, 2])
        noslip_loss = (jnp.mean(u_n ** 2) + jnp.mean(v_n ** 2)
                       + jnp.mean(w_n ** 2))

        # Outflow: p = 0 (theirs unchanged)
        c = self.out_coords
        _, _, _, p_o = self.uvwp_fn(params, c[:, 0], c[:, 1], c[:, 2])
        p_out_loss = jnp.mean(p_o ** 2)

        # [sota patch] symmetry plane: w = 0 and du/dz = dv/dz = 0
        c = self.sym_coords
        _, _, w_s, _ = self.uvwp_fn(params, c[:, 0], c[:, 1], c[:, 2])
        u_z = vmap(grad(self.u_net, argnums=3), (None, 0, 0, 0))(
            params, c[:, 0], c[:, 1], c[:, 2])
        v_z = vmap(grad(self.v_net, argnums=3), (None, 0, 0, 0))(
            params, c[:, 0], c[:, 1], c[:, 2])
        sym_loss = jnp.mean(w_s ** 2) + jnp.mean(u_z ** 2) + jnp.mean(v_z ** 2)

        res_losses = self.compute_residual_losses(
            params, state, batch, pseudo_time=self.config.pseudo_time.enabled)

        return {"far": far_loss, "noslip": noslip_loss,
                "p_out": p_out_loss, "sym": sym_loss, **res_losses}

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, coords, V_test):
        """[sota patch] the scored quantity is the velocity magnitude."""
        u, v, w, _ = self.uvwp_fn(params, coords[:, 0], coords[:, 1], coords[:, 2])
        V = jnp.sqrt(u ** 2 + v ** 2 + w ** 2)
        return jnp.linalg.norm(V - V_test) / jnp.linalg.norm(V_test)


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, coords, V_ref):
        self.log_dict["l2_error"] = model.compute_l2_error(params, coords, V_ref)

    def __call__(self, model, state, loss_dict, batch, coords, V_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, coords, V_ref)

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
