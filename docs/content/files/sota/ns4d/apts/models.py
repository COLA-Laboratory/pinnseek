from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator

from utils import forcing   # [sota patch] problem.md's f, not their body force


class NavierStokes4D(ForwardIVP):
    """[sota patch] their 2D+time Kolmogorov flow, moved onto this 3D+time task.

    Unchanged: the class it derives from (`ForwardIVP`), the `losses` shape
    (initial-condition terms plus the causally-weighted residual losses
    `compute_residual_losses` builds, which is where the pseudo-time stepping
    the paper contributes lives), the `t / t_max` input normalisation, and the
    way each term is written. What changed:

      * `neural_net` takes (t, x, y, z) instead of (t, x, y) and returns the
        three velocity components instead of (u, v, p). **There is no pressure
        here**: the equation is in vorticity-velocity form, where pressure has
        been eliminated by taking the curl.

      * `r_net` is the task's equation,
            w_t + (u . grad) w - (w . grad) u - nu lap(w) - f = 0,  div u = 0
        with `w = curl u`. The vorticity is *not* a network output, so every
        `w` in that line is a first derivative of the velocity and `lap(w)` is
        a **third** derivative of it. Their Kolmogorov residual needed only
        second derivatives; this is the one structural cost of the change.

      * the body force `2 sin(4 pi y)` becomes `problem.md`'s three-component
        forcing (see utils.py), which is also SPINN's own
        `navier_stokes4d_forcing_term`.

      * the initial condition covers three components rather than two, taken
        from `task/ns4d_ic.csv`.

    Periodicity is handled the same way theirs is and costs nothing: the
    architecture's `periodicity` block embeds x, y and z with period 2 pi, so
    the boundary condition is exact and there is no boundary loss.
    """

    def __init__(self, config, lr, tx, arch, state, t_max, nu):
        super().__init__(config, lr, tx, arch, state)

        self.t_max = t_max
        self.nu = nu

        # [sota patch] the pseudo-time correction needs one proximal term per
        # residual component, and `PINN.__init__` builds it from `neural_net`:
        #     res_pred + pts_weights[:, None] * (sols_pred - sols_prev)
        # In every published APTS example the two counts agree, because the
        # residual has one line per emitted field. Here they do not: the network
        # emits three velocity components but the residual has four lines, and
        # the quantities the three evolution equations advance are the
        # VORTICITY components, not the velocity. Left as built, this raises
        #     mul got incompatible shapes: (4, 1), (3, 8192)
        # So `sol_pred_fn` is rebuilt on the quantities the residuals actually
        # evolve: (wx, wy, wz) for the three vorticity equations, and zero for
        # the divergence, which carries no time derivative and so takes no
        # proximal term. That is the same reasoning their own code encodes --
        # one proximal term per evolution equation -- and the same choice the
        # Scale-PINN port of this case makes for its ER term.
        self.sol_pred_fn = vmap(self.sol_net, (None, 0, 0, 0, 0))

        # vmap functions
        self.ux_pred_fn = vmap(vmap(self.ux_net, (None, None, 0, 0, 0)),
                               (None, 0, None, None, None))
        self.wx_pred_fn = vmap(vmap(self.wx_net, (None, None, 0, 0, 0)),
                               (None, 0, None, None, None))
        self.wy_pred_fn = vmap(vmap(self.wy_net, (None, None, 0, 0, 0)),
                               (None, 0, None, None, None))
        self.wz_pred_fn = vmap(vmap(self.wz_net, (None, None, 0, 0, 0)),
                               (None, 0, None, None, None))

    def neural_net(self, params, t, x, y, z):
        t = t / self.t_max
        w = jnp.stack([t, x, y, z])
        outputs = self.state.apply_fn(params, w)
        return outputs[0], outputs[1], outputs[2]      # velocity (ux, uy, uz)

    def ux_net(self, params, t, x, y, z):
        ux, _, _ = self.neural_net(params, t, x, y, z)
        return ux

    def uy_net(self, params, t, x, y, z):
        _, uy, _ = self.neural_net(params, t, x, y, z)
        return uy

    def uz_net(self, params, t, x, y, z):
        _, _, uz = self.neural_net(params, t, x, y, z)
        return uz

    # ── vorticity: the autodiff curl of the velocity ───────────────────
    def wx_net(self, params, t, x, y, z):
        uz_y = grad(self.uz_net, argnums=3)(params, t, x, y, z)
        uy_z = grad(self.uy_net, argnums=4)(params, t, x, y, z)
        return uz_y - uy_z

    def wy_net(self, params, t, x, y, z):
        ux_z = grad(self.ux_net, argnums=4)(params, t, x, y, z)
        uz_x = grad(self.uz_net, argnums=2)(params, t, x, y, z)
        return ux_z - uz_x

    def wz_net(self, params, t, x, y, z):
        uy_x = grad(self.uy_net, argnums=2)(params, t, x, y, z)
        ux_y = grad(self.ux_net, argnums=3)(params, t, x, y, z)
        return uy_x - ux_y

    def sol_net(self, params, t, x, y, z):
        """[sota patch] the quantities the four residual lines evolve."""
        return (self.wx_net(params, t, x, y, z),
                self.wy_net(params, t, x, y, z),
                self.wz_net(params, t, x, y, z),
                jnp.zeros_like(t))

    def r_net(self, params, t, x, y, z):
        ux, uy, uz = self.neural_net(params, t, x, y, z)
        wx = self.wx_net(params, t, x, y, z)
        wy = self.wy_net(params, t, x, y, z)
        wz = self.wz_net(params, t, x, y, z)

        # velocity gradients, for the stretching term and the divergence
        (ux_x, ux_y, ux_z) = grad(self.ux_net, argnums=(2, 3, 4))(params, t, x, y, z)
        (uy_x, uy_y, uy_z) = grad(self.uy_net, argnums=(2, 3, 4))(params, t, x, y, z)
        (uz_x, uz_y, uz_z) = grad(self.uz_net, argnums=(2, 3, 4))(params, t, x, y, z)

        fx, fy, fz = forcing(t, x, y, z)

        def component(vor_net, vel_grads, f):
            v_t, v_x, v_y, v_z = grad(vor_net, argnums=(1, 2, 3, 4))(
                params, t, x, y, z)
            h = hessian(vor_net, argnums=(2, 3, 4))(params, t, x, y, z)
            lap = h[0][0] + h[1][1] + h[2][2]
            g_x, g_y, g_z = vel_grads
            return (v_t + ux * v_x + uy * v_y + uz * v_z
                    - (wx * g_x + wy * g_y + wz * g_z)
                    - self.nu * lap - f)

        rx = component(self.wx_net, (ux_x, ux_y, ux_z), fx)
        ry = component(self.wy_net, (uy_x, uy_y, uy_z), fy)
        rz = component(self.wz_net, (uz_x, uz_y, uz_z), fz)
        rc = ux_x + uy_y + uz_z

        return rx, ry, rz, rc

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # Unpack batch
        ics_batch = batch["ics"]
        res_batch = batch["res"]

        # Initial condition loss
        coords_batch, uvw_batch = ics_batch
        u_batch, v_batch, w_batch = uvw_batch[:, 0], uvw_batch[:, 1], uvw_batch[:, 2]

        ux_ic, uy_ic, uz_ic = vmap(self.neural_net, (None, None, 0, 0, 0))(
            params, 0.0, coords_batch[:, 0], coords_batch[:, 1], coords_batch[:, 2]
        )
        u_ic_loss = jnp.mean((ux_ic - u_batch) ** 2)
        v_ic_loss = jnp.mean((uy_ic - v_batch) ** 2)
        w_ic_loss = jnp.mean((uz_ic - w_batch) ** 2)

        res_losses = self.compute_residual_losses(params, state, res_batch,
                                                  pseudo_time=self.config.pseudo_time.enabled,
                                                  causal=self.config.causal.enabled)

        loss_dict = {
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "w_ic": w_ic_loss,          # [sota patch] three components, not two
            **res_losses
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t_ref, coords, w_ref):
        """[sota patch] the scored quantity is the vorticity, all three
        components, each normalised on its own -- the task's own protocol."""
        wx = self.wx_pred_fn(params, t_ref, coords[:, 0], coords[:, 1], coords[:, 2])
        wy = self.wy_pred_fn(params, t_ref, coords[:, 0], coords[:, 1], coords[:, 2])
        wz = self.wz_pred_fn(params, t_ref, coords[:, 0], coords[:, 1], coords[:, 2])
        pred = jnp.stack([wx, wy, wz], axis=-1)
        errs = [jnp.linalg.norm(pred[..., i] - w_ref[..., i])
                / jnp.linalg.norm(w_ref[..., i]) for i in range(3)]
        return errs[0], errs[1], errs[2]


class NavierStokes4DEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, t, coords, w_ref):
        ex, ey, ez = model.compute_l2_error(params, t, coords, w_ref)
        self.log_dict["wx_error"] = ex
        self.log_dict["wy_error"] = ey
        self.log_dict["wz_error"] = ez

    def __call__(self, model, state, loss_dict, batch, t, coords, w_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch["res"])

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, t, coords, w_ref)

        if self.config.causal.enabled and self.config.logging.log_causal_weights:
            causal_weight = model.compute_causal_weights(state, batch["res"])
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
