from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator

# [sota patch] the task's driven inflow, its geometry, and the boundary sets
from utils import inflow_profile, in_fluid, sample_boundary_sets, XU


class NavierStokes2D(ForwardIVP):
    """[sota patch] their unsteady Kolmogorov flow, moved onto this channel.

    Unchanged: the class it derives from (`ForwardIVP`), the `losses` shape
    (initial-condition terms plus the causally-weighted residual losses
    `compute_residual_losses` builds, which is where the pseudo-time stepping
    the paper contributes lives), the `t / t_max` input normalisation, and the
    residual itself -- their Kolmogorov residual *is* the task's momentum and
    continuity equations. What changed:

      * **the body force is gone.** Theirs adds `2 sin(4 pi y)` to `ru`; this
        flow is driven by its inlet, not by a body force.

      * **the domain has a hole and four real boundaries.** Their box is
        periodic in both directions and carries no boundary loss at all. Here
        the residual is masked outside the cylinder, and four boundary terms
        are added, following `examples/bfs_flow/models.py`, which is the
        repository's own channel-with-inlet-and-outlet case:
            `u_in` / `v_in` -- the driven inlet profile at x = 0,
            `uv_wall`       -- no-slip on the two channel walls and the
                               cylinder surface,
            `p_out`         -- p = 0 at x = 2.2.

      * `nu` is the task's 1e-3 rather than read from their .npy.

    Because the pressure is a network output, the outlet condition can be
    written down here. The SPINN port on this same case cannot write it: its
    published formulation is in vorticity form, and taking the curl is exactly
    what removes the pressure.
    """

    def __init__(self, config, lr, tx, arch, state, t_max, nu):
        super().__init__(config, lr, tx, arch, state)

        self.t_max = t_max
        self.nu = nu

        # [sota patch] boundary coordinates; theirs is periodic and has none
        self.x_in, self.x_out, self.x_wall = sample_boundary_sets(128, 128)

        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))
        self.v_pred_fn = vmap(vmap(self.v_net, (None, None, 0, 0)), (None, 0, None, None))
        self.u_flat_fn = vmap(self.u_net, (None, 0, 0, 0))
        self.v_flat_fn = vmap(self.v_net, (None, 0, 0, 0))
        self.p_flat_fn = vmap(self.p_net, (None, 0, 0, 0))

    def neural_net(self, params, t, x, y):
        t = t / self.t_max
        z = jnp.stack([t, x, y])
        outputs = self.state.apply_fn(params, z)
        return outputs[0], outputs[1], outputs[2]

    def u_net(self, params, t, x, y):
        u, _, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v, _ = self.neural_net(params, t, x, y)
        return v

    def p_net(self, params, t, x, y):
        _, _, p = self.neural_net(params, t, x, y)
        return p

    def r_net(self, params, t, x, y):
        u, v, p = self.neural_net(params, t, x, y)
        (u_t, u_x, u_y), (v_t, v_x, v_y), (_, p_x, p_y) = jacrev(
            self.neural_net, argnums=(1, 2, 3))(params, t, x, y)

        u_hessian = hessian(self.u_net, argnums=(2, 3))(params, t, x, y)
        v_hessian = hessian(self.v_net, argnums=(2, 3))(params, t, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]
        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        # [sota patch] no body force; this flow is driven by its inlet
        ru = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        rv = v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        rc = u_x + v_y

        # [sota patch] the cylinder is cut out of the residual; the 1/kept
        # rescale in `losses` turns ForwardIVP's plain mean into the masked one
        keep = in_fluid(x, y)
        return ru * keep, rv * keep, rc * keep

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        ics_batch = batch["ics"]
        res_batch = batch["res"]

        # Initial conditions loss
        coords_batch, uv_batch = ics_batch
        u_batch, v_batch = uv_batch[:, 0], uv_batch[:, 1]
        u_ic_pred, v_ic_pred, _ = vmap(self.neural_net, (None, None, 0, 0))(
            params, 0.0, coords_batch[:, 0], coords_batch[:, 1]
        )
        u_ic_loss = jnp.mean((u_ic_pred - u_batch) ** 2)
        v_ic_loss = jnp.mean((v_ic_pred - v_batch) ** 2)

        # [sota patch] the four boundary terms. Time is drawn from the residual
        # batch's own time column, so the boundary is enforced over exactly the
        # window the residual is being sampled on. The draw is a *stride* over
        # that column, not its head: jaxpi's `UniformSampler` sorts every batch
        # by its first coordinate (`sort_axis=0`, for the causal weighting), so
        # `t_b[:m]` would be the m EARLIEST times in the batch. With m = 128
        # inlet points out of a batch of 8192 that is t in [0, 0.018] of a
        # [0, 1.2] window -- 1.5% of it -- and the inlet, the walls and the
        # outlet would go unconstrained over the rest. Measured, seed 1, same
        # 1360 steps and the same wall clock: head 0.7993, stride 0.6037.
        t_b = res_batch[:, 0]
        n_b = t_b.shape[0]

        def at(coords, fn):
            m = coords.shape[0]
            idx = jnp.linspace(0, n_b - 1, m).astype(jnp.int32)
            tt = t_b[idx]
            return fn(params, tt, coords[:, 0], coords[:, 1]), tt

        (u_in, t_in) = at(self.x_in, self.u_flat_fn)
        (v_in, _) = at(self.x_in, self.v_flat_fn)
        u_in_target = inflow_profile(t_in, self.x_in[:, 1])
        u_in_loss = jnp.mean((u_in - u_in_target) ** 2)
        v_in_loss = jnp.mean(v_in ** 2)

        (u_w, _) = at(self.x_wall, self.u_flat_fn)
        (v_w, _) = at(self.x_wall, self.v_flat_fn)
        uv_wall_loss = jnp.mean(u_w ** 2) + jnp.mean(v_w ** 2)

        (p_o, _) = at(self.x_out, self.p_flat_fn)
        p_out_loss = jnp.mean(p_o ** 2)

        res_losses = self.compute_residual_losses(params, state, res_batch,
                                                  pseudo_time=self.config.pseudo_time.enabled,
                                                  causal=self.config.causal.enabled)
        # [sota patch] undo the dilution the mask causes in their plain mean
        kept = jnp.mean(in_fluid(res_batch[:, 1], res_batch[:, 2]))
        res_losses = {k: v / kept for k, v in res_losses.items()}

        return {"u_ic": u_ic_loss, "v_ic": v_ic_loss,
                "u_in": u_in_loss, "v_in": v_in_loss,
                "uv_wall": uv_wall_loss, "p_out": p_out_loss,
                **res_losses}

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, coords, uv_ref):
        u = self.u_flat_fn(params, coords[:, 0], coords[:, 1], coords[:, 2])
        v = self.v_flat_fn(params, coords[:, 0], coords[:, 1], coords[:, 2])
        pred = jnp.stack([u, v], axis=1)
        return jnp.linalg.norm(pred - uv_ref) / jnp.linalg.norm(uv_ref)


class NavierStokes2DEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, coords, uv_ref):
        self.log_dict["l2_error"] = model.compute_l2_error(params, coords, uv_ref)

    def __call__(self, model, state, loss_dict, batch, coords, uv_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch["res"])

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, coords, uv_ref)

        if self.config.causal.enabled and self.config.logging.log_causal_weights:
            causal_weight = model.compute_causal_weights(state, batch["res"])
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
