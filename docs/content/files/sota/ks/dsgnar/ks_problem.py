"""DSGNAR's own KS problem definition, from examples/ks.ipynb.

Cell 1 of the notebook is reproduced below unchanged (`KSProblem`): the same
equation and coefficients (alpha = 100/16, beta = 100/16**2,
gamma = 100/16**4), the same periodic domain x in [0, 2*pi], and the same
initial condition cos(x)(1 + sin(x)) as the task's problem.md and ks_ic.csv
(which agree to 1e-9).

Two things differ from the notebook, both because the notebook solves a
longer time interval than the task asks for, and both recorded here rather
than buried:

  1. `t_max` is 0.4, the task's final time; the notebook's is 1.0.
  2. `slab_boundaries` is the notebook's own list **truncated at t = 0.4**
     rather than rescaled. Their boundaries are
     `[0, .2344, .4375, .6094, .75, .8594, .9375, .9844, 1]` — clustered
     towards t = 1 "where the dynamics are sharpest", so the widths *shrink*
     with time. Truncating keeps exactly the resolution the authors chose for
     the interval the task actually covers ([0, .2344] then [.2344, .4]);
     rescaling the whole list onto [0, 0.4] would instead impose their
     late-time refinement on early time, which is not what they asked for.

Everything else in `RunConfig` is cell 2 of the notebook verbatim, plus the
seed and headless logging.
"""
from __future__ import annotations

import jax.numpy as jnp

import pinn
from pinn import operators as op, sampling

X, T = 0, 1

TASK_T_MAX = 0.4

# the notebook's own line, cell 2:
#   slab_boundaries = jnp.hstack([0.0, 1.0 - ((jnp.arange(8) / 8)**2)[::-1]])
NOTEBOOK_SLABS = [float(v) for v in
                  jnp.hstack([0.0, 1.0 - ((jnp.arange(8) / 8) ** 2)[::-1]])]
TASK_SLABS = ([b for b in NOTEBOOK_SLABS if b < TASK_T_MAX] + [TASK_T_MAX])


class KSProblem(pinn.Problem):
    """u_t + alpha u u_x + beta u_xx + gamma u_xxxx = 0, periodic on [0, 2pi]."""

    x_min, x_max = 0.0, 2.0 * jnp.pi
    t_max = TASK_T_MAX                      # notebook: 1.0; the task ends at 0.4
    alpha, beta, gamma = 100.0 / 16.0, 100.0 / 16.0**2, 100.0 / 16.0**4
    problem_name = "KS"
    ref_path = pinn.reference_path("ks")

    def __init__(self, *, n_pde, n_ic, n_bc=0):
        self.n_pde, self.n_ic, self.n_bc = n_pde, n_ic, n_bc

    def residual_fns(self):
        return {"pde": self.pde_residual, "ic": self.ic_residual}

    def pde_residual(self, model, coords):
        u = lambda c: model(c)[0]
        r = (op.grad(u, coords, T)
             + self.alpha * u(coords) * op.grad(u, coords, X)
             + self.beta * op.grad2(u, coords, X)
             + self.gamma * op.grad_n(u, coords, X, 4))
        return jnp.array([r])

    def ic_residual(self, model, coords):
        return jnp.array([model(coords[:2])[0] - coords[2]])

    def samplers(self):
        return {"pde": sampling.interior(self.x_min, self.x_max),
                "ic":  sampling.initial_line(self.x_min, self.x_max)}

    def analytical_ic(self, x):
        return jnp.cos(x) * (1.0 + jnp.sin(x))


def make_config(seed: int) -> "pinn.RunConfig":
    """Cell 2 of the notebook, with the slab list truncated to the task's
    time window, plus the seed and headless logging."""
    return pinn.RunConfig(
        seed=seed,
        slab_boundaries=list(TASK_SLABS),
        network=lambda key: pinn.MLP(
            key, KSProblem, hidden_dims=(70, 70, 70, 70),
            periodic_bc=True, n_inputs=2, n_outputs=1,
        ),
        n_pde=2**15, n_ic=2**14,
        residual_sketch=5000, parameter_sketch=5000,
        batch_size=2**12, probe_batch_size=2**8,
        pde_weight=1e-3,
        # display-only: the notebook logs/plots every 5 steps; headless under a
        # wall-clock budget we log only at pass boundaries
        log_every=10**9, plot_every=10**9,
    )
