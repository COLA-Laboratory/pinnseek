"""DSGNAR's own KdV problem definition, verbatim from examples/kdv.ipynb.

The equation, domain, dispersion (mu = 0.022) and initial condition
u(x,0) = cos(pi x) are identical to the task's problem.md — this is the
authors' own example of exactly this benchmark. Cells 2 and 3 of the
notebook are reproduced below unchanged; the only additions are the module
imports the notebook did at the top.
"""
from __future__ import annotations

import jax.numpy as jnp

import pinn
from pinn import operators as op, sampling

X, T = 0, 1


class KdVProblem(pinn.Problem):
    """u_t + eta u u_x + mu^2 u_xxx = 0, periodic on (-1, 1)."""

    x_min, x_max, t_max = -1.0, 1.0, 1.0
    eta, mu = 1.0, 0.022
    problem_name = "KdV"
    ref_path = pinn.reference_path("kdv")

    def __init__(self, *, n_pde, n_ic, n_bc=0):
        self.n_pde, self.n_ic, self.n_bc = n_pde, n_ic, n_bc

    def residual_fns(self):
        return {"pde": self.pde_residual, "ic": self.ic_residual}

    def pde_residual(self, model, coords):
        u = lambda c: model(c)[0]
        r = (op.grad(u, coords, T)
             + self.eta * u(coords) * op.grad(u, coords, X)
             + self.mu**2 * op.grad_n(u, coords, X, 3))
        return jnp.array([r])

    def ic_residual(self, model, coords):
        return jnp.array([model(coords[:2])[0] - coords[2]])

    def samplers(self):
        return {"pde": sampling.interior(self.x_min, self.x_max),
                "ic":  sampling.initial_line(self.x_min, self.x_max)}

    def analytical_ic(self, x):
        return jnp.cos(jnp.pi * x)


def make_config(seed: int) -> "pinn.RunConfig":
    """Cell 3 of the notebook, plus the seed."""
    return pinn.RunConfig(
        seed=seed,
        dt=0.25,
        network=lambda key: pinn.SIREN(
            key, KdVProblem, hidden_dims=(60, 60, 60, 60),
            periodic_bc=True, n_inputs=2, n_outputs=1,
        ),
        n_pde=2**15, n_ic=2**14,
        residual_sketch=4000, parameter_sketch=4000,
        batch_size=2**14, probe_batch_size=2**10,
        pde_weight=1e-4,
        # display-only: the notebook logs/plots every 5 steps; headless under a
        # wall-clock budget we log only at pass boundaries
        log_every=10**9, plot_every=10**9,
    )
