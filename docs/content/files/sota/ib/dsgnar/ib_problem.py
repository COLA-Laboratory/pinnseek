"""DSGNAR on inviscid Burgers — their own Burgers example, with the viscosity removed.

`examples/burgers_double_precision.ipynb` cell 2 is `BurgersProblem`, and this
file is that class with **two** changes:

  * `nu = 0.01` -> the viscous term dropped from the residual entirely, so it
    reads `u_t + u u_x`, the task's conservation law;
  * `t_max = 1.0` -> `1.5`, the task's final time.

Everything else is theirs verbatim and, unusually for this comparison, already
matches the task exactly: `x_min, x_max = -1.0, 1.0` is the task's domain,
`periodic_bc=True` is the task's boundary condition, and
`analytical_ic = -sin(pi x)` is the task's initial condition — which
`ib_ic.csv` reproduces to 5e-11. Their samplers (`sampling.interior`,
`sampling.initial_line`) are untouched.

**NATIVE = False**, and the viscosity is why. nu = 0.01 keeps the solution
smooth: the steepening front is resolved by a diffusive layer of width ~nu, and
a smooth network can represent it. At nu = 0 the characteristics genuinely
cross and a discontinuity forms at x = 0 at t = 1/pi and stands there for the
remaining two thirds of the time domain. That is a different problem for a PINN,
not a parameter tweak, so it is marked as a port rather than their benchmark.

`dt` is set to 1.5 rather than their 1.0. Their `dt` equals their `t_max`, i.e.
the recipe is "one slab over the whole time domain"; keeping the literal 1.0 on
a 1.5-long domain would instead produce slabs [0, 1] and [1, 1.5], a split they
never chose. Preserving the structure they chose is the closer reading, and it
is recorded here rather than buried.

`ref_path` points at `ib_reference.npz`, built by `make_reference.py` from the
task's own reference rather than their shipped `burgers` entry, which is the
viscous solution. It feeds reporting only — see that script.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

import pinn
from pinn import operators as op, sampling

X, T = 0, 1  # coordinate indices, theirs
HERE = Path(__file__).parent


class InviscidBurgersProblem(pinn.Problem):
    """u_t + u u_x = 0 on (-1, 1) x (0, 1.5), periodic in x."""

    x_min, x_max, t_max = -1.0, 1.0, 1.5      # [sota patch] t_max 1.0 -> 1.5
    problem_name = "InviscidBurgers"
    ref_path = str(HERE / "ib_reference.npz")

    def __init__(self, *, n_pde, n_ic, n_bc=0):
        self.n_pde, self.n_ic, self.n_bc = n_pde, n_ic, n_bc

    def residual_fns(self):
        return {"pde": self.pde_residual, "ic": self.ic_residual}

    def pde_residual(self, model, coords):
        _u = lambda c: model(c)[0]

        u = _u(coords)
        ut = op.grad(_u, coords, T)
        ux = op.grad(_u, coords, X)

        # [sota patch] theirs is `ut + u*ux - self.nu*uxx` with nu = 0.01;
        # this task is inviscid, so the second-order term is gone.
        return jnp.array([ut + u * ux])

    def ic_residual(self, model, coords):
        # coords = [x, t, u_target]; enforce u(x, t) = u_target.
        return jnp.array([model(coords[:2])[0] - coords[2]])

    def samplers(self):
        return {
            "pde": sampling.interior(self.x_min, self.x_max),
            "ic":  sampling.initial_line(self.x_min, self.x_max),
        }

    def analytical_ic(self, x):
        return -jnp.sin(jnp.pi * x)


def make_config(seed: int) -> "pinn.RunConfig":
    """The burgers notebook's cell 4, plus the seed, dt and headless logging."""
    return pinn.RunConfig(
        seed=seed,
        dt=1.5,                       # [sota patch] theirs is 1.0 = their t_max
        network=lambda key: pinn.SIREN(
            key, InviscidBurgersProblem,
            hidden_dims=(40, 40, 40, 40),
            periodic_bc=True, n_inputs=2, n_outputs=1,
        ),
        n_pde=2**15,
        n_ic=2**14,
        residual_sketch=2000,
        parameter_sketch=2000,
        batch_size=2**13,
        probe_batch_size=2**9,
        pde_weight=1e-2,              # initial PDE weight relative to the IC (1.0)
        log_every=10**9, plot_every=10**9,   # headless; not part of the method
    )
