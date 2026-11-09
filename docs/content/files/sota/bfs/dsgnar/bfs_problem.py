"""DSGNAR on the backward-facing step — their `ldc` recipe on a different geometry.

DSGNAR's `examples/ldc.ipynb` is steady 2D Navier-Stokes with (u, v, p) on a
square, driven entirely by wall data — the same equation and the same solver
regime as this task, on a different domain with different boundary values. So
the *method* here is theirs unchanged (the `Problem` interface, `operators`,
the `SIREN`, the `RunConfig`, the sketched trust-region solver via the
step-capped `trainer_capped.py`), and what is written here is the geometry and
the boundary conditions. NATIVE = False.

Point by point:

  * **Equations** — copied from their `LidDrivenCavityProblem.pde_residual`,
    including their `2.0 *` weighting on the divergence residual. Only the
    viscosity changes: they write `1/Re` with Re = 100, this task has Re = 800.
  * **Domain** — a rectangle, not a square, so `x_min`/`x_max` are given as
    per-axis tuples. `networks._domain_bounds` already reads vector bounds
    (`if hasattr(val, "__len__")`); no shipped example uses that path.
  * **Boundary values** — the task's table rather than a lid: inflow
    `24 y (0.5 - y)` for y > 0 and 0 below at x = 0, no-slip on y = +-0.5,
    and `p = 0` at x = 15. Their cavity residual returns
    `[u - u_target, v, 0.0]`; here the third slot carries the outlet pressure
    condition, which is the one thing a cavity never has.
  * **Sampling** — their `_sample_pde`/`_sample_bc` cluster points at the lid
    and its corners, which is where *their* solution is singular. The analogue
    here is the step face and the shear layer behind it, so the interior
    sampler puts 1/8 of its points in x < 2 and the boundary sampler weights
    the inlet plane; the proportions (7/8 bulk, the rest clustered) are theirs.
  * **Steady** — no `t_max` attribute, which is how `trainer64` detects a
    steady problem (`is_steady = not hasattr(problem, 't_max')`), and no `ic`.

`RunConfig` is the ldc notebook's cell 2 verbatim apart from `n_inputs`/
`n_outputs` (unchanged at 2/3) and the seed.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

import pinn
from pinn import operators as op

X, Y = 0, 1
HERE = Path(__file__).parent

XL, XU = 0.0, 15.0
YL, YU = -0.5, 0.5


def _inflow_u(y):
    return jnp.where(y > 0, 24.0 * y * (0.5 - y), 0.0)


class BFSProblem(pinn.Problem):
    """Steady backward-facing step at Re = 800; outputs (u, v, p)."""

    x_min, x_max = (XL, YL), (XU, YU)      # per-axis bounds
    Re = 800.0
    problem_name = "BackwardFacingStep"
    ref_path = str(HERE / "bfs_reference.npz")

    def __init__(self, *, n_pde, n_bc, n_ic=0):
        self.n_pde, self.n_bc, self.n_ic = n_pde, n_bc, n_ic

    def residual_fns(self):
        return {"pde": self.pde_residual, "bc": self.bc_residual}

    def pde_residual(self, model, coords):
        u_ = lambda c: model(c)[0]
        v_ = lambda c: model(c)[1]
        p_ = lambda c: model(c)[2]
        u, v = u_(coords), v_(coords)
        inv_Re = 1.0 / self.Re
        res_u = (u * op.grad(u_, coords, X) + v * op.grad(u_, coords, Y)
                 + op.grad(p_, coords, X)
                 - inv_Re * (op.grad2(u_, coords, X) + op.grad2(u_, coords, Y)))
        res_v = (u * op.grad(v_, coords, X) + v * op.grad(v_, coords, Y)
                 + op.grad(p_, coords, Y)
                 - inv_Re * (op.grad2(v_, coords, X) + op.grad2(v_, coords, Y)))
        res_div = op.grad(u_, coords, X) + op.grad(v_, coords, Y)
        return jnp.array([res_u, res_v, 2.0 * res_div])

    def bc_residual(self, model, coords):
        x, y = coords[0], coords[1]
        u, v, p = model(coords)[0], model(coords)[1], model(coords)[2]
        at_out = jnp.isclose(x, XU)
        # velocity is prescribed everywhere but the outlet
        u_target = jnp.where(jnp.isclose(x, XL), _inflow_u(y), 0.0)
        r_u = jnp.where(at_out, 0.0, u - u_target)
        r_v = jnp.where(at_out, 0.0, v)
        # the outlet carries p = 0 and nothing else
        r_p = jnp.where(at_out, p, 0.0)
        return jnp.array([r_u, r_v, r_p])

    def samplers(self):
        return {"pde": self._sample_pde, "bc": self._sample_bc}

    def _sample_pde(self, key, n, *_):
        # 7/8 bulk, 1/8 clustered behind the step where the shear layer sits
        n_step = n // 8
        n_bulk = n - n_step
        k_b, k_sx, k_sy = jax.random.split(key, 3)
        bulk = jax.random.uniform(k_b, (n_bulk, 2),
                                  minval=jnp.array([XL, YL]),
                                  maxval=jnp.array([XU, YU]))
        step = jnp.stack([2.0 * jax.random.uniform(k_sx, (n_step,)) ** 2,
                          jax.random.uniform(k_sy, (n_step,), minval=YL, maxval=YU)],
                         axis=1)
        return jnp.concatenate([bulk, step], axis=0)

    def _sample_bc(self, key, n, *_):
        # the four sides, with the inlet plane over-weighted (it is the only
        # one carrying a non-constant profile)
        q = n // 4
        k1, k2, k3, k4 = jax.random.split(key, 4)
        inlet = jnp.stack([jnp.full((q,), XL),
                           jax.random.uniform(k1, (q,), minval=YL, maxval=YU)], axis=1)
        outlet = jnp.stack([jnp.full((q,), XU),
                            jax.random.uniform(k2, (q,), minval=YL, maxval=YU)], axis=1)
        top = jnp.stack([jax.random.uniform(k3, (q,), minval=XL, maxval=XU),
                         jnp.full((q,), YU)], axis=1)
        bot = jnp.stack([jax.random.uniform(k4, (n - 3 * q,), minval=XL, maxval=XU),
                         jnp.full((n - 3 * q,), YL)], axis=1)
        return jnp.concatenate([inlet, outlet, top, bot], axis=0)


def make_config(seed: int) -> "pinn.RunConfig":
    """The ldc notebook's cell 2, plus the seed and headless logging."""
    return pinn.RunConfig(
        seed=seed,
        network=lambda key: pinn.SIREN(
            key, BFSProblem, time_dependent=False,
            hidden_dims=(60, 60, 60, 60), periodic_bc=False,
            n_inputs=2, n_outputs=3,
        ),
        n_pde=2**15, n_bc=2**14,
        residual_sketch=4000, parameter_sketch=4000,
        batch_size=2**12, probe_batch_size=2**8,
        pde_weight=1e-2,
        log_every=10**9, plot_every=10**9,
    )
