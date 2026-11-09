"""DSGNAR on the lid-driven cavity at Re = 3200 — their own cavity, at this task's operating point.

`examples/ldc.ipynb` cell 2 is `LidDrivenCavityProblem`, and this file is that
class with **two** changes, both of which move it onto the task:

  * `Re = 100.0` -> `Re = 3200.0`  (the task's nu = 1/3200);
  * the lid target `4 x (1 - x)` -> `1.0`.

Everything else — the residual (including their `2.0 *` weight on the
divergence), the domain, the corner-clustered `_sample_pde` and `_sample_bc`,
the `RunConfig` — is theirs verbatim.

**NATIVE = False**, and the second change is why. Their lid is *regularized*:
`4 x (1 - x)` vanishes at both top corners, so their velocity field is
continuous everywhere. The task's lid is u = 1 all the way to the corners,
which makes the boundary data discontinuous and the pressure singular there.
Combined with Re = 3200 against their Re = 100 — a single primary vortex
becomes a primary vortex plus secondary and tertiary corner vortices — this is
their geometry and their solver on a flow they do not publish. Marking it
native would overclaim; the comparison's other cavity entries (Scale-PINN and
APTS) genuinely do publish Re = 3200 with u = 1 and are marked native.

Their corner clustering is kept exactly as written and is if anything more
apt here than in their own notebook: `_sample_pde` puts 1/16 of the interior
points in `y > 0.9` and another 1/16 inside the two top corners, and
`_sample_bc` adds `n/16` extra points on the lid within 0.1 of each corner.
That is where this task's singularity lives.

`ref_path` points at `ldc_reference.npz`, built by `make_reference.py` from the
task's own reference rather than their shipped `ldc` entry, which is the
Re = 100 regularized solution. It feeds reporting only — see that script.

Steady: no `t_max` attribute, which is how `trainer64` detects a steady problem
(`is_steady = not hasattr(problem, 't_max')`), and no `ic`.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

import pinn
from pinn import operators as op

HERE = Path(__file__).parent


class LDCProblem(pinn.Problem):
    """Steady lid-driven cavity at Re = 3200 with a non-regularized lid; outputs (u, v, p)."""

    x_min, x_max = 0.0, 1.0
    Re = 3200.0                                    # [sota patch] theirs is 100
    problem_name = "LidDrivenCavity"
    ref_path = str(HERE / "ldc_reference.npz")

    def __init__(self, *, n_pde, n_bc, n_ic=0):
        self.n_pde, self.n_bc, self.n_ic = n_pde, n_bc, n_ic

    def residual_fns(self):
        return {"pde": self.pde_residual, "bc": self.bc_residual}

    def pde_residual(self, model, coords):
        x, y = coords[0], coords[1]
        u_ = lambda c: model(c)[0]
        v_ = lambda c: model(c)[1]
        p_ = lambda c: model(c)[2]
        u, v = u_(coords), v_(coords)
        inv_Re = 1.0 / self.Re
        res_u = (u * op.grad(u_, coords, 0) + v * op.grad(u_, coords, 1)
                 + op.grad(p_, coords, 0)
                 - inv_Re * (op.grad2(u_, coords, 0) + op.grad2(u_, coords, 1)))
        res_v = (u * op.grad(v_, coords, 0) + v * op.grad(v_, coords, 1)
                 + op.grad(p_, coords, 1)
                 - inv_Re * (op.grad2(v_, coords, 0) + op.grad2(v_, coords, 1)))
        res_div = op.grad(u_, coords, 0) + op.grad(v_, coords, 1)
        return jnp.array([res_u, res_v, 2.0 * res_div])

    def bc_residual(self, model, coords):
        x, y = coords[0], coords[1]
        u, v = model(coords)[0], model(coords)[1]
        # [sota patch] the task's lid is u = 1, not their regularized 4x(1-x)
        u_target = jnp.where(jnp.isclose(y, 1.0), 1.0, 0.0)
        return jnp.array([u - u_target, v, 0.0])

    def samplers(self):
        return {"pde": self._sample_pde, "bc": self._sample_bc}

    def _sample_pde(self, key, n, *_):
        # 7/8 bulk points, 1/16 near the moving lid, 1/16 in the top corners.
        n_lid, n_corner = n // 16, n // 16
        n_bulk = n - n_lid - n_corner
        k_bulk, k_lx, k_ly, k_l, k_r = jax.random.split(key, 5)
        bulk = jax.random.uniform(k_bulk, (n_bulk, 2), minval=0.0, maxval=1.0)
        lid = jnp.stack([
            jax.random.uniform(k_lx, (n_lid,)),
            jax.random.uniform(k_ly, (n_lid,), minval=0.9, maxval=1.0)], axis=1)
        half = n_corner // 2
        s1 = jax.random.uniform(k_l, (half, 2))
        left = jnp.stack([0.1 * s1[:, 0]**2, 1.0 - 0.1 * s1[:, 1]**2], axis=1)
        s2 = jax.random.uniform(k_r, (half, 2))
        right = jnp.stack([1.0 - 0.1 * s2[:, 0]**2, 1.0 - 0.1 * s2[:, 1]**2], axis=1)
        return jnp.concatenate([bulk, lid, left, right], axis=0)

    def _sample_bc(self, key, n, *_):
        n_corner = n // 16
        n_uniform = n - n_corner
        half = n_corner // 2
        k_pt, k_dim, k_face, k_l, k_r = jax.random.split(key, 5)
        pts = jax.random.uniform(k_pt, (n_uniform, 2), minval=0.0, maxval=1.0)
        face_dim = jax.random.randint(k_dim, (n_uniform,), 0, 2)
        face_val = jax.random.randint(k_face, (n_uniform,), 0, 2).astype(pts.dtype)
        pts = pts.at[jnp.arange(n_uniform), face_dim].set(face_val)
        left = jnp.stack([0.1 * jax.random.uniform(k_l, (half,)),
                          jnp.ones((half,))], axis=1)
        right = jnp.stack([1.0 - 0.1 * jax.random.uniform(k_r, (half,)),
                           jnp.ones((half,))], axis=1)
        return jnp.concatenate([pts, left, right], axis=0)


def make_config(seed: int) -> "pinn.RunConfig":
    """The ldc notebook's cell 3 verbatim, plus the seed and headless logging."""
    return pinn.RunConfig(
        seed=seed,
        network=lambda key: pinn.SIREN(
            key, LDCProblem, time_dependent=False,
            hidden_dims=(60, 60, 60, 60), periodic_bc=False,
            n_inputs=2, n_outputs=3,
        ),
        n_pde=2**15, n_bc=2**14,
        residual_sketch=4000, parameter_sketch=4000,
        batch_size=2**12, probe_batch_size=2**8,
        pde_weight=1e-2,
        log_every=10**9, plot_every=10**9,     # headless; not part of the method
    )
