"""DSGNAR on steady 3D flow over a finite wing.

The structural donor is `examples/ldc.ipynb` -- their own steady Navier-Stokes
with (u, v, p), from which the residual layout and the `2.0 *` weighting on the
divergence come. Everything that is the method -- `SIREN`, `operators`, the
`RunConfig`, the sketched trust-region solver -- is imported unchanged.
NATIVE = False: DSGNAR publishes no 3D flow and nothing around a body.

The equation is the task's, their cavity's three lines widened to four:

    u u_x + v u_y + w u_z + p_x - nu lap(u) = 0     (and y, z alike)
    u_x + v_y + w_z = 0

with `nu = 1e-3` (Re = 1000). Their cavity writes `1/Re` with Re = 100.

**The geometry is not analytic here**, unlike every other ported case in this
comparison. The wing is given as a signed-distance level set on a shipped
cloud (`collocation.npy`, `phi <= 0` in the fluid, and it is *already* filtered
to the fluid) plus a separate surface file (`wing_surface.npy`). So the
samplers do not generate points: they draw rows from those two files, which is
what `task/baseline.py` does as well. Their cavity's `_sample_pde` clustering
has no analogue -- there is no lid and no corner singularity -- so the interior
sampler is a plain uniform draw over the shipped fluid points.

Five conditions, from the task's table:

  * `pde`  -- the interior fluid points (faces excluded).
  * `bc`   -- the farfield: inlet `x = -1.5`, lateral `y = +-2`, top `z = 3.5`,
              all carrying `u = 1, v = w = 0`.
  * `bc2`  -- the wing surface, no-slip.
  * `bc3`  -- the outlet `x = 3.5`, carrying `p = 0` and nothing else.
  * `bc4`  -- the symmetry plane `z = 0`: `w = 0` and `du/dz = dv/dz = 0`.
              This is the one condition here that is not a plain value match,
              and it is why the fourth key is used.

`bc2_weight`, `bc3_weight` and `bc4_weight` are stock `RunConfig` fields, so
none of this needs a change to the method. Their defaults (1e-2, 1e-4, 1e-6)
would bury the wing, which is the whole problem, so all four boundary weights
are set to 1.0 -- a deliberate choice, recorded here. No published notebook
uses `bc2`/`bc3`/`bc4` at all (every example has at most one `bc` key), so
these defaults are an unexercised hook rather than a published value being
overridden.

`n_bc` is `2**13` where the ldc donor uses `2**14`, and that is deliberate too.
`n_bc2`, `n_bc3` and `n_bc4` each default to `n_bc`, so the count is paid four
times over. The donor has one boundary set and runs 2**14 boundary against
2**15 interior points, a 1:2 ratio. Four sets at 2**14 would be 65536 against
32768, i.e. 2:1 -- twice as many boundary points as interior ones. Four sets at
2**13 give 32768 against 32768, i.e. 1:1. Neither reproduces the donor's 1:2
exactly; 2**13 is the closer of the two, and that is why it is used.

Steady: no `t_max` attribute, which is how `trainer64` detects a steady problem
(`is_steady = not hasattr(problem, 't_max')`), and no `ic`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

import pinn
from pinn import operators as op

X, Y, Z = 0, 1, 2
HERE = Path(__file__).parent
TASK = Path("~/wing3d/task")

XL, XU = -1.5, 3.5
YL, YU = -2.0, 2.0
ZL, ZU = 0.0, 3.5
NU = 1e-3
U_INF = 1.0
TOL = 1e-5


def _sets():
    """The five fixed point sets, sliced out of the two shipped files exactly
    as `task/baseline.py:_build_dataset` slices them."""
    col = np.load(TASK / "collocation.npy")            # x, y, z, phi
    xyz = np.ascontiguousarray(col[:, :3], dtype=np.float32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    on_inlet = np.abs(x - XL) < TOL
    on_out = np.abs(x - XU) < TOL
    on_lat = (np.abs(y - YL) < TOL) | (np.abs(y - YU) < TOL)
    on_top = np.abs(z - ZU) < TOL
    on_sym = np.abs(z - ZL) < TOL
    on_face = on_inlet | on_out | on_lat | on_top | on_sym
    wing = np.load(TASK / "wing_surface.npy").astype(np.float32)
    return (jnp.asarray(xyz[~on_face]),
            jnp.asarray(xyz[(on_inlet | on_lat | on_top) & ~on_out]),
            jnp.asarray(wing),
            jnp.asarray(xyz[on_out]),
            jnp.asarray(xyz[on_sym]))


_INT, _FAR, _WING, _OUT, _SYM = _sets()


def _draw(pool):
    def sample(key, n, *_):
        return pool[jax.random.choice(key, pool.shape[0], (n,))]
    return sample


class Wing3DProblem(pinn.Problem):
    """Steady 3D Navier-Stokes over a finite wing at Re = 1000; outputs (u,v,w,p)."""

    x_min, x_max = (XL, YL, ZL), (XU, YU, ZU)    # per-axis bounds
    nu = NU
    problem_name = "Wing3D"
    ref_path = str(HERE / "wing3d_reference.npz")

    def __init__(self, *, n_pde, n_bc, n_bc2=None, n_bc3=None, n_bc4=None, n_ic=0):
        self.n_pde, self.n_bc, self.n_ic = n_pde, n_bc, n_ic
        self.n_bc2 = n_bc if n_bc2 is None else n_bc2
        self.n_bc3 = n_bc if n_bc3 is None else n_bc3
        self.n_bc4 = n_bc if n_bc4 is None else n_bc4

    def residual_fns(self):
        return {"pde": self.pde_residual, "bc": self.far_residual,
                "bc2": self.wing_residual, "bc3": self.out_residual,
                "bc4": self.sym_residual}

    def pde_residual(self, model, coords):
        u_ = lambda c: model(c)[0]
        v_ = lambda c: model(c)[1]
        w_ = lambda c: model(c)[2]
        p_ = lambda c: model(c)[3]
        u, v, w = u_(coords), v_(coords), w_(coords)
        sp = (X, Y, Z)

        def mom(f_, axis):
            return (u * op.grad(f_, coords, X) + v * op.grad(f_, coords, Y)
                    + w * op.grad(f_, coords, Z)
                    + op.grad(p_, coords, axis)
                    - self.nu * op.laplacian(f_, coords, sp))

        res_div = (op.grad(u_, coords, X) + op.grad(v_, coords, Y)
                   + op.grad(w_, coords, Z))
        return jnp.array([mom(u_, X), mom(v_, Y), mom(w_, Z),
                          2.0 * res_div])          # their own 2.0

    def far_residual(self, model, coords):
        """Freestream on the inlet, the lateral faces and the top."""
        pred = model(coords)
        return jnp.array([pred[0] - U_INF, pred[1], pred[2], 0.0])

    def wing_residual(self, model, coords):
        """No-slip on the wing surface."""
        pred = model(coords)
        return jnp.array([pred[0], pred[1], pred[2], 0.0])

    def out_residual(self, model, coords):
        """The outlet carries p = 0 and nothing else."""
        return jnp.array([0.0, 0.0, 0.0, model(coords)[3]])

    def sym_residual(self, model, coords):
        """Symmetry plane z = 0: w = 0 and du/dz = dv/dz = 0."""
        u_ = lambda c: model(c)[0]
        v_ = lambda c: model(c)[1]
        return jnp.array([op.grad(u_, coords, Z), op.grad(v_, coords, Z),
                          model(coords)[2], 0.0])

    def samplers(self):
        return {"pde": _draw(_INT), "bc": _draw(_FAR), "bc2": _draw(_WING),
                "bc3": _draw(_OUT), "bc4": _draw(_SYM)}


def make_config(seed: int) -> "pinn.RunConfig":
    """The ldc notebook's cell 3, widened to three inputs and four outputs.

    The four boundary weights are set to 1.0 rather than kept at their stock
    1e-2 / 1e-4 / 1e-6 defaults: those defaults exist to rank a primary
    condition above secondary ones, and here the wing's no-slip surface would
    land at 1e-2 while the outlet and the symmetry plane fell to 1e-4 and 1e-6.
    Every one of these is a first-class condition of the task.
    """
    return pinn.RunConfig(
        seed=seed,
        network=lambda key: pinn.SIREN(
            key, Wing3DProblem, time_dependent=False,
            hidden_dims=(60, 60, 60, 60), periodic_bc=False,
            n_inputs=3, n_outputs=4,
        ),
        n_pde=2**15, n_bc=2**13,
        residual_sketch=4000, parameter_sketch=4000,
        batch_size=2**12, probe_batch_size=2**8,
        pde_weight=1e-2,
        bc_weight=1.0, bc2_weight=1.0, bc3_weight=1.0, bc4_weight=1.0,
        log_every=10**9, plot_every=10**9,     # headless; not part of the method
    )
