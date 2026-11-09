"""DSGNAR on the 3D screened Poisson problem in a cube minus two spheres.

`examples/poisson5d.ipynb` cell 2 is `Poisson5DProblem`: a steady Laplacian on
`[0, 1]^d` with a `bc` residual and `sampling.box` / `sampling.boundary_faces`.
This file is that class retargeted, and every change is marked `[sota patch]`:

  * **d = 5 -> 3**, and the equation gains the screening term. Theirs is
    `-lap(u) - f`; this task's is

        -(u_xx + u_yy + u_zz) + K(z)^2 u - f(x, y, z) = 0

    with `K = 8` for `z < 0.5` and `K = 10` above, and `f` the two-part source
    from the frozen header. Both are taken from `task/baseline.py` and
    reproduced there term for term (`screening`, `source`).

  * **Dirichlet -> homogeneous Neumann on every boundary.** Theirs matches a
    known value on the faces; this task has `du/dn = 0` on all eight
    boundaries — six cube faces and two sphere surfaces — so the residual is
    the normal derivative itself. The normal is not carried alongside the
    point (their sampler contract passes coordinates only), so it is recovered
    from the coordinate: on a cube face exactly one coordinate sits on 0 or 1,
    and on a sphere the outward-of-the-domain normal is radially *inward*.

  * **two boundary conditions instead of one.** `bc` is the cube faces and
    `bc2` is the sphere surfaces; the trainer already keys weights and counts
    by name (`n = getattr(problem, f"n_{name}", problem.n_pde)`, and
    `bc2_weight` is a stock `RunConfig` field), so this needs no method change.
    They are split because a single mixed set cannot be sampled uniformly over
    two shapes of very different area.

  * **the spheres are cut out of the interior sampler.** Their `box` fills the
    whole cube; here points landing inside either sphere are pushed out along
    the radius, which keeps the count exact and the sampler jittable
    (rejection sampling is not, under a fixed `n`).

Everything that is the method — `GaborNet`, `operators`, the sketched
trust-region solver, the `RunConfig` of their Poisson notebook — is imported
unchanged. NATIVE = False: the screened operator, the geometry and the Neumann
data are not theirs, only the solver and the Poisson skeleton are.

**Why pure Neumann is well posed here.** With `K^2 u` present the constant is
not in the kernel, so the solution is unique and no mean constraint is needed.
Drop the screening term and this would be a floating problem — worth stating
because the trivial anchors do *not* bracket it: the reference has a nonzero
mean, so `u == 0` scores 1.0 and the best constant still scores 0.929.

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

# task/baseline.py frozen header, verbatim
SPHERES = ((0.4, 0.3, 0.6, 0.2), (0.6, 0.7, 0.6, 0.2))
Z_IFACE = 0.5
K_LO, K_HI = 8.0, 10.0
SRC_A1, SRC_A2 = 20.0, 100.0
SRC_M1, SRC_M2, SRC_M3 = 1.0, 10.0, 5.0


def screening(z):
    """K(z)^2 — task/baseline.py:screening, verbatim."""
    return jnp.where(z < Z_IFACE, K_LO ** 2, K_HI ** 2)


def source(x, y, z):
    """f(x, y, z) — task/baseline.py:source, verbatim. Independent of u."""
    r2 = x ** 2 + y ** 2 + z ** 2
    part1 = (jnp.exp(jnp.sin(SRC_M1 * jnp.pi * x)
                     + jnp.sin(SRC_M2 * jnp.pi * y)
                     + jnp.sin(SRC_M3 * jnp.pi * z))
             * (r2 - 1.0) / (r2 + 1.0))
    part2 = (jnp.sin(SRC_M1 * jnp.pi * x)
             * jnp.sin(SRC_M2 * jnp.pi * y)
             * jnp.sin(SRC_M3 * jnp.pi * z))
    return SRC_A1 * part1 + SRC_A2 * part2


class PoissonCGProblem(pinn.Problem):
    """Steady screened Poisson on the unit cube minus two spheres; output u."""

    x_min, x_max = 0.0, 1.0
    problem_name = "PoissonCG"
    ref_path = str(HERE / "poissoncg_reference.npz")

    def __init__(self, *, n_pde, n_bc, n_bc2=None, n_ic=0):
        self.n_pde, self.n_bc, self.n_ic = n_pde, n_bc, n_ic
        # [sota patch] the trainer sizes each condition by `n_<name>`
        self.n_bc2 = n_bc if n_bc2 is None else n_bc2

    def residual_fns(self):
        # [sota patch] their {"pde", "bc"} gains the sphere surfaces as "bc2"
        return {"pde": self.pde_residual,
                "bc": self.bc_face_residual,
                "bc2": self.bc_sphere_residual}

    # ── equation ───────────────────────────────────────────────────────
    def pde_residual(self, model, coords):
        """[sota patch] theirs is -lap(u) - f; this one screens with K(z)^2."""
        u_ = lambda c: model(c)[0]
        x, y, z = coords[0], coords[1], coords[2]
        r = (-op.laplacian(u_, coords)
             + screening(z) * u_(coords)
             - source(x, y, z))
        return jnp.array([r])

    # ── boundaries: du/dn = 0 everywhere ───────────────────────────────
    def bc_face_residual(self, model, coords):
        """[sota patch] Neumann on the six cube faces.

        `boundary_faces` pins exactly one coordinate to 0 or 1, so the outward
        normal is `+e_i` at the high face and `-e_i` at the low one. Only the
        squared value matters to the loss, but the sign is written out anyway
        so the expression reads as the physical normal derivative.
        """
        u_ = lambda c: model(c)[0]
        at_hi = (coords >= self.x_max).astype(coords.dtype)
        at_lo = (coords <= self.x_min).astype(coords.dtype)
        nrm = at_hi - at_lo                       # (3,), one entry +-1
        du = jnp.array([op.grad(u_, coords, i) for i in range(3)])
        return jnp.array([jnp.dot(du, nrm)])

    def bc_sphere_residual(self, model, coords):
        """[sota patch] Neumann on the two sphere surfaces.

        The normal that points out of the *fluid* domain points into the
        removed sphere — `task/baseline.py` says so where it builds `nrm_sph`.
        The point is assigned to whichever sphere it lies on.
        """
        u_ = lambda c: model(c)[0]
        cs = jnp.array([[cx, cy, cz] for cx, cy, cz, _ in SPHERES])
        rs = jnp.array([r for *_, r in SPHERES])
        d = jnp.linalg.norm(coords[None, :] - cs, axis=1) - rs
        k = jnp.argmin(jnp.abs(d))
        nrm = -(coords - cs[k]) / rs[k]           # inward: out of the domain
        du = jnp.array([op.grad(u_, coords, i) for i in range(3)])
        return jnp.array([jnp.dot(du, nrm)])

    # ── samplers ───────────────────────────────────────────────────────
    def samplers(self):
        return {"pde": self._sample_pde,
                "bc": self._sample_faces,
                "bc2": self._sample_spheres}

    def _sample_pde(self, key, n, *_):
        """[sota patch] their `sampling.box`, with the two spheres cut out.

        A point inside a sphere is pushed to just outside it along its own
        radius. That keeps `n` fixed — which rejection sampling cannot do
        under jit — and leaves the bulk untouched, since only the ~3.3% of the
        cube the spheres occupy moves at all.
        """
        pts = jax.random.uniform(key, (n, 3), minval=self.x_min, maxval=self.x_max)
        for cx, cy, cz, r in SPHERES:
            c = jnp.array([cx, cy, cz])
            v = pts - c
            d = jnp.linalg.norm(v, axis=1, keepdims=True)
            inside = d < r
            safe = jnp.where(d > 1e-12, d, 1e-12)
            pushed = c + v / safe * r * 1.0001
            pts = jnp.where(inside, pushed, pts)
        return pts

    def _sample_faces(self, key, n, *_):
        """Their `sampling.boundary_faces(0, 1, 3)`, inlined so the sphere
        cut-out can share this file. Unchanged in behaviour."""
        k_pt, k_dim, k_face = jax.random.split(key, 3)
        x = jax.random.uniform(k_pt, (n, 3), minval=self.x_min, maxval=self.x_max)
        face_dim = jax.random.randint(k_dim, (n,), minval=0, maxval=3)
        faces = jnp.array([self.x_min, self.x_max])
        face_val = faces[jax.random.randint(k_face, (n,), minval=0, maxval=2)]
        return x.at[jnp.arange(n), face_dim].set(face_val)

    def _sample_spheres(self, key, n, *_):
        """[sota patch] uniform points on the two sphere surfaces.

        Gaussian directions normalised to the unit sphere — the standard
        area-uniform draw. `task/baseline.py` uses a fixed Fibonacci lattice
        instead; random points are used here because every other sampler in
        this method is redrawn each step and a frozen set would be the odd one
        out. Both cover the surface uniformly.
        """
        k_dir, k_which = jax.random.split(key)
        v = jax.random.normal(k_dir, (n, 3))
        v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
        cs = jnp.array([[cx, cy, cz] for cx, cy, cz, _ in SPHERES])
        rs = jnp.array([r for *_, r in SPHERES])
        j = jax.random.randint(k_which, (n,), minval=0, maxval=len(SPHERES))
        return cs[j] + v * rs[j][:, None]


def make_config(seed: int) -> "pinn.RunConfig":
    """The poisson5d notebook's cell 3 verbatim, plus the seed, the third
    condition's weight and headless logging. `n_inputs` drops 5 -> 3."""
    return pinn.RunConfig(
        seed=seed,
        network=lambda key: pinn.GaborNet(
            key, PoissonCGProblem, n_gabor=2 ** 10,
            periodic_bc=False, n_inputs=3, n_outputs=1,
        ),
        n_pde=2 ** 15, n_bc=2 ** 14,
        residual_sketch=4000, parameter_sketch=4000,
        batch_size=2 ** 12, probe_batch_size=2 ** 8,
        pde_weight=1e-4, bc_weight=1.0,
        bc2_weight=1.0,                 # [sota patch] the spheres are a
                                        # boundary like any other, not a
                                        # secondary condition
        window_scale=6.0, n_probes=64,
        log_every=10 ** 9, plot_every=10 ** 9,   # headless; not the method
    )
