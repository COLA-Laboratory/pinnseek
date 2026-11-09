"""[sota patch] the LDC example's utils.py, retargeted to this geometry.

`get_dataset` loaded their cavity .mat; this task ships no training data at all
— geometry, source and every boundary are analytic — so the only thing that
needs loading is the reference, and that feeds the evaluator's `l2_error`
logging alone (`train.py` passes it to `PoissonEvaluator`, never to a loss).
In a benchmarked run that logging fires once, at step 0, and never again.

`sample_points_on_square_boundary` walked the four sides of the unit square.
This domain has eight boundaries — the six cube faces and two sphere surfaces —
so it is replaced by one function per shape, each returning points *and* the
outward normal, which the Neumann condition needs and a Dirichlet one does not.

The normals follow `task/baseline.py:_build_dataset`: axis-aligned and outward
on the faces, and radially *inward* on the spheres, because the normal that
leaves the fluid domain points into the removed ball.
"""
import numpy as np

import jax
import jax.numpy as jnp

REF = "~/poissoncg/task/ref_data.csv"

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


def get_dataset():
    """(coords[N,3], u_ref[N]) — the task's reference, for logging only."""
    csv = np.loadtxt(REF, delimiter=",", skiprows=1, dtype=np.float64)
    return jnp.asarray(csv[:, 0:3], dtype=jnp.float32), \
        jnp.asarray(csv[:, 3], dtype=jnp.float32)


def sample_points_on_cube_faces(num_pts_per_face):
    """[sota patch] the six faces of the unit cube, with outward normals.

    Their square version laid points out on a `linspace` per side; this keeps
    that deterministic-grid character, using a `n x n` lattice per face.
    No `eps` shrink: this task has no corner singularity to back away from —
    the boundary data is zero everywhere, and continuous.
    """
    a = jnp.linspace(0.0, 1.0, num_pts_per_face)
    A, B = jnp.meshgrid(a, a, indexing="ij")
    A, B = A.reshape(-1), B.reshape(-1)
    one, zero = jnp.ones_like(A), jnp.zeros_like(A)
    pts, nrm = [], []
    for axis in range(3):
        for value, sign in ((0.0, -1.0), (1.0, 1.0)):
            cols = [A, B]
            p = [None, None, None]
            p[axis] = jnp.full_like(A, value)
            for k in (i for i in range(3) if i != axis):
                p[k] = cols.pop(0)
            pts.append(jnp.stack(p, axis=1))
            n = [zero, zero, zero]
            n[axis] = sign * one
            nrm.append(jnp.stack(n, axis=1))
    return jnp.concatenate(pts), jnp.concatenate(nrm)


def sample_points_on_spheres(num_pts_per_sphere):
    """[sota patch] both sphere surfaces, with the normal that leaves the domain.

    `task/baseline.py` uses a Fibonacci lattice for exactly this, and it is
    reproduced here rather than replaced by a random draw, so the two sides
    place their sphere points identically.
    """
    # in float64 through numpy, as baseline.py does: theta runs to ~1.7e4 rad
    # and cos/sin of that in float32 drifts by ~5e-4, which would put these
    # points on a measurably different lattice from the task's own.
    idx = np.arange(num_pts_per_sphere) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / num_pts_per_sphere)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
    unit = np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)
    pts = np.concatenate([np.array([cx, cy, cz]) + r * unit
                          for cx, cy, cz, r in SPHERES]).astype(np.float32)
    nrm = np.concatenate([-unit for _ in SPHERES]).astype(np.float32)
    return jnp.asarray(pts), jnp.asarray(nrm)


def outside_spheres(pts):
    """[sota patch] mask of the batch points that are in the domain.

    Their cavity was the whole square, so nothing like this was needed. The
    residual sampler here is `jaxpi`'s stock `UniformSampler` over the cube;
    the 6.7% of each batch that lands inside a sphere (the two balls are
    2 * (4/3) pi 0.2^3 = 0.0670 of the unit cube) is masked out of the residual
    mean rather than resampled, which keeps the batch shape static.
    """
    keep = jnp.ones(pts.shape[0], dtype=pts.dtype)
    for cx, cy, cz, r in SPHERES:
        d2 = ((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
              + (pts[:, 2] - cz) ** 2)
        keep = keep * (d2 > r ** 2)
    return keep
