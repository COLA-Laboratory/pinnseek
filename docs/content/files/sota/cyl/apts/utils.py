"""[sota patch] the Kolmogorov-flow example's utils.py, retargeted to this task.

`get_dataset` loaded their `data/kolmogorov_flow_Re10000.npy`. This task ships
`task/cyl_ic.csv` (the velocity on a 441 x 83 grid at t = 0) and
`task/ref_data.csv` (the scoring snapshots). Only the initial condition feeds
training; the reference feeds the evaluator's `l2_error` logging, which in a
benchmarked run fires once, at step 0, and never again.

Their loader also sliced a `time_range` out of a stored trajectory. There is no
stored trajectory here -- the window is the whole `[0, 1.2]` the task
specifies -- so the slicing is gone.

The boundary sets are new. Their Kolmogorov flow is periodic in both directions
and needs none; this task has a driven inlet, two no-slip walls, a pressure
outlet and a no-slip cylinder. The three velocity sets and the outlet set are
built here, following `examples/bfs_flow/utils.py`, which is the repository's
own channel-with-inlet-and-outlet case.
"""
import numpy as np

import jax.numpy as jnp

TASK = "~/cyl/task"

NU = 1e-3                    # problem.md
H = 0.41
XL, XU = 0.0, 2.2
YL, YU = 0.0, 0.41
T_MAX = 1.2
CX, CY, CR = 0.2, 0.2, 0.05


def mean_inflow(t):
    """U_bar(t) = sin(pi (t + 4.8) / 8) -- problem.md's driven inflow."""
    return jnp.sin(jnp.pi * (t + 4.8) / 8.0)


def inflow_profile(t, y):
    """u at x = 0. v = 0 there. This is bfs_flow's `inflow_profile` with the
    task's parabola and the time factor its steady channel does not have."""
    return 6.0 * mean_inflow(t) * y * (H - y) / H ** 2


def get_dataset():
    """(ic_coords, ic_uv, ref_coords, ref_uv, nu).

    ic_coords : (N_ic, 2) as (x, y), the task's own t = 0 grid, cylinder removed
    ic_uv     : (N_ic, 2) the velocity there
    ref_coords: (N_ref, 3) as (t, x, y), the scoring points
    ref_uv    : (N_ref, 2) the scored velocity
    """
    ic = np.loadtxt(f"{TASK}/cyl_ic.csv", delimiter=",", skiprows=1, dtype=np.float64)
    ref = np.loadtxt(f"{TASK}/ref_data.csv", delimiter=",", skiprows=1, dtype=np.float64)
    return (jnp.asarray(ic[:, 0:2], dtype=jnp.float32),
            jnp.asarray(ic[:, 2:4], dtype=jnp.float32),
            jnp.asarray(ref[:, 0:3], dtype=jnp.float32),
            jnp.asarray(ref[:, 3:5], dtype=jnp.float32),
            NU)


def sample_boundary_sets(n_side, n_cyl):
    """[sota patch] the four channel sides and the cylinder, as (x, y) pairs.

    Deterministic lattices, the way `bfs_flow/utils.py` builds its inflow and
    wall coordinates with `linspace`. Time is drawn separately by the model, so
    these carry space only.
    """
    s = jnp.linspace(0.0, 1.0, n_side)
    ys = YL + (YU - YL) * s
    xs = XL + (XU - XL) * s
    inlet = jnp.stack([jnp.full_like(ys, XL), ys], axis=1)
    outlet = jnp.stack([jnp.full_like(ys, XU), ys], axis=1)
    bot = jnp.stack([xs, jnp.full_like(xs, YL)], axis=1)
    top = jnp.stack([xs, jnp.full_like(xs, YU)], axis=1)
    a = jnp.linspace(0.0, 2.0 * jnp.pi, n_cyl, endpoint=False)
    cyl = jnp.stack([CX + CR * jnp.cos(a), CY + CR * jnp.sin(a)], axis=1)
    walls = jnp.concatenate([bot, top, cyl], axis=0)      # all no-slip
    return inlet, outlet, walls


def in_fluid(x, y):
    """[sota patch] the analytic geometry: outside the cylinder. Their
    Kolmogorov box has no hole in it and needs nothing like this."""
    return (((x - CX) ** 2 + (y - CY) ** 2) > CR ** 2).astype(x.dtype)
