"""[sota patch] the Kolmogorov-flow example's utils.py, retargeted to this task.

`get_dataset` loaded their `data/kolmogorov_flow_Re10000.npy`. This task ships
`task/ns4d_ic.csv` (a 32^3 grid of x, y, z, u, v, w, wx, wy, wz at t = 0) and
`task/ref_data.csv` (the 20^4 scoring grid). Only the initial condition feeds
training; the reference feeds the evaluator's `l2_error` logging, which in a
benchmarked run fires once, at step 0, and never again.

Their loader also sliced a `time_range` out of a stored trajectory. There is no
stored trajectory here -- the window is the whole `[0, 5]` the task specifies --
so the slicing is gone and `t_star` is built from the reference's own 20 time
levels.
"""
import numpy as np

import jax.numpy as jnp

TASK = "~/ns4d/task"
NU = 0.05                      # problem.md; their loader read it from the .npy


def get_dataset():
    """(ic_coords, ic_uvw, t_star, ref_coords, ref_w, nu).

    ic_coords : (32^3, 3) as (x, y, z), the task's own initial-condition grid
    ic_uvw    : (32^3, 3) the velocity there
    t_star    : (20,) the reference's time levels, [0, 5]
    ref_coords: (20^3, 3) the spatial part of the scoring grid
    ref_w     : (20, 20^3, 3) the scored vorticity, indexed [time, point, comp]
    """
    ic = np.loadtxt(f"{TASK}/ns4d_ic.csv", delimiter=",", skiprows=1, dtype=np.float64)
    ic_coords = jnp.asarray(ic[:, 0:3], dtype=jnp.float32)
    ic_uvw = jnp.asarray(ic[:, 3:6], dtype=jnp.float32)

    ref = np.loadtxt(f"{TASK}/ref_data.csv", delimiter=",", skiprows=1, dtype=np.float64)
    t = ref[:, 0]
    t_star = np.unique(t)
    # the reference is a full 20^4 tensor grid written t-major
    n_sp = len(ref) // len(t_star)
    order = np.lexsort((ref[:, 3], ref[:, 2], ref[:, 1], ref[:, 0]))
    r = ref[order]
    ref_coords = jnp.asarray(r[:n_sp, 1:4], dtype=jnp.float32)
    ref_w = jnp.asarray(r[:, 7:10].reshape(len(t_star), n_sp, 3), dtype=jnp.float32)

    return ic_coords, ic_uvw, jnp.asarray(t_star, dtype=jnp.float32), \
        ref_coords, ref_w, NU


def forcing(t, x, y, z):
    """[sota patch] problem.md's f, in place of their `2 sin(4 pi y)` body force.

    Identical to SPINN's `navier_stokes4d_forcing_term`, which is where the
    task's own reference comes from.
    """
    e = jnp.exp(-18.0 * NU * t)
    return (-6.0 * e * jnp.sin(4.0 * y) * jnp.sin(2.0 * z),
            -6.0 * e * jnp.sin(4.0 * x) * jnp.sin(2.0 * z),
            6.0 * e * jnp.sin(4.0 * x) * jnp.sin(4.0 * y))
