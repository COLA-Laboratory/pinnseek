"""[sota patch] the backward-facing-step example's utils.py, retargeted to this task.

`get_dataset` read their `data/flow.vtu` with pyvista and built the inflow,
outflow and wall coordinate sets by `linspace`. This task ships its geometry
instead: `task/collocation.npy` is `(x, y, z, phi)` -- already filtered to the
fluid, every point has `phi <= 0` -- and `task/wing_surface.npy` is the wing's
no-slip surface. The five point sets are sliced out of those two files exactly
as `task/baseline.py:_build_dataset` slices them, so both sides see the same
geometry.

`inflow_profile` was a parabola; here the inlet, the lateral faces and the top
all carry the same freestream `u = 1, v = w = 0`, so there is no profile to
evaluate and the function is gone.

The reference is loaded for the evaluator's `l2_error` logging only, and
logging fires once, at step 0, and never again.
"""
import numpy as np

import jax.numpy as jnp

TASK = "~/wing3d/task"

RE = 1000.0
NU = 1.0 / RE
U_INF = 1.0
XL, XU = -1.5, 3.5
YL, YU = -2.0, 2.0
ZL, ZU = 0.0, 3.5
TOL = 1e-5


def get_dataset():
    """(coords, far_coords, wing_coords, out_coords, sym_coords,
        ref_coords, ref_V, nu).

    coords     : the interior fluid points -- the residual set
    far_coords : inlet x=-1.5, lateral y=+-2, top z=3.5, all freestream
    wing_coords: the wing's no-slip surface
    out_coords : the outlet x=3.5, p = 0
    sym_coords : the symmetry plane z=0, w = 0 and du/dz = dv/dz = 0
    ref_*      : the scoring points and the velocity magnitude there, for
                 logging only
    """
    col = np.load(f"{TASK}/collocation.npy")           # x, y, z, phi
    xyz = np.ascontiguousarray(col[:, :3], dtype=np.float32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    on_inlet = np.abs(x - XL) < TOL
    on_out = np.abs(x - XU) < TOL
    on_lat = (np.abs(y - YL) < TOL) | (np.abs(y - YU) < TOL)
    on_top = np.abs(z - ZU) < TOL
    on_sym = np.abs(z - ZL) < TOL
    on_face = on_inlet | on_out | on_lat | on_top | on_sym
    wing = np.load(f"{TASK}/wing_surface.npy").astype(np.float32)

    ref = np.load(f"{TASK}/ref_data.npy")              # x,y,z,u,v,w,p,phi
    ref_V = np.sqrt(ref[:, 3] ** 2 + ref[:, 4] ** 2 + ref[:, 5] ** 2)

    return (jnp.asarray(xyz[~on_face]),
            jnp.asarray(xyz[(on_inlet | on_lat | on_top) & ~on_out]),
            jnp.asarray(wing),
            jnp.asarray(xyz[on_out]),
            jnp.asarray(xyz[on_sym]),
            jnp.asarray(ref[:, 0:3], dtype=jnp.float32),
            jnp.asarray(ref_V, dtype=jnp.float32),
            NU)
