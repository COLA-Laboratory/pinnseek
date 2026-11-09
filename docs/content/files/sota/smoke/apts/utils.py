"""Dataset loader for the wildfire-smoke task — replaces the 1D advection loader.

The donor example builds an analytic 1D travelling wave with a constant speed.
This task is 2D with ERA5/GFAS coefficient fields read from disk, so the loader
is rewritten rather than patched.

**The coefficient fields are stored `(t, y, x)` while the coordinates are
`(t, x, y)`**, and the task's own `baseline.py` says so in as many words:

    "the fields are stored (t, y, x), so the coordinate order below is not (t,x,y)"

`coef_at` here is `task/baseline.py:coef_at` reproduced exactly — same
`map_coordinates`, same `order=1` (trilinear, as `problem.md` specifies), same
`mode="nearest"`, and the same `(t, y, x)` coordinate order. The fields are left
in their stored layout rather than transposed, so there is one fewer place to
get the order wrong, and `../verify_coef.py` checks the result against the
task's function point by point.
"""
import numpy as np
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

TASK = "~/smoke/task"
GRID_NT, GRID_NY, GRID_NX = 73, 153, 241


def coef_at(field, t, x, y):
    """task/baseline.py:coef_at, reproduced exactly. Fields are (t, y, x)."""
    coords = jnp.stack([t * (GRID_NT - 1), y * (GRID_NY - 1), x * (GRID_NX - 1)])
    return map_coordinates(field, coords, order=1, mode="nearest")


def get_dataset():
    z = np.load(f"{TASK}/smoke_coef.npz")
    A, B, C, F = (jnp.asarray(z[k], dtype=jnp.float32) for k in ("a", "b", "c", "f"))
    assert A.shape == (GRID_NT, GRID_NY, GRID_NX), A.shape
    dxc, dyc = float(z["dxc"]), float(z["dyc"])

    ref = np.loadtxt(f"{TASK}/ref_data.csv", delimiter=",", skiprows=1, dtype=np.float64)
    t_star, x_star, y_star = (np.unique(ref[:, i]) for i in range(3))
    order = np.lexsort((ref[:, 2], ref[:, 1], ref[:, 0]))   # t, then x, then y
    u_ref = ref[order, 3].reshape(len(t_star), len(x_star), len(y_star))
    assert np.allclose(ref[order, 0].reshape(u_ref.shape), t_star[:, None, None])
    assert np.allclose(ref[order, 2].reshape(u_ref.shape), y_star[None, None, :])

    return dict(A=A, B=B, C=C, F=F, dxc=dxc, dyc=dyc,
                t_star=jnp.array(t_star), x_star=jnp.array(x_star),
                y_star=jnp.array(y_star), u_ref=jnp.array(u_ref))
