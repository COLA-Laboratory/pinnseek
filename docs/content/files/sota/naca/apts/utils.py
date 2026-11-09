"""Dataset loader for the NACA0012 task — replaces the cavity's `get_dataset`.

The lid-driven-cavity example this port comes from reads a .mat of a square
solve. This task's geometry is an airfoil immersed in a box, and it is supplied
by the task package, so the loader is rewritten here rather than patched:

  * `pts_int` — the fluid collocation points, taken as `phi <= 0` from the
    task's own `collocation.csv` with the outer edges removed. That is the
    Scale-PINN convention and it is what `task/baseline.py:_build_dataset`
    itself does; sampling a box uniformly would put residual points inside the
    airfoil, which has no meaning here.
  * `pts_bc`, `u_bc` — the boundary set: the outer box (freestream u = 1,
    v = 0) followed by 400 analytic NACA0012 surface points (no-slip). Every
    v target is zero, so the cavity's `v_bc_loss = mean(v_pred**2)` carries
    over unchanged.
  * the airfoil points are `task/baseline.py:_naca_airfoil_points` reproduced
    exactly: cosine-clustered NACA0012 at 12% thickness, rotated -7 degrees
    about (0.5, 0).

`u_ref`/`v_ref` are returned for the evaluator's progress readout only.
"""
import numpy as np
import jax.numpy as jnp

TASK = "~/naca/task"
X_MIN, X_MAX, Y_MIN, Y_MAX = -3.0, 5.0, -2.0, 2.0
U_INF = 1.0
AOA_DEG = 7.0


def naca_airfoil_points(n=400, thickness=0.12):
    """task/baseline.py:_naca_airfoil_points, reproduced exactly."""
    beta = np.linspace(0.0, np.pi, n // 2)
    xc = 0.5 * (1.0 - np.cos(beta))
    yt = 5 * thickness * (0.2969 * np.sqrt(xc) - 0.1260 * xc
                          - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)
    xa = np.concatenate([xc, xc])
    ya = np.concatenate([yt, -yt])
    a = np.radians(-AOA_DEG)
    xr = 0.5 + (xa - 0.5) * np.cos(a) - ya * np.sin(a)
    yr = 0.0 + (xa - 0.5) * np.sin(a) + ya * np.cos(a)
    return np.stack([xr, yr], axis=1)


def get_dataset():
    col = np.loadtxt(f"{TASK}/collocation.csv", delimiter=",", skiprows=1, dtype=np.float64)
    xy, phi = col[:, 0:2], col[:, 2]
    edge = ((xy[:, 0] == X_MIN) | (xy[:, 0] == X_MAX)
            | (xy[:, 1] == Y_MIN) | (xy[:, 1] == Y_MAX))
    fluid = phi <= 0.0

    pts_int = xy[fluid & ~edge]
    pts_far = xy[edge]
    pts_air = naca_airfoil_points(400)

    pts_bc = np.vstack([pts_far, pts_air])
    u_bc = np.concatenate([np.full(len(pts_far), U_INF), np.zeros(len(pts_air))])

    # the reference is a structured 801 x 401 grid, so it reshapes onto its own
    # axes; this feeds the evaluator's progress readout only (see run.py)
    ref = np.loadtxt(f"{TASK}/ref_data.csv", delimiter=",", skiprows=1, dtype=np.float64)
    x_star, y_star = np.unique(ref[:, 0]), np.unique(ref[:, 1])
    order = np.lexsort((ref[:, 1], ref[:, 0]))
    V = np.sqrt(ref[order, 2] ** 2 + ref[order, 3] ** 2).reshape(len(x_star), len(y_star))
    assert np.allclose(ref[order, 0].reshape(len(x_star), len(y_star)), x_star[:, None])
    assert np.allclose(ref[order, 1].reshape(len(x_star), len(y_star)), y_star[None, :])

    return (jnp.array(pts_int), jnp.array(pts_bc), jnp.array(u_bc),
            jnp.array(x_star), jnp.array(y_star), jnp.array(V))
