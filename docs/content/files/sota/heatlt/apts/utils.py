"""Dataset for the long-time nonlinear heat task — replaces the 1D advection loader.

Everything this problem needs is analytic, so nothing is read from disk except
the reference used for the evaluator's progress readout:

  * the source is `A sin(u^2) (1 + 2 sin(pi t / 4)) sin(4 pi x) sin(2 pi y)`
    with D = 0.001 and A = 5, from `problem.md`;
  * the initial condition is `sin(4 pi x) sin(3 pi y)`, which the task also
    ships as `heat_ic.csv` on a 128 x 128 grid — the file reproduces the
    formula, and the formula is what is used;
  * the boundary is u = 0 on all four sides.

The reference is the COMSOL mesh: 501 snapshots on 16 x 12 = 192 nodes, so the
distinct spatial coordinates are few and the grid is small.
"""
import numpy as np
import jax.numpy as jnp

TASK = "~/heatlt/task"
D_COEF = 0.001
A_COEF = 5.0
T_MAX = 100.0


def get_dataset():
    ref = np.loadtxt(f"{TASK}/ref_data.csv", delimiter=",", skiprows=1, dtype=np.float64)
    t_star, x_star, y_star = (np.unique(ref[:, i]) for i in range(3))
    return dict(t_star=jnp.array(t_star), x_star=jnp.array(x_star),
                y_star=jnp.array(y_star),
                coords=jnp.array(ref[:, 0:3]), u_ref=jnp.array(ref[:, 3]))
