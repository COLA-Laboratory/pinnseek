"""Dataset loader for the multiscale wave task — replaces the 1D wave's `get_dataset`.

The donor example builds an analytic 1D standing wave. This task is 2D with a
medium read from file, so the loader is rewritten rather than patched:

  * `x_star`, `y_star` — the reference's own 129-point axes, used for the two
    initial-condition terms and the four Neumann boundary terms;
  * `u0` — u(0,x,y) = exp(-((x+0.5)^2 + y^2) / (2*0.3^2)) on that lattice,
    the task's analytic initial condition (the reference's t = 0 slice matches
    it to 5e-11);
  * `c_grid` — the task's own 256x256 wave-speed field, returned with its axes
    so the residual can interpolate bilinearly as `problem.md` specifies;
  * `u_ref`, `t_star` — for the evaluator's progress readout only.
"""
import numpy as np
import jax.numpy as jnp

TASK = "~/wavems/task"
SIGMA = 0.3


def get_dataset():
    ref = np.loadtxt(f"{TASK}/ref_data.csv", delimiter=",", skiprows=1, dtype=np.float64)
    t, x, y, u = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]
    t_star, x_star, y_star = np.unique(t), np.unique(x), np.unique(y)
    order = np.lexsort((y, x, t))                       # t-major, then x, then y
    u_ref = u[order].reshape(len(t_star), len(x_star), len(y_star))
    assert np.allclose(t[order].reshape(u_ref.shape), t_star[:, None, None])
    assert np.allclose(y[order].reshape(u_ref.shape), y_star[None, None, :])

    xx, yy = np.meshgrid(x_star, y_star, indexing="ij")
    u0 = np.exp(-((xx + 0.5) ** 2 + yy ** 2) / (2.0 * SIGMA ** 2))

    coef = np.loadtxt(f"{TASK}/wavems_coef.csv", delimiter=",", skiprows=1, dtype=np.float64)
    cx, cy = np.unique(coef[:, 0]), np.unique(coef[:, 1])
    corder = np.lexsort((coef[:, 1], coef[:, 0]))
    c_grid = coef[corder, 2].reshape(len(cx), len(cy))
    assert np.allclose(coef[corder, 0].reshape(c_grid.shape), cx[:, None])

    return (jnp.array(u_ref), jnp.array(t_star), jnp.array(x_star), jnp.array(y_star),
            jnp.array(u0), jnp.array(cx), jnp.array(cy), jnp.array(c_grid))
