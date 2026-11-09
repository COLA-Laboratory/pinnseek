"""DSGNAR on the NACA0012 airfoil at Re = 1000 — their cavity recipe on an immersed body.

`examples/ldc.ipynb` cell 2 is `LidDrivenCavityProblem`: steady 2D
Navier-Stokes with (u, v, p) on a box, driven entirely by prescribed wall
velocities and with no pressure condition. That is the same equation and the
same solver regime as this task; what differs is the geometry — the fluid
region here is the farfield box minus an airfoil — and the boundary values.
So the *method* is theirs unchanged (the `Problem` interface, `operators`, the
`SIREN`, the `RunConfig`, the sketched trust-region solver via the step-capped
`trainer_capped.py`), and what is written here is the geometry and the
boundary conditions. NATIVE = False.

Point by point:

  * **Equations** — copied from their `LidDrivenCavityProblem.pde_residual`,
    including their `2.0 *` weighting on the divergence residual. Only the
    viscosity changes: they write `1/Re` with Re = 100, this task has
    Re = 1000.
  * **Domain** — a rectangle, not a square, so `x_min`/`x_max` are given as
    per-axis tuples. `networks._domain_bounds` already reads vector bounds
    (`if hasattr(val, "__len__")`); no shipped example uses that path.
  * **Boundary values** — the task's table rather than a lid: freestream
    u = 1, v = 0 on the outer box, and no-slip on the airfoil surface. Their
    cavity residual returns `[u - u_target, v, 0.0]`; the same three slots are
    used here, the third staying zero because this task, like their cavity,
    prescribes no pressure.
  * **The immersed body is the part their samplers cannot express.** Their
    `_sample_pde` draws uniformly in the box and clusters at the lid and its
    corners, which is where *their* solution is singular. Here the interior
    sampler draws from the task's own fluid point set (`phi <= 0` in
    `collocation.csv`) so that no residual point lands inside the airfoil, and
    clusters an eighth of its draws near the surface, which is the analogue of
    their lid clustering. The boundary sampler splits between the outer box
    and the 400 analytic airfoil points, weighting the airfoil the way theirs
    weights the lid.
  * **Airfoil geometry** — `task/baseline.py:_naca_airfoil_points` reproduced
    exactly: cosine-clustered NACA0012 at 12% thickness rotated -7 degrees
    about (0.5, 0). Checked against the task's own function, max |diff| 3e-8
    (its float32 cast).
  * **Steady** — no `t_max` attribute, which is how `trainer64` detects a
    steady problem (`is_steady = not hasattr(problem, 't_max')`), and no `ic`.

`RunConfig` is the ldc notebook's cell 3 verbatim apart from the seed and
headless logging.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import pinn
from pinn import operators as op

HERE = Path(__file__).parent
TASK = "~/naca/task"

XL, XU = -3.0, 5.0
YL, YU = -2.0, 2.0
U_INF = 1.0
AOA_DEG = 7.0


def _airfoil_points(n=400, thickness=0.12):
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


def _fluid_points():
    col = np.loadtxt(f"{TASK}/collocation.csv", delimiter=",", skiprows=1, dtype=np.float64)
    xy, phi = col[:, 0:2], col[:, 2]
    edge = ((xy[:, 0] == XL) | (xy[:, 0] == XU) | (xy[:, 1] == YL) | (xy[:, 1] == YU))
    return xy[(phi <= 0.0) & ~edge]


# module level, as numpy: the trainer flips jax_enable_x64 between passes, so a
# frozen jnp constant would break the second one (see GUIDE.md, dsgnar section)
_AIR = _airfoil_points(400)
_FLUID = _fluid_points()
_NEAR = _FLUID[(_FLUID[:, 0] > -0.3) & (_FLUID[:, 0] < 1.6)
               & (_FLUID[:, 1] > -0.4) & (_FLUID[:, 1] < 0.4)]


class NACAProblem(pinn.Problem):
    """Steady flow over a NACA0012 at Re = 1000; outputs (u, v, p)."""

    x_min, x_max = (XL, YL), (XU, YU)      # per-axis bounds
    Re = 1000.0
    problem_name = "NACA0012"
    ref_path = str(HERE / "naca_reference.npz")

    def __init__(self, *, n_pde, n_bc, n_ic=0):
        self.n_pde, self.n_bc, self.n_ic = n_pde, n_bc, n_ic

    def residual_fns(self):
        return {"pde": self.pde_residual, "bc": self.bc_residual}

    def pde_residual(self, model, coords):
        u_ = lambda c: model(c)[0]
        v_ = lambda c: model(c)[1]
        p_ = lambda c: model(c)[2]
        u, v = u_(coords), v_(coords)
        inv_Re = 1.0 / self.Re
        res_u = (u * op.grad(u_, coords, 0) + v * op.grad(u_, coords, 1)
                 + op.grad(p_, coords, 0)
                 - inv_Re * (op.grad2(u_, coords, 0) + op.grad2(u_, coords, 1)))
        res_v = (u * op.grad(v_, coords, 0) + v * op.grad(v_, coords, 1)
                 + op.grad(p_, coords, 1)
                 - inv_Re * (op.grad2(v_, coords, 0) + op.grad2(v_, coords, 1)))
        res_div = op.grad(u_, coords, 0) + op.grad(v_, coords, 1)
        return jnp.array([res_u, res_v, 2.0 * res_div])

    def bc_residual(self, model, coords):
        """Freestream on the outer box, no-slip on the airfoil.

        A point is on the outer box when either coordinate sits on a domain
        edge; everything else in this set is an airfoil surface point.
        """
        x, y = coords[0], coords[1]
        u, v = model(coords)[0], model(coords)[1]
        on_box = (jnp.isclose(x, XL) | jnp.isclose(x, XU)
                  | jnp.isclose(y, YL) | jnp.isclose(y, YU))
        u_target = jnp.where(on_box, U_INF, 0.0)
        return jnp.array([u - u_target, v, 0.0])

    def samplers(self):
        return {"pde": self._sample_pde, "bc": self._sample_bc}

    def _sample_pde(self, key, n, *_):
        """7/8 from the task's fluid set, 1/8 clustered near the airfoil.

        Their cavity draws uniformly in the box and puts an eighth near the lid
        and its corners. The box is not the fluid region here, so the draws come
        from the task's own `phi <= 0` set; the clustered eighth goes to the
        near-airfoil window, which is where this solution's structure is.
        """
        dt = jnp.zeros(()).dtype
        pool = jnp.asarray(_FLUID, dt)
        near = jnp.asarray(_NEAR, dt)
        n_near = n // 8
        k_b, k_n = jax.random.split(key)
        bulk = pool[jax.random.randint(k_b, (n - n_near,), 0, pool.shape[0])]
        cl = near[jax.random.randint(k_n, (n_near,), 0, near.shape[0])]
        return jnp.concatenate([bulk, cl], axis=0)

    def _sample_bc(self, key, n, *_):
        """Half the outer box, half the airfoil surface.

        Their cavity draws uniformly over the four faces and adds an extra
        sixteenth on the lid, the face carrying the non-trivial value. Here the
        airfoil is that face: it is the condition the solution turns on, and it
        is 400 analytic points rather than a continuum, so it takes half the
        budget and the outer box the other half.
        """
        dt = jnp.zeros(()).dtype
        air = jnp.asarray(_AIR, dt)
        n_air = n // 2
        n_box = n - n_air
        k_pt, k_dim, k_face, k_a = jax.random.split(key, 4)
        lo = jnp.asarray([XL, YL], dt)
        hi = jnp.asarray([XU, YU], dt)
        pts = jax.random.uniform(k_pt, (n_box, 2), dtype=dt, minval=lo, maxval=hi)
        face_dim = jax.random.randint(k_dim, (n_box,), 0, 2)
        face_hi = jax.random.randint(k_face, (n_box,), 0, 2)
        face_val = jnp.where(face_dim == 0,
                             jnp.where(face_hi == 1, hi[0], lo[0]),
                             jnp.where(face_hi == 1, hi[1], lo[1]))
        pts = pts.at[jnp.arange(n_box), face_dim].set(face_val)
        surf = air[jax.random.randint(k_a, (n_air,), 0, air.shape[0])]
        return jnp.concatenate([pts, surf], axis=0)


def make_config(seed: int) -> "pinn.RunConfig":
    """The ldc notebook's cell 3 verbatim, plus the seed and headless logging."""
    return pinn.RunConfig(
        seed=seed,
        network=lambda key: pinn.SIREN(
            key, NACAProblem, time_dependent=False,
            hidden_dims=(60, 60, 60, 60), periodic_bc=False,
            n_inputs=2, n_outputs=3,
        ),
        n_pde=2**15, n_bc=2**14,
        residual_sketch=4000, parameter_sketch=4000,
        batch_size=2**12, probe_batch_size=2**8,
        pde_weight=1e-2,
        log_every=10**9, plot_every=10**9,     # headless; not part of the method
    )
