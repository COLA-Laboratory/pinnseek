"""DSGNAR on wildfire-smoke transport — their method on a problem they do not ship.

DSGNAR publishes nothing with advection or a variable coefficient. The closest
example structurally is `examples/laplacian.ipynb`: Dirichlet boundaries with
their `sampling.boundary_faces`, `pde_weight` far below `bc_weight`, and a
multiscale right-hand side. That is where the boundary handling and the
`RunConfig` shape come from; the time axis follows the 2D+time pattern the gs,
gl, kf and wavems cases here already use (`D = 2` plus the per-case samplers).
NATIVE = False.

Point by point:

  * **Equation** — the task's, written directly:
        u_t + a u_x + b u_y + c u - DXC u_xx - DYC u_yy - f = 0
    with a, b, c, f the ERA5/GFAS fields and DXC/DYC the frozen constants from
    `problem.md`.
  * **Coefficient lookup** — `task/baseline.py:coef_at` reproduced exactly:
    `map_coordinates` with `order=1` (trilinear, as `problem.md` specifies) and
    the **(t, y, x)** storage order the task warns about, while the coordinates
    are (t, x, y). `cases/smoke/methods/verify_coef.py` checks the same
    construction against the task's own function on 4096 random points: max
    |diff| 0.000, against up to 57 for the transposed order. The tables are kept
    as numpy and converted at call time, because the trainer flips
    `jax_enable_x64` between passes and a frozen jnp constant breaks the other
    one (GUIDE.md, dsgnar section).
  * **Two spatial dimensions** — `D = 2`, the hook their `trainer64` reads
    (`n_spatial = problem.D if hasattr(problem, 'D') else 1`).
  * **Initial condition** — u(0, x, y) = 0 everywhere. The atmosphere starts
    clean and the source f builds the plume, so `analytical_ic` returns zeros.
  * **Boundary** — u = 0 on all four sides, the same Dirichlet trace their
    laplacian example returns, sampled on the four faces over the slab's time
    window.

`ref_path` points at `smoke_reference.npz`, built by `make_reference.py` from
the task's own reference. It feeds reporting only — see that script.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

import pinn
from pinn import operators as op

X, Y, T = 0, 1, 2
HERE = Path(__file__).parent
TASK = "~/smoke/task"
GRID_NT, GRID_NY, GRID_NX = 73, 153, 241

_z = np.load(f"{TASK}/smoke_coef.npz")
_A, _B, _C, _F = (np.asarray(_z[k], dtype=np.float64) for k in ("a", "b", "c", "f"))
DXC, DYC = float(_z["dxc"]), float(_z["dyc"])
assert _A.shape == (GRID_NT, GRID_NY, GRID_NX), _A.shape


def _coef_at(table, t, x, y):
    """task/baseline.py:coef_at — trilinear, fields stored (t, y, x)."""
    dt = t.dtype
    fld = jnp.asarray(table, dt)
    coords = jnp.stack([t * (GRID_NT - 1), y * (GRID_NY - 1), x * (GRID_NX - 1)])
    return map_coordinates(fld, coords, order=1, mode="nearest")


def _interior_2d(x_min, x_max):
    """Uniform points over the square x time. Returns [x, y, t]."""
    def sample(key, n, t0, t1):
        kx, ky, kt = jax.random.split(key, 3)
        x = jax.random.uniform(kx, (n,), minval=x_min, maxval=x_max)
        y = jax.random.uniform(ky, (n,), minval=x_min, maxval=x_max)
        t = jax.random.uniform(kt, (n,), minval=t0, maxval=t1)
        return jnp.stack([x, y, t], axis=1)
    return sample


def _initial_plane(x_min, x_max):
    """Evenly spaced points on the square at the slab start. Returns [x, y, t0]."""
    def sample(key, n, t0):
        m = int(jnp.floor(jnp.sqrt(n)))
        g = jnp.linspace(x_min, x_max, m, endpoint=False)
        gx, gy = jnp.meshgrid(g, g, indexing="ij")
        return jnp.stack([gx.reshape(-1), gy.reshape(-1),
                          jnp.full((m * m,), t0)], axis=1)
    return sample


def _dirichlet_faces(x_min, x_max):
    """The four sides over the slab's time window — `sampling.boundary_faces`
    one dimension wider, which is what their laplacian example uses in 2D."""
    def sample(key, n, t0, t1):
        q = n // 4
        ks, kt = jax.random.split(key)
        s = jax.random.uniform(ks, (4 * q,), minval=x_min, maxval=x_max)
        t = jax.random.uniform(kt, (4 * q,), minval=t0, maxval=t1)
        face = jnp.repeat(jnp.arange(4), q)
        x = jnp.where(face == 0, x_min, jnp.where(face == 1, x_max, s))
        y = jnp.where(face == 2, x_min, jnp.where(face == 3, x_max, s))
        return jnp.stack([x, y, t], axis=1)
    return sample


class SmokeProblem(pinn.Problem):
    """u_t + a u_x + b u_y + c u - DXC u_xx - DYC u_yy - f = 0 on [0,1]^3."""

    D = 2                                    # spatial dimensions
    x_min, x_max, t_max = 0.0, 1.0, 1.0
    problem_name = "WildfireSmoke"
    ref_path = str(HERE / "smoke_reference.npz")

    def __init__(self, *, n_pde, n_ic, n_bc):
        self.n_pde, self.n_ic, self.n_bc = n_pde, n_ic, n_bc

    def residual_fns(self):
        return {"pde": self.pde_residual, "ic": self.ic_residual,
                "bc": self.bc_residual}

    def pde_residual(self, model, coords):
        u_ = lambda c: model(c)[0]
        u = u_(coords)
        a = _coef_at(_A, coords[T], coords[X], coords[Y])
        b = _coef_at(_B, coords[T], coords[X], coords[Y])
        c = _coef_at(_C, coords[T], coords[X], coords[Y])
        f = _coef_at(_F, coords[T], coords[X], coords[Y])
        r = (op.grad(u_, coords, T)
             + a * op.grad(u_, coords, X) + b * op.grad(u_, coords, Y)
             + c * u
             - DXC * op.grad2(u_, coords, X) - DYC * op.grad2(u_, coords, Y)
             - f)
        return jnp.array([r])

    def ic_residual(self, model, coords):
        # coords = [x, y, t0, u_target]; the task's u(0,.) = 0
        return jnp.array([model(coords[:3])[0] - coords[3]])

    def bc_residual(self, model, coords):
        # u = 0 on all four sides — the Dirichlet trace, as in their laplacian
        return jnp.array([model(coords)[0]])

    def samplers(self):
        return {"pde": _interior_2d(self.x_min, self.x_max),
                "ic":  _initial_plane(self.x_min, self.x_max),
                "bc":  _dirichlet_faces(self.x_min, self.x_max)}

    def analytical_ic(self, xy):
        """u(0, x, y) = 0 — the atmosphere starts clean."""
        return jnp.zeros_like(xy[X])[None, :]


def make_config(seed: int) -> "pinn.RunConfig":
    """The laplacian notebook's cell 3, plus the seed, a time axis and headless logging.

    `n_inputs` is 3 rather than 2 and the network is time-dependent; `dt` is the
    whole window, i.e. one slab, because the task's time span is already
    normalised to [0, 1] and their laplacian donor is steady so it offers no
    slab count to copy. The loss weights are theirs verbatim
    (`pde_weight=1e-8`, `bc_weight=1.0`): this problem is driven entirely by the
    source term through the residual, with u = 0 on the boundary and at t = 0,
    so leaning on the boundary and initial terms is the right side of that
    trade — exactly as it is for their Laplacian.
    """
    return pinn.RunConfig(
        seed=seed,
        dt=1.0,
        network=lambda key: pinn.SIREN(
            key, SmokeProblem, n_fourier=4,
            hidden_dims=(60, 60, 60, 60), periodic_bc=False,
            n_inputs=3, n_outputs=1,
        ),
        n_pde=2**15, n_ic=2**14, n_bc=2**14,
        residual_sketch=4000, parameter_sketch=4000,
        batch_size=2**14, probe_batch_size=2**10,
        pde_weight=1e-8, bc_weight=1.0,
        log_every=10**9, plot_every=10**9,     # headless; not part of the method
    )
