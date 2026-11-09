"""DSGNAR on the multiscale wave equation — their 1D wave recipe, in 2D with a variable medium.

`examples/wave.ipynb` cell 2 is `WaveProblem`: `u_tt - c^2 u_xx = 0` on a line
with Dirichlet walls, a constant speed and a standing-wave start. That is the
same *kind* of equation — second order in time, hyperbolic, with both u(0,.)
and u_t(0,.) prescribed — and its four-residual structure
(`pde` / `ic` / `ic_t` / `bc`) is exactly what this task needs. What differs is
everything about the setting: two spatial dimensions instead of one, a
coefficient that varies in space instead of a constant, and Neumann walls
instead of Dirichlet. NATIVE = False.

Point by point:

  * **Equation** — their `op.grad2(u, coords, T) - c^2 * op.grad2(u, coords, X)`
    becomes `u_tt - c(x,y) * (u_xx + u_yy)`. Note the task writes the
    coefficient as `c` multiplying the Laplacian directly (wave speed
    sqrt(c)), so there is no square here; theirs is written with `c^2` and
    c = 2. The residual code follows the task's form.
  * **Two spatial dimensions** — `D = 2`, the hook their `trainer64` already
    reads (`n_spatial = problem.D if hasattr(problem, 'D') else 1`) but no
    shipped example sets. The samplers are the 2D versions used by the gs, gl
    and kf cases here, written in the idiom of their own
    `sampling.interior`/`initial_line`.
  * **Variable coefficient** — `c(x,y)` takes 1 and 12 on the task's 256x256
    grid and `problem.md` specifies bilinear interpolation between grid
    points, so the residual interpolates rather than snapping. The table is
    kept as numpy and converted at call time: the trainer flips
    `jax_enable_x64` between passes, and a module-level jnp constant would be
    frozen at one precision and break the other (GUIDE.md, dsgnar section).
  * **Neumann walls** — their `bc_residual` returns `model(coords)[0]`, the
    Dirichlet trace. Here it returns the outward normal derivative, which is
    zero on all four sides. The sampler puts points on the four faces rather
    than the two endpoints.
  * **Both initial conditions** — kept as their two separate residuals, `ic`
    for u(0,.) and `ic_t` for u_t(0,.) = 0, which is their own decomposition.

**u == 0 satisfies the equation and both Neumann conditions exactly**, so only
the initial condition rules it out. Their `pde_weight` starts the PDE term at
1e-2 relative to the IC at 1.0, which is the right side of that trade; it is
left at the notebook's value and not tuned.

`ref_path` points at `wavems_reference.npz`, built by `make_reference.py` from
the task's own reference. It feeds reporting only — see that script.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import pinn
from pinn import operators as op

X, Y, T = 0, 1, 2
HERE = Path(__file__).parent
TASK = "~/wavems/task"

XL, XU = -1.0, 1.0
T_MAX = 5.0
SIGMA = 0.3


def _medium():
    csv = np.loadtxt(f"{TASK}/wavems_coef.csv", delimiter=",", skiprows=1, dtype=np.float64)
    xs, ys = np.unique(csv[:, 0]), np.unique(csv[:, 1])
    order = np.lexsort((csv[:, 1], csv[:, 0]))
    c = csv[order, 2].reshape(len(xs), len(ys))
    assert np.allclose(csv[order, 0].reshape(len(xs), len(ys)), xs[:, None])
    return xs, ys, c


# numpy at module level; converted to the active precision at call time
_CX, _CY, _CC = _medium()


def _c_at(x, y):
    """Bilinear interpolation of the task's c(x,y) at a single point."""
    dt = x.dtype
    cx, cy, cc = jnp.asarray(_CX, dt), jnp.asarray(_CY, dt), jnp.asarray(_CC, dt)
    fx = jnp.clip((x - cx[0]) / (cx[1] - cx[0]), 0.0, cx.shape[0] - 1.001)
    fy = jnp.clip((y - cy[0]) / (cy[1] - cy[0]), 0.0, cy.shape[0] - 1.001)
    i0, j0 = jnp.floor(fx).astype(jnp.int32), jnp.floor(fy).astype(jnp.int32)
    wx, wy = fx - i0, fy - j0
    return ((1 - wx) * (1 - wy) * cc[i0, j0] + wx * (1 - wy) * cc[i0 + 1, j0]
            + (1 - wx) * wy * cc[i0, j0 + 1] + wx * wy * cc[i0 + 1, j0 + 1])


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
        x, y = gx.reshape(-1), gy.reshape(-1)
        t = jnp.full((m * m,), t0)
        return jnp.stack([x, y, t], axis=1)
    return sample


def _neumann_faces(x_min, x_max):
    """Points on the four sides of the square, over the slab's time window.

    `sampling.dirichlet_walls`' idiom, one dimension wider: a quarter of the
    points on each face. The face index is carried in a fourth column so the
    residual knows which normal derivative to take.
    """
    def sample(key, n, t0, t1):
        q = n // 4
        ks, kt = jax.random.split(key)
        s = jax.random.uniform(ks, (4 * q,), minval=x_min, maxval=x_max)
        t = jax.random.uniform(kt, (4 * q,), minval=t0, maxval=t1)
        face = jnp.repeat(jnp.arange(4), q)
        x = jnp.where(face == 0, x_min, jnp.where(face == 1, x_max, s))
        y = jnp.where(face == 2, x_min, jnp.where(face == 3, x_max, s))
        return jnp.stack([x, y, t, face.astype(s.dtype)], axis=1)
    return sample


class WaveMSProblem(pinn.Problem):
    """u_tt - c(x,y) (u_xx + u_yy) = 0 on [-1,1]^2 x [0,5], zero Neumann."""

    D = 2                                    # spatial dimensions
    x_min, x_max, t_max = XL, XU, T_MAX
    problem_name = "WaveMultiscale"
    ref_path = str(HERE / "wavems_reference.npz")

    def __init__(self, *, n_pde, n_ic, n_bc):
        self.n_pde, self.n_ic, self.n_bc = n_pde, n_ic, n_bc

    def residual_fns(self):
        return {"pde": self.pde_residual, "ic": self.ic_residual,
                "ic_t": self.ic_t_residual, "bc": self.bc_residual}

    def pde_residual(self, model, coords):
        u_ = lambda c: model(c)[0]
        c = _c_at(coords[X], coords[Y])
        r = op.grad2(u_, coords, T) - c * (op.grad2(u_, coords, X)
                                           + op.grad2(u_, coords, Y))
        return jnp.array([r])

    def ic_residual(self, model, coords):
        # coords = [x, y, t0, u_target]
        return jnp.array([model(coords[:3])[0] - coords[3]])

    def ic_t_residual(self, model, coords):
        # coords = [x, y, t0, ut_target]; the task's u_t(0,.) = 0
        u_ = lambda c: model(c)[0]
        return jnp.array([op.grad(u_, coords[:3], T) - coords[3]])

    def bc_residual(self, model, coords):
        """Zero Neumann: u_x on x = +-1, u_y on y = +-1. Face index in coords[3]."""
        u_ = lambda c: model(c)[0]
        face = coords[3]
        du_x = op.grad(u_, coords[:3], X)
        du_y = op.grad(u_, coords[:3], Y)
        return jnp.array([jnp.where(face < 1.5, du_x, du_y)])

    def samplers(self):
        return {"pde": _interior_2d(self.x_min, self.x_max),
                "ic":  _initial_plane(self.x_min, self.x_max),
                "ic_t": _initial_plane(self.x_min, self.x_max),
                "bc":  _neumann_faces(self.x_min, self.x_max)}

    def analytical_ic(self, xy):
        """u(0,x,y) = exp(-((x+0.5)^2 + y^2) / (2 sigma^2))."""
        return jnp.array([jnp.exp(-((xy[X] + 0.5) ** 2 + xy[Y] ** 2)
                                  / (2.0 * SIGMA ** 2))])

    def analytical_ic_t(self, xy):
        """u_t(0,x,y) = 0 — the task's second initial condition.

        Their wave notebook writes exactly `jnp.zeros_like(x)` for the same
        thing; here the shape is (out_dim, n) with out_dim = 1, so the row is
        taken from one spatial coordinate.
        """
        return jnp.zeros_like(xy[X])[None, :]


def make_config(seed: int) -> "pinn.RunConfig":
    """The wave notebook's config, plus the seed, 3 inputs and headless logging.

    `dt` is kept at their **0.5**, not rescaled. Their `t_max` is 1.0, so 0.5 is
    two slabs; this task runs to t = 5, so the same 0.5 is ten. Keeping the
    number (their temporal *resolution*) rather than the count is the same call
    the ks case made, and it is the right one for a hyperbolic problem: a wave
    at speed sqrt(12) crosses this domain several times, and stretching a slab
    to 2.5 would ask one model to carry far more of the solution than they ever
    asked of theirs. Under a 100 s budget the step cap will stop part-way
    through the slab sequence; `info.slabs_solved` records how far it got, and
    points beyond are evaluated by the last solved slab.

    The loss weights are theirs verbatim: `pde_weight=1e-6`, `ic_weight=1.0`,
    `ic_t_weight=1e-3`, `bc_weight=1.0`. That ordering matters here — u == 0
    satisfies the PDE and both Neumann conditions exactly, so a recipe that
    leaned on the PDE term would collapse to it; theirs leans on the IC.
    """
    return pinn.RunConfig(
        seed=seed,
        dt=0.5,
        network=lambda key: pinn.SIREN(
            key, WaveMSProblem, hidden_dims=(60, 60, 60, 60),
            periodic_bc=False, n_inputs=3, n_outputs=1,
        ),
        n_pde=2**15, n_ic=2**14, n_bc=2**14,
        residual_sketch=4000, parameter_sketch=4000,
        batch_size=2**14, probe_batch_size=2**10,
        pde_weight=1e-6, ic_weight=1.0, ic_t_weight=1e-3, bc_weight=1.0,
        log_every=10**9, plot_every=10**9,     # headless; not part of the method
    )
