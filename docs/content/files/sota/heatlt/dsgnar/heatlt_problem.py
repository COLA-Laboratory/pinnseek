"""DSGNAR on the long-time nonlinear heat equation — their method on a problem they do not ship.

DSGNAR publishes no heat equation and nothing with a solution-dependent source.
The closest example structurally is `examples/laplacian.ipynb`: Dirichlet
boundaries through their `sampling.boundary_faces`, `pde_weight` far below
`bc_weight`, and a multiscale right-hand side. That is where the boundary
handling and the `RunConfig` shape come from; the time axis follows the 2D+time
pattern the gs, gl, kf, wavems and smoke cases here already use (`D = 2` plus
per-case samplers). NATIVE = False.

Point by point:

  * **Equation** — the task's, written directly:
        u_t - D (u_xx + u_yy) - A sin(u^2) (1 + 2 sin(pi t / 4)) sin(4 pi x) sin(2 pi y) = 0
    with D = 0.001 and A = 5 from `problem.md`. Everything is analytic; there is
    no table to interpolate.
  * **Two spatial dimensions** — `D = 2`, the hook their `trainer64` reads.
  * **Initial condition** — `sin(4 pi x) sin(3 pi y)`, analytic, so
    `analytical_ic` is a formula rather than a lookup. (The task also ships it
    as `heat_ic.csv`; the file reproduces the formula.)
  * **Boundary** — u = 0 on all four sides, the same Dirichlet trace their
    laplacian example returns.
  * **A long window.** t runs to 100, which is 12.5 periods of the forcing
    (period 8), so `dt` is the whole window and the trainer solves it as one
    slab; their laplacian donor is steady and offers no slab count to copy.

**The trap: u == 0 satisfies the PDE and all four sides exactly**, because the
source carries `sin(u^2)` and vanishes with u. Their `pde_weight = 1e-8`
against `bc_weight = 1.0` leans on the boundary and initial terms, which is the
right side of that trade and is left at their value.

`ref_path` points at `heatlt_reference.npz`, built by `make_reference.py` from
the task's own reference. It feeds reporting only.
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

D_COEF = 0.001
A_COEF = 5.0


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


class HeatLTProblem(pinn.Problem):
    """u_t - D lap(u) - A sin(u^2) (1+2 sin(pi t/4)) sin(4 pi x) sin(2 pi y) = 0."""

    D = 2                                    # spatial dimensions
    x_min, x_max, t_max = 0.0, 1.0, 100.0
    problem_name = "HeatLongTime"
    ref_path = str(HERE / "heatlt_reference.npz")

    def __init__(self, *, n_pde, n_ic, n_bc):
        self.n_pde, self.n_ic, self.n_bc = n_pde, n_ic, n_bc

    def residual_fns(self):
        return {"pde": self.pde_residual, "ic": self.ic_residual,
                "bc": self.bc_residual}

    def pde_residual(self, model, coords):
        u_ = lambda c: model(c)[0]
        u = u_(coords)
        src = (A_COEF * jnp.sin(u ** 2)
               * (1.0 + 2.0 * jnp.sin(jnp.pi * coords[T] / 4.0))
               * jnp.sin(4.0 * jnp.pi * coords[X])
               * jnp.sin(2.0 * jnp.pi * coords[Y]))
        r = (op.grad(u_, coords, T)
             - D_COEF * (op.grad2(u_, coords, X) + op.grad2(u_, coords, Y))
             - src)
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
        """u(0, x, y) = sin(4 pi x) sin(3 pi y)."""
        return (jnp.sin(4.0 * jnp.pi * xy[X])
                * jnp.sin(3.0 * jnp.pi * xy[Y]))[None, :]


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
        dt=100.0,
        network=lambda key: pinn.SIREN(
            key, HeatLTProblem, n_fourier=4,
            hidden_dims=(60, 60, 60, 60), periodic_bc=False,
            n_inputs=3, n_outputs=1,
        ),
        n_pde=2**15, n_ic=2**14, n_bc=2**14,
        residual_sketch=4000, parameter_sketch=4000,
        batch_size=2**14, probe_batch_size=2**10,
        pde_weight=1e-8, bc_weight=1.0,
        log_every=10**9, plot_every=10**9,     # headless; not part of the method
    )
