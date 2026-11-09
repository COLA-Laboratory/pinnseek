"""DSGNAR on Grey-Scott — the method's own components, a problem it does not ship.

DSGNAR publishes burgers / kdv / ks / laplacian / ldc / poisson5d / poisson10d /
wave; there is no Grey-Scott example, so this problem class is written here.
Everything that is the *method* is imported unchanged from
`reference/DSGNAR/src/pinn`: the `Problem` interface, `operators`, the periodic
`MLP`, the `RunConfig`, and the sketched trust-region solver in `trainer64.py`
(via the step-capped copy `trainer_capped.py`, unchanged from the kdv case).

What is written here, and why it looks the way it does:

  * **Coordinates.** DSGNAR puts time last: this problem is (x, y, t), and
    `D = 2` tells `trainer64._build_conditions` how many leading columns are
    spatial. `D` is a hook their trainer already reads
    (`problem.D if hasattr(problem, 'D') else 1`); no shipped example is
    2D-in-space *and* time-dependent, so this is the first use of it.
  * **Equations** — the task's, with the same six constants Scale-PINN
    hard-codes and APTS reads out of `gray_scott.mat`.
  * **Initial condition** — the task's analytic pair, which is its
    ref_data.csv first frame to 3e-7. `analytical_ic` takes (2, n) and returns
    (2, n), the shape the trainer's IC branch expects for a two-output problem.
  * **Boundary** — periodic in both x and y, enforced *by construction*:
    `MLP(..., periodic_bc=True, n_inputs=3)` embeds the two spatial
    coordinates as cos/sin of 2*pi*(x - x_min)/(x_max - x_min), i.e. exactly
    period 2 on [-1, 1] (`networks._embed`). So there is no `bc` residual —
    the same reason their own periodic examples (kdv, ks) have none.
  * **Samplers** — `sampling.py` only ships 1D-interval-times-time and steady
    box/face samplers, so the 2D+time pair below is written in its idiom:
    uniform interior points over the square crossed with the slab's time
    window, and an even spatial lattice at the slab start.

The `RunConfig` is the **ks** notebook's (cell 2), that being their closest
published problem: periodic, time-dependent, tanh `MLP`. Only what the problem's
shape forces is different — `n_inputs=3`, `n_outputs=2` — plus the slab list.

Slabs: their two periodic time-dependent examples disagree on slabbing (kdv
uses a uniform `dt`, ks a list clustered towards its final time "where the
dynamics are sharpest"). Grey-Scott's error is flat in time rather than
concentrated late, so the uniform choice is the right analogue, and `dt` is set
to give kdv's four uniform slabs over this window: 0.78 / 4 = 0.195.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

import pinn
from pinn import operators as op

X, Y, T = 0, 1, 2

TASK_T_MAX = 0.78
N_SLABS = 4                      # kdv's uniform slab count
HERE = Path(__file__).parent


def _interior_2d(x_min, x_max):
    """Uniform points over the square x time. Returns [x, y, t].
    sampling.interior's idiom, one spatial dimension wider."""
    def sample(key, n, t0, t1):
        kx, ky, kt = jax.random.split(key, 3)
        x = jax.random.uniform(kx, (n,), minval=x_min, maxval=x_max)
        y = jax.random.uniform(ky, (n,), minval=x_min, maxval=x_max)
        t = jax.random.uniform(kt, (n,), minval=t0, maxval=t1)
        return jnp.stack([x, y, t], axis=1)
    return sample


def _initial_plane(x_min, x_max):
    """Evenly spaced points on the square at the slab start. Returns [x, y, t0].
    sampling.initial_line's idiom, on a lattice instead of a line."""
    def sample(key, n, t0):
        m = int(jnp.floor(jnp.sqrt(n)))
        g = jnp.linspace(x_min, x_max, m, endpoint=False)
        gx, gy = jnp.meshgrid(g, g, indexing="ij")
        x, y = gx.reshape(-1), gy.reshape(-1)
        t = jnp.full((m * m,), t0)
        return jnp.stack([x, y, t], axis=1)
    return sample


class GrayScottProblem(pinn.Problem):
    """u_t - e1 lap u - b1(1-u) + c1 u v^2 = 0,
       v_t - e2 lap v + b2 v - c2 u v^2 = 0, periodic on [-1,1]^2."""

    D = 2                                    # spatial dimensions
    x_min, x_max, t_max = -1.0, 1.0, TASK_T_MAX
    eps1, eps2 = 0.2, 0.1
    b1, b2 = 40.0, 100.0
    c1, c2 = 1000.0, 1000.0
    problem_name = "GreyScott"
    ref_path = str(HERE / "gs_reference.npz")

    def __init__(self, *, n_pde, n_ic, n_bc=0):
        self.n_pde, self.n_ic, self.n_bc = n_pde, n_ic, n_bc

    def residual_fns(self):
        return {"pde": self.pde_residual, "ic": self.ic_residual}

    def pde_residual(self, model, coords):
        u_ = lambda c: model(c)[0]
        v_ = lambda c: model(c)[1]
        u, v = u_(coords), v_(coords)
        lap_u = op.grad2(u_, coords, X) + op.grad2(u_, coords, Y)
        lap_v = op.grad2(v_, coords, X) + op.grad2(v_, coords, Y)
        ru = (op.grad(u_, coords, T) - self.eps1 * lap_u
              - self.b1 * (1.0 - u) + self.c1 * u * v ** 2)
        rv = (op.grad(v_, coords, T) - self.eps2 * lap_v
              + self.b2 * v - self.c2 * u * v ** 2)
        return jnp.array([ru, rv])

    def ic_residual(self, model, coords):
        # coords = [x, y, t0, u_target, v_target]
        pred = model(coords[:3])
        return jnp.array([pred[0] - coords[3], pred[1] - coords[4]])

    def samplers(self):
        return {"pde": _interior_2d(self.x_min, self.x_max),
                "ic":  _initial_plane(self.x_min, self.x_max)}

    def analytical_ic(self, xy):
        """xy : (2, n) -> (2, n). problem.md's u0, v0."""
        x, y = xy[0], xy[1]
        u0 = 1.0 - jnp.exp(-10.0 * ((x + 0.05) ** 2 + (y + 0.02) ** 2))
        v0 = jnp.exp(-10.0 * ((x - 0.05) ** 2 + (y - 0.02) ** 2))
        return jnp.stack([u0, v0])


def make_config(seed: int) -> "pinn.RunConfig":
    """The ks notebook's cell 2, with n_inputs/n_outputs set by this problem's
    shape and kdv's uniform slabbing, plus the seed and headless logging."""
    return pinn.RunConfig(
        seed=seed,
        dt=TASK_T_MAX / N_SLABS,
        network=lambda key: pinn.MLP(
            key, GrayScottProblem, hidden_dims=(70, 70, 70, 70),
            periodic_bc=True, n_inputs=3, n_outputs=2,
        ),
        n_pde=2**15, n_ic=2**14,
        residual_sketch=5000, parameter_sketch=5000,
        batch_size=2**12, probe_batch_size=2**8,
        pde_weight=1e-3,
        # display-only: the notebooks log/plot every 5 steps; headless under a
        # wall-clock budget we log only at pass boundaries
        log_every=10**9, plot_every=10**9,
    )
