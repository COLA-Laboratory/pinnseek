"""DSGNAR on Kolmogorov flow — the method's own components, a problem it does not ship.

DSGNAR publishes burgers / kdv / ks / laplacian / ldc / poisson5d / poisson10d /
wave. `ldc` is 2D Navier-Stokes but steady; there is no unsteady flow example,
so this problem class is written here. Everything that is the *method* is
imported unchanged from `reference/DSGNAR/src/pinn`: the `Problem` interface,
`operators`, the periodic `MLP`, the `RunConfig`, and the sketched trust-region
solver in `trainer64.py` (via the step-capped copy `trainer_capped.py`,
unchanged from the kdv case).

What is written here, and why:

  * **Coordinates** — DSGNAR puts time last: (x, y, t), with `D = 2` telling
    `trainer64._build_conditions` how many leading columns are spatial.
  * **Equations** — the task's three, written the way their `ldc` example
    writes the steady pair (`op.grad` per component, `2.0 *` on the continuity
    residual is *their* weighting there and is kept), plus the unsteady terms
    and the Kolmogorov body force 2 sin(4 pi y) on the x-momentum.
    nu = 5e-4 (Re = 2000), the task's value.
  * **Initial condition** — the task's `kf_ic.csv` velocity field, not an
    analytic profile. `_initial_plane` samples exactly the reference grid's
    [::2] subsample (the 128x128 lattice the metric uses), so
    `analytical_ic` is an exact table lookup rather than an interpolation.
    Only u and v are prescribed: the pressure of an incompressible flow is
    fixed only up to a constant, the task does not give it at t = 0 and does
    not score it, so `ic_residual` leaves the third output free.
  * **Boundary** — periodic in both x and y, by construction:
    `MLP(..., periodic_bc=True, n_inputs=3)` embeds the two spatial
    coordinates as cos/sin of 2*pi*(x - x_min)/(x_max - x_min), i.e. exactly
    period 1 on [0, 1] (`networks._embed`). Hence no `bc` residual, the same
    as their own periodic examples.
  * **Samplers** — `sampling.py` ships no 2D+time sampler, so the pair below
    is written in its idiom.

The `RunConfig` is the **ks** notebook's (cell 2) — their closest published
problem that is periodic and time-dependent — with `n_inputs=3`,
`n_outputs=3` forced by this problem's shape, and kdv's uniform slabbing at
four slabs over the window: 0.75 / 4 = 0.1875.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import pinn
from pinn import operators as op

X, Y, T = 0, 1, 2

TASK_T_MAX = 0.75
N_SLABS = 4
HERE = Path(__file__).parent
IC_CSV = Path("~/kf/task/kf_ic.csv")


def _load_ic():
    """The task's t=0 velocity on the metric's own 128x128 subsample."""
    csv = np.loadtxt(IC_CSV, delimiter=",", skiprows=1, dtype=np.float64)
    xs = np.unique(csv[:, 0]); ys = np.unique(csv[:, 1])
    nx, ny = len(xs), len(ys)
    u = csv[:, 2].reshape(nx, ny)[::2, ::2]      # rows are y-fastest
    v = csv[:, 3].reshape(nx, ny)[::2, ::2]
    # kept as numpy, not jnp: the trainer runs a float32 pass and a float64
    # pass and flips `jax_enable_x64` between them, so a jnp array built at
    # import time would be frozen at one precision and the other pass would
    # fail to lower ("tensor<f64>" vs "tensor<f32>"). Converted at call time.
    return xs[::2], ys[::2], u, v


_XS, _YS, _U0, _V0 = _load_ic()


def _interior_2d(x_min, x_max):
    """Uniform points over the square x time. Returns [x, y, t]."""
    def sample(key, n, t0, t1):
        kx, ky, kt = jax.random.split(key, 3)
        x = jax.random.uniform(kx, (n,), minval=x_min, maxval=x_max)
        y = jax.random.uniform(ky, (n,), minval=x_min, maxval=x_max)
        t = jax.random.uniform(kt, (n,), minval=t0, maxval=t1)
        return jnp.stack([x, y, t], axis=1)
    return sample


def _initial_plane(key, n, t0):
    """The reference IC lattice at the slab start. Returns [x, y, t0]."""
    m = min(int(np.floor(np.sqrt(n))), _XS.shape[0])
    step = _XS.shape[0] // m
    xs, ys = _XS[::step][:m], _YS[::step][:m]
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    x, y = gx.reshape(-1), gy.reshape(-1)
    dt = jnp.zeros(()).dtype                     # follows jax_enable_x64
    return jnp.stack([jnp.asarray(x, dt), jnp.asarray(y, dt),
                      jnp.full(x.shape, t0, dt)], axis=1)


class KolmogorovFlowProblem(pinn.Problem):
    """u_t + u.grad u + grad p - nu lap u = f,  div u = 0, periodic on [0,1]^2."""

    D = 2
    x_min, x_max, t_max = 0.0, 1.0, TASK_T_MAX
    nu = 5e-4
    problem_name = "KolmogorovFlow"
    ref_path = str(HERE / "kf_reference.npz")

    def __init__(self, *, n_pde, n_ic, n_bc=0):
        self.n_pde, self.n_ic, self.n_bc = n_pde, n_ic, n_bc

    def residual_fns(self):
        return {"pde": self.pde_residual, "ic": self.ic_residual}

    def pde_residual(self, model, coords):
        u_ = lambda c: model(c)[0]
        v_ = lambda c: model(c)[1]
        p_ = lambda c: model(c)[2]
        u, v = u_(coords), v_(coords)
        f = 2.0 * jnp.sin(4.0 * jnp.pi * coords[Y])
        res_u = (op.grad(u_, coords, T)
                 + u * op.grad(u_, coords, X) + v * op.grad(u_, coords, Y)
                 + op.grad(p_, coords, X)
                 - self.nu * (op.grad2(u_, coords, X) + op.grad2(u_, coords, Y))
                 - f)
        res_v = (op.grad(v_, coords, T)
                 + u * op.grad(v_, coords, X) + v * op.grad(v_, coords, Y)
                 + op.grad(p_, coords, Y)
                 - self.nu * (op.grad2(v_, coords, X) + op.grad2(v_, coords, Y)))
        res_div = op.grad(u_, coords, X) + op.grad(v_, coords, Y)
        # the 2.0 on the divergence is their ldc example's own weighting
        return jnp.array([res_u, res_v, 2.0 * res_div])

    def ic_residual(self, model, coords):
        # coords = [x, y, t0, u_target, v_target, p_target]; pressure left free
        pred = model(coords[:3])
        return jnp.array([pred[0] - coords[3], pred[1] - coords[4], 0.0])

    def samplers(self):
        return {"pde": _interior_2d(self.x_min, self.x_max),
                "ic":  _initial_plane}

    def analytical_ic(self, xy):
        """xy : (2, n) -> (3, n). Exact lookup: the sampler returns lattice points."""
        dt = xy.dtype
        xs, ys = jnp.asarray(_XS, dt), jnp.asarray(_YS, dt)
        u0, v0 = jnp.asarray(_U0, dt), jnp.asarray(_V0, dt)
        ix = jnp.clip(jnp.searchsorted(xs, xy[0]), 0, xs.shape[0] - 1)
        iy = jnp.clip(jnp.searchsorted(ys, xy[1]), 0, ys.shape[0] - 1)
        return jnp.stack([u0[ix, iy], v0[ix, iy], jnp.zeros_like(xy[0])])


def make_config(seed: int) -> "pinn.RunConfig":
    return pinn.RunConfig(
        seed=seed,
        dt=TASK_T_MAX / N_SLABS,
        network=lambda key: pinn.MLP(
            key, KolmogorovFlowProblem, hidden_dims=(70, 70, 70, 70),
            periodic_bc=True, n_inputs=3, n_outputs=3,
        ),
        n_pde=2**15, n_ic=2**14,
        residual_sketch=5000, parameter_sketch=5000,
        batch_size=2**12, probe_batch_size=2**8,
        pde_weight=1e-3,
        log_every=10**9, plot_every=10**9,
    )
