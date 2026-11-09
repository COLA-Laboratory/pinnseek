"""DSGNAR on the von Karman vortex street past a cylinder.

The structural donors are `examples/ldc.ipynb` (their own Navier-Stokes, from
which the residual layout and the divergence weighting come) and
`examples/wave.ipynb` (from which the time axis comes). Everything that is the
method -- `SIREN`, `operators`, the `RunConfig`, the sketched trust-region
solver -- is imported unchanged. NATIVE = False: DSGNAR publishes no unsteady
flow and no flow around an obstacle.

The equation is the task's:

    u_t + u u_x + v u_y + p_x - nu (u_xx + u_yy) = 0
    v_t + u v_x + v v_y + p_y - nu (v_xx + v_yy) = 0
    u_x + v_y = 0

with `nu = 1e-3`. Their `ldc` writes the same three lines steady and with
`1/Re`; the two time derivatives are what is added. The `2.0 *` on the
divergence residual is their own weighting, kept.

Coordinate order is DSGNAR's own: `(x, y, t)`, spatial first and time last.

Four conditions rather than one:

  * `ic` -- the task's `cyl_ic.csv` velocity at t = 0, looked up bilinearly on
    its own 441 x 83 grid. The pressure is not supplied there and is not
    scored, so its slot is left free.
  * `bc` -- the four sides of the channel. The inlet carries the driven profile
    `u = 6 U(t) y (H - y) / H^2` with `U(t) = sin(pi (t + 4.8) / 8)`, the walls
    are no-slip, and the outlet carries `p = 0` and nothing else. This is the
    `bfs` port's `jnp.where(at_out, ...)` construction, which is in turn their
    cavity residual's `[u - target, v, 0.0]` with the third slot used.
  * `bc2` -- the cylinder surface, no-slip. It gets its own key because a
    circle cannot be sampled uniformly alongside four straight edges; the
    trainer already keys weights and counts by name.

Nothing else about the method changes. Unlike the SPINN port, the pressure is a
network output here, so the outlet condition *can* be written -- that is a
consequence of solving in primitive variables rather than vorticity form.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

import pinn
from pinn import operators as op

X, Y, T = 0, 1, 2                 # DSGNAR order: spatial first, time last
HERE = Path(__file__).parent
TASK = Path("~/cyl/task")

XL, XU = 0.0, 2.2
YL, YU = 0.0, 0.41
T_MAX = 1.2
NU = 1e-3
H = 0.41
CX, CY, CR = 0.2, 0.2, 0.05


def _ic_grid():
    """(x_axis, y_axis, u, v) from cyl_ic.csv; the cylinder's hole is filled
    with the no-slip value 0, which is the physical value there and is never
    sampled anyway."""
    csv = np.loadtxt(TASK / "cyl_ic.csv", delimiter=",", skiprows=1, dtype=np.float64)
    xs, ys = np.unique(csv[:, 0]), np.unique(csv[:, 1])
    u = np.zeros((len(xs), len(ys)))
    v = np.zeros((len(xs), len(ys)))
    u[np.searchsorted(xs, csv[:, 0]), np.searchsorted(ys, csv[:, 1])] = csv[:, 2]
    v[np.searchsorted(xs, csv[:, 0]), np.searchsorted(ys, csv[:, 1])] = csv[:, 3]
    return xs, ys, u, v


_XS, _YS, _U0, _V0 = _ic_grid()
_XS_J, _YS_J = jnp.asarray(_XS), jnp.asarray(_YS)
_U0_J, _V0_J = jnp.asarray(_U0), jnp.asarray(_V0)
_DX = float(_XS[1] - _XS[0])
_DY = float(_YS[1] - _YS[0])


def _ic_lookup(x, y):
    """Bilinear interpolation of the t = 0 field on its own uniform grid.

    Everything is cast to the coordinate's own dtype. `trainer64` runs its
    first `float32_steps` in single precision and the rest in double, so a
    lookup table pinned to float64 makes the traced function return f64 where
    the pass expects f32 and the jaxpr fails to lower. The rt port takes the
    same precaution.
    """
    dt = x.dtype
    xs0, ys0 = _XS_J.astype(dt)[0], _YS_J.astype(dt)[0]
    fx = jnp.clip((x - xs0) / jnp.asarray(_DX, dt), 0.0, _XS_J.shape[0] - 1.001)
    fy = jnp.clip((y - ys0) / jnp.asarray(_DY, dt), 0.0, _YS_J.shape[0] - 1.001)
    i0, j0 = jnp.floor(fx).astype(int), jnp.floor(fy).astype(int)
    wx, wy = fx - i0, fy - j0

    def bil(g):
        g = g.astype(dt)
        return ((1 - wx) * (1 - wy) * g[i0, j0] + wx * (1 - wy) * g[i0 + 1, j0]
                + (1 - wx) * wy * g[i0, j0 + 1] + wx * wy * g[i0 + 1, j0 + 1])

    return bil(_U0_J), bil(_V0_J)


def mean_inflow(t):
    """U_bar(t) = sin(pi (t + 4.8) / 8) -- problem.md's driven inflow."""
    return jnp.sin(jnp.pi * (t + 4.8) / 8.0)


def inlet_u(t, y):
    return 6.0 * mean_inflow(t) * y * (H - y) / H ** 2


class CylProblem(pinn.Problem):
    """Unsteady 2D Navier-Stokes past a cylinder; outputs (u, v, p)."""

    x_min, x_max = (XL, YL), (XU, YU)      # per-axis bounds, as the bfs port
    t_max = T_MAX
    nu = NU
    problem_name = "CylinderWake"
    ref_path = str(HERE / "cyl_reference.npz")

    def __init__(self, *, n_pde, n_ic, n_bc, n_bc2=None):
        self.n_pde, self.n_ic, self.n_bc = n_pde, n_ic, n_bc
        self.n_bc2 = n_bc if n_bc2 is None else n_bc2

    def residual_fns(self):
        return {"pde": self.pde_residual, "ic": self.ic_residual,
                "bc": self.bc_residual, "bc2": self.cyl_residual}

    def pde_residual(self, model, coords):
        u_ = lambda c: model(c)[0]
        v_ = lambda c: model(c)[1]
        p_ = lambda c: model(c)[2]
        u, v = u_(coords), v_(coords)
        res_u = (op.grad(u_, coords, T)
                 + u * op.grad(u_, coords, X) + v * op.grad(u_, coords, Y)
                 + op.grad(p_, coords, X)
                 - self.nu * (op.grad2(u_, coords, X) + op.grad2(u_, coords, Y)))
        res_v = (op.grad(v_, coords, T)
                 + u * op.grad(v_, coords, X) + v * op.grad(v_, coords, Y)
                 + op.grad(p_, coords, Y)
                 - self.nu * (op.grad2(v_, coords, X) + op.grad2(v_, coords, Y)))
        res_div = op.grad(u_, coords, X) + op.grad(v_, coords, Y)
        return jnp.array([res_u, res_v, 2.0 * res_div])   # their own 2.0

    def ic_residual(self, model, coords):
        # coords = [x, y, t0, u0, v0, p0-unused]
        pred = model(coords[:3])
        return jnp.array([pred[0] - coords[3], pred[1] - coords[4], 0.0])

    def bc_residual(self, model, coords):
        """The four channel sides. Velocity is prescribed everywhere but the
        outlet, which carries p = 0 and nothing else."""
        x, y, t = coords[X], coords[Y], coords[T]
        u, v, p = model(coords)[0], model(coords)[1], model(coords)[2]
        at_out = jnp.isclose(x, XU)
        u_target = jnp.where(jnp.isclose(x, XL), inlet_u(t, y), 0.0)
        r_u = jnp.where(at_out, 0.0, u - u_target)
        r_v = jnp.where(at_out, 0.0, v)
        r_p = jnp.where(at_out, p, 0.0)
        return jnp.array([r_u, r_v, r_p])

    def cyl_residual(self, model, coords):
        """No-slip on the cylinder surface."""
        pred = model(coords)
        return jnp.array([pred[0], pred[1], 0.0])

    # ── samplers ───────────────────────────────────────────────────────
    def samplers(self):
        return {"pde": self._sample_pde, "ic": self._sample_ic,
                "bc": self._sample_bc, "bc2": self._sample_cyl}

    def _sample_pde(self, key, n, t0, t1):
        """7/8 bulk, 1/8 clustered in the near wake, which is where their own
        cavity sampler's clustering idea points on this geometry."""
        n_wake = n // 8
        n_bulk = n - n_wake
        k_b, k_t, k_wx, k_wy, k_wt = jax.random.split(key, 5)
        bulk = jax.random.uniform(k_b, (n_bulk, 2),
                                  minval=jnp.array([XL, YL]),
                                  maxval=jnp.array([XU, YU]))
        wake = jnp.stack([jax.random.uniform(k_wx, (n_wake,), minval=0.15, maxval=1.0),
                          jax.random.uniform(k_wy, (n_wake,), minval=0.05, maxval=0.35)],
                         axis=1)
        s = jnp.concatenate([bulk, wake], axis=0)
        s = _push_out(s)
        t = jax.random.uniform(k_t, (n, 1), minval=t0, maxval=t1)
        return jnp.concatenate([s, t], axis=1)

    def _sample_ic(self, key, n, t0):
        s = jax.random.uniform(key, (n, 2), minval=jnp.array([XL, YL]),
                               maxval=jnp.array([XU, YU]))
        s = _push_out(s)
        return jnp.concatenate([s, jnp.full((n, 1), t0)], axis=1)

    def _sample_bc(self, key, n, t0, t1):
        q = n // 4
        k1, k2, k3, k4, kt = jax.random.split(key, 5)
        inlet = jnp.stack([jnp.full((q,), XL),
                           jax.random.uniform(k1, (q,), minval=YL, maxval=YU)], axis=1)
        outlet = jnp.stack([jnp.full((q,), XU),
                            jax.random.uniform(k2, (q,), minval=YL, maxval=YU)], axis=1)
        top = jnp.stack([jax.random.uniform(k3, (q,), minval=XL, maxval=XU),
                         jnp.full((q,), YU)], axis=1)
        bot = jnp.stack([jax.random.uniform(k4, (n - 3 * q,), minval=XL, maxval=XU),
                         jnp.full((n - 3 * q,), YL)], axis=1)
        s = jnp.concatenate([inlet, outlet, top, bot], axis=0)
        t = jax.random.uniform(kt, (n, 1), minval=t0, maxval=t1)
        return jnp.concatenate([s, t], axis=1)

    def _sample_cyl(self, key, n, t0, t1):
        k_a, k_t = jax.random.split(key)
        a = jax.random.uniform(k_a, (n,), minval=0.0, maxval=2.0 * jnp.pi)
        s = jnp.stack([CX + CR * jnp.cos(a), CY + CR * jnp.sin(a)], axis=1)
        t = jax.random.uniform(k_t, (n, 1), minval=t0, maxval=t1)
        return jnp.concatenate([s, t], axis=1)

    def analytical_ic(self, xy):
        """The task's own t = 0 velocity, looked up. Returns (out_dim, n)."""
        u0, v0 = _ic_lookup(xy[X], xy[Y])
        return jnp.stack([u0, v0, jnp.zeros_like(u0)])


def _push_out(s):
    """A point inside the cylinder is pushed to just outside it along its own
    radius. That keeps `n` fixed -- which rejection sampling cannot do under
    jit -- and leaves the bulk untouched, since the cylinder covers 0.9% of the
    channel."""
    c = jnp.array([CX, CY])
    v = s - c
    d = jnp.linalg.norm(v, axis=1, keepdims=True)
    safe = jnp.where(d > 1e-12, d, 1e-12)
    return jnp.where(d < CR, c + v / safe * CR * 1.0001, s)


def make_config(seed: int) -> "pinn.RunConfig":
    """The ldc notebook's cell 3, given a time axis and a fourth condition."""
    return pinn.RunConfig(
        seed=seed,
        dt=T_MAX,
        network=lambda key: pinn.SIREN(
            key, CylProblem, hidden_dims=(60, 60, 60, 60),
            periodic_bc=False, n_inputs=3, n_outputs=3,
        ),
        n_pde=2**15, n_ic=2**14, n_bc=2**14,
        residual_sketch=4000, parameter_sketch=4000,
        batch_size=2**12, probe_batch_size=2**8,
        pde_weight=1e-2, ic_weight=1.0, bc_weight=1.0,
        bc2_weight=1.0,               # the cylinder is a boundary like any other
        log_every=10**9, plot_every=10**9,     # headless; not part of the method
    )
