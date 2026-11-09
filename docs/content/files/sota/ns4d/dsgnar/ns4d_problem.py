"""DSGNAR on 3D unsteady Navier-Stokes in vorticity-velocity form.

The structural donors are `examples/ldc.ipynb` (their own Navier-Stokes, from
which the residual layout and the divergence weighting come) and
`examples/wave.ipynb` (from which the time axis and the two-initial-condition
pattern come). Everything that is the method -- `SIREN`, `operators`, the
`RunConfig`, the sketched trust-region solver -- is imported unchanged.
NATIVE = False: DSGNAR publishes no 3D flow and nothing 4-dimensional.

The equation is the task's, which is also SPINN's published one:

    w_t + (u . grad) w - (w . grad) u - nu lap(w) - f = 0,   div u = 0

with `w = curl u` **not** a network output, so `lap(w)` is a *third* derivative
of the velocity. The network emits (u, v, w_z-velocity) and every vorticity in
the residual is built from it by `op.grad`, the same way SPINN's
`utils/vorticity.py` builds it by `jvp`. nu = 0.05 and the forcing are
`problem.md`'s, reproduced here and cross-checked against SPINN's
`navier_stokes4d_forcing_term`.

Coordinate order is DSGNAR's own: `(x, y, z, t)`, spatial first and time last.

Three things worth stating:

  * **the boundary condition costs nothing.** The domain is 2pi-periodic in all
    three spatial directions, and `SIREN(periodic_bc=True)` embeds each spatial
    coordinate as `[cos(2 pi x~), sin(2 pi x~)]` with `x~` normalised by
    `x_min`/`x_max`. With `x_min = 0` and `x_max = 2 pi` that is *exactly*
    2pi-periodic by construction, so there is no boundary residual at all and
    no boundary weight to tune. SPINN, by contrast, has to add a boundary loss
    -- and supervises it from the analytic solution.

  * **the initial condition is the task's, in two parts.** `ic` matches the
    velocity at t = 0 and `ic_t` -- their stock second-initial-condition slot --
    matches the vorticity there. Both are values `task/ns4d_ic.csv` supplies on
    its 32^3 grid; they are evaluated from the closed form the file was
    generated from, which agrees with the file to 2.5e-9 (checked in
    ../../problem.py). The vorticity term is redundant in principle, since the
    curl of a matched velocity is matched too, but it is what the task ships
    and what SPINN's own `initial_loss` uses.

  * **the divergence weight is theirs, not SPINN's.** `ldc.ipynb` writes
    `jnp.array([res_u, res_v, 2.0 * res_div])`, so the incompressibility
    residual carries a factor 2 here as well. SPINN uses `lbda_c = 100` for the
    same term; each method keeps its own published choice.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

import pinn
from pinn import operators as op

HERE = Path(__file__).parent

X, Y, Z, T = 0, 1, 2, 3           # DSGNAR order: spatial first, time last
NU = 0.05
L = 2.0 * jnp.pi
T_MAX = 5.0


def forcing(t, x, y, z):
    """problem.md's f -- identical to SPINN's navier_stokes4d_forcing_term."""
    e = jnp.exp(-18.0 * NU * t)
    return (-6.0 * e * jnp.sin(4.0 * y) * jnp.sin(2.0 * z),
            -6.0 * e * jnp.sin(4.0 * x) * jnp.sin(2.0 * z),
            6.0 * e * jnp.sin(4.0 * x) * jnp.sin(4.0 * y))


def exact_u(t, x, y, z):
    """The closed form task/ns4d_ic.csv was generated from, used at t = 0 only."""
    e = jnp.exp(-9.0 * NU * t)
    return (2.0 * e * jnp.cos(2.0 * x) * jnp.sin(2.0 * y) * jnp.sin(z),
            -1.0 * e * jnp.sin(2.0 * x) * jnp.cos(2.0 * y) * jnp.sin(z),
            -2.0 * e * jnp.sin(2.0 * x) * jnp.sin(2.0 * y) * jnp.cos(z))


def exact_w(t, x, y, z):
    """The same file's vorticity columns, in closed form."""
    e = jnp.exp(-9.0 * NU * t)
    return (-3.0 * e * jnp.sin(2.0 * x) * jnp.cos(2.0 * y) * jnp.cos(z),
            6.0 * e * jnp.cos(2.0 * x) * jnp.sin(2.0 * y) * jnp.cos(z),
            -6.0 * e * jnp.cos(2.0 * x) * jnp.cos(2.0 * y) * jnp.sin(z))


class NS4DProblem(pinn.Problem):
    """3D unsteady Navier-Stokes, vorticity-velocity form; outputs (u, v, w)."""

    x_min, x_max = 0.0, float(L)
    t_max = T_MAX
    nu = NU
    problem_name = "NavierStokes4D"
    ref_path = str(HERE / "ns4d_reference.npz")

    def __init__(self, *, n_pde, n_ic, n_bc=0):
        self.n_pde, self.n_ic, self.n_bc = n_pde, n_ic, n_bc
        self.n_ic_t = n_ic

    def residual_fns(self):
        # no "bc": periodicity is exact in the embedding, not a penalty
        return {"pde": self.pde_residual, "ic": self.ic_residual,
                "ic_t": self.ic_t_residual}

    # ── the three vorticity components, as functions of a coordinate ────
    def _vel(self, model):
        return (lambda c: model(c)[0], lambda c: model(c)[1], lambda c: model(c)[2])

    def _vor(self, model):
        u_, v_, w_ = self._vel(model)
        wx = lambda c: op.grad(w_, c, Y) - op.grad(v_, c, Z)
        wy = lambda c: op.grad(u_, c, Z) - op.grad(w_, c, X)
        wz = lambda c: op.grad(v_, c, X) - op.grad(u_, c, Y)
        return wx, wy, wz

    def pde_residual(self, model, coords):
        u_, v_, w_ = self._vel(model)
        wx_, wy_, wz_ = self._vor(model)
        x, y, z, t = coords[X], coords[Y], coords[Z], coords[T]
        u, v, w = u_(coords), v_(coords), w_(coords)
        wx, wy, wz = wx_(coords), wy_(coords), wz_(coords)
        fx, fy, fz = forcing(t, x, y, z)
        sp = (X, Y, Z)

        def component(vor, vel, f):
            adv = (u * op.grad(vor, coords, X) + v * op.grad(vor, coords, Y)
                   + w * op.grad(vor, coords, Z))
            stretch = (wx * op.grad(vel, coords, X) + wy * op.grad(vel, coords, Y)
                       + wz * op.grad(vel, coords, Z))
            return (op.grad(vor, coords, T) + adv - stretch
                    - self.nu * op.laplacian(vor, coords, sp) - f)

        r_x = component(wx_, u_, fx)
        r_y = component(wy_, v_, fy)
        r_z = component(wz_, w_, fz)
        r_div = (op.grad(u_, coords, X) + op.grad(v_, coords, Y)
                 + op.grad(w_, coords, Z))
        # 2.0 on the divergence is ldc.ipynb's own weighting
        return jnp.array([r_x, r_y, r_z, 2.0 * r_div])

    def ic_residual(self, model, coords):
        # coords = [x, y, z, t0, u0, v0, w0]
        pred = model(coords[:4])
        return jnp.array([pred[0] - coords[4], pred[1] - coords[5],
                          pred[2] - coords[6]])

    def ic_t_residual(self, model, coords):
        """[their `ic_t` slot] the vorticity at t = 0, which the task also ships."""
        c = coords[:4]
        wx_, wy_, wz_ = self._vor(model)
        return jnp.array([wx_(c) - coords[4], wy_(c) - coords[5],
                          wz_(c) - coords[6]])

    # ── samplers ───────────────────────────────────────────────────────
    def samplers(self):
        return {"pde": self._sample_pde, "ic": self._sample_ic,
                "ic_t": self._sample_ic}

    def _sample_pde(self, key, n, t0, t1):
        k_s, k_t = jax.random.split(key)
        s = jax.random.uniform(k_s, (n, 3), minval=self.x_min, maxval=self.x_max)
        t = jax.random.uniform(k_t, (n, 1), minval=t0, maxval=t1)
        return jnp.concatenate([s, t], axis=1)

    def _sample_ic(self, key, n, t0):
        s = jax.random.uniform(key, (n, 3), minval=self.x_min, maxval=self.x_max)
        return jnp.concatenate([s, jnp.full((n, 1), t0)], axis=1)

    def analytical_ic(self, xyz):
        """Velocity at t = 0. Returns (out_dim, n)."""
        u0, v0, w0 = exact_u(0.0, xyz[X], xyz[Y], xyz[Z])
        return jnp.stack([u0, v0, w0])

    def analytical_ic_t(self, xyz):
        """Vorticity at t = 0 -- the task's own wx, wy, wz columns."""
        wx0, wy0, wz0 = exact_w(0.0, xyz[X], xyz[Y], xyz[Z])
        return jnp.stack([wx0, wy0, wz0])


def make_config(seed: int) -> "pinn.RunConfig":
    """The ldc notebook's cell 3, widened to a 3D+time problem.

    `periodic_bc=True` is what makes the boundary condition exact rather than
    penalised; `dt = t_max` means a single slab over the whole window, as the
    other 2D+time cases here use.
    """
    return pinn.RunConfig(
        seed=seed,
        dt=T_MAX,
        network=lambda key: pinn.SIREN(
            key, NS4DProblem, hidden_dims=(60, 60, 60, 60),
            periodic_bc=True, n_inputs=4, n_outputs=3,
        ),
        n_pde=2**15, n_ic=2**14, n_bc=0,
        residual_sketch=4000, parameter_sketch=4000,
        batch_size=2**12, probe_batch_size=2**8,
        pde_weight=1e-2, ic_weight=1.0, ic_t_weight=1.0,
        log_every=10**9, plot_every=10**9,     # headless; not part of the method
    )
