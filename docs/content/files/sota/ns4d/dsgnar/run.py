"""DSGNAR on 3D unsteady Navier-Stokes — their method on a problem they do not ship.

`ns4d_problem.py` carries the equation, the initial condition and the samplers;
see its docstring for every choice and its source. Everything that is the
method — the `SIREN`, the `operators`, the `RunConfig`, the sketched
trust-region solver via the step-capped `trainer_capped.py` — is imported
unchanged from `reference/DSGNAR/src/pinn`. The structural donors are
`examples/ldc.ipynb` (their own Navier-Stokes) and `examples/wave.ipynb` (the
time axis and the two-initial-condition pattern). NATIVE = False: DSGNAR
publishes no 3D flow and nothing 4-dimensional.

The residual was checked term by term against `problem.md` and against SPINN's
`navier_stokes4d.py`, which solves the identical problem; the forcing and the
closed-form initial fields were checked against SPINN's `data_utils.py` and
against `task/ns4d_ic.csv` (2.5e-9 on 32768 points).

`ns4d_reference.npz` is built by `make_reference.py` from the task's own
reference and feeds reporting only.

The method is convergence-driven, so there is no published step count to
compress. `steps` is a plain cap sized by hand-run probes. `dt = t_max` means a
single slab over the whole 5-unit window.

**The periodicity is free here.** `SIREN(periodic_bc=True)` embeds each spatial
coordinate as `[cos(2 pi x~), sin(2 pi x~)]`, which on `[0, 2 pi]` is exactly
the task's boundary condition — no boundary residual, no boundary weight, and
no boundary data. That is a real structural advantage on this case and is
recorded in `info` as `bc`.

Reading the score: predicting zero gives exactly 1.0 and freezing the initial
condition 1.342 — the field decays to 10.5% of its amplitude by t = 5, so
holding t = 0 is worse than giving up.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "dsgnar"
NATIVE = False
SOURCE = "reference/DSGNAR/src/pinn (method) + ldc/wave notebooks' RunConfig; problem written here"
DSGNAR_SRC = "~/reference/DSGNAR/src"


def train(*, steps: int, seed: int, gpu: str) -> dict:
    for p in (str(HERE), DSGNAR_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)

    import jax

    jax.config.update("jax_enable_x64", True)   # their notebooks' cell 0

    import jax.numpy as jnp
    import equinox as eqx

    # Compat shim for the frozen jax 0.4.33: DSGNAR's optimiser calls
    # jnp.minimum.accumulate (a numpy-ufunc method jax grew later); a running
    # minimum is exactly lax.cummin. The reference source stays unmodified.
    if not hasattr(jnp.minimum, "accumulate"):
        _orig_minimum = jnp.minimum

        class _MinimumWithAccumulate:
            def __call__(self, *a, **k):
                return _orig_minimum(*a, **k)

            @staticmethod
            def accumulate(x, axis=0):
                return jax.lax.cummin(x, axis=axis)

        jnp.minimum = _MinimumWithAccumulate()

    import ns4d_problem as prob
    import trainer_capped
    from pinn import operators as op

    cfg = prob.make_config(seed)
    cfg.cache_dir = str(HERE / ".cache" / "jax")

    results = trainer_capped.train(prob.NS4DProblem, cfg,
                                   max_global_steps=int(steps))

    slabs = results["slab_models"]          # dt = t_max: exactly one entry
    static = slabs[-1][3]
    params = {"p": slabs[-1][2]}

    def predict(prm, Xt):
        """Xt is (N,4) as (t, x, y, z); the DSGNAR model takes (x, y, z, t).

        The vorticity is the autodiff curl of the velocity, built with the same
        `op.grad` the residual uses — the task requires the curl, not a fourth
        network output."""
        model = eqx.combine(prm["p"], static)
        coords = jnp.stack([Xt[:, 1], Xt[:, 2], Xt[:, 3], Xt[:, 0]], axis=1)
        X_, Y_, Z_ = prob.X, prob.Y, prob.Z

        def curl(c):
            u_ = lambda q: model(q)[0]
            v_ = lambda q: model(q)[1]
            w_ = lambda q: model(q)[2]
            return jnp.array([op.grad(w_, c, Y_) - op.grad(v_, c, Z_),
                              op.grad(u_, c, Z_) - op.grad(w_, c, X_),
                              op.grad(v_, c, X_) - op.grad(u_, c, Y_)])

        f = jax.jit(jax.vmap(curl))
        outs = [f(coords[k:k + 16384]) for k in range(0, len(coords), 16384)]
        w = jnp.concatenate(outs, axis=0)
        return {"wx": w[:, 0], "wy": w[:, 1], "wz": w[:, 2]}

    return {"params": params, "predict": predict,
            "info": {"dt": float(cfg.dt), "n_slabs_planned": 1,
                     "network": "SIREN(60x4, periodic embedding, 4 in / 3 out)",
                     "n_pde": 2**15, "n_ic": 2**14, "n_ic_t": 2**14,
                     "residual_sketch": 4000, "parameter_sketch": 4000,
                     "batch_size": 2**12,
                     "pde_weight": 1e-2, "ic_weight": 1.0, "ic_t_weight": 1.0,
                     "config_template": "examples/ldc.ipynb cell 3 + a time axis",
                     "nu": prob.NU, "t_max": prob.T_MAX,
                     "divergence_weight": 2.0,
                     "bc": ("exact: SIREN(periodic_bc=True) embeds cos/sin of the "
                            "normalised spatial coordinates, which on [0, 2pi] is "
                            "the task's periodicity by construction -- no boundary "
                            "residual and no boundary data"),
                     "ic": "velocity at t=0 (ic) and vorticity at t=0 (ic_t), both shipped by the task",
                     "vorticity": "autodiff curl of the velocity via op.grad",
                     "max_global_steps": int(steps),
                     "global_steps_run": int(results["global_steps_run"]),
                     "budget_exhausted": bool(results["budget_exhausted"])}}
