"""DSGNAR on steady 3D flow over a finite wing — their method on a flow they do not ship.

`wing3d_problem.py` carries the equation, the geometry and the samplers; see
its docstring for every choice and its source. Everything that is the method —
the `SIREN`, the `operators`, the `RunConfig`, the sketched trust-region solver
via the step-capped `trainer_capped.py` — is imported unchanged from
`reference/DSGNAR/src/pinn`. The structural donor is `examples/ldc.ipynb`,
their own steady Navier-Stokes. NATIVE = False: DSGNAR publishes no 3D flow and
nothing around a body.

The residual was checked line by line against `problem.md`, and the five point
sets against the slicing in the task's own `baseline.py:_build_dataset`.

`wing3d_reference.npz` is built by `make_reference.py` from the task's own
reference and feeds reporting only.

The method is convergence-driven, so there is no published step count to
compress. `steps` is a plain cap sized by hand-run probes. The problem is
steady, so there is a single slab and no time axis at all.

Reading the score: predicting zero gives exactly 1.0 and the freestream
`u = 1, v = w = 0` everywhere gives 0.2116 — that is the anchor to beat.
`problem.md` also states an **error floor of 0.0071**: the reference is a
finite-volume solve, and an independent solve agrees with it only to that, so a
score below about 0.01 is fitting discretisation error rather than physics.
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

    import wing3d_problem as prob
    import trainer_capped

    cfg = prob.make_config(seed)
    cfg.cache_dir = str(HERE / ".cache" / "jax")

    results = trainer_capped.train(prob.Wing3DProblem, cfg,
                                   max_global_steps=int(steps))

    slabs = results["slab_models"]          # steady: exactly one entry
    static = slabs[-1][3]
    params = {"p": slabs[-1][2]}

    def predict(prm, X):
        """X is (N,3) as (x, y, z) — the same order the model takes. The scorer
        forms the velocity magnitude from the three components."""
        f = jax.vmap(eqx.combine(prm["p"], static))
        coords = jnp.asarray(X)
        outs = [f(coords[k:k + 65536]) for k in range(0, len(coords), 65536)]
        o = jnp.concatenate(outs, axis=0)
        return {"u": o[:, 0], "v": o[:, 1], "w": o[:, 2]}

    return {"params": params, "predict": predict,
            "info": {"steady": True,
                     "network": "SIREN(60x4, 3 in / 4 out)",
                     "n_pde": 2**15, "n_bc": 2**13,
                     "residual_sketch": 4000, "parameter_sketch": 4000,
                     "batch_size": 2**12,
                     "pde_weight": 1e-2,
                     "bc_weight": 1.0, "bc2_weight": 1.0,
                     "bc3_weight": 1.0, "bc4_weight": 1.0,
                     "config_template": "examples/ldc.ipynb cell 3, widened to 3 in / 4 out",
                     "nu": prob.NU, "Re": 1000.0,
                     "divergence_weight": 2.0,
                     "bc": ("farfield u=1 (bc), wing no-slip (bc2), outlet p=0 (bc3), "
                            "symmetry plane w=0 and du/dz=dv/dz=0 (bc4)"),
                     "boundary_weights": "all four set to 1.0; the stock 1e-2/1e-4/1e-6 would bury the wing",
                     "geometry": "shipped: collocation.npy level set + wing_surface.npy",
                     "error_floor": 0.0071,
                     "max_global_steps": int(steps),
                     "global_steps_run": int(results["global_steps_run"]),
                     "budget_exhausted": bool(results["budget_exhausted"])}}
