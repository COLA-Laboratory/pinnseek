"""DSGNAR on the 3D screened Poisson problem in a cube minus two spheres.

`poissoncg_problem.py` carries the equation, the geometry and the samplers; see
its docstring for every choice and its source. Everything that is the method —
the `GaborNet`, the `operators`, the `RunConfig`, the sketched trust-region
solver via the step-capped `trainer_capped.py` — is imported unchanged from
`reference/DSGNAR/src/pinn`. The structural donor is `examples/poisson5d.ipynb`,
their own steady Poisson, dropped from five dimensions to three and screened.
NATIVE = False.

The residual was checked term by term against `task/baseline.py` (`screening`
and `source` are reproduced verbatim there), and the Neumann normals against
the same file's `nrm_face` / `nrm_sph` construction — including the sign
convention that the sphere normal points *into* the removed ball.

`poissoncg_reference.npz` is built by `make_reference.py` from the task's own
reference and feeds reporting only.

The method is convergence-driven, so there is no published step count to
compress. `steps` is a plain cap sized by hand-run probes. The problem is
steady, so there is a single slab and no time axis at all.

Reading the score: predicting zero gives exactly 1.0 and the best constant
gives 0.929 — the field has a nonzero mean, so the two floors do not coincide
the way they do in the oscillating cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "dsgnar"
NATIVE = False
SOURCE = "reference/DSGNAR/src/pinn (method) + examples/poisson5d.ipynb; geometry written here"
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

    import poissoncg_problem as prob
    import trainer_capped

    cfg = prob.make_config(seed)
    cfg.cache_dir = str(HERE / ".cache" / "jax")

    results = trainer_capped.train(prob.PoissonCGProblem, cfg,
                                   max_global_steps=int(steps))

    slabs = results["slab_models"]          # steady: exactly one entry
    static = slabs[-1][3]
    params = {"p": slabs[-1][2]}

    def predict(prm, X):
        """X is (N, 3) as (x, y, z) — the same order the model takes."""
        f = jax.vmap(eqx.combine(prm["p"], static))
        coords = jnp.asarray(X)
        outs = [f(coords[k:k + 65536]) for k in range(0, len(coords), 65536)]
        return {"u": jnp.concatenate(outs, axis=0)[:, 0]}

    return {"params": params, "predict": predict,
            "info": {"steady": True,
                     "network": "GaborNet(n_gabor=1024, 3 in / 1 out)",
                     "n_pde": 2 ** 15, "n_bc": 2 ** 14, "n_bc2": 2 ** 14,
                     "residual_sketch": 4000, "parameter_sketch": 4000,
                     "batch_size": 2 ** 12, "n_probes": 64,
                     "window_scale": 6.0,
                     "pde_weight": 1e-4, "bc_weight": 1.0, "bc2_weight": 1.0,
                     "config_template": "examples/poisson5d.ipynb cell 3, n_inputs 5 -> 3",
                     "K_lo": prob.K_LO, "K_hi": prob.K_HI,
                     "z_interface": prob.Z_IFACE,
                     "spheres": [list(s) for s in prob.SPHERES],
                     "bc": "homogeneous Neumann on 6 faces + 2 sphere surfaces",
                     "max_global_steps": int(steps),
                     "global_steps_run": int(results["global_steps_run"]),
                     "budget_exhausted": bool(results["budget_exhausted"])}}
