"""DSGNAR on the NACA0012 airfoil at Re = 1000 — their cavity recipe on an immersed body.

`naca_problem.py` carries the geometry and the boundary conditions; see its
docstring for every choice and its source. Everything that is the method — the
`SIREN`, the `operators`, the `RunConfig`, the sketched trust-region solver via
the step-capped `trainer_capped.py` — is imported unchanged from
`reference/DSGNAR/src/pinn`. The donor is `examples/ldc.ipynb`, their steady 2D
Navier-Stokes case: same equation, same solver regime, no pressure condition on
either side. NATIVE = False — they publish a square cavity, not an airfoil.

`trainer_capped.py` is their `trainer64.py` with a four-line benchmark patch
(see its docstring): a hard cap on optimiser steps, and the slab models
returned so the task grid can be scored. It is the same file the kdv, ks, gs,
bfs, ldc and ib cases use, copied here unchanged.

`naca_reference.npz` is built by `make_reference.py` from the task's own
reference, because the trainer loads `problem.ref_path` unconditionally and
their package ships no airfoil reference. It feeds reporting only — see that
script's docstring.

The method is convergence-driven — it trains until its trust radius collapses
— so it has no published step count to compress the way scale_pinn's or APTS's
schedules were. `steps` is therefore a plain cap on solver steps sized by
hand-run probes to fill the wall-clock budget; a run that converges before the
cap simply finishes early. This case is steady, so there is a single model over
the whole domain rather than a sequence of time slabs.

The XLA persistent compilation cache lives under `.cache/jax` next to this
file, so seeds after the first skip most of the compile cost — the same
treatment the trainer's own dry-run mechanism gives repeat runs.

Reading the score: predicting zero gives exactly 1.0, the uniform freestream
(u=1, v=0) gives 0.480, and the best possible constant 0.403.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "dsgnar"
NATIVE = False
SOURCE = "reference/DSGNAR/examples/ldc.ipynb, re-pointed at the NACA0012 geometry"
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

    import naca_problem
    import trainer_capped

    cfg = naca_problem.make_config(seed)
    cfg.cache_dir = str(HERE / ".cache" / "jax")

    results = trainer_capped.train(naca_problem.NACAProblem, cfg,
                                   max_global_steps=int(steps))

    slabs = results["slab_models"]          # [(t0, t1, params, static), ...]
    statics = [s for (_, _, _, s) in slabs]
    bounds = [(float(t0), float(t1)) for (t0, t1, _, _) in slabs]
    params = {"slabs": [p for (_, _, p, _) in slabs]}

    def predict(prm, Xt):
        """Xt is (N, 2) as (x, y) — steady, so there is no time column and no
        slab masking: the single frozen model covers the whole domain."""
        coords = Xt

        def eval_slab(i):
            f = jax.vmap(eqx.combine(prm["slabs"][i], statics[i]))
            outs = [f(coords[k:k + 65536]) for k in range(0, len(coords), 65536)]
            o = jnp.concatenate(outs, axis=0)
            return o[:, 0], o[:, 1]

        u, v = eval_slab(len(statics) - 1)
        return {"u": u, "v": v}

    return {"params": params, "predict": predict,
            "info": {"steady": True,
                     "network": "SIREN(60x4, 2 in / 3 out)",
                     "n_pde": 2**15, "n_bc": 2**14,
                     "residual_sketch": 4000, "parameter_sketch": 4000,
                     "batch_size": 2**12, "pde_weight": 1e-2,
                     "config_template": "examples/ldc.ipynb cell 3",
                     "Re": 1000.0, "donor_case": "lid_driven_cavity",
                     "bc": "freestream on the outer box, no-slip on 400 analytic airfoil points",
                     "pde_sampling": "task fluid set (phi<=0), 1/8 clustered near the airfoil",
                     "max_global_steps": int(steps),
                     "global_steps_run": int(results["global_steps_run"]),
                     "budget_exhausted": bool(results["budget_exhausted"]),
                     "n_frozen": len(statics)}}
