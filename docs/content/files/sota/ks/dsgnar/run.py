"""DSGNAR on KS — the authors' own implementation of this problem, on the
task's time window.

`ks_problem.py` is cell 1 of `reference/DSGNAR/examples/ks.ipynb` verbatim
(equation, coefficients, periodic domain, initial condition) plus cell 2's
`RunConfig` — the same periodic tanh-MLP and solver settings the authors
publish for exactly this benchmark. Two values there are the task's rather
than the notebook's, because the notebook integrates to t = 1 and the task
stops at t = 0.4: `t_max`, and the slab list, which is their own list
*truncated* at 0.4 rather than rescaled (see `ks_problem.py` for why). NATIVE
is therefore False: it is their KS recipe, but not run over their interval.

`trainer_capped.py` is their `trainer64.py` with a four-line benchmark patch
(see its docstring): a hard cap on optimiser steps, and the slab models
returned so the task grid can be scored. It is the same file the kdv case
uses, copied here unchanged.

The method is convergence-driven — each time slab trains until its trust
radius collapses — so it has no published step count to compress the way
scale_pinn's or APTS's schedules were. `steps` is therefore a plain cap on
solver steps sized by bench to fill the wall-clock budget; a run that
converges before the cap simply finishes early, and a run cut off mid-way
keeps every slab it solved. Points past the last solved slab are evaluated by
that slab's model (extrapolation) — the method's best available answer, rather
than a NaN that would void the whole score.

The XLA persistent compilation cache lives under `.cache/jax` next to this
file, so seeds after the first skip most of the compile cost — the same
treatment the trainer's own dry-run mechanism gives repeat runs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "dsgnar"
NATIVE = False
SOURCE = "reference/DSGNAR/examples/ks.ipynb (slabs truncated to the task's t<=0.4)"
DSGNAR_SRC = "~/reference/DSGNAR/src"


def train(*, steps: int, seed: int, gpu: str) -> dict:
    for p in (str(HERE), DSGNAR_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)

    import jax

    jax.config.update("jax_enable_x64", True)   # notebook cell 0: double precision

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

    import ks_problem
    import trainer_capped

    cfg = ks_problem.make_config(seed)
    cfg.cache_dir = str(HERE / ".cache" / "jax")

    results = trainer_capped.train(ks_problem.KSProblem, cfg,
                                   max_global_steps=int(steps))

    slabs = results["slab_models"]          # [(t0, t1, params, static), ...]
    statics = [s for (_, _, _, s) in slabs]
    bounds = [(float(t0), float(t1)) for (t0, t1, _, _) in slabs]
    params = {"slabs": [p for (_, _, p, _) in slabs]}

    def predict(prm, X):
        """X is (N, 2) as (t, x); the DSGNAR model takes (x, t)."""
        coords_xt = jnp.stack([X[:, 1], X[:, 0]], axis=1)
        t = X[:, 0]
        out = jax.vmap(eqx.combine(prm["slabs"][0], statics[0]))(coords_xt)[:, 0]
        for i in range(1, len(statics)):
            t0, t1 = bounds[i]
            last = i == len(statics) - 1
            mask = (t >= t0) if last else ((t >= t0) & (t < t1))
            u = jax.vmap(eqx.combine(prm["slabs"][i], statics[i]))(coords_xt)[:, 0]
            out = jnp.where(mask, u, out)
        return {"u": out}

    return {"params": params, "predict": predict,
            "info": {"slab_boundaries": list(ks_problem.TASK_SLABS),
                     "notebook_slab_boundaries": list(ks_problem.NOTEBOOK_SLABS),
                     "network": "MLP(70x4, periodic)",
                     "n_pde": 2**15, "n_ic": 2**14,
                     "residual_sketch": 5000, "parameter_sketch": 5000,
                     "batch_size": 2**12, "pde_weight": 1e-3,
                     "max_global_steps": int(steps),
                     "global_steps_run": int(results["global_steps_run"]),
                     "budget_exhausted": bool(results["budget_exhausted"]),
                     "slabs_solved": [[t0, t1] for t0, t1 in bounds],
                     "published_recipe": "convergence-driven (no step count); "
                                         "8 slabs clustered towards t=1"}}
