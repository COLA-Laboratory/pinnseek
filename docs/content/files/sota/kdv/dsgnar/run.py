"""DSGNAR on KdV — the authors' own implementation of this problem.

`kdv_problem.py` is cells 2-3 of `reference/DSGNAR/examples/kdv.ipynb`
verbatim: the same equation, dispersion (mu = 0.022), domain, periodic SIREN
and solver settings the authors publish for exactly this benchmark.
`trainer_capped.py` is their `trainer64.py` with a four-line benchmark patch
(see its docstring): a hard cap on optimiser steps, and the slab models
returned so the task grid can be scored.

The method is convergence-driven — each 0.25-wide time slab trains until its
trust radius collapses — so it has no published step count to compress the
way scale_pinn's or APTS's schedules were. `steps` is therefore a plain cap
on solver steps sized by bench to fill the wall-clock budget; a run that
converges before the cap simply finishes early, and a run cut off mid-way
keeps every slab it solved. Points past the last solved slab are evaluated
by that slab's model (extrapolation) — the method's best available answer,
rather than a NaN that would void the whole score.

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
NATIVE = True
SOURCE = "reference/DSGNAR/examples/kdv.ipynb"
DSGNAR_SRC = "~/reference/DSGNAR/src"


def train(*, steps: int, seed: int, gpu: str) -> dict:
    for p in (str(HERE), DSGNAR_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)

    import jax

    jax.config.update("jax_enable_x64", True)   # notebook cell 1: double precision

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

    import kdv_problem
    import trainer_capped

    cfg = kdv_problem.make_config(seed)
    cfg.cache_dir = str(HERE / ".cache" / "jax")

    results = trainer_capped.train(kdv_problem.KdVProblem, cfg,
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
            "info": {"dt": 0.25, "network": "SIREN(60x4, periodic)",
                     "n_pde": 2**15, "n_ic": 2**14,
                     "residual_sketch": 4000, "parameter_sketch": 4000,
                     "batch_size": 2**14, "pde_weight": 1e-4,
                     "max_global_steps": int(steps),
                     "global_steps_run": int(results["global_steps_run"]),
                     "budget_exhausted": bool(results["budget_exhausted"]),
                     "slabs_solved": [[t0, t1] for t0, t1 in bounds],
                     "published_recipe": "convergence-driven (no step count); "
                                         "4 slabs of dt=0.25"}}
