"""DSGNAR on inviscid Burgers — their own Burgers example, viscosity removed.

`ib_problem.py` is `examples/burgers_double_precision.ipynb` cell 2 with the
viscous term dropped and `t_max` moved from 1.0 to 1.5; see its docstring for
why that makes NATIVE = False rather than True, and for the `dt` choice.
Everything that is the method — the periodic `SIREN`, the `operators`, the
`RunConfig`, the sketched trust-region solver via the step-capped
`trainer_capped.py` — is imported unchanged from `reference/DSGNAR/src/pinn`.

Of all sixteen method/case pairings built so far this is the closest donor:
their own example already has the task's domain, the task's periodic boundary
and the task's `-sin(pi x)` initial condition. Only the viscosity differs.

`trainer_capped.py` is their `trainer64.py` with a four-line benchmark patch
(see its docstring): a hard cap on optimiser steps, and the slab models
returned so the task grid can be scored. It is the same file the kdv, ks, gs,
bfs and ldc cases use, copied here unchanged.

`ib_reference.npz` is built by `make_reference.py` from the task's own
reference, because the trainer loads `problem.ref_path` unconditionally and
their shipped `burgers` entry is the viscous solution. It feeds reporting
only — see that script's docstring.

The method is convergence-driven — it trains until its trust radius collapses
— so it has no published step count to compress the way scale_pinn's or APTS's
schedules were. `steps` is therefore a plain cap on solver steps sized by
hand-run probes to fill the wall-clock budget; a run that converges before the
cap simply finishes early. `dt = t_max` means a single slab over the whole time
domain, so there is no slab masking in `predict`.

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
SOURCE = "reference/DSGNAR/examples/burgers_double_precision.ipynb, viscosity removed"
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

    import ib_problem
    import trainer_capped

    cfg = ib_problem.make_config(seed)
    cfg.cache_dir = str(HERE / ".cache" / "jax")

    results = trainer_capped.train(ib_problem.InviscidBurgersProblem, cfg,
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
            "info": {"dt": 1.5, "network": "SIREN(40x4, periodic)",
                     "n_pde": 2**15, "n_ic": 2**14,
                     "residual_sketch": 2000, "parameter_sketch": 2000,
                     "batch_size": 2**13, "pde_weight": 1e-2,
                     "nu": 0.0, "t_max": 1.5,
                     "config_template": "examples/burgers_double_precision.ipynb cell 4",
                     "max_global_steps": int(steps),
                     "global_steps_run": int(results["global_steps_run"]),
                     "budget_exhausted": bool(results["budget_exhausted"]),
                     "slabs_solved": [[t0, t1] for t0, t1 in bounds],
                     "published_recipe": "convergence-driven (no step count); "
                                         "one slab, dt = t_max"}}
