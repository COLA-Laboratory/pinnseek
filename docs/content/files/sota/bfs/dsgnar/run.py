"""DSGNAR on backward-facing step — their method, on a problem they do not ship.

`bfs_problem.py` writes the problem class (see its docstring for every choice
and its source); everything that is the method — the periodic `MLP`, the
`operators`, the `RunConfig` fields, the sketched trust-region solver — is
imported unchanged from `reference/DSGNAR/src/pinn`. NATIVE = False: the
solver settings are their ks notebook's, but the problem is not one of theirs.

`trainer_capped.py` is their `trainer64.py` with a four-line benchmark patch
(see its docstring): a hard cap on optimiser steps, and the slab models
returned so the task grid can be scored. It is the same file the kdv, ks and gs
cases use, copied here unchanged.

`bfs_reference.npz` is built by `make_reference.py` from the task's own
reference, because the trainer loads `problem.ref_path` unconditionally and
their package ships no backward-facing step reference. It feeds reporting only — see
that script's docstring.

The method is convergence-driven — each time slab trains until its trust
radius collapses — so it has no published step count to compress the way
scale_pinn's or APTS's schedules were. `steps` is therefore a plain cap on
solver steps sized by bench to fill the wall-clock budget; a run that converges
before the cap simply finishes early, and a run cut off mid-way keeps every
slab it solved. Points past the last solved slab are evaluated by that slab's
model (extrapolation) — the method's best available answer, rather than a NaN
that would void the whole score. `info.slabs_solved` records how far it got.

The XLA persistent compilation cache lives under `.cache/jax` next to this
file, so seeds after the first skip most of the compile cost — the same
treatment the trainer's own dry-run mechanism gives repeat runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "dsgnar"
NATIVE = False
SOURCE = "reference/DSGNAR/src/pinn (method) + ks notebook's RunConfig; problem written here"
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

    import bfs_problem
    import trainer_capped

    cfg = bfs_problem.make_config(seed)
    cfg.cache_dir = str(HERE / ".cache" / "jax")

    results = trainer_capped.train(bfs_problem.BFSProblem, cfg,
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
                     "config_template": "examples/ldc.ipynb cell 2",
                     "max_global_steps": int(steps),
                     "global_steps_run": int(results["global_steps_run"]),
                     "budget_exhausted": bool(results["budget_exhausted"]),
                     "n_frozen": len(statics)}}
