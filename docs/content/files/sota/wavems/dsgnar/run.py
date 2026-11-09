"""DSGNAR on the multiscale wave equation — their 1D wave recipe, in 2D with a variable medium.

`wavems_problem.py` carries the problem; see its docstring for every choice and
its source, including why `dt` is kept at their 0.5 rather than rescaled.
Everything that is the method — the `SIREN`, the `operators`, the `RunConfig`,
the sketched trust-region solver via the step-capped `trainer_capped.py` — is
imported unchanged from `reference/DSGNAR/src/pinn`. The donor is
`examples/wave.ipynb`: same equation type (second order in time, both u(0,.)
and u_t(0,.) prescribed) and the same four-residual decomposition, in 1D with a
constant speed and Dirichlet walls. NATIVE = False.

`trainer_capped.py` is their `trainer64.py` with a four-line benchmark patch
(see its docstring): a hard cap on optimiser steps, and the slab models
returned so the task grid can be scored.

`wavems_reference.npz` is built by `make_reference.py` from the task's own
reference, because the trainer loads `problem.ref_path` unconditionally and
their shipped `wave` entry is the 1D constant-speed case. It feeds reporting
only — see that script's docstring.

The method is convergence-driven, so it has no published step count to
compress. `steps` is a plain cap on solver steps sized by hand-run probes to
fill the wall-clock budget; a run cut off mid-way keeps every slab it solved,
and points past the last solved slab are evaluated by that slab's model rather
than returned as NaN. `info.slabs_solved` records how far it got.

The XLA persistent compilation cache lives under `.cache/jax` next to this
file, so seeds after the first skip most of the compile cost.

Reading the score: predicting zero gives exactly 1.0, the best constant 0.781,
and freezing the initial condition 1.274. **u == 0 solves the equation and both
Neumann conditions exactly**, so a run that collapses to it lands on 1.0.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "dsgnar"
NATIVE = False
SOURCE = "reference/DSGNAR/examples/wave.ipynb, extended to 2D with a variable medium"
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

    import wavems_problem
    import trainer_capped

    cfg = wavems_problem.make_config(seed)
    cfg.cache_dir = str(HERE / ".cache" / "jax")

    results = trainer_capped.train(wavems_problem.WaveMSProblem, cfg,
                                   max_global_steps=int(steps))

    slabs = results["slab_models"]          # [(t0, t1, params, static), ...]
    statics = [s for (_, _, _, s) in slabs]
    bounds = [(float(t0), float(t1)) for (t0, t1, _, _) in slabs]
    params = {"slabs": [p for (_, _, p, _) in slabs]}

    def predict(prm, Xt):
        """Xt is (N, 3) as (t, x, y); the DSGNAR model takes (x, y, t)."""
        coords = jnp.stack([Xt[:, 1], Xt[:, 2], Xt[:, 0]], axis=1)
        t = Xt[:, 0]

        def eval_slab(i):
            f = jax.vmap(eqx.combine(prm["slabs"][i], statics[i]))
            outs = [f(coords[k:k + 65536]) for k in range(0, len(coords), 65536)]
            return jnp.concatenate(outs, axis=0)[:, 0]

        u = eval_slab(0)
        for i in range(1, len(statics)):
            t0, t1 = bounds[i]
            last = i == len(statics) - 1
            mask = (t >= t0) if last else ((t >= t0) & (t < t1))
            u = jnp.where(mask, eval_slab(i), u)
        return {"u": u}

    return {"params": params, "predict": predict,
            "info": {"dt": float(cfg.dt),
                     "n_slabs_planned": int(round(wavems_problem.T_MAX / float(cfg.dt))),
                     "network": "SIREN(60x4, 3 in / 1 out)",
                     "n_pde": 2**15, "n_ic": 2**14, "n_bc": 2**14,
                     "residual_sketch": 4000, "parameter_sketch": 4000,
                     "batch_size": 2**14,
                     "pde_weight": 1e-6, "ic_weight": 1.0,
                     "ic_t_weight": 1e-3, "bc_weight": 1.0,
                     "config_template": "examples/wave.ipynb (dt kept at 0.5)",
                     "coefficient": "task 256x256 c(x,y), bilinear",
                     "bc": "zero Neumann on all four sides",
                     "max_global_steps": int(steps),
                     "global_steps_run": int(results["global_steps_run"]),
                     "budget_exhausted": bool(results["budget_exhausted"]),
                     "slabs_solved": [[t0, t1] for t0, t1 in bounds]}}
