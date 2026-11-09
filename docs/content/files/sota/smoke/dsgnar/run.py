"""DSGNAR on wildfire-smoke transport — their method on a problem they do not ship.

`smoke_problem.py` carries the equation, the coefficient lookup and the
samplers; see its docstring for every choice and its source. Everything that is
the method — the `SIREN`, the `operators`, the `RunConfig`, the sketched
trust-region solver via the step-capped `trainer_capped.py` — is imported
unchanged from `reference/DSGNAR/src/pinn`. The structural donor is
`examples/laplacian.ipynb` (Dirichlet faces, `pde_weight` far below
`bc_weight`, a multiscale forcing); the time axis follows the 2D+time pattern
the gs, gl, kf and wavems cases here already use. NATIVE = False.

`trainer_capped.py` is their `trainer64.py` with a four-line benchmark patch
(see its docstring): a hard cap on optimiser steps, and the slab models
returned so the task grid can be scored.

`smoke_reference.npz` is built by `make_reference.py` from the task's own
reference, because the trainer loads `problem.ref_path` unconditionally. It
feeds reporting only — see that script's docstring.

The method is convergence-driven, so there is no published step count to
compress. `steps` is a plain cap sized by hand-run probes to fill the
wall-clock budget. `dt = t_max` means a single slab over the whole window, so
there is no slab masking in `predict`.

The XLA persistent compilation cache lives under `.cache/jax` next to this
file, so seeds after the first skip most of the compile cost.

Reading the score: predicting zero gives exactly 1.0 and the best possible
constant 0.972 — the field is 98.9% empty sky, so anything at or above ~0.97
means the plume was never found.

**The single slab was measured, not assumed.** `dt = t_max` disables the time
marching that is DSGNAR's headline mechanism, so the alternative was run:
dt = 0.25 (four slabs), seed 1, the same step cap. It scores **0.99998** against **1.00005** for the
single slab -- the same number, but at 259.7 s against a 180 s budget, because the
slab restarts cost their own compile time. Under this budget the trainer never gets past its first
slab anyway (`slabs_solved` in `info` shows one entry everywhere), so slabbing
only shrinks the window that first slab covers and leaves the rest to
extrapolation -- and each new slab re-triggers the float32/float64 compile
passes on top. The single slab stays.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "dsgnar"
NATIVE = False
SOURCE = "reference/DSGNAR/src/pinn (method) + laplacian notebook's RunConfig; problem written here"
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

    import smoke_problem
    import trainer_capped

    cfg = smoke_problem.make_config(seed)
    cfg.cache_dir = str(HERE / ".cache" / "jax")

    results = trainer_capped.train(smoke_problem.SmokeProblem, cfg,
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
            "info": {"dt": float(cfg.dt), "n_slabs_planned": 1,
                     "network": "SIREN(60x4, n_fourier=4, 3 in / 1 out)",
                     "n_pde": 2**15, "n_ic": 2**14, "n_bc": 2**14,
                     "residual_sketch": 4000, "parameter_sketch": 4000,
                     "batch_size": 2**14,
                     "pde_weight": 1e-8, "bc_weight": 1.0,
                     "config_template": "examples/laplacian.ipynb cell 3 + a time axis",
                     "dxc": smoke_problem.DXC, "dyc": smoke_problem.DYC,
                     "coefficients": "task/baseline.py:coef_at, trilinear, (t,y,x) storage",
                     "ic": "u(0,.) = 0", "bc": "zero Dirichlet on all four sides",
                     "max_global_steps": int(steps),
                     "global_steps_run": int(results["global_steps_run"]),
                     "budget_exhausted": bool(results["budget_exhausted"]),
                     "slabs_solved": [[t0, t1] for t0, t1 in bounds]}}
