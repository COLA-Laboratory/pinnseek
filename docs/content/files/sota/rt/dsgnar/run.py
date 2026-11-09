"""DSGNAR on Rayleigh-Taylor — their steady-cavity recipe, given a time axis and a temperature.

`rt_problem.py` carries the four Boussinesq equations, the boundary treatment
and the file-supplied initial condition; see its docstring for every choice and
its source. Everything that is the method — the `SIREN`, the `operators`, the
`RunConfig`, the sketched trust-region solver via the step-capped
`trainer_capped.py` — is imported unchanged from `reference/DSGNAR/src/pinn`.
The donor is `examples/ldc.ipynb`, their incompressible Navier-Stokes case.
NATIVE = False.

Verified against the task rather than assumed (see the checks in
`rt_problem.py`'s docstring and the numbers below):

  * the four residuals reproduce the task's equations coefficient by
    coefficient, including the buoyancy term `-T` in the v-momentum equation
    and `NU = 8.426150e-4`, `KAPPA = 1.186782e-3`;
  * the initial condition is a lookup into the task's own `rt_ic.csv`. It is a
    *staircase* lookup, not an exact one: the initial sampler spreads its points
    evenly over the domain while that grid is stretched, so none of them lands
    on a stored node (0 of 128 on the x axis). Against bilinear interpolation
    of the same table the error is 0.0180 in relative L2. Sampling the stored
    nodes instead was measured and scores worse (1.0161 vs 0.9933, seed 1) —
    see `rt_problem.py`'s docstring;
  * the wall sampler lands on y = 0 and y = 2, the periodic sampler on x = 0.

`rt_reference.npz` holds **T only**, because only T is scored; it is built by
`make_reference.py` from the task's reference and feeds reporting only.

The method is convergence-driven, so there is no published step count to
compress. `steps` is a plain cap sized by hand-run probes. `dt = t_max` means a
single slab over the whole window.

Only T is scored, so `predict` returns the key "T"; u, v and p are trained
because the buoyancy coupling needs them, but are not part of the metric.

Reading the score: predicting zero gives exactly 1.0 and the best possible
constant 0.740.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "dsgnar"
NATIVE = False
SOURCE = "reference/DSGNAR/examples/ldc.ipynb, given a time axis and a temperature field"
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

    import rt_problem
    import trainer_capped

    cfg = rt_problem.make_config(seed)
    cfg.cache_dir = str(HERE / ".cache" / "jax")

    results = trainer_capped.train(rt_problem.RTProblem, cfg,
                                   max_global_steps=int(steps))

    slabs = results["slab_models"]          # [(t0, t1, params, static), ...]
    statics = [s for (_, _, _, s) in slabs]
    bounds = [(float(t0), float(t1)) for (t0, t1, _, _) in slabs]
    params = {"slabs": [p for (_, _, p, _) in slabs]}

    def predict(prm, Xt):
        """Xt is (N,3) as (t, x, y); the model takes (x, y, t). Only T is scored."""
        coords = jnp.stack([Xt[:, 1], Xt[:, 2], Xt[:, 0]], axis=1)
        t = Xt[:, 0]

        def eval_slab(i):
            f = jax.vmap(eqx.combine(prm["slabs"][i], statics[i]))
            outs = [f(coords[k:k + 65536]) for k in range(0, len(coords), 65536)]
            return jnp.concatenate(outs, axis=0)[:, 3]   # T is output 3

        u = eval_slab(0)
        for i in range(1, len(statics)):
            t0, t1 = bounds[i]
            last = i == len(statics) - 1
            mask = (t >= t0) if last else ((t >= t0) & (t < t1))
            u = jnp.where(mask, eval_slab(i), u)
        return {"T": u}

    return {"params": params, "predict": predict,
            "info": {"dt": float(cfg.dt), "n_slabs_planned": 1,
                     "network": "SIREN(60x4, 3 in / 4 out)",
                     "n_pde": 2**15, "n_ic": 2**14, "n_bc": 2**14,
                     "residual_sketch": 4000, "parameter_sketch": 4000,
                     "batch_size": 2**12, "pde_weight": 1e-2,
                     "config_template": "examples/ldc.ipynb cell 3 + a time axis",
                     "NU": rt_problem.NU, "KAPPA": rt_problem.KAPPA,
                     "Pr": rt_problem.PR, "Ra": rt_problem.RA,
                     "ic": "task rt_ic.csv, nearest-neighbour lookup (exact on its grid)",
                     "bc": "walls u=v=T=0 at y=0,2; periodic in x with period 1",
                     "scored_field": "T only",
                     "max_global_steps": int(steps),
                     "global_steps_run": int(results["global_steps_run"]),
                     "budget_exhausted": bool(results["budget_exhausted"]),
                     "slabs_solved": [[t0, t1] for t0, t1 in bounds]}}
