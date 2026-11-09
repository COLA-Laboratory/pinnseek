"""DSGNAR on the von Karman vortex street — their method on a flow they do not ship.

`cyl_problem.py` carries the equation, the geometry, the initial condition and
the samplers; see its docstring for every choice and its source. Everything
that is the method — the `SIREN`, the `operators`, the `RunConfig`, the
sketched trust-region solver via the step-capped `trainer_capped.py` — is
imported unchanged from `reference/DSGNAR/src/pinn`. The structural donors are
`examples/ldc.ipynb` (their own Navier-Stokes) and `examples/wave.ipynb` (the
time axis). NATIVE = False.

The residual was checked line by line against `problem.md`, and the driven
inflow and the geometry against the task's own `baseline.py`.

`cyl_reference.npz` is built by `make_reference.py` from the task's own
reference and feeds reporting only.

The method is convergence-driven, so there is no published step count to
compress. `steps` is a plain cap sized by hand-run probes. `dt = t_max` means a
single slab over the whole 1.2-unit window — one shedding cycle.

**The pressure is a network output here**, so the outlet's `p = 0` can be
written down; the SPINN port on this same case cannot write it, because its
published formulation is in vorticity form and taking the curl is what removes
the pressure. Worth reading the two `info` blocks together.

Reading the score: predicting zero gives exactly 1.0, the best constant 0.538,
and freezing the initial condition 0.399.
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

    import cyl_problem as prob
    import trainer_capped

    cfg = prob.make_config(seed)
    cfg.cache_dir = str(HERE / ".cache" / "jax")

    results = trainer_capped.train(prob.CylProblem, cfg,
                                   max_global_steps=int(steps))

    slabs = results["slab_models"]          # dt = t_max: exactly one entry
    static = slabs[-1][3]
    params = {"p": slabs[-1][2]}

    def predict(prm, Xt):
        """Xt is (N,3) as (t, x, y); the DSGNAR model takes (x, y, t)."""
        f = jax.vmap(eqx.combine(prm["p"], static))
        coords = jnp.stack([Xt[:, 1], Xt[:, 2], Xt[:, 0]], axis=1)
        outs = [f(coords[k:k + 65536]) for k in range(0, len(coords), 65536)]
        o = jnp.concatenate(outs, axis=0)
        return {"u": o[:, 0], "v": o[:, 1]}

    return {"params": params, "predict": predict,
            "info": {"dt": float(cfg.dt), "n_slabs_planned": 1,
                     "network": "SIREN(60x4, 3 in / 3 out)",
                     "n_pde": 2**15, "n_ic": 2**14, "n_bc": 2**14, "n_bc2": 2**14,
                     "residual_sketch": 4000, "parameter_sketch": 4000,
                     "batch_size": 2**12,
                     "pde_weight": 1e-2, "ic_weight": 1.0,
                     "bc_weight": 1.0, "bc2_weight": 1.0,
                     "config_template": "examples/ldc.ipynb cell 3 + a time axis",
                     "nu": prob.NU, "t_max": prob.T_MAX,
                     "cylinder": [prob.CX, prob.CY, prob.CR],
                     "divergence_weight": 2.0,
                     "bc": ("inlet profile + two walls + outlet p = 0 (bc) and the "
                            "cylinder's no-slip surface (bc2)"),
                     "outlet": "p = 0 is expressible here: the pressure is a network output",
                     "ic": "task/cyl_ic.csv velocity, bilinear on its 441 x 83 grid",
                     "max_global_steps": int(steps),
                     "global_steps_run": int(results["global_steps_run"]),
                     "budget_exhausted": bool(results["budget_exhausted"])}}
