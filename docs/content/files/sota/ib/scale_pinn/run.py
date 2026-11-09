"""Scale-PINN on inviscid Burgers — their KdV recipe, with the dispersion switched off.

Scale-PINN publishes no Burgers case. Its closest example by a wide margin is
`kdv/plot_benchmark_kdv.ipynb`: 1D + time on x in [-1, 1] with a periodic
boundary, and a residual their own code already writes as

    pde = u_t + v1*u*u_x + v2*u_xxx

— that is, **the task's equation is theirs with v2 = 0**. `src.py` is that
notebook extracted verbatim by tools/extract_scale_pinn.py, and it carries
three `[sota patch]` marks, all inside `main()`:

  1. `v2 = 0.022^2` -> `0.0`. This is the physics change and the reason
     NATIVE = False: KdV is third-order dispersive, this task is a first-order
     conservation law whose characteristics cross into a standing shock at
     x = 0 from t = 1/pi. Zeroing v2 also cancels the
     `sqrt(v2)*(u_xx - u0_xx)/ER_xx` half of their ER correction, which is the
     dispersion-scaled stabiliser — so the correction degenerates to the
     `(u - u_0)/ER` term on its own, which is what their formulation reduces to
     when there is no high-order term. `ER_xx` is left at the published value
     and is inert.
  2. `kdv.mat` -> the task's `ref_data.csv`, laid out t-major/x-minor exactly
     as their own three lines do it (asserted, not assumed). It plays the same
     role theirs does: the collocation grid, the IC and periodic labels, and
     the `rl2` progress print. Only the t = 0 row and the periodic column enter
     the loss; the interior values are reporting.
  3. `x_u = x_l + 2.0`. Their KdV grid includes both endpoints, so
     `x_u - x_l` is exactly the period and `pbc = u(t, x_u) - u(t, x_l)` is the
     periodic condition as written. This task's grid is **cell-centred**
     (-0.998046875 to +0.998046875, spanning 1.99609375), so those two columns
     are not a period apart and their term would impose the wrong constraint.
     The period is 2 wherever the samples sit.

Nothing else moves: the sin-feature network, the skip structure, the minibatch,
the ER mechanism, the loss shape and the schedule are all theirs.

Hyperparameters are the donor's own — row 8 of the paper's Table A1, which is
also the live call in the KdV notebook:

    ER=0.1, ER_xx=20, weight_ic=1000, weight_bc=1000, max_lr=1e-3, exponent=0.8

with 300000 published iterations. That count does **not** fit a 100 s budget,
so `max_iters` is set to what does; `main` feeds it to the schedule's
`decay_steps`, so the run executes the same learning-rate schedule compressed
rather than being cut off part-way. (When a published count *does* fit, it is
used as published — see GUIDE.md rule 5 and the ldc case.)

Reading the score: predicting zero gives exactly 1.0, a per-point constant also
1.0, and freezing the initial condition 0.744.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = False
SOURCE = "reference/scale-pinn/kdv/plot_benchmark_kdv.ipynb, dispersion zeroed for inviscid Burgers"
PAPER_ARGS = dict(ER=0.1, ER_xx=20, weight_ic=1000, weight_bc=1000,
                  max_lr=1e-3, exponent=0.8)
PUBLISHED_MAX_ITER = 300000


def train(*, steps: int, seed: int, gpu: str) -> dict:
    sys.path.insert(0, str(HERE))
    try:
        import src as sp
        # the module pins CUDA_VISIBLE_DEVICES='0' at import and main() ignores its
        # `gpu` argument; JAX has not touched a device yet, so placement is set here
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        import jax.numpy as jnp
        params, _ = sp.main(seed=seed, gpu=int(gpu or 0), max_iter=steps, **PAPER_ARGS)
        net = sp.PINN(128)          # the authors' own class: 6 outputs, u first

        def predict(prm, X):
            """X is (N,2) as (t, x) — the order their network takes."""
            outs = [net.apply(prm, X[i:i + 4096]) for i in range(0, len(X), 4096)]
            return {"u": jnp.concatenate(outs, axis=0)[:, 0]}

        return {"params": params, "predict": predict,
                "info": {**PAPER_ARGS, "max_iter": steps,
                         "published_max_iter": PUBLISHED_MAX_ITER,
                         "donor_case": "kdv (Table A1 row 8)",
                         "v1": 1.0, "v2": 0.0,
                         "periodic": "u(t, x_l+2) = u(t, x_l); grid is cell-centred"}}
    finally:
        pass
