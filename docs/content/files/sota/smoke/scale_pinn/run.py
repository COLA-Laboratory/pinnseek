"""Scale-PINN on wildfire-smoke transport — their Gray-Scott recipe, retargeted.

Scale-PINN publishes no advection case and nothing with a variable coefficient.
Its only 2D + time example is `gs`, and that is the donor: same input shape
(t, x, y), same sin-feature network with the skip structure, same minibatch
composition, same ER correction, same schedule. `src.py` is that notebook
extracted verbatim by tools/extract_scale_pinn.py and then changed in thirteen
marked places. NATIVE = False.

What changed:

  * **one field instead of two.** Gray-Scott carries (u, v); this task carries
    u alone.
  * **the equation.** Theirs is `u_t - eps*lap(u) - reaction`; this one is
        u_t + a u_x + b u_y + c u - DXC u_xx - DYC u_yy - f
    with a (wind u), b (wind v), c (wind divergence) and f (emission flux) the
    task's ERA5/GFAS fields, and DXC/DYC the frozen constants from problem.md.
    It stays first order in time, like the donor.
  * **the coefficient lookup.** `coef_at` is `task/baseline.py:coef_at`
    reproduced exactly — trilinear (`order=1`, as problem.md specifies) with the
    **(t, y, x)** storage order the task explicitly warns about while the
    coordinates are (t, x, y). `cases/smoke/methods/verify_coef.py` checks this
    construction against the task's own function on 4096 random points: max
    |diff| 0.000, against up to 57 for the transposed order. That mistake runs
    silently and swaps the wind components, so it is checked, not assumed.
  * **zero initial condition and Dirichlet sides.** u(0,x,y) = 0 and u = 0 on
    all four faces, so both label sets are zeros — the source term f builds the
    entire plume. The donor's periodic face-matching is replaced by the
    Dirichlet trace.
  * the segment-stitching machinery and the results-file writing their batch
    loop needs are removed; this task is solved in one window.

The donor's feature map is kept **exactly** as published, `t - t_l` and
`t - t_u` unscaled: this task's window is [0,1] against their [0,0.78], so the
`sin(2*pi * ...)` feature sees the O(1) range they tuned it on. An earlier
version of this port rescaled the two time features to [-1,1], carrying over a
patch the `wavems` port needs (its window is 5, and the raw form was measured to
diverge there). Nothing here justified it, and it is a deviation from the recipe
being benchmarked, so it is gone. Both arms are measured over seeds 1-5 at
62000 steps:

    raw t (published form)  ->  mean 0.3161 (0.2961-0.3269)
    t rescaled to [-1,1]    ->  mean 0.3012 (0.2937-0.3136)

The rescale is worth about 5%, and that is exactly why it is recorded rather
than kept: it is a real improvement, but it is not Scale-PINN's, and this table
answers what the published recipe does with the seconds it is given. The
published form is reported; the rescaled arm is kept here and in `info` so the
difference is visible rather than buried.

Hyperparameters are the donor's own, from the paper's Table A1 row for
Gray-Scott and the notebook's live call:

    ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000, max_lr=2e-3, exponent=1

with 300000 published iterations, which does not fit the budget, so `max_iters`
is set to what does; `main` feeds it to the schedule's `decay_steps`, so the
same learning-rate schedule runs compressed rather than being cut off part-way.

Unlike the wavems port, the ER correction is kept at its published value here:
this task is first order in time, which is the setting their `(u - u_0)/ER`
pseudo-time term was formulated for.

Reading the score: predicting zero gives exactly 1.0 and the best possible
constant 0.972 — the field is 98.9% empty sky, so anything at or above ~0.97
means the plume was never found.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = False
SOURCE = "reference/scale-pinn/gs/plot_benchmark_gs.ipynb, retargeted to variable-coefficient advection-diffusion"
PAPER_ARGS = dict(ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000,
                  max_lr=2e-3, exponent=1)
PUBLISHED_MAX_ITER = 300000


def train(*, steps: int, seed: int, gpu: str) -> dict:
    sys.path.insert(0, str(HERE))
    import src as sp
    # the module pins CUDA_VISIBLE_DEVICES='0' at import and main() ignores its
    # `gpu` argument; JAX has not touched a device yet, so placement is set here
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import jax.numpy as jnp
    params, _ = sp.main(seed=seed, gpu=int(gpu or 0), max_iter=steps, **PAPER_ARGS)
    net = sp.DNN(128, 0.0, 1.0)      # their plain forward net: u, weighted lap

    def predict(prm, X):
        """X is (N,3) as (t, x, y) — the order their network takes."""
        outs = [net.apply(prm, X[i:i + 8192]) for i in range(0, len(X), 8192)]
        return {"u": jnp.concatenate(outs, axis=0)[:, 0]}

    return {"params": params, "predict": predict,
            "info": {**PAPER_ARGS, "max_iter": steps,
                     "published_max_iter": PUBLISHED_MAX_ITER,
                     "donor_case": "gs (their only 2D+time example)",
                     "dxc": sp.DXC, "dyc": sp.DYC,
                     "coefficients": "task/baseline.py:coef_at, trilinear, (t,y,x) storage",
                     "ic": "u(0,.) = 0", "bc": "zero Dirichlet on all four sides",
                     "time_features": "the donor's raw t - t_l / t - t_u, unscaled",
                     "t_rescaled_arm": {"mean_rRMSE": 0.3012, "min": 0.2937, "max": 0.3136,
                                        "steps": 62000, "seeds": "1-5",
                                        "note": "t mapped to [-1,1]; not used -- a [0,1] window "
                                                "needs no rescale and the donor does not do it"}}}
