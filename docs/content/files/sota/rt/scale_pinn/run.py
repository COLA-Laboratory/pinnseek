"""Scale-PINN on Rayleigh-Taylor — their Gray-Scott recipe, widened to Boussinesq.

Scale-PINN publishes no buoyancy-driven flow and no unsteady Navier-Stokes with
a temperature. Its only 2D + time example is `gs`, and that is the donor: same
input shape (t, x, y), same sin-feature network with the skip structure, same
minibatch composition, same ER correction, same schedule. `src.py` is that
notebook extracted verbatim by tools/extract_scale_pinn.py and then changed in
twelve marked places. NATIVE = False.

What changed:

  * **four fields instead of two.** Gray-Scott carries (u, v); this task carries
    (u, v, p, T), so there are four output heads.
  * **the equations.** The task's four, Boussinesq at Pr = 0.71, Ra = 1e6:
        u_t + u u_x + v u_y + p_x - NU lap(u)      = 0
        v_t + u v_x + v v_y + p_y - NU lap(v) - T  = 0
        u_x + v_y                                   = 0
        T_t + u T_x + v T_y - KAPPA lap(T)          = 0
    with NU = sqrt(Pr/Ra) = 8.426150e-4 and KAPPA = 1/sqrt(Pr*Ra) = 1.186782e-3,
    the values problem.md states. The buoyancy term `-T` is what couples the
    temperature back into the momentum equation.
  * **the ER correction is applied per evolution equation**, each with its own
    diffusivity in the high-order half (the paper's S_Mu / S_Mv with
    gamma = the problem's own diffusivity), and the continuity residual carries
    the paper's S_Cn = (p^k - p^(k-1)) / tau_sc. Both donors implement S_Cn and
    the paper introduces it specifically "to explicitly provide a relation
    between pressure and the continuity equation, thereby improving
    convergence".
  * **two boundary kinds.** The donor's domain is periodic in both directions;
    here y = 0 and y = 2 are no-slip, no-flux walls (u = v = T = 0) and only x
    is periodic, with period 1.
  * **the domain comes from problem.md, not from the data file.** The donor
    read its domain off its own grid, which ran to the domain edge. This
    reference is cell-centred and stretched — x runs 0.0015386..0.9984614 and
    y runs 0.0015178..1.9984822 — so neither wall nor either periodic face is
    sampled at all. Taking min/max would (a) pair two x columns one cell short
    of a period and (b) pin u = v = T = 0 on the near-wall cell centres, where
    the true field still reaches |u| 0.047 and |T| 0.028. So `[0,1] x [0,2]` is
    used and both boundary sets are built analytically. The ib and kf cases hit
    the same cell-centred trap; spinn and dsgnar put the rt walls at y = 0 and
    y = 2 too.
  * the segment-stitching machinery and the results-file writing are removed;
    this task is solved in one window.

Hyperparameters are the donor's own, from the paper's Table A1 row for
Gray-Scott and the notebook's live call:

    ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000, max_lr=2e-3, exponent=1

with 300000 published iterations, which does not fit the budget, so `max_iters`
is set to what does; `main` feeds it to the schedule's `decay_steps`, so the
same schedule runs compressed rather than being cut off part-way.

The initial condition is the task's own t = 0 fields for u, v and T; the
pressure is not supplied by the task and is not scored, so it is left free.

Only T is scored, so `predict` returns the key "T".

What the two corrections above bought, seeds 1-5 at the same 143000 steps and
the same wall clock: mean 0.5478 (0.4611-0.6570) before, 0.4967 (0.4303-0.5784)
after — both ends of the spread improved.

Reading the score: predicting zero gives exactly 1.0 and the best possible
constant 0.740.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = False
SOURCE = "reference/scale-pinn/gs/plot_benchmark_gs.ipynb, widened to Boussinesq Rayleigh-Taylor"
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
    net = sp.DNN(128, 0.0, 4.75)     # their plain forward net: u, v, p, T, laps

    def predict(prm, X):
        """X is (N,3) as (t, x, y). Only T is scored, so only T is returned."""
        outs = [net.apply(prm, X[i:i + 8192]) for i in range(0, len(X), 8192)]
        return {"T": jnp.concatenate(outs, axis=0)[:, 3]}

    return {"params": params, "predict": predict,
            "info": {**PAPER_ARGS, "max_iter": steps,
                     "published_max_iter": PUBLISHED_MAX_ITER,
                     "donor_case": "gs (their only 2D+time example)",
                     "NU": sp.NU, "KAPPA": sp.KAPPA, "Pr": sp.PR, "Ra": sp.RA,
                     "bc": "walls u=v=T=0 at y=0,2; periodic in x with period 1",
                     "domain": "problem.md's [0,1] x [0,2]; the cell-centred grid's "
                               "min/max samples neither wall nor either periodic face",
                     "s_cn": "the paper's (p^k - p^(k-1))/tau_sc on the continuity residual",
                     "scored_field": "T only"}}
