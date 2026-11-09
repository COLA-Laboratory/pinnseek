"""Scale-PINN on the long-time nonlinear heat equation — their Gray-Scott recipe, retargeted.

Scale-PINN publishes no heat equation and nothing with a solution-dependent
source. Its only 2D + time example is `gs`, and that is the donor: same input
shape (t, x, y), same sin-feature network with the skip structure, same
minibatch composition, same ER correction, same schedule. `src.py` is that
notebook extracted verbatim and then changed in eleven marked places.
NATIVE = False.

What changed:

  * **one field instead of two.** Gray-Scott carries (u, v); this task carries u.
  * **the equation.** Theirs is `u_t - eps lap(u) - reaction`; this one is
        u_t - D lap(u) - A sin(u^2) (1 + 2 sin(pi t / 4)) sin(4 pi x) sin(2 pi y)
    with D = 0.001 and A = 5 from `problem.md`. It stays first order in time,
    like the donor.
  * **zero Dirichlet sides** in place of their periodic face matching, and the
    task's analytic `sin(4 pi x) sin(3 pi y)` as the initial condition.
  * **the domain comes from problem.md, not from the data file.** The donor
    read its domain off its own grid, whose points ran to the domain edge.
    This reference is a COMSOL mesh whose 16 x 12 nodes are *all interior*
    (x in 0.03125..0.96875, y in 0.041667..0.958333), so taking min/max would
    put the zero-Dirichlet trace on two interior lines — where the true field
    has rms 0.85 against a global rms of 1.066, on 27% of the scored points —
    and would shrink the collocation box to that rectangle. `[0,1]^2` is used
    and the boundary collocation is built analytically on the four real faces,
    which is where the hero and the other three baselines put it. Measured over
    seeds 1-5 at the same 80000 steps and the same wall clock: mean 0.4747
    (0.4710-0.4780) with the data-derived box, 0.2683 (0.2557-0.2844) with
    problem.md's.
  * the segment-stitching machinery and the results-file writing are removed.

The reference is a COMSOL mesh (501 snapshots on 16 x 12 = 192 nodes) rather
than a tensor grid. That is fine here: their minibatch draws rows and never
reshapes.

**The trap: u == 0 satisfies the PDE and all four sides exactly**, because the
source carries `sin(u^2)` and vanishes with u. The donor's `weight_ic = 5000`
against an unweighted PDE term is on the right side of that, which is why the
donor's weights are kept rather than retuned.

The ER correction is kept at its published value: this task is first order in
time, the setting their `(u - u_0)/ER` pseudo-time term was formulated for.
(The wavems port had to switch it off for exactly the opposite reason.)

Hyperparameters are the donor's own:

    ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000, max_lr=2e-3, exponent=1

with 300000 published iterations, which does not fit the budget, so `max_iters`
is set to what does; `main` feeds it to the schedule's `decay_steps`.

Reading the score: predicting zero gives exactly 1.0, and so does the best
possible constant, because the field is a zero-mean oscillation.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = False
SOURCE = "reference/scale-pinn/gs/plot_benchmark_gs.ipynb, retargeted to 2D nonlinear heat"
PAPER_ARGS = dict(ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000,
                  max_lr=2e-3, exponent=1)
PUBLISHED_MAX_ITER = 300000


def train(*, steps: int, seed: int, gpu: str) -> dict:
    sys.path.insert(0, str(HERE))
    import src as sp
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import jax.numpy as jnp
    params, _ = sp.main(seed=seed, gpu=int(gpu or 0), max_iter=steps, **PAPER_ARGS)
    net = sp.DNN(128, 0.0, 100.0)     # their plain forward net: u, lap_u

    def predict(prm, X):
        """X is (N,3) as (t, x, y) — the order their network takes."""
        outs = [net.apply(prm, X[i:i + 8192]) for i in range(0, len(X), 8192)]
        return {"u": jnp.concatenate(outs, axis=0)[:, 0]}

    return {"params": params, "predict": predict,
            "info": {**PAPER_ARGS, "max_iter": steps,
                     "published_max_iter": PUBLISHED_MAX_ITER,
                     "donor_case": "gs (their only 2D+time example)",
                     "D": sp.D_COEF, "A": sp.A_COEF, "t_max": 100.0,
                     "ic": "sin(4 pi x) sin(3 pi y)",
                     "bc": "zero Dirichlet on all four sides",
                     "domain": "problem.md's [0,1]^2; the COMSOL reference mesh is "
                               "entirely interior and does not sample any face"}}
