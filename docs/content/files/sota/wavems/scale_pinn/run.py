"""Scale-PINN on the multiscale wave equation — their Gray-Scott recipe, retargeted.

Scale-PINN publishes no wave case. Its only 2D + time example is `gs`
(`gs/plot_benchmark_gs.ipynb`), and that is the donor: same input shape
(t, x, y), same sin-feature network with the skip structure, same minibatch
composition, same ER correction, same schedule. `src.py` is that notebook
extracted verbatim by tools/extract_scale_pinn.py and then changed in twelve
marked places. NATIVE = False, and the changes are substantive enough that this
is a port rather than a retarget:

  * **one field instead of two.** Gray-Scott carries (u, v); this task carries
    u alone, so the second output head and every paired term go.
  * **second order in time.** Their residual is `u_t - eps*lap(u) - ...`; this
    one is `u_tt - c(x,y)*lap(u)`. The t derivative is taken twice, with the
    same `jacfwd` nesting their spatial derivatives already use.
  * **a variable coefficient.** c(x,y) takes 1 and 12 on the task's 256x256
    table, and `problem.md` specifies bilinear interpolation between grid
    points, so `c_at` interpolates. It appears both in the residual and in the
    high-order half of their ER correction, where the donor has its constant
    diffusivity.
  * **two initial terms.** The task prescribes u(0,.) *and* u_t(0,.) = 0. A
    first-order donor has only the first. Both are carried at the same weight,
    as the donor carries its single one.
  * **zero Neumann instead of periodic.** Their BC term matches opposite faces;
    this one is the squared normal derivative on the four sides.
  * the segment-stitching machinery (solve [T_0, T_1], hand the last slice to
    the next segment as an IC) is removed — this task is solved in one window —
    and so is the results-file writing their batch loop needs.

Hyperparameters are the donor's own, from the paper's Table A1 row for
Gray-Scott and from the notebook's live call — **except `ER`, which is set to
0**:

    ER=0 (theirs is 0.015), ER_xx=10, weight_ic=5000, weight_bc=1000,
    max_lr=2e-3, exponent=1

with 300000 published iterations.

**Why ER = 0, and how it was decided.** ER is Scale-PINN's headline
contribution: the residual gains `(u - u_0)/ER`, a proximal term against the
previous iterate. In every case they publish, the residual contains `u_t`, and
that term reads as a pseudo-time increment. This task's residual is `u_tt`;
there is no first derivative for it to correspond to, and at their published
ER = 0.015 the coefficient is 1/ER ~= 67. Measured at 30000 iterations, seed 1:

    ER = 0.015  ->  rRMSE 70.88   (diverging; 2.03 at 3000, 20.6 at 30000)
    ER = 0      ->  rRMSE 0.888   (converging, below the 1.0 trivial floor)

So the divergence is the ER term specifically, not the network, the schedule or
the loss. **ER = 0 is the authors' own control arm**, not something invented
here: `main()` guards the whole correction with `if (ER > 0)`, and their ns-ldc
notebook runs exactly this configuration as its reference (`# 5 runs (ref)`,
`ER=0`). Reporting the ER arm would say "Scale-PINN diverges on waves" when
what it actually shows is that a first-order construction was carried onto a
second-order operator; reporting the control arm says what their network and
schedule do with the seconds they are given, which is the question this
comparison asks. Both numbers are recorded in `info`.

A second change was needed before either arm was usable: **the time features
are rescaled to [-1,1]** (see `src.py`). Their sin feature multiplies the first
layer by 2*pi, and the donor's t window is 0.78 long while this task's is 5, so
a raw `t - t_l` reached ~31 rad; at 30000 iterations the unscaled ER arm hit
2.7e5. Derivatives are still taken with respect to physical t, so the PDE is
unchanged and no chain-rule factor is involved. That does not fit a 100 s budget, so
`max_iters` is set to what does; `main` feeds it to the schedule's
`decay_steps`, so the same learning-rate schedule runs compressed rather than
being cut off part-way.

**The trap in this case is that u == 0 satisfies the equation and both Neumann
conditions exactly.** Only the initial terms rule it out, and the donor's
`weight_ic = 5000` against an unweighted PDE term is on the right side of that
— which is why the donor's weights are kept rather than retuned.

Reading the score: predicting zero gives exactly 1.0, the best constant 0.781,
and freezing the initial condition 1.274.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = False
SOURCE = "reference/scale-pinn/gs/plot_benchmark_gs.ipynb, retargeted to the 2D wave"
PAPER_ARGS = dict(ER=0.0, ER_xx=10, weight_ic=5000, weight_bc=1000,
                  max_lr=2e-3, exponent=1)     # ER=0: their control arm, see the docstring
DONOR_ER = 0.015                                # their Gray-Scott value, measured below
ER_ARM_MEASURED = {"ER=0.015": 70.88, "ER=0": 0.888, "steps": 30000, "seed": 1}
PUBLISHED_MAX_ITER = 300000


def train(*, steps: int, seed: int, gpu: str) -> dict:
    sys.path.insert(0, str(HERE))
    import src as sp
    # the module pins CUDA_VISIBLE_DEVICES='0' at import and main() ignores its
    # `gpu` argument; JAX has not touched a device yet, so placement is set here
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import jax.numpy as jnp
    params, _ = sp.main(seed=seed, gpu=int(gpu or 0), max_iter=steps, **PAPER_ARGS)
    net = sp.DNN(128, 0.0, 5.0)      # their plain forward net: u, lap_u

    def predict(prm, X):
        """X is (N,3) as (t, x, y) — the order their network takes."""
        outs = [net.apply(prm, X[i:i + 8192]) for i in range(0, len(X), 8192)]
        return {"u": jnp.concatenate(outs, axis=0)[:, 0]}

    return {"params": params, "predict": predict,
            "info": {**PAPER_ARGS, "max_iter": steps,
                     "published_max_iter": PUBLISHED_MAX_ITER,
                     "donor_case": "gs (their only 2D+time example)",
                     "donor_ER": DONOR_ER, "er_arm_measured": ER_ARM_MEASURED,
                     "er_note": "ER=0 is the authors' own control arm (ns-ldc '# 5 runs (ref)'); "
                                "their ER correction is a first-order pseudo-time term and "
                                "diverges on this second-order operator",
                     "coefficient": "task 256x256 c(x,y), bilinear",
                     "bc": "zero Neumann on all four sides",
                     "ic": "u(0,.) and u_t(0,.)=0, both at weight_ic"}}
