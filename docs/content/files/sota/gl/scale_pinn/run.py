"""Scale-PINN on Ginzburg-Landau — their method, ported from their Grey-Scott case.

Scale-PINN publishes no Ginzburg-Landau benchmark, so NATIVE = False.
`src.py` is their `gs/plot_benchmark_gs.ipynb` (extracted verbatim by
tools/extract_scale_pinn.py) with four marked `[sota patch]` edits — the
residual, the coefficients, the ER_xx gamma and the data file. Its docstring
lists them; the diff against `cases/gs/methods/scale_pinn/src.py` is exactly
those four hunks.

Grey-Scott is the right donor: of their eight published cases it is the only
one that is 2D + time with two coupled fields on a periodic [-1,1]^2 square,
which is this problem's shape exactly. So the architecture, the feature map
(2*pi frequency annealing), the batch construction and the loss all transfer
without a judgement call, and only the physics changes.

Hyperparameters are their Grey-Scott row for the same reason (Table A1 row 7,
which is also the notebook's own live call):

    ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000, max_lr=2e-3, exponent=1

(the table gives tau_sc = 0.02 where the notebook runs 0.015; the notebook's
value is used, as in the gs case). The published iteration count is 300000.

Set from outside: `T_1 = 0.79`, the task's final time — the same `main()`
argument their own sequential time-window machinery drives — and `max_iter`,
sized by bench to fill the wall-clock budget. The latter is not a departure
from the recipe: `main` feeds `max_iter` straight to the schedule's
`decay_steps`, so a shorter run executes the *same* learning-rate schedule
compressed into the time available rather than being cut off part-way.

Axis convention: checked, and **no swap is needed here** (unlike the
Grey-Scott case). `main` pairs mesh point (x_i, y_j) with
`usol[t].reshape(-1)[i*200+j] = usol[t][i,j]`, and for this .mat that is the
right reading — the initial condition u0 = 10 x exp(-25 r^2) is asymmetric in
x <-> y and matches `usol[0]` to 2.8e-10 while the transpose is off by 1.2.

Reading the score: A == 0 is an exact zero-residual solution of this equation
with a large basin, and the trivial floor is flat at 1.0 (predicting zero *or*
the per-field mean both give exactly 1.0). Anything at or above ~0.99 means the
run produced nothing.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
GL_MAT = Path("~/reference/APTS/examples/ginzburg_landau/data/"
              "ginzburg_landau_square.mat")
METHOD_LIBS = "scale-pinn"
NATIVE = False
SOURCE = "reference/scale-pinn/gs/plot_benchmark_gs.ipynb, residual replaced by GL's"
PAPER_ARGS = dict(ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000,
                  max_lr=2e-3, exponent=1)
PUBLISHED_MAX_ITER = 300000
TASK_T1 = 0.79


def train(*, steps: int, seed: int, gpu: str) -> dict:
    sys.path.insert(0, str(HERE))
    # a scratch cwd: src.py reads its data by relative path and appends a
    # per-run log file, and neither reference tree is written to
    work = HERE.parent.parent / "runs" / f"scale_pinn_s{seed}_{steps}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / GL_MAT.name).symlink_to(GL_MAT)
    cwd = os.getcwd()
    os.chdir(work)
    try:
        import src as sp
        # the module pins CUDA_VISIBLE_DEVICES='0' at import and main() ignores its
        # `gpu` argument; JAX has not touched a device yet, so placement is set here
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        import jax.numpy as jnp
        params, _, _ = sp.main(seed=seed, gpu=int(gpu or 0), max_iter=steps,
                               T_0=0.0, T_1=TASK_T1, pred_T=[], **PAPER_ARGS)
        # the authors' own evaluation net (their notebook cells 7-9): DNN shares
        # PINN's parameter tree and returns [u, v, lap_u, lap_v]
        net = sp.DNN(128, 0.0, TASK_T1)

        def predict(prm, X):
            """X is (N,3) as (t, x, y) — their axes, no swap needed here."""
            outs = [net.apply(prm, X[i:i + 4096]) for i in range(0, len(X), 4096)]
            out = jnp.concatenate(outs, axis=0)
            return {"u": out[:, 0], "v": out[:, 1]}

        return {"params": params, "predict": predict,
                "info": {**PAPER_ARGS, "max_iter": steps,
                         "published_max_iter": PUBLISHED_MAX_ITER,
                         "T_0": 0.0, "T_1": TASK_T1,
                         "donor_case": "gs (Table A1 row 7)",
                         "eps": 0.004, "k": 10.0, "c": 1.5,
                         "xy_swapped_at_predict": False}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
