"""Scale-PINN on Grey-Scott — the authors' own implementation of this problem.

`src.py` is `reference/scale-pinn/gs/plot_benchmark_gs.ipynb` concatenated
verbatim up to its first top-level `main(...)` call (tools/extract_scale_pinn.py).
Nothing is patched.

Same problem, checked against the task rather than assumed:

  * coefficients — `main` sets b1,b2 = 40,100; c1,c2 = 1000,1000;
    ep1,ep2 = 0.2,0.1, and the residual it forms is
    u_t - ep1*(u_xx+u_yy) - b1*(1-u) + c1*u*v^2 and
    v_t - ep2*(v_xx+v_yy) + b2*v - c2*u*v^2 — the task's two equations;
  * data — their `grey_scott.mat` carries those six constants and is the same
    200x200x101 ETDRK4 solution APTS ships; the task's ref_data.csv is its
    first 40 frames subsampled [::2,::2] (max |diff| 5e-11, CSV rounding);
  * space — [-1,1]^2, periodic, enforced by matching u and v at x=+-1 and
    y=+-1 (their `pbcx`/`pbcy` terms);
  * labels — only the t=0 slice enters the loss (`ic` mask). The interior
    reference values ride along in the batch but multiply out of every loss
    term; they are used solely for the progress print.

**Two things are set from outside, both recorded here.**

1. *The time window.* `main` takes `T_0`/`T_1`; the published run is
   T_1 = 0.5 (their notebook's live call, cell 4), and the task's window is
   [0, 0.78]. T_1 is set to 0.78 — the same knob their own sequential
   time-window machinery (`pred_T`) drives. Their subsetting is `t <= T_1` on
   the .mat's own dt = 0.02 grid, so this selects exactly the task's 40
   snapshots. Note this makes the window 56% wider than the one they
   published, in one shot rather than chained; running two chained windows
   instead would have required inventing a budget split they never published.

2. *`max_iter`*, to the number of iterations that fills the case's wall-clock
   budget. Not a departure from the recipe: `main` sets `max_iters = max_iter`
   and feeds it to the schedule's `decay_steps`, so a shorter run executes the
   *same* learning-rate schedule compressed into the time available rather
   than being cut off part-way. The published value is 300000.

The remaining arguments are the notebook's own live call (cell 4, the one that
produced the weights its cells 8-9 report):

    ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000, max_lr=2e-3, exponent=1

These match row 7 of the paper's Table A1 (lambda_ic 5000, lambda_bc 1000,
lr 2e-3, 300k iterations, batch 1,000 = `BS_ALL`, tau_alpha 10) except for
tau_sc, which the table gives as 0.02 and the notebook runs at 0.015. The
notebook's value is used: it is the one that actually produced their result.

**Axis convention — the one real trap in this case.** `main` pairs the mesh
point (x[i], y[j]) with `usol[t].reshape(-1)[i*200+j] = usol[t][i,j]`, but the
.mat stores `usol` as [t, y, x]: `usol[0].T` matches the paper's initial
condition 1-exp(-10((x+0.05)^2+(y+0.02)^2)) to 1e-4 while `usol[0]` itself is
off by 1.1e-1, and against the task grid the transposed reading agrees to
5e-11 and the untransposed one to 0.69. So their `x` is the task's `y`.
Because the Laplacian is symmetric and the domain is a periodic square, a
model trained in their labelling is *exactly* a solution of the task's problem
read with the two spatial arguments swapped — nothing about the training is
wrong or needs changing, and `predict` below swaps them. (APTS's example reads
the file the same way and gets the same treatment.)

Verified on the trained weights, not just on the data: scoring
`weights/scale_pinn_s1.pkl` and `_s2.pkl` with the swap gives 0.0155 and
0.0307, and without it 0.1617 and 0.1659 — the x<->y mirror floor (0.164)
that the reference's own mirror predicts. Reading the axes the other way
would have inflated these methods' error roughly tenfold.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
GS_DIR = Path("~/reference/scale-pinn/gs")
METHOD_LIBS = "scale-pinn"
NATIVE = True
SOURCE = "reference/scale-pinn/gs/plot_benchmark_gs.ipynb (T_1 0.5 -> the task's 0.78)"
PAPER_ARGS = dict(ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000,
                  max_lr=2e-3, exponent=1)
PUBLISHED_MAX_ITER = 300000
PUBLISHED_T1 = 0.5
TASK_T1 = 0.78


def train(*, steps: int, seed: int, gpu: str) -> dict:
    sys.path.insert(0, str(HERE))
    # a scratch cwd, not the reference directory: the notebook reads its data by
    # relative path *and* appends a per-run log file, and the reference tree stays
    # read-only
    work = HERE.parent.parent / "runs" / f"scale_pinn_s{seed}_{steps}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "grey_scott.mat").symlink_to(GS_DIR / "grey_scott.mat")
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
        # the authors' own evaluation net (notebook cells 7-9): DNN shares PINN's
        # parameter tree and returns [u, v, lap_u, lap_v] without the residual
        net = sp.DNN(128, 0.0, TASK_T1)

        def predict(prm, X):
            """X is (N,3) as (t, x, y); their model's x/y are the transpose of
            the task's, so the two spatial columns are swapped on the way in."""
            Xs = jnp.stack([X[:, 0], X[:, 2], X[:, 1]], axis=1)
            outs = [net.apply(prm, Xs[i:i + 4096]) for i in range(0, len(Xs), 4096)]
            out = jnp.concatenate(outs, axis=0)
            return {"u": out[:, 0], "v": out[:, 1]}

        return {"params": params, "predict": predict,
                "info": {**PAPER_ARGS, "max_iter": steps,
                         "published_max_iter": PUBLISHED_MAX_ITER,
                         "T_0": 0.0, "T_1": TASK_T1,
                         "published_T_1": PUBLISHED_T1,
                         "paper_tau_sc": 0.02, "used_tau_sc": PAPER_ARGS["ER"],
                         "xy_swapped_at_predict": True}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
