"""Scale-PINN on 3D unsteady Navier-Stokes in vorticity-velocity form.

`src.py` is their Gray-Scott notebook — Scale-PINN's only 2D + time example and
the donor the `rt` and `heatlt` ports here use as well — extracted by
tools/extract_scale_pinn.py and then changed in the fifteen places marked
`[sota patch]`. See that file's docstring for each change and its source.
NATIVE = False: Scale-PINN publishes nothing 3-dimensional in space and nothing
in vorticity-velocity form.

What is theirs and untouched: the sin-feature trunk with the skip-concat
(`2 pi` on the first layer, every odd layer's output concatenated), the
`t - t_l` / `t - t_u` time framing, the `he_uniform` init, the per-field linear
heads without bias, the `BS_ALL=1000` / `BS_IC=50` / `BS_BC=50` minibatch
composition, the `(u - u_0)/ER` pseudo-time correction with its separate
`ER_xx` term on the diffusion part, the warmup-cosine schedule, and the paired
`PINN`/`DNN` design where the second net supplies the previous iterate.

What is not theirs: four inputs instead of three, the vorticity-velocity
Navier-Stokes residual with its third derivatives, a third periodic direction,
and collocation drawn from the analytic box.

**Training reads only `task/ns4d_ic.csv`.** The scoring truth is never opened —
not even for collocation coordinates, which the donor would have taken from its
data file.

Hyperparameters are the donor notebook's own live call:

    ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000, max_lr=2e-3, exponent=1

with 300000 published iterations, which does not fit the budget, so `max_iter`
is set to what does; `main` feeds it to the schedule's `decay_steps`, so the
same schedule runs compressed rather than being cut off part-way.

Prediction returns the vorticity, which `DNN` emits directly as the autodiff
curl of its velocity — the task requires the curl, not a network output.

Reading the score: predicting zero gives exactly 1.0 and freezing the initial
condition 1.342.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = False
SOURCE = "reference/scale-pinn/gs/plot_benchmark_gs.ipynb (their only 2D+time case), widened to 3D+time"
PAPER_ARGS = dict(ER=0.015, ER_xx=10, weight_ic=5000, weight_bc=1000,
                  max_lr=2e-3, exponent=1)
PUBLISHED_MAX_ITER = 300000


def train(*, steps: int, seed: int, gpu: str) -> dict:
    sys.path.insert(0, str(HERE))
    work = HERE.parent.parent / "runs" / f"scale_pinn_s{seed}_{steps}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cwd = os.getcwd()
    os.chdir(work)
    try:
        import src as sp
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        import jax.numpy as jnp
        params, _ = sp.main(seed=seed, gpu=int(gpu or 0), max_iter=steps, **PAPER_ARGS)
        net = sp.DNN(128, 0.0, sp.T_MAX)   # ux, uy, uz, wx, wy, wz, lwx, lwy, lwz

        def predict(prm, X):
            """X is (N,4) as (t, x, y, z) — the order the network takes.
            Columns 3, 4 and 5 of `DNN` are the vorticity."""
            outs = [net.apply(prm, X[i:i + 4096]) for i in range(0, len(X), 4096)]
            o = jnp.concatenate(outs, axis=0)
            return {"wx": o[:, 3], "wy": o[:, 4], "wz": o[:, 5]}

        return {"params": params, "predict": predict,
                "info": {**PAPER_ARGS, "n_nodes": 128, "max_iter": steps,
                         "published_max_iter": PUBLISHED_MAX_ITER,
                         "donor_case": "gs (their only 2D+time example)",
                         "nu": sp.NU, "t_max": sp.T_MAX, "period": float(sp.L),
                         "bs_all": 1000, "bs_ic": 50, "bs_bc": 50,
                         "bc": "periodic in x, y and z, matched face to face",
                         "ic": "task/ns4d_ic.csv velocity on its 32^3 grid",
                         "training_data": "the initial condition only; ref_data.csv is never read",
                         "vorticity": "autodiff curl of the velocity; lap(w) is a third derivative"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
