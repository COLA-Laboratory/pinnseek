"""Scale-PINN on the lid-driven cavity at Re = 3200 — their own published case.

This is the one method/case pair in the comparison where nothing had to be
ported. `src.py` is `ns-ldc/plot_benchmark_pinn_ldc3200.ipynb`, extracted
verbatim by tools/extract_scale_pinn.py, and the task it solves is the task the
notebook was written for: the unit square, Re = 3200, lid u = 1 on the top edge
and no-slip on the other three. NATIVE = True.

Three `[sota patch]` marks exist in `src.py` and all three are file paths made
absolute so the module can be imported from anywhere — their own reference csv
and their own two Ghia cut files. No physics, geometry, boundary condition,
network, schedule or hyperparameter is touched.

Hyperparameters are the notebook's own live call (cell 4, the five-seed loop):

    ER=0.06, ER_xx=1, weight_bc=15, max_lr=5e-4, exponent=0.8,
    max_iters=50000, seeds [90, 10, 70, 30, 50]

50000 iterations cost about 90 s and the budget is 150 s, so the published
count fits and is what runs — `used_published_count` in the calibration file.
Filling the remaining seconds by inflating `max_iters` would make the result
worse, not better: `max_iters` is also `decay_steps` of the cosine schedule, so
a longer run holds the learning rate high for longer. That was measured on this
exact case (85k steps degraded all five seeds, mean 5.5e-2 -> 9.4e-2), and the
comparison reports the published recipe rather than a stretched one.

Training uses no interior reference values. `eval_loss` is
`pde_loss + weight_bc*bc_loss`, and `bc_loss` is masked to the boundary, so the
csv contributes only the boundary values (verified: lid u = 1 exactly, other
walls 0 to 1e-19) plus the collocation grid. The interior columns feed the
notebook's own progress prints only.

Their csv is a 102x102 solve and the task's reference is 256x256 from a
different solver; interpolating one onto the other agrees to 2.5e-2 in relative
L2 over the interior, which is discretisation difference, not a different
problem.

Prediction uses `model_0` (their `DNN`), which is the plain forward network the
notebook itself uses for its final error report — `model` returns the residual
stack alongside the fields.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = True
SOURCE = "reference/scale-pinn/ns-ldc/plot_benchmark_pinn_ldc3200.ipynb (their own LDC Re=3200 case)"
PAPER_ARGS = dict(ER=0.06, ER_xx=1, weight_bc=15, max_lr=5e-4, exponent=0.8)
PUBLISHED_MAX_ITERS = 50000
PAPER_SEEDS = [90, 10, 70, 30, 50]


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
        params, _ = sp.main(seed=seed, gpu=int(gpu or 0), max_iters=steps, **PAPER_ARGS)
        net = sp.DNN(32)          # their evaluation net: u, v, p, lap_u, lap_v

        def predict(prm, X):
            """X is (N,2) as (x, y) on the unit square — the network's own frame."""
            outs = [net.apply(prm, X[i:i + 8192]) for i in range(0, len(X), 8192)]
            o = jnp.concatenate(outs, axis=0)
            return {"u": o[:, 0], "v": o[:, 1]}

        return {"params": params, "predict": predict,
                "info": {**PAPER_ARGS, "max_iters": steps,
                         "published_max_iters": PUBLISHED_MAX_ITERS,
                         "paper_seeds": PAPER_SEEDS,
                         "Re": 3200, "steady": True,
                         "collocation": "their 102x102 csv grid, corners excluded",
                         "labels_used": "boundary u,v only (bc_loss is masked)"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
