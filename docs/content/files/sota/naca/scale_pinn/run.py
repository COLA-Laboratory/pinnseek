"""Scale-PINN on the NACA0012 airfoil at Re = 1000 — their own published case.

Like ldc, this is a method/case pair where nothing had to be ported. `src.py`
is `ns-naca/plot_naca_1_re1000.ipynb`, extracted verbatim by
tools/extract_scale_pinn.py, and the task it solves is the task the notebook
was written for. NATIVE = True.

The reference is not merely equivalent, it is **the same file**: every one of
the 321201 rows and all six columns (x, y, u, v, p, phi) of
`naca/task/ref_data.csv` match
`ns-naca/naca0012_atk7_Re1000_801x401_ls.csv` exactly, max |diff| = 0. The
task's scoring protocol says as much — `worker_eval` calls it "matching
Scale-PINN" and takes the relative L2 of the velocity magnitude over the
near-airfoil window, which is the `l2_V` their own `eval_loss` reports.

One `[sota patch]` in `src.py`: the csv path made absolute. (The whitespace
separator is also rewritten to a raw string, which is the same Python string
and changes nothing.) No physics, geometry, boundary condition,
network, schedule or hyperparameter is touched.

Hyperparameters are the notebook's own live call (cell 7):

    ER=0.1, ER_xx=0.5, weight_bc=5, batch=4000, max_lr=5e-3, exponent=1.2,
    max_iter=50000, seed=30

Training uses no interior reference values. `eval_loss` is
`pde_loss + weight_bc*bc_loss`, and every boundary term is masked: freestream
u,v on the inflow/top/bottom, p = 0 on the outflow, and no-slip on the 400
analytic airfoil points from their own `sample_points_on_airfoil` (the same
NACA0012 at 7 degrees the task specifies). The csv contributes the collocation
mesh, the level set that removes the airfoil interior, and the boundary values;
its interior columns feed the notebook's progress prints only.

Prediction uses `model_0` (their `DNN`), the plain forward network — `model`
returns the residual stack alongside the fields.

Reading the score: predicting zero gives exactly 1.0, the uniform freestream
(u=1, v=0) gives 0.480, and the best possible constant 0.403. A PINN that
learns only the farfield condition sits at 0.48, so anything above ~0.40 means
the airfoil was never resolved.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = True
SOURCE = "reference/scale-pinn/ns-naca/plot_naca_1_re1000.ipynb (their own NACA0012 Re=1000 case)"
PAPER_ARGS = dict(ER=0.1, ER_xx=0.5, weight_bc=5, batch=4000,
                  max_lr=5e-3, exponent=1.2)
PUBLISHED_MAX_ITER = 50000


def train(*, steps: int, seed: int, gpu: str) -> dict:
    sys.path.insert(0, str(HERE))
    import src as sp
    # the module pins CUDA_VISIBLE_DEVICES='0' at import and main() ignores its
    # `gpu` argument; JAX has not touched a device yet, so placement is set here
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import jax.numpy as jnp
    params, _u, _v, _p = sp.main(seed=seed, gpu=int(gpu or 0), max_iter=steps, **PAPER_ARGS)
    net = sp.DNN(32)          # their evaluation net: u, v, p, lap_u, lap_v

    def predict(prm, X):
        """X is (N,2) as (x, y) — the network's own frame, no rescaling."""
        outs = [net.apply(prm, X[i:i + 8192]) for i in range(0, len(X), 8192)]
        o = jnp.concatenate(outs, axis=0)
        return {"u": o[:, 0], "v": o[:, 1]}

    return {"params": params, "predict": predict,
            "info": {**PAPER_ARGS, "max_iter": steps,
                     "published_max_iter": PUBLISHED_MAX_ITER,
                     "Re": 1000, "steady": True,
                     "airfoil_bc_points": 400,
                     "reference": "bit-identical to the task's ref_data.csv",
                     "labels_used": "boundary u,v and outflow p only (masked)"}}
