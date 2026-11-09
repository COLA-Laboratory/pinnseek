"""Scale-PINN on the 3D screened Poisson problem in a cube minus two spheres.

`src.py` is `ns-ldc/plot_benchmark_pinn_ldc3200.ipynb` — Scale-PINN's only
steady example, and so the only one whose loop has no time axis to remove —
extracted the same way every other Scale-PINN port here is, then changed in the
seventeen places marked `[sota patch]`. See that file's docstring for each
change and its source. NATIVE = False.

What is theirs and untouched: the `sin(2 pi * Dense(...))` feature map, the
shared trunk with a split head, the `he_uniform` init, the `BS_ALL=400` /
`BS_BC=50` minibatch composition, the `(u - u_0)/ER` pseudo-time correction
with its separate `ER_xx` term on the diffusion part, the warmup-cosine
schedule, and the paired `PINN`/`DNN` design where the second net is the plain
forward evaluator used for prediction.

What is not theirs: three inputs instead of two, one field instead of three,
the screened-Poisson residual, homogeneous Neumann on eight boundaries instead
of Dirichlet on four sides, and an analytic collocation set — this task ships
no data at all beyond the scoring truth, so there is no csv grid to borrow.

The ER correction is kept on. This is a steady problem, which is the setting
their pseudo-time term was formulated for; the `wavems` port had to switch it
off for the opposite reason.

Hyperparameters are the donor notebook's own live call (cell 4, the five-seed
loop):

    ER=0.06, ER_xx=1, weight_bc=15, max_lr=5e-4, exponent=0.8, n_nodes=32

with 50000 published iterations. `max_iters` is also `decay_steps` of the
cosine schedule, so the step count is compressed to what fits the budget rather
than the schedule being cut off part-way.

Reading the score: predicting zero gives exactly 1.0 and the best constant
0.929 — the field has a nonzero mean, so the two floors differ.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = False
SOURCE = "reference/scale-pinn/ns-ldc/plot_benchmark_pinn_ldc3200.ipynb (their only steady case), retargeted"
PAPER_ARGS = dict(ER=0.06, ER_xx=1, weight_bc=15, max_lr=5e-4, exponent=0.8)
PUBLISHED_MAX_ITERS = 50000


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
        net = sp.DNN(32)          # their evaluation net: u, m

        def predict(prm, X):
            """X is (N,3) as (x, y, z) — steady, no time column. The network
            reads the first three columns, so the reference array goes in as
            it is."""
            outs = [net.apply(prm, X[i:i + 8192]) for i in range(0, len(X), 8192)]
            return {"u": jnp.concatenate(outs, axis=0)[:, 0]}

        return {"params": params, "predict": predict,
                "info": {**PAPER_ARGS, "n_nodes": 32, "max_iters": steps,
                         "published_max_iters": PUBLISHED_MAX_ITERS,
                         "donor_case": "ns-ldc (their only steady example)",
                         "steady": True, "K_lo": sp.K_LO, "K_hi": sp.K_HI,
                         "z_interface": sp.Z_IFACE,
                         "spheres": [list(s) for s in sp.SPHERES],
                         "bc": "homogeneous Neumann on 6 faces + 2 sphere surfaces",
                         "collocation": "analytic 46^3 open lattice minus the spheres",
                         "bs_all": 400, "bs_bc": 50}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
