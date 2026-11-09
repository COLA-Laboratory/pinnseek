"""Scale-PINN on KS — the authors' own implementation of this problem.

`src.py` is `reference/scale-pinn/ks/plot_benchmark_ks.ipynb` concatenated
verbatim up to its first top-level `main(...)` call (tools/extract_scale_pinn.py;
in this notebook every call is commented out, so the boundary is the first cell
that shows one). Nothing is patched.

Same problem, checked against the task rather than assumed:

  * coefficients — `src.py` sets v1 = 100/16, v2 = 100/16**2, v3 = 100/16**4,
    the task's nu_1, nu_2, nu_3;
  * time window — `src.py` subsets its data with `T = 0.4`, exactly the task's
    [0, 0.4]; space is x in [0, 2*pi] from the data;
  * reference — their `ks_chaotic.mat` is byte-identical to APTS's copy, and the
    task's ref_data.csv is its first 101 frames (max |diff| 5e-10, CSV rounding);
  * initial condition — the data's first frame is cos(x)(1+sin(x)) to 2e-15, and
    the task's ks_ic.csv is that same profile to 1e-9.

`main()` is called with the paper's own arguments for this case, which are the
arguments of the (commented-out) call in the notebook itself and of row 6 of the
paper's Table A1 (`ER`/`ER_xx` are the table's tau_sc/tau_alpha = 0.2/1.5, the
weights are its 500/5000, the learning rate its 1e-3; the table's batch size
1,000 is `BS_ALL` inside `main`):

    ER=0.2, ER_xx=1.5, weight_ic=500, weight_bc=5000, max_lr=1e-3, exponent=0.9

Only `max_iter` is set from outside, to the number of iterations that fills the
case's wall-clock budget. That is not a departure from the recipe: `main` sets
`max_iters = max_iter` and feeds it straight to the schedule's `decay_steps`, so
a shorter run executes the *same* learning-rate schedule compressed into the
time available rather than being cut off in the middle of it. The published
value is 200000 iterations.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = True
SOURCE = "reference/scale-pinn/ks/plot_benchmark_ks.ipynb"
PAPER_ARGS = dict(ER=0.2, ER_xx=1.5, weight_ic=500, weight_bc=5000,
                  max_lr=1e-3, exponent=0.9)
PUBLISHED_MAX_ITER = 200000


def train(*, steps: int, seed: int, gpu: str) -> dict:
    sys.path.insert(0, str(HERE))
    cwd = os.getcwd()
    os.chdir("~/reference/scale-pinn/ks")   # the notebook reads its data by relative path
    try:
        import src as sp
        # the module pins CUDA_VISIBLE_DEVICES='0' at import and main() ignores its
        # `gpu` argument; JAX has not touched a device yet, so placement is set here
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        import jax.numpy as jnp
        params, _ = sp.main(seed=seed, gpu=int(gpu or 0), max_iter=steps, **PAPER_ARGS)
        net = sp.PINN(128)          # the authors' own class: 6 outputs, u first

        def predict(prm, X):
            outs = [net.apply(prm, X[i:i + 4096]) for i in range(0, len(X), 4096)]
            return {"u": jnp.concatenate(outs, axis=0)[:, 0]}

        return {"params": params, "predict": predict,
                "info": {**PAPER_ARGS, "max_iter": steps,
                         "published_max_iter": PUBLISHED_MAX_ITER}}
    finally:
        os.chdir(cwd)
