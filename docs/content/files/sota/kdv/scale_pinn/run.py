"""Scale-PINN on KdV — the authors' own implementation of this problem.

`src.py` is `reference/scale-pinn/kdv/plot_benchmark_kdv.ipynb` concatenated
verbatim up to its first top-level `main(...)` call (tools/extract_scale_pinn.py).
Nothing is patched.

`main()` is called with the paper's own arguments for this case, which are also
the arguments of the call in the notebook itself and of row 8 of the paper's
Table A1 (`ER`/`ER_xx` are the table's tau_sc/tau_alpha = 0.1/20, the weights are
its 1000/1000, the learning rate its 1e-3):

    ER=0.1, ER_xx=20, weight_ic=1000, weight_bc=1000, max_lr=1e-3, exponent=0.8

Only `max_iter` is set from outside, to the number of iterations that fills the
case's wall-clock budget. That is not a departure from the recipe: `src.py:229`
sets `max_iters = max_iter` and feeds it straight to the schedule's
`decay_steps`, so a shorter run executes the *same* learning-rate schedule
compressed into the time available rather than being cut off in the middle of
it. The published value is 300000 iterations.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = True
SOURCE = "reference/scale-pinn/kdv/plot_benchmark_kdv.ipynb"
PAPER_ARGS = dict(ER=0.1, ER_xx=20, weight_ic=1000, weight_bc=1000,
                  max_lr=1e-3, exponent=0.8)
PUBLISHED_MAX_ITER = 300000


def train(*, steps: int, seed: int, gpu: str) -> dict:
    sys.path.insert(0, str(HERE))
    cwd = os.getcwd()
    os.chdir("~/reference/scale-pinn/kdv")   # the notebook reads its data by relative path
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
