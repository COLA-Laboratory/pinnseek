"""Scale-PINN on Kolmogorov flow — their method, ported from their Grey-Scott case.

Scale-PINN publishes no Kolmogorov-flow benchmark, so NATIVE = False.
`src.py` is their `gs/plot_benchmark_gs.ipynb` (extracted verbatim by
tools/extract_scale_pinn.py) with the physics replaced; its docstring explains
why Grey-Scott is the code donor and Rayleigh-Benard the hyperparameter donor,
and marks every change `[sota patch]`.

Hyperparameters are Table A1 row 5 (Rayleigh-Benard), the paper's only
time-dependent Navier-Stokes row:

    ER=0.1 (tau_sc), ER_xx=1.5 (tau_alpha), weight_ic=1, weight_bc=10,
    max_lr=5e-3, exponent=1

with 50000 published iterations and batch 4,000 (`BS_ALL` in `main`).
`exponent` is not in the table; it is the schedule shape and 1 is the value
their Grey-Scott call uses, which is the code this is built on.

Set from outside: `T_1 = 0.75`, the task's final time, and `max_iter`, sized by
bench to fill the wall-clock budget — `main` feeds it straight to the
schedule's `decay_steps`, so a shorter run executes the *same* learning-rate
schedule compressed rather than being cut off part-way.

Axis convention: no swap. The collocation grid is built from the task's own
`ref_data.csv` rows, which carry (t, x, y) explicitly, so there is no
reshape to get wrong.

Periodicity: that grid is **cell-centred** (0.001953125 to 0.994140625), so its
max is one cell short of a period. The donor's grid included both endpoints, so
its `u(t, x_u, y) - u(t, x_l, y)` was the periodic condition as written; here
those two columns are not a period apart and the true solution violates that
pairing at rms 0.082 against a field rms of 0.763. `x_u`/`y_u` are therefore set
to `x_l + 1` / `y_l + 1`, the same patch the ib and rt ports carry. Measured
over seeds 1-5 at the same 26559 steps: mean 0.3719 (0.3328-0.3996) with the
data-derived pairing, 0.3734 (0.3658-0.3866) with the true periodicity — the
same number with a tighter spread, so this is a correctness fix to the record
rather than a change to the result.

Only u and v are scored; the pressure the network also outputs is not part of
the metric and is left unconstrained at t = 0 (it is defined up to a constant).

Reading the score: u = v = 0 scores exactly 1.0 on this task and the b00 anchor
is 1.1187 — *above* the trivial field. A score near 1.0 can mean collapse
rather than progress.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = False
SOURCE = "reference/scale-pinn/gs/plot_benchmark_gs.ipynb, physics replaced by KF's; Table A1 row 5"
PAPER_ARGS = dict(ER=0.1, ER_xx=1.5, weight_ic=1, weight_bc=10,
                  max_lr=5e-3, exponent=1)
PUBLISHED_MAX_ITER = 50000
TASK_T1 = 0.75


def train(*, steps: int, seed: int, gpu: str) -> dict:
    sys.path.insert(0, str(HERE))
    work = HERE.parent.parent / "runs" / f"scale_pinn_s{seed}_{steps}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cwd = os.getcwd()
    os.chdir(work)          # src.py appends a per-run log file to its cwd
    try:
        import src as sp
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        import jax.numpy as jnp
        params, _, _ = sp.main(seed=seed, gpu=int(gpu or 0), max_iter=steps,
                               T_0=0.0, T_1=TASK_T1, pred_T=[], **PAPER_ARGS)
        net = sp.DNN(128, 0.0, TASK_T1)     # their own evaluation net: u, v, p, lap_u, lap_v

        def predict(prm, X):
            outs = [net.apply(prm, X[i:i + 4096]) for i in range(0, len(X), 4096)]
            out = jnp.concatenate(outs, axis=0)
            return {"u": out[:, 0], "v": out[:, 1]}

        return {"params": params, "predict": predict,
                "info": {**PAPER_ARGS, "max_iter": steps,
                         "published_max_iter": PUBLISHED_MAX_ITER,
                         "T_0": 0.0, "T_1": TASK_T1,
                         "code_donor": "gs notebook", "hparam_donor": "Table A1 row 5 (ns-rb)",
                         "nu": 5e-4, "Re": 2000,
                         "body_force": "2 sin(4 pi y) on x-momentum",
                         "periodic_fix": "x_u = x_l + 1, y_u = y_l + 1 (grid is cell-centred)"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
