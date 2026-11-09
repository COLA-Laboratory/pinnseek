"""Scale-PINN on the backward-facing step — their method, ported from their cavity.

Scale-PINN publishes no backward-facing step, so NATIVE = False. `src.py` is
their `ns-ldc/plot_benchmark_pinn_ldc3200.ipynb` (extracted verbatim by
tools/extract_scale_pinn.py) with the physics, the geometry, the boundary
conditions and the input scaling changed; its docstring lists every edit and
each is marked `[sota patch]` in place.

Hyperparameters are the notebook's own live call — row 1a of the paper's
Table A1:

    ER=0.06 (tau_sc), ER_xx=1 (tau_alpha), weight_bc=15, max_lr=5e-4,
    exponent=0.8

with 50000 published iterations and batch 400 (`BS_ALL` in `main`).

`max_iters` is the **published 50000**, not a number sized to fill the clock.
`main` feeds it to the schedule's `decay_steps`, so stretching it past the
published value would hold the learning rate high for longer rather than train
longer at the published rate. 50000 iterations cost about 98 s of the 180 s
budget and the rest is deliberately left unspent; see GUIDE.md rule 5.

Only u and v are scored; the pressure is defined up to a constant, pinned here
by the task's outlet condition, and not part of the metric.

Reading the score: predicting zero gives exactly 1.0 and a per-field constant
0.667; the task's b00 anchor is 0.9975, which is near-trivial rather than a
real ceiling (the same baseline with rescaled inputs reaches 0.348).

**This recipe gets worse with more training on this geometry, and that is the
result rather than a defect of the port.** A 3000-iteration smoke of exactly
this file scores 0.387 (and its own internal RL2 print agrees with problem.py
to four digits, so the scoring path is sound), while the published 50000 lands
at 1.09-3.23 across seeds. Their schedule decays the learning rate over
`max_iters`, so a long run spends far more of it at the peak rate — on a cavity
that is fine, on a Re = 800 channel with a recirculation bubble it is not. An
earlier version of this file inflated `max_iters` to 99159 to fill the clock
and scored 1.17-1.73; that was a harness error, not a tuning knob, and the
number reported is the published schedule's. Nothing here is tuned to fix the
underlying behaviour: the point of the comparison is what the published recipe
does with the seconds it is given.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = False
SOURCE = "reference/scale-pinn/ns-ldc/plot_benchmark_pinn_ldc3200.ipynb, re-pointed at BFS"
PAPER_ARGS = dict(ER=0.06, ER_xx=1, weight_bc=15, max_lr=5e-4, exponent=0.8)
PUBLISHED_MAX_ITERS = 50000
SX, SY = 2.0 / 15.0, 2.0


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
        out = sp.main(seed=seed, gpu=int(gpu or 0), max_iters=steps, **PAPER_ARGS)
        params = out[0]
        net = sp.DNN(32)          # their own evaluation net: u, v, p, lap_u, lap_v

        def predict(prm, X):
            """X is (N,2) as (x, y); the network sees the rescaled coordinates."""
            Xs = jnp.stack([(X[:, 0] - 0.0) * SX - 1.0, X[:, 1] * SY], axis=1)
            outs = [net.apply(prm, Xs[i:i + 4096]) for i in range(0, len(Xs), 4096)]
            o = jnp.concatenate(outs, axis=0)
            return {"u": o[:, 0], "v": o[:, 1]}

        return {"params": params, "predict": predict,
                "info": {**PAPER_ARGS, "max_iters": steps,
                         "published_max_iters": PUBLISHED_MAX_ITERS,
                         "donor_case": "ns-ldc (Table A1 row 1a)",
                         "Re": 800, "steady": True,
                         "input_rescale": "x,y mapped to [-1,1]; residual carries the factors"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
