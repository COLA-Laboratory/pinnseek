"""Scale-PINN on the vortex street grown from rest.

**This is the `cyl` port with the ramped inflow, the longer window and the
at-rest initial condition**; the donors, the hyperparameters and the boundary
construction are identical, so the two rows can be read against each other.

`src.py` is their `ns-fps` notebook — Scale-PINN's own flow past obstacles in a
channel, with an inlet, a `p = 0` outlet and walls — extracted by
tools/extract_scale_pinn.py and then changed in the thirteen places marked
`[sota patch]`, with the time framing and the initial-condition term taken from
their `gs` notebook. See that file's docstring for each change and its source.
NATIVE = False: their case is steady and this one is not.

What is theirs and untouched: the `sin(pi * Dense(...))` feature map over
obstacle-relative coordinates, the shared trunk with three split heads, the
`he_uniform` init, the `(u - u_0)/ER` pseudo-time correction with its separate
`ER_xx` term on the diffusion part, the `uv_bc` / `p_bc` mask construction, the
obstacle points appended at the end of the batch so the residual can drop them,
the warmup-cosine schedule, and the paired `PINN`/`DNN` design.

What is not theirs: the time derivative, the initial-condition term, no-slip
walls in place of their `u_y = 0` symmetry condition, a driven time-dependent
inlet in place of their fixed profile, a circular obstacle in place of three
squares, and an analytic collocation set.

**Training reads nothing at all.** The initial condition, the geometry and
every boundary are analytic, and the scoring truth is never opened — not even
for collocation coordinates, which the donor took from its data file.

Hyperparameters are the ns-fps notebook's own live call (cell 6):

    ER=0.035, ER_xx=15, weight_bc=5, batch=4000, max_lr=5e-3, exponent=1.3

with 80000 published iterations. `weight_ic` is the one weight neither donor
supplies for this combination — ns-fps is steady — and is set to 5x
`weight_bc`, the ratio the gs notebook uses (5000 against 1000).

`max_iter` is also `decay_steps` of the cosine schedule, so the step count is
compressed to what fits the budget rather than the schedule being cut off
part-way.

Reading the score: the fluid starts at rest, so freezing the initial condition
*is* predicting zero and both score exactly 1.0. The best constant gives 0.585
and the inlet profile imposed everywhere 0.430 — the last is the anchor to
beat.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = False
SOURCE = "reference/scale-pinn/ns-fps (their flow past obstacles) + gs (the time axis)"
PAPER_ARGS = dict(ER=0.035, ER_xx=15, weight_bc=5, weight_ic=25, batch=4000,
                  max_lr=5e-3, exponent=1.3)
PUBLISHED_MAX_ITER = 80000


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
        net = sp.DNN(32)          # their evaluation net: u, v, p, mom1, mom2

        def predict(prm, X):
            """X is (N,3) as (t, x, y) — the order the network takes."""
            outs = [net.apply(prm, X[i:i + 8192]) for i in range(0, len(X), 8192)]
            o = jnp.concatenate(outs, axis=0)
            return {"u": o[:, 0], "v": o[:, 1]}

        return {"params": params, "predict": predict,
                "info": {**PAPER_ARGS, "n_nodes": 32, "max_iter": steps,
                         "published_max_iter": PUBLISHED_MAX_ITER,
                         "donor_case": "ns-fps (flow past obstacles) + gs (time axis)",
                         "nu": sp.NU, "t_max": sp.T_MAX,
                         "cylinder": [sp.CX, sp.CY, sp.CR],
                         "bc": "driven inlet + no-slip walls + no-slip cylinder + p = 0 outlet",
                         "outlet": "p = 0 is expressible here: the pressure is a network output",
                         "ic": "u = v = 0 (analytic: the fluid starts at rest)",
                         "derived_from": "cases/cyl/methods/scale_pinn; only U(t), T_MAX and the IC differ",
                         "training_data": "nothing is read: every condition is analytic"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
