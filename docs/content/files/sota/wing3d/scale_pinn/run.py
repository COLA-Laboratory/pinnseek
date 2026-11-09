"""Scale-PINN on steady 3D flow over a finite wing.

`src.py` is their `ns-fps` notebook — Scale-PINN's own steady flow past bodies
immersed in a channel — extracted by tools/extract_scale_pinn.py and then
changed in the eleven places marked `[sota patch]`. See that file's docstring
for each change and its source. NATIVE = False.

The donor is unusually close for a 3D case: `ns-fps` already reads a
signed-distance level set, selects the fluid with `phi <= 0`, appends the
body-surface points at the end of the minibatch so the residual can drop them,
and carries an inlet, a `p = 0` outlet and wall conditions. This task uses
**the same level-set convention** — `task/collocation.npy` is `(x, y, z, phi)`
with `phi <= 0` in the fluid — so the whole immersed-body machinery transfers
directly.

What is theirs and untouched: the `sin(pi * Dense(...))` feature map over
body-relative coordinates, the shared trunk with per-field split heads, the
`he_uniform` init, the `(u - u_0)/ER` pseudo-time correction with its separate
`ER_xx` term on the diffusion part, the boundary-mask construction, the
body-points-last batch layout, the warmup-cosine schedule, and the paired
`PINN`/`DNN` design where the second net supplies the previous iterate.

What is not theirs: a third spatial dimension and a fourth field, the wing in
place of three squares, a symmetry plane at `z = 0` (`w = 0`,
`du/dz = dv/dz = 0`), freestream lateral and top faces in place of channel
walls, and `Re = 1000` in place of 25.

Hyperparameters are the ns-fps notebook's own live call (cell 6):

    ER=0.035, ER_xx=15, weight_bc=5, batch=4000, max_lr=5e-3, exponent=1.3

with 80000 published iterations. `max_iter` is also `decay_steps` of the cosine
schedule, so the step count is compressed to what fits the budget rather than
the schedule being cut off part-way.

Training reads `collocation.npy` and `wing_surface.npy` and nothing else;
`ref_data.npy`, the scoring truth, is never opened.

Reading the score: predicting zero gives exactly 1.0 and the freestream
`u = 1, v = w = 0` everywhere gives 0.2116 — that is the anchor to beat.
`problem.md` states an **error floor of 0.0071**, so a score below about 0.01 is
fitting discretisation error rather than physics.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
METHOD_LIBS = "scale-pinn"
NATIVE = False
SOURCE = "reference/scale-pinn/ns-fps (their immersed-body flow, same level-set convention), lifted to 3D"
PAPER_ARGS = dict(ER=0.035, ER_xx=15, weight_bc=5, batch=4000,
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
        net = sp.DNN(32)          # their evaluation net: u, v, w, p, mom_u/v/w

        def predict(prm, X):
            """X is (N,3) as (x, y, z) — steady, no time column. The scorer
            forms the velocity magnitude from the three components."""
            outs = [net.apply(prm, X[i:i + 8192]) for i in range(0, len(X), 8192)]
            o = jnp.concatenate(outs, axis=0)
            return {"u": o[:, 0], "v": o[:, 1], "w": o[:, 2]}

        return {"params": params, "predict": predict,
                "info": {**PAPER_ARGS, "n_nodes": 32, "max_iter": steps,
                         "published_max_iter": PUBLISHED_MAX_ITER,
                         "donor_case": "ns-fps (immersed bodies, phi <= 0 level set)",
                         "steady": True, "nu": sp.NU, "Re": sp.RE,
                         "bc": ("farfield u=1 on inlet/lateral/top, wing no-slip, "
                                "outlet p=0, symmetry plane w=0 and du/dz=dv/dz=0"),
                         "geometry": "shipped: collocation.npy level set + wing_surface.npy",
                         "training_data": "the two geometry files only; ref_data.npy is never read",
                         "error_floor": 0.0071}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
