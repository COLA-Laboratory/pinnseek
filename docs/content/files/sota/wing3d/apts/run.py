"""APTS on steady 3D flow over a finite wing.

The donor is `reference/APTS/examples/bfs_flow/` — their own **steady**
wall-bounded incompressible flow with an inlet, a `p = 0` outlet and no-slip
walls, built on `ForwardBVP`, the base this task needs (no time coordinate, no
initial condition). `main.py` is theirs verbatim; `train.py`, `models.py`,
`utils.py` and `configs/base.py` are theirs with the changes marked
`[sota patch]` in each file. The shared `jaxpi` library is imported from the
upstream repository unmodified.

The run uses `configs/pseudo_time.py`, which *is* APTS — the
`strategy="dynamic"` pseudo-time stepping the paper contributes.
`fixed_pseudo_time.py` is the paper's PTS control and `baseline.py` turns both
off, so either of those would report the control as the SOTA.

NATIVE = False. APTS publishes nothing three-dimensional and nothing around a
body.

What this problem needs that the backward-facing step does not have:

  * **three input coordinates and four fields.** `config.input_dim = 3`,
    `arch.out_dim = 4`; `jaxpi` unpacks the batch generically, so nothing in
    the library changes. The residual gains its z-momentum equation and the
    `w_z` term in the divergence.
  * **a body in the flow.** The wing is not analytic: it comes as a
    signed-distance level set (`task/collocation.npy`, already filtered to
    `phi <= 0`) plus a surface file (`task/wing_surface.npy`). The residual
    sampler draws from the level-set cloud, so no masking is needed — the
    shipped points are all fluid. Their `MeshSampler` over a point cloud is
    exactly the right sampler for that and is used unchanged.
  * **a symmetry plane.** `z = 0` carries `w = 0` and `du/dz = dv/dz = 0`. It
    is the only condition here that is not a plain value match, and nothing in
    the donor resembles it.
  * **a freestream inlet.** Theirs is a parabolic profile; here the inlet, the
    two lateral faces and the top all carry `u = 1, v = w = 0`, so three
    boundary sets collapse into one term.

Settings are the repository's own (`configs/base.py`): ModifiedMlp, out_dim
raised to 4, batch 2048, 100000 published steps, every boundary weighted 100
against 1 on the residual.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config (2000 each) rather
than tied to the step count, so simply lowering `max_steps` would leave the
method still in warm-up when its time ran out. They are scaled by the same
factor as the step count. The two *weight-update* cadences are left at their
published values, as on the other cases.

Training reads the two geometry files; `ref_data.npy` is read only for the
evaluator's `l2_error` logging, and logging fires once, at step 0, during a benchmarked
run (`log_every_steps` is set past the end).

Reading the score: predicting zero gives exactly 1.0 and the freestream
`u = 1, v = w = 0` everywhere gives 0.2116 — that is the anchor to beat.
`problem.md` states an **error floor of 0.0071**, so a score below about 0.01 is
fitting discretisation error rather than physics.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
METHOD_LIBS = "apts"
NATIVE = False
SOURCE = "reference/APTS/examples/bfs_flow (their steady wall-bounded flow), lifted to 3D"
PUBLISHED_MAX_STEPS = 100000


def train(*, steps: int, seed: int, gpu: str) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"    # from the authors' main.py
    os.environ["WANDB_MODE"] = "disabled"         # logging only
    sys.path[:0] = [str(APTS), str(HERE)]

    import jax, jax.numpy as jnp
    jax.config.update("jax_default_matmul_precision", "highest")   # from main.py

    work = HERE.parent.parent / "runs" / f"apts_s{seed}_{steps}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cwd = os.getcwd()
    os.chdir(work)
    try:
        import train as apts_train, models
        from configs.pseudo_time import get_config

        cfg = get_config()
        cfg.seed = seed
        scale = steps / PUBLISHED_MAX_STEPS
        orig = dict(max_steps=int(cfg.training.max_steps),
                    warmup_steps=int(cfg.optim.warmup_steps),
                    decay_steps=int(cfg.optim.decay_steps))
        cfg.training.max_steps = steps
        cfg.optim.warmup_steps = max(int(round(orig["warmup_steps"] * scale)), 1)
        cfg.optim.decay_steps = max(int(round(orig["decay_steps"] * scale)), 1)
        cfg.saving.save_every_steps = 0               # the model is returned directly
        # Logging is not the method: their evaluator predicts on the reference
        # every `log_every_steps`, so its cost would land inside the budget
        # while producing nothing the comparison uses.
        cfg.logging.log_every_steps = steps + 10      # fires once, at step 0

        model = apts_train.train_and_evaluate(cfg)

        uvwp_fn = jax.jit(jax.vmap(model.neural_net, (None, 0, 0, 0)))

        def predict(prm, X):
            """X is (N,3) as (x, y, z) — steady, no time column. The scorer
            forms the velocity magnitude from the three components."""
            us, vs, ws = [], [], []
            for i in range(0, len(X), 16384):
                b = X[i:i + 16384]
                u, v, w, _ = uvwp_fn(prm, b[:, 0], b[:, 1], b[:, 2])
                us.append(u); vs.append(v); ws.append(w)
            return {"u": jnp.concatenate(us), "v": jnp.concatenate(vs),
                    "w": jnp.concatenate(ws)}

        return {"params": model.state.params, "predict": predict,
                "info": {"config": "configs/pseudo_time.py",
                         "arch": str(cfg.arch.arch_name),
                         "batch_size": int(cfg.training.batch_size),
                         "steps": steps,
                         "published_max_steps": PUBLISHED_MAX_STEPS,
                         "schedule_scaled_by": round(scale, 6),
                         "warmup_steps": int(cfg.optim.warmup_steps),
                         "decay_steps": int(cfg.optim.decay_steps),
                         "published_schedule": orig,
                         "steady": True, "input_dim": 3, "out_dim": 4,
                         "nu": 1e-3, "Re": 1000.0,
                         "loss_keys": ["far", "noslip", "p_out", "sym",
                                       "ru", "rv", "rw", "rc"],
                         "bc": ("farfield u=1 on inlet/lateral/top, wing no-slip, "
                                "outlet p=0, symmetry plane w=0 and du/dz=dv/dz=0"),
                         "geometry": "shipped: collocation.npy level set + wing_surface.npy",
                         "residual_mask": "none needed: the shipped collocation cloud is already fluid-only",
                         "error_floor": 0.0071}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
