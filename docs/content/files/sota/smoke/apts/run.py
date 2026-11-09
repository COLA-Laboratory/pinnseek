"""APTS on wildfire-smoke transport — their advection recipe, in 2D with real coefficient fields.

APTS publishes an advection case, `examples/advection`, and it is the right
donor by structure: a first-order transport equation with a causally-weighted
residual and an initial-condition term, which is what this task is. It is 1D
with a constant speed and a periodic boundary; this task is 2D with wind,
divergence and emission fields from ERA5/GFAS, plus diffusion and Dirichlet
sides. NATIVE = False.

`main.py` and `configs/` are `reference/APTS/examples/advection/` verbatim;
`models.py` and `train.py` carry `[sota patch]` marks; `utils.py` is rewritten
for this task's data. The run uses `configs/pseudo_time.py`, which *is* APTS.

What changed, and only this:

  * `models.py` — `Advection1D` becomes `Smoke2D`: the network takes (t, x, y),
    the residual is the task's
    `u_t + a u_x + b u_y + c u - DXC u_xx - DYC u_yy - f`, the IC term is
    `u(0,.) = 0` on a 64x64 lattice (the atmosphere starts clean — the source f
    builds the whole plume), and a Dirichlet term for `u = 0` on the four sides
    is added because the periodic donor has no boundary loss. The loss *shape*
    and the class it derives from are untouched.
  * `train.py` — the domain box gains a third axis; the dataset comes from the
    rewritten loader.
  * `config.input_dim` 2 -> 3, and **`arch.periodicity` is switched off**. The
    donor is periodic in x; this task is Dirichlet on all four sides, so a
    periodic embedding would impose a condition the problem does not have.

**The coefficient lookup is the task's own.** `utils.coef_at` reproduces
`task/baseline.py:coef_at` exactly — trilinear (`order=1`, as `problem.md`
specifies) with the `(t, y, x)` storage order the task explicitly warns about,
while the coordinates are `(t, x, y)`. `cases/smoke/methods/verify_coef.py`
checks ours against the task's function on 4096 random points: max |diff| is
0.000 for all four fields, while the transposed order differs by up to 57. That
mistake would run silently and swap the wind components, so it is checked rather
than assumed.

Settings are otherwise the repository's own (`configs/base.py`): PirateNet,
3 layers of 256 with a Fourier embedding, SOAP, batch 4096, causal weighting
with 16 chunks, 100000 published steps.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute (2000 each), so lowering
`max_steps` alone would leave the method in warm-up when its time ran out. They
are scaled by the same factor as the step count.

Reading the score: predicting zero gives exactly 1.0 and the best possible
constant 0.972 — the field is 98.9% empty sky, so a method has to find the plume
to score below about 0.97 at all.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
METHOD_LIBS = "apts"
NATIVE = False
SOURCE = "reference/APTS/examples/advection, extended to 2D with ERA5/GFAS coefficient fields"
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
        from jaxpi.models import (create_lr_schedule, create_optimizer,
                                  create_arch, create_train_state)
        from jaxpi.checkpointing import create_checkpoint_manager, restore_checkpoint
        from utils import get_dataset

        cfg = get_config()
        cfg.input_dim = 3                 # [sota patch] (t, x, y)
        cfg.arch.periodicity = None       # [sota patch] Dirichlet, not periodic
        cfg.seed = seed
        scale = steps / PUBLISHED_MAX_STEPS
        orig = dict(max_steps=int(cfg.training.max_steps),
                    warmup_steps=int(cfg.optim.warmup_steps),
                    decay_steps=int(cfg.optim.decay_steps))
        cfg.training.max_steps = steps
        cfg.optim.warmup_steps = max(int(round(orig["warmup_steps"] * scale)), 1)
        cfg.optim.decay_steps = max(int(round(orig["decay_steps"] * scale)), 1)
        cfg.saving.save_every_steps = steps
        # Logging is not the method: their evaluator predicts on the whole
        # 37x121x77 reference grid, so its cost would land inside the budget.
        cfg.logging.log_every_steps = steps + 10

        apts_train.train_and_evaluate(cfg)

        lr = create_lr_schedule(cfg.optim)
        tx = create_optimizer(cfg.optim, lr)
        arch = create_arch(cfg.arch)
        state = create_train_state(cfg, tx, arch)
        mngr = create_checkpoint_manager(
            cfg.saving, os.path.join(os.getcwd(), cfg.wandb.name, "ckpt"))
        state = restore_checkpoint(mngr, state)

        d = get_dataset()
        model = models.Smoke2D(cfg, lr, tx, arch, state, d)
        u_fn = jax.jit(jax.vmap(model.neural_net, (None, 0, 0, 0)))

        def predict(prm, X):
            """X is (N,3) as (t, x, y) — the order the network takes."""
            outs = [u_fn(prm, X[i:i + 16384, 0], X[i:i + 16384, 1], X[i:i + 16384, 2])
                    for i in range(0, len(X), 16384)]
            return {"u": jnp.concatenate(outs, axis=0)}

        return {"params": state.params, "predict": predict,
                "info": {"config": "configs/pseudo_time.py", "arch": str(cfg.arch.arch_name),
                         "batch_size": int(cfg.training.batch_size),
                         "steps": steps, "published_max_steps": PUBLISHED_MAX_STEPS,
                         "schedule_scaled_by": round(scale, 5),
                         "warmup_steps": int(cfg.optim.warmup_steps),
                         "decay_steps": int(cfg.optim.decay_steps),
                         "published_schedule": orig,
                         "causal_chunks": int(cfg.causal.num_chunks),
                         "donor_case": "advection (1D, constant c, periodic)",
                         "periodicity": "off (task is Dirichlet u=0 on all four sides)",
                         "coefficients": "task/baseline.py:coef_at, trilinear, (t,y,x) storage",
                         "dxc": d["dxc"], "dyc": d["dyc"], "input_dim": 3}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
