"""APTS on KdV — the authors' own implementation of this problem.

`main.py`, `train.py`, `models.py`, `utils.py` and `configs/` are
`reference/APTS/examples/kdv/` verbatim; the shared `jaxpi` library is imported
from the upstream repository. The run uses `configs/pseudo_time.py`, which *is*
APTS — the `strategy="dynamic"` pseudo-time stepping the paper contributes.
`fixed_pseudo_time.py` is the paper's PTS control and `baseline.py` turns both
off, so either of those would report the control as the SOTA.

Settings are the repository's own (`configs/base.py`): PirateNet, swish,
RFF 2.0, batch 2048, 50000 published steps. Note that these disagree with the
paper's own Table for KdV, which gives Tanh, batch 4,096 and 1e5 iterations —
see NOTES.md. The source's settings are what is used here.

**The schedule is compressed to the budget, not cut off.** `warmup_steps` and
`decay_steps` are absolute in this config (2000 each) rather than tied to the
step count, so simply lowering `max_steps` would leave the method still in
warm-up when its time ran out — at this budget roughly 60% of the run. They are
scaled by the same factor as the step count so the method executes its whole
learning-rate schedule inside the time available. That is the treatment that
flatters the method; the alternative would be reporting a number about warm-up.

Set from outside, none of it part of the method: the seed, the step count and
the schedule scaling above, `WANDB_MODE=disabled`, and the working directory.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
EXAMPLE = APTS / "examples" / "kdv"
METHOD_LIBS = "apts"
NATIVE = True
SOURCE = "reference/APTS/examples/kdv"
PUBLISHED_MAX_STEPS = 50000


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
    (work / "data").symlink_to(EXAMPLE / "data")
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
        cfg.seed = seed
        scale = steps / PUBLISHED_MAX_STEPS
        orig = dict(max_steps=int(cfg.training.max_steps),
                    warmup_steps=int(cfg.optim.warmup_steps),
                    decay_steps=int(cfg.optim.decay_steps))
        cfg.training.max_steps = steps
        cfg.optim.warmup_steps = max(int(round(orig["warmup_steps"] * scale)), 1)
        cfg.optim.decay_steps = max(int(round(orig["decay_steps"] * scale)), 1)
        cfg.saving.save_every_steps = steps          # one save, at the end
        # Logging is not the method. `train.py` calls its evaluator every
        # `log_every_steps`, and that evaluator predicts on the whole reference
        # grid — so its compilation and its per-call cost land inside the budget
        # while producing nothing the comparison uses (W&B is already disabled).
        # Measured on this case: switching it off cuts the fixed cost from 78.5s
        # to 67.7s, which at a 100s budget is the difference between 621 and 802
        # training steps. It does not touch `model.state`.
        cfg.logging.log_every_steps = steps + 10      # fires once, at step 0

        apts_train.train_and_evaluate(cfg)

        lr = create_lr_schedule(cfg.optim)
        tx = create_optimizer(cfg.optim, lr)
        arch = create_arch(cfg.arch)
        state = create_train_state(cfg, tx, arch)
        mngr = create_checkpoint_manager(
            cfg.saving, os.path.join(os.getcwd(), cfg.wandb.name, "ckpt"))
        state = restore_checkpoint(mngr, state)

        u_ref, t_star, x_star = get_dataset()
        model = models.KDV(cfg, lr, tx, arch, state, u_ref[0, :], t_star, x_star)

        def predict(prm, X):
            outs = [model.sol_pred_fn(prm, X[i:i + 8192, 0], X[i:i + 8192, 1])
                    for i in range(0, len(X), 8192)]
            return {"u": jnp.concatenate(outs, axis=0)}

        return {"params": state.params, "predict": predict,
                "info": {"config": "configs/pseudo_time.py", "arch": str(cfg.arch.arch_name),
                         "batch_size": int(cfg.training.batch_size),
                         "steps": steps, "published_max_steps": PUBLISHED_MAX_STEPS,
                         "schedule_scaled_by": round(scale, 5),
                         "warmup_steps": int(cfg.optim.warmup_steps),
                         "decay_steps": int(cfg.optim.decay_steps),
                         "published_schedule": orig}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
