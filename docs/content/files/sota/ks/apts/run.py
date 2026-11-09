"""APTS on KS — the authors' own implementation of this problem.

`main.py`, `train.py`, `models.py`, `utils.py` and `configs/` are
`reference/APTS/examples/ks/` verbatim; the shared `jaxpi` library is imported
from the upstream repository. The run uses `configs/pseudo_time.py`, which *is*
APTS — the `strategy="dynamic"` pseudo-time stepping the paper contributes.
`fixed_pseudo_time.py` is the paper's PTS control and `baseline.py` turns both
off, so either of those would report the control as the SOTA.

Same problem, checked against the task rather than assumed:

  * residual — `models.py:r_net` is u_t + 100/16 u u_x + 100/16**2 u_xx +
    100/16**4 u_xxxx, the task's nu_1, nu_2, nu_3;
  * reference — their `data/ks_chaotic.mat` is byte-identical to Scale-PINN's
    copy, and the task's ref_data.csv is its first 101 frames (max |diff| 5e-10,
    CSV rounding). Its first frame is cos(x)(1+sin(x)) to 2e-15, which is the
    task's ks_ic.csv to 1e-9;
  * space — x in [0, 2*pi] from that data, with the architecture's
    `periodicity` period 1.0 on axis 1, i.e. cos(x)/sin(x) features
    (`jaxpi/archs.py:PeriodEmbs` multiplies, so period *is* the wavenumber) —
    the task's 2*pi-periodic boundary, enforced by construction.

**One config field is set to the task's time window.** The repository ships
`config.time_range = [0.0, 0.35]`, and `utils.get_dataset` slices the 251-frame
reference by `int(time_range[1] * 251)` — 87 frames, t <= 0.344. The task is
t in [0, 0.4], so the field is set to 101/251, which is that same expression's
input for the 101 frames ending exactly at t = 0.4. Nothing else about the
problem changes; `time_range` is a knob their own code exposes for precisely
this. It matters because `models.py:neural_net` normalises t by `t_star[-1]`,
so a model trained to 0.344 would be extrapolating over the last seventh of
the scoring grid.

Settings are otherwise the repository's own (`configs/base.py`): PirateNet,
tanh, RFF 2.0, batch 4096, causal weighting in 16 chunks, 100000 published
steps.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config (2000 each) rather
than tied to the step count, so simply lowering `max_steps` would leave the
method still in warm-up when its time ran out. They are scaled by the same
factor as the step count so the method executes its whole schedule inside the
time available. The two *weight-update* cadences
(`loss_weighting.update_schedule`, `pseudo_time.update_schedule`) are left at
their published values, as on kdv: they are the method's own adaptation
triggers rather than a schedule that would strand the run, and at this budget
they fire at their `start` step and then not again.

Set from outside, none of it part of the method: the seed, the step count, the
lr-schedule scaling and the time window above, `WANDB_MODE=disabled`, and the
working directory.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
EXAMPLE = APTS / "examples" / "ks"
METHOD_LIBS = "apts"
NATIVE = True
SOURCE = "reference/APTS/examples/ks (config.time_range set to the task's t<=0.4)"
PUBLISHED_MAX_STEPS = 100000
N_REF_FRAMES = 251          # frames in ks_chaotic.mat
TASK_FRAMES = 101           # frames of it that end at t = 0.4


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
        published_time_range = list(cfg.time_range)
        cfg.time_range = [0.0, TASK_FRAMES / N_REF_FRAMES]
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
        # It does not touch `model.state`.
        cfg.logging.log_every_steps = steps + 10      # fires once, at step 0

        apts_train.train_and_evaluate(cfg)

        lr = create_lr_schedule(cfg.optim)
        tx = create_optimizer(cfg.optim, lr)
        arch = create_arch(cfg.arch)
        state = create_train_state(cfg, tx, arch)
        mngr = create_checkpoint_manager(
            cfg.saving, os.path.join(os.getcwd(), cfg.wandb.name, "ckpt"),
            suffix="time_window_1")
        state = restore_checkpoint(mngr, state)

        u_ref, t_star, x_star = get_dataset(time_range=cfg.time_range)
        model = models.KS(cfg, lr, tx, arch, state, u_ref[0, :], t_star, x_star)

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
                         "published_schedule": orig,
                         "time_range": list(cfg.time_range),
                         "published_time_range": published_time_range,
                         "t_max_trained": float(t_star[-1])}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
