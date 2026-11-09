"""APTS on inviscid Burgers — the authors' own example, on the task's initial condition.

`main.py`, `train.py`, `models.py`, `utils.py` and `configs/` are
`reference/APTS/examples/inviscid_burgers/` verbatim except for a single
`[sota patch]` in `utils.py` (the data path); the shared `jaxpi` library is
imported from the upstream repository. The run uses `configs/pseudo_time.py`,
which *is* APTS — the `strategy="dynamic"` pseudo-time stepping the paper
contributes. `fixed_pseudo_time.py` is the paper's PTS control and
`baseline.py` turns both off, so either of those would report the control as
the SOTA. NATIVE = True.

Same problem, checked against the task rather than assumed:

  * equation — `models.py:r_net` is `u_t + u * u_x`, the task's conservation
    law with no viscosity, written exactly that way;
  * domain — `train.py` builds `dom` from the data's own `t_star`/`x_star`
    endpoints: t in [0, 1.5] and the task's 101 time levels to the last bit.
    The task's x grid is **cell-centred**, so those endpoints are
    +-0.998046875 rather than +-1 and the residual box is half a cell short of
    the domain at each end. That is their own code's behaviour on their own
    data (theirs is 500 cell centres) and it costs nothing here, because the
    periodic boundary is imposed by the architecture rather than by sampling —
    see the next item;
  * periodic boundary — `configs/base.py` sets
    `arch.periodicity = {"period": (pi,), "axis": (1,)}` and jaxpi's
    `PeriodEmbs` embeds `cos(period*x), sin(period*x)`, so `period = pi` is a
    *spatial* period of 2, i.e. exactly periodic on [-1, 1]. The task's BC is
    imposed in the architecture, not as a penalty;
  * **initial condition — this is the one thing that changes.** Their shipped
    `data/inviscid_burger.npy` is the same equation on the same domain from a
    decaying bump (their `utils.py` comment calls it the Riemann problem); the
    task starts from `-sin(pi x)` and its shock is stationary at x = 0 from
    t = 1/pi. The IC is *data* in their code — `train.py` takes
    `u0 = u_ref[0, :]` and evaluates the IC loss on `x_star` — so
    `make_dataset.py` writes the task's reference in their dict format and the
    patch points `get_dataset` at it. Nothing about the method moves: network,
    embedding, losses, causal weighting, optimiser and schedule are all theirs.
    Checked before writing: the task reference's t = 0 row equals `ib_ic.csv`
    bit for bit, and its x grid equals the IC file's.

Their `u_ref` also feeds `compute_l2_error`, their own progress readout, so
this substitution makes that number meaningful instead of misleading.

Settings are the repository's own (`configs/base.py`): ModifiedMlp, 3 layers of
256 with a Fourier embedding, tanh, SOAP, batch 4096, causal weighting with 16
chunks, 100000 published steps.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config (2000 each)
rather than tied to the step count, so simply lowering `max_steps` would leave
the method still in warm-up when its time ran out. They are scaled by the same
factor as the step count. The two *weight-update* cadences are left at their
published values, as on the other cases.

Reading the score: predicting zero gives exactly 1.0, a per-field constant also
1.0, and freezing the initial condition 0.744 — so anything near 1.0 means the
run produced nothing, and 0.744 is what doing nothing but copying the IC buys.

Set from outside, none of it part of the method: the seed, the step count, the
lr-schedule scaling above, `WANDB_MODE=disabled`, and the working directory.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
METHOD_LIBS = "apts"
NATIVE = True
SOURCE = "reference/APTS/examples/inviscid_burgers (their own case; IC data swapped for the task's)"
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
    (work / "data_task.npy").symlink_to(HERE / "data_task.npy")
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
        # Logging is not the method: their evaluator predicts on the whole
        # reference grid every `log_every_steps`, so its cost would land inside
        # the budget while producing nothing the comparison uses.
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
        model = models.InviscidBurgers(cfg, lr, tx, arch, state,
                                       u_ref[0, :], t_star, x_star)

        def predict(prm, X):
            """X is (N,2) as (t, x) — the same order the network takes."""
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
                         "causal_chunks": int(cfg.causal.num_chunks),
                         "periodicity": "cos/sin(pi x) -> spatial period 2 on [-1,1]",
                         "ic_source": "task ref_data t=0 row (= ib_ic.csv)"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
