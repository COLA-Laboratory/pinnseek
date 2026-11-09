"""APTS on the backward-facing step — the authors' own implementation of this problem.

`main.py`, `train.py`, `models.py`, `utils.py` and `configs/` are
`reference/APTS/examples/bfs_flow/` verbatim; the shared `jaxpi` library is
imported from the upstream repository. The run uses `configs/pseudo_time.py`,
which *is* APTS — the `strategy="dynamic"` pseudo-time stepping the paper
contributes. `fixed_pseudo_time.py` is the paper's PTS control and
`baseline.py` turns both off, so either of those would report the control as
the SOTA.

Same problem, checked against the task rather than assumed:

  * residual — `models.py:r_net` is u u_x + v u_y + p_x - nu(u_xx+u_yy),
    u v_x + v v_y + p_y - nu(v_xx+v_yy), and u_x + v_y: the task's three
    steady equations;
  * viscosity — `utils.py` sets nu = 1/800, the task's Re = 800;
  * inflow — `utils.py:inflow_profile` is `24 y (0.5 - y)` for y > 0 and 0
    below, exactly the task's step-face inlet, on x = 0;
  * walls and outlet — top/bottom no-slip at y = +-0.5, p = 0 at x = 15,
    matching the task's table;
  * reference — their `data/flow.vtu` **is** the task's reference: matching
    the 31265 points by coordinate, every one is present and |du| <= 5e-10,
    |dv| <= 5e-11 (CSV rounding). No axis convention to resolve — `train.py`
    uses the file's own `points` array as its collocation mesh rather than
    rebuilding a grid.

This case is **steady**: no time coordinate, no initial condition, and the
metric scores only u and v (the pressure the network also outputs is defined
up to a constant and is not scored).

Settings are the repository's own (`configs/base.py`): ModifiedMlp, out_dim 3,
batch 2048, 100000 published steps.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config (2000 each)
rather than tied to the step count, so simply lowering `max_steps` would leave
the method still in warm-up when its time ran out. They are scaled by the same
factor as the step count. The two *weight-update* cadences are left at their
published values, as on the other cases.

Reading the score: the task's b00 anchor is 0.9975 — near-trivial (predicting
zero scores 1.0, a per-field constant 0.667). That anchor is not a hard
ceiling: re-running the same baseline with its inputs rescaled to [-1,1] and
the boundary weight raised reaches 0.348, so the task has plenty of headroom
and a score near 1.0 means the run produced nothing.

Set from outside, none of it part of the method: the seed, the step count, the
lr-schedule scaling above, `WANDB_MODE=disabled`, and the working directory.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
EXAMPLE = APTS / "examples" / "bfs_flow"
METHOD_LIBS = "apts"
NATIVE = True
SOURCE = "reference/APTS/examples/bfs_flow"
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
    (work / "data").symlink_to(EXAMPLE / "data")
    cwd = os.getcwd()
    os.chdir(work)
    try:
        import train as apts_train, models
        from configs.pseudo_time import get_config
        from jaxpi.models import (create_lr_schedule, create_optimizer,
                                  create_arch, create_train_state)
        from jaxpi.checkpointing import create_checkpoint_manager, restore_checkpoint
        from utils import get_dataset, inflow_profile

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
        # Logging is not the method: their evaluator predicts on the reference
        # grid every `log_every_steps`, so its cost would land inside the budget
        # while producing nothing the comparison uses.
        cfg.logging.log_every_steps = steps + 10      # fires once, at step 0

        apts_train.train_and_evaluate(cfg)

        lr = create_lr_schedule(cfg.optim)
        tx = create_optimizer(cfg.optim, lr)
        arch = create_arch(cfg.arch)
        state = create_train_state(cfg, tx, arch)
        mngr = create_checkpoint_manager(
            cfg.saving, os.path.join(os.getcwd(), cfg.wandb.name, "ckpt"))
        state = restore_checkpoint(mngr, state)

        (u_ref, v_ref, p_ref, coords, inflow_coords, outflow_coords,
         wall_coords, nu) = get_dataset()
        u_inflow, _ = inflow_profile(inflow_coords[:, 1])
        model = models.NavierStokes2D(cfg, lr, tx, arch, state, u_inflow,
                                      inflow_coords, outflow_coords, wall_coords, nu)
        uvp_fn = jax.jit(jax.vmap(model.neural_net, (None, 0, 0)))

        def predict(prm, X):
            """X is (N,2) as (x, y) — steady, no time column."""
            us, vs = [], []
            for i in range(0, len(X), 8192):
                b = X[i:i + 8192]
                u, v, _ = uvp_fn(prm, b[:, 0], b[:, 1])
                us.append(u); vs.append(v)
            return {"u": jnp.concatenate(us), "v": jnp.concatenate(vs)}

        return {"params": state.params, "predict": predict,
                "info": {"config": "configs/pseudo_time.py", "arch": str(cfg.arch.arch_name),
                         "batch_size": int(cfg.training.batch_size),
                         "steps": steps, "published_max_steps": PUBLISHED_MAX_STEPS,
                         "schedule_scaled_by": round(scale, 5),
                         "warmup_steps": int(cfg.optim.warmup_steps),
                         "decay_steps": int(cfg.optim.decay_steps),
                         "published_schedule": orig,
                         "nu": float(nu), "Re": round(1.0 / float(nu)),
                         "steady": True}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
