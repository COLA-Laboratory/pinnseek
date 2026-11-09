"""APTS on the lid-driven cavity at Re = 3200 — the authors' own case.

`main.py`, `train.py`, `models.py`, `utils.py` and `configs/` are
`reference/APTS/examples/lid_driven_cavity/` verbatim (`diff -q` clean); the
shared `jaxpi` library is imported from the upstream repository. The run uses
`configs/pseudo_time.py`, which *is* APTS — the `strategy="dynamic"`
pseudo-time stepping the paper contributes. `fixed_pseudo_time.py` is the
paper's PTS control and `baseline.py` turns both off, so either of those would
report the control as the SOTA. NATIVE = True.

Same problem, checked against the task rather than assumed:

  * residual — `models.py:r_net` is u u_x + v u_y + p_x - nu(u_xx+u_yy),
    u v_x + v v_y + p_y - nu(v_xx+v_yy), and u_x + v_y: the task's three
    steady equations;
  * viscosity — `train.py` sets nu = 1/config.Re, and `config.Re` is set to
    3200 here (their `base.py` default is 5000, a different Re they also ship
    data for). `ldc_Re3200.mat` carries `nu = 3.125e-4 = 1/3200`, so the file
    and the config agree;
  * geometry and boundary — `utils.py:sample_points_on_square_boundary` walks
    the unit square top, bottom, left, right, and `models.py` sets
    `u_bc[:num_pts] = 1` with everything else zero: lid u = 1, no-slip on the
    other three walls, the task's table exactly. Their `eps=0.005` pulls the
    side samples just short of the lid to avoid the corner singularity;
  * reference — their `data/ldc_Re3200.mat` is 256x256 on the same grid as the
    task's `ref_data.csv` (`x` matches to the last digit) and is indexed
    `[ix, iy]`; the lid row `u[:, -1]` equals the task's y = 1 row exactly.
    The interiors are two independent solves and differ by 2.4e-2 in relative
    L2 away from the lid corners, which is discretisation, not a different
    problem — Scale-PINN's own 102x102 cavity file sits 2.5e-2 from the task
    and 3.1e-2 from this one, so all three agree to the same tolerance.

The task's reference is the ground truth for the metric either way; their .mat
is used here only to confirm the problem and is never read during training.
Training uses no reference values at all: the loss is residual plus the
analytic boundary condition.

This case is **steady** — no time coordinate, no initial condition — and the
metric scores only u and v, with the boundary overridden by the known values
before the norm (see ../../problem.py). The pressure the network also outputs
is defined up to a constant and is not scored.

Settings are the repository's own (`configs/base.py`): ModifiedMlp, 3 layers of
256 with a Fourier embedding, out_dim 3, SOAP, batch 1024, 50000 published
steps.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config (2000 each)
rather than tied to the step count, so simply lowering `max_steps` would leave
the method still in warm-up when its time ran out. They are scaled by the same
factor as the step count. The two *weight-update* cadences are left at their
published values, as on the other cases.

Set from outside, none of it part of the method: `config.Re = 3200`, the seed,
the step count, the lr-schedule scaling above, `WANDB_MODE=disabled`, and the
working directory.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
EXAMPLE = APTS / "examples" / "lid_driven_cavity"
METHOD_LIBS = "apts"
NATIVE = True
SOURCE = "reference/APTS/examples/lid_driven_cavity (their own LDC case, Re set to the task's 3200)"
PUBLISHED_MAX_STEPS = 50000
RE = 3200


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

        cfg = get_config()
        cfg.Re = RE
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

        nu = 1.0 / cfg.Re                            # train.py's own line
        model = models.NavierStokes2D(cfg, lr, tx, arch, state, nu)
        uvp_fn = jax.jit(jax.vmap(model.neural_net, (None, 0, 0)))

        def predict(prm, X):
            """X is (N,2) as (x, y) on the unit square — steady, no time column."""
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
                         "nu": nu, "Re": RE, "steady": True}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
