"""APTS on the 3D screened Poisson problem in a cube minus two spheres.

The donor is `reference/APTS/examples/lid_driven_cavity/` — their only *steady*
example, and so the only one built on `ForwardBVP`, the base class this task
needs (no time coordinate, no initial condition). `main.py` is theirs verbatim;
`train.py`, `models.py`, `utils.py` and `configs/base.py` are theirs with the
changes marked `[sota patch]` in each file. The shared `jaxpi` library is
imported from the upstream repository unmodified.

The run uses `configs/pseudo_time.py`, which *is* APTS — the
`strategy="dynamic"` pseudo-time stepping the paper contributes.
`fixed_pseudo_time.py` is the paper's PTS control and `baseline.py` turns both
off, so either of those would report the control as the SOTA.

NATIVE = False. APTS publishes no Poisson case, no 3D case and no
complex-geometry case; what carries over is the method (the pseudo-time
weighting, the dynamic loss re-weighting, PirateNet/ModifiedMlp with a Fourier
embedding, SOAP) and the steady training loop.

What this problem needs that the cavity does not have:

  * **three input coordinates.** `config.input_dim = 3`; `jaxpi` unpacks the
    batch generically (`coords = tuple(batch[:, i] for i in range(...))`), so
    nothing in the library changes. Their own 2D+time cases already run at
    input_dim 3.
  * **one output field.** `arch.out_dim = 1`, and the loss keys drop from five
    to three.
  * **a domain with holes.** Two spheres of radius 0.2 are cut out. The
    residual is masked — see `models.py:r_net` and the `1/kept` rescale in
    `losses`, which together make `ForwardBVP`'s plain mean the masked mean.
  * **Neumann on every boundary.** The cavity matches values; here `du/dn = 0`
    on the six cube faces and both sphere surfaces. The sphere points and
    normals are the task's own Fibonacci lattice, reproduced bit for bit —
    checked against `task/baseline.py:_build_dataset` (max |difference| 0.0 on
    both points and normals).

Settings are the repository's own (`configs/base.py`): ModifiedMlp, 3 layers of
256 with a Fourier embedding, SOAP, batch 1024, 50000 published steps.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config (2000 each) rather
than tied to the step count, so simply lowering `max_steps` would leave the
method still in warm-up when its time ran out. They are scaled by the same
factor as the step count. The two *weight-update* cadences are left at their
published values, as on the other cases.

Set from outside, none of it part of the method: the seed, the step count, the
lr-schedule scaling above, `WANDB_MODE=disabled`, and the working directory.

Reading the score: predicting zero gives exactly 1.0 and the best constant
0.929 — the field has a nonzero mean, so the two floors differ.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
METHOD_LIBS = "apts"
NATIVE = False
SOURCE = "reference/APTS/examples/lid_driven_cavity (their only steady example), retargeted to 3D screened Poisson"
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
    cwd = os.getcwd()
    os.chdir(work)
    try:
        import train as apts_train, models
        from configs.pseudo_time import get_config
        from jaxpi.models import (create_lr_schedule, create_optimizer,
                                  create_arch, create_train_state)
        from jaxpi.checkpointing import create_checkpoint_manager, restore_checkpoint

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
        # 65202-point reference every `log_every_steps`, so its cost would land
        # inside the budget while producing nothing the comparison uses.
        cfg.logging.log_every_steps = steps + 10      # fires once, at step 0

        apts_train.train_and_evaluate(cfg)

        lr = create_lr_schedule(cfg.optim)
        tx = create_optimizer(cfg.optim, lr)
        arch = create_arch(cfg.arch)
        state = create_train_state(cfg, tx, arch)
        mngr = create_checkpoint_manager(
            cfg.saving, os.path.join(os.getcwd(), cfg.wandb.name, "ckpt"))
        state = restore_checkpoint(mngr, state)

        model = models.PoissonCG3D(cfg, lr, tx, arch, state)
        u_fn = jax.jit(jax.vmap(model.neural_net, (None, 0, 0, 0)))

        def predict(prm, X):
            """X is (N,3) as (x, y, z) — steady, no time column."""
            us = []
            for i in range(0, len(X), 8192):
                b = X[i:i + 8192]
                us.append(u_fn(prm, b[:, 0], b[:, 1], b[:, 2]))
            return {"u": jnp.concatenate(us)}

        return {"params": state.params, "predict": predict,
                "info": {"config": "configs/pseudo_time.py",
                         "arch": str(cfg.arch.arch_name),
                         "batch_size": int(cfg.training.batch_size),
                         "steps": steps,
                         "published_max_steps": PUBLISHED_MAX_STEPS,
                         "schedule_scaled_by": round(scale, 5),
                         "warmup_steps": int(cfg.optim.warmup_steps),
                         "decay_steps": int(cfg.optim.decay_steps),
                         "published_schedule": orig,
                         "steady": True, "input_dim": 3, "out_dim": 1,
                         "loss_keys": ["bc_face", "bc_sph", "r"],
                         "bc": "homogeneous Neumann on 6 faces + 2 sphere surfaces",
                         "residual_mask": "analytic: batch points inside a sphere zeroed, mean rescaled",
                         "sphere_points": "task/baseline.py's Fibonacci lattice, reproduced exactly"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
