"""APTS on the multiscale wave equation — their 1D wave recipe, in 2D with a variable medium.

APTS publishes a wave case, `examples/wave`, and it is the right donor by
structure: second order in time, both u(0,.) and u_t(0,.) prescribed, and a
loss built from exactly the terms this task needs (`u0`, `u_t0`, `bcs`, plus
the causally-weighted residual). It is 1D with a constant speed and Dirichlet
ends; this task is 2D with c(x,y) taking 1 and 12 and zero Neumann walls.
NATIVE = False.

`main.py` and `configs/` are `reference/APTS/examples/wave/` verbatim;
`models.py` and `train.py` each carry `[sota patch]` marks; `utils.py` is
rewritten for this task's data. The shared `jaxpi` library is imported from the
upstream repository. The run uses `configs/pseudo_time.py`, which *is* APTS.

What changed, and only this:

  * `models.py` — `Wave1D` becomes `Wave2DMS`: the network takes (t, x, y),
    the residual is `u_tt - c(x,y)(u_xx + u_yy)` with c interpolated
    bilinearly from the task's 256x256 table as `problem.md` specifies, the two
    initial terms are evaluated on the (x, y) lattice instead of a line, and
    `bcs_loss` is the squared normal derivative on the four sides instead of
    the Dirichlet trace at two ends. The loss *shape* and the class it derives
    from are untouched.
  * `train.py` — the domain box gains a third axis and the dataset comes from
    the rewritten loader.
  * `config.input_dim` 2 -> 3, which is what tells jaxpi's sampler and arch how
    many coordinates there are.

Settings are otherwise the repository's own (`configs/base.py`): PirateNet,
3 layers of 256 with a Fourier embedding, SOAP, batch 4096, causal weighting
with 16 chunks, all four loss weights at 1.0, 100000 published steps.

**The trap in this case is that u == 0 satisfies the equation and both Neumann
conditions exactly.** Only the initial terms rule it out. Their published
weights put `u0`, `u_t0`, `bcs` and `res` all at 1.0 and let the dynamic
weighting move them; that is left alone, and where it lands is the measurement.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute (2000 each), so lowering
`max_steps` alone would leave the method in warm-up when its time ran out. They
are scaled by the same factor as the step count; the weight-update cadences are
left at their published values.

Reading the score: predicting zero gives exactly 1.0, the best constant 0.781,
and freezing the initial condition 1.274.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
METHOD_LIBS = "apts"
NATIVE = False
SOURCE = "reference/APTS/examples/wave, extended to 2D with a variable medium"
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
        # 51x129x129 reference grid, so its cost would land inside the budget.
        cfg.logging.log_every_steps = steps + 10      # fires once, at step 0

        apts_train.train_and_evaluate(cfg)

        lr = create_lr_schedule(cfg.optim)
        tx = create_optimizer(cfg.optim, lr)
        arch = create_arch(cfg.arch)
        state = create_train_state(cfg, tx, arch)
        mngr = create_checkpoint_manager(
            cfg.saving, os.path.join(os.getcwd(), cfg.wandb.name, "ckpt"))
        state = restore_checkpoint(mngr, state)

        u_ref, t_star, x_star, y_star, u0, cx, cy, c_grid = get_dataset()
        model = models.Wave2DMS(cfg, lr, tx, arch, state, u0, t_star,
                                x_star, y_star, cx, cy, c_grid)
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
                         "donor_case": "wave (1D, constant c, Dirichlet)",
                         "coefficient": "task 256x256 c(x,y), bilinear",
                         "bc": "zero Neumann on all four sides",
                         "input_dim": 3}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
