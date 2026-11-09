"""APTS on the long-time nonlinear heat equation — their advection recipe, retargeted.

APTS publishes no heat equation and nothing with a solution-dependent source.
Its closest example is `examples/advection`: a first-order-in-time transport
problem with a causally-weighted residual and an initial-condition term, which
is the shape this task has. It is 1D, constant-coefficient and periodic; this
task is 2D with diffusion, a `sin(u^2)` source and Dirichlet sides.
NATIVE = False.

`main.py` and `configs/` are `reference/APTS/examples/advection/` verbatim
apart from one `[sota patch]` in `base.py`; `models.py` and `train.py` carry
`[sota patch]` marks; `utils.py` is rewritten. The run uses
`configs/pseudo_time.py`, which *is* APTS.

What changed:

  * `models.py` — `Advection1D` becomes `HeatLT2D`: the network takes (t, x, y),
    the residual is
    `u_t - D (u_xx + u_yy) - A sin(u^2) (1 + 2 sin(pi t / 4)) sin(4 pi x) sin(2 pi y)`
    with D = 0.001 and A = 5, the IC term is `sin(4 pi x) sin(3 pi y)` on a
    64x64 lattice, and a Dirichlet term for `u = 0` on the four sides is added.
  * `configs/base.py` — `loss_weights` gains a `bcs` entry at 1.0. jaxpi checks
    that the weight dict matches the loss dict key for key, so adding a loss
    term without its weight is an error rather than a silent default; the smoke
    caught exactly that.
  * `train.py` — the domain box gains a third axis and spans t in [0, 100].
  * `config.input_dim` 2 -> 3 and **`arch.periodicity` is switched off**: the
    donor is periodic in x, this task is Dirichlet on all four sides.

**The trap: u == 0 satisfies the PDE and all four sides exactly**, because the
source carries `sin(u^2)`. Only the initial term rules it out. The published
weights start every term at 1.0 and let the dynamic weighting move them; that
is left alone.

The window is long — t to 100 is 12.5 periods of the forcing (period 8) — and
nothing is rescaled. Their causal weighting over 16 chunks is what is supposed
to handle a long window, so leaving it to do that is the honest test.

**The `t / t_max` arm was measured, not argued about.** Feeding raw t on a
window 50x longer than the donor's (advection runs to t = 2) is the one place
this port could be accused of handicapping the method: five of their examples
(ks, gray_scott, ginzburg_landau, kolmogorov_flow, rayleigh_taylor) normalise t
inside `neural_net`, and the three that do not — advection, kdv, wave — all have
t_max <= 2. So the alternative was run: `t = t / self.t1` added to
`neural_net`, everything else identical, seed 1, the same 1900 steps and the
same wall clock (179.2 s). It scores **0.99537**, against **0.99730** for the
source form reported here. Both sit on the trivial floor (zero scores exactly
1.0, and so does the best constant). The input scaling is not what is failing
here, so the source form is what is reported.

Reading the score: predicting zero gives exactly 1.0, and so does the best
possible constant, because the field is a zero-mean oscillation.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
METHOD_LIBS = "apts"
NATIVE = False
SOURCE = "reference/APTS/examples/advection, retargeted to 2D nonlinear heat"
PUBLISHED_MAX_STEPS = 100000


def train(*, steps: int, seed: int, gpu: str) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    sys.path[:0] = [str(APTS), str(HERE)]

    import jax, jax.numpy as jnp
    jax.config.update("jax_default_matmul_precision", "highest")

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
        model = models.HeatLT2D(cfg, lr, tx, arch, state, d)
        u_fn = jax.jit(jax.vmap(model.neural_net, (None, 0, 0, 0)))

        def predict(prm, X):
            """X is (N,3) as (t, x, y)."""
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
                         "D": 0.001, "A": 5.0, "t_max": 100.0,
                         "periodicity": "off (task is Dirichlet u=0 on all four sides)",
                         "input_dim": 3,
                         "t_input": "raw (donor form); the t/t_max arm was "
                                    "measured at 0.99537 vs 0.99730, seed 1, "
                                    "1900 steps -- both on the trivial floor"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
