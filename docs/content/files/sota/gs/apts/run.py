"""APTS on Grey-Scott — the authors' own implementation of this problem.

`main.py`, `models.py`, `utils.py` and `configs/` are
`reference/APTS/examples/gray_scott/` verbatim; the shared `jaxpi` library is
imported from the upstream repository. `train.py` is their file with **one**
change, marked `[sota patch]` in it: as shipped, its residual-batch condition
is the negation of the same construct in their `bfs_flow`,
`lid_driven_cavity` and `inviscid_burgers` examples, and under this config's
own default (`training.random_sampling = True`) it indexes `fixed_batches`
before anything fills it — `KeyError('res')` on step 0, i.e. the example
cannot run at all as published. The patch restores the sibling examples'
reading: fresh residual and IC batches every step. The run uses `configs/pseudo_time.py`,
which *is* APTS — the `strategy="dynamic"` pseudo-time stepping the paper
contributes. `fixed_pseudo_time.py` is the paper's PTS control and
`baseline.py` turns both off, so either of those would report the control as
the SOTA.

Same problem, checked against the task rather than assumed:

  * residual — `models.py:r_net` is
    u_t - b1(1-u) + c1 u v^2 - eps1(u_xx+u_yy) and
    v_t + b2 v - c2 u v^2 - eps2(v_xx+v_yy), with the six constants read out of
    the .mat: b1=40, b2=100, c1=c2=1000, eps1=0.2, eps2=0.1 — the task's;
  * time window — `config.time_range = [0.0, 0.4]` and `utils.get_dataset`
    slices `int(0.4 * 101) = 40` frames, i.e. t = 0 ... 0.78. **That is exactly
    the task's window**; nothing is changed here (unlike the ks case);
  * space — [-1,1]^2 from the data, with `periodicity` period pi on axes 1 and
    2, i.e. cos(pi x)/sin(pi x) features (`jaxpi/archs.py:PeriodEmbs`
    multiplies, so period is the wavenumber) — spatial period 2, the task's
    periodic square, enforced by construction;
  * reference — their `data/gray_scott.mat` is the same file Scale-PINN ships;
    the task's ref_data.csv is its first 40 frames subsampled [::2,::2]
    (max |diff| 5e-11, CSV rounding), and its first frame is the task's
    analytic IC.

Settings are the repository's own (`configs/base.py`): PirateNet, 3x256,
swish, RFF 2.0, out_dim 2, batch 8192, random residual resampling, causal
weighting in 16 chunks, 100000 published steps.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config (2000 each)
rather than tied to the step count, so simply lowering `max_steps` would leave
the method still in warm-up when its time ran out. They are scaled by the same
factor as the step count so the method executes its whole schedule inside the
time available. The two *weight-update* cadences are left at their published
values, as on kdv and ks.

**Axis convention.** `train.py` builds its mesh as
`meshgrid(x_star, y_star, indexing="ij").reshape(-1,2)` and pairs entry
`i*200+j` with `usol[t].reshape(-1)[i*200+j] = usol[t][i,j]` — but the .mat
stores `usol` as [t, y, x]: `usol[0].T` matches the paper's initial condition
1-exp(-10((x+0.05)^2+(y+0.02)^2)) to 1e-4 while `usol[0]` itself is off by
1.1e-1, and against the task grid the transposed reading agrees to 5e-11 and
the untransposed one to 0.69. So their `x` is the task's `y`. Because the
Laplacian is symmetric and the domain is a periodic square, a model trained in
their labelling is *exactly* a solution of the task's problem read with the two
spatial arguments swapped — the training is untouched, and `predict` below
swaps them. (Getting this wrong is not a rounding error: the x<->y mirror of
the reference scores 0.164 against the reference itself.)

Verified on the trained weights, not just on the data: scoring
`weights/scale_pinn_s1.pkl` and `_s2.pkl` with the swap gives 0.0155 and
0.0307, and without it 0.1617 and 0.1659 — the x<->y mirror floor (0.164)
that the reference's own mirror predicts. Reading the axes the other way
would have inflated these methods' error roughly tenfold.

Set from outside, none of it part of the method: the seed, the step count, the
lr-schedule scaling above, `WANDB_MODE=disabled`, and the working directory.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
EXAMPLE = APTS / "examples" / "gray_scott"
METHOD_LIBS = "apts"
NATIVE = True
SOURCE = "reference/APTS/examples/gray_scott (train.py: one [sota patch] typo fix)"
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
        # Logging is not the method: `train.py` calls its evaluator every
        # `log_every_steps` and that evaluator predicts on a reference grid, so
        # its compilation and per-call cost would land inside the budget while
        # producing nothing the comparison uses. It does not touch model.state.
        cfg.logging.log_every_steps = steps + 10      # fires once, at step 0

        apts_train.train_and_evaluate(cfg, os.getcwd())

        lr = create_lr_schedule(cfg.optim)
        tx = create_optimizer(cfg.optim, lr)
        arch = create_arch(cfg.arch)
        state = create_train_state(cfg, tx, arch)
        mngr = create_checkpoint_manager(
            cfg.saving, os.path.join(os.getcwd(), cfg.wandb.name, "ckpt"),
            suffix="time_window_1")
        state = restore_checkpoint(mngr, state)

        u_ref, v_ref, t_star, x_star, y_star, b1, b2, c1, c2, e1, e2 = \
            get_dataset(time_range=cfg.time_range)
        # train.py's own t_max for the single time window
        dt = t_star[1] - t_star[0]
        t_max = float(t_star[-1] + 1.1 * dt)
        model = models.GrayScott(cfg, lr, tx, arch, state, t_max=t_max,
                                 b1=b1, b2=b2, c1=c1, c2=c2, eps1=e1, eps2=e2)
        uv_fn = jax.jit(jax.vmap(model.neural_net, (None, 0, 0, 0)))

        def predict(prm, X):
            """X is (N,3) as (t, x, y); their model's x/y are the transpose of
            the task's, so the two spatial columns are swapped on the way in."""
            us, vs = [], []
            for i in range(0, len(X), 8192):
                b = X[i:i + 8192]
                u, v = uv_fn(prm, b[:, 0], b[:, 2], b[:, 1])
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
                         "time_range": list(cfg.time_range),
                         "t_trained_max": float(t_star[-1]), "t_norm": t_max,
                         "coeffs": [float(b1), float(b2), float(c1), float(c2),
                                    float(e1), float(e2)],
                         "xy_swapped_at_predict": True}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
