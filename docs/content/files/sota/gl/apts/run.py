"""APTS on Ginzburg-Landau — the authors' own implementation of this problem.

`main.py`, `train.py`, `models.py`, `utils.py` and `configs/` are
`reference/APTS/examples/ginzburg_landau/` verbatim; the shared `jaxpi` library
is imported from the upstream repository. The run uses `configs/pseudo_time.py`,
which *is* APTS — the `strategy="dynamic"` pseudo-time stepping the paper
contributes. `fixed_pseudo_time.py` is the paper's PTS control and
`baseline.py` turns both off, so either of those would report the control as
the SOTA.

Same problem, checked against the task rather than assumed:

  * residual — `models.py:r_net` is
    u_t - eps(u_xx+u_yy) - k(u - u(u^2+v^2) + 1.5 v(u^2+v^2)) and
    v_t - eps(v_xx+v_yy) - k(v - v(u^2+v^2) - 1.5 u(u^2+v^2)), i.e. the task's
    pair with c = 1.5; eps = 0.004 and k = 10 are read out of the .mat and are
    the task's values;
  * time window — `config.time_range = [0.0, 0.8]` and `utils.get_dataset`
    slices `int(0.8 * 101) = 80` frames, t = 0 ... 0.79. **That is exactly the
    task's window**; nothing is changed;
  * space — [-1,1]^2 from the data, `periodicity` period pi on axes 1 and 2,
    i.e. cos(pi x)/sin(pi x) (`jaxpi/archs.py:PeriodEmbs` multiplies, so period
    is the wavenumber) — spatial period 2, the task's periodic square, enforced
    by construction;
  * axis convention — checked, and unlike the Grey-Scott case **no swap is
    needed**. `train.py` pairs mesh point (x_i, y_j) with
    `usol[t].reshape(-1)[i*200+j] = usol[t][i,j]`, and here that reading is the
    right one: the initial condition u0 = 10 x exp(-25 r^2) is asymmetric in
    x <-> y, and `usol[0]` matches it to 2.8e-10 while `usol[0].T` is off by
    1.2. Against the task grid the direct reading agrees to 5e-9. (The
    transpose is not harmless here either way — it would flip the sign of c.)

Settings are the repository's own (`configs/base.py`): PirateNet, 3x256, swish,
RFF 2.0, out_dim 2, batch 8192, causal weighting in 16 chunks, 100000 published
steps.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config (2000 each) rather
than tied to the step count, so simply lowering `max_steps` would leave the
method still in warm-up when its time ran out. They are scaled by the same
factor as the step count so the method executes its whole schedule inside the
time available. The two *weight-update* cadences are left at their published
values, as on kdv, ks and gs.

Set from outside, none of it part of the method: the seed, the step count, the
lr-schedule scaling above, `WANDB_MODE=disabled`, and the working directory.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
EXAMPLE = APTS / "examples" / "ginzburg_landau"
METHOD_LIBS = "apts"
NATIVE = True
SOURCE = "reference/APTS/examples/ginzburg_landau"
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
        # Logging is not the method: their evaluator predicts on a reference
        # grid every `log_every_steps`, so its compilation and per-call cost
        # would land inside the budget while producing nothing the comparison
        # uses. It does not touch model.state.
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

        u_ref, v_ref, t_star, x_star, y_star, eps, k = get_dataset(
            time_range=cfg.time_range)
        dt = t_star[1] - t_star[0]
        t_max = float(t_star[-1] + 1.1 * dt)          # train.py's own t1
        model = models.GinzburgLandau(cfg, lr, tx, arch, state,
                                      t_max=t_max, eps=eps, k=k)
        uv_fn = jax.jit(jax.vmap(model.neural_net, (None, 0, 0, 0)))

        def predict(prm, X):
            """X is (N,3) as (t, x, y) — their axes, no swap needed here."""
            us, vs = [], []
            for i in range(0, len(X), 8192):
                b = X[i:i + 8192]
                u, v = uv_fn(prm, b[:, 0], b[:, 1], b[:, 2])
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
                         "eps": float(eps), "k": float(k), "c": 1.5}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
