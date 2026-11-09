"""APTS on Kolmogorov flow — the authors' own implementation of this problem.

`main.py`, `models.py`, `train.py` and `configs/` are
`reference/APTS/examples/kolmogorov_flow/` verbatim; the shared `jaxpi`
library is imported from the upstream repository. `utils.py` is their file
with **one** change, marked `[sota patch]` in it: as published it loads
`data/kolmogorov_flow_Re10000.npy`, a file the repository does not contain —
the only dataset shipped is `data/kolmogorov_flow.npy`, so the example cannot
load anything as it stands. The patch points at the file that is there.

The run uses `configs/pseudo_time.py`, which *is* APTS — the
`strategy="dynamic"` pseudo-time stepping the paper contributes.
`fixed_pseudo_time.py` is the paper's PTS control and `baseline.py` turns both
off, so either of those would report the control as the SOTA.

Same problem, checked against the task rather than assumed:

  * residual — `models.py:r_net` is
    u_t + u u_x + v u_y + p_x - nu(u_xx+u_yy) - f, v_t + u v_x + v v_y + p_y -
    nu(v_xx+v_yy), and u_x + v_y, with `body_force_fn = 2 sin(4 pi y)` acting
    on the x-momentum only — the task's three equations and its forcing;
  * viscosity — `nu` is read out of the dataset and is 5e-4, i.e. Re = 2000,
    the task's value. (The filename the example asks for claims Re = 10000;
    the file that exists is Re = 2000, and the task was built from it. The
    stored nu was independently confirmed by fitting the vorticity equation
    to the field itself: 5.26e-4.);
  * time window — `config.time_range = [0.0, 0.4]` and `utils.get_dataset`
    slices `int(0.4 * 40) = 16` frames and shifts them to start at zero,
    giving t = 0, 0.05, ..., 0.75. **That is exactly the task's window**;
  * space — [0,1]^2, with `periodicity` period 2*pi on axes 1 and 2, i.e.
    cos(2 pi x)/sin(2 pi x) features (`jaxpi/archs.py:PeriodEmbs` multiplies,
    so period is the wavenumber) — spatial period 1, the task's periodic unit
    square, enforced by construction;
  * axis convention — no swap needed, and not for the reason gs needed one:
    `train.py` uses the dataset's own `coords` array as its mesh rather than
    rebuilding it with `meshgrid`, so each velocity value stays paired with
    the point it was stored against. (The flattened arrays do run x-fastest,
    which is why the task's (t,x,y)-ordered grid is the transpose of a naive
    `reshape(n,n)` — that trap is avoided here by not reshaping at all.)

Settings are the repository's own (`configs/base.py`): PirateNet, out_dim 3
(u, v, p), RFF 4.0, batch 8192, causal weighting in 16 chunks, 100000
published steps.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config (2000 each)
rather than tied to the step count, so simply lowering `max_steps` would leave
the method still in warm-up when its time ran out. They are scaled by the same
factor as the step count. The two *weight-update* cadences are left at their
published values, as on kdv, ks, gs and gl.

Only u and v are scored; the pressure the network also outputs is not part of
the metric.

Reading the score: u = v = 0 scores exactly 1.0 on this task, and the task's
own b00 anchor is 1.1187 — *above* the trivial field, with the right amplitude
but a decorrelated field. A score near 1.0 can mean collapse rather than
progress.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
EXAMPLE = APTS / "examples" / "kolmogorov_flow"
METHOD_LIBS = "apts"
NATIVE = True
SOURCE = "reference/APTS/examples/kolmogorov_flow (utils.py: one [sota patch] data-path fix)"
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

        u_ref, v_ref, w_ref, t_star, mesh, nu = get_dataset(
            time_range=cfg.time_range)
        dt = t_star[1] - t_star[0]
        t_max = float(t_star[-1] + 1.1 * dt)          # train.py's own t1
        model = models.NavierStokes2D(cfg, lr, tx, arch, state,
                                      t_max=t_max, nu=nu)
        uvp_fn = jax.jit(jax.vmap(model.neural_net, (None, 0, 0, 0)))

        def predict(prm, X):
            """X is (N,3) as (t, x, y) — their axes, no swap needed here."""
            us, vs = [], []
            for i in range(0, len(X), 8192):
                b = X[i:i + 8192]
                u, v, _ = uvp_fn(prm, b[:, 0], b[:, 1], b[:, 2])
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
                         "nu": float(nu), "Re": round(1.0 / float(nu)),
                         "body_force": "2 sin(4 pi y) on x-momentum"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
