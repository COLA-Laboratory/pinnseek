"""APTS on Rayleigh-Taylor — the authors' own case, on the authors' own data.

`main.py`, `models.py`, `utils.py` and `configs/` are
`reference/APTS/examples/rayleigh_taylor/` verbatim; `train.py` carries one
`[sota patch]`. The shared `jaxpi` library is imported from the upstream
repository. The run uses `configs/pseudo_time.py`, which *is* APTS.
NATIVE = True.

**The task's reference is this example's own simulation, cut and re-zeroed.**
That is not obvious from the metadata — the task has 20 snapshots over
t in [0, 4.75], theirs has 40 over t in [0.01, 9.76] — so it was checked rather
than assumed:

  * the spatial meshes are identical point for point (100 x 200 cell centres,
    max |dxy| 4.9e-10);
  * the task's t = 0 frame equals their **t = 1.01** frame exactly
    (relative L2 0.0000, found by scanning the task's first frame against all
    forty of theirs);
  * aligning `task t -> their t + 1.01`, all 20 snapshots and all three fields
    (u, v, T) agree to **5e-10**, which is the CSV's rounding.

So the data is theirs, and `get_dataset(time_range=...)` already slices by
fraction and shifts the start to zero. The fraction that selects frames 4..23
is `[0.1, 0.6]` — **which is exactly the value their own `configs/base.py`
ships**. Nothing had to be changed to match the task's window.

Everything else lines up too, and each was checked:

  * coefficients — `alpha1 = 8.426149773e-4` and `alpha4 = 1.1867816582e-3` in
    their data file are the task's `sqrt(Pr/Ra)` and `1/sqrt(Pr*Ra)` to the last
    digit, with Ra = 1e6 and Pr = 0.71 as `problem.md` states;
  * domain — x in [0,1], y in [0,2], set in `train.py`;
  * periodicity — `arch.periodicity` is `period = 2*pi` on axis 1, and jaxpi's
    `PeriodEmbs` embeds `cos(period*x)`, so that is a *spatial* period of 1:
    the task's periodic-in-x-with-period-1 condition;
  * the wall conditions and the four outputs (u, v, p, T) are their model's.

The one `[sota patch]`, in `train.py`, makes `train_and_evaluate` return the
trained model. Their checkpoints are written with
`suffix="time_window_{idx}"`, and restoring without the same suffix silently
returns the *initial* state — a failure mode this repository has hit before
(GUIDE.md, apts section). Returning the model removes it.

Only T is scored (see ../../problem.py), and `predict` returns the key "T"
accordingly; u, v and p are trained but not part of the metric.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config, so lowering
`max_steps` alone would leave the method in warm-up when its time ran out.

Reading the score: predicting zero gives exactly 1.0 and the best possible
constant 0.740.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
EXAMPLE = APTS / "examples" / "rayleigh_taylor"
METHOD_LIBS = "apts"
NATIVE = True
SOURCE = "reference/APTS/examples/rayleigh_taylor (their own case; the task's reference is this simulation cut at t=1.01)"
PUBLISHED_MAX_STEPS = 100000
TIME_RANGE = [0.1, 0.6]     # their own default; selects frames 4..23 = task t 0..4.75


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
        import train as apts_train
        from configs.pseudo_time import get_config

        cfg = get_config()
        cfg.seed = seed
        cfg.time_range = TIME_RANGE
        scale = steps / PUBLISHED_MAX_STEPS
        orig = dict(max_steps=int(cfg.training.max_steps),
                    warmup_steps=int(cfg.optim.warmup_steps),
                    decay_steps=int(cfg.optim.decay_steps))
        cfg.training.max_steps = steps
        cfg.optim.warmup_steps = max(int(round(orig["warmup_steps"] * scale)), 1)
        cfg.optim.decay_steps = max(int(round(orig["decay_steps"] * scale)), 1)
        cfg.saving.save_every_steps = steps
        # Logging is not the method: their evaluator predicts on the reference
        # grid every `log_every_steps`, so its cost would land inside the budget.
        cfg.logging.log_every_steps = steps + 10

        model = apts_train.train_and_evaluate(cfg, os.getcwd())
        params = model.state.params
        uvpt = jax.jit(jax.vmap(model.neural_net, (None, 0, 0, 0)))

        def predict(prm, X):
            """X is (N,3) as (t, x, y). Only T is scored, so only T is returned."""
            outs = []
            for i in range(0, len(X), 16384):
                b = X[i:i + 16384]
                _u, _v, _p, T = uvpt(prm, b[:, 0], b[:, 1], b[:, 2])
                outs.append(T)
            return {"T": jnp.concatenate(outs, axis=0)}

        return {"params": params, "predict": predict,
                "info": {"config": "configs/pseudo_time.py", "arch": str(cfg.arch.arch_name),
                         "batch_size": int(cfg.training.batch_size),
                         "steps": steps, "published_max_steps": PUBLISHED_MAX_STEPS,
                         "schedule_scaled_by": round(scale, 5),
                         "warmup_steps": int(cfg.optim.warmup_steps),
                         "decay_steps": int(cfg.optim.decay_steps),
                         "published_schedule": orig,
                         "time_range": TIME_RANGE,
                         "time_range_note": "their own base.py default; frames 4..23, shifted to t=0",
                         "num_time_windows": int(cfg.training.num_time_windows),
                         "causal_chunks": int(cfg.causal.num_chunks),
                         "scored_field": "T only"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
