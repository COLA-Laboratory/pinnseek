"""APTS on 3D unsteady Navier-Stokes in vorticity-velocity form.

The donor is `reference/APTS/examples/kolmogorov_flow/` — their own unsteady
incompressible flow, and the closest thing in the repository to this task.
`main.py` is theirs verbatim; `train.py`, `models.py`, `utils.py` and
`configs/base.py` are theirs with the changes marked `[sota patch]` in each
file. The shared `jaxpi` library is imported from the upstream repository
unmodified.

The run uses `configs/pseudo_time.py`, which *is* APTS — the
`strategy="dynamic"` pseudo-time stepping the paper contributes.
`fixed_pseudo_time.py` is the paper's PTS control and `baseline.py` turns both
off, so either of those would report the control as the SOTA.

NATIVE = False. APTS publishes no 3D flow and nothing 4-dimensional; what
carries over is the method (pseudo-time weighting, dynamic loss re-weighting,
causal weighting over 16 chunks, PirateNet with a Fourier embedding, SOAP) and
the unsteady training loop.

What this problem needs that Kolmogorov flow does not have:

  * **four input coordinates and no pressure.** `config.input_dim = 4`;
    `jaxpi` unpacks the batch generically. The equation is in vorticity-velocity
    form — pressure has been eliminated by taking the curl — so the network's
    three outputs are the three velocity components.
  * **third derivatives.** `w = curl u` is not a network output, so `lap(w)` in
    the residual is a third derivative of the velocity. Their Kolmogorov
    residual needed only second derivatives. This is the one structural cost of
    the change and it is what makes the per-step cost here what it is.
  * **a third periodic axis.** `arch.periodicity` gains z, so the boundary
    condition stays exact and free — the same advantage their own 2D case has.
  * **the task's forcing** in place of `2 sin(4 pi y)`; see utils.py. It is
    identical to SPINN's `navier_stokes4d_forcing_term`, which is where the
    task's reference comes from.

Settings are the repository's own (`configs/base.py`): PirateNet, 3 layers of
256 with a Fourier embedding, SOAP, batch 8192, causal weighting with 16
chunks, 100000 published steps.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config (2000 each) rather
than tied to the step count, so simply lowering `max_steps` would leave the
method still in warm-up when its time ran out. They are scaled by the same
factor as the step count. The two *weight-update* cadences are left at their
published values, as on the other cases.

Training data is the task's initial condition only. `task/ref_data.csv` is read
for the evaluator's `l2_error` logging, and `log_every_steps` is set past the
end of the run, so it fires once, at step 0, and never again.

Reading the score: predicting zero gives exactly 1.0 and freezing the initial
condition 1.342.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
METHOD_LIBS = "apts"
NATIVE = False
SOURCE = "reference/APTS/examples/kolmogorov_flow (their own unsteady flow), retargeted to 3D+time"
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

        cfg = get_config()
        cfg.seed = seed
        scale = steps / PUBLISHED_MAX_STEPS
        orig = dict(max_steps=int(cfg.training.max_steps),
                    warmup_steps=int(cfg.optim.warmup_steps),
                    decay_steps=int(cfg.optim.decay_steps))
        cfg.training.max_steps = steps
        cfg.optim.warmup_steps = max(int(round(orig["warmup_steps"] * scale)), 1)
        cfg.optim.decay_steps = max(int(round(orig["decay_steps"] * scale)), 1)
        cfg.saving.save_every_steps = 0               # the model is returned directly
        # Logging is not the method: their evaluator predicts the vorticity on
        # the whole reference grid every `log_every_steps`, and that is a third
        # derivative of the network -- its cost would land inside the budget
        # while producing nothing the comparison uses.
        cfg.logging.log_every_steps = steps + 10      # fires once, at step 0

        model = apts_train.train_and_evaluate(cfg, os.getcwd())

        wx_fn = jax.jit(jax.vmap(model.wx_net, (None, 0, 0, 0, 0)))
        wy_fn = jax.jit(jax.vmap(model.wy_net, (None, 0, 0, 0, 0)))
        wz_fn = jax.jit(jax.vmap(model.wz_net, (None, 0, 0, 0, 0)))

        def predict(prm, X):
            """X is (N,4) as (t, x, y, z). The vorticity is the autodiff curl of
            the velocity — the task requires the curl, not a network output."""
            out = {"wx": [], "wy": [], "wz": []}
            for i in range(0, len(X), 4096):
                b = X[i:i + 4096]
                t, x, y, z = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
                out["wx"].append(wx_fn(prm, t, x, y, z))
                out["wy"].append(wy_fn(prm, t, x, y, z))
                out["wz"].append(wz_fn(prm, t, x, y, z))
            return {k: jnp.concatenate(v) for k, v in out.items()}

        return {"params": model.state.params, "predict": predict,
                "info": {"config": "configs/pseudo_time.py",
                         "arch": str(cfg.arch.arch_name),
                         "batch_size": int(cfg.training.batch_size),
                         "steps": steps,
                         "published_max_steps": PUBLISHED_MAX_STEPS,
                         "schedule_scaled_by": round(scale, 6),
                         "warmup_steps": int(cfg.optim.warmup_steps),
                         "decay_steps": int(cfg.optim.decay_steps),
                         "published_schedule": orig,
                         "input_dim": 4, "out_dim": 3, "nu": 0.05, "t_max": 5.0,
                         "causal_chunks": int(cfg.causal.num_chunks),
                         "loss_keys": ["u_ic", "v_ic", "w_ic", "rx", "ry", "rz", "rc"],
                         "bc": ("exact: arch.periodicity embeds x, y and z with "
                                "period 2 pi -- no boundary loss"),
                         "ic": "task/ns4d_ic.csv velocity on its 32^3 grid",
                         "vorticity": "autodiff curl of the velocity; lap(w) is a third derivative"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
