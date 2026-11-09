"""APTS on the NACA0012 airfoil at Re = 1000 — their cavity recipe, on an immersed body.

APTS publishes no airfoil case. Its closest example is `lid_driven_cavity`:
2D steady Navier-Stokes with (u, v, p), velocity prescribed on every boundary
and **no pressure condition** — structurally the same boundary problem this
task poses, the difference being that the Dirichlet set here is the farfield
box plus an immersed body rather than the four walls of a square.
NATIVE = False.

`main.py`, `models.py` and `configs/` are
`reference/APTS/examples/lid_driven_cavity/` verbatim apart from one
`[sota patch]`; `utils.py` is rewritten for this geometry; `train.py` carries
two `[sota patch]` marks. The shared `jaxpi` library is imported from the
upstream repository. The run uses `configs/pseudo_time.py`, which *is* APTS —
the `strategy="dynamic"` pseudo-time stepping the paper contributes.

What changed and why:

  * `models.py:__init__` — the cavity calls
    `sample_points_on_square_boundary` and sets `u_bc[:num_pts] = 1` for the
    lid. Here the boundary point set and its u targets are passed in instead.
    **Neither loss is touched**: every v target is zero in this problem exactly
    as in the cavity, so `v_bc_loss = mean(v_pred**2)` still states the
    condition, and `u_bc_loss` is the same expression against a different
    target vector. `r_net` — u u_x + v u_y + p_x - nu(u_xx+u_yy), etc. — is
    already the task's three steady equations, and `nu = 1/config.Re` with
    `Re = 1000` is the task's viscosity.
  * `train.py` — the cavity fills a square, so it draws residual points from
    the domain box with `UniformSampler`. The fluid region here is the box
    minus the airfoil, given only as a level set, so residual points come from
    the task's own fluid set through jaxpi's `MeshSampler` — their own class,
    called the way their own `bfs_flow` example calls it for an unstructured
    mesh. Sampling the box uniformly would place residual points inside the
    airfoil.
  * `utils.py` — rewritten to load this task's geometry. `pts_int` is
    `phi <= 0` from `collocation.csv` with the outer edges removed, which is
    what `task/baseline.py:_build_dataset` does and what Scale-PINN's own
    notebook does. The 400 airfoil points reproduce
    `task/baseline.py:_naca_airfoil_points` — checked against the task's own
    function, max |diff| 3e-8 (its float32 cast).

Settings are the repository's own (`configs/base.py`): ModifiedMlp, 3 layers of
256 with a Fourier embedding, out_dim 3, SOAP, batch 1024, 50000 published
steps. `config.Re` is set to the task's 1000 (their cavity default is 5000).

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config (2000 each), so
lowering `max_steps` alone would leave the method still in warm-up when its
time ran out. They are scaled by the same factor as the step count. The
weight-update cadences are left at their published values.

Reading the score: predicting zero gives exactly 1.0, the uniform freestream
(u=1, v=0) gives 0.480, and the best possible constant 0.403. A run that learns
only the farfield condition sits at 0.48, so anything above ~0.40 means the
airfoil was never resolved.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
METHOD_LIBS = "apts"
NATIVE = False
SOURCE = "reference/APTS/examples/lid_driven_cavity, re-pointed at the NACA0012 geometry"
PUBLISHED_MAX_STEPS = 50000
RE = 1000


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
        # Logging is not the method: their evaluator predicts on the whole
        # 801x401 reference grid, so its cost would land inside the budget while
        # producing nothing the comparison uses.
        cfg.logging.log_every_steps = steps + 10      # fires once, at step 0

        apts_train.train_and_evaluate(cfg)

        lr = create_lr_schedule(cfg.optim)
        tx = create_optimizer(cfg.optim, lr)
        arch = create_arch(cfg.arch)
        state = create_train_state(cfg, tx, arch)
        mngr = create_checkpoint_manager(
            cfg.saving, os.path.join(os.getcwd(), cfg.wandb.name, "ckpt"))
        state = restore_checkpoint(mngr, state)

        pts_int, pts_bc, u_bc, _x, _y, _V = get_dataset()
        model = models.NavierStokes2D(cfg, lr, tx, arch, state,
                                      1.0 / cfg.Re, pts_bc, u_bc)
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
                         "donor_case": "lid_driven_cavity",
                         "nu": 1.0 / RE, "Re": RE, "steady": True,
                         "n_fluid_pts": int(pts_int.shape[0]),
                         "n_bc_pts": int(pts_bc.shape[0]),
                         "sampler": "MeshSampler over the task's fluid set"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
