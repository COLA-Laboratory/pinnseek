"""APTS on the vortex street grown from rest.

**This is the `cyl` port with the ramped inflow `U(t) = sin(pi t / 8)`, the
longer window `t in [0, 6]` and an at-rest initial condition**; the donors,
the hyperparameters, the masking and the boundary construction are identical,
so the two rows can be read against each other.

Two donors from `reference/APTS/examples/`, both of them theirs:

  * `kolmogorov_flow` — their own **unsteady** 2D incompressible Navier-Stokes,
    and the base here: `ForwardIVP`, the causal weighting over 16 chunks, the
    `t / t_max` input normalisation, the `{ics, res}` sampler pair, and a
    residual that already *is* this task's momentum and continuity equations.
  * `bfs_flow` — their own channel with an inlet, two no-slip walls and a
    `p = 0` outlet, which is where the four boundary terms come from. Their
    Kolmogorov box is periodic and carries no boundary loss at all.

`main.py` is theirs verbatim; `train.py`, `models.py`, `utils.py` and
`configs/base.py` are theirs with the changes marked `[sota patch]` in each
file. The shared `jaxpi` library is imported from the upstream repository
unmodified.

The run uses `configs/pseudo_time.py`, which *is* APTS — the
`strategy="dynamic"` pseudo-time stepping the paper contributes.
`fixed_pseudo_time.py` is the paper's PTS control and `baseline.py` turns both
off, so either of those would report the control as the SOTA.

NATIVE = False. APTS publishes no flow around an obstacle, and no case that is
both unsteady and wall-bounded.

What this problem needs that neither donor has on its own:

  * **an obstacle.** The residual is masked outside the cylinder — see
    `models.py:r_net` and the `1/kept` rescale in `losses`, which together make
    `ForwardIVP`'s plain mean the masked mean.
  * **a ramped inlet.** `u = 6 U(t) y (H - y) / H^2` with
    `U(t) = sin(pi t / 8)`, starting from zero; `bfs_flow`'s inlet is steady.
  * **no periodicity.** `arch.periodicity` is switched off, because the channel
    is periodic in neither direction. That removes a structural advantage the
    Kolmogorov case has and turns four conditions into loss terms.

**The pressure is a network output here**, so the outlet's `p = 0` can be
written down. The SPINN port on this same case cannot write it, because its
published formulation is in vorticity form.

Settings are the repository's own (`configs/base.py`): PirateNet, 3 layers of
256 with a Fourier embedding, SOAP, batch 8192, causal weighting with 16
chunks, 100000 published steps.

**The learning-rate schedule is compressed to the budget, not cut off.**
`warmup_steps` and `decay_steps` are absolute in this config (2000 each) rather
than tied to the step count, so simply lowering `max_steps` would leave the
method still in warm-up when its time ran out. They are scaled by the same
factor as the step count. The two *weight-update* cadences are left at their
published values, as on the other cases.

Reading the score: the fluid starts at rest, so freezing the initial condition
*is* predicting zero and both score exactly 1.0. The best constant gives 0.585
and the inlet profile imposed everywhere 0.430 — the last is the anchor to
beat.

**Two corrections from the 2026-08-26 APTS review**, inherited from `cyl` along
with the rest of this port:

  * the four boundary terms take their time coordinate from the residual
    batch's own time column, and that column is **sorted** — jaxpi's
    `UniformSampler` sorts every batch by its first coordinate for the causal
    weighting. Taking the head of it enforced the inlet and the outlet over
    only the earliest 1.5% of this case's [0, 6] window (t <= 0.09) and the
    walls over the earliest 4.5%. The draw is now a stride over the whole
    column.
  * `p_out` carried weight 1.0 while `bfs_flow` — the donor these four terms
    are built from — uses 100.0 for the same term, and the config comment
    claimed it followed that 100.0. Set to 100.0.

Both were measured on `cyl`, this case's parent (seed 1, 1360 steps, same wall
clock): 0.7993 head -> 0.6037 stride -> 0.5721 with `p_out` at 100.0. They were
not measured separately here. Seeds 1-5 were re-run on this case after both
changes and the mean moved 0.9896 -> 0.8808; the superseded rows were deleted,
not annotated.
"""
from __future__ import annotations

import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
APTS = Path("~/reference/APTS")
METHOD_LIBS = "apts"
NATIVE = False
SOURCE = "reference/APTS/examples/kolmogorov_flow (unsteady base) + bfs_flow (channel BCs)"
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
        # Logging is not the method: their evaluator predicts on the reference
        # every `log_every_steps`, so its cost would land inside the budget
        # while producing nothing the comparison uses.
        cfg.logging.log_every_steps = steps + 10      # fires once, at step 0

        model = apts_train.train_and_evaluate(cfg, os.getcwd())

        u_fn = jax.jit(jax.vmap(model.u_net, (None, 0, 0, 0)))
        v_fn = jax.jit(jax.vmap(model.v_net, (None, 0, 0, 0)))

        def predict(prm, X):
            """X is (N,3) as (t, x, y)."""
            us, vs = [], []
            for i in range(0, len(X), 16384):
                b = X[i:i + 16384]
                us.append(u_fn(prm, b[:, 0], b[:, 1], b[:, 2]))
                vs.append(v_fn(prm, b[:, 0], b[:, 1], b[:, 2]))
            return {"u": jnp.concatenate(us), "v": jnp.concatenate(vs)}

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
                         "input_dim": 3, "out_dim": 3, "nu": 1e-3, "t_max": 6.0,
                         "causal_chunks": int(cfg.causal.num_chunks),
                         "loss_keys": ["u_ic", "v_ic", "u_in", "v_in",
                                       "uv_wall", "p_out", "ru", "rv", "rc"],
                         "periodicity": "off -- the channel is periodic in neither direction",
                         "residual_mask": "analytic: batch points inside the cylinder zeroed, mean rescaled",
                         "outlet": "p = 0 is expressible here: the pressure is a network output",
                         "ic": "u = v = 0 (analytic: the fluid starts at rest)",
                         "derived_from": "cases/cyl/methods/apts; only U(t), t_max and the IC differ"}}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
