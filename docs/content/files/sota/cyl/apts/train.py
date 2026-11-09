import time
import os
from absl import logging

import jax
from jax import vmap
import jax.numpy as jnp

import wandb

from jaxpi.models import create_lr_schedule, create_optimizer, create_arch, create_train_state
from jaxpi.samplers import UniformSampler, MeshSampler
from jaxpi.checkpointing import create_checkpoint_manager
from jaxpi.logging import Logger
from jaxpi.checkpointing import save_checkpoint, restore_checkpoint
from jaxpi.utils import create_update_scheduler

import models
from utils import get_dataset


def train_time_window(config, model, evaluator, logger, samplers, ref_coords, ref_uv, window_idx):
    # create checkpoint manager
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path, suffix="time_window_{}".format(window_idx))

    # Weights update scheduler
    loss_update = create_update_scheduler(**config.loss_weighting.update_schedule)
    pseudo_time_update = create_update_scheduler(**config.pseudo_time.update_schedule)

    # jit warm up
    start_time = time.time()  # Initialize before the loop
    step_offset = (window_idx - 1) * config.training.max_steps
    print("Waiting for JIT...")
    init_state = model.state  # Initialize prev_state for pseudo-time weights update
    for step in range(config.training.max_steps):
        # Sample mini-batch
        batch = {}
        for key, sampler in samplers.items():
            batch[key] = next(sampler)

        model.state, loss, loss_dict = model.step(model.state, batch)

        if config.pseudo_time.strategy == "dynamic":
            if pseudo_time_update(step):
                model.state = model.update_pts_weights(model.state, init_state, batch['res'])

        # Update weights if necessary
        if config.loss_weighting.strategy == "dynamic":
            if loss_update(step):
                model.state = model.update_loss_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                end_time = time.time()
                # Compute evaluation metrics and log to W&B
                log_dict = evaluator(model, model.state, loss_dict, batch,
                                     ref_coords[::16], ref_uv[::16])   # [sota patch]
                wandb.log(log_dict, step + step_offset)

                # Report training metrics
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = time.time()

        # Save checkpoint
        if config.saving.save_every_steps > 0:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                    step + 1
            ) == config.training.max_steps:
                save_checkpoint(ckpt_mngr, model.state)

    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    if config.saving.save_every_steps > 0:
        save_checkpoint(ckpt_mngr, model.state)
        ckpt_mngr.wait_until_finished()

    return model


def train_and_evaluate(config, workdir):
    # [sota patch] the task ships an initial condition and scoring snapshots,
    # not a stored trajectory, so there is no time_range to slice: the window is
    # the whole [0, 1.2] problem.md specifies, and the box is the channel.
    ic_mesh, ic_uv, ref_coords, ref_uv, nu = get_dataset()

    t0 = 0.0
    t1 = float(config.t_max)
    dom = jnp.array([[t0, t1], [0.0, 2.2], [0.0, 0.41]])

    # Initialize the model
    lr = create_lr_schedule(config.optim)
    tx = create_optimizer(config.optim, lr)
    arch = create_arch(config.arch)
    state = create_train_state(config, tx, arch)

    # Initialize model
    model = models.NavierStokes2D(config, lr, tx, arch, state, t_max=t1, nu=nu)   # [sota patch] nu from problem.md

    # Initialize evaluator
    evaluator = models.NavierStokes2DEvaluator(config)

    # Initialize logger
    logger = Logger()

    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))
        if config.training.transfer_learning:
            if idx > 0:
                # restore the checkpoint from the previous time window
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
                ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path, suffix="time_window_{}".format(idx))

                state = restore_checkpoint(ckpt_mngr, model.state)
                model.state = create_train_state(
                    config,
                    tx=tx,
                    arch=arch,
                    params=state.params,

                )
                print("Restored checkpoint from previous time window {} at step {}".format(idx, state.step))

        # Initialize the samplers
        # [sota patch] the IC lives on its own 441 x 83 grid, cylinder removed
        ics_sampler = MeshSampler(ic_mesh, ic_uv, config.training.batch_size)
        res_sampler = UniformSampler(dom, config.training.batch_size)

        samplers = {
            "ics": iter(ics_sampler),
            "res": iter(res_sampler),
        }

        # Training the current time window
        model = train_time_window(
            config, model, evaluator, logger, samplers, ref_coords, ref_uv, idx + 1)

        #  Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            u0, v0, _ = vmap(model.neural_net, in_axes=(None, None, 0, 0))(
                model.state.params, t1 * idx / config.training.num_time_windows,
                ic_mesh[:, 0], ic_mesh[:, 1]
            )
            ic_uv = jnp.stack([u0, v0], axis=-1)

    # [sota patch] return the trained model, so the benchmark can score it
    # without going through a checkpoint. Their own `train_and_evaluate` ends
    # here and returns None -- the checkpoint on disk is how their `eval.ipynb`
    # picks the weights up, and its `suffix="time_window_{}"` naming is easy to
    # get wrong from outside. Nothing about the training changes.
    return model
