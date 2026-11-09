import ml_collections

import jax.numpy as jnp


def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "JAXPI-Cylinder"   # [sota patch]
    wandb.name = "baseline"
    wandb.tag = None

    # [sota patch] the task has no stored trajectory to slice; the window is
    # the whole [0, 1.2] problem.md specifies.
    config.t_max = 1.2

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 3
    arch.hidden_dim = 256
    arch.out_dim = 3
    arch.activation = "tanh"
    # [sota patch] the channel is not periodic in either direction -- inlet,
    # outlet and two walls -- so the periodic embedding their box relies on is
    # switched off and the conditions are carried by the loss instead.
    arch.periodicity = None
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 4.0, "embed_dim": 256}
    )
    arch.nonlinearity = 0.2

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "soap"
    optim.lr_schedule = "exponential_decay"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.warmup_steps = 2000
    optim.staircase = False
    optim.schedule_free = True

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000
    training.batch_size = 8192
    training.num_time_windows = 1
    training.transfer_learning = True

    # Global weightings for different loss terms
    config.loss_weighting = loss_weighting = ml_collections.ConfigDict()
    loss_weighting.strategy = "dynamic"  # "constant" or "constant", constant means fixed weights
    # [sota patch] their two IC terms and three residual terms, plus the four
    # boundary terms their periodic box does not need. Every boundary weight is
    # 100.0: that is `examples/bfs_flow/configs/base.py`'s own value for these
    # same four term names (`u_in`, `v_in`, `p_out`, `u_noslip`, `v_noslip`),
    # and bfs_flow is where the boundary construction here comes from. It also
    # matches their 100.0 on the initial condition. `p_out` was 1.0 here until
    # a review caught the disagreement with both the donor and this comment;
    # measured on cyl, seed 1, 1360 steps: 0.6037 at 1.0, 0.5721 at 100.0.
    loss_weighting.loss_weights = ml_collections.ConfigDict(
        {"u_ic": 100.0, "v_ic": 100.0,
         "u_in": 100.0, "v_in": 100.0, "uv_wall": 100.0, "p_out": 100.0,
         "ru": 1.0, "rv": 1.0, "rc": 1.0}
    )
    loss_weighting.update_schedule = ml_collections.ConfigDict({
        "start": 100,
        "every": 1000,  # used when schedule="constant"
    })
    loss_weighting.momentum = 0.9

    # Pseudo-time stepping for the PDE residuals
    config.pseudo_time = pseudo_time = ml_collections.ConfigDict()
    pseudo_time.enabled = False
    pseudo_time.strategy = "constant"  # "dynamic" or "constant" constant means fixed weights
    pseudo_time.pts_weights = ml_collections.ConfigDict(
        {"ru": 1.0, "rv": 1.0, "rc": 1.0})
    pseudo_time.update_schedule = ml_collections.ConfigDict({
        "start": 100,
        "every": 1000,  # used when schedule="constant"
    })
    pseudo_time.momentum = 0.9
    pseudo_time.shrink = shrink = ml_collections.ConfigDict()
    shrink.enabled = True
    shrink.start_log_drop = 2.0
    shrink.end_log_drop = 6.0
    shrink.min_factor = 0.1

    config.causal = causal = ml_collections.ConfigDict()
    causal.enabled = True
    causal.num_chunks = 16
    causal.tol = 1.0

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_lr = True
    logging.log_losses = True
    logging.log_raw_losses = False
    logging.log_loss_weights = True
    logging.log_pts_weights = True
    logging.log_causal_weights = True
    logging.log_grads = False
    logging.log_nonlinearities = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 2

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
