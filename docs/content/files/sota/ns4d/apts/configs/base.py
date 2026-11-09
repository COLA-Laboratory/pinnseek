import ml_collections

import jax.numpy as jnp


def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "JAXPI-NS4D"   # [sota patch]
    wandb.name = "baseline"
    wandb.tag = None

    # [sota patch] the task has no stored trajectory to slice; the window is
    # the whole [0, 5] problem.md specifies.
    config.t_max = 5.0

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 3
    arch.hidden_dim = 256
    arch.out_dim = 3
    arch.activation = "tanh"
    # [sota patch] three periodic spatial axes instead of two; the input is
    # (t, x, y, z), so the spatial axes are 1, 2 and 3.
    # The value is a WAVENUMBER, not a length: `jaxpi/archs.py:PeriodEmbs`
    # embeds cos(period*x)/sin(period*x), so period = 2*pi/L. Their Kolmogorov
    # box is L = 1 and uses 2*pi; this task's box is L = 2*pi, so the value is
    # 1.0 -- which is exactly what their own `examples/ks` (also L = 2*pi) sets.
    # Carrying their 2*pi over unchanged would have imposed period-1
    # periodicity on a 2*pi-long axis: a constraint the true solution does not
    # satisfy, rather than the task's boundary condition.
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (1.0, 1.0, 1.0), "axis": (1, 2, 3),
         "trainable": (False, False, False)}
    )
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
    # [sota patch] three velocity components at t=0 and three vorticity
    # residual components; their 100.0 on the IC and 1.0 on the residual stand
    loss_weighting.loss_weights = ml_collections.ConfigDict(
        {"u_ic": 100.0, "v_ic": 100.0, "w_ic": 100.0,
         "rx": 1.0, "ry": 1.0, "rz": 1.0, "rc": 1.0}
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
        {"rx": 1.0, "ry": 1.0, "rz": 1.0, "rc": 1.0})   # [sota patch]
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
    config.input_dim = 4   # [sota patch] (t, x, y, z)

    # Integer for PRNG random seed.
    config.seed = 42

    return config
