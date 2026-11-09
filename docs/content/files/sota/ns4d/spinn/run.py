"""SPINN on 3D unsteady Navier-Stokes — the authors' own published case.

This is the second method/case pair in the comparison where nothing had to be
ported. `reference/SPINN/navier_stokes4d.py` *is* this task: same domain
`[0,5] x [0,2pi]^3`, same `nu = 0.05`, same forcing, and — checked point by
point — the same solution. The task's `ref_data.csv` agrees with their
`utils/data_utils.py:navier_stokes4d_exact_w` to 8e-10 in relative L2 on all
three vorticity components, and with `navier_stokes4d_exact_u` to the same
tolerance on the velocity. The task's `eval.py` says as much in its own
docstring ("matching SPINN"), including the per-component averaging of the
relative L2. NATIVE = True.

Everything is imported from `reference/SPINN` unmodified and nothing is copied
into this directory:

  * `SPINN4d` from networks/physics_informed_neural_networks.py,
  * `vorx`/`vory`/`vorz` from utils/vorticity.py — the autodiff curl the task
    requires, since the vorticity is not a network output,
  * `hvp_fwdfwd` from networks/hessian_vector_products.py,
  * `_spinn_train_generator_navier_stokes4d` from utils/data_generators.py,
    which builds the collocation grid, the forcing, the initial condition and
    the six boundary sets,
  * `update_model` and the `apply_model_spinn` loss from navier_stokes4d.py
    itself — imported, not re-typed, so the residual in force here is
    literally the one in their file.

Hyperparameters are that file's own defaults: modified_mlp, n_layers=5,
features=64, r=128, out_dim=3, nc=32, nu=0.05, lbda_c=100, lbda_ic=10,
Adam(1e-3), 50000 epochs, collocation resampled every 100 epochs. Adam runs at
a constant learning rate with no schedule tied to the step count, so extra
epochs are simply more training and filling the budget is the right mapping
(GUIDE.md rule 5).

**Two things about their published recipe that have to be said out loud.**

  1. `initial_loss` supervises `u` and `w` at t = 0 from the analytic solution.
     That is the same information `task/ns4d_ic.csv` supplies (checked: their
     analytic `w` at t = 0 matches the shipped IC to 2.5e-9 on all 32768
     points), so it is not extra data.

  2. `boundary_loss` supervises `w` on six hyperplanes **from the analytic
     solution too** — and the task supplies no such data. Its planes are also
     hard-coded as `x = -1` and `x = +1` (likewise y, z), which lie outside the
     task's `[0, 2pi]` domain; the fields are pi-periodic in x and y and
     2pi-periodic in z, so those planes are equivalent to interior ones. Either
     way this is ground-truth supervision on six slices that the hero does not
     get. It is kept exactly as published, because the comparison reports what
     the method's own recipe does, and because where the choice is free
     GUIDE.md says to take the treatment that flatters the baseline. It is
     recorded in `info` as `boundary_supervision`.

Reading the score: predicting zero gives exactly 1.0 and freezing the initial
condition 1.342 — the field decays to 10.5% of its amplitude by t = 5, so
holding t = 0 is worse than giving up.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).parent
METHOD_LIBS = "spinn"
NATIVE = True
SOURCE = "reference/SPINN/navier_stokes4d.py (their own 4D Navier-Stokes case, verbatim)"
SPINN_ROOT = "~/reference/SPINN"

NC = 32
FEATURES = 64
N_LAYERS = 5
RANK = 128
OUT_DIM = 3
LR = 1e-3
MLP = "modified_mlp"
NU = 0.05
LBDA_C = 100
LBDA_IC = 10
PUBLISHED_EPOCHS = 50000


def train(*, steps: int, seed: int, gpu: str) -> dict:
    if SPINN_ROOT not in sys.path:
        sys.path.insert(0, SPINN_ROOT)

    import jax
    import jax.numpy as jnp
    import optax
    from utils.data_generators import generate_train_data
    from utils.training_utils import setup_networks, update_model
    from utils.vorticity import vorx, vory, vorz
    import importlib.util

    # their navier_stokes4d.py, imported for `apply_model_spinn` — the loss in
    # force is the one in their file, not a copy of it
    spec = importlib.util.spec_from_file_location(
        "spinn_ns4d", f"{SPINN_ROOT}/navier_stokes4d.py")
    ns4d = importlib.util.module_from_spec(spec)
    sys.modules["spinn_ns4d"] = ns4d
    spec.loader.exec_module(ns4d)          # guarded by `if __name__ == '__main__'`
    apply_model_spinn = ns4d.apply_model_spinn

    args = SimpleNamespace(model="spinn", equation="navier_stokes4d",
                           nc=NC, nc_test=20, seed=seed, lr=LR,
                           epochs=int(steps), mlp=MLP, n_layers=N_LAYERS,
                           features=FEATURES, r=RANK, out_dim=OUT_DIM,
                           nu=NU, lbda_c=LBDA_C, lbda_ic=LBDA_IC)

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key, 2)
    apply_fn, params = setup_networks(args, subkey)

    optim = optax.adam(learning_rate=LR)
    state = optim.init(params)

    key, subkey = jax.random.split(key, 2)
    train_data = generate_train_data(args, subkey)

    for e in range(1, int(steps) + 1):
        if e % 100 == 0:                       # theirs: resample every 100
            key, subkey = jax.random.split(key, 2)
            train_data = generate_train_data(args, subkey)
        loss, gradient = apply_model_spinn(apply_fn, params, NU, LBDA_C, LBDA_IC,
                                           *train_data)
        params, state = update_model(optim, gradient, params, state)
        if e % 10000 == 0:
            print(f'Epoch: {e}/{steps} --> total loss: {loss:.8f}')

    def predict(prm, X):
        """X is (N,4) as (t, x, y, z). The reference is a full 20^4 tensor grid,
        so the factorised prediction is built on its distinct axis values and
        the points are gathered out of it. The vorticity is the autodiff curl
        of the velocity, through their own `vorx`/`vory`/`vorz`."""
        Xn = np.asarray(X)
        vals, inv = [], []
        for k in range(4):
            v, i = np.unique(Xn[:, k], return_inverse=True)
            vals.append(jnp.asarray(v[:, None].astype(np.float32)))
            inv.append(i)
        t, x, y, z = vals
        wx = np.asarray(vorx(apply_fn, prm, t, x, y, z))
        wy = np.asarray(vory(apply_fn, prm, t, x, y, z))
        wz = np.asarray(vorz(apply_fn, prm, t, x, y, z))
        g = (inv[0], inv[1], inv[2], inv[3])
        return {"wx": jnp.asarray(wx[g]), "wy": jnp.asarray(wy[g]),
                "wz": jnp.asarray(wz[g])}

    return {"params": params, "predict": predict,
            "info": {"mlp": MLP, "n_layers": N_LAYERS, "features": FEATURES,
                     "r": RANK, "out_dim": OUT_DIM, "nc": NC, "lr": LR,
                     "nu": NU, "lbda_c": LBDA_C, "lbda_ic": LBDA_IC,
                     "epochs": int(steps), "published_epochs": PUBLISHED_EPOCHS,
                     "template": "navier_stokes4d.py, unmodified",
                     "resample_every": 100,
                     "initial_supervision": "analytic u and w at t=0 (equals the shipped IC to 2.5e-9)",
                     "boundary_supervision": ("analytic w on six hyperplanes "
                                              "x,y,z = -1 and +1 -- ground truth the "
                                              "task does not supply; kept as published"),
                     "vorticity": "autodiff curl of the velocity (their vorx/vory/vorz)"}}
