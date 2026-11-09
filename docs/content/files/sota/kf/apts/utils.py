"""APTS's examples/kolmogorov_flow/utils.py, verbatim except the data path.

Copy of reference/APTS/examples/kolmogorov_flow/utils.py with exactly one
change, marked `[sota patch]`: the filename it loads. Everything else is
theirs.
"""
import jax.numpy as jnp


def get_dataset(time_range=[0.0, 1.0]):
    # [sota patch] the repository ships data/kolmogorov_flow.npy and no
    # kolmogorov_flow_Re10000.npy, so this example cannot load anything as
    # published. The shipped file stores nu = 5e-4 (Re = 2000), which is the
    # task's coefficient. See run.py.
    data = jnp.load("data/kolmogorov_flow.npy", allow_pickle=True).item()
    w_ref = jnp.array(data["vorticity"])
    velocity = jnp.array(data["velocity"])

    u_ref = velocity[..., 0]
    v_ref = velocity[..., 1]

    t_star = jnp.array(data["t"]).flatten()

    start_time_step = int(time_range[0] * len(t_star))
    end_time_step = int(time_range[1] * len(t_star))
    t_star = t_star[start_time_step: end_time_step] - t_star[start_time_step]

    # Truncate data
    u_ref = u_ref[start_time_step:end_time_step]
    v_ref = v_ref[start_time_step:end_time_step]
    w_ref = w_ref[start_time_step:end_time_step]

    coords = jnp.array(data["coords"])
    nu = data["nu"]

    return u_ref, v_ref, w_ref, t_star, coords, nu
