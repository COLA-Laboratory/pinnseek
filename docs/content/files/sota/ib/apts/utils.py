import scipy.io
import numpy as np


def get_dataset():
    # Exact soluiton to Riemann problem
    # [sota patch] their file is the same equation on the same domain but a
    # different initial condition; this one carries the task's -sin(pi x).
    # Built by make_dataset.py — see its docstring.
    burger_data = np.load('data_task.npy', allow_pickle=True).item()

    u_ref = burger_data['u']
    t_star = burger_data['t']
    x_star = burger_data['x']

    return u_ref, t_star, x_star
