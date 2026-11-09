"""Build the DSGNAR-format reference .npz for inviscid Burgers from the task's own data.

    ~/PINNSeek/.venv/bin/python make_reference.py

DSGNAR's trainer loads `problem.ref_path` unconditionally (`Reference.load`).
Their package ships a `burgers` reference, but it is the **viscous** solution
their notebook solves (nu = 0.01), not this task's inviscid one, so pointing at
it would report errors against a different flow. One is built here from
`ib/task/ref_data.csv` — the same 101 x 512 field the scoring
uses — written through the authors' own `pinn.reference.save_reference` so the
schema is theirs.

This file feeds the trainer's *reporting* only: the reference is used for the
`rel_l2`/`pure_l2` history, the plot grid and the progress display, never by
the optimiser (grep `u_ref_flat` in trainer64.py — every use is an append to a
history list or an event dict). Training sees the equation, the analytic
initial condition and the collocation samplers, nothing else.

Coordinate order is **(x, t)**, which is what their `BurgersProblem` uses
(`X, T = 0, 1`); the task grid is (t, x), so the columns are swapped here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "~/reference/DSGNAR/src")
from pinn.reference import save_reference

TASK = Path("~/ib/task/ref_data.csv")
OUT = Path(__file__).parent / "ib_reference.npz"

csv = np.loadtxt(TASK, delimiter=",", skiprows=1, dtype=np.float64)
t, x, u = csv[:, 0], csv[:, 1], csv[:, 2]

coords = np.stack([x, t], axis=1)                  # (x, t), their order
solution = u.reshape(-1, 1)

X, T = np.unique(x), np.unique(t)
order = np.lexsort((t, x))                         # x-major, t-minor
U = u[order].reshape(len(X), len(T))
assert np.allclose(x[order].reshape(len(X), len(T)), X[:, None])
assert np.allclose(t[order].reshape(len(X), len(T)), T[None, :])

save_reference(
    str(OUT),
    coords=coords, solution=solution, channels=["u"],
    plot_axes=("x", "t"), plot_x0=X, plot_x1=T,
    plot_grids={"u": U},
)
print(f"wrote {OUT}  coords {coords.shape}  solution {solution.shape}  "
      f"grid {len(X)} x {len(T)}")
