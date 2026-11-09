"""Build the DSGNAR-format reference .npz for Kolmogorov flow from the task's own data.

    ~/PINNSeek/.venv/bin/python make_reference.py

DSGNAR's trainer loads `problem.ref_path` unconditionally (`Reference.load`)
and its shipped `src/pinn/references/` has no Kolmogorov flow entry, so one is built
here from `kf/task/ref_data.csv` — the same 40x100x100 field the
scoring uses — written through the authors' own `pinn.reference.save_reference`
so the schema is theirs.

This file feeds the trainer's *reporting* only: the reference is used for the
`rel_l2`/`pure_l2` history, the plot grid and the progress display, never by
the optimiser (grep `u_ref_flat` in trainer64.py — every use is an append to a
history list or an event dict). Training sees the equation, the analytic
initial condition and the collocation samplers, nothing else.

Coordinate order is DSGNAR's: (x, y, t), with t last.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "~/reference/DSGNAR/src")
from pinn.reference import save_reference

TASK = Path("~/kf/task/ref_data.csv")
OUT = Path(__file__).parent / "kf_reference.npz"

csv = np.loadtxt(TASK, delimiter=",", skiprows=1, dtype=np.float64)
t, x, y, u, v = (csv[:, i] for i in range(5))
T, X, Y = np.unique(t), np.unique(x), np.unique(y)

coords = np.stack([x, y, t], axis=1)                 # DSGNAR order
solution = np.stack([u, v], axis=1)

# plot grid: u,v over (x, t) at the y plane nearest 0
U = u.reshape(len(T), len(X), len(Y))
V = v.reshape(len(T), len(X), len(Y))
j = int(np.argmin(np.abs(Y)))
save_reference(
    str(OUT),
    coords=coords, solution=solution, channels=["u", "v"],
    plot_axes=("x", "t"), plot_x0=X, plot_x1=T,
    plot_grids={"u": U[:, :, j].T.copy(), "v": V[:, :, j].T.copy()},
    slice_val=float(Y[j]),
)
print(f"wrote {OUT}  coords {coords.shape}  solution {solution.shape}  "
      f"y slice {Y[j]:.6f}")
