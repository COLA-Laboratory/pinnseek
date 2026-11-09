"""Build the DSGNAR-format reference .npz for the multiscale wave from the task's data.

    ~/PINNSeek/.venv/bin/python make_reference.py

DSGNAR's trainer loads `problem.ref_path` unconditionally (`Reference.load`).
Their package ships a `wave` reference, but it is the 1D constant-speed
standing wave their notebook solves, so one is built here from
`wavems/task/ref_data.csv` — the same 51 x 129 x 129 field the
scoring uses — written through the authors' own `pinn.reference.save_reference`
so the schema is theirs.

This file feeds the trainer's *reporting* only: the reference is used for the
`rel_l2`/`pure_l2` history, the plot grid and the progress display, never by
the optimiser (grep `u_ref_flat` in trainer64.py — every use is an append to a
history list or an event dict).

Coordinate order is **(x, y, t)**, which is what this problem class uses
(`X, Y, T = 0, 1, 2`); the task grid is (t, x, y), so the columns are permuted
here. The plot grid is x by t at the middle y slice, matching how the gs, gl
and kf cases draw their 2D+time fields.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "~/reference/DSGNAR/src")
from pinn.reference import save_reference

TASK = Path("~/wavems/task/ref_data.csv")
OUT = Path(__file__).parent / "wavems_reference.npz"

csv = np.loadtxt(TASK, delimiter=",", skiprows=1, dtype=np.float64)
t, x, y, u = csv[:, 0], csv[:, 1], csv[:, 2], csv[:, 3]

coords = np.stack([x, y, t], axis=1)          # (x, y, t), this problem's order
solution = u.reshape(-1, 1)

T, Xs, Ys = np.unique(t), np.unique(x), np.unique(y)
order = np.lexsort((t, y, x))                 # x-major, then y, then t
U = u[order].reshape(len(Xs), len(Ys), len(T))
assert np.allclose(x[order].reshape(len(Xs), len(Ys), len(T)), Xs[:, None, None])
assert np.allclose(t[order].reshape(len(Xs), len(Ys), len(T)), T[None, None, :])

mid = len(Ys) // 2
save_reference(
    str(OUT),
    coords=coords, solution=solution, channels=["u"],
    plot_axes=("x", "t"), plot_x0=Xs, plot_x1=T,
    plot_grids={"u": U[:, mid, :]},
    slice_val=float(Ys[mid]),
)
print(f"wrote {OUT}  coords {coords.shape}  solution {solution.shape}  "
      f"grid {len(Xs)} x {len(T)} (y slice {Ys[mid]:.4f})")
