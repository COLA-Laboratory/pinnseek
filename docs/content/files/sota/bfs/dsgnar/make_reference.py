"""Build the DSGNAR-format reference .npz for backward-facing step from the task's own data.

    ~/PINNSeek/.venv/bin/python make_reference.py

DSGNAR's trainer loads `problem.ref_path` unconditionally (`Reference.load`)
and its shipped `src/pinn/references/` has no backward-facing step entry, so one is built
here from `bfs/task/ref_data.csv` — the same 40x100x100 field the
scoring uses — written through the authors' own `pinn.reference.save_reference`
so the schema is theirs.

This file feeds the trainer's *reporting* only: the reference is used for the
`rel_l2`/`pure_l2` history, the plot grid and the progress display, never by
the optimiser (grep `u_ref_flat` in trainer64.py — every use is an append to a
history list or an event dict). Training sees the equation, the analytic
initial condition and the collocation samplers, nothing else.

This case is steady, so the coordinates are just (x, y) — no time column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "~/reference/DSGNAR/src")
from pinn.reference import save_reference

TASK = Path("~/bfs/task/ref_data.csv")
OUT = Path(__file__).parent / "bfs_reference.npz"

csv = np.loadtxt(TASK, delimiter=",", skiprows=1, dtype=np.float64)
x, y, u, v = csv[:, 0], csv[:, 1], csv[:, 2], csv[:, 3]

coords = np.stack([x, y], axis=1)                    # steady: (x, y), no time
solution = np.stack([u, v], axis=1)

# plot grid: the vtu mesh is a structured 481 x 65 grid, so u,v reshape onto it
X, Y = np.unique(x), np.unique(y)
order = np.lexsort((y, x))
U = u[order].reshape(len(X), len(Y))
V = v[order].reshape(len(X), len(Y))
save_reference(
    str(OUT),
    coords=coords, solution=solution, channels=["u", "v"],
    plot_axes=("x", "y"), plot_x0=X, plot_x1=Y,
    plot_grids={"u": U, "v": V},
)
print(f"wrote {OUT}  coords {coords.shape}  solution {solution.shape}  "
      f"grid {len(X)} x {len(Y)}")
