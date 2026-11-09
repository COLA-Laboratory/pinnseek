"""Build the DSGNAR-format reference .npz for CYL from the task's own data.

    ~/PINNSeek/.venv/bin/python make_reference.py

DSGNAR's trainer loads `problem.ref_path` unconditionally (`Reference.load`),
and their package ships nothing like this flow, so one is built here from
`cyl/task/ref_data.csv` -- the same 220503 points and the same
two velocity components the scoring uses -- written through the authors' own
`pinn.reference.save_reference` so the schema is theirs.

This file feeds the trainer's *reporting* only: the reference is used for the
`rel_l2`/`pure_l2` history, the plot grid and the progress display, never by
the optimiser (grep `u_ref_flat` in trainer64.py -- every use is an append to a
history list or an event dict). Training sees the equation, the initial
condition and the collocation samplers, nothing else.

Coordinate order is DSGNAR's: spatial first, time last, so (x, y, t) rather
than the task's (t, x, y). The plot grid is the (x, t) plane at fixed
y = slice_val, filled by nearest neighbour -- the cylinder punches a hole in
the tensor grid, so a reshape would not line up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "~/reference/DSGNAR/src")
from pinn.reference import save_reference

TASK = Path("~/cyl/task/ref_data.csv")
OUT = Path(__file__).parent / "cyl_reference.npz"

csv = np.loadtxt(TASK, delimiter=",", skiprows=1, dtype=np.float64)
t, x, y = csv[:, 0], csv[:, 1], csv[:, 2]
uv = csv[:, 3:5]

coords = np.stack([x, y, t], axis=1)              # DSGNAR order
T, X = np.unique(t), np.unique(x)
Y = np.unique(y)
slice_val = float(Y[len(Y) // 2])

grids = {}
sel = np.isclose(y, slice_val)
for i, ch in enumerate(("u", "v")):
    g = np.zeros((len(X), len(T)))
    xi = np.searchsorted(X, x[sel])
    ti = np.searchsorted(T, t[sel])
    g[xi, ti] = uv[sel][:, i]
    grids[ch] = g

save_reference(
    str(OUT),
    coords=coords, solution=uv, channels=["u", "v"],
    plot_axes=("x", "t"), plot_x0=X, plot_x1=T,
    plot_grids=grids, slice_val=slice_val,
)
print(f"wrote {OUT}  coords {coords.shape}  solution {uv.shape}  "
      f"grid {len(X)} x {len(T)} at y = {slice_val:.4f} ({sel.sum()} points)")
