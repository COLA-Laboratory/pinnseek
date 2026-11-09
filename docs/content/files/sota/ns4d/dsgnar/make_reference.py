"""Build the DSGNAR-format reference .npz for NS4D from the task's own data.

    ~/PINNSeek/.venv/bin/python make_reference.py

DSGNAR's trainer loads `problem.ref_path` unconditionally (`Reference.load`),
and their package ships nothing 4-dimensional, so one is built here from
`ns4d/task/ref_data.csv` -- the same 160000 points and the same
three vorticity components the scoring uses -- written through the authors' own
`pinn.reference.save_reference` so the schema is theirs.

This file feeds the trainer's *reporting* only: the reference is used for the
`rel_l2`/`pure_l2` history, the plot grid and the progress display, never by
the optimiser (grep `u_ref_flat` in trainer64.py -- every use is an append to a
history list or an event dict). Training sees the equation, the initial
condition and the collocation samplers, nothing else.

Coordinate order is DSGNAR's: spatial first, time last, so (x, y, z, t) rather
than the task's (t, x, y, z). The plot grid is the (x, t) plane at fixed
y = z = slice_val, taken straight off the 20^4 tensor grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "~/reference/DSGNAR/src")
from pinn.reference import save_reference

TASK = Path("~/ns4d/task/ref_data.csv")
OUT = Path(__file__).parent / "ns4d_reference.npz"

csv = np.loadtxt(TASK, delimiter=",", skiprows=1, dtype=np.float64)
t, x, y, z = csv[:, 0], csv[:, 1], csv[:, 2], csv[:, 3]
w = csv[:, 7:10]                                  # wx, wy, wz -- the scored field

coords = np.stack([x, y, z, t], axis=1)           # DSGNAR order: space, then time
T, X = np.unique(t), np.unique(x)
Y, Z = np.unique(y), np.unique(z)
slice_val = float(Y[len(Y) // 2])
assert np.isclose(slice_val, Z[len(Z) // 2]), "y and z grids differ"

sel = np.isclose(y, slice_val) & np.isclose(z, slice_val)
order = np.lexsort((t[sel], x[sel]))              # x major, t minor
grids = {ch: w[sel][order, i].reshape(len(X), len(T))
         for i, ch in enumerate(("wx", "wy", "wz"))}
assert x[sel][order].reshape(len(X), len(T))[:, 0].shape == X.shape

save_reference(
    str(OUT),
    coords=coords, solution=w, channels=["wx", "wy", "wz"],
    plot_axes=("x", "t"), plot_x0=X, plot_x1=T,
    plot_grids=grids, slice_val=slice_val,
)
print(f"wrote {OUT}  coords {coords.shape}  solution {w.shape}  "
      f"grid {len(X)} x {len(T)} at y = z = {slice_val:.4f}")
