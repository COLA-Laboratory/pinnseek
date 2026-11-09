"""Build the DSGNAR-format reference .npz for POISSONCG from the task's own data.

    ~/PINNSeek/.venv/bin/python make_reference.py

DSGNAR's trainer loads `problem.ref_path` unconditionally (`Reference.load`),
and their package ships nothing for this problem, so one is built here from
`poissoncg/task/ref_data.csv` — the same 65202 points the
scoring uses — written through the authors' own `pinn.reference.save_reference`
so the schema is theirs.

This file feeds the trainer's *reporting* only: the reference is used for the
`rel_l2`/`pure_l2` history, the plot grid and the progress display, never by
the optimiser (grep `u_ref_flat` in trainer64.py — every use is an append to a
history list or an event dict). Training sees the equation and the collocation
samplers, nothing else.

The reference here is an unstructured cloud, not a tensor grid, so the plot
grid cannot be a reshape. It is built by nearest-neighbour lookup on a
`z = slice_val` slab — good enough for a picture, and the picture is never
drawn anyway (`plot_every` is set past the end of the run).

Steady, 3 inputs: coords are (x, y, z) with no time column, and `slice_val`
holds z fixed on the plot grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "~/reference/DSGNAR/src")
from pinn.reference import save_reference

TASK = Path("~/poissoncg/task/ref_data.csv")
OUT = Path(__file__).parent / "poissoncg_reference.npz"
SLICE_Z = 0.25            # below the sphere band and below the k jump at z = 0.5
NG = 96

csv = np.loadtxt(TASK, delimiter=",", skiprows=1, dtype=np.float64)
coords, u = csv[:, 0:3], csv[:, 3]
solution = u[:, None]

# nearest-neighbour picture on a z = SLICE_Z slab
band = np.abs(coords[:, 2] - SLICE_Z) < 0.06
pts, vals = coords[band][:, :2], u[band]
X = np.linspace(0.0, 1.0, NG)
Y = np.linspace(0.0, 1.0, NG)
GX, GY = np.meshgrid(X, Y, indexing="ij")
q = np.stack([GX.ravel(), GY.ravel()], axis=1)
d2 = ((q[:, None, 0] - pts[None, :, 0]) ** 2
      + (q[:, None, 1] - pts[None, :, 1]) ** 2)
U = vals[d2.argmin(axis=1)].reshape(NG, NG)

save_reference(
    str(OUT),
    coords=coords, solution=solution, channels=["u"],
    plot_axes=("x", "y"), plot_x0=X, plot_x1=Y,
    plot_grids={"u": U}, slice_val=SLICE_Z,
)
print(f"wrote {OUT}  coords {coords.shape}  solution {solution.shape}  "
      f"slab points {band.sum()}  grid {NG} x {NG}")
