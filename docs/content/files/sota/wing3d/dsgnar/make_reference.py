"""Build the DSGNAR-format reference .npz for WING3D from the task's own data.

    ~/PINNSeek/.venv/bin/python make_reference.py

DSGNAR's trainer loads `problem.ref_path` unconditionally (`Reference.load`),
and their package ships nothing three-dimensional in space, so one is built
here from `wing3d/task/ref_data.npy` -- the same 388475 points
the scoring uses -- written through the authors' own
`pinn.reference.save_reference` so the schema is theirs.

This file feeds the trainer's *reporting* only: the reference is used for the
`rel_l2`/`pure_l2` history, the plot grid and the progress display, never by
the optimiser (grep `u_ref_flat` in trainer64.py -- every use is an append to a
history list or an event dict). Training sees the equation, the geometry files
and the collocation samplers, nothing else.

The scored quantity is the velocity *magnitude*, but the network emits
components, so the three components are carried as the channels and the
magnitude is formed by ../../problem.py at scoring time -- the same split the
naca case uses.

Steady, 3 inputs: coords are (x, y, z) with no time column, and `slice_val`
holds z fixed on the plot grid. The reference is an unstructured
finite-volume cloud, so the plot grid is a nearest-neighbour picture on a
z-slab; it is never drawn (`plot_every` is set past the end of the run).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "~/reference/DSGNAR/src")
from pinn.reference import save_reference

TASK = Path("~/wing3d/task/ref_data.npy")
OUT = Path(__file__).parent / "wing3d_reference.npz"
SLICE_Z = 0.5
NG = 96

ref = np.load(TASK)                       # x,y,z,u,v,w,p,phi
coords = ref[:, 0:3].astype(np.float64)
sol = ref[:, 3:6].astype(np.float64)      # u, v, w

band = np.abs(coords[:, 2] - SLICE_Z) < 0.08
pts, vals = coords[band][:, :2], sol[band]
X = np.linspace(-0.2, 2.2, NG)
Y = np.linspace(-0.35, 0.35, NG)
GX, GY = np.meshgrid(X, Y, indexing="ij")
q = np.stack([GX.ravel(), GY.ravel()], axis=1)
step = 4096
idx = np.empty(len(q), dtype=np.int64)
for k in range(0, len(q), step):
    d2 = ((q[k:k + step, None, 0] - pts[None, :, 0]) ** 2
          + (q[k:k + step, None, 1] - pts[None, :, 1]) ** 2)
    idx[k:k + step] = d2.argmin(axis=1)
grids = {ch: vals[idx, i].reshape(NG, NG) for i, ch in enumerate(("u", "v", "w"))}

save_reference(
    str(OUT),
    coords=coords, solution=sol, channels=["u", "v", "w"],
    plot_axes=("x", "y"), plot_x0=X, plot_x1=Y,
    plot_grids=grids, slice_val=SLICE_Z,
)
print(f"wrote {OUT}  coords {coords.shape}  solution {sol.shape}  "
      f"slab points {int(band.sum())}  grid {NG} x {NG}")
