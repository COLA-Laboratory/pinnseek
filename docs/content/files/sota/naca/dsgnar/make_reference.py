"""Build the DSGNAR-format reference .npz for NACA0012 from the task's own data.

    ~/PINNSeek/.venv/bin/python make_reference.py

DSGNAR's trainer loads `problem.ref_path` unconditionally (`Reference.load`)
and its shipped `src/pinn/references/` has no airfoil entry, so one is built
here from `naca/task/ref_data.csv` — the same 801 x 401 field
the scoring uses — written through the authors' own
`pinn.reference.save_reference` so the schema is theirs.

Only the fluid points (`phi <= 0`) go in: the airfoil interior carries solver
fill values that are not a solution of anything, and including them would make
the trainer's progress readout meaningless. The plot grid keeps the full
rectangle with the interior set to NaN-free zeros, since it is only drawn.

This file feeds the trainer's *reporting* only: the reference is used for the
`rel_l2`/`pure_l2` history, the plot grid and the progress display, never by
the optimiser (grep `u_ref_flat` in trainer64.py — every use is an append to a
history list or an event dict). Training sees the equation and the collocation
samplers, nothing else.

This case is steady, so the coordinates are just (x, y) — no time column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "~/reference/DSGNAR/src")
from pinn.reference import save_reference

TASK = Path("~/naca/task/ref_data.csv")
OUT = Path(__file__).parent / "naca_reference.npz"

csv = np.loadtxt(TASK, delimiter=",", skiprows=1, dtype=np.float64)
x, y, u, v, phi = csv[:, 0], csv[:, 1], csv[:, 2], csv[:, 3], csv[:, 5]

fluid = phi <= 0.0
coords = np.stack([x[fluid], y[fluid]], axis=1)      # steady: (x, y), no time
solution = np.stack([u[fluid], v[fluid]], axis=1)

X, Y = np.unique(x), np.unique(y)
order = np.lexsort((y, x))
U = u[order].reshape(len(X), len(Y))
V = v[order].reshape(len(X), len(Y))
assert np.allclose(x[order].reshape(len(X), len(Y)), X[:, None])
assert np.allclose(y[order].reshape(len(X), len(Y)), Y[None, :])

save_reference(
    str(OUT),
    coords=coords, solution=solution, channels=["u", "v"],
    plot_axes=("x", "y"), plot_x0=X, plot_x1=Y,
    plot_grids={"u": U, "v": V},
)
print(f"wrote {OUT}  coords {coords.shape}  solution {solution.shape}  "
      f"grid {len(X)} x {len(Y)}  (fluid {fluid.mean():.4f})")
