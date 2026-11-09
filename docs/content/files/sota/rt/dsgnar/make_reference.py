"""Build the DSGNAR-format reference .npz for Rayleigh-Taylor from the task's own data.

    ~/PINNSeek/.venv/bin/python make_reference.py

DSGNAR's trainer loads `problem.ref_path` unconditionally and their package
ships nothing like this, so one is built from
`rt/task/ref_data.csv` — the same 20 x 100 x 200 field the
scoring uses — through the authors' own `pinn.reference.save_reference`.

**Only T is stored**, because only T is scored. The velocity is trained (the
buoyancy coupling needs it) but does not enter the metric, and putting it in
the reference would make the trainer's progress readout measure something the
comparison does not.

This file feeds reporting only: `rel_l2`/`pure_l2` history, the plot grid and
the progress display, never the optimiser.

Coordinate order is **(x, y, t)**, which is what this problem class uses; the
task grid is (t, x, y), so the columns are permuted. The plot grid is y by t at
the middle x slice — the interface moves in y, so that is the informative cut.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "~/reference/DSGNAR/src")
from pinn.reference import save_reference

TASK = Path("~/rt/task/ref_data.csv")
OUT = Path(__file__).parent / "rt_reference.npz"

csv = np.loadtxt(TASK, delimiter=",", skiprows=1, dtype=np.float64)
t, x, y, T = csv[:, 0], csv[:, 1], csv[:, 2], csv[:, 5]

coords = np.stack([x, y, t], axis=1)          # (x, y, t), this problem's order
solution = T.reshape(-1, 1)

Ts, Xs, Ys = np.unique(t), np.unique(x), np.unique(y)
order = np.lexsort((t, y, x))                 # x-major, then y, then t
G = T[order].reshape(len(Xs), len(Ys), len(Ts))
assert np.allclose(x[order].reshape(G.shape), Xs[:, None, None])
assert np.allclose(t[order].reshape(G.shape), Ts[None, None, :])

mid = len(Xs) // 2
save_reference(
    str(OUT),
    coords=coords, solution=solution, channels=["T"],
    plot_axes=("y", "t"), plot_x0=Ys, plot_x1=Ts,
    plot_grids={"T": G[mid, :, :]},
    slice_val=float(Xs[mid]),
)
print(f"wrote {OUT}  coords {coords.shape}  solution {solution.shape}  "
      f"grid {len(Ys)} x {len(Ts)} (x slice {Xs[mid]:.4f})")
