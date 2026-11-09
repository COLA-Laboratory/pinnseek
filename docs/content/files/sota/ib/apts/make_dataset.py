"""Build APTS's dataset dict for inviscid Burgers from the task's own reference.

    ~/PINNSeek/.venv/bin/python make_dataset.py

`utils.py:get_dataset` loads a pickled dict with keys `u` (nt, nx), `t` (nt)
and `x` (nx). The file the repository ships, `data/inviscid_burger.npy`, is the
*same equation on the same domain over the same time span* but a different
initial condition — a decaying bump (their docstring calls it the Riemann
problem), where this task starts from `-sin(pi x)`. Their `x` is 500 cell
centres, the task's is 512.

The initial condition is data in their code (`train.py` takes `u0 = u_ref[0]`
and the IC loss is evaluated on `x_star`), so pointing it at the task's
reference is a data substitution and nothing else: equation, domain, periodic
embedding, network, losses, optimiser and schedule are untouched. Verified
before writing: the reference's t = 0 row is `ib_ic.csv` to the last bit, and
the two x grids are identical.

`u_ref` also feeds their `compute_l2_error` progress readout, which this makes
meaningful rather than misleading.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TASK = Path("~/ib/task")
OUT = Path(__file__).parent / "data_task.npy"

csv = np.loadtxt(TASK / "ref_data.csv", delimiter=",", skiprows=1, dtype=np.float64)
t, x = np.unique(csv[:, 0]), np.unique(csv[:, 1])
order = np.lexsort((csv[:, 1], csv[:, 0]))
u = csv[order, 2].reshape(len(t), len(x))
assert np.allclose(csv[order, 0].reshape(len(t), len(x)), t[:, None])
assert np.allclose(csv[order, 1].reshape(len(t), len(x)), x[None, :])

ic = np.loadtxt(TASK / "ib_ic.csv", delimiter=",", skiprows=1, dtype=np.float64)
assert np.allclose(x, ic[:, 0]) and np.abs(u[0] - ic[:, 1]).max() == 0.0

np.save(OUT, {"u": u, "t": t, "x": x}, allow_pickle=True)
print(f"wrote {OUT}  u {u.shape}  t {t.shape}  x {x.shape}")
