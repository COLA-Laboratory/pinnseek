"""Measurement probe (not a solver): averages the predictions of four already
trained marched PINNs (b01_tm14, b02_tw, b02_ovl, b02_hbc) to price readout-only
deep ensembling before paying to train an ensemble inside one 300 s run.
Changed vs b01_tm14.py: train() replays saved pickles; predict_fn averages them.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 3
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u", "v", "p")

# --- Domain ---
T_MIN, T_MAX = 0.0, 1.2
X_MIN, X_MAX = 0.0, 2.2
Y_MIN, Y_MAX = 0.0, 0.41
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 3   # ru, rv, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("inlet", "outlet", "walls", "cylinder")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "channel_with_cylinder"

# --- Geometry (DFG 2D benchmark, Schaefer & Turek) ---
CYL_X, CYL_Y = 0.2, 0.2      # cylinder centre; offset from the channel
CYL_R        = 0.05          # mid-height 0.205 to break the symmetry

# --- Physical parameters (DFG 2D-3, Schaefer & Turek: driven inflow) ---
U_PEAK  = 1.5        # peak centreline inflow; U(s) = U_PEAK*sin(pi*s/T_DRIVE)
T_DRIVE = 8.0        # driving period of the benchmark, in simulation time s
T_START = 4.8        # task t = 0 is benchmark time s = T_START
NU      = 1.0e-3     # kinematic viscosity (rho = 1)
RE      = (2.0 / 3.0) * U_PEAK * (2.0 * CYL_R) / NU   # = 100 at the drive peak
                     # the window itself runs at Re(t) = 95 -> 71

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 300.0     # total wall-clock seconds for the WHOLE process,
                       # from _T0: imports, JIT compile, training and param
                       # save all included — eval.py kills the process at
                       # this wall; return params with margin to spare
MAX_ITER   = 1_000_000 # safety cap on the loop (not expected to bind)

# --- Eval gate: full fidelity only under task/eval.py ---
if _os.environ.get("FORGE_EVAL_TOKEN") != "task/eval.py":
    TRAIN_TIME = 10.0                                # off-eval runs are smoke:
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")   # CPU, 10 s
# ═══════════════════════════════════════════════════════════════

import pickle
import sys
import types
from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp

SRC = [("b01", "b01_tm14"), ("b02", "b02_tw"), ("b02", "b02_ovl"), ("b02", "b02_hbc")]
_MODS: list | None = None


def _mods():
    global _MODS
    if _MODS is None:
        out = []
        for blk, stem in SRC:
            code = (Path.cwd() / "blocks" / blk / f"{stem}.py").read_text(encoding="utf-8")
            m = types.ModuleType("_e_" + stem)
            sys.modules["_e_" + stem] = m
            exec(compile(code, stem, "exec"), m.__dict__)
            out.append(m)
        _MODS = out
    return _MODS


def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u","v","p"} each [BS]."""
    acc = None
    for m, p in zip(_mods(), params):
        o = m.predict_fn(p, X)
        s = jnp.stack([o["u"], o["v"], o["p"]], axis=1)
        acc = s if acc is None else acc + s
    out = acc / len(params)
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Probe: no training — replay the saved parameters of the source models."""
    ps = []
    for blk, stem in SRC:
        with open(Path.cwd() / "blocks" / blk / f"{stem}.pkl", "rb") as f:
            ps.append(pickle.load(f))
    return ps, 0
