"""Measured wing sections for `FluidSolver.section`, from the tables in
`sections/` (UIUC Low-Speed Airfoil Tests; `sections/SOURCES.md`).

A section is built from its drag runs only: each has the lift and the
wake-rake drag at the same angles, from about -5 deg to just short of stall.
The separate lift runs (to 18-24 deg, no drag) are kept out of the model and
used to check it (`w3_section.py`).
"""
import glob
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "sections"


def _rows(path):
    return np.array([[float(x) for x in line.split(",")] for line in open(path)
                     if line[0] not in "#a"])


def tables(name: str, min_points: int = 8):
    out = []
    for p in sorted(glob.glob(str(HERE / f"{name}_Re*k.csv"))):
        a = _rows(p)
        if len(a) < min_points:
            continue
        k = int(re.search(r"_Re(\d+)k", p).group(1))
        a = a[np.argsort(a[:, 0])]
        out.append((1e3 * k, np.radians(a[:, 0]), a[:, 1], a[:, 2]))
    if not out:
        raise FileNotFoundError(f"no drag run for section {name!r} in {HERE}")
    return out


def load(name: str):
    from dytiscidae.physics.fluid import SectionPolar
    return SectionPolar(name, tables(name))


def lift_runs(name: str):
    """(reynolds, alpha_rad, cl) of each lift run's rising sweep."""
    out = []
    for p in sorted(glob.glob(str(HERE / f"{name}_Re*k_lift.csv"))):
        a = _rows(p)
        top = int(np.argmax(a[:, 0]))
        a = a[: top + 1]
        k = int(re.search(r"_Re(\d+)k", p).group(1))
        out.append((1e3 * k, np.radians(a[:, 0]), a[:, 1]))
    return out
