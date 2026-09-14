"""Checks, tolerances and provenance.  Deliberately tiny and self-contained.

`benchmarks/` does not import from `experiments/`: a verification ladder that
depends on another directory having been merged is not a foundation.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    """One comparison between a reference and the simulator.

    `reference` never comes from the simulator.  It is a closed form, or a
    tight integration of a force law written out independently.
    """

    layer: int
    name: str
    reference: float
    simulated: float
    tolerance: float
    #: What the reference is, so a reader can attack it: "closed form",
    #: "RK45 rtol 1e-10 of the same force law", "conservation law".
    reference_kind: str
    note: str = ""
    #: Compare absolutely rather than relatively.  For a reference of exactly
    #: zero -- a conserved quantity, a residual -- a relative error is always 1
    #: and says nothing.
    absolute: bool = False

    @property
    def error(self) -> float:
        if self.absolute:
            return abs(self.simulated - self.reference)
        denom = max(abs(self.reference), abs(self.simulated), 1e-300)
        return abs(self.simulated - self.reference) / denom

    @property
    def agrees(self) -> bool:
        return self.error <= self.tolerance

    def line(self) -> str:
        mark = "ok     " if self.agrees else "DEPARTS"
        kind = "abs" if self.absolute else "rel"
        return (f"  [{mark}] {self.name:<54} ref {self.reference: .6g}  "
                f"sim {self.simulated: .6g}  {kind} {self.error:.3g} "
                f"(tol {self.tolerance:g})")

    def to_dict(self) -> dict:
        return {"layer": self.layer, "name": self.name,
                "reference": float(self.reference),
                "simulated": float(self.simulated),
                "reference_kind": self.reference_kind,
                "error": float(self.error), "absolute": self.absolute,
                "tolerance": float(self.tolerance), "agrees": bool(self.agrees),
                "note": self.note}


@dataclass
class Layer:
    number: int
    title: str
    adds: str
    checks: list[Check] = field(default_factory=list)

    @property
    def agrees(self) -> bool:
        return all(c.agrees for c in self.checks)

    @property
    def worst(self) -> float:
        return max((c.error for c in self.checks), default=0.0)


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def write(layers: list[Layer], out_dir: Path, started: float) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import mujoco
    payload = {
        "commit": git_commit(),
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_s": round(time.time() - started, 3),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "platform": platform.platform(),
        "first_departing_layer": next(
            (lay.number for lay in layers if not lay.agrees), None),
        "layers": [
            {"number": lay.number, "title": lay.title, "adds": lay.adds,
             "agrees": lay.agrees, "worst_error": lay.worst,
             "checks": [c.to_dict() for c in lay.checks]}
            for lay in layers
        ],
    }
    path = out_dir / "result.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def rk45(rhs, y0: np.ndarray, t_end: float, *, rtol: float = 1e-11,
         atol: float = 1e-13) -> np.ndarray:
    """Tight Runge-Kutta reference for a force law written out independently.

    Used where no closed form exists.  The point of the tight tolerance is that
    the reference's own integration error is several orders below the
    discrepancy being measured, so a departure is attributable to the force law
    rather than to the integrator.
    """
    from scipy.integrate import solve_ivp
    sol = solve_ivp(rhs, (0.0, t_end), np.asarray(y0, float),
                    method="RK45", rtol=rtol, atol=atol, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"reference integration failed: {sol.message}")
    return sol.y[:, -1]
