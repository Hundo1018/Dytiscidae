"""The MAP-Elites archive: a map of what is achievable, not a single winner.

Ordinary optimisation returns one design and throws away everything it learned
on the way.  That is the wrong output for this project, where the user asked the
system to *discover the specification* rather than hit one.  A quality-diversity
archive answers the harder question directly: for every combination of mass,
buoyancy, flight competence and dive competence, what is the best machine of
that description, and does one exist at all?

The resulting grid is readable as an engineering result.  "Nothing above 12 kg
ever flies" and "neutral buoyancy costs half the air competence" are conclusions
you can only draw from a populated map, and they are exactly the conclusions
that tell you whether a 15 kg triphibian is worth pursuing.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Elite:
    """One occupant of one cell."""

    genome: object
    fitness: float
    descriptor: np.ndarray
    cell: tuple[int, ...]
    meta: dict = field(default_factory=dict)
    #: How many times this cell has been overwritten.  A cell that keeps being
    #: improved is a productive place to look; one that never changes is done.
    improvements: int = 0
    #: Offspring produced from this elite, and how many landed anywhere in the
    #: archive.  The ratio is what the curator calls curiosity.
    offspring: int = 0
    offspring_placed: int = 0
    born_at: int = 0
    tier: int = 1

    @property
    def curiosity(self) -> float:
        if self.offspring < 3:
            return 0.5  # optimistic prior: untried parents deserve a chance
        return self.offspring_placed / self.offspring


class Archive:
    """An N-dimensional grid of elites.

    Parameters
    ----------
    axes:
        List of ``(name, lo, hi, bins)``.  Values outside ``[lo, hi]`` are
        clamped into the edge bins rather than rejected, so a design that is
        heavier than the map anticipated still lands somewhere and can be seen.
    """

    def __init__(self, axes: list[tuple[str, float, float, int]]) -> None:
        self.axes = axes
        self.names = [a[0] for a in axes]
        self.lo = np.array([a[1] for a in axes], float)
        self.hi = np.array([a[2] for a in axes], float)
        self.bins = np.array([a[3] for a in axes], int)
        self.cells: dict[tuple[int, ...], Elite] = {}
        self.generation = 0
        self.history: list[dict] = []
        #: Cells that produced a simulator exploit.  Kept so the curator can
        #: refuse to re-seed from a region that is known to be a trap.
        self.tainted: dict[tuple[int, ...], int] = {}

    # ------------------------------------------------------------------ basics

    @property
    def capacity(self) -> int:
        return int(np.prod(self.bins))

    @property
    def coverage(self) -> float:
        return len(self.cells) / self.capacity

    @property
    def qd_score(self) -> float:
        """Sum of elite fitnesses: the standard scalar summary of a QD run.

        It rises both by finding better designs and by finding *more kinds* of
        design, which is the behaviour we want to reward.
        """
        return float(sum(e.fitness for e in self.cells.values()))

    @property
    def best(self) -> Elite | None:
        return max(self.cells.values(), key=lambda e: e.fitness, default=None)

    def cell_of(self, descriptor: np.ndarray) -> tuple[int, ...]:
        d = np.asarray(descriptor, float)
        frac = (d - self.lo) / np.maximum(self.hi - self.lo, 1e-12)
        idx = np.floor(frac * self.bins).astype(int)
        idx = np.clip(idx, 0, self.bins - 1)
        return tuple(int(i) for i in idx)

    # -------------------------------------------------------------------- add

    def add(
        self,
        genome,
        fitness: float,
        descriptor: np.ndarray,
        meta: dict | None = None,
        tier: int = 1,
    ) -> str:
        """Insert a candidate.  Returns 'new', 'improved' or 'rejected'."""
        if not math.isfinite(fitness):
            return "rejected"
        cell = self.cell_of(descriptor)
        cur = self.cells.get(cell)
        if cur is None:
            self.cells[cell] = Elite(
                genome=genome, fitness=fitness, descriptor=np.asarray(descriptor, float),
                cell=cell, meta=meta or {}, born_at=self.generation, tier=tier,
            )
            return "new"
        if fitness > cur.fitness:
            self.cells[cell] = Elite(
                genome=genome, fitness=fitness, descriptor=np.asarray(descriptor, float),
                cell=cell, meta=meta or {}, improvements=cur.improvements + 1,
                offspring=cur.offspring, offspring_placed=cur.offspring_placed,
                born_at=self.generation, tier=tier,
            )
            return "improved"
        return "rejected"

    # ------------------------------------------------------------- inspection

    def neighbour_density(self, cell: tuple[int, ...], radius: int = 1) -> int:
        """How crowded the neighbourhood of a cell is.

        Used by the curator to prefer parents on the *frontier* of the explored
        region, where offspring have somewhere new to land, over parents deep
        inside a region that is already fully mapped.
        """
        count = 0
        for other in self.cells:
            if all(abs(a - b) <= radius for a, b in zip(cell, other)):
                count += 1
        return count

    def project(self, ax_x: int, ax_y: int) -> tuple[np.ndarray, np.ndarray]:
        """2D projection for plotting: max fitness and occupancy per (x, y) bin."""
        nx, ny = self.bins[ax_x], self.bins[ax_y]
        best = np.full((nx, ny), np.nan)
        count = np.zeros((nx, ny), int)
        for cell, e in self.cells.items():
            i, j = cell[ax_x], cell[ax_y]
            count[i, j] += 1
            if math.isnan(best[i, j]) or e.fitness > best[i, j]:
                best[i, j] = e.fitness
        return best, count

    def snapshot(self) -> dict:
        """A JSON-safe summary, recorded once per generation for the dashboard."""
        fits = [e.fitness for e in self.cells.values()]
        b = self.best
        return {
            "generation": self.generation,
            "filled": len(self.cells),
            "capacity": self.capacity,
            "coverage": self.coverage,
            "qd_score": self.qd_score,
            "best_fitness": max(fits) if fits else 0.0,
            "mean_fitness": float(np.mean(fits)) if fits else 0.0,
            "best_cell": list(b.cell) if b else None,
            "best_meta": (b.meta or {}) if b else {},
            "tainted_cells": len(self.tainted),
        }

    # --------------------------------------------------------------- storage

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"axes": self.axes, "cells": self.cells, "generation": self.generation,
                 "history": self.history, "tainted": self.tainted},
                f,
            )

    @staticmethod
    def load(path: str | Path) -> "Archive":
        with open(path, "rb") as f:
            d = pickle.load(f)
        a = Archive(d["axes"])
        a.cells = d["cells"]
        a.generation = d["generation"]
        a.history = d.get("history", [])
        a.tainted = d.get("tainted", {})
        return a

    def export_json(self, path: str | Path) -> None:
        """Human- and browser-readable dump for the dashboard."""
        rows = []
        for cell, e in self.cells.items():
            rows.append(
                {
                    "cell": list(cell),
                    "descriptor": [float(x) for x in e.descriptor],
                    "fitness": float(e.fitness),
                    "tier": e.tier,
                    "improvements": e.improvements,
                    "curiosity": e.curiosity,
                    "meta": {k: v for k, v in (e.meta or {}).items()
                             if isinstance(v, (int, float, str, bool, list))},
                }
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"axes": [{"name": n, "lo": lo, "hi": hi, "bins": b}
                          for n, lo, hi, b in self.axes],
                 "summary": self.snapshot(), "history": self.history, "elites": rows},
                indent=1,
            )
        )
