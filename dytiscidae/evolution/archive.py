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


# ``eq=False`` so that ``==`` is identity.  The generated __eq__ compares every
# field, two of which are numpy arrays, and ``array == array`` is an array --
# so any use of ``==`` or ``in`` on Elites raised "truth value of an array with
# more than one element is ambiguous" from deep inside an unrelated call.  That
# is what it did, five generations into a three-thousand-generation run.
@dataclass(eq=False)
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
    #: Objective vector, all higher-is-better.  This, not ``fitness``, decides
    #: whether a challenger takes the cell.  ``fitness`` survives for reporting
    #: and for the curator's parent weighting, where a scalar is unavoidable.
    objectives: np.ndarray = field(default_factory=lambda: np.zeros(0))

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
        #: The full non-dominated set per cell.  ``cells`` names one
        #: representative from each of these.
        self.fronts: dict[tuple[int, ...], list[Elite]] = {}
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

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """True when ``a`` is at least as good everywhere and better somewhere."""
        return bool(np.all(a >= b - 1e-12) and np.any(a > b + 1e-12))

    #: How many mutually non-dominated designs one cell may hold.  Small on
    #: purpose: the point is to stop discarding a design because of an exchange
    #: rate I invented, not to turn every cell into an archive of its own.
    front_capacity: int = 4

    @staticmethod
    def _crowding(front: list["Elite"]) -> np.ndarray:
        """Crowding distance over the objective vectors (NSGA-II).

        Used to decide who leaves when a cell's front is over capacity, so what
        survives spans the trade-off rather than clustering on one corner of it.
        """
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        obj = np.array([e.objectives for e in front], float)
        dist = np.zeros(n)
        for k in range(obj.shape[1]):
            order = np.argsort(obj[:, k])
            span = obj[order[-1], k] - obj[order[0], k]
            dist[order[0]] = dist[order[-1]] = np.inf
            if span < 1e-12:
                continue
            for i in range(1, n - 1):
                dist[order[i]] += (obj[order[i + 1], k] - obj[order[i - 1], k]) / span
        return dist

    def front(self, cell: tuple[int, ...]) -> list["Elite"]:
        """The non-dominated set occupying a cell."""
        return self.fronts.get(cell, [])

    @property
    def front_size(self) -> int:
        """Total designs held across all cells, fronts included."""
        return sum(len(f) for f in self.fronts.values())

    def add(
        self,
        genome,
        fitness: float,
        descriptor: np.ndarray,
        meta: dict | None = None,
        tier: int = 1,
        objectives: np.ndarray | None = None,
    ) -> str:
        """Insert a candidate.  Returns 'new', 'improved' or 'rejected'.

        Each cell holds a bounded *Pareto front* rather than one winner
        (Multi-Objective MAP-Elites; Pierrot et al. 2022).  What it replaced was
        a weighted sum whose coefficients I chose -- "a tenth of the structural
        margin is worth a tenth of the energy margin is worth a tenth of the
        land competence" -- and the cost of choosing them was invisible.  A
        design that gave up a hundredth of its mission fraction for three times
        the structural margin was accepted or discarded purely according to
        three numbers I typed, and nothing in the run ever reported which.

        Under dominance neither of those designs displaces the other: both stay,
        in the same cell, and the trade-off between them becomes a thing the run
        can show rather than a thing I decided in advance.  ``cells`` continues
        to name one representative per cell -- the one with the highest mission
        fraction, the single component that is unambiguously the task -- so
        coverage, the QD score and every existing consumer keep working.
        """
        if not math.isfinite(fitness):
            return "rejected"
        obj = np.zeros(0) if objectives is None else np.asarray(objectives, float)
        cell = self.cell_of(descriptor)
        front = self.fronts.setdefault(cell, [])

        cand = Elite(
            genome=genome, fitness=fitness, descriptor=np.asarray(descriptor, float),
            cell=cell, meta=meta or {}, born_at=self.generation, tier=tier,
            objectives=obj,
        )

        if not front:
            self.fronts[cell] = [cand]
            self.cells[cell] = cand
            return "new"

        # Inherit the cell's exploration bookkeeping: it describes the region,
        # not the individual, and resetting it every time an occupant changes
        # makes every cell look permanently untried to the curator.
        incumbent = self.cells.get(cell)
        if incumbent is None:
            # A front with no representative should be impossible, but repairing
            # it is strictly better than raising from inside an evaluation.
            incumbent = max(front, key=lambda e: e.objectives[0] if e.objectives.size else e.fitness)
            self.cells[cell] = incumbent
        cand.offspring = incumbent.offspring
        cand.offspring_placed = incumbent.offspring_placed
        cand.improvements = incumbent.improvements + 1

        if obj.size == 0 or any(e.objectives.size != obj.size for e in front):
            # No objective vector to compare on: fall back to the scalar.
            if fitness > incumbent.fitness:
                self.fronts[cell] = [cand]
                self.cells[cell] = cand
                return "improved"
            return "rejected"

        if any(self._dominates(e.objectives, obj) for e in front):
            return "rejected"

        kept = [e for e in front if not self._dominates(obj, e.objectives)]
        kept.append(cand)
        if len(kept) > self.front_capacity:
            order = np.argsort(-self._crowding(kept))
            kept = [kept[i] for i in order[: self.front_capacity]]
            if not any(e is cand for e in kept):
                # Survived dominance but lost on crowding: it sits on top of
                # designs the cell already has.
                self.fronts[cell] = kept
                self.cells[cell] = max(kept, key=lambda e: e.objectives[0])
                return "rejected"
        self.fronts[cell] = kept
        self.cells[cell] = max(kept, key=lambda e: e.objectives[0])
        return "improved"

    def remove(self, cell: tuple[int, ...]) -> bool:
        """Empty a cell completely.

        The only supported way to drop a cell.  Occupancy lives in two
        structures now -- the front and the representative -- and a caller that
        knows about only one leaves the other behind.  That is exactly what
        happened: quarantine and pruning both deleted from ``cells`` alone, so
        the next candidate landing in that cell found a non-empty front with no
        representative and the run died on a KeyError, sixty-eight generations
        in.  Deleting through one method is what keeps the two in step.
        """
        had = cell in self.cells or cell in self.fronts
        self.cells.pop(cell, None)
        self.fronts.pop(cell, None)
        return had

    # ----------------------------------------------------------------- rebin

    def rebin(self, axes: list[tuple[str, float, float, int]], reproject) -> dict:
        """Rebuild the grid under new axes, re-placing every elite.

        Needed because the descriptor axes are *learned* and therefore move.  An
        archive built under one projection is not comparable to one built under
        another, so when the projection is refitted every occupant has to be
        re-projected and re-filed.

        Collisions are resolved by fitness, which means re-binning can lose
        elites: two designs that the old axes called different may be the same
        under the new ones.  That is not a bug to be papered over, it is the
        cost of letting the system decide what "different" means, and the number
        lost is returned so a run can report it rather than quietly shrink.
        """
        before = len(self.cells)
        old = self.cells
        self.axes = axes
        self.names = [a[0] for a in axes]
        self.lo = np.array([a[1] for a in axes], float)
        self.hi = np.array([a[2] for a in axes], float)
        self.bins = np.array([a[3] for a in axes], int)
        self.cells = {}
        self.fronts = {}
        for e in old.values():
            d = reproject(e)
            if d is None:
                continue
            cell = self.cell_of(d)
            e.descriptor = np.asarray(d, float)
            e.cell = cell
            cur = self.cells.get(cell)
            if cur is None or e.fitness > cur.fitness:
                self.cells[cell] = e
            self.fronts.setdefault(cell, []).append(e)
        # Rebuilding fronts exactly would need a full dominance pass per cell;
        # trimming to capacity by fitness is enough, because the next insert
        # into a cell re-establishes the dominance invariant for it.
        for c, f in self.fronts.items():
            if len(f) > self.front_capacity:
                self.fronts[c] = sorted(f, key=lambda e: -e.fitness)[: self.front_capacity]
        # Taint marks are cell coordinates, which no longer mean anything.
        self.tainted = {}
        return {"before": before, "after": len(self.cells), "merged": before - len(self.cells)}

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
        """Write the archive, atomically.

        Through a temp file and a rename, because this is the one artefact of a
        run that cannot be recomputed -- every cell in it is an evaluation
        somebody paid seconds of simulation for -- and the environments this
        runs in are reclaimed without warning.  A plain ``open(path, "wb")``
        truncates the previous checkpoint before writing the new one, so a
        machine that goes away mid-write leaves neither: a run that has been
        going for hours is destroyed by an interruption that lands in the
        wrong millisecond.  A rename cannot half-happen.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(
                {"axes": self.axes, "cells": self.cells, "generation": self.generation,
                 "history": self.history, "tainted": self.tainted},
                f,
            )
        tmp.replace(path)

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
