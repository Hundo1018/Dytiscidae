"""Islands: specialists and generalists evolved in parallel, and crossed.

The argument from biology
-------------------------
There are almost no triphibian animals, and the reason is not that the niche is
worthless.  It is that specialisation is close to irreversible.  A lineage that
commits to water loses the structures flight needed; a lineage that commits to
flight loses the mass budget diving needed.  The intermediate is worse at both
than either specialist, so selection never carries anything across the valley --
and the few animals that manage all three (a diving petrel, a dipper, this
project's namesake beetle) are conspicuously mediocre at each.

A single population scored on the full mission reproduces that trap exactly.
Mission fraction is built on the weakest domain, so a superb water specialist
scores the same as a bad everything, gets no selective advantage from what it is
good at, and is bred out.  The population converges on uniform mediocrity and
the parts a triphibian would need are never invented, because nothing was ever
rewarded for inventing them.

What this does
--------------
Separate populations with separate objectives, plus the one path biology does
not have: deliberate hybridisation.

    air, water, land   specialists, scored only on their own medium and the
                       crossings that touch it.  Free to give up everything else
                       and go as far as the physics allows.
    amphibian          two media and the crossing between them.  The rung real
                       animals actually occupy.
    generalist         the full mission, scored as everywhere else.

Islands evolve independently.  Periodically the best of each migrates to its
neighbours, and -- the part that has no biological counterpart -- specialists
from different islands are crossed directly, so a water specialist's hull can
meet an air specialist's surfaces without either lineage having had to survive
the valley between them.

That is the whole point of doing this in simulation rather than in a river.  The
irreversibility is a property of *inheritance*, not of physics, and a search
that can copy genes between lineages is not bound by it.

Migration is one-directional in effect: an immigrant competes on the receiving
island's own terms, so a water specialist arriving on the generalist island has
to earn its place under the full mission.  Nothing is protected by having come
from somewhere prestigious.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Which domains each island cares about, and which crossings.
ISLANDS: dict[str, dict] = {
    "air": {
        "domains": ("air",),
        "transitions": ("air_to_water", "water_to_air"),
        "note": "fly as well as the physics allows; owe nothing to the water",
    },
    "water": {
        "domains": ("water",),
        "transitions": ("air_to_water", "water_to_land"),
        "note": "dive as deep and hold as long as the hull allows",
    },
    "land": {
        "domains": ("land",),
        "transitions": ("water_to_land",),
        "note": "walk, climb the beach, carry itself",
    },
    "amphibian": {
        "domains": ("water", "land"),
        "transitions": ("water_to_land",),
        "note": "the rung real animals occupy: two media and the crossing",
    },
    "aerial_diver": {
        "domains": ("air", "water"),
        "transitions": ("air_to_water", "water_to_air"),
        "note": "the gannet problem: enter fast, come back out",
    },
    "generalist": {
        "domains": ("air", "water", "land"),
        "transitions": ("air_to_water", "water_to_air", "water_to_land"),
        "note": "the whole mission, scored as it is everywhere else",
    },
}


def island_score(island: str, result, transitions=None) -> float:
    """Score a design on one island's own terms.

    A specialist island reads only its own domains, so a design that has given
    up the others is not punished for having given them up.  That is the point:
    it is the only way anything ever gets far enough out along an axis to be
    worth crossing back in.
    """
    spec = ISLANDS.get(island)
    if spec is None:
        return float(getattr(result, "mission_fraction", 0.0))
    if island == "generalist":
        return float(getattr(result, "mission_fraction", 0.0))

    segs = getattr(result, "segments", {}) or {}
    comps = [float(getattr(segs[d], "competence", 0.0))
             for d in spec["domains"] if d in segs]
    if not comps:
        return 0.0
    # Weakest of the island's *own* domains, so an amphibian still has to do
    # both of its two -- the trap is only avoided across islands, not within one.
    base = float(np.min(comps)) ** 0.5 * float(np.mean(comps))

    quality = 1.0
    if transitions is not None:
        rel = [r for k, r in getattr(transitions, "results", {}).items()
               if k in spec["transitions"]]
        if rel:
            crossed = float(np.mean([1.0 if r.crossed else 0.0 for r in rel]))
            comp = float(np.mean([
                np.mean([r.shock, r.control, r.settle, r.exit_state]) for r in rel
            ]))
            quality = crossed * (0.4 + 0.6 * comp)
    # A specialist still has to be able to get into and out of its own medium,
    # but the floor stops a pure-domain achievement from being erased by a
    # crossing it has not learned yet.
    return float(base * max(quality, 0.15))


@dataclass(eq=False)
class Archipelago:
    """Several archives, evolved in parallel, with migration and hybridisation.

    Parameters
    ----------
    migrate_every:
        Generations between migrations.  Infrequent: migration is a shock to a
        receiving island's composition, and doing it constantly just merges the
        islands back into the single population this exists to avoid.
    n_migrants:
        How many of an island's best travel each time.
    hybridise:
        Whether to cross specialists from different islands directly.  This is
        the move biology cannot make and the reason this is worth doing.
    """

    migrate_every: int = 60
    n_migrants: int = 2
    hybridise: bool = True

    archives: dict = field(default_factory=dict)
    curators: dict = field(default_factory=dict)
    migrations: int = 0
    hybrids: int = 0
    log: list = field(default_factory=list)

    @property
    def names(self) -> list:
        return list(self.archives)

    def register(self, name: str, archive, curator) -> None:
        self.archives[name] = archive
        self.curators[name] = curator

    # ------------------------------------------------------------- migration

    def due(self, generation: int) -> bool:
        return generation > 0 and generation % max(self.migrate_every, 1) == 0

    def emigrants(self, name: str) -> list:
        """The designs this island sends abroad: its best, by its own lights."""
        a = self.archives.get(name)
        if a is None or not a.cells:
            return []
        best = sorted(a.cells.values(), key=lambda e: -e.fitness)
        return [e.genome for e in best[: self.n_migrants]]

    def migrate(self, generation: int, rng: np.random.Generator, crossover=None) -> list:
        """Move genomes between islands and cross specialists.

        Returns the genomes to be evaluated as immigrants, tagged with the
        island that should evaluate them.  Nothing is inserted directly: an
        immigrant has to earn its place under the receiving island's own
        objective, which is what stops a prestigious origin from being a free
        pass.
        """
        if len(self.archives) < 2:
            return []
        out = []
        names = self.names
        for i, src in enumerate(names):
            for g in self.emigrants(src):
                dst = names[(i + 1 + int(rng.integers(len(names) - 1))) % len(names)]
                if dst == src:
                    continue
                out.append({"island": dst, "genome": g, "origin": src, "kind": "migrant"})
                self.migrations += 1

        # Hybridisation: the move that has no biological counterpart.  A water
        # specialist's hull meets an air specialist's surfaces without either
        # lineage having had to survive the valley between them.
        if self.hybridise and crossover is not None:
            specialists = [n for n in names if n in ("air", "water", "land")]
            for a_name, b_name in _pairs(specialists):
                ga, gb = self.emigrants(a_name), self.emigrants(b_name)
                if not ga or not gb:
                    continue
                child = crossover(ga[0], gb[0], rng)
                for dst in ("generalist", "amphibian", "aerial_diver"):
                    if dst in self.archives:
                        out.append({
                            "island": dst, "genome": child,
                            "origin": f"{a_name}x{b_name}", "kind": "hybrid",
                        })
                        self.hybrids += 1
                        break

        self.log.append({
            "generation": generation, "moved": len(out),
            "migrations": self.migrations, "hybrids": self.hybrids,
        })
        return out

    def report(self) -> dict:
        return {
            "islands": {
                n: {"cells": len(a.cells),
                    "best": round(a.best.fitness, 4) if a.best else 0.0}
                for n, a in self.archives.items()
            },
            "migrations": self.migrations,
            "hybrids": self.hybrids,
        }


def _pairs(items: list) -> list:
    return [(items[i], items[j])
            for i in range(len(items)) for j in range(i + 1, len(items))]
