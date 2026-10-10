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
    aerial_diver       air and water: enter fast and come back out.
    land_air           land and air, and the crossing between them.  Added after
                       three runs filmed 0/2 transitions: `land_to_air` was in
                       no island's objective, so nothing had ever been asked to
                       leave the ground.
    triphibian         all three media at once, paid on the *weakest*, with a
                       gradient where it is zero.  Added 2026-10-03: no arch46
                       evaluation had competence > 0.15 in all three media, and
                       the generalist's mission is zero for 98.8% of them, so
                       nothing paid for the stepping stone between "good in
                       two" and "does all three".  See ``triphibian_score``.
    generalist         the full mission, scored as everywhere else.

Eight islands.  Islands evolve independently.  Periodically the best of each
migrates to its neighbours, and -- the part that has no biological counterpart --
specialists from different islands are crossed directly, so a water specialist's
hull can meet an air specialist's surfaces without either lineage having had to
survive the valley between them.  A pair island's champion is crossed with the
specialist of its missing medium, and that cross goes to the triphibian island
(``TRIPLE_HOME``).

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

from .curriculum import WEAKEST_BARS

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
        # `land_to_air` was in no island's transitions at all, so nothing in the
        # search has ever been asked to leave the ground -- while arch35 and
        # arch36 spent two runs building a take-off ladder, gating it, and
        # multiplying it into `mission_fraction`.  The land island's only
        # transition was `water_to_land`: arriving, never leaving.
        "transitions": ("water_to_land", "land_to_air"),
        "note": "walk, climb the beach, carry itself, and get off it",
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
    "land_air": {
        "domains": ("land", "air"),
        "transitions": ("land_to_air",),
        # The pair the archipelago never had.  The first six islands covered three
        # singles, water+land and air+water, and left out the one crossing the
        # mission is actually blocked on: three runs have filmed 0/2
        # transitions and 0 m of depth while every elite sat on the beach.
        "note": "the takeoff problem: carry yourself, then leave the ground",
    },
    "triphibian": {
        "domains": ("air", "water", "land"),
        # Owned for the curriculum's crossing and chain stages only; the
        # island's own objective carries no transition factor (see
        # ``triphibian_score``).
        "transitions": ("air_to_water", "water_to_air", "water_to_land",
                        "land_to_air"),
        # Read by ``island_score``, ``own_domain_score`` and ``curriculum_for``:
        # the weakest medium, not the island's mean or its best.
        "objective": "weakest",
        "note": "all three media at once, paid on the weakest",
    },
    "generalist": {
        "domains": ("air", "water", "land"),
        "transitions": ("air_to_water", "water_to_air", "water_to_land",
                        "land_to_air"),
        "note": "the whole mission, scored as it is everywhere else",
    },
}


#: The triphibian objective's floor.  Half the triphibian ladder's ``third`` bar
#: (``WEAKEST_BARS[1]`` = 0.012, the p95 of the weakest medium among arch46
#: evaluations that operate in two), so a machine with all three media at that
#: bar outranks every machine missing one.  See ``triphibian_score``.
TRIPHIBIAN_FLOOR: float = WEAKEST_BARS[1] / 2.0


#: Competence under which a medium counts as not done (ROADMAP 2026-10-10,
#: revised T1).  0.012 = curriculum.WEAKEST_BARS[1] = 2 x TRIPHIBIAN_FLOOR.
#: Certified against held-still machines on arch49 by experiments/no_model_gate
#: (results_arch49_table.md): water elites 64/229 vs still 3/229, land 51/229
#: vs 0/229.  NOT certified for air there (elites 6/229 vs still 11/229: a
#: leak); kept at 0.012 until the paired air score's gate says otherwise.
COMPETENCE_FLOOR = {"air": WEAKEST_BARS[1], "water": WEAKEST_BARS[1],
                    "land": WEAKEST_BARS[1]}


def island_media(island: str) -> tuple:
    """The media an island's score reads; every medium for an unknown name."""
    return tuple(ISLANDS.get(island, {}).get("domains", ("air", "water", "land")))


def below_competence_floor(island: str, result) -> bool:
    """True when the design clears the floor in none of its island's media."""
    segs = getattr(result, "segments", {}) or {}
    for d in island_media(island):
        if (d in segs
                and float(getattr(segs[d], "competence", 0.0)) >= COMPETENCE_FLOOR[d]):
            return False
    return True


def triphibian_score(comps, floor: float = TRIPHIBIAN_FLOOR) -> float:
    """The triphibian island's objective: a soft minimum of the three media.

    ``H = 3 / sum(1 / (c_i + e)) - e``, the harmonic mean of the competences
    lifted by a floor ``e`` and lowered by it again, with ``e`` =
    ``TRIPHIBIAN_FLOOR`` = 0.006.  A medium with no segment is a zero.

    Why this shape, measured on arch46's 7,804 Tier-1 evaluations:

    * **The weakest medium decides.**  A machine missing a medium scores below
      ``2e`` = 0.012 however good the other two are (as c1, c2 -> inf, H -> 2e);
      a machine with all three at or above 0.012 scores at least 0.012.  So any
      machine that clears the ``third`` bar in all three outranks every machine
      that does not.  The geometric mean with the same floor fails this: it
      scores air 0.9 + water 0.9 + land 0 at 0.164, above 0.1 in all three.
    * **A gradient where the minimum has none.**  ``min`` is flat in two of its
      three arguments everywhere and flat in all three wherever the weakest is
      0 -- 98.2% of arch46.  H is strictly increasing in every competence, so
      among machines with a zero medium it still ranks the second-best one, the
      stepping stone, and the marginal value of the weakest is
      ``((c_j + e) / (c_w + e))^2`` times that of a stronger one c_j (312x at
      c_j = 0.1, c_w = 0).  Only order matters: the blend ranks this score as a
      population quantile (``Curriculum.standing``), so a small magnitude is no
      handicap.
    * **No energy, transition or take-off factor.**  Those are the generalist's
      (``mission_fraction``), and they are why it is zero for 98.8% of
      evaluations.  Crossings enter this island at its curriculum's ``crossing``
      and ``chain`` stages, gated on the weakest medium.

    Held still (``experiments/triphibian_still``, 7 plans x 2 seeds), the worst
    reading is 0.0081 -- gannet seed 1 glides in air (0.128) and floats in
    water (0.020) with land 0 -- below the 0.012 any three-medium machine at the
    bar scores.
    """
    vals = list(comps.values()) if isinstance(comps, dict) else list(comps)
    vals = (vals + [0.0, 0.0, 0.0])[:3]
    e = float(floor)
    inv = sum(1.0 / (max(float(v), 0.0) + e) for v in vals)
    return float(3.0 / inv - e)


def _triphibian_comps(result, domains) -> list:
    segs = getattr(result, "segments", {}) or {}
    return [float(getattr(segs[d], "competence", 0.0)) if d in segs else 0.0
            for d in domains]


def island_score(island: str, result, transitions=None) -> float:
    """Score a design on one island's own terms.

    A specialist island reads only its own domains, so a design that has given
    up the others is not punished for having given them up.  That is the point:
    it is the only way anything ever gets far enough out along an axis to be
    worth crossing back in.  The triphibian island reads all three through
    ``triphibian_score`` and nothing else.
    """
    spec = ISLANDS.get(island)
    if spec is None:
        return float(getattr(result, "mission_fraction", 0.0))
    if island == "generalist":
        return float(getattr(result, "mission_fraction", 0.0))
    if spec.get("objective") == "weakest":
        return triphibian_score(_triphibian_comps(result, spec["domains"]))

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


def curriculum_for(island: str):
    """A curriculum that asks an island's questions in the island's own media.

    ``run_search`` built every island's curriculum with no idea which island it
    served, so each one scored "the best medium anywhere" -- see
    ``curriculum.stage_score``.  An island outside ``ISLANDS`` gets the
    unrestricted curriculum, which is what the generalist's set amounts to.
    """
    from .curriculum import Curriculum

    spec = ISLANDS.get(island)
    if spec is None:
        return Curriculum()
    return Curriculum(domains=tuple(spec["domains"]),
                      transition_names=tuple(spec["transitions"]),
                      weakest=spec.get("objective") == "weakest")


def own_domain_score(island: str, meta: dict) -> float:
    """How good a stored elite is at *its island's own* domains, from its meta.

    Not the same question as fitness, and at arch39's generation-160 checkpoint
    not the same answer on five islands of seven: fitness blends the island's
    score with the curriculum stage's as population quantiles, and stage 0
    ("single") reads the design's best medium *anywhere*, so the air island's
    fitness champion scored air 0.033 and land 0.744 while an elite with air
    0.768 sat in the same archive.  Filming "the air island's best" by fitness
    films a land machine.

    This is :func:`island_score`'s base term -- the weakest own domain, square
    rooted, times their mean -- computed from the competences an elite's meta
    stores.  The transition-quality factor is left out because the archive does
    not store it.  The generalist island's own objective is the mission.
    """
    meta = meta or {}
    spec = ISLANDS.get(island)
    if spec is None or island == "generalist":
        v = meta.get("mission_fraction")
        return float(v) if isinstance(v, (int, float)) else 0.0
    comps = [float(meta.get(d) or 0.0) for d in spec["domains"]]
    if spec.get("objective") == "weakest":
        return triphibian_score(comps)
    if not comps:
        return 0.0
    return float(np.min(comps)) ** 0.5 * float(np.mean(comps))


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
                # To the island whose objective is that pair, not to whichever
                # name came first.  This loop used to run over
                # ("generalist", "amphibian", "aerial_diver") and ``break`` on
                # the first that existed -- and ``generalist`` always exists, so
                # **every hybrid ever made went to generalist**: 39 of 39 in
                # arch36, 24 of 24 in arch35, and none ever reached `amphibian`
                # or `aerial_diver`.  The air x land cross in particular -- a
                # walking flyer, which is the whole point of the take-off work --
                # was built every migration and only ever scored on an island
                # that also demanded water.
                for dst in HYBRID_HOME.get(frozenset((a_name, b_name)), ()):
                    if dst in self.archives:
                        out.append({
                            "island": dst, "genome": child,
                            "origin": f"{a_name}x{b_name}", "kind": "hybrid",
                        })
                        self.hybrids += 1
                        break

            # The third medium: a pair island's champion crossed with the
            # specialist of the medium it lacks, filed on the island that pays
            # for all three.  A specialist pair stays with its pair island
            # (``HYBRID_HOME``): scored here it would be ranked on a medium
            # neither parent was selected for.
            for pair_name, (single, dst) in TRIPLE_HOME.items():
                if dst not in self.archives:
                    continue
                gp, gs = self.emigrants(pair_name), self.emigrants(single)
                if not gp or not gs:
                    continue
                out.append({
                    "island": dst, "genome": crossover(gp[0], gs[0], rng),
                    "origin": f"{pair_name}x{single}", "kind": "hybrid",
                })
                self.hybrids += 1

        self.log.append({
            "generation": generation, "moved": len(out),
            "migrations": self.migrations, "hybrids": self.hybrids,
        })
        return out

    def colonists(self, name: str, n: int) -> list:
        """Immigrants for an island that is empty, from every other island.

        A checkpoint written before an island existed resumes with that island
        empty, and an empty island breeds random genomes -- hundreds of
        generations behind every other island.  Instead it is offered the ``n``
        elites of the rest of the archipelago that its own objective rates
        highest (``own_domain_score`` on their stored meta), as pending
        migrants: each is evaluated and has to earn its cell under the island's
        objective like any immigrant.  Empty when the island already has cells.
        """
        a = self.archives.get(name)
        if a is None or a.cells or n <= 0:
            return []
        pool = [(own_domain_score(name, e.meta), src, e.genome)
                for src, arch in self.archives.items() if src != name
                for e in arch.cells.values()]
        pool.sort(key=lambda t: -t[0])
        out = [{"island": name, "genome": g, "origin": src, "kind": "migrant"}
               for _, src, g in pool[:n]]
        self.migrations += len(out)
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


#: Which island's objective a cross of two specialists belongs to.  Ordered, so
#: a destination that this run does not have falls through to the next.
HYBRID_HOME: dict[frozenset, tuple[str, ...]] = {
    frozenset(("air", "water")): ("aerial_diver", "generalist"),
    frozenset(("water", "land")): ("amphibian", "generalist"),
    frozenset(("air", "land")): ("land_air", "generalist"),
}

#: A pair island, the specialist of the medium it lacks, and where their cross
#: goes.  Three crosses per migration, one per pair island.
TRIPLE_HOME: dict[str, tuple[str, str]] = {
    "amphibian": ("air", "triphibian"),
    "aerial_diver": ("land", "triphibian"),
    "land_air": ("water", "triphibian"),
}


def _pairs(items: list) -> list:
    return [(items[i], items[j])
            for i in range(len(items)) for j in range(i + 1, len(items))]
