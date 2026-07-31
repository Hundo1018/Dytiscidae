"""Staged evaluation: learn one medium, then a crossing, then the chain.

Why staging, and why it is not the fidelity cascade
---------------------------------------------------
There is already a cascade in this project -- Tier 0 analytic, Tier 1 short
episodes, Tier 2 full mission -- but that is a *fidelity* ladder.  Every tier
asks the same question, more or less accurately, and the question is always the
whole mission.

This is a *difficulty* ladder, which is a different axis.  Its stages ask
genuinely different questions, and a design that cannot answer the first is
never asked the last.

The reason is the shape of the reward.  Mission fraction is built on
``min(competences)`` times a transition term, so a design that is superb in
water and cannot leave it scores essentially zero -- the same essentially zero
as a design that is bad at everything.  Between those two designs there is a
gradient that matters enormously and the score cannot see it.  That is the
classic sparse-reward problem, and the classic answer is to reward the
intermediate capability directly until it is reliable, then stop.

Staging also spends the budget where the information is.  A design that cannot
stay upright in water learns nothing from being flown through three domains and
nine transitions; it fails in the first eight seconds and the remaining ninety
percent of the evaluation is measuring the same failure repeatedly.

The stages
----------
``0  single``
    One medium at a time, from a placed start.  Can it operate at all.
``1  directed``
    One medium, but going somewhere: hold depth, hold height, make ground.
``2  crossing``
    One boundary, from a placed start, scored on how the crossing was made.
``3  chain``
    Two media and the crossing between them, continuously, with no reset.
``4  mission``
    The full schedule.

Promotion and demotion
----------------------
A design is evaluated at its own stage and one above, so there is always a
gradient pointing up.  It is promoted when it clears the stage's bar, and it can
be *demoted* -- because the judge's bar ratchets, and a design that was
promoted under a looser bar should not keep a stage it can no longer earn.
Demotion is what stops the curriculum from becoming a set of participation
awards as the standard rises.

The stage is recorded on the design and reported.  "Half the archive is at stage
2 and nothing has reached stage 4" is a sentence about the state of the search
that no scalar score can produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The difficulty ladder.  ``(name, what it asks, the bar to leave it)``.
#:
#: The bars are deliberately modest: this ladder decides *what a design is asked
#: to do next*, not how good it is.  How good it is, is the judge's ladder, and
#: conflating the two would put tuning constants back in the place this project
#: keeps removing them from.
STAGES: list[tuple[str, str, float]] = [
    ("single", "operate in one medium at a time", 0.25),
    ("directed", "hold depth, hold height, make ground", 0.35),
    ("crossing", "cross one boundary and arrive usable", 0.35),
    ("chain", "two media and the crossing between them, continuously", 0.30),
    ("mission", "the full schedule", 0.0),
]
N_STAGES = len(STAGES)


@dataclass(eq=False)
class StageResult:
    stage: int = 0
    score: float = 0.0
    passed: bool = False
    detail: dict = field(default_factory=dict)


def stage_score(stage: int, result, transitions=None) -> float:
    """How well a result answers the question *this* stage asks.

    Each stage reads a different projection of the same evaluation, which is
    what makes the ladder a difficulty ladder rather than a weighting.
    """
    segs = getattr(result, "segments", {}) or {}
    if not segs:
        return 0.0
    comps = {d: float(getattr(s, "competence", 0.0)) for d, s in segs.items()}
    meas = {d: dict(getattr(s, "measurements", {}) or {}) for d, s in segs.items()}
    tc = {}
    if transitions is not None:
        try:
            tc = transitions.component_means()
        except Exception:
            tc = {}

    if stage == 0:
        # Can it operate at all, anywhere.  The *best* medium, not the worst:
        # the whole point of this stage is to reward a specialist for being one.
        return max(comps.values(), default=0.0)

    if stage == 1:
        # Going somewhere, in whichever medium it is best at.  Reads the raw
        # ladder measurements rather than competence, because "held depth" and
        # "scored well in water" are not the same claim.
        best = 0.0
        air = meas.get("air", {})
        best = max(best, float(np.clip(1.0 - air.get("sink_rate", 9.9) / 3.0, 0.0, 1.0)))
        water = meas.get("water", {})
        best = max(best, float(np.clip(1.0 - water.get("depth_error", 99.0) / 10.0, 0.0, 1.0)))
        land = meas.get("land", {})
        best = max(best, float(np.clip(land.get("land_speed", 0.0) / 0.4, 0.0, 1.0)))
        return best

    if stage == 2:
        # One crossing, made well.  Conditioned on crossing at all so that
        # never attempting it cannot look like doing it cleanly.
        if not tc:
            return 0.0
        quality = float(np.mean([
            tc.get("shock", 0.0), tc.get("control", 0.0),
            tc.get("exit_state", 0.0), tc.get("settle", 0.0),
        ]))
        return float(tc.get("crossed", 0.0) * quality)

    if stage == 3:
        # Two media plus the crossing, so the weakest of a *pair* rather than of
        # all three.  This is the rung where an amphibian lives and a triphibian
        # does not yet have to.
        vals = sorted(comps.values(), reverse=True)
        if len(vals) < 2:
            return 0.0
        pair = float(np.sqrt(vals[0] * vals[1]))
        return pair * max(float(tc.get("crossed", 0.0)), 0.1)

    # stage 4: the mission as scored everywhere else.
    return float(getattr(result, "mission_fraction", 0.0))


@dataclass(eq=False)
class Curriculum:
    """Tracks what each lineage is ready to be asked.

    Stages are held per *cell* rather than per design, because a design is
    transient and the region of behaviour space it occupies is not.  A cell that
    has produced a stage-3 design should be asking its next occupants stage-3
    questions, even though the individual that earned it has been replaced.
    """

    stages: dict = field(default_factory=dict)
    promotions: int = 0
    demotions: int = 0

    def stage_of(self, cell) -> int:
        return int(self.stages.get(tuple(cell), 0))

    def evaluate(self, cell, result, transitions=None) -> StageResult:
        """Score a design at its cell's stage, and at the next one up."""
        s = self.stage_of(cell)
        here = stage_score(s, result, transitions)
        nxt = stage_score(min(s + 1, N_STAGES - 1), result, transitions)
        bar = STAGES[s][2]
        return StageResult(
            stage=s,
            # Always a gradient upward: the next stage's score is visible even
            # to a design that has not been promoted, so there is something to
            # climb toward rather than a cliff at the promotion boundary.
            score=float(here + 0.25 * nxt),
            passed=here >= bar,
            detail={"here": round(here, 4), "next": round(nxt, 4), "bar": bar},
        )

    def update(self, cell, sr: StageResult) -> str:
        """Promote or demote the cell.  Returns what happened."""
        key = tuple(cell)
        s = self.stage_of(key)
        if sr.passed and s < N_STAGES - 1:
            self.stages[key] = s + 1
            self.promotions += 1
            return "promoted"
        # Demotion: the judge's bar ratchets, so a stage earned under a looser
        # standard has to be re-earned.  Without this the curriculum turns into
        # a record of what was once true.
        if s > 0 and sr.detail.get("here", 0.0) < 0.4 * STAGES[s - 1][2]:
            self.stages[key] = s - 1
            self.demotions += 1
            return "demoted"
        return "held"

    def forget(self, cell) -> None:
        """Drop a cell's stage when the cell itself is gone.

        Without this the record outlives the design.  A cell that was promoted
        and then quarantined, pruned or invalidated left its stage behind, and
        the summary below reported it forever.
        """
        self.stages.pop(tuple(cell), None)

    def report(self) -> dict:
        if not self.stages:
            return {"stages": {}, "promotions": 0, "demotions": 0,
                    "reached": 0, "typical": 0}
        counts = {}
        for s in self.stages.values():
            counts[STAGES[s][0]] = counts.get(STAGES[s][0], 0) + 1
        vals = sorted(self.stages.values())
        return {
            "stages": counts,
            "promotions": self.promotions,
            "demotions": self.demotions,
            # The furthest any one cell has got, and where the archive actually
            # is.  Reporting only the first overstates progress badly: an
            # archive of twenty cells at "single" and one at "directed" was
            # being summarised as stage 2, because ``max`` was taken over cells
            # that had since been removed as well as those still present.
            "reached": vals[-1],
            "typical": vals[len(vals) // 2],
        }
