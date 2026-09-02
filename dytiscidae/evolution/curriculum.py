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

    #: Recent (island_score, curriculum_score) pairs *per stage*.  Bounded
    #: window: this is a question about the population now, not about the whole
    #: run.
    #:
    #: Keyed by stage because the stages ask different questions and their
    #: answers are not on one scale.  Pooling them, which is what the first
    #: version did, meant a stage-2 design (``crossed * quality``, usually zero)
    #: was ranked inside a window dominated by stage-0 designs (``max
    #: competence``, around 0.6), so it landed in the bottom quantile for having
    #: been promoted.  Measured over arch31's 9384 evaluations: mean fitness by
    #: stage 0.568 / 0.414 / 0.302 / 0.380 and corr(fitness, stage) = -0.35 --
    #: the search docked a design for advancing, and ``Curator.prune`` then
    #: dropped whatever sat below its neighbourhood's 25th percentile.
    _recent: dict = field(default_factory=dict)
    #: Unused.  Kept only so that unpickling a pre-2026-09 state does not fail;
    #: the applied weight has always been ``stage / (N_STAGES - 1)`` and the
    #: telemetry now reports that instead of this field, which nothing writes.
    _handover: float = 0.0
    window: int = 256
    #: Samples a stage's window needs before its quantiles mean anything.
    min_rank_samples: int = 12
    #: Floor under the island half's weight.  At ``stage/4`` alone the weight
    #: was 0 for the 69% of arch31's evaluations that sat at stage 0, so the
    #: island objective -- which for the generalist island *is* the mission --
    #: contributed nothing at all to most of the run.
    handover_floor: float = 0.25

    def stage_of(self, cell) -> int:
        return int(self.stages.get(tuple(cell), 0))

    def seed_stage(self, cell, stage: int) -> int:
        """Give an unvisited cell the stage its parent had earned.

        Stages are held per cell, and 58.9% of arch31's children landed in a
        cell nobody had occupied -- which started at stage 0 however advanced
        the parent was.  So 69.1% of all evaluations were asked the *easiest*
        question no matter what the lineage had already shown it could do, the
        handover ramp never left its floor, and the mission's share of the
        score stayed at 1.7% for six hundred generations.

        A region is not a lineage, so this is a starting guess and not a grant:
        the cell is still demoted the moment an occupant cannot hold the stage,
        which is what ``update`` already does.
        """
        key = tuple(cell)
        if key in self.stages:
            return int(self.stages[key])
        s = int(np.clip(stage, 0, N_STAGES - 1))
        self.stages[key] = s
        return s

    def observe_blend(self, island_score: float, curriculum_score: float,
                      stage: int = 0, mission: float = 0.0) -> None:
        """Record what each half of the blend said about one design.

        Filed under the stage the design was asked about, because that is the
        only population it is comparable with.
        """
        w = self._recent.setdefault(int(stage), [])
        w.append((float(island_score), float(curriculum_score), float(mission)))
        if len(w) > self.window:
            del w[: len(w) - self.window]

    def handover(self, stage: int) -> float:
        """How much weight the island's own objective has earned, in [0, 1].

        The islands are an *early* device.  They exist to grow terrain-adapted
        genes quickly and to stop a dark horse -- good at nothing yet, promising
        -- from being culled before it can show what it is.  Late in a run the
        thing being searched for is a triphibian again, so the island's own
        objective has to take the weight back.

        It used to be a flat 0.5 below stage 4, and that constant was doing real
        damage.  Measured on the 500-generation runs, the air island's champion
        had air competence 0.107 while its stored fitness was 0.5019: the island
        objective, ``competence**1.5``, could contribute at most 0.035, so 96% of
        the selection pressure on the *air* island came from a score that does
        not look at air at all.  Every stage below 4 reads the design's *best*
        medium regardless of island, and since water competence reaches 0.9 where
        air reaches 0.1, the island-blind half rewarded water everywhere.  The
        air island filled with wingless water specialists.

        So the weight is evidence now, in the shape the judge already uses: a
        fixed ladder for what is measured, a population statistic for where the
        line sits, and a ratchet so it only ever moves one way.

        It used to add a second term: weight whichever half was *discriminating*
        more, measured as the ratio of the two p90-p10 spreads.  That rule is
        removed, because measurement showed it does the opposite of its purpose.

        A hard objective has a small spread precisely because nothing can do it
        yet.  `mission_fraction` across arch30's archive spans 0.0437 from p10 to
        p90 while the curriculum score spans 0.5092, so the rule concluded the
        mission "does not discriminate" and handed its weight away -- which
        guarantees the search never learns to do the thing it is for.  Over
        arch30 it averaged 0.197, *below* the 0.25 stage floor it was meant to
        improve on, so it was strictly worse than the ramp it was added to.
        The result was corr(fitness, mission_fraction) = 0.1413 across 653
        elites: 98% of the selection pressure was deciding something other than
        the mission.

        What remains is the stage ramp alone.  It is a scaffold's proper shape:
        the curriculum answers an easier question while a cell cannot answer the
        real one, and hands back as the cell climbs.
        """
        if stage >= N_STAGES - 1:
            return 1.0
        ramp = stage / float(N_STAGES - 1)
        return float(np.clip(max(ramp, self.handover_floor), 0.0, 1.0))

    def standing(self, island_score: float, curriculum_score: float,
                 stage: int = 0):
        """Both halves of the blend as population quantiles in [0, 1].

        The blend is a convex combination, so it compares the two halves'
        *magnitudes* -- and they are not on the same scale.  Measured over
        arch30: the island half spans 0.0437 from p10 to p90 and the curriculum
        half spans 0.5092, an 11.7x mismatch, so at the weight the ramp actually
        applied (0.2794 mean) the mission contributed 0.0086 against 0.40, about
        2% of the selection variance.  That figure was arrived at two ways --
        from the weights and spreads, and independently as r-squared of the
        measured correlation -- and they agree to 0.001.

        Ranking each half within the recent population removes the mismatch at
        its source: a quantile is on [0, 1] by construction whatever the raw
        scale, so a weight of 0.25 means a quarter of the influence rather than
        a fortieth.

        The window is *per stage*.  A quantile only means anything against
        answers to the same question, and the stages ask different ones -- see
        ``_recent``.  A stage with too few samples to rank returns 0.5 for that
        half rather than a number: "cannot rank this yet" is the honest answer,
        and it neither rewards nor punishes the design for being rare, which is
        what pooling did.

        The cost is honest and worth stating: a quantile is relative to the
        window, so an elite's stored fitness is not comparable across a long run
        the way an absolute score would be.  MAP-Elites compares a challenger
        against the cell's current occupant, and both are scored against
        populations from different times.  The window is long enough that this
        drifts slowly, and the judge already ratchets population quantiles for
        the same reason, but it is a change in what a stored fitness means.
        """
        w = self._recent.get(int(stage), [])
        if len(w) < self.min_rank_samples:
            return 0.5, 0.5
        isl = np.array([a for a, _, _ in w], float)
        cur = np.array([b for _, b, _ in w], float)
        return (float(np.mean(isl <= island_score)),
                float(np.mean(cur <= curriculum_score)))

    def mission_standing(self, mission: float, stage: int = 0) -> float:
        """``mission_fraction`` as a population quantile, on the same scale.

        The blend's two halves decide *how well this design answered the
        question its cell was asked*, and measurement says that is not the same
        thing as the mission: over arch31's 9384 evaluations corr(archive
        fitness, mission_fraction) = 0.159, corr with the *weakest* competence --
        which is what ``mission_fraction`` is built on -- was 0.003, and corr
        with the energy margin 0.056.  The scalar that drives replacement,
        parent choice and pruning could not see two of the four factors in its
        own objective.

        Injecting the mission as a third quantile is the smallest change that
        fixes that: it arrives on [0, 1] like the other two, and because
        ``mission_fraction`` is already ``min(comp)^0.5 * mean(comp) *
        energy_fraction * transition_fraction`` it carries the weakest medium,
        the battery and the crossings in the proportions the task itself
        defines, rather than in weights someone typed.
        """
        w = self._recent.get(int(stage), [])
        if len(w) < self.min_rank_samples:
            return 0.5
        m = np.array([c for _, _, c in w], float)
        return float(np.mean(m <= float(mission)))

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
        """Promote or demote the cell.  Returns what happened.

        Promotion needs two things: passing this stage's bar, and a *nonzero*
        answer to the next stage's question.  The bar alone let a cell climb on
        its best medium and then face-plant on a question it had never touched:
        arch30 logged 764 promotions against 17 demotions while the typical
        cell sat at stage 1 and five cells ever reached "chain" -- swimmers
        passing "directed" on depth-hold, being promoted into "crossing", and
        scoring zero there forever.  Zero is not a tunable: a design that has
        never once crossed a boundary gains nothing from being asked about
        crossings, and one that has -- however badly -- has something for the
        gradient in ``evaluate`` to climb.
        """
        key = tuple(cell)
        s = self.stage_of(key)
        if sr.passed and s < N_STAGES - 1 and sr.detail.get("next", 0.0) > 0.0:
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

    def rebuild_from(self, archive) -> dict:
        """Re-key the stage record after the descriptor axes have moved.

        ``stages`` is keyed by cell coordinate, and a learned-descriptor refit
        re-files every elite, so after a refit the old keys name regions that no
        longer exist while the surviving elites have coordinates nobody has a
        stage for.  Measured on arch31's generalist island at generation 599:
        **1,040 stage entries against 251 live cells, of which only 100 (9.6%)
        named a cell that still existed, and 60.2% of live cells had no entry at
        all** -- so every one of them was asked the stage-0 question again.
        With twenty-three refits over the run, the curriculum was being reset
        underneath the population on a schedule.

        The stage travels on the elite (``meta["stage"]``), so it can simply be
        re-read from the archive under the new coordinates.  Where two designs
        merge into one cell the higher stage wins: the region has demonstrably
        produced an answer to the harder question, and ``update`` will demote it
        again within a generation if its occupant cannot hold it.
        """
        before = len(self.stages)
        fresh: dict = {}
        for cell, e in archive.cells.items():
            s = int((getattr(e, "meta", None) or {}).get("stage", 0))
            fresh[tuple(cell)] = max(fresh.get(tuple(cell), 0), s)
        self.stages = fresh
        return {"stages_before": before, "stages_after": len(fresh)}

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
            # The blend is now a moving quantity, so it has to be in the record
            # or a later reading of a run cannot tell what it was scored on.
            #
            # This used to log ``_handover``, a field nothing ever writes, so
            # every arch31 generation line reported handover 0.0 while the
            # weight actually applied was ``stage/4``.  A telemetry field that
            # reports a constant no code reads is worse than no field.
            "handover_floor": round(self.handover_floor, 4),
            "handover_typical": round(self.handover(vals[len(vals) // 2]), 4),
            "handover_mean": round(
                float(np.mean([self.handover(s) for s in self.stages.values()])), 4),
            "rank_windows": {STAGES[s][0]: len(w) for s, w in sorted(self._recent.items())},
        }
