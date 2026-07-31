"""The judge: a scoring standard that gets stricter as the population improves.

The problem with a fixed standard
---------------------------------
Every threshold in this project started as a number I typed.  "Full marks for
zero sink, nothing by 1.5 m/s."  "Ten metres is the dive target."  Each one
decides, invisibly, where the search stops trying -- once a population saturates
a threshold, the gradient vanishes and the search spends the rest of its budget
polishing something that is already scoring 1.0.  Worse, a saturated score hides
the difference between a machine that just clears the bar and one that clears it
by a factor of three, so the search has no reason to prefer the second.

The problem with a purely relative standard
-------------------------------------------
The obvious fix -- make the bar track the population's best -- destroys the
thing that makes a long run interpretable.  If the bar moves with the
population, 0.8 at generation 100 and 0.8 at generation 2000 are different
achievements and nothing in the record says so.  Worse, a bar that can move
*down* lets a population that has collapsed re-earn its old scores, which turns
the score into a measure of nothing.

What this does instead
----------------------
The two are separated.

**What is measured** is a fixed ladder, declared once and never changed.  Each
domain has a sequence of rungs describing qualitatively different achievements:
in air, "leaves the surface", then "glides", then "holds height", then "climbs",
then "manoeuvres while holding height".  The ladder is engineering, not tuning:
the rungs correspond to capabilities you would name in a design review, and a
design's rung is directly comparable across the whole run and between runs.

**Where the bar sits within the current rung** is a population quantile, and it
*ratchets*: it can tighten and never loosen.  So as soon as a real breakthrough
happens the standard for full marks moves up to meet it, and a design that would
have scored 1.0 last week scores 0.7 today -- while its *rung* and its raw
physical measurements are unchanged, so the record still says exactly what it
did.

Both are reported.  A design's score is (rung, fraction into the next rung), and
its absolute measurements travel with it, so nothing here can make the run
un-analysable later.

The ratchet is what makes this adversarial rather than merely adaptive: the
population is trying to score, and the judge answers every breakthrough by
raising the bar, so the only way to keep scoring is to keep breaking through.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# The fixed ladder
# --------------------------------------------------------------------------

#: Rungs per domain.  ``(name, metric, threshold)`` where ``metric`` names a raw
#: physical measurement and ``threshold`` is the value at which that rung is
#: considered reached.
#:
#: These thresholds are *not* tuning constants in the sense the rest of this
#: project has fought against.  They are the boundaries between qualitatively
#: different behaviours -- the difference between descending and not descending
#: is not a matter of taste -- and they are declared once so that a rung means
#: the same thing on day one and day five.  What adapts is the bar *inside* a
#: rung, which is where taste would otherwise creep in.
LADDER: dict[str, list[tuple[str, str, float]]] = {
    "air": [
        ("leaves_surface", "airborne_fraction", 0.10),
        ("stays_up", "airborne_fraction", 0.60),
        ("glides", "sink_rate", 3.0),          # sink below 3 m/s
        ("holds_height", "sink_rate", 0.5),    # sink below 0.5 m/s
        ("climbs", "sink_rate", -0.5),         # net climb
        ("manoeuvres", "turn_rate_held", 0.2),  # turns while holding height
    ],
    "water": [
        ("submerges", "max_depth", 0.5),
        ("dives", "max_depth", 3.0),
        ("reaches_depth", "max_depth", 10.0),
        ("holds_depth", "depth_error", 1.0),   # within 1 m of target
        ("manoeuvres", "water_speed", 0.5),    # makes way while holding depth
    ],
    "land": [
        ("stays_upright", "upright", 0.7),
        ("supports_itself", "contact_fraction", 0.5),
        ("moves", "land_speed", 0.1),
        ("walks", "land_speed", 0.4),
        ("climbs_slope", "slope_climbed", 0.5),
    ],
    "transition": [
        ("crosses", "crossed_fraction", 0.34),
        ("crosses_all", "crossed_fraction", 0.99),
        ("survives_entry", "shock", 0.5),
        ("stays_controlled", "control", 0.6),
        ("arrives_usable", "exit_state", 0.7),
        ("crosses_efficiently", "economy", 0.7),
    ],
}

#: Metrics where *lower* is better, so the rung is reached by going below the
#: threshold rather than above it.
LOWER_IS_BETTER = {"sink_rate", "depth_error"}


def rung_reached(domain: str, measurements: dict[str, float]) -> int:
    """How many rungs of ``domain``'s ladder these measurements clear.

    Rungs are cumulative and ordered: a design is at rung *k* when it clears
    rungs 0..k-1.  Clearing rung 3 while failing rung 2 counts as rung 2, which
    is deliberate -- the ladder describes a progression, and skipping a step
    usually means a measurement is being read in a regime where it does not
    mean what it normally means.
    """
    rungs = LADDER.get(domain, [])
    for i, (_, metric, threshold) in enumerate(rungs):
        v = measurements.get(metric)
        if v is None or not np.isfinite(v):
            return i
        ok = v <= threshold if metric in LOWER_IS_BETTER else v >= threshold
        if not ok:
            return i
    return len(rungs)


# --------------------------------------------------------------------------
# The ratchet
# --------------------------------------------------------------------------


@dataclass(eq=False)
class Ratchet:
    """A bar that tracks the population and never moves back down.

    Parameters
    ----------
    quantile:
        Where in the observed distribution the bar sits.  0.9 means "full marks
        requires being better than nine tenths of what has been seen".
    min_samples:
        Below this the bar stays at its declared floor, because a quantile of
        four numbers is not a standard, it is an accident.
    """

    metric: str
    lower_is_better: bool = False
    quantile: float = 0.9
    min_samples: int = 40
    floor: float = 0.0

    _seen: list = field(default_factory=list)
    bar: float = 0.0
    tightenings: int = 0
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bar = self.floor

    def observe(self, value: float) -> None:
        if np.isfinite(value):
            self._seen.append(float(value))
            if len(self._seen) > 4000:
                self._seen = self._seen[-3000:]

    def update(self, generation: int) -> bool:
        """Recompute the bar.  Returns True if it tightened."""
        if len(self._seen) < self.min_samples:
            return False
        q = 1.0 - self.quantile if self.lower_is_better else self.quantile
        proposed = float(np.quantile(self._seen, q))
        tighter = proposed < self.bar if self.lower_is_better else proposed > self.bar
        if not tighter:
            return False
        self.history.append(
            {"generation": generation, "from": self.bar, "to": proposed,
             "samples": len(self._seen)}
        )
        self.bar = proposed
        self.tightenings += 1
        return True

    def rollback(self) -> bool:
        """Undo the most recent tightening.

        The auditor's veto.  A bar that tightened because of a design that later
        failed audit has been moved by evidence that turned out not to be
        evidence, and leaving it there permanently penalises every honest design
        that follows.
        """
        if not self.history:
            return False
        last = self.history.pop()
        self.bar = float(last["from"])
        self.tightenings = max(self.tightenings - 1, 0)
        return True

    def score(self, value: float) -> float:
        """Fraction of the current bar this value achieves, in [0, 1]."""
        if not np.isfinite(value):
            return 0.0
        if self.lower_is_better:
            # floor is the "worthless" end, bar is the "full marks" end.
            span = self.floor - self.bar
            if span <= 1e-9:
                return 1.0 if value <= self.bar else 0.0
            return float(np.clip((self.floor - value) / span, 0.0, 1.0))
        span = self.bar - self.floor
        if span <= 1e-9:
            return 1.0 if value >= self.bar else 0.0
        return float(np.clip((value - self.floor) / span, 0.0, 1.0))


# --------------------------------------------------------------------------
# The judge
# --------------------------------------------------------------------------

#: The metric each domain's *headline* bar tracks -- the one that decides how
#: hard full marks is.  Chosen as the metric of the ladder's top rung, so the
#: bar tightens on the thing the population is currently trying hardest to do.
HEADLINE = {
    "air": ("sink_rate", True, 6.0),
    "water": ("max_depth", False, 0.0),
    "land": ("land_speed", False, 0.0),
    "transition": ("exit_state", False, 0.0),
}


@dataclass(eq=False)
class Judge:
    """Scores a design against a fixed ladder and a ratcheting bar."""

    quantile: float = 0.9
    update_every: int = 50
    ratchets: dict = field(default_factory=dict)
    generation: int = 0
    frozen: bool = False

    def __post_init__(self) -> None:
        if self.ratchets:
            return
        for dom, (metric, lower, floor) in HEADLINE.items():
            self.ratchets[dom] = Ratchet(
                metric=metric, lower_is_better=lower, quantile=self.quantile,
                floor=floor,
            )

    # ------------------------------------------------------------- observing

    def observe(self, measurements: dict[str, dict[str, float]]) -> None:
        """Record one design's raw measurements, keyed by domain."""
        for dom, r in self.ratchets.items():
            m = measurements.get(dom, {})
            if r.metric in m:
                r.observe(m[r.metric])

    def maybe_tighten(self, generation: int) -> list[dict]:
        """Tighten any bar that the population has outgrown."""
        self.generation = generation
        if self.frozen or generation % max(self.update_every, 1) != 0:
            return []
        moved = []
        for dom, r in self.ratchets.items():
            if r.update(generation):
                moved.append({"domain": dom, "metric": r.metric, **r.history[-1]})
        return moved

    # -------------------------------------------------------------- scoring

    def score(self, domain: str, measurements: dict[str, float]) -> dict:
        """Score one domain.  Returns rung, within-rung fraction, and total.

        ``rung`` is comparable across the whole run and between runs, because
        the ladder never changes.  ``within`` is comparable only against the
        judge version that produced it, which is why the judge's tightening
        count is reported alongside it.
        """
        rungs = LADDER.get(domain, [])
        k = rung_reached(domain, measurements)
        r = self.ratchets.get(domain)
        within = r.score(measurements.get(r.metric, np.nan)) if r else 0.0
        # Total in [0, 1]: rung progress dominates, the bar refines within it.
        #
        # The within-rung bonus is capped strictly below a whole rung, so a
        # design that has not cleared the next rung can never score as though it
        # had.  Without the cap a design at rung 5 with a perfect bar score tied
        # with one at rung 6, which is the saturation this whole design exists
        # to avoid, reintroduced one line from the end.
        base = k / max(len(rungs), 1)
        step = 0.95 / max(len(rungs), 1)
        return {
            "rung": k,
            "rung_name": rungs[k - 1][0] if 0 < k <= len(rungs) else "none",
            "within": within,
            "total": float(np.clip(base + step * within * (k < len(rungs)), 0.0, 1.0)),
        }

    def report(self) -> dict:
        return {
            "generation": self.generation,
            "frozen": self.frozen,
            "bars": {
                d: {"metric": r.metric, "bar": round(r.bar, 4),
                    "tightenings": r.tightenings, "samples": len(r._seen)}
                for d, r in self.ratchets.items()
            },
        }
