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
        # Four rungs on whether the airframe can fly at all, *below* the ones
        # that read where it happened to be.  `airborne_fraction` is satisfied
        # by falling -- the air segment releases the machine at 30 m and a body
        # dropped there is airborne for the 2.5 s it takes to arrive -- so
        # `leaves_surface` and `stays_up` were paying 53% of arch36 for free
        # fall, with a population median sink rate of 9.9 m/s.
        #
        # 130 of arch36's 179 elites (72.6%) could not produce their own weight
        # in lift at any speed between 6 and 30 m/s, so `_measure_trim_speed`
        # dropped them rather than launching them; and between "generates no
        # lift" and "flies" there was no gradient at all, for three quarters of
        # the population.  `lift_margin` is that gradient and it costs nothing:
        # the trim sweep already computed it and threw it away.
        #
        # Thresholds from the measured distribution over those 179 elites --
        # min -1.105 (pushing *down* at 30 m/s), median 0.333, p90 5.591 --
        # leaving 72.1% / 51.4% / ~36% / 27.4% standing.  The last is the same
        # set as "has a real trim speed", by both measurements, which is the
        # check that this is the right quantity.
        ("makes_lift", "lift_margin", 0.10),
        ("carries_a_third", "lift_margin", 0.30),
        ("nearly_flies", "lift_margin", 0.60),
        ("carries_itself", "lift_margin", 1.00),   # can hold itself up somewhere
        ("leaves_surface", "airborne_fraction", 0.10),
        ("stays_up", "airborne_fraction", 0.60),
        ("glides", "sink_rate", 3.0),          # sink below 3 m/s
        ("holds_height", "sink_rate", 0.5),    # sink below 0.5 m/s
        # Thrust, which nothing in this project measured for four runs.
        #
        # Everything above this point can be earned by a glider: `lift_margin`
        # is the airframe's static lift, and `sink_rate` rewards descending
        # slowly.  A glider converts height into speed and comes down.  The two
        # rungs *below* -- holding a station and climbing -- are exactly the
        # ones that need the flapping to produce net forward force, and they had
        # been reached once and never in 14,092 evaluations.  That is where lift
        # was before arch37: the capability had no measurement, so it had no
        # gradient, and the population optimised what it could see.
        #
        # `thrust_margin` is `(<Fx>_cycle - Fx_static) / drag` at the machine's
        # own trim -- 1.0 means the gait produces the airframe's whole drag, so
        # it holds speed instead of trading height for it.  Measured over
        # arch37's 182 elites at their own gaits: **median -0.0030, max +0.2390,
        # 3.3% above 0.10**, and every hand-built seed is in the same place (the
        # gannet -0.0001, the teal -0.3796 -- its flapping costs 38% more drag
        # than holding still).  Correlation with `lift_margin` is +0.242, so it
        # is not the question the ladder already asks.
        #
        # The thresholds are inside that measured distribution and no higher:
        # 0.00 is "flapping is not a net cost", which half the archive fails.
        # Nothing is placed above +0.239 because nothing has been there --
        # though a random search over 400 gaits reached **+0.8466** on the
        # gannet at 10.95 Hz, so the headroom is real and the rung is a slope
        # rather than a wall.  The gaits that make thrust run at 4.5-11 Hz with
        # the joints spread across the cycle; the population sits at 2.2 Hz.
        ("flaps_forward", "thrust_margin", 0.0),     # ~49% -- not a net cost
        ("makes_thrust", "thrust_margin", 0.05),     # 7.7%
        ("pushes_itself", "thrust_margin", 0.15),    # ~2%
        # Holding *a* height, not losing one slowly.  A machine gliding down at
        # 0.4 m/s clears ``holds_height`` for a whole segment while never
        # holding anything, which is why the two are separate rungs: this one
        # asks the machine to still be at the height it settled at.
        ("holds_station", "station_keeping", 0.6),
        ("climbs", "sink_rate", -0.5),         # net climb
        ("manoeuvres", "turn_rate_held", 0.2),  # turns while holding height
    ],
    "water": [
        # Depth as a gain over where the machine was released, not as an
        # absolute.  ``SPAWN[Domain.WATER]`` puts it four metres under, so the
        # minimum ``max_depth`` over arch37's 14,058 water segments is 3.34 m
        # and the old first two rungs -- 0.5 m and 3.0 m absolute -- were
        # cleared by **100% of every evaluation ever run**, by being dropped.
        # That is the same defect the air ladder had, where ``leaves_surface``
        # and ``stays_up`` were satisfied by falling, and it outlived the fix
        # to that one by three runs.
        #
        # Thresholds from the measured gain distribution over those segments
        # (p25 +0.16, median +2.23, p75 +5.45, p90 +8.11).  Shares left standing
        # are in the comments.  Those quantiles are `max_depth - 4.0` against a
        # release randomised by 0.2 m, so `submerges` is approximate to about
        # that much; arch38 measures the gain directly.  The gain is floored at
        # zero by construction -- a maximum cannot fall below the first sample --
        # so a machine that only rises sits at rung 0 rather than below it.
        ("submerges", "depth_gain", 0.25),      # 70.8% -- goes down at all
        ("dives", "depth_gain", 2.0),           # 51.9%
        ("reaches_depth", "depth_gain", 5.0),   # 28.8%
        ("goes_deep", "depth_gain", 8.0),       # 10.4%
        ("holds_depth", "depth_error", 1.0),   # within 1 m of target
        ("manoeuvres", "water_speed", 0.5),    # makes way while holding depth
    ],
    "land": [
        ("stays_upright", "upright", 0.7),
        ("supports_itself", "contact_fraction", 0.5),
        # Producing motion and sustaining it are different capabilities, and
        # arch34 had no rung between them: 61.6% of the population sat here,
        # upright and self-supporting and under the 0.1 m/s mean that `moves`
        # asks for.  A machine that covers half a metre in one second and then
        # falls averages an eighth of the speed it produced.  `land_peak_speed`
        # is the best one-second displacement rate over windows the machine
        # stayed upright throughout -- measured over 64 arch34 elites its
        # median is 0.086 m/s against a 0.026 m/s mean, so this rung sits
        # inside the population rather than above it.
        ("stirs", "land_peak_speed", 0.08),
        ("moves", "land_speed", 0.1),
        ("walks", "land_speed", 0.4),
        ("climbs_slope", "slope_climbed", 0.5),
    ],
    # Take-off, and it is deliberately a ladder of its own rather than rungs
    # appended to ``land``.  ``rung_reached`` stops at the first unmet rung, and
    # arch34's land distribution is 28.7% at rung 0, 61.6% stuck at rung 2 --
    # upright and self-supporting but under 0.1 m/s -- with 0.5% at the top.
    # Rungs added above ``climbs_slope`` would be visible to one design in two
    # hundred, which is a ladder nobody is standing on.
    #
    # The thresholds are set from the measured population, not from what a
    # take-off ought to look like: 0.05 m is the spawn clearance, arch34's best
    # elite reached 0.133 m, and 0.50 m is the bar ``transitions.py`` uses for
    # a crossing.  So the first rung is inside reach of what already exists and
    # the last is a real climb-out, which is what makes this a slope rather
    # than a wall.
    "takeoff": [
        ("unweights", "takeoff_height", 0.02),   # rises at all, gated on control
        ("hops", "takeoff_height", 0.10),        # clearly leaves the ground
        ("clears", "takeoff_height", 0.30),      # a crossing's worth of height
        ("climbs_out", "takeoff_height", 0.80),  # a departure, not a hop
    ],
    "transition": [
        # Rebuilt from arch37's 14,092 evaluations.  The old ladder put 70.6% of
        # the population on rung 1 and **0.3% above it**, for three reasons that
        # are all visible in the distribution:
        #
        # 1. `crossed_fraction` counts how many of four transitions were made,
        #    so it takes five values: 0, 0.25, 0.5, 0.75, 1.  A rung at 0.34
        #    means "two of four" (70.9%) and the next one at 0.99 means "all
        #    four" (0.3%).  Between them sat nothing, and three-of-four -- 17.1%
        #    of the population -- had no rung of its own.
        # 2. **Completeness came before quality.**  A machine that crossed two
        #    boundaries beautifully scored the same rung as one that crossed two
        #    badly, because every quality rung was placed above `crosses_all`
        #    and `crosses_all` is 0.3%.
        # 3. Two of the quality rungs were set at or above the population's
        #    maximum.  `arrives_usable` asked for `exit_state >= 0.7` and the
        #    **largest value ever measured is 0.689** -- a rung nobody can stand
        #    on, which is the mistake this file records under `moves` at
        #    0.1 m/s.  `stays_controlled` asked for `control >= 0.6` against a
        #    p90 of 0.296.
        #
        # Every threshold below is inside the measured distribution and the
        # shares decrease monotonically, so the ladder is a slope.  Shares are
        # marginal, as the lift and depth rungs' were.
        ("crosses", "crossed_fraction", 0.34),      # 70.9% -- two of four
        ("stays_controlled", "control", 0.20),      # 56.4%
        ("arrives_usable", "exit_state", 0.30),     # 48.2%
        ("arrives_well", "exit_state", 0.37),       # 27.0%
        ("crosses_three", "crossed_fraction", 0.60),  # 17.1% -- three of four
        ("crosses_efficiently", "economy", 0.75),   # 17.1%
        ("survives_entry", "shock", 0.50),          # 16.4%
        ("crosses_all", "crossed_fraction", 0.99),  # 0.3% -- all four
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
    # `depth_gain`, not `max_depth`, for the same reason the water ladder reads
    # it: the segment releases the machine 4 m under, so `max_depth` has a 4 m
    # floor built into it and a bar set on that quantity is set on the spawn as
    # much as on the machine.  Floor 0.0 now means "went no deeper than it was
    # put", which is a real zero.
    "water": ("depth_gain", False, 0.0),
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
