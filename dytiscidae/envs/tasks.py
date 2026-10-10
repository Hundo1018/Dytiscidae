"""What a segment asks the machine to do, phase by phase.

Why this exists
---------------

Until 2026-09-21 the controller was told which medium it was in and never what
to do there.  Its command channel was a three-way one-hot of the domain, so each
medium's score had to blend every goal the medium might have, and the policy --
rewarded once per segment with that blend -- could only learn one compromise.
The blend fought itself: over arch40's 6,288 water segments,
``corr(headway, depth_station_keeping)`` was -0.264, and the water formula paid a
perfect hover 0.20 against a perfect mover 0.80.

The user's rule, which this module states as data: **if the purpose is to go
forward, going forward scores; if the purpose is to hold still in the medium, it
should hold still then, and moving is penalised.**  So a segment is now a short
script of *phases*, each with one purpose, the controller observes the purpose
it is currently under, and each phase is scored on its own objective alone.

The standard this follows is command tracking from legged locomotion: the
policy observes a commanded velocity and is rewarded for matching it, and a
command of zero *is* standing still.

Two phases per segment, each half of it:

    water   hold at a commanded depth  |  cruise on a heading at that depth
            (order drawn per generation)
    air     cruise on the launch heading, holding height  |  cruise on a new
            heading, holding height -- a turn on command.  "Hold" in air is
            holding *height*: almost nothing here can hover, and a hover phase
            would be a wall the whole population stands at.
    land    walk on a heading  |  stop and stay stopped
            (order drawn per generation)

What is scored in each phase (``TriphibianEnv._task_scores``, reshaped
2026-09-22 after 102 generations of arch42 at a median of exactly zero): progress
along a commanded heading, which may be passive -- a glide is a capability --
and the things that must be chosen, which may not: holding at a commanded depth
and staying still, stopping when told, turning when told.  A passive body does
the same thing whatever it is commanded, so it cannot score on those, and no
passive twin has to run to know it.  Holding height counts only the samples
spent in the air: a body that fell into the sea floats at a sink rate of zero,
and without that gate it scored full marks for it.

Everything here is plain data and arithmetic: no MuJoCo, no numpy state.  The
scoring that reads it is ``TriphibianEnv._task_scores``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Commanded cruise speed per medium, in m/s.  Air is absent on purpose: a
#: flapping machine's cruise speed is a property of its wing loading, so the
#: air command is the machine's own trim speed (``TriphibianEnv.launch_speed``),
#: filled in by the environment.
#:
#: Set from the measured distribution of gross speed over arch40's first 400
#: generations (6,383 segments each):
#:
#:     water `water_speed`  p50 0.151  p75 0.271  p90 0.446  m/s
#:     land  `land_speed`   p50 0.020  p75 0.060  p90 0.116  m/s
#:
#: Water sits at about the 80th percentile.  Land was there too, at 0.08, and
#: arch41 stopped at generation 52 with `walk_tracking` p50 0 and p75 0.04: a
#: command the population could not approach is a wall, not a gradient.  It
#: is now 0.04, between land's p50 and p75.
#:
#: A tracking command is not a full-marks reference: overshooting it scores
#: less, so it has to be a speed the population can reach rather than the 0.993
#: quantile ``FORWARD_REF_SPEED`` used.
TASK_SPEED = {"water": 0.30, "land": 0.04}

#: Commanded hold depth in water, drawn uniformly per generation.  The machine
#: is released four metres under (``TriphibianEnv.SPAWN``), so every draw asks
#: it to *descend* at least a metre and stop: a hull that sits where it was put
#: fails the hold, and one that sinks through the target fails it too.
#:
#: Was 5.5-8.0 m.  arch41 stopped at generation 52 with the population's
#: maximum depth at p50 4.71 m and p75 6.93 m -- most machines never got near a
#: 1.5-4 m descent, and 89% scored zero on the hold.  5.0-6.5 m sits inside what
#: the population already reaches.
HOLD_DEPTH_RANGE = (5.0, 6.5)

#: The heading change the air segment's second phase commands, in radians,
#: drawn with a random sign.  Between 45 and 90 degrees: enough that holding the
#: launch heading is a clear miss, small enough to be flown in half a phase.
TURN_RANGE = (math.radians(45.0), math.radians(90.0))

#: The cruise heading when no draw is supplied: across the spawn attitude, not
#: along it.  Every machine is released with its body +x on world +x, so a
#: default of 0 is satisfied by a body that simply goes where it was pointed --
#: measured: the beetle, actuators held still, glides forward while it sinks
#: and scored 0.45 on tracking a heading of 0.  In a run the heading is drawn
#: per generation and a passive glide lines up with it only by chance; the
#: default should not line it up by construction.
DEFAULT_HEADING = math.pi / 2

#: How fast a machine may drift while told to hold and still score, m/s.  The
#: stillness term is scored against the distance this would cover over the
#: window, so drifting at it scores zero.
#:
#: **Set by the gate, not by the population.**  arch41's first 52 generations
#: measured hold-phase drift at p50 0.41 m and p75 0.64 m over the ~2 s window,
#: and a band at that p75 was tried: with the shallower depth range below, it
#: paid a beetle with its actuators held still 0.093 in water (max 0.240), up
#: from 0.011 -- because that population's drift *is* passive sinking, and a
#: threshold set from passive behaviour rewards passive behaviour.  0.45 m/s
#: still paid it 0.056.  At 0.30 m/s the held-still beetle scores 0.018 and a
#: hull sinking at 0.55 m/s scores exactly 0 on stillness.
HOLD_DRIFT_SPEED = 0.30

#: How fast a machine may still be sinking or rising, net over the window, and
#: be holding its depth, m/s.  Scored separately from ``HOLD_DRIFT_SPEED``
#: because that band is the cruise command itself, and a hull passing *through*
#: the commanded depth at a third of it held it well enough to score.
#:
#: **Set below the passive distribution, on purpose.**  Measured 2026-09-22 on
#: arch42's top 40 elites (gen 102): with every actuator held still, 5 of 40
#: water segments scored over 0.1 on hold, up to 0.63 -- an eel of density
#: 1.008 sinking at 0.094 m/s that reached its commanded 5.72 m inside the
#: window, a gannet at the turning point of a passive oscillation at 0.048 m/s --
#: and they crossed at 0.048 to 0.233 m/s.  The 7 *driven* holders crossed at
#: 0.026 to 0.172 m/s: no controller in that population held depth better than
#: a body doing nothing, so hold was paying coincidence.  At 0.05 m/s passing
#: through scores nothing and a controller that slows at the target earns it.
HOLD_SINK_SPEED = 0.05

#: Kinds of phase.  ``stop`` is ``hold`` on land, where there is no depth or
#: height to hold and the only thing to hold is position.
CRUISE, HOLD, STOP = "cruise", "hold", "stop"


@dataclass(frozen=True)
class Phase:
    """One purpose, over one stretch of a segment."""

    kind: str
    #: Fraction of the segment at which this phase begins.
    start: float
    #: World-frame heading in radians (cruise only; 0 is world +x).
    heading: float = 0.0
    #: Commanded horizontal speed, m/s.  Zero for hold and stop.  For air it is
    #: filled in from the machine's own trim speed.
    speed: float = 0.0
    #: Commanded depth in metres (water only).
    depth: float | None = None

    @property
    def moving(self) -> bool:
        return self.kind == CRUISE


@dataclass(frozen=True)
class TaskSchedule:
    """The phases one segment is made of, in order."""

    domain: str
    phases: tuple

    def index_at(self, frac: float) -> int:
        """The phase in force at ``frac`` of the way through the segment."""
        k = 0
        for i, ph in enumerate(self.phases):
            if frac >= ph.start:
                k = i
        return k

    def at(self, frac: float) -> Phase:
        return self.phases[self.index_at(frac)]

    def bounds(self, n: int) -> list:
        """Sample index ranges ``[(lo, hi), ...]`` for a segment of ``n`` samples.

        ``n`` is the length the segment was *asked* for, never the number of
        samples recorded, for the reason ``_score_segment`` documents: a
        segment that ended early is measured against what it was asked to do,
        so a phase it never reached has no samples and scores nothing.
        """
        starts = [int(round(ph.start * n)) for ph in self.phases] + [n]
        return [(starts[i], starts[i + 1]) for i in range(len(self.phases))]


def _key(domain) -> str:
    return getattr(domain, "value", domain)


def schedule_for(domain, rng=None, *, trim_speed: float = 0.0) -> TaskSchedule:
    """The two-phase script for one segment in ``domain``.

    ``rng`` draws the order, the heading and the depth.  The evaluators derive
    it from the evaluation seed and give every machine in a generation the same
    draw, as they do with ``scatter``, so candidates face the same task and a
    score stays reproducible.  Without one the schedule is a fixed default,
    which is what probes and tests get.

    ``trim_speed`` is the air command's speed -- the machine's own.
    """
    key = _key(domain)
    if key == "water":
        if rng is None:
            hold_first, heading, depth = True, DEFAULT_HEADING, 5.75
        else:
            hold_first = bool(rng.random() < 0.5)
            heading = float(rng.uniform(-math.pi, math.pi))
            depth = float(rng.uniform(*HOLD_DEPTH_RANGE))
        hold = Phase(HOLD, 0.0, depth=depth)
        cruise = Phase(CRUISE, 0.0, heading=heading, speed=TASK_SPEED["water"],
                       depth=depth)
        first, second = (hold, cruise) if hold_first else (cruise, hold)
    elif key == "air":
        if rng is None:
            turn = math.radians(60.0)
        else:
            turn = float(rng.uniform(*TURN_RANGE)) * (1.0 if rng.random() < 0.5 else -1.0)
        v = float(trim_speed)
        first = Phase(CRUISE, 0.0, heading=0.0, speed=v)
        second = Phase(CRUISE, 0.0, heading=_wrap(turn), speed=v)
    elif key == "land":
        if rng is None:
            walk_first, heading = True, DEFAULT_HEADING
        else:
            walk_first = bool(rng.random() < 0.5)
            heading = float(rng.uniform(-math.pi, math.pi))
        walk = Phase(CRUISE, 0.0, heading=heading, speed=TASK_SPEED["land"])
        stop = Phase(STOP, 0.0)
        first, second = (walk, stop) if walk_first else (stop, walk)
    else:
        raise ValueError(f"no task schedule for domain {domain!r}")
    return TaskSchedule(key, (_at(first, 0.0), _at(second, 0.5)))


def mirror(schedule: TaskSchedule) -> TaskSchedule:
    """The same air script with the drawn turn the other way.

    Air's headings are relative to the launch -- cruise straight, then a turn
    of 45-90 degrees to a drawn side -- so its pair is the mirror, not the
    antipode: the straight phase is the same in both halves and the turn goes
    to opposite sides.  A lateral velocity the command did not choose (a
    glider slipping to one side) reads toward the commanded side in one half
    and away from it in the other, and the turn response is taken on the mean
    of the two, so it cancels.  The user's answer of 2026-10-09: air, too, is
    scored on a commanded difference, after the fixed still machine cleared air
    >= 0.012 on 11 arch49 gannets against the elites' 6.
    """
    return TaskSchedule(schedule.domain, tuple(
        Phase(ph.kind, ph.start, _wrap(-ph.heading), ph.speed, ph.depth)
        if ph.kind == CRUISE else ph
        for ph in schedule.phases))


def pair_of(schedule: TaskSchedule) -> TaskSchedule:
    """The second half of a paired segment: the antipode in water and on land,
    the mirror in the air."""
    return mirror(schedule) if schedule.domain == "air" else antipode(schedule)


#: The media whose segment is scored as a pair from one initial state
#: (``pair_of``): water and land on the antipode of the drawn heading, air on
#: the mirror of the drawn turn (since 2026-10-10).
PAIRED_MEDIA: tuple = ("water", "land", "air")


def antipode(schedule: TaskSchedule) -> TaskSchedule:
    """The same script with every cruise heading turned by pi.

    Water and land are scored as a pair: the drawn heading and its opposite,
    from the same initial state, with the progress term taken on the *mean of
    the signed* speed along the two headings (``TriphibianEnv.pair_partner``).
    Any velocity that does not depend on the command -- a dense body sinking
    along a glide, a body that swims forward and does not steer -- moves the
    same way in both halves and cancels exactly.

    Measured on arch48 (docs/PAPERS_2610.md §3): a fresh seed took Tier-1 water
    passes from 15/16 to 3/16 and land from 13/16 to 0/16, because one heading
    was drawn per generation and the archive kept the best of each draw; in an
    eight-heading sweep every elite earned its credit on 1-3 headings, and the
    still machine cleared water 0.15 in 35 of 200 against the elites' 45.
    """
    return TaskSchedule(schedule.domain, tuple(
        Phase(ph.kind, ph.start, _wrap(ph.heading + math.pi), ph.speed, ph.depth)
        if ph.kind == CRUISE else ph
        for ph in schedule.phases))


def _at(ph: Phase, start: float) -> Phase:
    return Phase(ph.kind, start, ph.heading, ph.speed, ph.depth)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def task_seed(scatter_seed: int) -> int:
    """The task draw for one segment, derived from its scatter draw.

    One definition, imported by both evaluators, so the two paths cannot come
    to disagree about what a machine was asked to do.
    """
    return (int(scatter_seed) * 2862933555777941757 + 3037000493) & 0x7FFFFFFF
