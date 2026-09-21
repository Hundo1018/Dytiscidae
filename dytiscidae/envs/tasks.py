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

A passive machine does the same thing whatever it is commanded, so it cannot
score on both halves of one segment.  That is the job the passive twin did,
done by the command switch at no extra rollout.

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
#: generations (6,383 segments each), at about the 80th percentile -- a command
#: most of the population undershoots but that the upper fifth already makes:
#:
#:     water `water_speed`  p50 0.151  p75 0.271  p90 0.446  m/s
#:     land  `land_speed`   p50 0.020  p75 0.060  p90 0.116  m/s
#:
#: A tracking command is not a full-marks reference: overshooting it scores
#: less, so it has to be a speed the population can reach rather than the 0.993
#: quantile ``FORWARD_REF_SPEED`` used.
TASK_SPEED = {"water": 0.30, "land": 0.08}

#: Commanded hold depth in water, drawn uniformly per generation.  The machine
#: is released four metres under (``TriphibianEnv.SPAWN``), so every draw asks
#: it to *descend* at least 1.5 m and stop: a hull that sits where it was put
#: fails the hold, and one that sinks through the target fails it too.  The top
#: of the range stays short of the mission's 10 m because a 4 s phase has to
#: leave time to settle.
HOLD_DEPTH_RANGE = (5.5, 8.0)

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
            hold_first, heading, depth = True, DEFAULT_HEADING, 6.5
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
