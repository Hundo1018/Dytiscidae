"""The triphibian mission environment.

Mission
-------
Start in a randomly chosen domain, then cycle air -> water -> land -> air ...
spending five minutes in each, three times round, for forty-five minutes total.
While submerged, reach and hold ten metres.

Why this is evaluated at three fidelities
-----------------------------------------
Simulating forty-five minutes of physics at 250 Hz is 675,000 steps.  At the
cost of the fluid solver that is roughly a minute of wall clock per candidate,
which for a population-based search on four cores is about forty candidates an
hour.  That is not a search, it is a slideshow.

So evaluation is staged, and the great majority of candidates are killed by the
cheapest stage that can honestly kill them:

* **Tier 0 -- analytic, ~0.2 ms.**  Closed-form power and feasibility from
  geometry alone.  A machine whose wing loading implies a 40 m/s stall speed, or
  whose battery cannot supply cruise power, or whose spar snaps under its own
  flapping, is rejected here.  This removes ~90% of random genomes.

* **Tier 1 -- short dynamic, ~2-6 s.**  Ten-to-fifteen second episodes in each
  domain plus the four interesting transitions.  Measures what cannot be derived
  from geometry: whether it is controllable, what it actually costs to hold
  station, and whether the transitions destroy it.  The forty-five minute energy
  budget is then *extrapolated* from measured steady-state power, which is
  legitimate precisely because steady cruise is steady.

* **Tier 2 -- full mission, ~40-90 s.**  The real schedule with disturbances,
  run only on archive elites the curator promotes.  This is where extrapolation
  is checked against reality, and where an elite that only looked good because
  its Tier-1 window was too short gets found out.

The tiers are not independent estimates to be averaged; they are a filter
cascade, and each one's job is to be *cheaply wrong in the safe direction*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from enum import Enum

import numpy as np

from ..control.cpg import CPG, CPGParams, MobilityBasis, identify_mobility
from ..core.mjcf import compile_phenotype
from ..core.phenotype import Phenotype
from ..physics.energy import (
    PowerBudget,
    crawl_power_land,
    cruise_power_air,
    cruise_power_water,
    transition_energy,
)
from ..physics.fluid import CN_LEV, FluidSolver
from ..physics.medium import GRAVITY, MediumField, SeaState
from ..physics.structure import ballast_pump_power
from .tasks import (CRUISE, HOLD, HOLD_DRIFT_SPEED, HOLD_SINK_SPEED, STOP, TASK_SPEED, TaskSchedule,
                    schedule_for)


class Domain(str, Enum):
    AIR = "air"
    WATER = "water"
    LAND = "land"


DOMAIN_CYCLE = [Domain.AIR, Domain.WATER, Domain.LAND]

#: The four transitions that actually happen when cycling air->water->land->air.
CYCLE_TRANSITIONS = ["air_to_water", "water_to_land", "land_to_air"]


@dataclass
class MissionSpec:
    """The mission the user specified, with every number left adjustable.

    These are reference values, not hard constraints: the user asked the system
    to discover its own best specification, so the scorer treats them as targets
    to approach and reports how close the Pareto front gets.
    """

    cycles: int = 3
    seconds_per_domain: float = 300.0
    target_depth: float = 10.0
    #: Reference mass.  Not enforced; recorded so the archive can be read
    #: against the original 15 kg ambition.
    reference_mass: float = 15.0
    #: How far back from its interface each transition probe starts, metres
    #: (ROADMAP Y/O).  Empty is 0 for every kind: the probe starts at the
    #: waterline, the shore or 2.5 m over the sea, as it always has.  Set by the
    #: search's ``DistanceCurriculum`` as the population learns to cross; at its
    #: limit a probe is the continuous mission.  See ``transitions._place_for``.
    transition_back: dict = field(default_factory=dict)
    #: The air segment's launch height, metres; ``None`` is ``SPAWN`` (30 m).
    #: The same curriculum run the other way (item O): it steps *down* as the
    #: population learns to hold height, so a score stops being paid for time
    #: spent falling from a height nothing climbed to.
    air_launch_height: float | None = None

    @property
    def total_seconds(self) -> float:
        return self.cycles * len(DOMAIN_CYCLE) * self.seconds_per_domain

    @property
    def transitions(self) -> list[str]:
        return CYCLE_TRANSITIONS * self.cycles


def command_statistics(commands) -> tuple[float, float] | None:
    """How much, and how erratically, a controller's commands move.

    ``commands`` is the commanded body rate at each control decision.  Returns
    ``(rate, reversal)``: the mean size of a step between consecutive commands,
    and the fraction of consecutive steps that reverse direction.  A smooth
    controller reverses well under half the time; one chattering at the
    control rate reverses on most steps.  ROADMAP item Y measured the filmed
    arch40 machine at 72% on the beach, a loop through its body-rate channels.
    None when there are too few decisions to say.
    """
    c = np.asarray([np.asarray(x, float).ravel() for x in (commands or [])])
    if c.ndim != 2 or len(c) < 3:
        return None
    d = np.diff(c, axis=0)
    size = np.linalg.norm(d, axis=1)
    rate = float(size.mean())
    live = (size[:-1] > 1e-9) & (size[1:] > 1e-9)
    if not np.any(live):
        return rate, 0.0
    dots = np.einsum("ij,ij->i", d[:-1], d[1:])[live]
    return rate, float(np.mean(dots < 0.0))


@dataclass
class SegmentResult:
    """What one stretch of operating in one domain produced."""

    domain: Domain
    duration: float = 0.0
    distance: float = 0.0
    mean_speed: float = 0.0
    mean_power: float = 0.0
    survived: bool = True
    failure: str = ""
    #: MuJoCo bad-qacc events during this segment.  MuJoCo 3.x *auto-resets*
    #: the state to the initial pose when qacc goes non-finite, so a blowup
    #: does not trip the position-divergence guard -- the machine teleports to
    #: spawn mid-rollout and keeps being scored.  A segment with any of these
    #: is marked unstable rather than trusted.
    bad_qacc: int = 0
    # Domain-specific competence, all in [0, 1].
    competence: float = 0.0
    max_depth: float = 0.0
    depth_error: float = 0.0
    altitude_held: float = 0.0
    ground_contact_fraction: float = 0.0
    # Physical stress witnesses.
    peak_slam: float = 0.0
    max_actuator_overload: float = 0.0
    attitude_rms: float = 0.0
    #: Raw physical quantities the judge's fixed ladder is defined on.  Kept
    #: separate from ``competence`` because the ladder must stay comparable
    #: across a whole run while the scoring on top of it moves.
    measurements: dict = field(default_factory=dict)
    #: What the competence is made of, recorded by ``_score_segment`` at the
    #: exit it took: ``gate`` (state preconditions, multiplied) and ``control``
    #: (the medium's motion terms in [0, 1]).  Their product is ``competence``.
    parts: dict = field(default_factory=dict)

    @property
    def cost_of_transport(self) -> float:
        """Dimensionless energy per unit distance per unit weight."""
        if self.distance < 0.1:
            return float("inf")
        return self.mean_power * self.duration / max(self.distance, 1e-6)


@dataclass(eq=False)
class MissionResult:
    """Aggregate of a whole evaluation, at whatever fidelity produced it."""

    tier: int
    feasible: bool = False
    structural_margin: float = 0.0
    segments: dict[str, SegmentResult] = field(default_factory=dict)
    #: Tier-2 only: one leg in each medium the mission never reached, run for
    #: the critic's labels.  Kept apart so nothing that scores ``segments`` --
    #: the mission fraction, energy, ``fitness`` -- can see them.
    probe_segments: dict[str, SegmentResult] = field(default_factory=dict)
    transition_ok: dict[str, bool] = field(default_factory=dict)
    #: The graded record of every crossing attempted.  ``transition_ok`` is kept
    #: as the boolean summary because the curator and the telemetry read it, but
    #: the score comes from here.
    transitions: "object" = field(default_factory=lambda: __import__(
        "dytiscidae.envs.transitions", fromlist=["TransitionSet"]).TransitionSet())
    energy_required_wh: float = 0.0
    energy_available_wh: float = 0.0
    mission_fraction: float = 0.0
    mobility: dict[str, MobilityBasis] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    wall_time: float = 0.0
    #: Set when the evaluation detected the candidate exploiting the simulator
    #: rather than solving the task.  The curator culls these on sight.
    exploit: str = ""
    #: How many of this evaluation's rollouts (domain segments and transition
    #: legs) ended in numerical divergence, against how many were run.  A
    #: diverged rollout zeroes its competence, and mission_fraction is gated on
    #: the *minimum* competence -- so the divergence rate bounds how much of the
    #: selection signal is numerics rather than behaviour, and it has to be
    #: visible per generation to be managed at all.
    diverged_rollouts: int = 0
    n_rollouts: int = 0
    #: Flight gates this design fails, as reasons.  Computed closed-form at
    #: Tier 0 and carried forward, because Tier 1 used to simulate an air
    #: segment for a machine Tier 0 had already established has no wing, and
    #: score it.  See ``airworthiness``.
    air_gates: list[str] = field(default_factory=list)

    @property
    def energy_margin(self) -> float:
        if self.energy_required_wh <= 1e-6:
            return -1.0
        return self.energy_available_wh / self.energy_required_wh - 1.0


# --------------------------------------------------------------------------
# Tier 0: analytic
# --------------------------------------------------------------------------

#: Bounds on the launch airspeed, m/s.  Outside this band the quasi-steady
#: coefficients are extrapolating and the design is not one this mission is
#: about anyway.
LAUNCH_SPEED_RANGE = (6.0, 30.0)

#: The largest lift coefficient the strip model will produce: the separated
#: branch ``CN sin a cos a`` at 45 degrees with a full leading-edge vortex,
#: ``CN_LEV / 2`` = 1.7 (the attached branch peaks lower, ``2 pi`` times the
#: 11 degree stall angle, 1.21).  It was 1.8 against a model that capped at
#: ``1.2 * 1.9``; derived from the model since 2026-09-23 so the two cannot
#: drift apart.
CL_MAX = CN_LEV / 2.0

#: A lifting surface smaller than this is not a lifting surface.  The same
#: figure ``Phenotype.is_plausible_flyer`` uses to decide whether flight load
#: cases apply at all.
WING_AREA_FLOOR = 1e-3  # m^2

#: The highest scored take-off available to a design with no lifting surface.
#: Just under the `clears` rung at 0.30 m, so a wingless body keeps `unweights`
#: and `hops` -- it can be thrown, and leaving the ground is all those two rungs
#: ask -- and cannot reach `clears` ("a crossing's worth of height") or
#: `climbs_out` ("a departure, not a hop"), both of which are flight claims that
#: `airworthiness` already refuses such a design.
TAKEOFF_WINGLESS_CAP = 0.29  # m

#: Segment-mean posture a design must hold for its take-off to be scored at all.
#: The same 0.7 the `stays_upright` land rung uses, deliberately: a second bar
#: for the same property would be a second standard with no measurement behind
#: it.  Set from arch35's distribution -- see the block in `_score_segment`.
TAKEOFF_POSTURE_BAR = 0.7

#: Wing loading at which the stall speed is exactly the top of the launch band:
#: ``0.5 rho V^2 CL_max`` = 992 N/m^2 at 30 m/s.  Above it there is no speed
#: this mission will fly a machine at, and no attitude at that speed, which
#: produces the machine's own weight in lift.  Derived rather than chosen.
MAX_WING_LOADING = 0.5 * 1.225 * LAUNCH_SPEED_RANGE[1] ** 2 * CL_MAX

#: Body angular rate above which nothing measured over an air segment is an
#: attitude that was *held*.  One revolution per second turns the machine
#: through 160 degrees within a single stroke at the reference design's 2.2 Hz,
#: so its angle of attack sweeps the whole circle several times inside the
#: window the sink rate is averaged over.
MAX_SPIN_RATE = 2.0 * math.pi  # rad/s

#: Correlation between commanded and achieved angular rate below which an
#: attitude change was not asked for.  Zero is "the machine's rotation is
#: unrelated to its commands"; this is deliberately close to zero, because the
#: claim being defended against is a design with *no* control authority at all.
MIN_TURN_AUTHORITY = 0.3

#: Sink rate at which a descent stops being a glide and becomes a fall.
#:
#: At the bottom of the launch band the flight path is then steeper than 45
#: degrees, so nothing about the trajectory is being produced by lift.  This is
#: what replaced the flat ``+ 0.25`` the air score used to add for being off the
#: ground at all: gliding is a real capability and has to keep scoring
#: something, but it has to be *earned* by the descent rate rather than paid to
#: anything that has left the surface.  Measured over the seven seed plans, the
#: flappers sink at 10-14 m/s and the gannet at 1.7.
SINK_BALLISTIC = LAUNCH_SPEED_RANGE[0]  # m/s

#: What a gated design's air segment is worth.  Not zero.
#:
#: A hard zero would make ``mission_fraction`` -- which is gated on the *minimum*
#: competence -- exactly zero for every wingless machine at once, and with it
#: every gradient back toward growing a wing.  ``fitness`` declines to zero
#: infeasible designs for the same reason.  A twentieth separates the two
#: classes decisively without flattening one of them: measured on arch33,
#: winged designs scored 0.144 in air and wingless 0.136, and at this credit
#: the wingless class scores 0.007 against the same 0.144.
GATED_AIR_CREDIT = 0.05


#: The window the air score's height term looks for a drop in, s.  One second:
#: long enough that a stroke's own heave (a few Hz) averages out, short enough
#: that a tumble's fall does not.
HEIGHT_WINDOW = 1.0

#: The level-flight gate (ROADMAP AD): the air task's *flight* term -- holding
#: height, as distinct from gliding down -- is paid only to a machine whose
#: actuators can hold it level on the rig, ``level_margin >= LEVEL_GATE``.  Set
#: by the user 2026-09-30 from arch44's gen-200 distribution (p50 0.006, p90
#: 0.705, max 1.354): about one design in ten clears it, so it gates the part of
#: the score that means flight and leaves the glide term to everyone -- gating
#: all of air would put 90% of the population on the floor, the `moves` wall.
#: A margin the rig could not measure does not clear it.
LEVEL_GATE = 0.7


def rotor_lift_ratio(p: Phenotype) -> float:
    """Static thrust of every propeller at top speed, in air, over the weight."""
    from ..core.mjcf import ROTOR_TIP_SPEED
    from ..physics.medium import AIR
    from ..physics.rotor import bemt
    specs = [s.rotor for s in getattr(p, "segments", []) if getattr(s, "rotor", None) is not None]
    if not specs:
        return 0.0
    thrust = sum(bemt(r, ROTOR_TIP_SPEED / r.radius, 0.0, 0.0, AIR.rho, AIR.mu)[0] for r in specs)
    return float(thrust / max(p.mass * GRAVITY, 1e-9))


def airworthiness(p: Phenotype) -> list[str]:
    """Which flight gates a design fails, as reasons.  Empty means none.

    These are Tier-0 facts -- geometry and mass, no simulation -- about a
    family of designs the search kept finding and the scoring kept paying.
    arch33's mission champion was 12 kg with ``wing_area`` 0.0000 m^2, a wing
    loading of 1,178,337 N/m^2 and zero actuated degrees of freedom, and it was
    recorded *climbing* at 2.08 m/s while spinning at 31.4 rad/s.  It scored
    0.18, and the existing exploit flag needs 0.35 to disqualify, so nothing in
    the run objected.

    Tier 0 already knew: ``p_air`` is infinite for such a machine and the note
    reads "no lifting surface".  Tier 1 then simulated an air segment for it
    anyway and scored the result, so the closed-form finding was overwritten by
    a number produced by throwing the object.  These gates carry the Tier-0
    finding into the air score.

    What a gate does *not* do is reject the design.  A wingless submarine is a
    legitimate water specialist and the islands exist so it can be one; it
    simply cannot be credited with flight.
    """
    out: list[str] = []
    # A machine whose propellers can hold its weight up is airworthy without a
    # wing: that is what a multirotor is.  Static thrust at top speed, in air.
    if rotor_lift_ratio(p) >= 1.0:
        return out
    if p.wing_area < WING_AREA_FLOOR:
        out.append("no lifting surface")
    elif p.wing_loading > MAX_WING_LOADING:
        out.append(
            f"wing loading {p.wing_loading:.0f} N/m^2 exceeds "
            f"{MAX_WING_LOADING:.0f}: no launch speed carries the weight"
        )
    return out


def _turn_authority(commands, responses) -> tuple:
    """How much of a machine's rotation it actually asked for.

    Returns ``(correlation, mean_rate)``.  ``commands`` are the angular halves
    of the body twists the controller commanded, one per control decision, and
    ``responses`` are the angular body rates measured at those same decisions.
    The command at step *k* is paired with the response at *k+1*, so the machine
    is given one control interval to answer.

    Both are in the body frame and in rad/s -- ``MobilityBasis.twist_of`` is the
    forward model of the same identification the commands are expressed in -- so
    the correlation is between a request and its outcome on the same axes.

    A design with no controller, no mobility basis or no actuated degrees of
    freedom produces no commands, and the answer is zero: whatever it is doing
    with its attitude, it did not ask for it.
    """
    if commands is None or responses is None:
        return 0.0, 0.0
    c = np.asarray(commands, float)
    a = np.asarray(responses, float)
    n = min(len(c), len(a)) - 1
    if n < 2 or c.ndim != 2 or a.ndim != 2:
        return 0.0, 0.0
    x = c[:n].ravel()
    y = a[1:n + 1].ravel()
    rate = float(np.mean(np.linalg.norm(a[1:n + 1], axis=1)))
    if float(np.std(x)) < 1e-9 or float(np.std(y)) < 1e-9:
        return 0.0, rate
    corr = float(np.corrcoef(x, y)[0, 1])
    return (corr if np.isfinite(corr) else 0.0), rate


def evaluate_tier0(p: Phenotype, spec: MissionSpec | None = None) -> MissionResult:
    """Closed-form feasibility and energy budget.  No simulation."""
    spec = spec or MissionSpec()
    r = MissionResult(tier=0)
    r.structural_margin = p.report.min_margin
    r.feasible = p.report.ok

    mass = p.mass
    # Best cruise speed: fly at the CL that maximises L/D, approximated by the
    # speed where induced and profile drag balance.
    if p.wing_area > 1e-4:
        v_stall = math.sqrt(2 * mass * GRAVITY / (1.225 * p.wing_area * 1.8))
        v_cruise = float(np.clip(1.35 * v_stall, 4.0, 35.0))
        p_air, note_air = cruise_power_air(
            mass=mass, span=p.max_span, wing_area=p.wing_area, speed=v_cruise
        )
    else:
        v_cruise, p_air, note_air = 0.0, float("inf"), "no lifting surface"

    p_water, note_water = cruise_power_water(
        volume=p.displaced_volume,
        frontal_area=p.frontal_area,
        speed=1.0,
        seal_count=p.n_sealed,
    )
    # Holding depth against a compressing gas bladder costs continuous pumping.
    p_water += ballast_pump_power(spec.target_depth, flow_m3_s=1e-4 * max(p.depth_instability, 0.0))

    p_land, note_land = crawl_power_land(mass=mass, speed=0.3)

    for dom, power, note in (
        (Domain.AIR, p_air, note_air),
        (Domain.WATER, p_water, note_water),
        (Domain.LAND, p_land, note_land),
    ):
        ok = math.isfinite(power) and power < 1e5
        r.segments[dom.value] = SegmentResult(
            domain=dom,
            duration=spec.seconds_per_domain,
            mean_power=power if ok else 0.0,
            survived=ok,
            failure="" if ok else note,
            competence=1.0 if ok else 0.0,
        )

    cruise_j = sum(
        s.mean_power * spec.seconds_per_domain * spec.cycles for s in r.segments.values()
    )
    trans_j = sum(transition_energy(mass, k) for k in spec.transitions)
    r.energy_required_wh = (cruise_j + trans_j) / 3600.0
    r.energy_available_wh = p.genome.battery_wh * 0.85  # usable fraction of the pack

    if not math.isfinite(r.energy_required_wh) or r.energy_required_wh > 1e5:
        r.mission_fraction = 0.0
        r.notes.append("energy diverged: at least one domain is unflyable")
    else:
        r.mission_fraction = float(
            np.clip(r.energy_available_wh / max(r.energy_required_wh, 1e-6), 0.0, 1.0)
        )
    if not r.feasible:
        w = p.report.worst
        r.notes.append(f"structural: {w.name} margin {w.margin:+.2f}" if w else "structural")
    r.air_gates = airworthiness(p)
    r.notes.extend(f"air gate: {g}" for g in r.air_gates)
    r.notes.append(f"v_cruise={v_cruise:.1f}m/s P_air={p_air:.0f}W P_water={p_water:.0f}W")
    return r


# --------------------------------------------------------------------------
# Tier 1: short dynamic episodes
# --------------------------------------------------------------------------


#: Channels of body identity appended to every observation.  See
#: ``TriphibianEnv.morphology_context``.
MORPHOLOGY_DIM = 8


def morphology_channels(*, mass: float, density_ratio: float, wing_area: float,
                        span: float, aspect_ratio: float, wing_loading: float,
                        n_actuated: int, battery_wh: float) -> np.ndarray:
    """The eight body-identity channels, from scalars rather than a phenotype.

    Split out so that anything reading a stored design's morphology -- the
    distillation study reads it out of archive JSON, where every one of these
    is already recorded -- computes exactly the transform the policy was trained
    under.  A second copy of this arithmetic is how a stored policy comes to
    mean something different from what it meant when it was fitted.

    Every channel is scaled to roughly [-1, 1] by a *fixed* transform rather
    than by population statistics, because a normalisation that moves would make
    a stored policy mean something different in a later generation.
    """
    mass = float(max(mass, 1e-3))
    wing = float(max(wing_area, 0.0))
    return np.array([
        np.clip((np.log10(mass) - np.log10(0.4))
                / (np.log10(40.0) - np.log10(0.4)) * 2.0 - 1.0, -1.5, 1.5),
        np.clip(np.tanh(float(density_ratio) - 1.0), -1.0, 1.0),
        np.clip(np.tanh(wing / 0.5), 0.0, 1.0),
        np.clip(float(max(span, 0.0)) / 3.0, 0.0, 1.5),
        np.clip(float(aspect_ratio) / 20.0, 0.0, 1.5),
        np.clip(np.log10(max(float(wing_loading), 1.0)) / 3.0 - 1.0, -1.5, 1.5),
        np.clip(int(n_actuated) / 16.0, 0.0, 1.5),
        np.clip(float(battery_wh) / (100.0 * mass), 0.0, 1.5),
    ], float)


class TriphibianEnv:
    """A compiled machine in the triphibian world, steppable by a controller."""

    #: Spawn poses per domain.  Water is deep enough that the free surface is
    #: not doing the work, land is up the beach (the exact height is derived
    #: from the machine, see ``_clear_of_terrain``).
    #:
    #: Air is a *launch*, not a drop, and the difference decided the whole air
    #: score.  The old spawn released the machine at 6 m with zero airspeed:
    #: 1.1 s of free fall inside an 8 s segment, so the airborne fraction --
    #: which every air term is multiplied by -- could not exceed 0.15 however
    #: well the thing flew.  Measured across all five plans it was 0.136 to
    #: 0.173, and the observed air scores were 0.035 to 0.044.  The ceiling was
    #: set by the height of the drop, not by aerodynamics, so the search was
    #: being asked to optimise a number it could barely move.
    #:
    #: A flight test does not start with the aircraft at rest in mid-air, so
    #: the air segment is a launch.  The speed is each design's own measured
    #: trim speed (see ``launch_speed``), which is only defensible now that the
    #: two things that made it a *gift* are gone: a design with no trim speed
    #: anywhere is released at the bottom of the band rather than the top, and
    #: the air score no longer pays for the velocity it was handed.
    #: The window `sink_rate` and `station_keeping` are measured over, in
    #: seconds, independent of how long the segment is.  Without it both
    #: statistics change meaning with `segment_seconds`, and the rungs that read
    #: them stop being comparable between runs.  4.0 s is the value they had
    #: implicitly at an 8 s segment, chosen so nothing measured before this
    #: change moves.
    STATION_WINDOW = 4.0

    #: How long a machine must actually be airborne before the air segment will
    #: publish episode measurements at all -- an absolute duration, in seconds,
    #: for the same reason `STATION_WINDOW` is one.
    #:
    #: This was `0.35 * res.duration`, a fraction of a configurable window, and
    #: at `--segment-seconds 24` it silently became 8.4 s instead of 2.8 s.
    #: arch38's first launch is kept in `runs/arch38_void_window_gate/`: over its
    #: first 1,601 air segments **1,571 took the short-hop exit and 24 took the
    #: full one -- 1.5%** -- so `glides`, `holds_height`, the three thrust rungs,
    #: `holds_station`, `climbs` and `manoeuvres`, nine of fourteen, were
    #: unreachable for 99.6% of the population.  And the share was *falling*,
    #: 2.4% to 0.4% over the first hundred generations, because selection could
    #: not see the rungs above the gate and so stopped paying for staying up.
    #:
    #: 2.8 s is exactly what `0.35 * 8.0` meant, so nothing measured at an 8 s
    #: segment moves.  The survival question the longer segment is *supposed* to
    #: ask is still asked, by `airborne_fraction` and the two rungs that read
    #: it, which remain fractions and do get harder as the segment grows.
    MEASURABLE_AIR_SECONDS = 2.8

    SPAWN = {
        Domain.AIR: (-40.0, 0.0, 30.0),
        Domain.WATER: (-8.0, 0.0, -4.0),
        Domain.LAND: (15.0, 0.0, 0.9),
    }

    #: What the next segment will ask, set by the evaluator after ``scatter``
    #: from a draw every machine in the generation shares (``tasks.task_seed``).
    #: ``None`` means "whatever ``tasks.schedule_for`` gives by default", which
    #: is what probes and tests that call ``rollout`` directly get.
    task: "TaskSchedule | None" = None
    #: ``MissionSpec.air_launch_height`` for this machine; ``None`` is ``SPAWN``.
    #: Read by ``reset`` only -- the trim sweep and the level rig keep the
    #: canonical pose, since they measure the airframe, not the episode.
    air_launch_height: float | None = None
    #: The schedule in force while a segment is running, and the clock it runs
    #: on.  ``None`` outside a segment -- during a transition, or the continuous
    #: mission's own loop -- so the controller is told no task there, which is
    #: what it was trained with in those places.
    _active_task: "TaskSchedule | None" = None
    _seg_t0: float = 0.0
    _seg_T: float = 1.0
    #: Height above the surface when the segment began: the air hold's
    #: reference, and what the controller's height-error channel reads against.
    _height_ref: float = 0.0

    #: Tolerance for holding a commanded depth, in metres: the water ladder's
    #: ``holds_depth`` rung has always meant "within 1 m of the target", and the
    #: hold phase is that rung's task.
    DEPTH_BAND = 1.0


    def __init__(
        self,
        phenotype: Phenotype,
        *,
        sea_state: SeaState | None = None,
        current: np.ndarray | None = None,
        wind: np.ndarray | None = None,
        timestep: float = 0.004,
        seed: int = 0,
        perturb: dict | None = None,
        detail: bool = False,
    ) -> None:
        """``detail`` draws the surfaces as the shape the fluid solver reads
        rather than as the flat box that collides for them.  Rendering wants it;
        search does not, and pays about a quarter of its step budget for it."""
        self.p = phenotype
        #: Flight gates from geometry and mass alone; see ``airworthiness``.
        self.air_gates = airworthiness(phenotype)
        self._morph_ctx = None
        self.rng = np.random.default_rng(seed)
        #: Weight of the command-rate penalty (ROADMAP item Y), 0 = off.  Read
        #: from the environment because the search's worker processes build
        #: their own envs and inherit the parent's environment, not its
        #: objects; `run_search` sets it from `SearchConfig.action_rate_penalty`.
        import os as _os
        self.action_rate_penalty = float(
            _os.environ.get("DYTISCIDAE_ACTION_RATE_PENALTY", "0") or 0.0)
        self.medium = MediumField(sea_state=sea_state, current=current, wind=wind)
        self.timestep = timestep

        from ..core.mjcf import scene_xml

        scene = scene_xml(timestep=timestep)
        self.model, self.data, self.act_names, self.panels = compile_phenotype(
            phenotype, scene=scene, detail=detail
        )
        # The servo's own lag, kv/kp per actuator (biasprm = [0, -kp, -kv]).
        # A position actuator damps the joint's *absolute* velocity, so on its
        # own it is a first-order lag of kv/kp = 75 ms -- a 2.1 Hz corner
        # whatever the motor, and measured on the gannet it reached 27% of a
        # 7.3 Hz stroke with its torque never saturating.  A servo tracking a
        # trajectory feeds the reference rate forward; commanding
        # ``q_ref + (kv/kp) dq_ref/dt`` is exactly that, since
        # kp(q_ref + tau dq_ref - q) - kv dq = kp(q_ref - q) + kv(dq_ref - dq).
        # The torque limit is untouched, so what the motor cannot do it still
        # cannot: the same gannet reaches 98% at 7.3 Hz and saturates 21% of the
        # time (experiments/flight_audit/servo_probe.py).
        bp = np.asarray(self.model.actuator_biasprm, float)
        self.servo_lead = (np.where(bp[:, 1] < 0, bp[:, 2] / np.minimum(bp[:, 1], -1e-12), 0.0)
                           if self.model.nu else np.zeros(0))
        # ``perturb`` moves the model's own coefficients.  It exists for the
        # auditor: the only way to find out whether a design depends on the
        # model being exactly right is to make the model wrong on purpose.
        pert = dict(perturb or {})
        self.solver = FluidSolver(
            self.model, self.panels, self.medium,
            added_mass_scale=float(pert.get("added_mass_scale", 1.0)),
            cd_scale=float(pert.get("cd_scale", 1.0)),
            lift_scale=float(pert.get("lift_scale", 1.0)),
        )
        from ..core.phenotype import build_jets
        self.jets = build_jets(phenotype, self.model)
        from ..physics.rotor import RotorSet
        self.rotors = RotorSet(self.model, {
            f"{s.name}_rot": s.rotor for s in phenotype.segments
            if getattr(s, "rotor", None) is not None})

        import mujoco

        self._mj = mujoco
        self.root_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, phenotype.segments[0].name
        ) if phenotype.segments else 0

        # Joint travel limits for the CPG, plus the state addresses of the same
        # actuated joints so the observation can report stroke phase.
        ranges, qadr, vadr = [], [], []
        for name in self.act_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            jid = self.model.actuator_trnid[aid, 0]
            ranges.append(self.model.jnt_range[jid])
            qadr.append(self.model.jnt_qposadr[jid])
            vadr.append(self.model.jnt_dofadr[jid])
        self.joint_range = np.array(ranges) if ranges else np.zeros((0, 2))
        self._act_qadr = np.array(qadr, int)
        self._act_vadr = np.array(vadr, int)
        self.cpg = CPG(
            len(self.act_names),
            base_frequency=phenotype.genome.flap_frequency,
            joint_range=self.joint_range if len(ranges) else None,
        )
        # Seed the oscillator phases from the genome rather than from a fixed
        # linspace.  Without this ``Part.phase_offset`` is a gene nothing reads,
        # and a travelling wave along a serial chain -- the entire anguilliform
        # and batoid family of gaits -- stays unreachable no matter what the
        # search does.
        phases, amps, offs = [], [], []
        lo, hi = self.cpg.lo, self.cpg.hi
        for k, name in enumerate(self.act_names):
            seg_name = name[:-2]  # strip the "_a" suffix
            seg = next((x for x in phenotype.segments if x.name == seg_name), None)
            # Stroke amplitude and neutral position, likewise.  Both were
            # constants -- 0.45 of half-travel about the midpoint, for every
            # joint of every design -- which put every fixed-wing and every
            # folding-wing configuration outside the search space, because the
            # only way to stop a surface being shaken was to leave it
            # unactuated and an unactuated surface cannot fold or take weight.
            part = seg.part if seg is not None else None
            frac = float(getattr(part, "stroke_amplitude", 0.45)) if part else 0.45
            neut = float(getattr(part, "neutral", 0.5)) if part else 0.5
            half = 0.5 * (hi[k] - lo[k])
            amps.append(np.clip(frac, 0.0, 1.0) * half)
            offs.append(lo[k] + np.clip(neut, 0.0, 1.0) * (hi[k] - lo[k]))
            # Multiplied by chain depth, because ``phase_offset`` is a *phase
            # advance per link*, not an absolute phase.  Every segment in a
            # recursive chain shares one Part, so a flat offset makes the whole
            # chain beat in unison -- which is not a travelling wave, it is a
            # very long paddle.  Accumulating it down the chain is what makes
            # the wave travel, and the wave is where the thrust comes from.
            ph = seg.part.phase_offset * max(seg.depth, 1) if seg is not None else 0.0
            if name.endswith("_r") and seg is not None:
                # A rotor's channel is a speed: steady at the part's throttle,
                # no stroke.  The policy moves it through the offset.
                amps[-1] = 0.0
                offs[-1] = lo[k] + float(np.clip(getattr(seg.part, "rotor_throttle", 0.5),
                                                 0.0, 1.0)) * (hi[k] - lo[k])
            if name.endswith("_f") and seg is not None:
                # A universal joint's feathering channel: centred (the hinge is
                # symmetric about the span) and leading the stroke by the part's
                # ``feather_lead``.
                ph += float(getattr(seg.part, "feather_lead", math.pi / 2))
                offs[-1] = 0.5 * (lo[k] + hi[k])
            phases.append(ph)
        if phases:
            self.cpg.base.phase = np.array(phases, float)
            self.cpg.base.amplitude = np.array(amps, float)
            self.cpg.base.offset = np.array(offs, float)
        self.actuators = getattr(phenotype, "actuators", [])
        self.budget = PowerBudget(battery=phenotype.battery, actuators=self.actuators)
        self._saved = None

    # ---------------------------------------------------------------- lifecycle

    def reset(self, domain: Domain, *, randomise: bool = True) -> None:
        self._mj.mj_resetData(self.model, self.data)
        self.task = None
        self._active_task = None
        x, y, z = self.SPAWN[domain]
        if domain is Domain.AIR and self.air_launch_height is not None:
            z = float(self.air_launch_height)
        if randomise:
            x += float(self.rng.normal(0, 0.4))
            y += float(self.rng.normal(0, 0.4))
            z += float(self.rng.normal(0, 0.2))
        if self.model.nq >= 7:
            self.data.qpos[:3] = (x, y, z)
            self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            if domain is Domain.LAND:
                self.data.qpos[2] = self._clear_of_terrain(x, y, z)
        if domain is Domain.AIR and self.model.nv >= 6:
            # Free joint velocity is [linear, angular] in the world frame, and
            # the spawn attitude is identity, so body +x is world +x.
            self.data.qvel[0] = self.launch_speed
            # Released in trim: at the attitude that balances, not flat.
            a = self.launch_pitch
            self.data.qpos[3:7] = (math.cos(-a / 2), 0.0, math.sin(-a / 2), 0.0)
        self.solver.reset()
        # `JetSet.reset` existed and nothing called it, so a bell carried its
        # volume history -- and, since the pumping load was added, its damping
        # -- from one episode into the next.  `FluidSolver.reset` restores dry
        # inertia here for exactly the same reason.
        self.jets.reset(self.model)
        self.cpg.reset()
        self.budget.reset()
        self._mj.mj_forward(self.model, self.data)

    def scatter(self, rng, *, strength: float = 1.0) -> None:
        """Widen the initial condition, from a caller-supplied generator.

        Every rollout the policy learns from used to begin at the same point:
        one spawn pose per domain with a few centimetres of noise, identity
        attitude, zero velocity (or one fixed launch speed), a full battery and
        a stroke phase of exactly zero -- and the transitions had no noise at
        all, so each kind presented one entry state, repeated for every machine
        of every generation.

        Two of the channels added to the observation were constant across the
        whole training set because of it.  A battery that is always full and a
        stroke that always starts at the same point carry no information, so a
        policy cannot learn to act on either.  And the states a continuous
        mission actually presents -- arriving at the surface carrying the speed
        and attitude the previous leg left behind -- were never in the data.

        The generator is supplied rather than taken from ``self.rng`` so the
        caller decides what is shared: the batched evaluator gives every machine
        in a generation the same draw, which keeps candidates comparable with
        each other, and derives it from the evaluation seed, which keeps a score
        reproducible.
        """
        if self.model.nq >= 7:
            # A small random rotation, as an axis-angle applied to the current
            # attitude.  Bounded: this is meant to require a correction, not to
            # ask every design to recover from a tumble.
            axis = rng.normal(size=3)
            axis /= max(np.linalg.norm(axis), 1e-9)
            ang = float(rng.normal(0.0, 0.18 * strength))
            half = 0.5 * ang
            dq = np.array([math.cos(half), *(math.sin(half) * axis)])
            q = self.data.qpos[3:7]
            self.data.qpos[3:7] = np.array([
                dq[0]*q[0] - dq[1]*q[1] - dq[2]*q[2] - dq[3]*q[3],
                dq[0]*q[1] + dq[1]*q[0] + dq[2]*q[3] - dq[3]*q[2],
                dq[0]*q[2] - dq[1]*q[3] + dq[2]*q[0] + dq[3]*q[1],
                dq[0]*q[3] + dq[1]*q[2] - dq[2]*q[1] + dq[3]*q[0],
            ])
        if self.model.nv >= 6:
            # Scale whatever velocity the pose was set up with, then add to it,
            # so a deliberate entry speed stays an entry speed and stops being
            # the *only* entry speed.
            #
            # **The additive term is proportional to the speed already there.**
            # It used to be a flat `normal(0, 0.35, 3)`, and a land segment
            # begins at rest, so on land that term was the entire initial
            # velocity: every scored land segment started with a shove of about
            # 0.54 m/s and `land_speed` -- which is net displacement over the
            # segment -- measured the machine coasting away from it.
            #
            # Attributed over arch38's eight highest-mission elites, median
            # `land_speed`: 0.470 m/s scattered and actuated, **0.431 m/s
            # scattered with the actuators switched off**, and 0.008 m/s
            # actuated without the scatter.  Turning the machine off changed the
            # number by 8%; turning the shove off changed it by 98%.  88% of
            # them cleared the `moves` rung with no actuation at all.
            #
            # That is the fifth time a score in this project has paid for
            # uncontrolled motion, and the largest: `land_speed` carries two
            # rungs and the land competence, so the whole land ladder above
            # `stirs` was reading the shove.
            #
            # What the scatter is *for* survives intact: the attitude
            # perturbation, the angular kick, the stroke phase and the battery
            # state are all untouched, and an air launch at 13 m/s is still
            # widened by 15% of scale plus 35% of its own speed.  What is gone is
            # translation invented for a body that was standing still.
            # `min(speed, 1.0)`, not `speed`: the additive term keeps its
            # original size wherever there is a velocity to widen -- an air
            # launch at 13 m/s still gets the same 0.35 m/s it always got, and
            # scaling by the full speed would have given it 4.55, which is a
            # different experiment -- and goes to zero for a body at rest, which
            # is the case where it was inventing the motion the score then
            # measured.
            speed = float(np.linalg.norm(self.data.qvel[:3]))
            self.data.qvel[:3] *= 1.0 + float(rng.normal(0.0, 0.15 * strength))
            self.data.qvel[:3] += rng.normal(
                0.0, 0.35 * strength * min(speed, 1.0), 3)
            self.data.qvel[3:6] += rng.normal(0.0, 0.25 * strength, 3)
        b = self.budget.battery
        b.energy_j = float(rng.uniform(0.4, 1.0)) * b.capacity_j
        self.cpg.phase_offset = float(rng.uniform(0.0, 2.0 * math.pi))
        # Put the joints where that phase says they should be.  Offsetting the
        # phase alone leaves every episode starting from the same joint angles
        # and only diverging afterwards, and it is the angles the observation
        # reports.
        if len(self._act_qadr):
            self.data.qpos[self._act_qadr] = self.cpg.command(self.cpg.base, 0.0)
        self._mj.mj_forward(self.model, self.data)

    @property
    def launch_speed(self) -> float:
        """Airspeed the air segment begins at: the speed at which this design's
        measured lift actually balances its weight.

        This was ``sqrt(2W / (rho S CL))`` at CL = 0.9 -- the textbook trim
        speed for a wing that reaches CL = 0.9.  These surfaces do not.  Sitting
        at whatever dihedral and twist the CPPN gave them, driven by a pattern
        generator that is not holding them at an angle of attack, their
        *effective* CL at the natural attitude is around 0.35.  So every design
        was being launched at about 60% of the speed it needed, arriving at
        L/W of 0.4 to 0.5, and falling immediately -- measured across the five
        plans: launched at 9.6-30 m/s against real trim speeds of 15.4-29.3.

        No controller can fix that.  Training one on the ray moved its sink rate
        from 6.98 m/s to 6.62: the body was never given enough airspeed to fly,
        and the search was reading the result as "cannot fly" for every design
        at once.

        So the speed is *measured*, by the same solver that will fly the
        episode: sweep pitch at a series of speeds and find the lowest one where
        the best attitude produces at least the machine's weight.  A design with
        no such speed inside a sane range is launched at the cap and falls,
        which is the correct answer for a design that cannot fly.

        The same principle as replacing the entry-speed proxy with the measured
        slam load: where a quantity can be measured with the model that is about
        to be used, an idealised formula for it is a source of error nobody
        sees.
        """
        return float(self._trim()[0])

    @property
    def launch_pitch(self) -> float:
        """Nose-up attitude the air segment begins at, radians.

        A flight test releases the aircraft *in trim*: at the speed and the
        attitude where it balances.  Launching at the right speed and a level
        attitude is not a fair test of whether a machine can fly, it is a test
        of whether it can recover from being thrown flat -- and every design
        here failed it for the same reason, which is a sign the test rather than
        the designs was wrong.

        Holding trim once released is still entirely the machine's problem, and
        it is the problem worth measuring.
        """
        return float(self._trim()[1])

    @property
    def lift_margin(self) -> float:
        """Best lift this body makes at the top of the speed band, over its weight.

        1.0 is "there is some speed at which this can hold itself up"; below it
        the machine is falling however it is released, and the number says how
        far short.  Over arch36's 179 elites it ran from -1.105 -- pushing
        *down* at 30 m/s -- through a median of 0.333 to 7.367, and the set with
        ``lift_margin >= 1.0`` is exactly the set with a real trim speed, both
        27.4%.

        Free: the trim sweep computes it and caches it per phenotype.
        """
        return float(self._trim()[2])

    def flap_travel(self, phases: int = 16) -> float:
        """Peak-to-peak actuated-joint travel over one cycle, in radians.

        `thrust_margin` is a *ratio* and a still wing produces exactly 0.0 of
        it -- which is the honest answer, and is why the rung `flaps_forward` at
        `thrust_margin >= 0.0` was cleared by flapping that produces nothing.
        arch38 measured the consequence: selection bought the cheapest clearance,
        the share at `>= 0.05` fell 7.7% -> 1.8% and the share at exactly zero
        doubled.

        This separates the two cases the ratio cannot.  A design that does not
        move its joints and a design whose stroke cancels itself both read zero
        thrust; only one of them is flapping.  Costs nothing -- the pattern
        generator is evaluated, the solver is not.
        """
        base = self.cpg.base
        hz = max(float(getattr(base, "frequency", 2.0) or 2.0), 1e-3)
        dt = 1.0 / (hz * max(int(phases), 2))
        try:
            cmds = np.asarray(
                [np.asarray(self.cpg.command(base, k * dt), float)
                 for k in range(max(int(phases), 2))], float)
        except Exception:                                         # noqa: BLE001
            return 0.0
        if cmds.ndim != 2 or cmds.size == 0:
            return 0.0
        return float(np.mean(cmds.max(axis=0) - cmds.min(axis=0)))

    def thrust_margin(self, phases: int = 16) -> float:
        """What the flapping adds forward, over the airframe's own drag.

        ``lift_margin`` asks whether the airframe can hold its weight up.  This
        asks whether the *gait* can hold its speed -- and nothing in this
        project asked it before, in fourteen thousand evaluations across four
        runs.  A glider that cannot make thrust converts height into speed and
        sinks; ``holds_station`` and ``climbs`` are exactly the rungs that
        require thrust, and they had been reached once and never in 14,092
        evaluations.  That is precisely where lift was before arch37: the
        capability had no measurement, so it had no gradient.

            drag   = -Fx with the joints held at the cycle's mean angle
            thrust = <Fx>_cycle - Fx_static
            margin = thrust / drag

        1.0 is "the flapping produces the airframe's entire drag at the speed it
        trims for", i.e. level powered flight rather than a descent.

        Measured over arch37's 182 final elites at their own gaits: **median
        -0.0030, maximum +0.2390, and 3.3% above 0.10.**  Every hand-built seed
        plan is in the same place -- the gannet reads -0.0001 and the teal, a
        5 Hz flapper, reads -0.3796, so its flapping produces 38% more drag than
        holding still.  Correlation with ``lift_margin`` is +0.242, so it is a
        genuinely different question from the one the ladder already asks.

        **And positive thrust is reachable**, which is what makes this a rung
        and not a wall: a random search over 400 gaits took the gannet from
        -0.0001 to **+0.8466** at 10.95 Hz, the teal to +0.4633 at 7.78 Hz.  The
        gaits that make thrust run at 4.5-11 Hz with the joints spread across
        the cycle -- a travelling wave -- while the population sits at 2.2 Hz,
        the beetle seed default, with correlation +0.009 between ``flap_hz`` and
        this quantity because the whole archive is inside the dead zone.

        Quasi-static, like ``lift_at``: joint angles and rates are set around
        one cycle and the solver is evaluated, without stepping the dynamics.
        Same class of estimate the trim sweep already relies on, and about
        0.12 s per phenotype, cached.
        """
        import math

        cached = getattr(self.p, "_measured_thrust", None)
        if cached is not None:
            return None if cached is None else float(cached)
        if hasattr(self.p, "_measured_thrust"):
            return None

        # None, not 0.0, when this cannot be measured.
        #
        # The first version returned 0.0 for "no actuated joints", "no drag to
        # divide by" and "the pose could not be set" as well as for "the flapping
        # produces exactly nothing" -- four states, one number, and the rung
        # `flaps_forward` was set at `>= 0.0`, so the unmeasurable ones cleared
        # it.  Measured over 80 re-scored arch38 elites, 8 read exactly 0.0 and
        # their median `flap_travel` is 0.8998 rad: they are flapping nearly a
        # radian and the number is a guard, not a result.
        #
        # A missing metric stops `rung_reached` where it stands, which is the
        # correct reading of "unknown" and is what `excursion` already does.
        out = None
        try:
            mj, m, d = self._mj, self.model, self.data
            v, pitch, _lift = self._trim()
            jid = np.asarray(m.actuator_trnid[:, 0], int)
            ok = jid >= 0
            if m.nq >= 7 and m.nu > 0 and bool(ok.any()):
                qadr = np.asarray(m.jnt_qposadr, int)[jid[ok]]
                dadr = np.asarray(m.jnt_dofadr, int)[jid[ok]]
                base = self.cpg.base
                hz = max(float(getattr(base, "frequency", 2.0) or 2.0), 1e-3)
                dt = 1.0 / (hz * phases)
                x0, y0, z0 = self.SPAWN[Domain.AIR]

                def fx(angles, rates):
                    mj.mj_resetData(m, d)
                    d.qpos[:3] = (x0, y0, z0)
                    d.qpos[3:7] = (math.cos(-pitch / 2), 0.0,
                                   math.sin(-pitch / 2), 0.0)
                    d.qvel[:] = 0.0
                    d.qvel[0] = v
                    k = min(len(qadr), len(angles))
                    d.qpos[qadr[:k]] = angles[:k]
                    d.qvel[dadr[:k]] = rates[:k]
                    self.solver.reset()
                    mj.mj_forward(m, d)
                    d.xfrc_applied[:] = 0.0
                    with self.solver.steady():
                        self.solver.apply(d, 0.0)
                    return float(d.xfrc_applied[:, 0].sum())

                cmds = [np.asarray(self.cpg.command(base, k * dt), float)
                        for k in range(phases)]
                zero = np.zeros(len(dadr))
                # Held at the cycle's mean angle, not at zero, so the comparison
                # is against the same geometry the flapping passes through.
                static = fx(np.mean(cmds, axis=0), zero)
                moving = [
                    fx(cmds[k],
                       (cmds[(k + 1) % phases] - cmds[(k - 1) % phases]) / (2 * dt))
                    for k in range(phases)
                ]
                self.solver.reset()
                drag = -static
                if drag > 1e-9:
                    out = float((float(np.mean(moving)) - static) / drag)
        except Exception:
            # A body whose joints cannot be posed says nothing about thrust, and
            # "nothing" is not zero: zero is a measurement some designs earn.
            out = None
        out = (float(np.clip(out, -10.0, 10.0))
               if out is not None and np.isfinite(out) else None)
        try:
            self.p._measured_thrust = out
        except Exception:
            pass
        return out

    #: Rig timing for `level_margin`, seconds: settle, then average.  Long
    #: enough for a 1.5 Hz gait's full cycle in the averaging window.
    RIG_SETTLE, RIG_AVERAGE = 0.5, 0.7

    def flies_level(self) -> bool:
        """Whether ``level_margin`` clears ``LEVEL_GATE`` (ROADMAP AD).

        False when the rig could not measure it: absent is not a pass.
        """
        lm = self.level_margin()
        return lm is not None and np.isfinite(lm) and float(lm) >= LEVEL_GATE

    def level_margin(self):
        """Can this machine hold height *and* speed, with its own actuators?

        ``min(<Fz>/W, 1 + <Fx>/W)`` -- 1 or more is level flight -- measured on
        a fixed rig: the airframe held at its own trim speed and attitude (the
        air segment's launch), the joints driven through its own gait by the
        real servos and motors, the fluid with its full history, and the forces
        averaged after a settle.  ROADMAP AG.

        Why not `thrust_margin`: that is quasi-static, prescribed kinematics,
        and blind to lift.  Measured 2026-09-26, the gannet's best feathering
        gait read +2.50 there and pulled the machine *down* with 0.39 of its
        weight; the teal's level-flight gait read 1.28 with its kinematics
        prescribed and 0.42 with its joints free, the motors at their torque
        limit 60% of the time (MATH_AUDIT C-12, A-02).  This measures what the
        actuators deliver.  Rotor thrust is in it, so a multirotor reads its
        hover margin.  Cached per phenotype like `thrust_margin`; ``None`` when
        it cannot be measured.
        """
        if hasattr(self.p, "_measured_level"):
            v = self.p._measured_level
            return None if v is None else float(v)
        out = None
        import copy
        budget = copy.deepcopy(self.budget)      # the rig must not spend the battery
        try:
            mj, m, d = self._mj, self.model, self.data
            if m.nq >= 7:
                v, pitch = self._trim()[:2]
                mj.mj_resetData(m, d)
                x0, y0, z0 = self.SPAWN[Domain.AIR]
                q0 = np.array([x0, y0, z0, math.cos(-pitch / 2), 0.0,
                               math.sin(-pitch / 2), 0.0])
                d.qpos[:7] = q0
                self.solver.reset()
                self.cpg.reset()
                self.cpg.phase_offset = 0.0
                for k in range(len(self.rotors.dof)):
                    d.qvel[self.rotors.dof[k]] = 0.0
                mj.mj_forward(m, d)
                weight = float(self.solver._dry_mass.sum()) * GRAVITY
                n_set = int(self.RIG_SETTLE / self.timestep)
                n_avg = int(self.RIG_AVERAGE / self.timestep)
                F = np.zeros(3)
                t0 = float(d.time)
                for k in range(n_set + n_avg):
                    d.qpos[:7] = q0
                    d.qpos[0] = x0 + v * (float(d.time) - t0)
                    d.qvel[:6] = 0.0
                    d.qvel[0] = v
                    if not self.step(self.cpg.command(self.cpg.base, d.time)):
                        break
                    if k >= n_set:
                        f = d.xfrc_applied[:, :3].sum(0)
                        f[2] -= (m.body_mass.sum() - self.solver._dry_mass.sum()) * GRAVITY
                        F += f / n_avg
                else:
                    if weight > 1e-9 and np.all(np.isfinite(F)):
                        out = float(min(F[2] / weight, 1.0 + F[0] / weight))
                self.solver.reset()
                self.cpg.reset()
        except Exception:
            out = None
        finally:
            self.budget = budget
        out = float(np.clip(out, -10.0, 10.0)) if out is not None else None
        try:
            self.p._measured_level = out
        except Exception:
            pass
        return out

    def _trim(self) -> tuple:
        cached = getattr(self.p, "_measured_trim", None)
        if cached is not None:
            return cached
        lo, hi = LAUNCH_SPEED_RANGE
        out = self._measure_trim_speed(lo, hi)
        if rotor_lift_ratio(self.p) >= 1.0:
            # Held up by its propellers, not its airframe: it trims level, at
            # the bottom of the launch band, whatever the wing sweep said -- a
            # wingless machine's sweep falls back to its steepest attitude,
            # which for a multirotor is a 44 degree dive it did not choose.
            out = (lo, 0.0, out[2] if len(out) > 2 else 0.0)
        try:
            self.p._measured_trim = out
        except Exception:
            pass
        return out

    #: Margin over the stall-limited minimum speed at which the air segment
    #: begins.  Launching *at* the minimum is launching at the top of the lift
    #: curve, which is the one attitude from which any disturbance drops the
    #: machine; 1.2 Vs is the ordinary approach margin and the same reasoning.
    LAUNCH_MARGIN = 1.2

    def _measure_trim_speed(self, lo: float, hi: float, n_pitch: int = 25) -> tuple:
        """Airspeed and attitude the air segment begins at.  Returns
        ``(speed, pitch)``.

        Two steps, because the answer to "what is the slowest this can fly" is
        not the answer to "how should it be released".

        First the stall speed: bisect for the lowest airspeed at which *some*
        attitude produces at least the machine's weight.  The attitude that does
        it is by construction the one at maximum CL, which for this model is
        deep in the post-stall regime -- and releasing a machine there is
        releasing it stalled, at maximum drag, with no lift in reserve.  Every
        body plan was being launched at 34 degrees nose-up, the ceiling of the
        pitch sweep, because that is where a stalled plate makes the most lift.

        So the launch is at ``LAUNCH_MARGIN`` times that speed, at the *lowest*
        pitch that carries the weight there -- the unstalled root of the trim
        equation rather than the stalled one.  That is a flying trim: on the
        front side of the drag curve, with margin above stall in both speed and
        incidence.  Holding it once released remains the machine's problem.

        Bisection on speed with a pitch sweep inside it.  A few hundred solver
        evaluations, a few hundred milliseconds -- once per phenotype, cached,
        against an evaluation that costs seconds.
        """
        mj = self._mj
        # A scratch MjData, never the one being simulated.  This runs the first
        # time anything reads ``launch_speed`` -- which is inside ``reset``, for
        # the air, *after* the spawn pose and its random offset have been
        # written -- and it used to probe on ``self.data`` itself.  So the first
        # air reset of every phenotype object came out at the exact spawn with
        # the last probe's velocities in it, and every later one did not: an
        # air segment's initial condition depended on whether that phenotype's
        # trim had been computed before, i.e. on the history of the process.
        # Found 2026-09-21 when two evaluations of the same body in the same
        # process disagreed in air and nowhere else.
        m, d = self.model, mj.MjData(self.model)
        weight = self.p.mass * GRAVITY
        pitches = np.radians(np.linspace(-4.0, 44.0, n_pitch))

        def lift_at(v: float, a: float) -> float:
            # The pose is set directly rather than through ``reset``, because
            # ``reset`` reads ``launch_speed`` and this is what computes it.
            mj.mj_resetData(m, d)
            if m.nq >= 7:
                x, y, z = self.SPAWN[Domain.AIR]
                d.qpos[:3] = (x, y, z)
                d.qpos[3:7] = (math.cos(-a / 2), 0.0, math.sin(-a / 2), 0.0)
                d.qvel[:] = 0.0
                d.qvel[0] = v
            self.solver.reset()
            mj.mj_forward(m, d)
            d.xfrc_applied[:] = 0.0
            with self.solver.steady():
                self.solver.apply(d, 0.0)
            # Remove the added-mass gravity compensation: it is not lift.
            return float(d.xfrc_applied[:, 2].sum()) - self.solver.diag.added_mass * GRAVITY

        def best_lift(v: float) -> tuple:
            best, best_a = -1e18, pitches[0]
            for a in pitches:
                fz = lift_at(v, a)
                if fz > best:
                    best, best_a = fz, a
            self.solver.reset()
            return best, best_a

        def lowest_pitch(v: float, fallback: float) -> float:
            """First attitude in the sweep that carries the weight at ``v``."""
            for a in pitches:
                if lift_at(v, a) >= weight:
                    self.solver.reset()
                    return float(a)
            self.solver.reset()
            return float(fallback)

        top, top_a = best_lift(hi)
        # How much of its own weight this body can lift at the top of the
        # band, at the best attitude.  Already computed here and thrown
        # away until arch37; it is the only continuous answer this project
        # has to "how close is this design to flying", and 72.6% of
        # arch36's archive needed one -- they could not fly at any speed,
        # scored the two `airborne_fraction` rungs for falling, and had no
        # gradient between "generates no lift" and "flies".
        margin = float(top / weight) if weight > 0 else 0.0
        if top < weight:
            # Cannot fly at any speed we are willing to model.  Released at the
            # *bottom* of the band, at the attitude that does least badly.
            #
            # This used to be the top of the band, and that made the launch
            # inversely earned: the machines that could not fly at all were the
            # ones thrown hardest, at 30 m/s, and the air score then paid them
            # 0.2 for the speed they had been given and let them read as
            # climbing when the beach rose to meet them 75 m downrange.  The
            # comment that used to be here said this was "the correct answer for
            # a design that cannot fly"; the correct answer for a design that
            # cannot fly is to drop it, not to throw it.
            return float(lo), float(top_a), margin
        base, base_a = best_lift(lo)
        if base >= weight:
            v_stall = lo
        else:
            a, b = lo, hi
            for _ in range(10):
                mid = 0.5 * (a + b)
                f, _f_a = best_lift(mid)
                if f >= weight:
                    b = mid
                else:
                    a = mid
            v_stall = b
        v = float(np.clip(v_stall * self.LAUNCH_MARGIN, lo, hi))
        return v, lowest_pitch(v, top_a), margin

    def _clear_of_terrain(self, x: float, y: float, z: float, gap: float = 0.05) -> float:
        """Height at which the machine's lowest geometry sits ``gap`` above ground.

        The spawn height was a constant, so a machine larger than that constant
        started *inside* the beach.  A medusa began its land episode with 99
        contacts and was ejected to 8 m altitude within a second, and whatever
        the land score measured after that, it was not locomotion.  Since the
        search is free to invent machines of any size, the spawn has to be
        derived from the machine rather than assumed.

        Written in terms of ``clearance``, which already knows both the terrain
        and the machine's own extent.  The previous version mixed the requested
        ``z`` with whatever pose ``data`` happened to hold, so calling it with a
        z different from the current one lifted the machine metres into the air.
        """
        if self.model.nq < 7:
            return z
        self.data.qpos[0], self.data.qpos[1], self.data.qpos[2] = x, y, z
        self._mj.mj_forward(self.model, self.data)
        return float(z + (gap - self.clearance()))

    def snapshot(self) -> tuple:
        return (self.data.qpos.copy(), self.data.qvel.copy(), self.data.time)

    def restore(self, snap: tuple) -> None:
        self.data.qpos[:] = snap[0]
        self.data.qvel[:] = snap[1]
        self.data.time = snap[2]
        self.solver.reset()
        self._mj.mj_forward(self.model, self.data)

    # -------------------------------------------------------------------- state

    def body_twist(self) -> np.ndarray:
        """Root body velocity in its own frame: [vx vy vz wx wy wz]."""
        v = np.zeros(6)
        self._mj.mj_objectVelocity(
            self.model, self.data, self._mj.mjtObj.mjOBJ_BODY, self.root_body, v, 1
        )
        return np.concatenate([v[3:], v[:3]])

    def _touching_ground(self) -> bool:
        """True only for contacts against the terrain, not the machine itself.

        ``data.ncon > 0`` counts every contact including body-on-body, and a
        design with several surfaces close together -- a bat with six membranes,
        say -- is in self-contact continuously.  Reading that as "on the ground"
        marks it as landed for the whole episode, which zeroed its flight score
        and inflated its ground-contact score at the same time.
        """
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if b1 == 0 or b2 == 0:  # body 0 is the world
                return True
        return False

    def root_pos(self) -> np.ndarray:
        return self.data.xpos[self.root_body].copy()

    def depth(self) -> float:
        return float(self.medium.depth(self.root_pos()[None, :], self.data.time)[0])

    @property
    def _machine_geoms(self) -> np.ndarray:
        """The machine's *physical* geometry: every geom on its bodies that collides.

        Render-only geometry is excluded -- the tapered, twisted strips
        ``detail=True`` draws beside each wing's collision box, which carry no
        mass and no contact.  They were included, and they reach further than
        the box when a wing flaps, so ``clearance`` -- and through it the air
        segment's sink rate and the controller's height-lost channel -- read a
        different number on a model built for the camera.  Found 2026-09-21:
        the filmed evaluation of a body diverged from the unfilmed one at the
        eleventh control decision, on that channel and no other.
        """
        g = getattr(self, "_mgeoms", None)
        if g is None:
            m = self.model
            g = np.nonzero((m.geom_bodyid != 0)
                           & ((m.geom_contype != 0) | (m.geom_conaffinity != 0)))[0]
            self._mgeoms = g
        return g

    def ground_height(self, x: float, t: float | None = None) -> float:
        """Height of whatever is underneath position ``x``: water, or beach.

        The beach rises inland to more than three metres, so "above the water
        surface" and "off the ground" are different questions on this map, and
        only the second one is about flying.
        """
        return float(self.ground_heights(np.array([float(x)]), t)[0])

    def ground_heights(self, xs: np.ndarray, t: float | None = None) -> np.ndarray:
        """Vectorised ``ground_height``.

        Called once per geom per step, so the scalar version's array allocation
        and per-call wave evaluation showed up as a 75% slowdown of the whole
        search.  One call for the whole machine instead.
        """
        from ..core.mjcf import BEACH_SLOPE, SHORE_X, beach_extent, beach_surface_z

        xs = np.asarray(xs, float)
        probe = np.zeros((xs.size, 3))
        probe[:, 0] = xs
        surface = -np.asarray(
            self.medium.depth(probe, self.data.time if t is None else t), float
        )
        lo, hi = beach_extent()
        beach = (xs - SHORE_X) * BEACH_SLOPE + (beach_surface_z(SHORE_X))
        # The ramp is finite: no phantom ground out at sea or far inland.
        beach = np.where((xs >= lo) & (xs <= hi), beach, -np.inf)
        return np.maximum(surface, beach)

    def clearance(self) -> float:
        """Height of the machine above the ground beneath it, metres.

        This exists because ``depth`` measures against the waterline, and the
        waterline is not the ground.  A machine sitting on the beach ramp
        thirty metres inland is three metres *above* the waterline, so any test
        of the form "is it above the water" calls it airborne while it is
        resting on a hillside.  One design in a live run scored 0.75 for flight
        that way: no lifting surface at all, launched at the 30 m/s cap because
        its wing area rounded to zero, lobbed seventy-five metres downrange, and
        landed on rising terrain -- above the water for 85% of the episode and
        flying for none of it.

        Measured from the machine's *lowest* geometry, not from its root body.
        A machine at rest has its root roughly half a metre up, so a
        root-referenced clearance called the same design airborne while it sat
        on a hillside -- and because its clearance then stayed constant, the
        sink-rate term read it as holding altitude and scored it 0.93 for
        flight.  Referencing the lowest point makes a resting machine read as
        what it is: clearance zero.
        """
        g = self._machine_geoms
        if g.size == 0:
            pos = self.root_pos()
            return float(pos[2] - self.ground_height(float(pos[0])))
        # Lowest corner of each geom's own bounding box, rotated into the world.
        # ``geom_rbound`` is a bounding *sphere*, so for anything elongated it
        # puts the bottom far below the real one -- a machine resting on the
        # beach measured 0.39 m *underground* that way.
        aabb = self.model.geom_aabb.reshape(-1, 6)[g]
        centre_local, half = aabb[:, :3], aabb[:, 3:]
        R = self.data.geom_xmat[g].reshape(-1, 3, 3)
        # Lowest z of a box under rotation: centre minus the sum of the
        # projections of its half-extents onto world -z.
        centre_z = self.data.geom_xpos[g][:, 2] + np.einsum(
            "nij,nj->ni", R, centre_local
        )[:, 2]
        drop = np.einsum("nj,nj->n", np.abs(R[:, 2, :]), half)
        bottom = centre_z - drop
        # Per geom, vectorised.  Measured at 78 us against a 1865 us step, so
        # 4% -- the scalar-loop version it replaced was the expensive one, and
        # the search slowdown I first blamed on this was the new cross-flow and
        # anisotropic added-mass terms in the fluid solver, which are real work.
        ground = self.ground_heights(self.data.geom_xpos[g][:, 0])
        return float(np.min(bottom - ground))

    @property
    def morphology_context(self) -> np.ndarray:
        """Who this machine *is*, as eight bounded numbers.

        Everything else in the observation is a state; none of it says which
        body the state belongs to.  A policy shared across morphologies is then
        a function that must return the same command for two machines in the
        same measured state, however differently they are built -- and measured
        on twelve arch31 elites driven by eight constant twist commands, nine of
        the twelve had a command worth +0.0095 of mission fraction over
        commanding nothing, while *every* command's population mean was below
        commanding nothing.  Per-body optima exist and are large; a shared
        optimum does not.  The gap is exactly this information.

        So the policy is conditioned on the body, which is what the universal-
        controller literature converged on: MetaMorph (Gupta et al. 2022) feeds
        morphology as a context embedding, ModuMorph (Xiong et al. 2023)
        modulates the shared trunk by it.  Eight scalars is the cheap end of
        that idea and it fits an MLP; a limb-token transformer is the expensive
        end and needs a different policy class.

        Constant for the machine's lifetime, so it is computed once.  Every
        channel is scaled to roughly [-1, 1] by a *fixed* transform rather than
        by population statistics, because a normalisation that moves would make
        a stored policy mean something different in a later generation.
        """
        if self._morph_ctx is None:
            p = self.p
            self._morph_ctx = morphology_channels(
                mass=p.mass,
                density_ratio=p.density_ratio,
                wing_area=p.wing_area,
                span=p.max_span,
                aspect_ratio=p.aspect_ratio,
                wing_loading=p.wing_loading,
                n_actuated=len(self.act_names),
                battery_wh=float(getattr(p.genome, "battery_wh", 0.0)),
            )
        return self._morph_ctx

    def observation(self, target: "Domain | None" = None) -> np.ndarray:
        """What the controller senses, plus what it is being asked to do.

        The sensed half is deliberately close to what the real machine could
        measure: a rate gyro and accelerometer (as a body-frame gravity
        direction), a depth sensor, and a wetness estimate.  No global position,
        no ground-truth world velocity -- nothing a real hull could not supply.

        The commanded half is a one-hot of the domain the mission currently
        wants.  Without it the controller has no way to know whether it is
        supposed to be climbing away from the water or diving into it, and the
        best it can do is a single compromise gait.  A mission controller has to
        be told the mission.

        Four senses a real hull also has, added when measurement showed the
        controller could not act on what the mission scores:

        * depth *error* to the mission's own target.  ``tanh(d/5)`` reads 0.96
          at the 10 m target -- the one place the depth channel must have
          gradient is the one place it had none, so holding depth had to be
          inferred from the reward alone.  This channel is zero exactly at the
          target.
        * ground contact (a bump switch), without which walking and the land
          arrival cannot be sensed at all.
        * battery fraction remaining (a coulomb counter), without which economy
          is invisible to the thing being asked to economise.
        * stroke phase: the mean normalised actuated-joint position and its
          rate (joint encoders).  Resonant drive is a phase relationship; a
          controller that cannot sense its own stroke cannot seek resonance.

        And, since 2026-09-21, **what it is being asked to do in the medium**
        (``task_channels``): the domain one-hot said where, and nothing said
        what, so each medium's score had to blend every goal and the policy
        could only learn one compromise.  The depth-error channel reads the
        *commanded* depth while a water task is in force.
        """
        tw = self.body_twist()
        R = self.data.xmat[self.root_body].reshape(3, 3)
        gravity_body = R.T @ np.array([0.0, 0.0, -1.0])
        d = self.depth()
        ph = self._phase_now()
        depth_target = (ph.depth if ph is not None and ph.depth is not None
                        else self.TARGET_DEPTH)
        cmd = np.zeros(3)
        if target is not None:
            cmd[DOMAIN_CYCLE.index(target)] = 1.0
        b = self.budget.battery
        if len(self._act_qadr):
            mid = self.joint_range.mean(axis=1)
            half = np.maximum(
                0.5 * (self.joint_range[:, 1] - self.joint_range[:, 0]), 1e-6)
            qn = (self.data.qpos[self._act_qadr] - mid) / half
            stroke = float(np.clip(np.mean(qn), -1.0, 1.0))
            omega = 2.0 * np.pi * max(self.cpg.base.frequency, 0.1)
            stroke_rate = float(np.clip(
                np.mean(self.data.qvel[self._act_vadr] / half) / omega,
                -3.0, 3.0))
        else:
            stroke = stroke_rate = 0.0
        return np.concatenate(
            [
                np.clip(tw[:3] / 5.0, -3, 3),
                np.clip(tw[3:] / 4.0, -3, 3),
                gravity_body,
                [np.tanh(d / 5.0), self.solver.diag.mean_submerged],
                cmd,
                [
                    np.tanh((d - depth_target) / 3.0),
                    1.0 if self._touching_ground() else 0.0,
                    float(b.energy_j / max(b.capacity_j, 1e-9)),
                    stroke,
                    stroke_rate,
                ],
                self.task_channels(R, ph),
                self.morphology_context,
            ]
        )

    def _phase_now(self):
        """The task phase in force at this instant, or None outside a segment."""
        t = self._active_task
        if t is None:
            return None
        return t.at((float(self.data.time) - self._seg_t0) / self._seg_T)

    def task_channels(self, R=None, ph=None) -> np.ndarray:
        """Six channels saying what the current phase asks for.

        ``[cruise, hold, speed, cos(heading error), sin(heading error),
        height lost]``:

        * a one-hot of the phase's purpose -- move, or hold (``stop`` on land
          is a hold);
        * the commanded speed over the medium's own command scale, so 1 while
          cruising and 0 while holding;
        * the heading error in the horizontal plane, as a unit vector, between
          the command and where the body's +x axis points -- a compass, which a
          real hull has.  Zero while holding, where there is no heading to
          keep;
        * in air, the height lost since the segment began, ``tanh(m / 3)`` --
          a barometer, and the thing the air phases are scored on keeping.

        All six are zero outside a segment, which is what the controller sees
        during a transition and was trained with there.
        """
        out = np.zeros(6)
        if ph is None:
            ph = self._phase_now()
        if ph is None:
            return out
        if R is None:
            R = self.data.xmat[self.root_body].reshape(3, 3)
        if ph.kind == CRUISE:
            out[0] = 1.0
            ref = TASK_SPEED.get(self._active_task.domain) or max(self.launch_speed, 1e-6)
            out[2] = float(np.clip(ph.speed / max(ref, 1e-6), 0.0, 3.0))
            yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
            err = ph.heading - yaw
            out[3], out[4] = math.cos(err), math.sin(err)
        else:
            out[1] = 1.0
        if self._active_task.domain == "air":
            out[5] = math.tanh((self._height_ref - float(self.clearance())) / 3.0)
        return out

    #: 3 linear + 3 angular + 3 gravity + depth + wetness + 3 commanded domain
    #: + depth error + contact + battery + stroke phase and rate, + 6 task
    #: channels (``task_channels``), + 8 morphology.
    OBS_DIM = 25 + MORPHOLOGY_DIM

    #: The scorer's depth target, shared with the observation's error channel
    #: so the sensed error and the scored error cannot drift apart.
    TARGET_DEPTH = 10.0


    # ------------------------------------------------------------------ stepping

    def servo_command(self, target_angles) -> np.ndarray:
        """``ctrl`` for these target angles: the target plus the servo's
        velocity feed-forward, when the CPG that produced them left a rate."""
        tgt = np.asarray(target_angles, float)
        rate = self.cpg.pop_rate()
        if rate is None or len(rate) != len(tgt):
            return tgt
        return tgt + self.servo_lead[: len(tgt)] * rate

    def step(self, target_angles: np.ndarray) -> bool:
        """Advance one timestep.  Returns False when the battery is flat."""
        if len(self.act_names):
            self.data.ctrl[: len(target_angles)] = self.servo_command(target_angles)
        self.data.xfrc_applied[:] = 0.0
        self.solver.apply(self.data, self.data.time)
        if self.jets.n:
            self.jets.apply(self.model, self.data, self.medium,
                            self.data.time, self.timestep)
        if self.rotors.n:
            self.rotors.apply(self.model, self.data, self.medium, self.data.time)
        self._mj.mj_step(self.model, self.data)
        alive = self.budget.step(
            np.abs(self.data.actuator_force), np.abs(self.data.actuator_velocity), self.timestep
        )
        return alive

    def rollout(
        self,
        duration: float,
        *,
        params: CPGParams | None = None,
        policy=None,
        basis: MobilityBasis | None = None,
        domain: Domain = Domain.AIR,
        control_hz: float = 25.0,
        on_step=None,
    ) -> SegmentResult:
        """Run one segment and measure what happened.

        ``on_step(env, i)``, if given, is called after every physics step and
        must not touch the simulation.  It exists so a film can be taken *of
        the evaluation* rather than of a second rollout that is meant to
        resemble it -- see ``viz/film.py``, and ROADMAP item Y for what the
        second kind of film showed.

        One rollout, and nothing else: the passive twin this used to run first
        -- the same segment with every actuator held still, subtracted from the
        scored one -- was removed 2026-09-21.  It doubled the cost of every
        segment (2.084 s against 1.047 s on an 8 s water segment, +38% on a
        whole generation) to defend a score that is now defended by gates
        instead: the water branch's ``active`` requirement, and the state terms
        it stopped adding.  See ``_score_segment``'s water branch for the
        measurement that replaced it.
        """
        p = params or self.cpg.base
        res = SegmentResult(domain=domain, duration=duration)
        self._arm_task(domain, duration)
        n_steps = int(duration / self.timestep)
        control_every = max(1, int(1.0 / (control_hz * self.timestep)))

        start = self.root_pos().copy()
        bad0 = int(self.data.warning[
            self._mj.mjtWarning.mjWARN_BADQACC].number)
        depths, alts, ups, contacts, clearances = [], [], [], [], []
        # The attitude record.  ``spins`` is per step; ``commands`` and
        # ``responses`` are per control decision, and exist so the air score can
        # tell a commanded turn from a tumble -- see ``_turn_authority``.
        spins, commands, responses, vzs, xys = [], [], [], [], []
        # The commanded gait gain at each recorded step; stays empty for a
        # controller without the channel.
        gains, cur_gain = [], None
        peak_slam = 0.0
        cur = p

        for i in range(n_steps):
            if policy is not None and basis is not None and i % control_every == 0:
                cur, coeffs, cur_gain = basis.command_policy(
                    p, policy.act(self.observation(domain)), self.cpg.n, policy)
                commands.append(np.asarray(basis.twist_of(coeffs), float)[3:])
                responses.append(self.body_twist()[3:])
            angles = self.cpg.command(cur, self.data.time)
            if not self.step(angles):
                res.failure = "battery exhausted"
                break

            pos = self.root_pos()
            if not np.all(np.isfinite(pos)) or np.abs(pos).max() > 400.0:
                res.survived = False
                res.failure = "diverged"
                break
            depths.append(self.medium.depth(pos[None, :], self.data.time)[0])
            clearances.append(self.clearance())
            alts.append(pos[2])
            R = self.data.xmat[self.root_body].reshape(3, 3)
            ups.append(float(R[2, 2]))
            contacts.append(1.0 if self._touching_ground() else 0.0)
            spins.append(float(np.linalg.norm(self.body_twist()[3:])))
            vzs.append(float(self.body_twist()[2]))
            xys.append(pos[:2].copy())
            if cur_gain is not None:
                gains.append(cur_gain)
            peak_slam = max(peak_slam, self.solver.diag.slam)
            if on_step is not None:
                on_step(self, i)

        end = self.root_pos().copy()
        res.bad_qacc = int(self.data.warning[
            self._mj.mjtWarning.mjWARN_BADQACC].number) - bad0
        if res.bad_qacc > 0 and res.survived:
            res.survived = False
            res.failure = res.failure or "unstable"
        n = max(len(alts), 1)
        res.distance = float(np.linalg.norm((end - start)[:2]))
        res.mean_speed = res.distance / max(duration, 1e-6)
        res.mean_power = self.budget.mean_power
        res.peak_slam = peak_slam
        res.max_actuator_overload = self.budget.max_overload
        res.attitude_rms = float(np.std(ups)) if ups else 1.0
        res.ground_contact_fraction = float(np.mean(contacts)) if contacts else 0.0
        res.max_depth = float(max(depths)) if depths else 0.0
        res.competence = self._score_segment(
            domain, res, np.array(depths), np.array(alts),
            np.array(ups), np.array(contacts), np.array(clearances),
            spins=np.array(spins), commands=commands, responses=responses,
            vzs=np.array(vzs), xys=np.array(xys), gains=gains,
        )
        self._active_task = None
        return res

    def _arm_task(self, domain: Domain, duration: float) -> None:
        """Put this segment's task in force and start its clock.

        The evaluators set ``self.task`` from a draw the whole generation
        shares; anything that calls ``rollout`` directly gets the default
        schedule.  The air command's speed is the machine's own trim speed,
        filled in here because the shared draw cannot know it.  Both rollout
        paths call this, so they cannot disagree about what was asked.
        """
        t = self.task
        if t is None or t.domain != domain.value:
            t = schedule_for(domain)
        if domain is Domain.AIR:
            v = float(self.launch_speed)
            t = TaskSchedule(t.domain, tuple(_replace(ph, speed=v) for ph in t.phases))
        self._active_task = t
        self._seg_t0 = float(self.data.time)
        self._seg_T = max(float(duration), 1e-6)
        self._height_ref = float(self.clearance())

    def _publish_gains(self, res, gains) -> None:
        """The mean commanded gait gain over each kind of phase.

        ``gain_cruise``, ``gain_hold``, ``gain_stop``: whether the controller
        uses the channel the way the task asks -- down when told to stop or
        hold, up or level when told to go.  Published only when the controller
        has the channel; a run without it has no gain to report, and a 1.0 in
        its place would read as "commanded the design's own gait".
        """
        g = np.asarray(gains if gains is not None else [], float)
        t = self._active_task
        if not len(g) or t is None:
            return
        n = max(int(round(self._seg_T / self.timestep)), 1)
        by_kind: dict = {}
        for (lo, hi), ph in zip(t.bounds(n), t.phases):
            seg = g[lo:min(hi, len(g))]
            if len(seg):
                by_kind.setdefault(ph.kind, []).append(seg)
        for kind, parts in by_kind.items():
            res.measurements[f"gain_{kind}"] = float(np.mean(np.concatenate(parts)))

    def _task_scores(self, domain: Domain, depths, clearances, xys,
                     n_want: int, airborne=None) -> dict:
        """Score each phase of the segment on its own purpose, and combine them.

        Every phase is measured over its **late half** -- the first half is
        the machine's to turn, descend or brake in -- and over the samples the
        segment was *asked* for, so a phase the episode never reached scores
        nothing and a truncated one scores only the part it served.

        * **cruise**: the mean horizontal velocity over the late half against
          the commanded one, ``1 - |v - v_cmd| / |v_cmd|``.  A machine standing
          still scores 0, one drifting across the heading scores 0, and one
          going twice as fast as asked scores 0.  Multiplied by holding the
          vertical: in water the commanded depth, against how far the command
          asked it to descend; in air height above the surface, graded from
          level (1) through a glide to ballistic (0) as ``flight`` and ``glide``
          have been since arch37.
        * **hold** (water): at the commanded depth *and* not moving, both
          required.  Distance from the commanded depth is measured against how
          far the command asked the machine to descend, so a hull that sits
          where it was released scores 0; movement over the window is measured
          against the distance the cruise command would have covered, so one
          sinking through the target scores 0.
        * **stop** (land): the same, horizontally, against the distance the
          walk command would have covered -- walking on at the commanded speed
          scores 0.

        Combined: water and air are the mean of their two phases, because
        neither phase is free for a machine that does nothing.  On land a
        machine that does nothing stops perfectly, so stopping *qualifies* the
        walk instead of adding to it -- ``walk * (0.5 + 0.5 * stop)`` -- and a
        rock scores zero.
        """
        t = self._active_task
        if t is None or t.domain != domain.value:
            t = schedule_for(domain, trim_speed=float(self.launch_speed))
        xy = np.asarray(xys if xys is not None else np.zeros((0, 2)), float).reshape(-1, 2)
        dep = np.asarray(depths, float)
        clr = (np.asarray(clearances, float) if clearances is not None else -dep)
        n_got = min(len(xy), len(dep), len(clr))
        dt = float(self.timestep)
        # Where the machine was released, for the water phases' scale: an
        # error is measured against how far the command asked it to move, so a
        # hull that sits where it was put scores exactly zero and one that gets
        # halfway scores a half.  A flat 1 m band was measured first and gave
        # 97% of 64 designs exactly zero in water -- a wall, with no gradient
        # for selection or the policy to climb.
        d_start = float(dep[0]) if len(dep) else 0.0
        phases = []
        for (lo, hi), ph in zip(t.bounds(n_want), t.phases):
            mid = lo + (hi - lo) // 2
            top = min(hi, n_got)
            rec = {"kind": ph.kind, "score": 0.0, "measured": False}
            if top - mid >= 2:
                span = (top - 1 - mid) * dt
                served = (top - mid) / max(hi - mid, 1)
                v = (xy[top - 1] - xy[mid]) / max(span, 1e-6)
                rec.update(measured=True, v=v, served=served)
                if ph.kind == CRUISE:
                    h = np.array([math.cos(ph.heading), math.sin(ph.heading)])
                    want = ph.speed * h
                    # Progress along the commanded heading, the part of the
                    # score that may be passive (a glide, a hull that carries
                    # itself -- the user's rule, 2026-09-21): the speed along
                    # the heading over the command, capped at 1, times a soft
                    # penalty on the speed across it.  Standing still and
                    # drifting across the heading both score 0; going faster
                    # than asked is still forward.
                    along = float(v @ h)
                    across = float(abs(v[0] * h[1] - v[1] * h[0]))
                    rec["progress"] = float(np.clip(along / max(ph.speed, 1e-6), 0.0, 1.0)
                                            * math.exp(-(across / max(ph.speed, 1e-6)) ** 2))
                    track = float(np.clip(
                        1.0 - np.linalg.norm(v - want) / max(ph.speed, 1e-6), 0.0, 1.0))
                    if domain is Domain.WATER:
                        err = float(np.mean(np.abs(dep[mid:top] - ph.depth)))
                        scale = max(abs(float(ph.depth) - d_start), self.DEPTH_BAND)
                        vert = float(np.clip(1.0 - err / scale, 0.0, 1.0))
                    elif domain is Domain.AIR:
                        # Graded over the whole range between holding height and
                        # falling, as the air score has been since arch37:
                        # ``flight`` alone is zero for everything not very nearly
                        # level, and a 4 s phase of it measured zero for 131 of
                        # 132 open-loop segments.
                        #
                        # Holding height is only holding height in the air, so
                        # it is measured from the start of the phase for as long
                        # as the machine *stays* in the air, and weighted by
                        # the share of the phase that is: a body that fell into
                        # the sea floats at a sink rate of zero, and scored full
                        # marks for it (the seed flappers, 0.141, against the
                        # gannet's 0.187).  From the start, not over the late
                        # half the other terms use -- that half is time to turn
                        # or descend, and height needs none: the machine is
                        # launched level at its own trim speed.  Measured on
                        # arch42's top 40 at gen 102: 86% of air segments are
                        # in the sea by 2.65 s, so a late-half window read zero
                        # for every driven design (median 0.000, mean 0.004)
                        # and air had no gradient at all.  A rock launched from
                        # 30 m falls at ~12 m/s averaged over its flight and
                        # still scores 0; a glide scores, as it may.
                        ab = (np.ones(top - lo, bool) if airborne is None
                              else np.asarray(airborne, bool)[lo:top])
                        stay = int(np.argmin(ab)) if not ab.all() else len(ab)
                        if stay >= 2:
                            sink = float((clr[lo] - clr[lo + stay - 1])
                                         / max((stay - 1) * dt, 1e-6))
                            # And the worst second inside it (ROADMAP AE): the
                            # endpoints read zero for a body that drops and
                            # comes back, and arch42's best air scores after
                            # gen 120 were tumblers doing exactly that (wobble
                            # ratio 27-72).  `sink_rate` against
                            # `station_keeping` again, arch37.
                            win = int(round(HEIGHT_WINDOW / dt))
                            if stay > win:
                                seg_clr = np.asarray(clr[lo:lo + stay], float)
                                drops = (seg_clr[:-win] - seg_clr[win:]) / (win * dt)
                                sink = max(sink, float(drops.max()))
                            flight = float(np.clip(1.0 - sink / 1.5, 0.0, 1.0))
                            # A trajectory that holds height is flight only
                            # when the actuators could have held it (AD); a
                            # glider launched level at trim sinks slowly for
                            # a few seconds on its airframe alone.
                            if not self.flies_level():
                                flight = 0.0
                            glide = float(np.clip(1.0 - sink / SINK_BALLISTIC, 0.0, 1.0))
                            vert = ((0.55 * flight + 0.25 * glide) / 0.80
                                    * stay / max(hi - lo, 1))
                        else:
                            vert = 0.0
                        late = (np.asarray(airborne, bool)[mid:top] if airborne is not None
                                else np.ones(top - mid, bool))
                        rec["airborne"] = float(np.mean(late)) if len(late) else 0.0
                    else:
                        vert = 1.0
                    rec.update(tracking=track, vertical=vert, score=track * vert * served)
                elif ph.kind == HOLD:
                    # Two things, both required: be at the commanded depth,
                    # and do not move.  Depth alone is satisfied by a hull
                    # sinking *through* the target during the window -- the
                    # still eel, released at 4 m and sinking 0.55 m/s, crosses
                    # 6.5 m right on time -- so the motion over the window is
                    # scored separately, against the distance
                    # ``HOLD_DRIFT_SPEED`` would cover in it: holding still
                    # scores 1, drifting at that speed scores 0.  And the net
                    # vertical rate on its own, against ``HOLD_SINK_SPEED``:
                    # that band is the cruise speed, and arch42's held-still
                    # eel passed through its target at 0.094 m/s and scored
                    # 0.63 for holding it.
                    e_d = float(np.mean(np.abs(dep[mid:top] - float(ph.depth))))
                    p3 = np.column_stack([xy[mid:top], dep[mid:top]])
                    e_m = float(np.mean(np.linalg.norm(p3 - p3[0], axis=1)))
                    scale = max(abs(float(ph.depth) - d_start), self.DEPTH_BAND)
                    band_m = max(HOLD_DRIFT_SPEED * span / 2.0, 1e-3)
                    at = float(np.clip(1.0 - e_d / scale, 0.0, 1.0))
                    still = float(np.clip(1.0 - e_m / band_m, 0.0, 1.0))
                    sink = float(abs(dep[top - 1] - dep[mid]) / max(span, 1e-6))
                    level = float(np.clip(1.0 - sink / HOLD_SINK_SPEED, 0.0, 1.0))
                    rec.update(error=e_d + e_m, depth_error=e_d, drift=e_m, sink=sink,
                               score=at * still * level * served)
                else:
                    drift = float(np.mean(np.linalg.norm(xy[mid:top] - xy[mid], axis=1)))
                    band = max(TASK_SPEED["land"] * span / 2.0, 1e-3)
                    rec.update(drift=drift, score=float(
                        np.clip(1.0 - drift / band, 0.0, 1.0)) * served)
            phases.append(rec)

        # The combination, as measured on arch42's elites against the same
        # bodies held still (`runs/_logs/probe_shapes_score.py`, 2026-09-22).
        # The linear, multiplied form before this left the whole population's
        # median at exactly zero in every medium for 102 generations, so
        # selection was choosing among ties.  The rule it follows is the
        # user's: **progress may be passive, what must be chosen may not.**
        #
        # * progress along a commanded heading -- a glide, a hull carrying
        #   itself -- scores however it is made;
        # * turning on command is scored as a *response*: the change in
        #   velocity across the old heading toward the new one, which a body
        #   gliding on is not doing;
        # * holding is scored as being at the commanded depth *and* still, and
        #   stopping as a factor on the walk -- neither can pay a still body
        #   more than its own passive progress;
        # * sinking never scores by itself: the hold is at a commanded depth
        #   below the release, and it must also be still.
        #
        # Measured means, driven / held still: land progress 0.240 / 0.009;
        # water hold 0.056 / 0.048; air turn 0.061 / 0.040.  Water progress
        # 0.313 / 0.220 and air height 0.336 / 0.352 are mostly passive, which
        # the rule allows.  A stop *response* term on land was tried and
        # dropped: a still eel that slid to rest scored 0.097 on it.
        m: dict = {}
        def _served(*rs):
            return min((r.get("served", 0.0) for r in rs), default=0.0)

        if domain is Domain.LAND:
            walk = next(r for r in phases if r["kind"] == CRUISE)
            stop = next(r for r in phases if r["kind"] == STOP)
            wph = next(ph for ph in t.phases if ph.kind == CRUISE)
            # Progress, qualified by stopping when told: a *product*, not the
            # sum with a stop "response" that was tried first.  The response
            # was measured as the walk-phase velocity minus the stop-phase
            # velocity along the heading, and a body held still that slid and
            # came to rest on its own read as having stopped on command -- the
            # eel, 0.097 averaged over eight draws.  As a product, a still body
            # can never score more than its own passive progress, which the
            # user's rule allows; a rock scores zero.
            progress = walk.get("progress", 0.0) * walk.get("served", 0.0)
            task = progress * (0.5 + 0.5 * stop["score"])
            if walk["measured"]:
                m["walk_tracking"] = walk["tracking"]
                m["walk_progress"] = walk["progress"]
            if stop["measured"]:
                m["stop_score"] = stop["score"]
                m["stop_drift"] = stop["drift"]
            m["cmd_heading"] = wph.heading
            m["task_walk_first"] = float(t.phases[0].kind == CRUISE)
        elif domain is Domain.WATER:
            cr = next(r for r in phases if r["kind"] == CRUISE)
            ho = next(r for r in phases if r["kind"] == HOLD)
            task = (0.5 * cr.get("progress", 0.0) * cr.get("served", 0.0)
                    + 0.5 * ho["score"])
            if cr["measured"]:
                m["cruise_tracking"] = cr["tracking"]
                m["cruise_progress"] = cr["progress"]
                m["cruise_depth_hold"] = cr["vertical"]
                m["cruise_score"] = cr["score"]
            if ho["measured"]:
                # Metres from where it was told to be, *plus* metres it
                # moved: the ladder's `holds_depth` reads this at 1 m.
                m["hold_error"] = ho["error"]
                m["hold_depth_error"] = ho["depth_error"]
                m["hold_drift"] = ho["drift"]
                m["hold_sink_rate"] = ho["sink"]
                m["hold_score"] = ho["score"]
            m["cmd_depth"] = float(t.phases[0].depth)
            m["cmd_heading"] = next(ph.heading for ph in t.phases if ph.kind == CRUISE)
            m["task_hold_first"] = float(t.phases[0].kind == HOLD)
        else:
            a, b = phases
            pa, pb = t.phases
            turn = 0.0
            if a["measured"] and b["measured"]:
                # Toward the new heading *across* the old one only: a body
                # slowing down on its launch heading has a velocity change
                # with a component along any turn that is more than 90
                # degrees from straight ahead, and was scored as turning.
                ha = np.array([math.cos(pa.heading), math.sin(pa.heading)])
                hb = np.array([math.cos(pb.heading), math.sin(pb.heading)])
                n = np.array([-ha[1], ha[0]])
                if float(hb @ n) < 0.0:
                    n = -n
                # Against what a perfect turn to the commanded heading would add
                # across the old one, ``speed * sin(turn)``, so turning exactly
                # as told scores 1 at any commanded angle.
                want = max(pb.speed * float(hb @ n), 1e-6)
                # And the second phase's own velocity must point to the new
                # side, in the air.  arch42's held-still gannets fell 30 m,
                # slid sideways away from the commanded side on the water and
                # stopped: stopping is a velocity change *toward* it, and
                # scored 0.28 and 0.34 as turning.
                gain = min(float(b["v"] @ n), float((b["v"] - a["v"]) @ n))
                turn = (float(np.clip(gain / want, 0.0, 1.0)) * _served(a, b)
                        * min(a.get("airborne", 1.0), b.get("airborne", 1.0)))
                m["turn_response"] = turn
            height = float(np.mean([r.get("vertical", 0.0) * r.get("served", 0.0) for r in (a, b)]))
            task = 0.5 * turn + 0.5 * height
            if a["measured"]:
                m["cruise_tracking"] = a["tracking"]
                m["cruise_progress"] = a["progress"]
                m["cruise_height_hold"] = a["vertical"]
            if b["measured"]:
                m["turn_tracking"] = b["tracking"]
                m["turn_height_hold"] = b["vertical"]
            m["height_hold"] = height
            m["cmd_turn"] = float(pb.heading)
            m["cmd_speed"] = float(pa.speed)
        m["task_score"] = float(task)
        return {"task": float(task), "phases": phases, "measurements": m}

    def _score_segment(self, domain, res, depths, alts, ups, contacts,
                       clearances=None, *, spins=None, commands=None,
                       responses=None, vzs=None, xys=None, gains=None) -> float:
        """Domain competence in [0, 1].

        ``spins`` is the body angular rate magnitude at each recorded step, and
        ``commands``/``responses`` are the commanded and achieved angular rates
        at each control decision.  All three are optional and all three are only
        read by the air branch, where the difference between a tumble and a
        commanded turn is the difference between a score and an exploit.  A
        caller that does not supply them gets an air score with no manoeuvring
        credit, which is the right answer for a rollout that had no controller.

        Each domain is scored on what actually matters there, not on a generic
        "went far" reward -- flying is about not falling, diving is about
        holding depth, walking is about making progress while touching ground.

        Every time-fraction below is divided by the length the segment was
        *asked* for, never by the number of samples that happened to be
        recorded.  That distinction is the whole defence against the following
        exploit, which the search found within 800 generations:

            Build a machine with almost no wing and a battery it drains in a
            fraction of a second.  The episode terminates on the first step.
            The two or three samples that were recorded are all at the launch
            altitude, so the machine was "airborne" 100% of the time, its
            measured sink rate over that window is nil, and it kept its launch
            speed.  Air competence: 1.00.

        Twenty-one designs with wing loadings up to 480,000 N/m^2 -- objects
        with no lifting surface at all -- were scoring above 0.9 this way, and
        twenty of them had an energy margin of -0.98 or worse.  Dividing by the
        intended length makes a truncated episode score the truncation.
        """
        if not res.survived or len(alts) == 0:
            # The two airframe properties even here.  They are facts about the
            # body -- a static lift sweep and a cycle-averaged thrust estimate,
            # both computed from a fresh pose -- so they are the same numbers
            # whatever the episode did, and the air ladder's bottom rungs read
            # them.  Withholding them makes a design whose rollout went unstable
            # indistinguishable from one with no wings: 243 of arch37's 14,092
            # air segments left this path having published *nothing*, so they
            # scored rung 0 for a `bad_qacc` rather than for an airframe.
            #
            # Same reasoning as the two exits below, and the same defect that
            # voided arch38's first launch, one level lower down.
            if domain is Domain.AIR:
                res.measurements["lift_margin"] = float(self.lift_margin)
                res.measurements["flap_travel"] = float(self.flap_travel())
                _tm = self.thrust_margin()
                if _tm is not None:
                    res.measurements["thrust_margin"] = float(_tm)
                _lm = self.level_margin()
                if _lm is not None:
                    res.measurements["level_margin"] = float(_lm)
            return 0.0
        upright = float(np.clip(np.mean(ups), 0.0, 1.0))
        # Samples the segment should have produced had it run to term.
        if clearances is None:
            clearances = -np.asarray(depths, float)
        n_want = max(int(round(res.duration / self.timestep)), 1)
        n_got = len(alts)
        # Anything that ended early is measured against what it was asked to do.
        served = min(n_got / n_want, 1.0)
        # How the controller's commands moved, published on every segment that
        # had a controller -- a diagnostic no rung reads, so a run can measure
        # its distribution before anyone sets the penalty's weight from it.
        stats = command_statistics(commands)
        self._publish_gains(res, gains)
        chatter = 1.0
        if stats is not None:
            res.measurements["command_rate"], res.measurements["command_reversal"] = stats
            if self.action_rate_penalty > 0.0:
                chatter = 1.0 / (1.0 + self.action_rate_penalty * stats[0])

        if domain is Domain.AIR:
            # Flight, not slow descent.
            #
            # The first version scored ``1 - drop/6`` over a seven-second window,
            # which gives partial credit to anything that falls slowly.  The
            # archive's best "flyers" were then infeasible random genomes with a
            # wing loading of 1481 N/m^2 -- objects that cannot fly by any
            # measure, scoring 0.52 for descending in a controlled fashion.
            #
            # Sustained flight means the sink rate goes to zero.  So the metric
            # is built on the descent *rate over the second half* of the episode,
            # by which time a real flyer has settled: zero or negative sink is
            # full marks, and it falls to nothing by 1.5 m/s, which is a glide
            # rather than flight.  Gliding still scores something -- it is a real
            # capability -- but it can no longer be mistaken for flying.
            # Airborne means clear of the surface and touching nothing.  A
            # machine bobbing at the waterline has its hull centre a few
            # centimetres above the water and no ground contact, so any test
            # looser than this scores floating as flying -- which is how a
            # 937 N/m^2 medusa came to outscore a 54 N/m^2 ray at flight.
            # Clear of the *ground*, not merely of the waterline -- see
            # ``clearance``.  The old test read ``depths < -0.3``, which over a
            # beach that rises to three metres calls a landed machine airborne.
            airborne = (np.asarray(clearances) > 0.3) & (np.asarray(contacts) < 0.5)
            frac = float(np.sum(airborne) / n_want)
            # Absolute, like `MEASURABLE_AIR_SECONDS` and for the same reason:
            # this asks "did it ever get clear", which is a fact about the
            # machine, not a fact about how long we watched.  As a fraction it
            # was 0.4 s at an 8 s segment and 1.2 s at a 24 s one, so the same
            # half-second hop was classified two different ways by a flag.
            #
            # `airborne_fraction` itself stays a fraction, and the two rungs
            # that read it stay fractions -- *those* are the survival question,
            # and survival is supposed to get harder as the segment grows.  What
            # must not move with the window is whether a measurement is taken.
            if np.sum(airborne) * self.timestep < 0.4:
                # Even here.  `lift_margin` is what the airframe can lift, not
                # what this episode did with it, and the ladder's bottom four
                # rungs read it -- so leaving it out of any air path makes a
                # design that never got clear indistinguishable from one with no
                # wings.  Three paths leave this branch of `_score_segment` and
                # all three publish it; the test below enforces that, because
                # the first attempt at this changed one of the three and a grep
                # for another key said that was all of them.
                res.measurements["lift_margin"] = float(self.lift_margin)
                res.measurements["flap_travel"] = float(self.flap_travel())
                _tm = self.thrust_margin()
                if _tm is not None:
                    res.measurements["thrust_margin"] = float(_tm)
                _lm = self.level_margin()
                if _lm is not None:
                    res.measurements["level_margin"] = float(_lm)
                return 0.0  # never left the surface: no flight to score
            idx = np.flatnonzero(airborne)

            # --- the gates ---------------------------------------------
            #
            # Two of them are Tier-0 facts about the machine (``airworthiness``)
            # and the third is measured here: a body turning faster than a
            # revolution a second has no attitude that persists long enough for
            # anything below to be a measurement of flight.
            spin = 0.0
            if spins is not None and len(spins) == len(clearances) and len(idx):
                spin = float(np.mean(np.asarray(spins, float)[idx]))
            gates = list(self.air_gates)
            if spin > MAX_SPIN_RATE:
                gates.append(f"spinning at {spin:.1f} rad/s")
            credit = GATED_AIR_CREDIT if gates else 1.0

            # Sink is a rate, and a rate needs a baseline long enough to be one.
            # Over a tenth of a second every launched object has a sink rate of
            # nearly zero, including a brick.
            #
            # The partial credit for a brief hop was 0.25, which is what a
            # genuine glide at 1.0 m/s sink now scores.  A hop and a glide are
            # not the same achievement, so it is 0.10.
            if (np.sum(airborne) * self.timestep
                    < self.MEASURABLE_AIR_SECONDS):
                # The diagnostics, but deliberately not the ladder metrics: a
                # hop that was too short to measure a sink rate over should not
                # be able to climb a rung on the strength of having happened.
                # This is the path arch33's wingless "climbers" now take, and
                # what they left behind was a blank record.
                res.measurements.update({
                    "spin_rate": spin,
                    "measured_sink_rate": 9.9,
                    "air_gates": float(len(gates)),
                    "airborne_seconds": float(np.sum(airborne) * self.timestep),
                    # `lift_margin` crosses the "diagnostics but not the ladder
                    # metrics" line above, and has to: it is a property of the
                    # airframe, not of the episode.  Withholding it here says
                    # "this machine landed too quickly for us to know whether it
                    # has wings".  The rule this branch enforces is that a hop
                    # must not climb a rung *on the strength of having
                    # happened*, and a static lift measurement cannot -- it is
                    # the same number however the segment went.
                    #
                    # It matters because this is the branch every falling design
                    # takes.  The air spawn is 30 m, free fall from there is
                    # 2.5 s, and 2.5 s is 31% of an 8 s segment -- under the 35%
                    # bar above.  50 of arch37's first 180 evaluations came
                    # through here and scored air rung 0 whatever their airframe
                    # could do, which voided that launch.
                    "lift_margin": float(self.lift_margin),
                    "flap_travel": float(self.flap_travel()),
                })
                _tm = self.thrust_margin()
                if _tm is not None:
                    res.measurements["thrust_margin"] = float(_tm)
                _lm = self.level_margin()
                if _lm is not None:
                    res.measurements["level_margin"] = float(_lm)
                # And no score.  This paid ``credit * frac * 0.10`` until
                # 2026-09-23, and every body that takes this branch reached the
                # sea from the 30 m launch in under 2.8 s -- faster than free
                # fall's 2.47 s allows anything that flew.  It paid 0.033 for
                # that, while a body that stayed up longer and came down faster
                # than 6 m/s took the long branch and scored 0: across arch42,
                # 1,069 fallers had median 0.030-0.033 and 1,786 long exits
                # median 0.000, and air competence correlated -0.44 with time
                # aloft.  The score paid falling over staying up, and the famine
                # regime -- air starved in 113 of 178 generations -- selected
                # on it.  A fall is not flight, so it scores what flight it
                # showed: none.
                res.parts = {"gate": float(credit * frac), "control": 0.0}
                return 0.0

            # Sink rate measured only over the airborne stretch, and only its
            # later half, by which time a real flyer has settled.  Zero sink is
            # full marks; 1.5 m/s is a glide, which is a real capability but is
            # not flight and no longer scores as if it were.
            # Sink is the rate of loss of height *above the ground*, not of
            # world z.  The beach rises inland at 0.12, so a machine skimming up
            # it gains z at 3.6 m/s while flying at 30 -- and the design that
            # exposed this had no lifting surface at all.  It was launched at the
            # speed cap because its wing area rounded to zero, lobbed seventy
            # five metres downrange, and read as *climbing* over the second half
            # of its arc because the hill came up to meet it.  Scored 0.75 for
            # flight.  It is now released at the bottom of the launch band, and
            # gated above in any case.
            # The late half of the airborne stretch, **capped at a fixed
            # window**, so that what this measures does not change when the
            # segment length does.
            #
            # `station_keeping` is the fraction of this window spent within a
            # band of where the machine settled, so for a steady descent at v it
            # is `band / (v * T)` -- the rung threshold 0.6 therefore encodes a
            # sink rate that depends on T.  At an 8 s segment it asks for
            # <= 0.21 m/s; at 24 s it would silently ask for <= 0.069 m/s.  Same
            # rung name, two different physical requirements, which breaks this
            # ladder's own contract that a rung means the same thing on day one
            # and day five -- and would set a bar no design in arch37 came within
            # a factor of three of, which is the "moves at 0.1 m/s left 61.6% of
            # the population with nowhere to stand" mistake again.
            #
            # Capping the measurement window separates the two questions a long
            # segment asks.  *Survive the segment* is read by
            # `airborne_fraction`, and gets harder as the segment grows, which
            # is the point of growing it.  *Hold a height* is read here over a
            # constant window, and means the same thing at any segment length.
            #
            # At `segment_seconds` 8 this is exactly the old definition -- the
            # airborne stretch cannot exceed 8 s, so its late half cannot exceed
            # the 4 s cap -- so every arch37 number remains comparable.
            half = max(len(idx) // 2, 1)
            cap = max(int(self.STATION_WINDOW / self.timestep), 1)
            late = idx[-min(half, cap):]
            if len(late) > 1:
                span_s = (late[-1] - late[0]) * self.timestep
                sink = float(
                    (clearances[late[0]] - clearances[late[-1]]) / max(span_s, 1e-6)
                )
            else:
                sink = 9.9
            flight = float(np.clip(1.0 - sink / 1.5, 0.0, 1.0))
            # Gliding, graded over the whole range between flight and falling.
            # The 0.55/1.5 term above is zero for everything that is not very
            # nearly holding height -- no seed plan reaches it -- so without a
            # graded companion the air score is a step function nothing in the
            # population can climb, which is the failure the flat 0.25 was
            # papering over.  This one has to be earned.
            glide = float(np.clip(1.0 - sink / SINK_BALLISTIC, 0.0, 1.0))

            # Station keeping: holding *a* height, not merely losing height
            # slowly.  Nothing used to measure this, so the ``holds_height``
            # rung was satisfied by a fast shallow glide -- a machine trading
            # altitude for ground speed at 0.4 m/s clears a 0.5 m/s bar for the
            # whole segment while never holding anything.
            #
            # The band is a quarter of the machine's own span, floored at half
            # a metre: staying that close to the height it settled at is a
            # station-keeping tolerance, and expressing it in the machine's own
            # geometry makes it the same test for a 0.4 m machine and a 3 m one.
            #
            # Over an 8 s segment this separates a 0.4 m/s creep from level
            # flight and not much finer than that -- 0.4 m/s is 1.6 m of drift
            # over the window, which is the scale of the machine.  The
            # measurement earns its keep on the 60 s leg (``evaluate_tier1_5``),
            # where the same creep is 24 m.
            band = max(0.25 * float(getattr(self.p, "max_span", 2.0)), 0.5)
            if len(late) > 1:
                ref = float(clearances[late[0]])
                dev = np.abs(clearances[late] - ref)
                station = float(np.mean(dev < band))
                # The shape between the endpoints, which neither of the other
                # two measurements can see.  ``sink_rate`` is the difference
                # between ``clearances[late[0]]`` and ``clearances[late[-1]]``,
                # so it reads zero for a machine that drops and comes back;
                # ``station_keeping`` sees that something is wrong but not what.
                #
                # arch37 measured the gap and it is a factor of ten: over the
                # 173 segments holding height on the ladder's own measure
                # (``lift_margin`` >= 1, |sink| < 0.5 m/s), a straight-line
                # descent at the same sink rate would have scored 0.704 median
                # station with 59.5% clearing ``holds_station``; the measured
                # values are 0.062 and 0.58%.  A tenth of what the geometry
                # allows, so the paths are not straight -- and every statement
                # about *how* they are not straight was an inference, because
                # nothing published the excursion.
                #
                # Now something does.  ``excursion_ratio`` is the same quantity
                # in units of the machine's own station band, so a 0.5 m
                # machine and a 3 m one are comparable.
                excursion = float(dev.max())
            else:
                station = 0.0
                # Not zero.  A window too short to measure has no excursion to
                # report, and 0.0 would read as "perfectly steady" -- a
                # degenerate case scoring like the best possible one, which is
                # the shape of every scoring bug in this file's history.  The
                # key is simply absent, which is safe precisely because no rung
                # reads it.
                excursion = None

            # Oscillation, separated from trend, over the **whole** airborne
            # stretch rather than a window.
            #
            # `excursion` above is the deviation from where the machine settled,
            # so for a steady descent it is just the drift and it grows with the
            # window -- useful, and not window-invariant.  Subtracting the fitted
            # line leaves the part that is not a descent at all: the amplitude of
            # whatever the machine is doing around its own trajectory.  That
            # number does not grow with the window for a periodic motion, so it
            # means the same thing at 8 s and at 24 s, and it is the quantity the
            # arch37 inference was actually about -- `sink_rate` already carries
            # the trend, and 173 segments held height on it while scoring a tenth
            # of the station a straight descent would give.
            if len(idx) > 2:
                cc = np.asarray(clearances)[idx].astype(float)
                tt = np.arange(len(idx), dtype=float) * self.timestep
                slope, icept = np.polyfit(tt, cc, 1)
                wobble = float(np.max(np.abs(cc - (slope * tt + icept))))
            else:
                wobble = None

            # Manoeuvring means the attitude change was *asked for*.
            authority, turn = _turn_authority(commands, responses)

            res.altitude_held = flight
            res.measurements.update({
                "airborne_fraction": frac,
                # Whether this body can hold itself up at all, and by how
                # much it misses when it cannot.  A property of the
                # airframe rather than of the episode, which is the point:
                # the four rungs it carries sit *below* the
                # `airborne_fraction` ones, so a machine that makes no
                # lift cannot reach them by being dropped from 30 m.  That
                # is how 53% of arch36 was paid, and `rung_reached` stops
                # at the first unmet rung, so the ordering is the gate.
                "lift_margin": float(self.lift_margin),
                "flap_travel": float(self.flap_travel()),
                # The ladder reads these, so a gated design must not be able to
                # climb it on numbers that do not mean what they say.  The
                # measurements themselves are kept under their own names so the
                # record stays honest and the gate stays re-derivable.
                "sink_rate": 9.9 if gates else sink,
                "glide": 0.0 if gates else glide,
                "station_keeping": 0.0 if gates else station,
                "measured_sink_rate": sink,
                "measured_station_keeping": station,
                # Diagnostics, ungated and with **no rung reading them**, which
                # is the property that makes them safe to add mid-programme:
                # ``rung_reached`` stops at the first rung whose metric is
                # missing, and arch37's first launch was voided by exactly that
                # -- a metric published on one of the air branch's three exits
                # and read by a rung.  Nothing reads these, so a branch that
                # never computes them costs nothing.
                #
                # A threshold is deliberately not set here.  The distribution
                # is unknown, and setting a bar from what the capability ought
                # to look like rather than from what was measured is the
                # mistake this file keeps recording.
                #
                # Published just below, so a window too short to measure can
                # leave them out rather than report a flattering zero.
                "spin_rate": spin,
                "turn_authority": authority,
                # This used to be the mean rate of change of the up-vector over
                # the airborne stretch, gated only on not sinking, and the
                # arch33 exploit champion logged 31.4 rad/s of it while having
                # zero actuated degrees of freedom -- there was nothing aboard
                # that could have commanded a turn.
                "turn_rate_held": turn if (
                    sink < 0.5 and authority > MIN_TURN_AUTHORITY and not gates
                ) else 0.0,
                "air_gates": float(len(gates)),
            })
            if excursion is not None:
                res.measurements["altitude_excursion"] = excursion
                res.measurements["excursion_ratio"] = excursion / band
            if wobble is not None:
                res.measurements["altitude_wobble"] = wobble
                res.measurements["wobble_ratio"] = wobble / band
            # Conditional for the same reason as the other three exits: a thrust
            # margin that could not be measured is absent, not zero.
            _tm = self.thrust_margin()
            if _tm is not None:
                res.measurements["thrust_margin"] = float(_tm)
            _lm = self.level_margin()
            if _lm is not None:
                res.measurements["level_margin"] = float(_lm)
            # Both of the terms that paid for something other than flying are
            # gone.  The flat 0.25 for being off the ground at all is now the
            # graded glide term, and the 0.2 for the launch velocity the
            # environment handed the machine is simply removed -- scoring a
            # machine on speed it did not produce is what made the launch a
            # gift worth having.  Every term that remains is a property of the
            # trajectory the machine flew.
            # What the machine is *scored* on is the task it was given: cruise
            # on the launch heading holding height, then turn on command and
            # hold height on the new heading (``_task_scores``).  ``flight``,
            # ``glide`` and ``station`` stay above as the ladder's measurements;
            # they used to be the score, and all three were vertical -- the air
            # competence had no forward term at all, so it asked a flapping wing
            # to hold height without giving it the one thing that holds a
            # flapping wing up.  A glider launched from 30 m sinks and cannot
            # turn when told, so it scores on neither phase.
            ts = self._task_scores(domain, depths, clearances, xys, n_want,
                                   airborne=airborne)
            tm = ts["measurements"]
            if gates:
                # Gated designs publish zeros on the rungs, as every other air
                # ladder metric above does.
                tm = {k: (0.0 if k in ("cruise_tracking", "cruise_height_hold",
                                       "turn_tracking", "turn_height_hold")
                          else v) for k, v in tm.items()}
            res.measurements.update(tm)
            res.parts = {"gate": float(credit * frac), "control": ts["task"] * chatter}
            return float(credit * frac * ts["task"] * chatter)

        if domain is Domain.WATER:
            target = self.TARGET_DEPTH
            # Progress from where the machine was *released* toward the target,
            # not absolute depth.  The segment starts it 4 m under, so
            # `max_depth / target` handed every design 0.4 before it did
            # anything -- measured p25 was 0.416 across arch37, i.e. a quarter
            # of the population scored the spawn and nothing else.  Dividing the
            # gain by the distance that remained makes 0.0 mean "went no deeper
            # than it was put", and leaves the spread intact (new p25/median/p75
            # 0.03 / 0.37 / 0.91 against 0.42 / 0.62 / 0.95).
            #
            # The ladder reads `depth_gain` for the same reason; a rung saying
            # "gained nothing" beside a competence saying 0.4 is the kind of
            # split this file has been bitten by before.
            start_depth = float(depths[0]) if len(depths) else 0.0
            gain = float(res.max_depth - start_depth)
            submerged = float(np.sum(depths > 0.2) / n_want)
            # Holding depth matters as much as reaching it: a machine that
            # plummets to 10 m has not demonstrated depth control.
            settled = depths[len(depths) // 2 :]
            err = float(np.mean(np.abs(settled - target))) if len(settled) else target
            res.depth_error = err
            # Station keeping in depth, the same measurement the air branch
            # makes on height and for the same reason.  `depth_error` is the
            # mean distance from the target over the late half, so a machine
            # sinking straight through the target at a constant rate scores
            # well on it, and `depth_gain` is a maximum, which a brick earns by
            # being denser than water.  Neither can see the *shape*: this is
            # the fraction of the late submerged window spent within a band of
            # the depth the machine settled at.  Measured with the actuators
            # held still, the seed plans' water competence was 104% of their
            # driven competence -- sinking paid for `reached`, `hold`,
            # `submerged` and `upright` at once -- and this is the term that
            # separates diving from sinking without needing a passive twin to
            # subtract.
            sub_idx = np.flatnonzero(np.asarray(depths, float) > 0.2)
            if len(sub_idx) > 1:
                s_half = max(len(sub_idx) // 2, 1)
                s_cap = max(int(self.STATION_WINDOW / self.timestep), 1)
                s_late = sub_idx[-min(s_half, s_cap):]
                # The same band as the air branch: a quarter of the machine's
                # own span, floored at half a metre, so a 0.4 m machine and a
                # 3 m one face the same test.
                s_band = max(0.25 * float(getattr(self.p, "max_span", 2.0)), 0.5)
                s_ref = float(depths[s_late[0]])
                s_dev = np.abs(np.asarray(depths, float)[s_late] - s_ref)
                station = float(np.mean(s_dev < s_band))
                excursion = float(s_dev.max() / s_band)
            else:
                station, excursion = 0.0, 0.0
            # Depth as a *gain*, the way ``takeoff_height`` is a gain over
            # resting clearance -- because the water segment releases the
            # machine four metres under (``SPAWN[Domain.WATER]``) and
            # ``max_depth`` therefore starts at 4 m for a brick.
            #
            # Measured over arch37's 14,058 water segments: the **minimum**
            # ``max_depth`` is 3.34 m, so the first two rungs of the water
            # ladder -- ``submerges`` at 0.5 m and ``dives`` at 3.0 m -- were
            # cleared by 100% of every run ever made, by being dropped.  Same
            # defect as the air ladder's first two rungs being paid for falling,
            # and it survived three runs after that one was found.
            #
            # The gain separates where the absolute did not: p25 +0.16 m,
            # median +2.23 m, p90 +8.11 m.  It is floored at zero -- a maximum
            # cannot be below the first sample -- so a machine that only rises
            # scores 0 and now has somewhere to climb from, where before it
            # scored the same rung as one that dove nine metres.
            #
            # Those quantiles are `max_depth - 4.0` against a release that is
            # randomised by 0.2 m, so the lowest threshold is approximate to
            # about that much.  arch38 measures the gain directly.
            res.measurements.update({
                "max_depth": float(res.max_depth),
                "depth_gain": gain,
                "depth_error": err,
                "water_speed": float(res.mean_speed) if submerged > 0.5 else 0.0,
                "depth_station_keeping": station,
                "depth_excursion_ratio": excursion,
            })
            # ``served`` closes the same truncation hole here: holding depth and
            # staying upright for a tenth of a second is not a demonstration of
            # either, and a machine that ends its episode early has not done the
            # thing it was asked to do.
            #
            # Being underwater and being upright are *state*, so they gate the
            # task rather than adding to it.  Measured over the 985 segments of
            # `runs/arch40_stopped_passive_twin`: the additive form paid a
            # machine with its actuators held still **0.422** in water, two
            # fifths of it the constant ``0.2 * submerged + 0.2 * upright``.
            #
            # The task (``_task_scores``) is two phases with one purpose each:
            # hold at a commanded depth below the release, and cruise on a
            # commanded heading at that depth, in an order drawn per generation.
            # Until 2026-09-21 this was one blend of moving, holding and
            # diving, and the blend fought itself -- ``corr(headway,
            # depth_station_keeping)`` was -0.264 over arch40's 6,288 water
            # segments, and it paid a perfect hover 0.20 against a perfect mover
            # 0.80.  Sinking scores only as far as it reaches the commanded
            # depth and stops there, which is what makes it a chosen operation.
            ts = self._task_scores(domain, depths, clearances, xys, n_want)
            res.measurements.update(ts["measurements"])
            gate = float(served * submerged * (0.5 + 0.5 * upright))
            res.parts = {"gate": gate, "control": ts["task"] * chatter}
            return float(gate * ts["task"] * chatter)

        # LAND
        #
        # Contact *while upright*, not contact.  Measured over arch38's eight
        # highest-mission elites at one scattered initial condition, the ungated
        # fraction reads 0.8197 with the policy driving and **0.8227 with the
        # actuators held still** -- a machine lying on the beach is in contact
        # with it, and `supports_itself` is the rung that reads this.  Switching
        # the machine off moved the number by three parts in a thousand, in the
        # wrong direction.
        #
        # Same treatment `land_speed` and `slope_climbed` just had, and the
        # ungated figure is kept under its own name so the record stays
        # re-derivable.
        raw_contact = float(np.sum(np.asarray(contacts) > 0.5) / n_want)
        _c = np.asarray(contacts, float)
        _u = np.asarray(ups, float)
        _n = min(len(_c), len(_u))
        contact = float(
            np.sum((_c[:_n] > 0.5) & (_u[:_n] > 0.5)) / n_want) if _n else 0.0
        # Height gained against the beach's own slope: walking uphill is the
        # capability, not merely moving.
        from ..core.mjcf import beach_surface_z

        climbed = 0.0
        if len(alts) > 2:
            climbed = float(beach_surface_z(float(self.root_pos()[0])) - beach_surface_z(
                float(self.root_pos()[0]) - res.distance))
        # Take-off, measured where the machine actually starts on the ground.
        #
        # The air segment is a *launch* -- 30 m up at trim speed -- so nothing
        # in this project had ever measured whether a design can leave the
        # ground under its own power, and ``transitions.py`` records that none
        # of the seed plans can.  A probe over arch34's final archive put 96
        # elites on the beach for 30 s: the median never exceeded its 0.05 m
        # spawn clearance, and the best reached 0.133 m against a 0.50 m bar.
        #
        # A threshold on peak clearance is therefore the wrong instrument -- it
        # reads zero across a population whose entire signal lives below it.
        # This is the projectile estimate instead: the height the machine's
        # present upward velocity would carry it to, which is dense from the
        # first gram of unweighting and does not require leaving the ground at
        # all.  Same densification, and the same equations, as Wang et al.,
        # "Towards Quadrupedal Jumping and Walking for Dynamic Locomotion using
        # Reinforcement Learning" (arXiv 2510.24584).
        # Measured as a *gain* over where the machine rests, not as an absolute
        # clearance: a tall design sitting still would otherwise out-score a
        # short one that hopped, and the resting height is geometry rather than
        # an achievement.  Zero therefore means "never rose", for every body.
        # Gated on being off the ground and the right way up, and both gates
        # were put there by measurement.  Ungated over 64 arch34 elites this
        # read a 90th percentile of 0.378 m and a maximum of 1.035 m from a
        # population whose best controlled hop is 0.13 m: it was scoring the
        # bounce of a machine falling over.  That is the same defect as the old
        # air score paying for having been thrown and the old turn rate paying
        # for a tumble, arriving a third time by a different route.
        takeoff = 0.0
        if clearances is not None and len(np.asarray(clearances)):
            c = np.asarray(clearances, float)

            def _align(x, fill=0.0):
                a = np.asarray(x, float) if x is not None else np.full_like(c, fill)
                return (a[:len(c)] if len(a) >= len(c)
                        else np.pad(a, (0, len(c) - len(a)), constant_values=fill))

            v = _align(vzs)
            free = _align(contacts, 1.0) < 0.5      # not touching the ground
            level = _align(ups, 0.0) > 0.5          # not on its back or side
            apex = c + np.maximum(v, 0.0) ** 2 / (2.0 * 9.81)
            ok = np.isfinite(apex) & free & level
            rest = float(c[0]) if np.isfinite(c[0]) else 0.0
            takeoff = max(float(np.max(apex[ok])) - rest, 0.0) if ok.any() else 0.0
            # The estimate is dense and the achievement is not, so both are
            # kept -- the same split as `sink_rate` against
            # `measured_sink_rate`.  A machine rebounding off the beach is
            # airborne, upright and rising, so the gates above cannot tell a
            # push-off from a bounce; the height it *actually* reached can.
            got = np.isfinite(c) & free & level
            measured = max(float(np.max(c[got])) - rest, 0.0) if got.any() else 0.0

            # Two more gates on the *scored* height, both put here by arch35.
            #
            # First: a departure is a flight claim, and `airworthiness` already
            # says which designs may not make one.  A wingless body can still be
            # thrown, so it keeps `unweights` and `hops` -- leaving the ground is
            # what those rungs ask -- but `clears` is "a crossing's worth of
            # height" and `climbs_out` is "a departure, not a hop", and neither
            # is available without a lifting surface.  arch35 had 31 wingless
            # segments at `clears` and 13 at `climbs_out`, one of them reaching
            # 4.15 m *measured*: the fourth time a score has paid for
            # uncontrolled motion, and the fourth time the fix is a gate.
            #
            # Second: posture over the whole segment, not just at the apex.  The
            # per-sample `free & level` above catches "upright at the moment it
            # left", which a body flung by a contact impulse passes on its way
            # through.  Over arch35, splitting on this same 0.7 bar -- the one
            # `stays_upright` already uses, so this is not a second standard --
            # the designs *below* it scored 0.79-0.95x the mission of those
            # above once A's own multiplier was divided out.  They were being
            # paid for take-off and were worse at the mission.
            #
            # Thresholds set from the measured distribution, per the rule this
            # project keeps relearning: over arch35's 7806 land segments these
            # two together leave unweights 24.9%, hops 9.3%, clears 2.4% and
            # climbs_out 0.7% standing.  Thinner, and not empty -- a rung
            # nobody stands on carries no gradient.
            if float(getattr(self.p, "wing_area", 0.0)) < WING_AREA_FLOOR:
                takeoff = min(takeoff, TAKEOFF_WINGLESS_CAP)
            if float(upright) < TAKEOFF_POSTURE_BAR:
                takeoff = 0.0
            # `measured_takeoff_height` is deliberately left ungated by both.
            # It is the diagnostic that exists to catch the next instance of
            # this, and gating it would hide exactly the evidence that found
            # this one.
        # `land_speed` is net displacement over the *whole* segment, so a
        # machine that lurches half a metre in one second and then falls over
        # averages an eighth of the speed it actually produced.  61.6% of arch34
        # sat at the land rung below `moves` (0.1 m/s) for exactly this kind of
        # reason, with a population median of 0.041 m/s.
        #
        # `land_peak_speed` is the best displacement rate over any one-second
        # window: the same densification as `takeoff_height`, applied to the
        # other place where a threshold sits above the whole distribution.  It
        # is reported alongside the mean rather than replacing it -- sustaining
        # motion is a different capability from producing it, and the ladder
        # should be able to tell them apart.
        peak = 0.0
        gated_mean, held_any = 0.0, False
        if len(alts) > 2 and res.duration > 0:
            step = res.duration / max(len(alts) - 1, 1)
            w = max(int(round(1.0 / step)), 1)
            if xys is not None and len(xys) > w:
                t = np.asarray(xys, float)
                d = np.linalg.norm(t[w:] - t[:-w], axis=1)
                # Only windows the machine spent upright throughout.  Ungated
                # this peaked at 3.23 m/s across arch34's elites, which is a
                # body sliding down the beach on its side, not locomotion.
                u = np.asarray(ups, float)
                u = (u[:len(t)] if len(u) >= len(t)
                     else np.pad(u, (0, len(t) - len(u))))
                # Per-window minimum of the upright trace: a window counts
                # only if posture held for all of it.  (A suffix-minimum was
                # computed here first and overwritten on the next line without
                # ever being read.)
                held = np.array([u[i:i + w + 1].min() for i in range(len(t) - w)])
                d = np.where(np.isfinite(d) & (held > 0.5), d, 0.0)
                peak = float(np.max(d) / (w * step)) if d.size else 0.0
                # The same gate, as a mean rather than a peak.
                #
                # `land_speed` was `res.mean_speed` -- net displacement over the
                # segment, ungated -- and it is not a locomotion measurement.
                # Measured over arch38's eight highest-mission elites at the same
                # scattered initial condition, median `land_speed` is 0.4312 m/s
                # with the policy driving and **0.4262 m/s with the actuators
                # held still**: switching the machine off changes it by one
                # percent.  `slope_climbed` is the same, 0.4139 against 0.4091,
                # because it is computed from the same `res.distance`.  Between
                # them those two carried three of the six land rungs -- `moves`,
                # `walks` and `climbs_slope` -- so half the land ladder was
                # scoring a machine falling over and sliding down the beach.
                #
                # The gate that works is the one already above: `land_peak_speed`
                # and `takeoff_height` both read **exactly 0.0** for the passive
                # machine, and both are gated on holding posture. So this is the
                # same masked windows, averaged over all of them rather than
                # maximised -- zeros included, so a single lucky window cannot
                # earn it, and a machine that never held posture reads zero.
                gated_mean = float(np.mean(d) / (w * step)) if d.size else 0.0
                held_any = bool(d.size) and bool(np.any(d > 0.0))
        res.measurements.update({
            "upright": upright,
            "contact_fraction": contact,
            "measured_contact_fraction": raw_contact,
            # Gated, and the ungated figure kept beside it under its own name --
            # the same split as `sink_rate` against `measured_sink_rate` and
            # `takeoff_height` against `measured_takeoff_height`, so the record
            # stays honest and every old number stays re-derivable.
            "land_speed": gated_mean,
            "measured_land_speed": float(res.mean_speed),
            "land_peak_speed": peak,
            "slope_climbed": max(climbed, 0.0) if held_any else 0.0,
            "measured_slope_climbed": max(climbed, 0.0),
            "takeoff_height": takeoff,
            "measured_takeoff_height": measured,
        })
        # Posture *gates* locomotion rather than substituting for it.
        #
        # This was ``0.4*progress + 0.3*contact + 0.3*upright``, and measured
        # across the six plans progress is 0.003 to 0.13 while contact and
        # upright are both around 0.95.  So six tenths of a locomotion score was
        # awarded for lying still the right way up, every design scored 0.54 to
        # 0.60, and the spread between a machine that covered 0.63 m and one
        # that covered 0.01 m was 0.055.  A rock scores 0.57.
        #
        # A machine that falls over or leaves the ground cannot walk, so posture
        # belongs in front as a precondition -- which is how the air score
        # already treats being airborne.  And the climb term was computed here,
        # commented as "walking uphill is the capability, not merely moving",
        # and then left out of the return: measured, documented as the point,
        # and never read, which is exactly what the camber gene was doing.
        #
        # The absolute numbers drop a long way because nothing walks yet.  The
        # gradient is what matters and it improves by more than an order of
        # magnitude: the spread across these six plans goes from 1.1x to 36x.
        posture = 0.5 * contact + 0.5 * upright
        # The task: walk on a commanded heading at the commanded speed, and
        # stop when told to (``_task_scores``).  It replaces ``0.65 * progress
        # + 0.35 * climb``, where progress was displacement in *any* direction
        # -- drifting sideways was progress -- and the climb was the only
        # direction anything asked for.  ``slope_climbed`` is still measured
        # above, for the ladder.
        ts = self._task_scores(domain, depths, clearances, xys, n_want)
        res.measurements.update(ts["measurements"])
        res.parts = {"gate": float(served * posture), "control": ts["task"] * chatter}
        return float(served * posture * ts["task"] * chatter)

    # ------------------------------------------------------------- mobility ID

    def identify(self, domain: Domain, *, probe_time: float = 1.2,
                 n_probes: int = 24, seed: int = 0,
                 max_modes: int = 6) -> MobilityBasis:
        """Discover this body's control axes in one medium.

        ``n_probes`` and ``max_modes`` match `identify_batch`'s defaults on
        purpose: the same body identified through the two paths has to get the
        same basis, and they were 8/4 here against 24/6 there.
        """
        self.reset(domain, randomise=False)
        snap = self.snapshot()
        base = self.cpg.base

        def reset_fn():
            self.restore(snap)
            self.budget.reset()

        def step_fn(delta):
            params = CPGParams.from_flat(base.flat() + delta, self.cpg.n)
            before = self.root_pos().copy()
            acc = np.zeros(6)
            n = int(probe_time / self.timestep)
            for _ in range(n):
                self.step(self.cpg.command(params, self.data.time))
                acc += self.body_twist()
            if not np.all(np.isfinite(self.root_pos())):
                return np.zeros(6)
            return acc / max(n, 1)

        return identify_mobility(
            step_fn,
            reset_fn,
            self.cpg.n_params,
            n_probes=n_probes,
            medium=domain.value,
            rng=np.random.default_rng(seed),
            max_modes=max_modes,
        )
