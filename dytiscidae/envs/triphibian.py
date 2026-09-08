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
from ..physics.fluid import FluidSolver
from ..physics.medium import GRAVITY, MediumField, SeaState
from ..physics.structure import ballast_pump_power


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

    @property
    def total_seconds(self) -> float:
        return self.cycles * len(DOMAIN_CYCLE) * self.seconds_per_domain

    @property
    def transitions(self) -> list[str]:
        return CYCLE_TRANSITIONS * self.cycles


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

#: The largest lift coefficient the strip model will produce (``lift_coefficient``
#: caps at ``1.2 * cl_max`` with ``cl_max`` reaching 1.9 under a strong leading-
#: edge vortex).  1.8 is the figure Tier 0's stall-speed estimate already used;
#: naming it here keeps the gate below and that estimate from drifting apart.
CL_MAX = 1.8

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

    SPAWN = {
        Domain.AIR: (-40.0, 0.0, 30.0),
        Domain.WATER: (-8.0, 0.0, -4.0),
        Domain.LAND: (15.0, 0.0, 0.9),
    }


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
        self.medium = MediumField(sea_state=sea_state, current=current, wind=wind)
        self.timestep = timestep

        from ..core.mjcf import scene_xml

        scene = scene_xml(timestep=timestep)
        self.model, self.data, self.act_names, self.panels = compile_phenotype(
            phenotype, scene=scene, detail=detail
        )
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
            phases.append(seg.part.phase_offset * max(seg.depth, 1) if seg is not None else 0.0)
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
        x, y, z = self.SPAWN[domain]
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
            self.data.qvel[:3] *= 1.0 + float(rng.normal(0.0, 0.15 * strength))
            self.data.qvel[:3] += rng.normal(0.0, 0.35 * strength, 3)
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
            return float(cached)

        out = 0.0
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
            # A body whose joints cannot be posed says nothing about thrust; it
            # must not say "lots".  Zero is the honest answer and it is also the
            # ladder's first rung, so such a design stops there.
            out = 0.0
        out = float(np.clip(out, -10.0, 10.0)) if np.isfinite(out) else 0.0
        try:
            self.p._measured_thrust = out
        except Exception:
            pass
        return out

    def _trim(self) -> tuple:
        cached = getattr(self.p, "_measured_trim", None)
        if cached is not None:
            return cached
        lo, hi = LAUNCH_SPEED_RANGE
        out = self._measure_trim_speed(lo, hi)
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
        m, d = self.model, self.data
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
        g = getattr(self, "_mgeoms", None)
        if g is None:
            g = np.nonzero(self.model.geom_bodyid != 0)[0]
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
        """
        tw = self.body_twist()
        R = self.data.xmat[self.root_body].reshape(3, 3)
        gravity_body = R.T @ np.array([0.0, 0.0, -1.0])
        d = self.depth()
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
                    np.tanh((d - self.TARGET_DEPTH) / 3.0),
                    1.0 if self._touching_ground() else 0.0,
                    float(b.energy_j / max(b.capacity_j, 1e-9)),
                    stroke,
                    stroke_rate,
                ],
                self.morphology_context,
            ]
        )

    #: 3 linear + 3 angular + 3 gravity + depth + wetness + 3 commanded domain
    #: + depth error + contact + battery + stroke phase and rate + 8 morphology.
    OBS_DIM = 19 + MORPHOLOGY_DIM

    #: The scorer's depth target, shared with the observation's error channel
    #: so the sensed error and the scored error cannot drift apart.
    TARGET_DEPTH = 10.0

    # ------------------------------------------------------------------ stepping

    def step(self, target_angles: np.ndarray) -> bool:
        """Advance one timestep.  Returns False when the battery is flat."""
        if len(self.act_names):
            self.data.ctrl[: len(target_angles)] = target_angles
        self.data.xfrc_applied[:] = 0.0
        self.solver.apply(self.data, self.data.time)
        if self.jets.n:
            self.jets.apply(self.model, self.data, self.medium,
                            self.data.time, self.timestep)
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
    ) -> SegmentResult:
        """Run one segment and measure what happened."""
        p = params or self.cpg.base
        res = SegmentResult(domain=domain, duration=duration)
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
        peak_slam = 0.0
        cur = p

        for i in range(n_steps):
            if policy is not None and basis is not None and i % control_every == 0:
                coeffs = policy.act(self.observation(domain))
                cur = basis.command_params(p, coeffs, self.cpg.n)
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
            peak_slam = max(peak_slam, self.solver.diag.slam)

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
            vzs=np.array(vzs), xys=np.array(xys),
        )
        return res

    def _score_segment(self, domain, res, depths, alts, ups, contacts,
                       clearances=None, *, spins=None, commands=None,
                       responses=None, vzs=None, xys=None) -> float:
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
            return 0.0
        upright = float(np.clip(np.mean(ups), 0.0, 1.0))
        # Samples the segment should have produced had it run to term.
        if clearances is None:
            clearances = -np.asarray(depths, float)
        n_want = max(int(round(res.duration / self.timestep)), 1)
        n_got = len(alts)
        # Anything that ended early is measured against what it was asked to do.
        served = min(n_got / n_want, 1.0)

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
            if frac < 0.05:
                # Even here.  `lift_margin` is what the airframe can lift, not
                # what this episode did with it, and the ladder's bottom four
                # rungs read it -- so leaving it out of any air path makes a
                # design that never got clear indistinguishable from one with no
                # wings.  Three paths leave this branch of `_score_segment` and
                # all three publish it; the test below enforces that, because
                # the first attempt at this changed one of the three and a grep
                # for another key said that was all of them.
                res.measurements["lift_margin"] = float(self.lift_margin)
                res.measurements["thrust_margin"] = float(self.thrust_margin())
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
            if np.sum(airborne) * self.timestep < 0.35 * res.duration:
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
                    "thrust_margin": float(self.thrust_margin()),
                })
                return float(credit * frac * 0.10)

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
                "thrust_margin": float(self.thrust_margin()),
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
            # Both of the terms that paid for something other than flying are
            # gone.  The flat 0.25 for being off the ground at all is now the
            # graded glide term, and the 0.2 for the launch velocity the
            # environment handed the machine is simply removed -- scoring a
            # machine on speed it did not produce is what made the launch a
            # gift worth having.  Every term that remains is a property of the
            # trajectory the machine flew.
            return float(
                credit * frac * (0.55 * flight + 0.25 * glide + 0.20 * station)
            )

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
            reached = float(np.clip(gain / max(target - start_depth, 1e-6),
                                    0.0, 1.0))
            submerged = float(np.sum(depths > 0.2) / n_want)
            # Holding depth matters as much as reaching it: a machine that
            # plummets to 10 m has not demonstrated depth control.
            settled = depths[len(depths) // 2 :]
            err = float(np.mean(np.abs(settled - target))) if len(settled) else target
            res.depth_error = err
            hold = float(np.clip(1.0 - err / target, 0.0, 1.0))
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
            })
            # ``served`` closes the same truncation hole here: holding depth and
            # staying upright for a tenth of a second is not a demonstration of
            # either, and a machine that ends its episode early has not done the
            # thing it was asked to do.
            return served * float(
                0.35 * reached + 0.25 * hold + 0.2 * submerged + 0.2 * upright
            )

        # LAND
        contact = float(np.sum(np.asarray(contacts) > 0.5) / n_want)
        progress = float(np.clip(res.mean_speed / 0.6, 0.0, 1.0))
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
                held = np.minimum.accumulate(u[::-1])[::-1]
                held = np.array([u[i:i + w + 1].min() for i in range(len(t) - w)])
                d = np.where(np.isfinite(d) & (held > 0.5), d, 0.0)
                peak = float(np.max(d) / (w * step)) if d.size else 0.0
        res.measurements.update({
            "upright": upright,
            "contact_fraction": contact,
            "land_speed": float(res.mean_speed),
            "land_peak_speed": peak,
            "slope_climbed": max(climbed, 0.0),
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
        climb = float(np.clip(max(climbed, 0.0) / 0.5, 0.0, 1.0))
        return served * posture * float(0.65 * progress + 0.35 * climb)

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
