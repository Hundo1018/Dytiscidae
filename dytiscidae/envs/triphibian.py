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

    @property
    def energy_margin(self) -> float:
        if self.energy_required_wh <= 1e-6:
            return -1.0
        return self.energy_available_wh / self.energy_required_wh - 1.0


# --------------------------------------------------------------------------
# Tier 0: analytic
# --------------------------------------------------------------------------


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
    r.notes.append(f"v_cruise={v_cruise:.1f}m/s P_air={p_air:.0f}W P_water={p_water:.0f}W")
    return r


# --------------------------------------------------------------------------
# Tier 1: short dynamic episodes
# --------------------------------------------------------------------------


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
    #: A flight test does not start with the aircraft at rest in mid-air.  The
    #: launch is deliberately identical for every design -- a per-design trim
    #: speed would hand a tiny-winged machine a large free velocity and the
    #: speed term would pay it for that.
    SPAWN = {
        Domain.AIR: (-40.0, 0.0, 30.0),
        Domain.WATER: (-8.0, 0.0, -4.0),
        Domain.LAND: (15.0, 0.0, 0.9),
    }

    #: Bounds on the launch airspeed, m/s.  Outside this band the quasi-steady
    #: coefficients are extrapolating and the design is not one this mission is
    #: about anyway.
    LAUNCH_SPEED_RANGE = (6.0, 30.0)

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

        import mujoco

        self._mj = mujoco
        self.root_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, phenotype.segments[0].name
        ) if phenotype.segments else 0

        # Joint travel limits for the CPG.
        ranges = []
        for name in self.act_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            jid = self.model.actuator_trnid[aid, 0]
            ranges.append(self.model.jnt_range[jid])
        self.joint_range = np.array(ranges) if ranges else np.zeros((0, 2))
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
        phases = []
        for name in self.act_names:
            seg_name = name[:-2]  # strip the "_a" suffix
            seg = next((x for x in phenotype.segments if x.name == seg_name), None)
            # Multiplied by chain depth, because ``phase_offset`` is a *phase
            # advance per link*, not an absolute phase.  Every segment in a
            # recursive chain shares one Part, so a flat offset makes the whole
            # chain beat in unison -- which is not a travelling wave, it is a
            # very long paddle.  Accumulating it down the chain is what makes
            # the wave travel, and the wave is where the thrust comes from.
            phases.append(seg.part.phase_offset * max(seg.depth, 1) if seg is not None else 0.0)
        if phases:
            self.cpg.base.phase = np.array(phases, float)
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

    def _trim(self) -> tuple:
        cached = getattr(self.p, "_measured_trim", None)
        if cached is not None:
            return cached
        lo, hi = self.LAUNCH_SPEED_RANGE
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
        if top < weight:
            # Cannot fly at any speed we are willing to model.  Launched at the
            # cap, at the attitude that does least badly, and it will fall --
            # which is the correct answer for a design that cannot fly.
            return float(hi), float(top_a)
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
        return v, lowest_pitch(v, top_a)

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
        """
        tw = self.body_twist()
        R = self.data.xmat[self.root_body].reshape(3, 3)
        gravity_body = R.T @ np.array([0.0, 0.0, -1.0])
        d = self.depth()
        cmd = np.zeros(3)
        if target is not None:
            cmd[DOMAIN_CYCLE.index(target)] = 1.0
        return np.concatenate(
            [
                np.clip(tw[:3] / 5.0, -3, 3),
                np.clip(tw[3:] / 4.0, -3, 3),
                gravity_body,
                [np.tanh(d / 5.0), self.solver.diag.mean_submerged],
                cmd,
            ]
        )

    #: 3 linear + 3 angular + 3 gravity + depth + wetness + 3 commanded domain.
    OBS_DIM = 14

    # ------------------------------------------------------------------ stepping

    def step(self, target_angles: np.ndarray) -> bool:
        """Advance one timestep.  Returns False when the battery is flat."""
        if len(self.act_names):
            self.data.ctrl[: len(target_angles)] = target_angles
        self.data.xfrc_applied[:] = 0.0
        self.solver.apply(self.data, self.data.time)
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
        depths, alts, ups, contacts, clearances = [], [], [], [], []
        peak_slam = 0.0
        cur = p

        for i in range(n_steps):
            if policy is not None and basis is not None and i % control_every == 0:
                coeffs = policy.act(self.observation(domain))
                cur = basis.command_params(p, coeffs, self.cpg.n)
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
            peak_slam = max(peak_slam, self.solver.diag.slam)

        end = self.root_pos().copy()
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
        )
        return res

    def _score_segment(self, domain, res, depths, alts, ups, contacts,
                       clearances=None) -> float:
        """Domain competence in [0, 1].

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
                return 0.0  # never left the surface: no flight to score
            # Sink is a rate, and a rate needs a baseline long enough to be one.
            # Over a tenth of a second every launched object has a sink rate of
            # nearly zero, including a brick.
            if np.sum(airborne) * self.timestep < 0.35 * res.duration:
                return float(frac * 0.25)

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
            # flight.
            idx = np.flatnonzero(airborne)
            late = idx[len(idx) // 2:]
            if len(late) > 1:
                span_s = (late[-1] - late[0]) * self.timestep
                sink = float(
                    (clearances[late[0]] - clearances[late[-1]]) / max(span_s, 1e-6)
                )
            else:
                sink = 9.9
            flight = float(np.clip(1.0 - sink / 1.5, 0.0, 1.0))
            # Speed is scored against this design's own launch, not against a
            # fixed 8 m/s.  With the launch set to each machine's trim speed, an
            # absolute threshold would hand full marks to every design that
            # simply had a small wing, since a small wing means a fast launch.
            # Measured this way the term asks whether the machine *kept* the
            # speed it was given, which is what sustaining flight means.
            speed = float(np.clip(res.mean_speed / max(self.launch_speed, 1e-6), 0.0, 1.0))
            res.altitude_held = flight
            # Turn rate sustained *while* not losing height: manoeuvring and
            # falling out of a turn are different things.
            turn = float(np.mean(np.abs(np.diff(np.asarray(ups)[idx])))) / max(
                self.timestep, 1e-6
            ) if len(idx) > 2 else 0.0
            res.measurements.update({
                "airborne_fraction": frac,
                "sink_rate": sink,
                "turn_rate_held": turn if sink < 0.5 else 0.0,
            })
            # Every term is gated on actually being up there.
            return float(frac * (0.55 * flight + 0.25 + 0.2 * speed))

        if domain is Domain.WATER:
            target = 10.0
            reached = float(np.clip(res.max_depth / target, 0.0, 1.0))
            submerged = float(np.sum(depths > 0.2) / n_want)
            # Holding depth matters as much as reaching it: a machine that
            # plummets to 10 m has not demonstrated depth control.
            settled = depths[len(depths) // 2 :]
            err = float(np.mean(np.abs(settled - target))) if len(settled) else target
            res.depth_error = err
            hold = float(np.clip(1.0 - err / target, 0.0, 1.0))
            res.measurements.update({
                "max_depth": float(res.max_depth),
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
        res.measurements.update({
            "upright": upright,
            "contact_fraction": contact,
            "land_speed": float(res.mean_speed),
            "slope_climbed": max(climbed, 0.0),
        })
        return served * float(0.4 * progress + 0.3 * contact + 0.3 * upright)

    # ------------------------------------------------------------- mobility ID

    def identify(self, domain: Domain, *, probe_time: float = 1.2,
                 n_probes: int = 8, seed: int = 0) -> MobilityBasis:
        """Discover this body's control axes in one medium."""
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
        )
