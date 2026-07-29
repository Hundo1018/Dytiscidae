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

    @property
    def cost_of_transport(self) -> float:
        """Dimensionless energy per unit distance per unit weight."""
        if self.distance < 0.1:
            return float("inf")
        return self.mean_power * self.duration / max(self.distance, 1e-6)


@dataclass
class MissionResult:
    """Aggregate of a whole evaluation, at whatever fidelity produced it."""

    tier: int
    feasible: bool = False
    structural_margin: float = 0.0
    segments: dict[str, SegmentResult] = field(default_factory=dict)
    transition_ok: dict[str, bool] = field(default_factory=dict)
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

    #: Spawn poses per domain.  Air is well clear of the surface, water is deep
    #: enough that the free surface is not doing the work, land is up the beach.
    SPAWN = {
        Domain.AIR: (-8.0, 0.0, 6.0),
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
    ) -> None:
        self.p = phenotype
        self.rng = np.random.default_rng(seed)
        self.medium = MediumField(sea_state=sea_state, current=current, wind=wind)
        self.timestep = timestep

        from ..core.mjcf import scene_xml

        scene = scene_xml(timestep=timestep)
        self.model, self.data, self.act_names, self.panels = compile_phenotype(
            phenotype, scene=scene
        )
        self.solver = FluidSolver(self.model, self.panels, self.medium)

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
        self.solver.reset()
        self.cpg.reset()
        self.budget.reset()
        self._mj.mj_forward(self.model, self.data)

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

    def root_pos(self) -> np.ndarray:
        return self.data.xpos[self.root_body].copy()

    def depth(self) -> float:
        return float(self.medium.depth(self.root_pos()[None, :], self.data.time)[0])

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
        depths, alts, ups, contacts = [], [], [], []
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
            alts.append(pos[2])
            R = self.data.xmat[self.root_body].reshape(3, 3)
            ups.append(float(R[2, 2]))
            contacts.append(1.0 if self.data.ncon > 0 else 0.0)
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
        res.competence = self._score_segment(domain, res, np.array(depths), np.array(alts),
                                             np.array(ups))
        return res

    def _score_segment(self, domain, res, depths, alts, ups) -> float:
        """Domain competence in [0, 1].

        Each domain is scored on what actually matters there, not on a generic
        "went far" reward -- flying is about not falling, diving is about
        holding depth, walking is about making progress while touching ground.
        """
        if not res.survived or len(alts) == 0:
            return 0.0
        upright = float(np.clip(np.mean(ups), 0.0, 1.0))

        if domain is Domain.AIR:
            # Held altitude, stayed dry, went somewhere, stayed upright.
            drop = float(alts[0] - alts[-1])
            held = float(np.clip(1.0 - drop / 6.0, 0.0, 1.0))
            dry = float(np.mean(depths < 0.0))
            speed = float(np.clip(res.mean_speed / 8.0, 0.0, 1.0))
            res.altitude_held = held
            return float(0.45 * held + 0.2 * dry + 0.2 * speed + 0.15 * upright)

        if domain is Domain.WATER:
            target = 10.0
            reached = float(np.clip(res.max_depth / target, 0.0, 1.0))
            submerged = float(np.mean(depths > 0.2))
            # Holding depth matters as much as reaching it: a machine that
            # plummets to 10 m has not demonstrated depth control.
            settled = depths[len(depths) // 2 :]
            err = float(np.mean(np.abs(settled - target))) if len(settled) else target
            res.depth_error = err
            hold = float(np.clip(1.0 - err / target, 0.0, 1.0))
            return float(0.35 * reached + 0.25 * hold + 0.2 * submerged + 0.2 * upright)

        # LAND
        contact = res.ground_contact_fraction
        progress = float(np.clip(res.mean_speed / 0.6, 0.0, 1.0))
        return float(0.4 * progress + 0.3 * contact + 0.3 * upright)

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
