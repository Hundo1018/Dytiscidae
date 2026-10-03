"""Crossing between media, scored rather than merely survived.

Why this exists
---------------
A transition used to return a boolean.  ``crossed`` was True if the machine's
depth changed sign at any point in six seconds, and the three booleans became a
pass rate that multiplied the mission score.  Under that rule a machine that
fell through the surface out of control, tumbling, at twice the speed its hull
survives, scored exactly the same as one that entered cleanly and came out the
other side under command -- provided both changed sign.  The only graded
quantity anywhere was water-entry speed, and it was graded as a cliff: under the
hull limit, full marks; over it, total failure.

That is a bad way to score the hardest part of the mission.  Every real
triphibian machine that exists -- AquaMAV, Nezha, the Beihang and Northeastern
flying-swimming quadrotors -- is dominated in its design by the crossings, not
by the cruise phases.  The crossing is where the structure sees its peak load,
where the control authority collapses because the medium is changing underneath
the actuators, and where nearly all of the energy goes.  Handing all of that to
a boolean means the search cannot tell an elegant crossing from a survivable
accident, so it has no gradient to climb toward one.

What is measured
----------------
Six things, all in [0, 1], recorded separately so the judge can weight them and
so a run can say *which* part of a crossing a design is bad at:

``crossed``
    Did the machine actually get from one medium to the other and stay there.
    Necessary but not sufficient -- everything else is conditioned on it.
``shock``
    Peak structural load during the crossing against what the hull survives.
    Water entry is the case that matters: slam pressure goes as v^2.
``control``
    Attitude excursion through the crossing.  A machine that arrives inverted
    has not completed a transition in any useful sense, even if it arrives.
``settle``
    How long after the boundary it takes to reach a steady state in the new
    medium.  A crossing that ends in a two-second tumble has cost the mission
    two seconds of the next leg.
``economy``
    Energy spent on the crossing against the machine's own budget for it.
``exit_state``
    Whether the terminal state is one the next leg can start from: right way
    up, moving the right way, at a sensible depth or height.

None of these is combined here.  Combination is the judge's job, and the judge's
weights move; keeping the measurements separate is what lets the judge tighten
without the raw record of what happened changing underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .triphibian import Domain, TriphibianEnv

#: Which medium each crossing starts and ends in.
#: Which crossings the seeds can and cannot make, measured against the six body
#: plans with no controller, so that the next person starts from the answer.
#:
#: **Superseded 2026-10-03** (ARCH46_SPEC §8): the first two rows were passive.
#: `air_to_water` was a fall and `water_to_land` was credited for settling
#: *into* the water.  With the hold-then-go command and directional crossings,
#: no seed plan with no controller makes any crossing, and a still machine
#: earns at most 0.013 of a graded approach.
#:
#:     air_to_water    all six      falling into the sea is easy
#:     water_to_land   five of six  the ramp is reachable from x = 8
#:     land_to_air     none         nothing gets off the ground
#:
#: The last one is not a duration problem, which was the obvious guess: given 6,
#: 15 or 30 seconds the best clearance any plan reaches is 0.06 m against a bar
#: of 0.50.  Every one of these designs can *glide* once it is launched at trim
#: speed -- that is what the air segment tests -- and not one can accelerate
#: itself from rest to trim speed on the ground.  They have no ground run, no
#: jump, and not enough flapping thrust to hover out of it.
#:
#: What takeoff would take, measured rather than argued:
#:
#:     plan      W/S   v_trim   助跑@T/W=1   靜止拍翅升力
#:     ray        51    11.8      7.1 m         0.03 W
#:     gannet     86    13.3      9.1 m         0.06 W
#:     bat        86    15.8     12.8 m         0.01 W
#:     beetle    167    17.8     16.2 m         0.07 W
#:
#: Reaching trim speed costs 333 to 4701 J, which even at a thrust equal to the
#: machine's own weight is a 7 to 46 m ground run.  Nothing here produces
#: anything like that thrust: standing still and flapping, the best of them
#: lifts 7% of its weight.
#:
#: The obvious escape -- flap harder -- does not exist.  Swept on the beetle,
#: the only real flapper, at full stroke amplitude:
#:
#:     2.2 Hz  0.19 W   219 W   structure ok
#:     5.0 Hz  0.17 W   484 W   structure ok
#:    10.0 Hz  0.11 W   673 W   structure ok
#:    16.0 Hz  0.07 W   750 W   spar fails, margin -0.36
#:
#: Amplitude roughly triples lift; frequency *reduces* it while quadrupling the
#: power and eventually breaking the spar, which is the f^2 inertial load doing
#: exactly what the structural check exists to catch.  So the two knobs the
#: genome has are already at their best at 0.19 W, five times short of hovering,
#: and no amount of tuning gets there.  Takeoff needs a different machine --
#: far lower wing loading or far lighter -- not a better-tuned one.
#:
#: So takeoff is the binding constraint on the whole mission, and it is the same
#: shape of problem as flight was before the gannet: a capability nothing in the
#: seed set has, which the search must therefore cross a valley to reach rather
#: than improve its way to.  Adding a seed that can leave the ground is the
#: obvious answer and is deliberately not done here -- it is a change to what
#: the search starts from, and that is worth making on purpose rather than in
#: passing.
TRANSITION_ENDPOINTS: dict[str, tuple[Domain, Domain]] = {
    "air_to_water": (Domain.AIR, Domain.WATER),
    "water_to_air": (Domain.WATER, Domain.AIR),
    "water_to_land": (Domain.WATER, Domain.LAND),
    "land_to_water": (Domain.LAND, Domain.WATER),
    "land_to_air": (Domain.LAND, Domain.AIR),
    "air_to_land": (Domain.AIR, Domain.LAND),
}


@dataclass(eq=False)
class TransitionResult:
    """One crossing, measured.  Every field is a raw physical quantity or a
    normalised score in [0, 1]; nothing here is weighted."""

    kind: str
    crossed: bool = False
    failure: str = ""
    #: How far back from the interface this probe started, metres (Y/O).
    start_back: float = 0.0
    #: MuJoCo bad-qacc events during the crossing; same auto-reset channel the
    #: segment rollouts guard against (see SegmentResult.bad_qacc).
    bad_qacc: int = 0
    #: How well the machine stayed where it was told to during the hold phase,
    #: in [0, 1] (``CrossingTracker``).  The crossing counts only at
    #: ``HOLD_PASS`` or above, and the graded ``approach`` is scaled by it.
    hold: float = 0.0
    #: Share of the way from the start to the shoreline (SHORE_X) covered by the
    #: end, for a land target; graded like ``approach`` for an air target.
    shore_progress: float = 0.0
    #: The medium the placement actually started the machine in.  A probe that
    #: does not start in its start medium measures nothing (water_to_land did
    #: not, until 2026-10-03).
    started_in: str = ""

    # Raw measurements, kept so the record survives any change to the scoring.
    peak_entry_speed: float = 0.0
    survivable_entry_speed: float = 0.0
    peak_slam: float = 0.0
    min_upright: float = 1.0
    settle_seconds: float = 0.0
    energy_j: float = 0.0
    duration: float = 0.0
    exit_depth: float = 0.0
    exit_upright: float = 1.0
    exit_speed: float = 0.0
    #: Peak slamming pressure and the hull's capacity for it, Pa.
    slam_pressure: float = 0.0
    slam_capacity: float = 0.0
    #: Highest the machine's lowest geometry ever got, and the share of the
    #: episode it spent with no ground contact at all.  Raw, always recorded.
    peak_clearance: float = 0.0
    airborne_fraction: float = 0.0
    #: Share of the go phase spent clear of whatever is underneath -- the water
    #: surface or the ground -- by more than AIRBORNE_GAP.  What the graded
    #: approach to an air target reads.  ``airborne_fraction`` counts steps with
    #: no contact at all, which a body floating in water satisfies every step:
    #: it paid a still machine 0.300 on ``water_to_air`` until 2026-10-03.
    aloft_fraction: float = 0.0
    #: Highest clearance reached in the go phase.  ``peak_clearance`` covers the
    #: whole episode including where the probe *put* the machine, and the 5 cm
    #: placement gap alone paid a sitting gannet 0.031 of a takeoff.
    go_peak_clearance: float = 0.0

    #: How much of the crossing happened, in [0, 1], and 1.0 exactly when
    #: ``crossed`` is true.  This is what the fitness gate reads.
    #:
    #: It used to read ``crossed`` itself, and that was a cliff: the gate is
    #: multiplicative -- ``transition_fraction = crossed * (0.40 + 0.60 *
    #: quality)`` -- so every design that did not complete the crossing scored
    #: exactly zero for it, and they all scored the same zero.  A machine that
    #: leapt 0.566 m and came down was worth precisely as much as one that
    #: never moved, which is the same defect the buoyancy checks had before
    #: they were given graded margins, and it is fatal in the same way: the
    #: search cannot climb a quantity that does not vary.
    #:
    #: Takeoff is where it bites hardest, because ``land_to_air`` asks for
    #: clearance above half a metre *at the end of the episode* -- sustained
    #: flight, not a leap -- and a machine cannot get there in one mutation
    #: from a machine that is sitting on the ground.  Grading the approach is
    #: what makes the intermediate rungs -- leaves the ground at all, stays off
    #: it longer, comes down slower -- visible to selection, so the capability
    #: can be evolved toward instead of having to arrive complete.
    approach: float = 0.0

    # Normalised components, all higher-is-better.
    shock: float = 0.0
    control: float = 0.0
    settle: float = 0.0
    economy: float = 0.0
    exit_state: float = 0.0

    @property
    def components(self) -> dict[str, float]:
        """The scored parts, for the judge to weight."""
        return {
            # Graded: 1.0 iff the crossing actually completed, and a fraction
            # of it for an attempt that got part of the way.  See ``approach``.
            "crossed": float(self.approach),
            "shock": self.shock,
            "control": self.control,
            "settle": self.settle,
            "economy": self.economy,
            "exit_state": self.exit_state,
        }


def transition_scatter_seed(kind: str) -> int:
    """The entry-state draw every machine's ``kind`` crossing starts from.

    Stable across processes -- an index into a fixed list, never ``hash(str)``
    -- and shared by both evaluation paths.  Until 2026-09-30 only the batched
    path scattered a crossing's entry state; ``run_transition`` started every
    machine from the bare placement, so Tier-2, every film and every offline
    probe measured a different crossing from the one the search scored (peak
    entry speed differed by up to 7.3 m/s on the seed plans).
    """
    kinds = ("air_to_water", "water_to_air", "water_to_land", "land_to_air",
             "land_to_water", "air_to_land")
    return (0x9E3779B9 * (kinds.index(kind) + 1 if kind in kinds else 7)) & 0x7FFFFFFF


#: ``air_to_water`` starts level at the air launch speed this high over the sea,
#: and this far out (6 s at 30 m/s from here is still 20 m short of the beach).
AIR_TO_WATER_HEIGHT = 4.0  # m
AIR_TO_WATER_X = -200.0  # m

#: ``water_to_land`` starts with the root this deep, at the most shoreward x
#: where the machine fits above the ramp, beginning here (shore is at x ~ 8).
WATER_START_DEPTH = 0.15  # m
WATER_TO_LAND_X = 6.0  # m
#: Where the ramp meets the still waterline (``ground_height`` is 0.02 at 8).
SHORE_X = 8.0  # m

#: Every crossing is two commanded phases (ARCH46_SPEC §8): for HOLD_SECONDS the
#: controller observes the start medium and must stay in it, then it observes
#: the target and must cross.  A passive body does the same thing whatever it is
#: commanded, so it cannot both hold and then go.
HOLD_SECONDS = 1.5
#: The hold score a crossing needs in order to count at all.
HOLD_PASS = 0.5
#: Holding in air is holding height (``tasks.py``: "almost nothing here can
#: hover"): full marks for losing at most HOLD_ALT_FULL over the hold, nothing
#: at HOLD_ALT_ZERO.  Measured 2026-10-03, launched level at trim with the
#: actuators still, the seed plans lose 2.6 to 9.5 m in 1.5 s.
HOLD_ALT_FULL = 0.5  # m
HOLD_ALT_ZERO = 1.5  # m


#: How close to the ground counts as on it when classifying the start and the
#: land hold: every placement leaves a few centimetres so the machine does not
#: start inside the terrain.
GROUND_TOL = 0.10  # m
#: Clear of the water surface or the ground by more than this counts as aloft.
AIRBORNE_GAP = 0.05  # m


def medium_of(env: TriphibianEnv, ground_tol: float = 0.0) -> Domain:
    """Which medium the machine is in: root under water, else on the ground
    (or within ``ground_tol`` of it) with the root dry, else in the air."""
    if env.depth() > 0.0:
        return Domain.WATER
    if env._touching_ground() or (ground_tol > 0.0 and env.clearance() <= ground_tol):
        return Domain.LAND
    return Domain.AIR


def _rock_gap(env: TriphibianEnv) -> float:
    """Lowest machine geometry above the beach ramp, metres, at the current pose.

    Not ``clearance``: that is height above whatever is underneath, which over
    the sea is the water surface, so a submerged start never "clears" it.
    """
    from ..core.mjcf import beach_surface_z

    g = env._machine_geoms
    aabb = env.model.geom_aabb.reshape(-1, 6)[g]
    R = env.data.geom_xmat[g].reshape(-1, 3, 3)
    centre_z = env.data.geom_xpos[g][:, 2] + np.einsum("nij,nj->ni", R, aabb[:, :3])[:, 2]
    bottom = centre_z - np.einsum("nj,nj->n", np.abs(R[:, 2, :]), aabb[:, 3:])
    rock = np.array([beach_surface_z(float(x)) for x in env.data.geom_xpos[g][:, 0]])
    return float(np.min(bottom - rock))


class CrossingTracker:
    """What a crossing is, shared by ``run_transition`` and the batched path.

    Until 2026-10-03 each path kept its own copy of "a crossing happened": the
    first step at which the root's wetness differed from the start's, in either
    direction, or touching ground at the end for a land target.  So
    ``water_to_land``, which started the machine dry on the ramp, credited a
    still machine for settling *into* the water, and ``air_to_water`` paid a
    body for falling.  Measured with the actuators held still over 7 plans x 2
    seeds: 0.834 against the base gait's 0.693, and 0.865 against 0.861.

    Now:

    - it must start in the start medium (``started_in``), or nothing counts;
    - the first HOLD_SECONDS are a *hold* phase: the controller observes the
      start medium (``commanded``) and is scored on staying in it -- holding
      height, in air;
    - only then does it observe the target, and only a change *into* the target
      medium after that counts, and only if it is still there at the end.
      ``land_to_air`` and ``water_to_air`` keep their terminal clearance test.
    """

    def __init__(self, env: TriphibianEnv, kind: str):
        self.start, self.target = TRANSITION_ENDPOINTS[kind]
        self.hold_steps = max(int(round(HOLD_SECONDS / env.timestep)), 1)
        self.started_in = medium_of(env, GROUND_TOL)
        self.z0 = float(env.root_pos()[2])
        self.x0 = float(env.root_pos()[0])
        self.z_min = self.z0
        self.in_start = 0
        self.held_steps = 0
        self.left_early = False
        self.cross_step = -1
        self.entry_speed = 0.0
        self.go_steps = 0
        self.aloft_steps = 0
        self.go_peak = 0.0

    def commanded(self, i: int) -> Domain:
        """The medium the controller is told about at step ``i``."""
        return self.start if i < self.hold_steps else self.target

    def observe(self, env: TriphibianEnv, i: int) -> None:
        m = medium_of(env)
        if i < self.hold_steps:
            self.held_steps += 1
            if self.start is Domain.AIR:
                if m is Domain.AIR:
                    self.in_start += 1
                    self.z_min = min(self.z_min, float(env.root_pos()[2]))
                else:
                    self.left_early = True
            elif self.start is Domain.WATER:
                # Bobbing at the surface is still in the water; on land, or
                # clear of it by half a metre, is not.
                if m is not Domain.LAND and env.clearance() <= 0.5:
                    self.in_start += 1
                else:
                    self.left_early = True
            else:
                if medium_of(env, GROUND_TOL) is Domain.LAND:
                    self.in_start += 1
                elif env.clearance() > 0.5 or m is Domain.WATER:
                    self.left_early = True
            return
        self.go_steps += 1
        cl = float(env.clearance())
        self.go_peak = max(self.go_peak, cl)
        if cl > AIRBORNE_GAP:
            self.aloft_steps += 1
        if self.cross_step < 0 and self.target is not Domain.AIR and m is self.target:
            self.cross_step = i
            self.entry_speed = abs(float(env.body_twist()[2]))

    def hold_score(self) -> float:
        if self.started_in is not self.start or self.left_early or not self.held_steps:
            return 0.0
        stayed = self.in_start / self.held_steps
        if self.start is Domain.AIR:
            loss = self.z0 - self.z_min
            height = float(np.clip((HOLD_ALT_ZERO - loss) / (HOLD_ALT_ZERO - HOLD_ALT_FULL),
                                   0.0, 1.0))
            return float(stayed * height)
        return float(stayed)

    def finish(self, env: TriphibianEnv, r: TransitionResult, n_steps: int) -> int:
        """Set ``r.hold``, ``r.started_in``, ``r.crossed``; return the cross step."""
        r.hold = self.hold_score()
        r.started_in = self.started_in.value
        r.aloft_fraction = self.aloft_steps / max(self.go_steps, 1)
        r.go_peak_clearance = self.go_peak
        if self.target is Domain.LAND and SHORE_X > self.x0:
            r.shore_progress = float(np.clip(
                (float(env.root_pos()[0]) - self.x0) / (SHORE_X - self.x0), 0.0, 1.0))
        cross = self.cross_step
        if self.target is Domain.AIR:
            # Terminal, not peak: a machine that leapt and came down has not
            # crossed.
            cross = max(n_steps - 1, 0) if env.clearance() > 0.5 else -1
        elif cross >= 0 and medium_of(env) is not self.target:
            # Held: it has to be there at the end, not have passed through.
            cross = -1
        if cross >= 0:
            r.peak_entry_speed = max(r.peak_entry_speed, self.entry_speed)
        r.crossed = cross >= 0 and not r.failure and r.hold >= HOLD_PASS
        if not r.failure:
            if self.started_in is not self.start:
                r.failure = f"did not start in {self.start.value}"
            elif cross >= 0 and r.hold < HOLD_PASS:
                r.failure = "did not hold before the command to cross"
        return cross if r.crossed else -1


def _place_for(env: TriphibianEnv, kind: str, back: float = 0.0) -> None:
    """Put the machine where the crossing begins.

    ``back`` moves the start away from the interface, metres (ROADMAP Y/O):
    higher over the sea, deeper under it, further seaward of the ramp, further
    up the beach.  ``land_to_air`` has no interface to step back from and
    ignores it.  0 is the placement every run before Y/O used.
    """
    start, _ = TRANSITION_ENDPOINTS[kind]
    env.reset(start, randomise=False)
    if env.model.nq < 7:
        return
    back = max(float(back), 0.0)
    if kind == "air_to_water":
        # Level, at the air segment's launch speed, AIR_TO_WATER_HEIGHT over the
        # sea, and far enough out that 6 s at 30 m/s is still over water.  It
        # used to be a committed descent from 2.5 m at -1.5 m/s -- "the machine
        # has to arrive at the surface, not decide whether to" -- and a machine
        # with its actuators held still crossed 100% of the time and scored
        # 0.834 against the base gait's 0.693 (2026-10-03).  Now it must hold
        # height until told to go (``CrossingTracker``), which no passive body
        # can: launched level at trim, every seed plan held still loses at
        # least 2.6 m in the 1.5 s hold.
        env.data.qpos[0] = AIR_TO_WATER_X
        env.data.qpos[2] = AIR_TO_WATER_HEIGHT + back
    elif kind == "water_to_air":
        env.data.qpos[2] = -2.0 - back
    elif kind == "water_to_land":
        # Floating just above the submerged part of the ramp, a few metres
        # seaward of the shoreline: a machine that has swum up to the beach and
        # now has to get out of the water.
        #
        # The obvious placement -- a fixed depth below the surface -- puts the
        # machine *inside* the ramp, because the ramp is already above that
        # depth this close in.  That is the same mistake that made the land
        # domain unreachable for the whole project, so the height comes from the
        # terrain here too rather than from a constant.
        #
        # And *in* the water: the root submerged.  Until 2026-10-03 the height
        # put the machine's lowest geometry 0.02 m above the ramp at x = 8, which
        # for a tall body is standing in 20 cm of water with its root dry -- on
        # land already -- and a still machine was credited with the crossing for
        # settling back in.  So start where the root fits under
        # WATER_START_DEPTH without touching the ramp, walking seaward from
        # WATER_TO_LAND_X until it does: a deeper hull starts further out.
        x = WATER_TO_LAND_X
        while x > WATER_TO_LAND_X - 12.0:
            env.data.qpos[0], env.data.qpos[2] = x, -WATER_START_DEPTH
            env._mj.mj_forward(env.model, env.data)
            if _rock_gap(env) >= 0.05:
                break
            x -= 0.5
        env.data.qpos[0] = x - back
        env.data.qpos[2] = -WATER_START_DEPTH
    elif kind == "land_to_water":
        env.data.qpos[0] = 14.0 + back
        if back > 0.0:
            # Up the beach the ground is higher; set down on it, not in it.
            env.data.qpos[2] = env._clear_of_terrain(14.0 + back, 0.0,
                                                     float(env.data.qpos[2]))
    env._mj.mj_forward(env.model, env.data)


def reseat_after_scatter(env: TriphibianEnv, kind: str) -> None:
    """Put a land start back on the ground at the pose ``scatter`` left.

    ``scatter`` sets every actuated joint to the gait's pose at a random phase,
    after ``_place_for`` placed the machine with its joints at zero.  For the
    teal that left the lowest geometry 0.5 m off the ground, so every
    ``land_to_air`` began with a half-metre drop -- which the probe read as a
    leap -- and the start was not on land at all.  A submerged start next to
    the ramp has the opposite problem: the ray's scattered pose put its lowest
    geometry 8 cm into the ramp, so it is stepped seaward until it clears.
    Called by both paths.
    """
    if env.model.nq < 7:
        return
    start, _ = TRANSITION_ENDPOINTS[kind]
    q = env.data.qpos
    if start is Domain.LAND:
        q[2] = env._clear_of_terrain(float(q[0]), float(q[1]), float(q[2]), gap=0.02)
        env._mj.mj_forward(env.model, env.data)
    elif kind == "water_to_land":
        for _ in range(40):
            if _rock_gap(env) >= 0.02:
                break
            q[0] -= 0.25
            env._mj.mj_forward(env.model, env.data)


def run_transition(
    env: TriphibianEnv,
    kind: str,
    controller,
    *,
    duration: float = 6.0,
    back: float = 0.0,
) -> TransitionResult:
    """Simulate one crossing and measure it.  ``back``: see ``_place_for``."""
    r = TransitionResult(kind=kind, duration=duration, start_back=float(back))
    if kind not in TRANSITION_ENDPOINTS:
        r.failure = f"unknown transition {kind}"
        return r

    _, target = TRANSITION_ENDPOINTS[kind]
    _place_for(env, kind, back)
    # The same entry-state draw the batched path makes (see
    # ``transition_scatter_seed``).
    env.scatter(np.random.default_rng(transition_scatter_seed(kind)))
    reseat_after_scatter(env, kind)
    r.survivable_entry_speed = float(env.p.max_entry_speed)

    basis = controller.basis_for(Domain.WATER if "water" in kind else Domain.AIR)
    n = int(duration / env.timestep)
    control_every = max(1, int(1.0 / (25.0 * env.timestep)))
    cur = controller.params

    energy0 = float(env.budget.total_j)
    track = CrossingTracker(env, kind)
    uprights: list[float] = []
    speeds: list[float] = []
    slam_window: list[float] = []
    slam_n = max(int(0.010 / env.timestep), 1)
    airborne_steps = 0
    r.peak_clearance = float(env.clearance())

    for i in range(n):
        if controller.policy is not None and basis is not None and i % control_every == 0:
            cur, _, _g = basis.command_policy(
                controller.params,
                controller.policy.act(env.observation(track.commanded(i))),
                env.cpg.n,
                controller.policy,
            )
        if not env.step(env.cpg.command(cur, env.data.time)):
            r.failure = "battery exhausted mid-transition"
            break
        pos = env.root_pos()
        if not np.all(np.isfinite(pos)) or np.abs(pos).max() > 400:
            r.failure = "diverged"
            break

        up = float(env.data.xmat[env.root_body].reshape(3, 3)[2, 2])
        uprights.append(up)
        speeds.append(float(np.linalg.norm(env.body_twist()[:3])))
        r.min_upright = min(r.min_upright, up)
        # How far off the ground it ever got, and how long it stayed off it.
        # Recorded for every crossing, scored only where it means something.
        clear_now = float(env.clearance())
        if clear_now > r.peak_clearance:
            r.peak_clearance = clear_now
        if int(env.data.ncon) == 0:
            airborne_steps += 1
        # Slam over a short window, not a single step.
        #
        # ``diag.slam`` is a one-step finite difference of the entrained mass,
        # so as a number it is sharp, timestep-dependent and dominated by
        # whichever step happens to straddle the surface.  A shell does not
        # respond to that: it responds over its own natural period, and an
        # impulse far shorter than that period does not load it.  Averaging over
        # 10 ms -- the order of a PETG shell's first mode at this size -- is
        # both the physically meaningful load and a far less noisy estimator.
        slam_window.append(float(env.solver.diag.slam))
        if len(slam_window) > slam_n:
            slam_window.pop(0)
        if len(slam_window) == slam_n:
            r.peak_slam = max(r.peak_slam, float(np.mean(slam_window)))

        # Vertical speed at the moment of crossing is what the hull sees.
        track.observe(env, i)

    cross_step = track.finish(env, r, len(uprights))
    r.airborne_fraction = airborne_steps / max(len(uprights), 1)
    r.energy_j = float(env.budget.total_j - energy0)
    r.exit_depth = float(env.depth())
    r.exit_upright = float(uprights[-1]) if uprights else 0.0
    r.exit_speed = float(speeds[-1]) if speeds else 0.0

    _score(env, r, cross_step, np.array(uprights), np.array(speeds), target)
    if not r.crossed and not r.failure:
        r.failure = "never crossed the boundary"
    return r


def _score(
    env: TriphibianEnv,
    r: TransitionResult,
    cross_step: int,
    uprights: np.ndarray,
    speeds: np.ndarray,
    target: Domain,
) -> None:
    """Turn the raw measurements into the six normalised components."""
    if not r.crossed:
        # A failed crossing is not automatically worth nothing.  For an air
        # target the bar is terminal clearance above 0.5 m, and the distance
        # from "sitting on the ground" to that is far more than one mutation --
        # so the approach is graded, and the intermediate rungs become
        # something selection can see and climb.
        #
        # Two measurements, because either one alone is gameable.  Peak
        # clearance alone rewards a single ballistic hop that lands
        # immediately; airborne fraction alone rewards a machine that is
        # never quite touching the ground while going nowhere.  Together they
        # ask for height *and* time, which is what a takeoff is.
        #
        # Capped at 0.6 so that completing the crossing is always strictly
        # better than any approach to it.  The gate is multiplicative, so this
        # scales the whole transition score rather than adding to it: a leap
        # cannot out-earn a flight, it can only stop being worth zero.
        if target is Domain.AIR:
            height = float(np.clip(r.go_peak_clearance / 0.5, 0.0, 1.0))
            aloft = float(np.clip(r.aloft_fraction, 0.0, 1.0))
            r.approach = float(0.6 * np.clip(0.5 * height + 0.5 * aloft, 0.0, 1.0)
                               * r.hold)
        elif target is Domain.LAND:
            # The same reason for a land target: from the start to the shore is
            # 3 to 7.5 m, more than one mutation of swimming speed covers in the
            # go phase, so the distance closed is graded, scaled by the hold and
            # capped below a completed crossing.
            r.approach = float(0.6 * r.shore_progress * r.hold)
        return
    # 1.0 for a crossing that held perfectly first; never below HOLD_PASS.
    r.approach = float(r.hold)

    # --- shock ------------------------------------------------------------
    # The *hydrodynamic* slam load, not the speed of the machine's centre.
    #
    # ``diag.slam`` is |d(m_add)/dt . v_n|, the von Karman-Wagner slamming
    # force: the rate at which the body entrains fluid, times how fast it is
    # driving into it.  It was already being computed and recorded, and the
    # score was ignoring it in favour of the root body's vertical speed.
    #
    # That was unfair to exactly the designs worth finding.  A gannet enters at
    # 24 m/s and survives because it enters *nose first*, so the wetted area
    # grows slowly and dm/dt stays small; a flat hull at 6 m/s wets all at once
    # and is destroyed.  Scoring on centre-of-mass speed cannot tell those
    # apart, and it penalises the fast elegant entry more.  Scoring on the slam
    # load reads attitude, slenderness, deadrise and structural compliance for
    # free, because all four change dm/dt and all four are already in the
    # dynamics.
    #
    # Compared as a pressure against the hull's own membrane capacity, since
    # that is what breaks a shell.
    area = max(float(getattr(env.p, "frontal_area", 0.0)), 1e-3)
    capacity = max(float(getattr(env.p, "slam_pressure_capacity", 1e5)), 1e3)
    r.slam_pressure = float(r.peak_slam / area)
    r.slam_capacity = capacity
    util = r.slam_pressure / capacity
    r.shock = float(np.clip(1.0 - util**2, 0.0, 1.0))

    # --- control ----------------------------------------------------------
    # Worst attitude through the crossing.  A machine that goes past 90 degrees
    # has been thrown rather than flown, so that is where this reaches zero.
    r.control = float(np.clip(r.min_upright, 0.0, 1.0))

    # --- settle -----------------------------------------------------------
    # Time from the boundary until the attitude stops changing much.  Measured
    # rather than assumed: this is the part of a crossing that eats the next
    # leg, and no static analysis can predict it.
    after = uprights[cross_step:]
    if len(after) > 10:
        window = max(len(after) // 8, 5)
        settled_at = len(after)
        for i in range(len(after) - window):
            if float(np.std(after[i : i + window])) < 0.05:
                settled_at = i
                break
        r.settle_seconds = settled_at * env.timestep
    else:
        r.settle_seconds = r.duration
    # Two seconds is a slow but real crossing; beyond four it has not settled.
    r.settle = float(np.clip(1.0 - (r.settle_seconds - 1.0) / 3.0, 0.0, 1.0))

    # --- economy ----------------------------------------------------------
    # Against the machine's own hotel load for the same time, so a big machine
    # is not punished for being big.  Ten times idle is a hard crossing; a
    # hundred times is a machine throwing its whole battery at the problem.
    idle = max(env.budget.mean_power, 1.0) * r.duration
    ratio = r.energy_j / max(idle, 1e-6)
    r.economy = float(np.clip(1.0 - (ratio - 1.0) / 9.0, 0.0, 1.0))

    # --- exit state -------------------------------------------------------
    # Is this a state the next leg could start from?
    up = float(np.clip(r.exit_upright, 0.0, 1.0))
    if target is Domain.WATER:
        # Submerged, not bobbing on the surface half in and half out.
        placed = float(np.clip(r.exit_depth / 1.0, 0.0, 1.0))
    elif target is Domain.AIR:
        placed = float(np.clip(env.clearance() / 2.0, 0.0, 1.0))
    else:
        placed = 1.0 if env._touching_ground() else 0.0
    r.exit_state = float(0.5 * up + 0.5 * placed)


@dataclass(eq=False)
class TransitionSet:
    """All crossings attempted in one evaluation."""

    results: dict[str, TransitionResult] = field(default_factory=dict)

    @property
    def crossed_fraction(self) -> float:
        if not self.results:
            return 0.0
        return sum(1.0 for r in self.results.values() if r.crossed) / len(self.results)

    def component_means(self) -> dict[str, float]:
        """Mean of each component across the crossings that happened.

        Crossings that failed contribute zero to every component, so a design
        cannot raise its average by refusing to attempt the hard one.
        """
        if not self.results:
            return {k: 0.0 for k in
                    ("crossed", "shock", "control", "settle", "economy", "exit_state")}
        keys = next(iter(self.results.values())).components.keys()
        return {
            k: float(np.mean([r.components[k] for r in self.results.values()]))
            for k in keys
        }
