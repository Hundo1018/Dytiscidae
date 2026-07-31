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
            "crossed": 1.0 if self.crossed else 0.0,
            "shock": self.shock,
            "control": self.control,
            "settle": self.settle,
            "economy": self.economy,
            "exit_state": self.exit_state,
        }


def _place_for(env: TriphibianEnv, kind: str) -> None:
    """Put the machine where the crossing begins."""
    start, _ = TRANSITION_ENDPOINTS[kind]
    env.reset(start, randomise=False)
    if env.model.nq < 7:
        return
    if kind == "air_to_water":
        # Committed descent from low altitude: the machine has to arrive at the
        # surface, not decide whether to.
        env.data.qpos[2] = 2.5
        env.data.qvel[2] = -1.5
    elif kind == "water_to_air":
        env.data.qpos[2] = -2.0
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
        x = 8.0
        env.data.qpos[2] = env._clear_of_terrain(x, 0.0, 0.0, gap=0.02)
    elif kind == "land_to_water":
        env.data.qpos[0] = 14.0
    env._mj.mj_forward(env.model, env.data)


def run_transition(
    env: TriphibianEnv,
    kind: str,
    controller,
    *,
    duration: float = 6.0,
) -> TransitionResult:
    """Simulate one crossing and measure it."""
    r = TransitionResult(kind=kind, duration=duration)
    if kind not in TRANSITION_ENDPOINTS:
        r.failure = f"unknown transition {kind}"
        return r

    _, target = TRANSITION_ENDPOINTS[kind]
    _place_for(env, kind)
    r.survivable_entry_speed = float(env.p.max_entry_speed)

    basis = controller.basis_for(Domain.WATER if "water" in kind else Domain.AIR)
    n = int(duration / env.timestep)
    control_every = max(1, int(1.0 / (25.0 * env.timestep)))
    cur = controller.params

    energy0 = float(env.budget.total_j)
    was_wet = env.depth() > 0.0
    cross_step = -1
    uprights: list[float] = []
    speeds: list[float] = []
    slam_window: list[float] = []
    slam_n = max(int(0.010 / env.timestep), 1)

    for i in range(n):
        if controller.policy is not None and basis is not None and i % control_every == 0:
            cur = basis.command_params(
                controller.params,
                controller.policy.act(env.observation(target)),
                env.cpg.n,
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

        wet = env.depth() > 0.0
        if wet != was_wet:
            if cross_step < 0:
                cross_step = i
                # Vertical speed at the moment of crossing is what the hull sees.
                r.peak_entry_speed = max(
                    r.peak_entry_speed, abs(float(env.body_twist()[2]))
                )
            was_wet = wet

    # For a land crossing there is no waterline to cross, so use ground contact.
    if cross_step < 0 and target is Domain.LAND:
        if env._touching_ground():
            cross_step = max(len(uprights) - 1, 0)
    if cross_step < 0 and target is Domain.AIR and env.clearance() > 0.5:
        cross_step = max(len(uprights) - 1, 0)

    r.crossed = cross_step >= 0 and not r.failure
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
        return

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
