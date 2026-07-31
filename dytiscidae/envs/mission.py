"""The continuous mission: one unbroken simulation across all three domains.

Everything else in ``envs/`` evaluates domains separately, resetting the machine
into each one.  That is the right call for search -- it is cheap, it isolates
competence per domain, and it does not let a failure in one leg hide the others.
But it is not the mission.  The mission is *continuous*: the machine starts
somewhere, and to reach the next domain it has to physically get there, through
the surface, under its own control, carrying whatever state the previous leg
left it in.

So this module runs the schedule with no resets at all.  The commanded domain
changes on a timer and goes into the controller's observation; whether the
machine follows is what is being measured.  A design that cannot leave the water
simply spends the air leg in the water and is scored accordingly, which is the
honest result and exactly what the separate-leg evaluation cannot tell you.

Domain membership is judged from physics rather than from position:

    AIR    the machine is above the local surface and not touching ground
    WATER  more than a third of it is below the local surface
    LAND   it is in contact with terrain and not substantially submerged

so "which domain is it in" is an observation about the world, not a region of a
map, and it stays correct when the sea state moves the surface around.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .triphibian import DOMAIN_CYCLE, Domain, MissionSpec, TriphibianEnv


@dataclass
class LegRecord:
    """One commanded stretch of the mission."""

    index: int
    commanded: Domain
    start_t: float
    end_t: float = 0.0
    on_task_fraction: float = 0.0
    mean_power: float = 0.0
    max_depth: float = 0.0
    distance: float = 0.0
    entered: bool = False  # did it reach and hold the commanded domain
    entry_time: float | None = None  # seconds after the command to first arrival

    @property
    def duration(self) -> float:
        return max(self.end_t - self.start_t, 1e-9)


@dataclass
class ContinuousResult:
    """What one unbroken mission produced."""

    legs: list[LegRecord] = field(default_factory=list)
    survived: bool = True
    failure: str = ""
    total_time: float = 0.0
    energy_wh: float = 0.0
    max_depth: float = 0.0
    transitions_commanded: int = 0
    transitions_completed: int = 0
    peak_slam: float = 0.0
    #: Highest structural utilisation reached during the mission (1.0 = at the
    #: allowable stress).  The Tier-0 report checks the *design* load case; this
    #: is what the controller actually flew, and the two routinely disagree --
    #: a design can pass its static check with margin and still be torn apart by
    #: the loads its own controller commands.
    peak_stress: float = 0.0
    stress_overload_time: float = 0.0
    wall_time: float = 0.0
    #: What ``energy_wh`` was billed for, in seconds of mission.  A filmed run
    #: is time-compressed and the battery is not, so the energy is scaled up to
    #: the duration it represents -- and reporting that scaled figure next to
    #: the *unscaled* clock implied 4.7 kW for a machine drawing 190 W.
    billed_seconds: float = 0.0

    @property
    def on_task(self) -> float:
        """Fraction of the whole mission spent in the domain being asked for."""
        if not self.legs:
            return 0.0
        num = sum(leg.on_task_fraction * leg.duration for leg in self.legs)
        den = sum(leg.duration for leg in self.legs)
        return num / max(den, 1e-9)

    @property
    def transition_rate(self) -> float:
        if self.transitions_commanded == 0:
            return 0.0
        return self.transitions_completed / self.transitions_commanded

    @property
    def mean_power_w(self) -> float:
        """Mean draw, on the time base the energy was billed against."""
        secs = self.billed_seconds or self.total_time
        return self.energy_wh * 3600.0 / max(secs, 1e-9)

    def summary(self) -> str:
        return (
            f"on-task {self.on_task*100:.0f}%   "
            f"transitions {self.transitions_completed}/{self.transitions_commanded}   "
            f"max depth {self.max_depth:.1f} m   "
            f"peak stress {self.peak_stress*100:.0f}%   "
            + _energy_phrase(self)
            + (f"   FAILED: {self.failure}" if not self.survived else "")
        )


def _energy_phrase(r: "ContinuousResult") -> str:
    """Energy and the duration it was actually billed for."""
    if r.billed_seconds and abs(r.billed_seconds - r.total_time) > 1e-6:
        return (f"{r.energy_wh:.1f} Wh over {r.billed_seconds:.0f} s of mission "
                f"({r.mean_power_w:.0f} W, filmed in {r.total_time:.0f} s)")
    return f"{r.energy_wh:.1f} Wh over {r.total_time:.0f} s ({r.mean_power_w:.0f} W)"


#: Arrival is judged over a window, not on one step and not on an unbroken run.
#:
#: A transition used to count the first single step on which the actual domain
#: matched the commanded one, and one step is not a crossing.  ``current_domain``
#: reads AIR whenever nothing is touching and nothing is submerged, so a walking
#: machine reads AIR every time it lifts its feet: measured on a land episode,
#: the beetle reads airborne for 25.7% of steps and the gannet for 1.7%, in
#: bursts of 0.07 to 0.14 s.  Any of those bursts completed the land-to-air
#: transition, which is worth 35% of the continuous mission score across two
#: crossings -- so a design that never left the ground could bank half of it on
#: contact chatter.
#:
#: Requiring an *unbroken* half second instead fixes that and breaks something
#: else: a gait with an aerial phase is a real gait, and the beetle -- on the
#: ground for 74% of a land leg -- never held it unbroken for half a second, so
#: it never counted as having arrived anywhere.  Both errors are the same
#: mistake, treating an instantaneous reading as a state.
#:
#: So: in the domain for a majority of a half-second window.  Chatter at 26%
#: does not qualify; a runner at 74% does.
#:
#: Scope, because it is the first thing to ask about a scoring bug: this module
#: is used by controller training and by the showcase film, and by nothing else.
#: The search never saw it.  Tier 1 scores crossings in ``envs/transitions.py``,
#: which watches the waterline actually being crossed and demands half a metre
#: of clearance before calling something airborne; Tier 2 re-places the machine
#: for each leg and does not score transitions at all.  Both were checked.
ENTRY_WINDOW = 0.5
ENTRY_MAJORITY = 0.6


def current_domain(env: TriphibianEnv) -> Domain:
    """Which domain the machine is physically in right now."""
    submerged = env.solver.diag.mean_submerged
    touching = env.data.ncon > 0
    if submerged > 0.34:
        return Domain.WATER
    if touching:
        return Domain.LAND
    return Domain.AIR


def build_schedule(spec: MissionSpec, rng: np.random.Generator,
                   leg_seconds: float | None = None) -> list[tuple[Domain, float]]:
    """Random starting domain, then cycle, as specified."""
    start = int(rng.integers(len(DOMAIN_CYCLE)))
    secs = leg_seconds if leg_seconds is not None else spec.seconds_per_domain
    return [
        (DOMAIN_CYCLE[(start + i) % len(DOMAIN_CYCLE)], secs)
        for i in range(len(DOMAIN_CYCLE) * spec.cycles)
    ]


def run_continuous(
    env: TriphibianEnv,
    controller,
    schedule: list[tuple[Domain, float]],
    *,
    on_step=None,
    control_hz: float = 25.0,
    energy_scale: float = 1.0,
    stress_probe=None,
) -> ContinuousResult:
    """Fly the whole schedule without ever resetting.

    ``energy_scale`` multiplies the charged energy so a time-compressed run can
    still be billed for the real mission duration: the dynamics are shortened,
    the battery is not.

    ``on_step(env, commanded, actual, t)`` is called every step, which is how
    the renderer gets its frames without this module knowing anything about
    rendering.
    """
    t0 = time.time()
    result = ContinuousResult()
    if not schedule:
        return result

    # Start in the first commanded domain -- the only placement in the run.
    env.reset(schedule[0][0], randomise=False)
    params = controller.params if controller is not None else env.cpg.base
    ctrl_every = max(1, int(1.0 / (control_hz * env.timestep)))

    prev_actual = current_domain(env)
    step_i = 0
    clock = 0.0

    for leg_i, (commanded, seconds) in enumerate(schedule):
        leg = LegRecord(index=leg_i, commanded=commanded, start_t=clock)
        basis = controller.basis_for(commanded) if controller is not None else None
        cur = params
        on_task_steps = 0
        leg_steps = 0
        start_xy = env.root_pos()[:2].copy()
        power_acc = 0.0
        window: deque = deque()
        window_n = max(int(ENTRY_WINDOW / env.timestep), 1)

        if leg_i > 0:
            result.transitions_commanded += 1

        n = int(seconds / env.timestep)
        for _ in range(n):
            if controller is not None and controller.policy is not None \
                    and basis is not None and step_i % ctrl_every == 0:
                coeffs = controller.policy.act(env.observation(commanded))
                cur = basis.command_params(params, coeffs, env.cpg.n)

            alive = env.step(env.cpg.command(cur, env.data.time))
            pos = env.root_pos()
            if not np.all(np.isfinite(pos)) or np.abs(pos).max() > 400.0:
                result.survived = False
                result.failure = "diverged"
                break
            if not alive:
                result.survived = False
                result.failure = "battery exhausted"
                break

            actual = current_domain(env)
            in_domain = actual is commanded
            if in_domain:
                on_task_steps += 1
            # Time spent in the domain is counted every step -- that is a
            # fraction of time and chatter averages out of it.  *Arriving* is
            # judged over a window; see ENTRY_WINDOW.
            window.append(in_domain)
            if len(window) > window_n:
                window.popleft()
            if (not leg.entered and len(window) == window_n
                    and sum(window) >= ENTRY_MAJORITY * window_n):
                leg.entered = True
                leg.entry_time = max(clock - leg.start_t - ENTRY_WINDOW, 0.0)
                if leg_i > 0:
                    result.transitions_completed += 1
            leg_steps += 1
            step_i += 1
            clock += env.timestep
            power_acc += env.budget.mean_power
            result.max_depth = max(result.max_depth, env.depth())
            result.peak_slam = max(result.peak_slam, env.solver.diag.slam)
            if stress_probe is not None and step_i % 8 == 0:
                util = stress_probe.utilisation()
                if util:
                    peak = max(util.values())
                    result.peak_stress = max(result.peak_stress, peak)
                    if peak > 1.0:
                        result.stress_overload_time += env.timestep * 8
            prev_actual = actual

            if on_step is not None:
                on_step(env, commanded, actual, clock, step_i)

        leg.end_t = clock
        leg.on_task_fraction = on_task_steps / max(leg_steps, 1)
        leg.mean_power = power_acc / max(leg_steps, 1)
        leg.distance = float(np.linalg.norm(env.root_pos()[:2] - start_xy))
        leg.max_depth = result.max_depth
        result.legs.append(leg)
        if not result.survived:
            break

    result.total_time = clock
    result.energy_wh = env.budget.total_j / 3600.0 * energy_scale
    result.billed_seconds = clock * energy_scale
    result.wall_time = time.time() - t0
    return result


def continuous_score(result: ContinuousResult, spec: MissionSpec) -> float:
    """A single number for the continuous mission, in [0, 1].

    Weighted toward the two things the separate-leg evaluation cannot see:
    whether the machine actually *gets* to each domain, and whether it can do
    the whole sequence without dying somewhere in the middle.
    """
    if not result.legs:
        return 0.0
    depth_term = float(np.clip(result.max_depth / max(spec.target_depth, 1e-6), 0.0, 1.0))
    survive = 1.0 if result.survived else 0.35

    # Structural honesty.  A mission flown at three times the allowable stress
    # is not a completed mission, it is a machine that broke and kept going
    # because the simulator has no failure model.  Scoring it as a success is
    # exactly the kind of thing the optimiser will find and exploit, so the
    # penalty is steep and starts the moment the allowable is exceeded.
    if result.peak_stress > 1.0:
        survive *= float(np.clip(1.0 / result.peak_stress, 0.15, 1.0))

    return float(
        np.clip(
            (0.45 * result.on_task + 0.35 * result.transition_rate + 0.20 * depth_term)
            * survive,
            0.0,
            1.0,
        )
    )
