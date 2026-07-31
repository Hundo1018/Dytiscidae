"""The actuator skill bench: learning to operate components, not vehicles.

The mission environment asks "can this machine do the job".  This one asks the
prior question: "can anything learn to drive this component well".  They are
separated because component-level control is cheap to simulate (milliseconds,
no MuJoCo, no fluid solver), can be learned to convergence in seconds, and the
resulting skills transfer to every morphology that carries the same component.

Six tasks.  Three are the obvious ones the user named -- motors, pumps,
electromagnetic and vibrating elements.  Three are ones that turn out to matter
more than they look:

* **Resonance seeking** is the highest-value actuator skill in the whole
  project.  A compliant flapping wing driven at its resonant frequency costs a
  fraction of the power of the same wing driven off-resonance, because the
  elastic element recovers the inertial energy of stroke reversal instead of the
  motor paying for it twice per cycle.  The resonant frequency depends on the
  medium, so it *changes when the machine enters water* -- a controller that
  cannot re-find it will burn its battery in minutes.

* **Depth holding against a compressing gas store** is unstable open-loop.  Gas
  compresses with depth, so buoyancy falls as the machine sinks, so it sinks
  faster.  There is no passive fix; it has to be actively controlled, and the
  pump work scales with depth.

* **Thermal management** is what actually limits duty cycle.  Copper loss goes
  as torque squared, so a controller that meets its target by saturating the
  motor wins for thirty seconds and then has no actuator at all.

Every task exposes the same interface, so the same optimiser trains all of them,
and every learned skill is saved with the observation it needs so the mission
controller can call it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..physics.energy import Actuator
from ..physics.materials import SHAFT_SEAL_FRICTION
from ..physics.medium import GRAVITY, P_ATM, SEAWATER
from ..physics.structure import ballast_pump_power


# --------------------------------------------------------------------------
# Component models
# --------------------------------------------------------------------------


@dataclass
class BuoyancyEngine:
    """A reciprocating pump moving water between a tank and the sea.

    ``volume`` is the water currently held (positive = heavier).  Pump work is
    done against ambient pressure, so the same displacement costs ten times as
    much at 100 m as at 10 m -- and a controller that has learned to trim at the
    surface will find it cannot afford the same behaviour deep.
    """

    capacity: float = 0.004  # m^3
    max_flow: float = 8e-5  # m^3/s
    efficiency: float = 0.35
    volume: float = 0.0

    def step(self, command: float, depth: float, dt: float) -> float:
        """Move water; returns electrical energy spent, J."""
        flow = float(np.clip(command, -1.0, 1.0)) * self.max_flow
        new = float(np.clip(self.volume + flow * dt, 0.0, self.capacity))
        moved = abs(new - self.volume)
        self.volume = new
        # Pumping *out* against pressure costs; letting water in is nearly free.
        if flow < 0:
            return ballast_pump_power(depth, moved / max(dt, 1e-9), self.efficiency) * dt
        return 0.05 * ballast_pump_power(depth, moved / max(dt, 1e-9), self.efficiency) * dt


@dataclass
class GasStore:
    """A compressible gas bladder.  Boyle's law, which is the whole problem."""

    volume_surface: float = 0.003  # m^3 at 1 atm

    def volume_at(self, depth: float) -> float:
        return self.volume_surface * P_ATM / (P_ATM + SEAWATER.rho * GRAVITY * max(depth, 0.0))


@dataclass
class Solenoid:
    """An electromagnetic actuator: fast, binary-ish, and expensive to hold.

    Modelled with the two properties that matter operationally: force falls off
    as the inverse square of the gap (so it is weak until it is nearly closed,
    then violent), and holding costs continuous current with no mechanical
    output at all.  A controller that learns to pulse-and-latch rather than hold
    saves almost all of the energy.
    """

    force_closed: float = 40.0  # N at zero gap
    stroke: float = 0.006  # m
    resistance: float = 8.0  # ohm
    voltage: float = 12.0
    gap: float = 0.006
    velocity: float = 0.0
    mass: float = 0.02
    spring_k: float = 3000.0

    def step(self, duty: float, dt: float) -> float:
        duty = float(np.clip(duty, 0.0, 1.0))
        rel_gap = max(self.gap / self.stroke, 0.02)
        f_mag = self.force_closed * duty / rel_gap**2
        f_spring = self.spring_k * (self.stroke - self.gap)
        acc = (-f_mag + f_spring) / self.mass
        self.velocity += acc * dt
        self.gap = float(np.clip(self.gap + self.velocity * dt, 0.0, self.stroke))
        if self.gap in (0.0, self.stroke):
            self.velocity = 0.0
        return duty * self.voltage**2 / self.resistance * dt


@dataclass
class PiezoVibrator:
    """A resonant vibrating element: an antifouling/silt-shedding actuator.

    Included because it is the kind of component that is easy to bolt on and
    hard to operate: it is only useful within a percent or two of its resonance,
    and its resonance shifts with load and with whether it is wet.  Learning to
    track that shift is a real skill and is exactly the sort of thing the user
    asked to be surprised by.
    """

    f0: float = 480.0  # Hz, dry resonance
    q_factor: float = 45.0
    coupling: float = 1.0

    def response(self, drive_hz: float, wetted: float) -> float:
        """Vibration amplitude for a unit drive, normalised."""
        # Added fluid mass drags the resonance down and damps it heavily.
        f_res = self.f0 * (1.0 - 0.28 * wetted)
        q = self.q_factor * (1.0 - 0.75 * wetted)
        r = drive_hz / max(f_res, 1e-6)
        denom = math.sqrt((1 - r**2) ** 2 + (r / max(q, 1e-6)) ** 2)
        return self.coupling / max(denom, 1e-9)


@dataclass
class CompliantWingDrive:
    """A motor driving an inertial load through a spring: the flapping drivetrain.

    This is a second-order system.  Driven at its resonance the spring returns
    the wing's kinetic energy at each reversal and the motor only pays for
    aerodynamic damping; driven away from it the motor pays for the full
    inertial reversal twice per cycle.  The resonance is
    ``sqrt(k / J) / 2 pi``, and ``J`` includes the fluid's added mass, so it
    drops by an order of magnitude the moment the wing is submerged.
    """

    inertia_dry: float = 0.06  # kg m^2
    stiffness: float = 12.0  # N m / rad
    damping_air: float = 0.05
    damping_water: float = 2.4
    added_inertia_water: float = 1.9  # kg m^2 when fully submerged

    angle: float = 0.0
    rate: float = 0.0

    def inertia(self, wetted: float) -> float:
        return self.inertia_dry + wetted * self.added_inertia_water

    def resonance_hz(self, wetted: float) -> float:
        return math.sqrt(self.stiffness / self.inertia(wetted)) / (2 * math.pi)

    def step(self, torque: float, wetted: float, dt: float) -> None:
        j = self.inertia(wetted)
        c = self.damping_air + wetted * (self.damping_water - self.damping_air)
        acc = (torque - c * self.rate - self.stiffness * self.angle) / j
        self.rate += acc * dt
        self.angle += self.rate * dt


# --------------------------------------------------------------------------
# Task framework
# --------------------------------------------------------------------------


@dataclass
class SkillResult:
    score: float
    energy_j: float
    detail: dict = field(default_factory=dict)


class SkillTask:
    """Base class.  ``obs_dim`` and ``act_dim`` define the policy shape."""

    name = "task"
    obs_dim = 4
    act_dim = 1
    duration = 20.0
    dt = 0.01
    #: One sentence explaining what competence at this task buys the vehicle.
    why = ""

    def run(self, policy, rng: np.random.Generator) -> SkillResult:  # pragma: no cover
        raise NotImplementedError


class DepthHold(SkillTask):
    """Hold a commanded depth using a buoyancy engine against a gas store."""

    name = "depth_hold"
    obs_dim = 5
    act_dim = 1
    duration = 90.0
    dt = 0.05
    why = "Open-loop depth is unstable: carried gas compresses, so sinking makes it sink faster."

    def run(self, policy, rng) -> SkillResult:
        target = float(rng.uniform(4.0, 12.0))
        dry_mass = 6.0
        rigid_volume = 0.0072
        engine = BuoyancyEngine(capacity=0.0035)
        gas = GasStore(volume_surface=float(rng.uniform(0.001, 0.004)))
        engine.volume = engine.capacity * 0.5

        depth, vz = float(rng.uniform(1.0, 8.0)), 0.0
        energy = 0.0
        err_acc, n = 0.0, 0
        drag_c = 55.0

        for _ in range(int(self.duration / self.dt)):
            disp = rigid_volume + gas.volume_at(depth)
            mass = dry_mass + engine.volume * SEAWATER.rho
            net = SEAWATER.rho * GRAVITY * disp - mass * GRAVITY
            acc = (net - drag_c * vz * abs(vz)) / max(mass, 1e-3)
            vz += acc * self.dt
            depth = max(depth - vz * self.dt, 0.0)

            obs = np.array([
                np.tanh((depth - target) / 4.0), np.tanh(vz / 0.6),
                depth / 15.0, engine.volume / engine.capacity, 1.0,
            ])
            cmd = float(np.clip(policy(obs)[0], -1.0, 1.0))
            energy += engine.step(cmd, depth, self.dt)
            err_acc += abs(depth - target)
            n += 1

        mean_err = err_acc / max(n, 1)
        score = float(np.clip(1.0 - mean_err / 6.0, 0.0, 1.0)) * float(
            np.clip(1.0 - energy / 9000.0, 0.05, 1.0)
        )
        return SkillResult(score, energy, {"target": target, "mean_error_m": mean_err})


class ResonanceSeek(SkillTask):
    """Find and hold the drivetrain's resonance as the medium changes.

    The medium is switched part-way through the episode, which moves the
    resonance by roughly a factor of five.  A controller that memorises one
    frequency scores badly; one that senses amplitude-per-torque and hill-climbs
    scores well.  That difference is worth more battery than any other single
    control decision in this project.
    """

    name = "resonance_seek"
    obs_dim = 5
    act_dim = 1
    # Halved when the oracle reference was added, since each scored episode now
    # runs the dynamics twice.
    duration = 18.0
    # The drivetrain resonance is 0.4 Hz submerged to 2.3 Hz dry, so 5 ms still
    # gives ~90 steps per cycle at the fastest. Finer than that only bought
    # simulation cost, and the cost mattered: this task is inside a CMA-ES loop.
    dt = 0.005
    why = "Resonant flapping cuts drive power several-fold, and resonance moves when it gets wet."

    def _episode(self, policy, stiffness: float, switch_at: float, oracle: bool):
        """One run.  ``oracle`` pins the drive frequency to the true resonance.

        The oracle is the reference the policy is scored against.  Scoring
        against an absolute constant instead -- which is what this task did
        first -- silently saturated: a do-nothing policy scored a perfect 1.0
        while sitting at 7 Hz with resonance at 2.4 Hz and 0.42 Hz.  A metric
        that a broken controller can max out is not measuring the skill.
        """
        drive = CompliantWingDrive(stiffness=stiffness)
        motor = Actuator(motor_class="geared", mass=0.3, gear_ratio=10.0, sealed=True)
        f_cmd, amp_target = 2.0, 0.55
        energy, work = 0.0, 0.0
        peak_tracker = 0.0
        t, phase = 0.0, 0.0

        for _ in range(int(self.duration / self.dt)):
            wetted = 1.0 if t > switch_at else 0.0
            if oracle:
                f_cmd = drive.resonance_hz(wetted)
            else:
                obs = np.array([
                    np.tanh(peak_tracker / amp_target - 1.0), np.tanh(f_cmd / 6.0 - 1.0),
                    wetted, np.tanh(drive.rate / 6.0), 1.0,
                ])
                # The policy sets a frequency *slew rate*, not a frequency: it
                # has to search for resonance rather than recall a number, which
                # is the whole point when resonance moves as the wing wets.
                f_cmd = float(np.clip(f_cmd + policy(obs)[0] * 2.0 * self.dt, 0.3, 12.0))
            phase += 2 * math.pi * f_cmd * self.dt
            torque = motor.stall_torque * 0.6 * math.sin(phase)
            drive.step(torque + math.copysign(SHAFT_SEAL_FRICTION, -drive.rate), wetted, self.dt)

            peak_tracker = max(peak_tracker * 0.999, abs(drive.angle))
            p = float(motor.electrical_power(np.array([abs(torque)]),
                                             np.array([abs(drive.rate)]))[0])
            energy += p * self.dt
            work += abs(drive.angle) * self.dt
            t += self.dt

        return work / max(energy, 1e-9), energy, f_cmd, peak_tracker, drive

    def run(self, policy, rng) -> SkillResult:
        stiffness = float(rng.uniform(8.0, 20.0))
        switch_at = self.duration * float(rng.uniform(0.35, 0.6))

        eff, energy, f_final, amp, drive = self._episode(policy, stiffness, switch_at, False)
        eff_oracle, _, _, _, _ = self._episode(None, stiffness, switch_at, True)

        # Amplitude per joule, as a fraction of what a controller that already
        # knew the answer would achieve.
        score = float(np.clip(eff / max(eff_oracle, 1e-12), 0.0, 1.0))
        return SkillResult(
            score, energy,
            {"final_hz": round(f_final, 3),
             "resonance_air": round(drive.resonance_hz(0.0), 3),
             "resonance_water": round(drive.resonance_hz(1.0), 3),
             "efficiency": eff, "oracle_efficiency": eff_oracle,
             "amplitude": float(amp)},
        )


class ThermalThrust(SkillTask):
    """Maximise sustained output without cooking the motor.

    Copper loss goes as torque squared while useful work goes as torque, so the
    greedy policy always saturates and always overheats.  The competent policy
    finds the duty cycle its heat sink supports.
    """

    name = "thermal_thrust"
    obs_dim = 4
    act_dim = 1
    duration = 120.0
    dt = 0.05
    why = "Duty cycle, not peak torque, is what actually limits a mission."

    def run(self, policy, rng) -> SkillResult:
        motor = Actuator(motor_class="bldc", mass=float(rng.uniform(0.08, 0.25)),
                         gear_ratio=6.0)
        temp = 25.0
        ambient = float(rng.uniform(4.0, 28.0))  # sea is cold, air is not
        limit = 95.0
        thermal_mass = 180.0 * motor.mass
        conductance = 1.1 + 6.0 * float(rng.uniform(0.0, 1.0))  # water cools far better

        energy, thrust_acc, n = 0.0, 0.0, 0
        overheated = 0.0
        for _ in range(int(self.duration / self.dt)):
            obs = np.array([(temp - ambient) / 80.0, (limit - temp) / 80.0,
                            conductance / 8.0, 1.0])
            duty = float(np.clip(policy(obs)[0] * 0.5 + 0.5, 0.0, 1.0))
            torque = duty * motor.stall_torque
            omega = 30.0 * (1.0 - 0.7 * duty)
            p = float(motor.electrical_power(np.array([torque]), np.array([omega]))[0])
            loss = p - abs(torque * omega)
            temp += (loss - conductance * (temp - ambient)) / thermal_mass * self.dt
            if temp > limit:
                overheated += self.dt
                torque *= 0.2  # thermal fold-back, as any real ESC does
            energy += p * self.dt
            thrust_acc += torque * omega * self.dt
            n += 1

        score = float(np.clip(thrust_acc / (energy + 1e-6) * 2.2, 0.0, 1.0)) * float(
            np.clip(1.0 - overheated / self.duration * 3.0, 0.0, 1.0)
        )
        return SkillResult(score, energy,
                           {"peak_temp": temp, "overheat_s": overheated, "ambient": ambient})


class SolenoidLatch(SkillTask):
    """Actuate an electromagnetic latch without holding current.

    Rewards pulsing to close and then releasing, which is what a mechanical
    latch allows and what a naive controller never discovers because holding
    also satisfies the position objective.
    """

    name = "solenoid_latch"
    obs_dim = 4
    act_dim = 1
    duration = 6.0
    # The armature's natural period is ~16 ms; 1 ms resolves it adequately.
    dt = 0.001
    why = "Ballast and wing-fold latches are held for minutes; holding current is pure loss."

    def run(self, policy, rng) -> SkillResult:
        sol = Solenoid(spring_k=float(rng.uniform(2000, 4500)))
        want_closed_from = self.duration * 0.25
        energy, err, n = 0.0, 0.0, 0
        latched = False
        for i in range(int(self.duration / self.dt)):
            t = i * self.dt
            want = 1.0 if t > want_closed_from else 0.0
            closed = sol.gap < 0.15 * sol.stroke
            if closed and want:
                latched = True  # a mechanical detent holds it from here
            obs = np.array([want, sol.gap / sol.stroke, float(latched), 1.0])
            duty = float(np.clip(policy(obs)[0] * 0.5 + 0.5, 0.0, 1.0))
            if latched:
                sol.gap = 0.0
                energy += duty * sol.voltage**2 / sol.resistance * self.dt  # wasted if commanded
            else:
                energy += sol.step(duty, self.dt)
            err += abs(want - (1.0 if sol.gap < 0.15 * sol.stroke else 0.0)) * self.dt
            n += 1
        score = float(np.clip(1.0 - err / (self.duration * 0.5), 0.0, 1.0)) * float(
            np.clip(1.0 - energy / 60.0, 0.05, 1.0)
        )
        return SkillResult(score, energy, {"latched": latched})


class VibratorTrack(SkillTask):
    """Keep a piezo element on resonance as it wets and dries."""

    name = "vibrator_track"
    obs_dim = 4
    act_dim = 1
    duration = 25.0
    dt = 0.01
    why = "Silt shedding and antifouling only work on resonance, which moves when submerged."

    def run(self, policy, rng) -> SkillResult:
        v = PiezoVibrator(f0=float(rng.uniform(400, 560)))
        f = 400.0
        energy, amp_acc, n = 0.0, 0.0, 0
        for i in range(int(self.duration / self.dt)):
            t = i * self.dt
            wetted = 0.5 + 0.5 * math.sin(2 * math.pi * t / 9.0)
            amp = v.response(f, wetted)
            obs = np.array([np.tanh(amp / 8.0), f / 600.0, wetted, 1.0])
            f = float(np.clip(f + policy(obs)[0] * 4.0, 250.0, 650.0))
            energy += (0.4 + 0.02 * amp) * self.dt
            amp_acc += amp * self.dt
            n += 1
        score = float(np.clip(amp_acc / self.duration / 12.0, 0.0, 1.0))
        return SkillResult(score, energy, {"final_hz": f, "mean_amp": amp_acc / self.duration})


class BallastSequence(SkillTask):
    """Dive to a depth as fast as possible without overshooting past it.

    Tests coordination rather than regulation: vent, flood, then arrest.  The
    penalty for overshoot is what makes it hard -- a machine that simply floods
    everything reaches depth quickly and then keeps going.
    """

    name = "ballast_sequence"
    obs_dim = 5
    act_dim = 2
    duration = 60.0
    dt = 0.05
    why = "A five-minute dive leg spent descending is a five-minute leg wasted."

    def run(self, policy, rng) -> SkillResult:
        target = 10.0
        dry_mass, rigid = 6.0, 0.0075
        engine = BuoyancyEngine(capacity=0.004)
        gas = GasStore(volume_surface=0.004)
        vented = 0.0
        depth, vz = 0.0, 0.0
        energy, n = 0.0, 0
        first_arrival = None
        overshoot = 0.0

        for i in range(int(self.duration / self.dt)):
            gas_now = gas.volume_at(depth) * (1.0 - vented)
            disp = rigid + gas_now
            mass = dry_mass + engine.volume * SEAWATER.rho
            net = SEAWATER.rho * GRAVITY * disp - mass * GRAVITY
            acc = (net - 55.0 * vz * abs(vz)) / max(mass, 1e-3)
            vz += acc * self.dt
            depth = max(depth - vz * self.dt, 0.0)

            obs = np.array([np.tanh((depth - target) / 5.0), np.tanh(vz / 0.8),
                            vented, engine.volume / engine.capacity, 1.0])
            a = policy(obs)
            vented = float(np.clip(vented + np.clip(a[0], -1, 1) * 0.05, 0.0, 1.0))
            energy += engine.step(float(np.clip(a[1], -1, 1)), depth, self.dt)

            if depth >= target * 0.95 and first_arrival is None:
                first_arrival = i * self.dt
            overshoot = max(overshoot, depth - target)
            n += 1

        settle = abs(depth - target)
        speed = 1.0 if first_arrival is None else float(
            np.clip(1.0 - first_arrival / self.duration, 0.0, 1.0)
        )
        score = (
            (0.0 if first_arrival is None else 0.45 + 0.25 * speed)
            * float(np.clip(1.0 - overshoot / 4.0, 0.0, 1.0))
            * float(np.clip(1.0 - settle / 4.0, 0.0, 1.0))
        )
        return SkillResult(score, energy,
                           {"arrival_s": first_arrival, "overshoot_m": overshoot,
                            "settled_error_m": settle})


SKILL_TASKS: dict[str, type[SkillTask]] = {
    t.name: t
    for t in (DepthHold, ResonanceSeek, ThermalThrust, SolenoidLatch, VibratorTrack,
              BallastSequence)
}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@dataclass(eq=False)
class LinearSkillPolicy:
    """A one-hidden-layer tanh policy, small enough for CMA-ES to solve fast."""

    obs_dim: int
    act_dim: int
    hidden: int = 12
    weights: np.ndarray | None = None

    @property
    def n_weights(self) -> int:
        return self.obs_dim * self.hidden + self.hidden + self.hidden * self.act_dim + self.act_dim

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        w = self.weights
        if w is None:
            return np.zeros(self.act_dim)
        i = 0
        W1 = w[i:i + self.obs_dim * self.hidden].reshape(self.obs_dim, self.hidden)
        i += self.obs_dim * self.hidden
        b1 = w[i:i + self.hidden]; i += self.hidden
        W2 = w[i:i + self.hidden * self.act_dim].reshape(self.hidden, self.act_dim)
        i += self.hidden * self.act_dim
        b2 = w[i:i + self.act_dim]
        return np.tanh(np.tanh(obs @ W1 + b1) @ W2 + b2)


def train_skill(
    task_name: str,
    *,
    iterations: int = 30,
    episodes_per_eval: int = 3,
    seed: int = 0,
    hidden: int = 12,
    on_iteration=None,
) -> dict:
    """Learn one skill with CMA-ES.

    ``episodes_per_eval`` averages over randomised task instances, which is what
    stops the optimiser from solving one particular target depth or one
    particular motor and calling it a policy.
    """
    from ..evolution.cmaes import CMAES

    task = SKILL_TASKS[task_name]()
    proto = LinearSkillPolicy(task.obs_dim, task.act_dim, hidden)
    n = proto.n_weights
    es = CMAES(np.zeros(n), sigma0=0.5, seed=seed)
    rng = np.random.default_rng(seed)
    history = []

    for it in range(iterations):
        pop = es.ask()
        scores = np.zeros(len(pop))
        for k, w in enumerate(pop):
            pol = LinearSkillPolicy(task.obs_dim, task.act_dim, hidden, w)
            total = 0.0
            for ep in range(episodes_per_eval):
                r = task.run(pol, np.random.default_rng(seed * 1000 + it * 97 + ep))
                total += r.score
            scores[k] = total / episodes_per_eval
        es.tell(pop, scores)
        history.append({"iteration": it, "best": float(scores.max()),
                        "mean": float(scores.mean()), "sigma": es.sigma})
        if on_iteration is not None:
            on_iteration(it, history[-1])

    best_pol = LinearSkillPolicy(task.obs_dim, task.act_dim, hidden, es.best_x)
    final = task.run(best_pol, np.random.default_rng(seed + 12345))
    return {
        "task": task_name,
        "why": task.why,
        "weights": es.best_x.tolist(),
        "obs_dim": task.obs_dim,
        "act_dim": task.act_dim,
        "hidden": hidden,
        "best_score": float(es.best_f),
        "final_score": final.score,
        "final_detail": final.detail,
        "history": history,
    }


def baseline_score(task_name: str, seed: int = 0, episodes: int = 5) -> float:
    """Score of a do-nothing policy, for reference.

    Reported alongside every trained skill.  Without it a score of 0.4 is
    meaningless -- it might be excellent or it might be what you get for
    outputting zeros.
    """
    task = SKILL_TASKS[task_name]()
    pol = LinearSkillPolicy(task.obs_dim, task.act_dim, 12, np.zeros(
        LinearSkillPolicy(task.obs_dim, task.act_dim, 12).n_weights))
    return float(np.mean([
        task.run(pol, np.random.default_rng(seed + i)).score for i in range(episodes)
    ]))
