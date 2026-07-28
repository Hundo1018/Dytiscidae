"""Electrical power train: actuators, battery, and the mission energy budget.

A generative pipeline left to itself will produce machines that are mechanically
beautiful and electrically impossible.  This module is the accountant.  It
converts every newton-metre the controller asks for into joules out of a pack of
finite size, and it charges for the things that are easy to forget: winding
resistance at peak torque, gearbox loss, iron loss, standby avionics draw, and
the friction of every dynamic seal that crosses the hull boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .materials import CELLS, MOTOR_CLASSES, SHAFT_SEAL_FRICTION, Cell, MotorClass


@dataclass
class Actuator:
    """One electromechanical actuator sized against a motor family.

    Parameters
    ----------
    motor_class:
        Key into ``materials.MOTOR_CLASSES``.
    mass:
        Motor + drive mass, kg.  Sets the continuous power rating.
    gear_ratio:
        Output reduction.  Torque multiplies, speed divides, efficiency drops.
    sealed:
        True if this actuator's output shaft penetrates the pressure boundary.
        Costs a seal mass and a constant friction torque whenever it moves.
    """

    motor_class: str = "bldc"
    mass: float = 0.08
    gear_ratio: float = 1.0
    sealed: bool = False

    def __post_init__(self) -> None:
        self.spec: MotorClass = MOTOR_CLASSES[self.motor_class]
        self.mass = float(np.clip(self.mass, self.spec.mass_min, self.spec.mass_max))
        # Continuous mechanical rating.
        self.p_cont = self.spec.specific_power * self.mass
        # Gearboxes lose ~3% per stage; approximate stages as log_5(ratio).
        stages = max(0.0, np.log(max(self.gear_ratio, 1.0)) / np.log(5.0))
        self.eta_gear = 0.97**stages
        # Motor constant km = tau / sqrt(P_copper).  Scales as mass^(3/4) for a
        # geometrically similar motor; anchored so a 100 g outrunner has the
        # right order of magnitude (km ~ 0.05 N.m/sqrt(W)).
        self.km = 0.05 * (self.mass / 0.1) ** 0.75 * self.spec.efficiency_peak / 0.88
        # No-load / iron loss coefficients, referred to output speed.
        self.k_visc = 2e-5 * self.mass / 0.1  # N.m per rad/s
        self.k_iron = 0.02 * self.p_cont / 1000.0  # W per (rad/s) of motor speed

    @property
    def stall_torque(self) -> float:
        """Torque at the continuous thermal limit, N.m at the output shaft."""
        return self.km * np.sqrt(self.p_cont) * self.gear_ratio * self.eta_gear

    @property
    def max_speed(self) -> float:
        """Output speed at the continuous rating, rad/s."""
        return self.p_cont / max(self.stall_torque, 1e-6)

    def electrical_power(self, torque: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Electrical power drawn for an output torque and speed, W.

        Never negative: regeneration through a hobby ESC is not a thing, and
        assuming it would let the optimiser build a perpetual motion machine
        out of a flapping wing.
        """
        torque = np.asarray(torque, float)
        omega = np.asarray(omega, float)

        if self.sealed:
            torque = torque + np.sign(omega) * SHAFT_SEAL_FRICTION

        # Reflect to the motor shaft.
        tau_m = np.abs(torque) / (self.gear_ratio * self.eta_gear)
        om_m = np.abs(omega) * self.gear_ratio

        p_mech = np.abs(torque * omega)
        p_copper = (tau_m / max(self.km, 1e-9)) ** 2
        p_iron = self.k_iron * om_m
        p_visc = self.k_visc * om_m**2
        return p_mech + p_copper + p_iron + p_visc

    def thermal_overload(self, torque: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Ratio of dissipated power to the continuous rating.

        Values above 1 mean the actuator is being run past its thermal limit.
        Sustained overload is a hard constraint, not a soft penalty: a motor
        that cooks halfway through the mission has failed the mission.
        """
        tau_m = np.abs(np.asarray(torque, float)) / (self.gear_ratio * self.eta_gear)
        p_copper = (tau_m / max(self.km, 1e-9)) ** 2
        return p_copper / max(0.35 * self.p_cont, 1e-6)


@dataclass
class Battery:
    """Pack-level energy store."""

    chemistry: str = "liion"
    wh: float = 150.0

    def __post_init__(self) -> None:
        self.cell: Cell = CELLS[self.chemistry]
        self.capacity_j = self.wh * 3600.0
        self.mass = self.cell.pack_mass(self.wh)
        self.p_max = self.cell.max_power(self.wh)
        self.reset()

    def reset(self) -> None:
        self.energy_j = self.capacity_j
        self.peak_draw = 0.0
        self.overdraw_time = 0.0

    @property
    def soc(self) -> float:
        return float(np.clip(self.energy_j / self.capacity_j, 0.0, 1.0))

    def draw(self, power_w: float, dt: float) -> bool:
        """Remove energy.  Returns False once the pack is empty.

        Draw beyond the continuous rating is allowed but tracked, and the
        efficiency penalty of doing so is applied through a simple internal
        resistance model -- pulling hard from a small pack wastes energy as
        heat, which is precisely the trap a hover-capable design falls into.
        """
        self.peak_draw = max(self.peak_draw, power_w)
        overdraw = power_w / max(self.p_max, 1e-9)
        if overdraw > 1.0:
            self.overdraw_time += dt
            power_w *= 1.0 + 0.25 * (overdraw - 1.0)
        self.energy_j -= power_w * dt
        return self.energy_j > 0.0


@dataclass
class PowerBudget:
    """Tracks the whole machine's electrical consumption over an episode."""

    battery: Battery
    actuators: list[Actuator] = field(default_factory=list)
    #: Constant hotel load: flight controller, IMU, depth sensor, radio, lights.
    #: A Pico-class MCU plus sensors and a radio is a few watts once you include
    #: the buck converters; at the 15 kg scale the avionics are larger.
    avionics_w: float = 6.0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.battery.reset()
        self.total_j = 0.0
        self.actuator_j = 0.0
        self.avionics_j = 0.0
        self.t = 0.0
        self.max_overload = 0.0
        self.samples: list[tuple[float, float]] = []

    @property
    def actuator_mass(self) -> float:
        return sum(a.mass for a in self.actuators)

    def step(self, torques: np.ndarray, speeds: np.ndarray, dt: float) -> bool:
        """Charge one control step.  Returns False when the pack is flat."""
        p = self.avionics_w
        for i, act in enumerate(self.actuators):
            if i >= len(torques):
                break
            p += float(act.electrical_power(torques[i], speeds[i]))
            self.max_overload = max(
                self.max_overload, float(act.thermal_overload(torques[i], speeds[i]))
            )
        self.avionics_j += self.avionics_w * dt
        self.actuator_j += (p - self.avionics_w) * dt
        self.total_j += p * dt
        self.t += dt
        self.samples.append((self.t, p))
        return self.battery.draw(p, dt)

    @property
    def mean_power(self) -> float:
        return self.total_j / max(self.t, 1e-9)

    def endurance_at_current_rate(self) -> float:
        """Seconds of remaining runtime if the present mean draw continued."""
        p = self.mean_power
        return self.battery.energy_j / max(p, 1e-9)


# --------------------------------------------------------------------------
# Analytic mission energy: the Tier-0 filter
# --------------------------------------------------------------------------


@dataclass
class DomainEnergyEstimate:
    """Closed-form power estimate for one operating domain."""

    domain: str
    power_w: float
    duration_s: float
    feasible: bool
    limiter: str = ""

    @property
    def energy_wh(self) -> float:
        return self.power_w * self.duration_s / 3600.0


def cruise_power_air(
    mass: float,
    span: float,
    wing_area: float,
    speed: float,
    *,
    ld_ratio: float | None = None,
    propulsive_eff: float = 0.55,
    drivetrain_eff: float = 0.70,
    rho: float = 1.225,
) -> tuple[float, str]:
    """Electrical power for steady forward flapping flight, W.

    Uses the classical drag polar rather than a hover estimate, because hover is
    what makes a 15 kg flapping machine impossible and cruise is what makes it
    merely very hard.  For reference: at 15 kg, hovering with a 1 m^2 actuator
    disc needs about 3.2 kW electrical, while cruising at L/D = 6 needs about
    500 W.  Any design that has to hover has already lost.
    """
    from .medium import GRAVITY

    weight = mass * GRAVITY
    if wing_area <= 1e-6 or speed <= 0.1:
        return float("inf"), "degenerate wing"
    ar = span**2 / wing_area
    cl = weight / (0.5 * rho * speed**2 * wing_area)
    if cl > 2.2:
        return float("inf"), f"CL={cl:.2f} beyond any flapping wing"
    cd0 = 0.035  # a fat amphibious fuselage with legs and a hull is not clean
    cdi = cl**2 / (np.pi * 0.75 * max(ar, 0.8))
    cd = cd0 + cdi
    if ld_ratio is None:
        ld_ratio = cl / cd
    p_aero = weight * speed / max(ld_ratio, 0.5)
    p_elec = p_aero / (propulsive_eff * drivetrain_eff)
    return float(p_elec), f"L/D={ld_ratio:.1f} CL={cl:.2f} AR={ar:.1f}"


def cruise_power_water(
    volume: float,
    frontal_area: float,
    speed: float,
    *,
    cd: float = 0.25,
    propulsive_eff: float = 0.45,
    drivetrain_eff: float = 0.70,
    rho: float = 1025.0,
    seal_count: int = 0,
) -> tuple[float, str]:
    """Electrical power for steady submerged cruise, W.

    Underwater is the cheap domain, which is counterintuitive until you notice
    that a submerged body needs no lift: buoyancy is free, so all the power goes
    into drag, and drag at 1 m/s is tiny.  The real underwater costs are seal
    friction and holding depth, not going forwards.
    """
    drag = 0.5 * rho * cd * frontal_area * speed**2
    p_hydro = drag * speed
    p_elec = p_hydro / (propulsive_eff * drivetrain_eff)
    # Every dynamic seal costs friction whenever anything moves.
    p_elec += seal_count * SHAFT_SEAL_FRICTION * 20.0
    return float(p_elec), f"drag={drag:.1f}N"


def crawl_power_land(
    mass: float,
    speed: float,
    *,
    cost_of_transport: float = 4.0,
    drivetrain_eff: float = 0.55,
) -> tuple[float, str]:
    """Electrical power for legged or wheeled locomotion, W.

    ``cost_of_transport`` is dimensionless (E / (m g d)).  Legged robots land
    between 2 and 10; the default of 4 is optimistic-but-defensible for a
    sprawling amphibious gait.
    """
    from .medium import GRAVITY

    p_mech = cost_of_transport * mass * GRAVITY * speed
    return float(p_mech / drivetrain_eff), f"CoT={cost_of_transport:.1f}"


def transition_energy(mass: float, kind: str) -> float:
    """Energy for one domain transition, J.

    Transitions are not free and are routinely ignored in papers.  Getting a
    15 kg machine out of water and into the air is the single most expensive
    event in the mission: it must break the surface, shed entrained water,
    accelerate to flying speed, and climb clear of the wave field.
    """
    from .medium import GRAVITY

    table = {
        # Breaching: potential energy to lift clear of the surface + kinetic
        # energy to reach flying speed + a large allowance for entrained water.
        "water_to_air": mass * GRAVITY * 1.5 + 0.5 * mass * 12.0**2 + mass * GRAVITY * 2.0,
        # Entry is nearly free in energy terms, but it is a structural event.
        "air_to_water": 0.15 * mass * GRAVITY * 1.0,
        "water_to_land": mass * GRAVITY * 0.8 * 3.0,  # dragging out of the surf
        "land_to_water": 0.2 * mass * GRAVITY * 0.5,
        "land_to_air": 0.5 * mass * 12.0**2 + mass * GRAVITY * 3.0,
        "air_to_land": 0.1 * mass * GRAVITY * 1.0,
    }
    return float(table.get(kind, 0.5 * mass * GRAVITY))


def mission_energy_wh(
    estimates: list[DomainEnergyEstimate], transitions: list[str], mass: float
) -> float:
    """Total mission energy including transitions, Wh."""
    e = sum(d.energy_wh for d in estimates)
    e += sum(transition_energy(mass, k) for k in transitions) / 3600.0
    return float(e)
