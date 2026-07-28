"""Structural feasibility: spars, hulls, seals, and the loads that break them.

Generative structure search will confidently propose a 2.4 m wing on a 6 mm
printed spar.  These checks exist so that such a design is scored as dead rather
than as an elite.  Each check returns a *margin* (allowable / applied - 1), so
the optimiser sees a gradient and can climb toward feasibility rather than
falling off a cliff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .materials import (
    SEALING_MASS_FRACTION,
    SHAFT_SEAL_MASS,
    Material,
)
from .medium import GRAVITY, P_ATM, SEAWATER


@dataclass
class Check:
    """One structural criterion."""

    name: str
    applied: float
    allowable: float
    unit: str = "Pa"
    note: str = ""

    @property
    def margin(self) -> float:
        """(allowable / applied) - 1.  Negative means failure."""
        if self.applied <= 1e-12:
            return 10.0
        return self.allowable / self.applied - 1.0

    @property
    def ok(self) -> bool:
        return self.margin >= 0.0


@dataclass
class StructuralReport:
    checks: list[Check] = field(default_factory=list)
    mass_penalty: float = 0.0  # kg of structure/sealing implied by the checks

    def add(self, c: Check) -> Check:
        self.checks.append(c)
        return c

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def worst(self) -> Check | None:
        return min(self.checks, key=lambda c: c.margin) if self.checks else None

    @property
    def min_margin(self) -> float:
        return min((c.margin for c in self.checks), default=10.0)

    def summary(self) -> str:
        w = self.worst
        if w is None:
            return "no checks"
        return f"{'PASS' if self.ok else 'FAIL'} worst={w.name} margin={w.margin:+.2f}"


# --------------------------------------------------------------------------
# Beams
# --------------------------------------------------------------------------


def tube_section(outer_d: float, wall: float) -> tuple[float, float, float]:
    """Return ``(area, second_moment, section_modulus)`` of a hollow round tube."""
    ro = outer_d / 2.0
    ri = max(ro - wall, 0.0)
    area = math.pi * (ro**2 - ri**2)
    i = math.pi * (ro**4 - ri**4) / 4.0
    z = i / ro if ro > 0 else 0.0
    return area, i, z


def spar_check(
    *,
    lift_n: float,
    semi_span: float,
    outer_d: float,
    wall: float,
    material: Material,
    load_factor: float = 3.0,
    cycles: float = 1e5,
    report: StructuralReport | None = None,
) -> Check:
    """Root bending of a wing spar under an elliptically distributed lift.

    The resultant of an elliptic lift distribution acts at 4/(3*pi) of the
    semi-span, so the root moment is ``L_semi * semi_span * 4/(3*pi)``.

    ``load_factor`` covers manoeuvre and gust.  Three is modest for an aircraft
    and *low* for a flapping wing, whose inertial loads at stroke reversal can
    exceed its aerodynamic loads -- which is why ``flapping_inertial_check``
    exists separately.
    """
    _, _, z = tube_section(outer_d, wall)
    if z <= 0:
        z = 1e-12
    l_semi = lift_n * load_factor / 2.0
    moment = l_semi * semi_span * 4.0 / (3.0 * math.pi)
    sigma = moment / z
    c = Check(
        "spar_root_bending",
        applied=sigma,
        allowable=material.allowable_stress(cycles=cycles),
        note=f"M={moment:.1f}N.m Z={z*1e9:.1f}mm^3 {material.name}",
    )
    if report is not None:
        report.add(c)
    return c


def flapping_inertial_check(
    *,
    wing_mass: float,
    semi_span: float,
    flap_freq: float,
    flap_amplitude_rad: float,
    outer_d: float,
    wall: float,
    material: Material,
    report: StructuralReport | None = None,
) -> Check:
    """Root bending from the wing's own inertia at stroke reversal.

    A wing swinging at +/-``amplitude`` at ``f`` Hz sees a peak angular
    acceleration of ``amplitude * (2*pi*f)^2``.  The distributed wing mass, with
    its centroid at roughly 40% of the semi-span, generates a root moment that
    scales with frequency *squared*.  This is the term that stops the optimiser
    from simply flapping faster to make more lift, and it is the reason large
    flapping machines converge on low frequencies and long wings.
    """
    omega = 2.0 * math.pi * flap_freq
    ang_acc = flap_amplitude_rad * omega**2
    r_cg = 0.40 * semi_span
    # Second moment of the wing about the root, thin-rod-like distribution.
    i_root = wing_mass * (0.45 * semi_span) ** 2
    moment = i_root * ang_acc
    _, _, z = tube_section(outer_d, wall)
    sigma = moment / max(z, 1e-12)
    c = Check(
        "spar_inertial_reversal",
        applied=sigma,
        allowable=material.allowable_stress(cycles=1e6),  # once per half stroke
        note=f"a={ang_acc:.0f}rad/s^2 M={moment:.1f}N.m r_cg={r_cg:.2f}m",
    )
    if report is not None:
        report.add(c)
    return c


def spar_deflection(
    *, lift_n: float, semi_span: float, outer_d: float, wall: float, material: Material
) -> float:
    """Tip deflection under load as a fraction of semi-span.

    Not a failure criterion but an efficiency one: a wing that deflects more
    than ~10% of its semi-span has lost most of its intended twist distribution
    and will not fly the way the optimiser believes it does.
    """
    _, i, _ = tube_section(outer_d, wall)
    if i <= 0:
        return 1e3
    # Uniformly distributed load approximation: delta = q L^4 / (8 E I).
    q = lift_n / 2.0 / max(semi_span, 1e-6)
    delta = q * semi_span**4 / (8.0 * material.E * i)
    return delta / max(semi_span, 1e-6)


# --------------------------------------------------------------------------
# Pressure hull
# --------------------------------------------------------------------------


def hull_pressure_check(
    *,
    depth_m: float,
    radius: float,
    wall: float,
    length: float,
    material: Material,
    report: StructuralReport | None = None,
) -> tuple[Check, Check]:
    """Hoop stress and elastic buckling of a cylindrical pressure hull.

    At the 10 m design depth the gauge pressure is only 1 bar, so hoop stress is
    trivially satisfied by any printable wall.  **Buckling is the real
    constraint** -- a thin cylinder under external pressure collapses long
    before it yields, and printed polymers have low modulus, so the critical
    pressure scales as ``E * (t/r)^3`` and falls off a cliff as the hull gets
    bigger.  This is the check that quietly forces large designs toward small
    dry volumes or toward pressure-tolerant (flooded, oil-filled) construction.
    """
    p_gauge = SEAWATER.rho * GRAVITY * max(depth_m, 0.0)

    hoop = p_gauge * radius / max(wall, 1e-6)
    c_hoop = Check(
        "hull_hoop_stress",
        applied=hoop,
        allowable=material.allowable_stress(cycles=1e4),
        note=f"p={p_gauge/1e5:.2f}bar r={radius*1e3:.0f}mm t={wall*1e3:.1f}mm",
    )

    # Long-cylinder external pressure buckling, with a 0.6 knockdown for the
    # imperfections a printed hull certainly has.
    nu = material.poisson
    p_cr = 0.6 * 2.0 * material.E / (1.0 - nu**2) * (wall / max(radius, 1e-6)) ** 3
    c_buck = Check(
        "hull_buckling",
        applied=p_gauge,
        allowable=p_cr,
        unit="Pa",
        note=f"p_cr={p_cr/1e5:.2f}bar",
    )
    if report is not None:
        report.add(c_hoop)
        report.add(c_buck)
    return c_hoop, c_buck


def hull_mass(
    *, radius: float, wall: float, length: float, material: Material, endcaps: bool = True
) -> float:
    """Mass of a cylindrical hull with hemispherical end caps, kg."""
    shell = 2.0 * math.pi * radius * wall * length
    caps = 4.0 * math.pi * radius**2 * wall if endcaps else 0.0
    return (shell + caps) * material.rho


def sealing_mass(n_penetrations: int, wetted_mass: float) -> float:
    """Mass of dynamic seals and waterproofing, kg.

    Every actuated axis that crosses the pressure boundary costs a rotary shaft
    seal.  The sensible design response is to keep actuators outside the hull
    (flooded and pressure-tolerant) and only penetrate for power, but that trade
    is the optimiser's to discover -- it just has to be charged for either way.
    """
    return n_penetrations * SHAFT_SEAL_MASS + wetted_mass * SEALING_MASS_FRACTION


# --------------------------------------------------------------------------
# Buoyancy and trim
# --------------------------------------------------------------------------


@dataclass
class BuoyancyState:
    """Static buoyancy accounting at a given depth."""

    mass: float
    displaced_volume: float
    ballast_volume: float = 0.0  # variable volume currently flooded, m^3
    gas_volume_surface: float = 0.0  # compressible gas carried, m^3 at 1 atm

    def net_buoyancy(self, depth_m: float = 0.0) -> float:
        """Net upward force, N.  Positive floats, negative sinks.

        The gas volume is compressed by Boyle's law, which is the single most
        important instability in shallow diving: a vehicle trimmed neutral at
        the surface becomes *more* negative the deeper it goes, so it runs away
        downward unless it actively pumps.  A diving beetle solves this with an
        elytral air store it can vent; the optimiser is free to reinvent that.
        """
        p_ratio = P_ATM / (P_ATM + SEAWATER.rho * GRAVITY * max(depth_m, 0.0))
        gas_now = self.gas_volume_surface * p_ratio
        v = self.displaced_volume - self.ballast_volume + gas_now - self.gas_volume_surface
        return SEAWATER.rho * GRAVITY * v - self.mass * GRAVITY

    def depth_stability(self, depth_m: float) -> float:
        """d(net buoyancy)/d(depth), N/m.  Negative is unstable in depth."""
        eps = 0.05
        return (self.net_buoyancy(depth_m + eps) - self.net_buoyancy(depth_m - eps)) / (2 * eps)

    def trim_authority(self, depth_m: float = 10.0) -> float:
        """Range of net buoyancy the ballast system can command, N."""
        full = BuoyancyState(self.mass, self.displaced_volume, 0.0, self.gas_volume_surface)
        empty = BuoyancyState(
            self.mass, self.displaced_volume, self.ballast_volume, self.gas_volume_surface
        )
        return abs(full.net_buoyancy(depth_m) - empty.net_buoyancy(depth_m))


def ballast_pump_power(depth_m: float, flow_m3_s: float, efficiency: float = 0.35) -> float:
    """Power to pump water out of a ballast tank against ambient pressure, W.

    This is the cost of active depth control and it grows linearly with depth.
    At 10 m, moving 100 mL/s costs about 30 W hydraulic, ~85 W electrical --
    small, but continuous, and it is why a passively stable trim is worth a lot.
    """
    dp = SEAWATER.rho * GRAVITY * max(depth_m, 0.0)
    return dp * flow_m3_s / max(efficiency, 1e-3)


# --------------------------------------------------------------------------
# Water entry
# --------------------------------------------------------------------------


def slam_pressure(impact_speed: float, deadrise_deg: float = 20.0) -> float:
    """Peak slamming pressure on water entry, Pa (Wagner / Chuang).

    A flat surface hitting water at 10 m/s sees pressures of order 500 kPa --
    far above anything the wing sees in flight.  A deadrise angle (a V shape)
    reduces it dramatically, which is why every seaplane hull has one and why a
    flat-bottomed generated design should be penalised.
    """
    dr = math.radians(max(deadrise_deg, 1.0))
    # Wagner: p_max ~ 0.5 * rho * v^2 * (pi/tan(beta))^2 for small beta,
    # limited by the compressible/ventilated regime at very small deadrise.
    k = min((math.pi / math.tan(dr)) ** 2, 250.0)
    return 0.5 * SEAWATER.rho * impact_speed**2 * k


def water_entry_stress(
    *, impact_speed: float, deadrise_deg: float, radius: float, wall: float,
    flat_panel_width: float | None = None,
) -> tuple[float, float]:
    """Stress in a structure under slamming, and the pressure that caused it.

    Two load paths, and picking the wrong one changes the answer by two orders
    of magnitude:

    * **Curved shell** (the default).  A cylindrical hull carries external
      pressure in hoop membrane stress, ``sigma = p * r / t``.  This is the
      right path for any rounded hull, and it is why boats are not flat.
    * **Flat panel** (``flat_panel_width`` given).  A flat unsupported panel has
      no membrane path and must take the load in bending,
      ``sigma = 0.30 * p * b^2 / t^2``.  For a 200 mm panel this is roughly 40x
      the shell stress, which is a real and correct penalty on flat bottoms.
    """
    p = slam_pressure(impact_speed, deadrise_deg)
    if flat_panel_width is not None:
        return 0.30 * p * flat_panel_width**2 / max(wall, 1e-4) ** 2, p
    return p * max(radius, 1e-3) / max(wall, 1e-4), p


def water_entry_check(
    *,
    impact_speed: float,
    deadrise_deg: float,
    radius: float,
    wall: float,
    material: Material,
    flat_panel_width: float | None = None,
    report: StructuralReport | None = None,
) -> Check:
    """Whether the hull survives water entry at ``impact_speed``."""
    sigma, p = water_entry_stress(
        impact_speed=impact_speed, deadrise_deg=deadrise_deg, radius=radius,
        wall=wall, flat_panel_width=flat_panel_width,
    )
    c = Check(
        "water_entry_hull",
        applied=sigma,
        allowable=material.allowable_stress(cycles=1e3),
        note=(
            f"v={impact_speed:.1f}m/s p={p/1e3:.0f}kPa "
            f"{'flat panel' if flat_panel_width else 'curved shell'} "
            f"deadrise={deadrise_deg:.0f}deg"
        ),
    )
    if report is not None:
        report.add(c)
    return c
