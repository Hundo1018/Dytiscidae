"""Material and component property database.

Numbers here are deliberately conservative and cite the manufacturing route,
because the whole point of a generative pipeline is that it will happily invent
a wing spar out of a material that cannot survive being a wing spar.  Printed
polymers in particular are anisotropic and fatigue badly, so the knockdown
factors matter more than the headline strengths.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """An isotropic-equivalent structural material.

    ``strength_knockdown`` folds together the layer-adhesion penalty of fused
    deposition printing and a generic manufacturing scatter allowance.  Applied
    strengths are ``yield_MPa * strength_knockdown``.

    ``fatigue_knockdown_1e5`` is the further factor applied when a part sees
    ~1e5 fully reversed load cycles, which is what a flapping spar accumulates
    in a single 45 minute mission at 5 Hz (5 Hz * 900 s of flight = 4500 cycles
    per mission, but the design life is many missions).
    """

    name: str
    rho: float  # kg/m^3
    E: float  # Young's modulus, Pa
    yield_MPa: float  # tensile yield / ultimate, MPa
    strength_knockdown: float
    fatigue_knockdown_1e5: float
    poisson: float = 0.35
    note: str = ""

    def allowable_stress(self, cycles: float = 0.0, safety_factor: float = 2.0) -> float:
        """Allowable stress in Pa for a given cyclic life and safety factor."""
        s = self.yield_MPa * 1e6 * self.strength_knockdown
        if cycles > 1e3:
            # Log-linear interpolation of the knockdown between 1e3 and 1e5 cycles.
            import math

            frac = min(1.0, (math.log10(cycles) - 3.0) / 2.0)
            s *= 1.0 + frac * (self.fatigue_knockdown_1e5 - 1.0)
        return s / safety_factor


# --------------------------------------------------------------------------
# Structural materials
# --------------------------------------------------------------------------

PETG = Material(
    name="PETG (FDM printed)",
    rho=1270.0,
    E=2.0e9,
    yield_MPa=50.0,
    # Printed PETG reaches roughly 60-70% of injection-moulded strength across
    # layer lines, and Z-direction is the weak axis of every printed part.
    strength_knockdown=0.60,
    # Unreinforced thermoplastics lose roughly half their strength by 1e5 cycles.
    fatigue_knockdown_1e5=0.45,
    poisson=0.40,
    note="Tough, hygroscopically stable, bonds well, prints watertight with enough perimeters.",
)

PETG_CF = Material(
    name="PETG-CF (20% chopped carbon)",
    rho=1300.0,
    E=5.5e9,
    yield_MPa=55.0,
    strength_knockdown=0.55,  # stiffer but more brittle, worse layer adhesion
    fatigue_knockdown_1e5=0.40,
    note="Higher stiffness per mass than PETG; abrasive on nozzles; more porous.",
)

CFRP_TUBE = Material(
    name="Pultruded carbon tube",
    rho=1550.0,
    E=1.35e11,
    yield_MPa=800.0,
    strength_knockdown=0.80,
    fatigue_knockdown_1e5=0.80,  # carbon composites are excellent in fatigue
    poisson=0.30,
    note="The only realistic spar material at the 15 kg scale. Bought, not printed.",
)

AL6061 = Material(
    name="Aluminium 6061-T6",
    rho=2700.0,
    E=6.9e10,
    yield_MPa=276.0,
    strength_knockdown=0.90,
    fatigue_knockdown_1e5=0.45,  # aluminium has no fatigue limit
    poisson=0.33,
    note="Pressure hull / fitting material. Galvanic issues in seawater.",
)

MYLAR_MEMBRANE = Material(
    name="Mylar membrane (wing skin)",
    rho=1390.0,
    E=4.0e9,
    yield_MPa=170.0,
    strength_knockdown=0.70,
    fatigue_knockdown_1e5=0.60,
    note="Thin-film wing surface. Contributes area and mass but carries no bending.",
)

SYNTACTIC_FOAM = Material(
    name="Syntactic foam (buoyancy)",
    rho=380.0,
    E=1.5e9,
    yield_MPa=25.0,
    strength_knockdown=0.70,
    fatigue_knockdown_1e5=0.70,
    note="Positive-buoyancy filler that survives pressure. The cheapest way to trim.",
)

STRUCTURAL_MATERIALS: dict[str, Material] = {
    m.name.split()[0].lower().replace("-", "_"): m
    for m in (PETG, PETG_CF, CFRP_TUBE, AL6061, MYLAR_MEMBRANE, SYNTACTIC_FOAM)
}
# Explicit, stable keys for the genome to reference.
STRUCTURAL_MATERIALS = {
    "petg": PETG,
    "petg_cf": PETG_CF,
    "cfrp": CFRP_TUBE,
    "alu": AL6061,
    "membrane": MYLAR_MEMBRANE,
    "foam": SYNTACTIC_FOAM,
}


# --------------------------------------------------------------------------
# Energy storage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """A battery chemistry, at pack level (cells + BMS + wiring + case)."""

    name: str
    wh_per_kg: float  # pack-level specific energy
    max_c_cont: float  # continuous discharge, multiples of capacity per hour
    internal_r_mohm_per_wh: float  # pack resistance scaling
    note: str = ""

    def pack_mass(self, wh: float) -> float:
        return wh / self.wh_per_kg

    def max_power(self, wh: float) -> float:
        """Continuous power the pack can deliver, W."""
        return wh * self.max_c_cont


LI_ION_ENERGY = Cell(
    "Li-ion 21700 (energy cell)",
    wh_per_kg=200.0,  # ~260 Wh/kg cell, ~200 at pack level
    max_c_cont=3.0,
    internal_r_mohm_per_wh=0.9,
    note="Best endurance per kg. Cannot supply flapping-flight peak power alone.",
)
LI_PO_POWER = Cell(
    "Li-Po (power cell)",
    wh_per_kg=130.0,
    max_c_cont=25.0,
    internal_r_mohm_per_wh=0.15,
    note="Handles flight power peaks. Poor endurance per kg.",
)
LI_S = Cell(
    "Li-S (developmental)",
    wh_per_kg=350.0,
    max_c_cont=2.0,
    internal_r_mohm_per_wh=2.0,
    note="Optimistic. Included so the optimiser can show what a better cell would buy.",
)

CELLS: dict[str, Cell] = {"liion": LI_ION_ENERGY, "lipo": LI_PO_POWER, "lis": LI_S}


# --------------------------------------------------------------------------
# Actuators
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MotorClass:
    """A scalable motor family.

    Real motors do not scale freely, so each family carries the range over which
    its specific-power figure is honest.  ``specific_power`` is mechanical watts
    per kilogram of motor at continuous rating; ``efficiency_peak`` is at the
    best point of the efficiency map, and the runtime model degrades from there.
    """

    name: str
    specific_power: float  # W_mech / kg
    efficiency_peak: float
    kv_ref: float  # rpm per volt at the reference size
    mass_min: float
    mass_max: float
    note: str = ""


CORELESS_MICRO = MotorClass(
    "Coreless brushed micro",
    specific_power=150.0,
    efficiency_peak=0.55,
    kv_ref=20000.0,
    mass_min=1e-3,
    mass_max=0.02,
    note="What is on the bench today. Brushes wear fast and it will not scale to 15 kg.",
)
BLDC_OUTRUNNER = MotorClass(
    "BLDC outrunner",
    specific_power=3000.0,
    efficiency_peak=0.88,
    kv_ref=900.0,
    mass_min=0.02,
    mass_max=2.0,
    note="The workhorse for the flight actuator at this scale.",
)
BLDC_GEARED = MotorClass(
    "BLDC + planetary reduction",
    specific_power=1200.0,  # gearbox mass and loss included
    efficiency_peak=0.75,
    kv_ref=200.0,
    mass_min=0.05,
    mass_max=3.0,
    note="High torque at low speed: flapping drive, legs, buoyancy pump.",
)

MOTOR_CLASSES: dict[str, MotorClass] = {
    "coreless": CORELESS_MICRO,
    "bldc": BLDC_OUTRUNNER,
    "geared": BLDC_GEARED,
}


# --------------------------------------------------------------------------
# Seawater / environment penalties that are easy to forget
# --------------------------------------------------------------------------

#: Mass fraction added to any part exposed to seawater for sealing, coating and
#: fasteners.  Seals are where amphibious vehicles actually fail.
SEALING_MASS_FRACTION = 0.08

#: Mass of a dynamic shaft seal rated to the design depth, per penetration, kg.
#: Every actuated degree of freedom that crosses the hull boundary costs one.
SHAFT_SEAL_MASS = 0.035

#: Parasitic friction torque of one dynamic shaft seal at depth, N.m.  This is
#: a real and frequently fatal power cost: a machine with 20 sealed joints can
#: spend more power turning its seals than moving.
SHAFT_SEAL_FRICTION = 0.02
