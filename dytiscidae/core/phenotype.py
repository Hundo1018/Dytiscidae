"""Genome -> phenotype: expand the module graph, size everything, weigh it.

This is where a design stops being a data structure and acquires a mass, a
displaced volume, a set of load paths and a power train.  Almost all of the
"physics the user did not think of" lives here, in the mass budget and the
feasibility report, because that is where an attractive-looking design gets
told it weighs 40 kg and cannot close.

Frame conventions for a segment (matching ``physics/fluid.py``):

    local +X  span axis, root -> tip
    local -Y  chord axis, leading edge -> trailing edge
    local +Z  surface normal, = chord x span

Placement of a child relative to its parent is by a direction vector built from
(azimuth, elevation), so the neutral attachment (both zero) is a lateral wing --
the body plan the seed prior starts from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..physics import structure
from ..physics.energy import Actuator, Battery
from ..physics.fluid import BLUFF, WING, PanelSet
from ..physics.materials import STRUCTURAL_MATERIALS, Material
from ..physics.medium import GRAVITY, SEAWATER
from .cppn import SurfaceField, sample_surface
from .genome import (
    BALLAST,
    BELL,
    FIN,
    FOOT,
    HULL,
    MEMBRANE,
    PADDLE,
    STRUT,
    Genome,
    Part,
)
from .genome import WING as WING_KIND

#: Fixed avionics mass: flight controller, IMU, depth sensor, radio, wiring.
AVIONICS_MASS = 0.30

#: Membrane skin areal density, kg/m^2 (50 micron mylar plus adhesive).
SKIN_AREAL_DENSITY = 0.085


def hull_wall(radius: float) -> float:
    """Printable wall thickness for a pressure hull of a given radius, m.

    Sized by external-pressure buckling, which is the binding constraint at
    shallow depth: ``p_cr ~ E * (t/r)^3`` means a PETG hull needs t/r ~ 0.035 to
    survive 1 bar gauge, which for a 90 mm radius is a very printable 3.2 mm.

    This being a *fraction of radius* rather than a constant is the whole point:
    it is what makes a big dry volume expensive and pushes the search toward
    small pressure hulls with everything else flooded.
    """
    return float(np.clip(0.038 * radius, 0.0018, 0.012))


def tube_wall(radius: float) -> float:
    """Wall thickness for a structural tube or wing spar, m.

    t/r ~ 0.12 is where real pultruded carbon and printed tubes live; thinner
    buckles locally, thicker is just wasted mass that never carries bending.
    """
    return float(np.clip(0.12 * radius, 0.0004, 0.004))


def _look_at(direction: np.ndarray, roll: float) -> np.ndarray:
    """Rotation matrix whose +X is ``direction``, rolled about that axis.

    The chord axis is placed by referencing world up, so an untwisted lateral
    surface naturally lies with its chord fore-aft.  ``roll`` then lets evolution
    set incidence and dihedral orientation freely.
    """
    x = np.asarray(direction, float)
    n = np.linalg.norm(x)
    x = x / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(x @ up)) > 0.98:
        up = np.array([1.0, 0.0, 0.0])
    y = np.cross(x, up)
    ny = np.linalg.norm(y)
    y = y / ny if ny > 1e-9 else np.array([0.0, 1.0, 0.0])
    z = np.cross(x, y)
    R = np.stack([x, y, z], axis=1)
    c, s = math.cos(roll), math.sin(roll)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    return R @ Rx


@dataclass
class Segment:
    """One instantiated module in the expanded body."""

    index: int
    part_index: int
    part: Part
    parent: int  # segment index, -1 for the root
    name: str
    depth: int
    mirrored: bool

    # Placement relative to the parent segment's frame.
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Resolved dimensions after the edge scale chain and the global scale.
    length: float = 0.1
    radius: float = 0.01
    span: float = 0.2
    root_chord: float = 0.08
    scale: float = 1.0

    # Filled in during sizing.
    surface: SurfaceField | None = None
    mass: float = 0.0
    #: Outer envelope, m^3.  Drives form drag and added mass -- a free-flooding
    #: fairing is just as big and just as heavy to accelerate as a sealed one.
    volume: float = 0.0
    #: Volume that actually generates net buoyancy, m^3: the solid material plus
    #: whatever rigid gas space is sealed inside.  A free-flooding structure
    #: displaces its envelope but is full of water, so it contributes nothing
    #: beyond its own material.  Conflating the two is what makes every
    #: generated design come out as a balloon.
    volume_buoyant: float = 0.0
    #: BELL only: enclosed cavity volume and nozzle area for the jet model.
    bell_volume: float = 0.0
    orifice_area: float = 0.0
    #: Free-form occupancy field for non-surface parts.  When present it, not a
    #: capsule, is what the mass budget, the buoyancy and the fluid panels read.
    field: object | None = None
    actuator: Actuator | None = None

    @property
    def kind(self) -> str:
        return self.part.kind

    @property
    def is_surface(self) -> bool:
        return self.part.is_surface

    @property
    def axis_length(self) -> float:
        """Extent along the segment's own +X."""
        return self.span if self.is_surface else self.length

    @property
    def material(self) -> Material:
        return STRUCTURAL_MATERIALS.get(self.part.material, STRUCTURAL_MATERIALS["petg"])


@dataclass
class MassBudget:
    """Where the kilograms went.  Reported verbatim in telemetry."""

    structure: float = 0.0
    skin: float = 0.0
    actuators: float = 0.0
    battery: float = 0.0
    avionics: float = 0.0
    seals: float = 0.0
    ballast_water: float = 0.0

    @property
    def dry(self) -> float:
        return (
            self.structure + self.skin + self.actuators + self.battery
            + self.avionics + self.seals
        )

    @property
    def total(self) -> float:
        return self.dry + self.ballast_water

    def as_dict(self) -> dict[str, float]:
        return {
            "structure": self.structure,
            "skin": self.skin,
            "actuators": self.actuators,
            "battery": self.battery,
            "avionics": self.avionics,
            "seals": self.seals,
            "ballast_water": self.ballast_water,
            "dry_total": self.dry,
            "total": self.total,
        }


@dataclass
class Phenotype:
    """A fully sized machine, ready to be compiled to MJCF and evaluated."""

    genome: Genome
    segments: list[Segment]
    budget: MassBudget
    battery: Battery
    report: structure.StructuralReport

    # Derived geometry.
    wing_area: float = 0.0
    max_span: float = 0.0
    body_length: float = 0.0
    frontal_area: float = 0.0
    displaced_volume: float = 0.0  # outer envelope, for drag and added mass
    buoyant_volume: float = 0.0  # rigid, incompressible buoyancy
    gas_volume: float = 0.0  # compressible bladder, collapses with depth

    # Derived counts.
    n_actuated: int = 0
    n_sealed: int = 0

    #: Fastest water entry the hull survives, m/s.  Set by the structural pass.
    max_entry_speed: float = 4.0
    #: Volume of solid material, m^3.  Sealed gas space is buoyant minus this.
    _material_volume: float = 0.0
    #: N/m of buoyancy lost per metre of depth.  Positive means unstable.
    depth_instability: float = 0.0
    #: Per-segment limit on sweep tip speed in water, m/s.  The controller has
    #: to stay under it; the dynamic stress probe checks whether it does.
    max_sweep_speed: dict = field(default_factory=dict)

    @property
    def mass(self) -> float:
        return self.budget.total

    @property
    def wing_loading(self) -> float:
        """N/m^2.  Above ~200 the machine needs a runway it does not have."""
        return self.mass * GRAVITY / max(self.wing_area, 1e-4)

    @property
    def aspect_ratio(self) -> float:
        if self.wing_area < 1e-4:
            return 0.0  # no lifting surface: AR is undefined, not enormous
        return self.max_span**2 / self.wing_area

    @property
    def is_plausible_flyer(self) -> bool:
        """Whether flight load cases apply to this design at all.

        Wing loading above ~450 N/m^2 puts the stall speed past 25 m/s, and an
        aspect ratio below 1.5 means the surfaces are arranged along the body
        rather than across it -- an eel's fins, not a wing.  Neither machine
        will ever fly, so sizing its spars for flight loads and its battery for
        flight power judges it on a mission it does not attempt, and deletes
        whole body plans from the search for failing a test that never applies.

        Flight *competence* remains a performance outcome measured in
        simulation.  This only decides which structural load cases are real.
        """
        return self.wing_area > 1e-3 and self.wing_loading < 450.0 and self.aspect_ratio > 1.5

    @property
    def density_ratio(self) -> float:
        """Mean density relative to seawater.  1.0 is neutrally buoyant."""
        return self.mass / max(self.buoyant_volume * SEAWATER.rho, 1e-6)

    def buoyancy_state(self, ballast_flooded: float = 0.0) -> structure.BuoyancyState:
        """Static buoyancy with the ballast tanks ``ballast_flooded`` full.

        Flooding a tank is accounted for *once*, as displacement lost.  Charging
        it again as added mass -- which reads plausibly and is easy to write --
        doubles the effect and makes every design able to sink.
        """
        return structure.BuoyancyState(
            mass=self.budget.dry,
            displaced_volume=self.buoyant_volume,
            ballast_volume=ballast_flooded * self.ballast_volume,
            gas_volume_surface=self.gas_volume,
        )

    @property
    def ballast_volume(self) -> float:
        """Total buoyancy the machine can shed, expressed as a volume.

        Two independent mechanisms, and leaving the second one out is what made
        the reference design fail to dive by 3%:

        * Flooding the sealed rigid gas space.  Bounded by ``ballast_fraction``
          -- you cannot pump water into solid carbon fibre, and you would not
          flood the battery bay even if you could.
        * **Venting the gas bladder.**  A diving beetle dives by releasing the
          air held under its elytra, and a machine carrying a compressible gas
          store can do exactly the same.  The bladder is the cheapest buoyancy
          to carry and the cheapest to throw away.
        """
        gas_space = max(self.buoyant_volume - self._material_volume, 0.0)
        return self.genome.ballast_fraction * gas_space + self.gas_volume

    def summary(self) -> str:
        return (
            f"m={self.mass:.2f}kg span={self.max_span:.2f}m S={self.wing_area:.3f}m^2 "
            f"AR={self.aspect_ratio:.1f} W/S={self.wing_loading:.0f}N/m^2 "
            f"rho_rel={self.density_ratio:.2f} dof={self.n_actuated} "
            f"E={self.genome.battery_wh:.0f}Wh {self.report.summary()}"
        )


# --------------------------------------------------------------------------
# Expansion
# --------------------------------------------------------------------------


def expand(genome: Genome, *, max_segments: int = 22) -> list[Segment]:
    """Walk the module graph into a concrete tree of segments.

    Recursion is counted per edge along the current path, so an edge with
    ``recursion=3`` produces a chain of three, not a combinatorial explosion.
    The hard cap on segment count is a safety valve: an unlucky mutation can
    make a graph whose expansion is exponential, and an evaluation that never
    starts is worse than a bad one.
    """
    segments: list[Segment] = []
    if not genome.parts:
        return segments

    root_idx = min(genome.root, len(genome.parts) - 1)
    by_parent: dict[int, list[tuple[int, object]]] = {}
    for ei, e in enumerate(genome.edges):
        if 0 <= e.parent < len(genome.parts) and 0 <= e.child < len(genome.parts):
            by_parent.setdefault(e.parent, []).append((ei, e))

    def emit(part_idx: int, parent_seg: int, scale: float, depth: int,
             counts: dict[int, int], mirrored: bool, placement) -> None:
        if len(segments) >= max_segments or depth > 6:
            return
        part = genome.parts[part_idx]
        s = Segment(
            index=len(segments),
            part_index=part_idx,
            part=part,
            parent=parent_seg,
            name=f"seg{len(segments)}_{part.kind}",
            depth=depth,
            mirrored=mirrored,
            scale=scale,
        )
        s.length = part.length * scale
        s.radius = max(part.radius * scale, 0.002)
        s.span = part.span * scale
        s.root_chord = max(part.root_chord * scale, 0.01)
        if placement is not None:
            s.offset, s.rotation = placement
        segments.append(s)
        me = s.index

        for ei, e in by_parent.get(part_idx, ()):
            if counts.get(ei, 0) >= e.recursion:
                continue
            child_counts = dict(counts)
            child_counts[ei] = child_counts.get(ei, 0) + 1
            # Radial replication and bilateral reflection are alternative
            # symmetries, not composable ones: a ring of six limbs mirrored is
            # a ring of six limbs.  Radial wins when set, because reaching a
            # medusa or a radial limb array should cost one gene, not a lucky
            # accumulation of separate attachments.
            n_radial = int(np.clip(e.radial, 1, 8))
            variants = []
            if n_radial > 1:
                for k in range(n_radial):
                    variants.append((1.0, mirrored, e.azimuth + 2.0 * math.pi * k / n_radial))
            else:
                variants.append((1.0, mirrored, e.azimuth))
                if e.reflect:
                    variants.append((-1.0, not mirrored, -e.azimuth))
            for sign, mir, az in variants:
                roll = e.roll * sign
                # Neutral (az=0, el=0) points the child straight out to the side,
                # so the default body plan is a winged one.
                d = np.array(
                    [
                        math.sin(e.elevation),
                        sign * math.cos(e.elevation) * math.cos(az),
                        math.cos(e.elevation) * math.sin(az),
                    ]
                )
                R = _look_at(d, roll)
                off = np.array([e.pos_u * s.axis_length, 0.0, 0.0])
                emit(e.child, me, scale * e.scale, depth + 1, child_counts, mir, (off, R))

    emit(root_idx, -1, genome.scale, 0, {}, False, None)
    return segments


# --------------------------------------------------------------------------
# Sizing and mass
# --------------------------------------------------------------------------


def _surface_of(genome: Genome, seg: Segment) -> SurfaceField:
    idx = seg.part.surface_cppn
    if 0 <= idx < len(genome.cppns):
        cppn = genome.cppns[idx]
    else:
        cppn = genome.cppns[0] if genome.cppns else None
    if cppn is None:
        u = np.linspace(0.0, seg.span, 10)
        n = len(u)
        return SurfaceField(
            u=u,
            chord=np.full(n, seg.root_chord),
            twist=np.zeros(n),
            camber=np.zeros(n),
            thickness=np.full(n, 0.08),
            dihedral=np.zeros(n),
        )
    return sample_surface(
        cppn,
        span=seg.span,
        root_chord=seg.root_chord,
        stations=10,
        lateral_offset=1.0 if seg.mirrored else 0.0,
    )


def build(genome: Genome) -> Phenotype:
    """Expand, size, weigh and structurally check a genome."""
    segments = expand(genome)
    budget = MassBudget()
    report = structure.StructuralReport()

    wing_area = 0.0
    displaced = 0.0
    buoyant = 0.0
    material_volume = 0.0
    n_actuated = 0
    n_sealed = 0
    actuators: list[Actuator] = []

    # --- per-segment sizing -------------------------------------------------
    for s in segments:
        mat = s.material
        if s.is_surface:
            s.surface = _surface_of(genome, s)
            area = float(np.trapezoid(s.surface.chord, s.surface.u))
            if s.part.has_own_spar:
                # Spar: a tube running the span, sized by the part's radius gene.
                wall = tube_wall(s.radius)
                a_sec, _, _ = structure.tube_section(2 * s.radius, wall)
                spar_mass = a_sec * s.span * mat.rho
                skin_mass = 2.0 * area * SKIN_AREAL_DENSITY
            else:
                # A membrane has no bending member of its own; its loads go into
                # whatever strut it hangs from.  It pays for a heavier, tougher
                # skin instead, because it is the structure.
                spar_mass = 0.0
                skin_mass = 2.4 * area * SKIN_AREAL_DENSITY
            s.mass = spar_mass + skin_mass
            budget.structure += spar_mass
            budget.skin += skin_mass
            # A surface displaces its own enclosed volume; treat it as an
            # elliptic section, so V = (pi/4) * t * c integrated over span.
            s.volume = float(np.trapezoid(0.785 * s.surface.thickness * s.surface.chord**2,
                                          s.surface.u))
            # Every surface that makes lift counts, not just the ones called
            # "wing".  A bat's membrane and a ray's pectoral fin are its wings;
            # counting only WING made every non-avian plan report zero area and
            # then fail the flight-power check for having no wing.
            if s.kind in (WING_KIND, MEMBRANE, FIN):
                wing_area += area
        elif not s.is_surface and s.part.body_cppn >= 0:
            # Free-form volume from the implicit field.  Mass is a shell of the
            # real wetted surface rather than a formula for a cylinder, so a
            # lobed or hollow body is charged for the skin it actually has.
            from .sdf import sample_body

            bc = genome.body_cppns[s.part.body_cppn] \
                if s.part.body_cppn < len(genome.body_cppns) else None
            fld = sample_body(bc, length=max(s.length, 0.02), radius=max(s.radius, 0.01))
            s.field = fld
            wall = hull_wall(s.radius) if s.kind in (HULL, BALLAST, BELL) else tube_wall(s.radius)
            s.mass = fld.surface_area * wall * mat.rho
            budget.structure += s.mass
            s.volume = fld.volume
            if s.kind == BELL:
                s.bell_volume = fld.volume
                s.orifice_area = max(
                    math.pi * s.radius**2 * float(np.clip(s.part.jet_area_ratio, 0.02, 0.9)),
                    1e-5,
                )
        elif s.kind == BELL:
            # A contracting cavity.  Its wall must flex, so it is thin and
            # elastomeric rather than a rigid pressure shell.
            wall = max(0.35 * hull_wall(s.radius), 0.0012)
            s.mass = structure.hull_mass(
                radius=s.radius, wall=wall, length=s.length, material=mat
            )
            budget.structure += s.mass
            s.volume = math.pi * s.radius**2 * s.length
            s.bell_volume = s.volume
            s.orifice_area = max(
                math.pi * s.radius**2 * float(np.clip(s.part.jet_area_ratio, 0.02, 0.9)),
                1e-5,
            )
        elif s.kind in (HULL, BALLAST):
            wall = hull_wall(s.radius)
            s.mass = structure.hull_mass(
                radius=s.radius, wall=wall, length=s.length, material=mat
            )
            budget.structure += s.mass
            s.volume = math.pi * s.radius**2 * s.length + (4.0 / 3.0) * math.pi * s.radius**3
        else:  # STRUT, FOOT
            wall = tube_wall(s.radius)
            a_sec, _, _ = structure.tube_section(2 * s.radius, wall)
            s.mass = a_sec * s.length * mat.rho
            budget.structure += s.mass
            s.volume = math.pi * s.radius**2 * s.length

        # Net buoyancy comes from the solid material plus the sealed gas space
        # inside it.  ``dry_fraction`` is the gene that decides whether a part is
        # a sealed float or a free-flooding fairing -- the same shape, radically
        # different behaviour in water, and a real design choice.
        mat_vol = s.mass / max(s.material.rho, 1.0)
        interior = max(s.volume - mat_vol, 0.0)
        s.volume_buoyant = mat_vol + interior * float(np.clip(s.part.dry_fraction, 0.0, 1.0))
        displaced += s.volume
        buoyant += s.volume_buoyant
        material_volume += mat_vol

        if s.part.joint != "none" and s.part.actuated:
            act = Actuator(
                motor_class=s.part.motor_class,
                mass=s.part.motor_mass * max(s.scale, 0.3),
                gear_ratio=s.part.gear_ratio,
                sealed=s.part.sealed,
            )
            s.actuator = act
            actuators.append(act)
            budget.actuators += act.mass
            n_actuated += 1
            if s.part.sealed:
                n_sealed += 1

    # --- global masses ------------------------------------------------------
    battery = Battery(genome.battery_chem, wh=genome.battery_wh)
    budget.battery = battery.mass
    budget.avionics = AVIONICS_MASS
    wetted = budget.structure + budget.skin
    budget.seals = structure.sealing_mass(n_sealed, wetted)

    # A compressible gas bladder carried outside the rigid structure -- the
    # diving beetle's elytral bubble.  It is not clipped by hull volume because
    # it does not live in the hull, but it is not free either: it costs membrane
    # mass, and Boyle's law takes it away exactly when the machine is deepest,
    # which is what makes shallow depth-keeping unstable.
    gas_volume = float(np.clip(genome.gas_volume, 0.0, 0.4))
    budget.structure += gas_volume * 0.9  # bladder + plumbing, kg per m^3-ish

    # --- geometry -----------------------------------------------------------
    # Tip positions in the root frame, to get span and length honestly.
    world_pos: dict[int, np.ndarray] = {}
    world_rot: dict[int, np.ndarray] = {}
    extremes: list[np.ndarray] = []
    for s in segments:
        if s.parent < 0:
            p, R = np.zeros(3), np.eye(3)
        else:
            Rp, pp = world_rot[s.parent], world_pos[s.parent]
            p = pp + Rp @ s.offset
            R = Rp @ s.rotation
        world_pos[s.index] = p
        world_rot[s.index] = R
        extremes.append(p)
        extremes.append(p + R @ np.array([s.axis_length, 0.0, 0.0]))

    pts = np.array(extremes) if extremes else np.zeros((1, 3))
    max_span = float(pts[:, 1].max() - pts[:, 1].min())
    body_length = float(pts[:, 0].max() - pts[:, 0].min())
    height = float(pts[:, 2].max() - pts[:, 2].min())
    frontal_area = max(0.25 * math.pi * max(max_span * 0.15, 0.02) * max(height, 0.05), 0.004)

    pheno = Phenotype(
        genome=genome,
        segments=segments,
        budget=budget,
        battery=battery,
        report=report,
        wing_area=wing_area,
        max_span=max(max_span, 1e-3),
        body_length=max(body_length, 1e-3),
        frontal_area=frontal_area,
        displaced_volume=max(displaced, 1e-6),
        buoyant_volume=max(buoyant, 1e-6),
        gas_volume=gas_volume,
        n_actuated=n_actuated,
        n_sealed=n_sealed,
    )
    pheno._material_volume = material_volume  # type: ignore[attr-defined]
    pheno.world_pos = world_pos  # type: ignore[attr-defined]
    pheno.world_rot = world_rot  # type: ignore[attr-defined]
    pheno.actuators = actuators  # type: ignore[attr-defined]

    _structural_checks(pheno)
    return pheno


def _structural_checks(p: Phenotype) -> None:
    """Run every failure mode that can kill this design, and record the margins.

    Order matters for readability of the report, not for the result: the worst
    margin is what the scorer uses.
    """
    g = p.genome
    mass = p.budget.total
    lift_required = mass * GRAVITY

    surfaces = [s for s in p.segments if s.is_surface and s.surface is not None]
    jets = [s for s in p.segments if s.kind == BELL and s.bell_volume > 0]
    if not surfaces and not jets:
        # No surface and no jet means no way to move through any fluid at all.
        # A design with only a jet is fine -- it is a squid.
        p.report.add(
            structure.Check("has_propulsor", applied=1.0, allowable=0.0,
                            note="no lifting surface and no jet")
        )
    by_index = {seg.index: seg for seg in p.segments}
    for s in surfaces:
        # Which member actually reacts this surface's loads?
        #
        # A wing, paddle or fin carries its own spar.  A membrane does not: it is
        # 2 mm of mylar with a section modulus of 4 mm^3, and checking it against
        # its own geometry reports a 300x overload for every membraned design --
        # which silently deleted the bat and the ray from the search entirely.
        # Its loads go into the digit it is stretched across, and that is the
        # whole structural point of the architecture.
        carrier = s
        if not s.part.has_own_spar:
            parent = by_index.get(s.parent)
            if parent is not None and not parent.is_surface:
                carrier = parent
            else:
                continue  # membrane hanging off a membrane: nothing to check

        wall = tube_wall(carrier.radius)
        share = float(np.trapezoid(s.surface.chord, s.surface.u)) / max(p.wing_area, 1e-6)
        share = min(max(share, 0.05), 1.0)
        if not p.is_plausible_flyer:
            # Not a flyer: skip the flight load case, but the water sweep and
            # inertial checks below still apply -- those happen in every domain.
            structure.hydrodynamic_sweep_check(
                span=s.span, chord_distribution=s.surface.chord,
                span_stations=s.surface.u, outer_d=2 * carrier.radius, wall=wall,
                material=carrier.material, compliant=s.kind in (FIN, MEMBRANE),
                report=p.report,
            )
            p.max_sweep_speed[s.index] = structure.max_sweep_tip_speed(
                chord_distribution=s.surface.chord, span_stations=s.surface.u,
                span=s.span, outer_d=2 * carrier.radius, wall=wall,
                material=carrier.material, compliant=s.kind in (FIN, MEMBRANE),
            )
            continue
        area_s = float(np.trapezoid(s.surface.chord, s.surface.u))
        # Most lift this surface could ever make: CL_max at a generous airspeed.
        # 30 m/s is well past anything a machine of this class will reach, so it
        # is a bound rather than an assumption about how fast it flies.
        generable = 1.8 * 0.5 * 1.225 * 30.0**2 * area_s
        structure.spar_check(
            lift_n=lift_required * share,
            semi_span=s.span,
            outer_d=2 * carrier.radius,
            wall=wall,
            material=carrier.material,
            load_factor=3.0,
            cycles=1e5,
            generable_lift_n=generable,
            report=p.report,
        )
        # Every surface that can end up in water gets checked against the
        # hydrodynamic sweep load, which is the largest load case in the whole
        # machine and the one an air-only spar check never sees.
        # Note ``carrier``, not ``s``.  A membrane has no section of its own to
        # check -- it is 2 mm of mylar, section modulus 4 mm^3 -- so checking it
        # against its own geometry reports a 300x overload for every membraned
        # design and silently deletes the bat and ray body plans from the search.
        # The load goes into the digit.
        structure.hydrodynamic_sweep_check(
            span=s.span,
            chord_distribution=s.surface.chord,
            span_stations=s.surface.u,
            outer_d=2 * carrier.radius,
            wall=wall,
            material=carrier.material,
            compliant=s.kind in (FIN, MEMBRANE),
            report=p.report,
        )
        p.max_sweep_speed[s.index] = structure.max_sweep_tip_speed(
            chord_distribution=s.surface.chord, span_stations=s.surface.u,
            span=s.span, outer_d=2 * carrier.radius, wall=wall,
            material=carrier.material, compliant=s.kind in (FIN, MEMBRANE),
        )
        structure.flapping_inertial_check(
            wing_mass=s.mass,
            semi_span=s.span,
            flap_freq=g.flap_frequency,
            flap_amplitude_rad=abs(s.part.joint_range[1] - s.part.joint_range[0]) / 2.0,
            outer_d=2 * carrier.radius,
            wall=wall,
            material=carrier.material,
            report=p.report,
        )

    hulls = [s for s in p.segments if s.kind in (HULL, BALLAST)]
    for s in hulls:
        # Design depth is a free variable: check the hull at the depth its own
        # buoyancy system could actually reach.
        structure.hull_pressure_check(
            depth_m=12.0,
            radius=s.radius,
            wall=hull_wall(s.radius),
            length=s.length,
            material=s.material,
            report=p.report,
        )

    # Water entry, checked on the hull underside rather than on a wing.
    #
    # Checking it on the membrane instead makes *every* design infeasible: an
    # unsupported 100 mm panel meeting water at 8 m/s sees ~24 bar, which no
    # skin survives.  That is not a modelling artefact, it is the reason real
    # amphibious aircraft fold or feather their surfaces before entry -- so the
    # wing is not the structure that takes this load.  The hull is.
    #
    # The nominal 4 m/s here sizes the hull for a *controlled* entry.  Whether a
    # given controller actually achieves one is not knowable from the geometry,
    # so the peak slam that really occurs is measured in the dynamic episode and
    # checked against ``max_entry_speed`` by the scorer.
    if hulls:
        big_hull = max(hulls, key=lambda s: s.radius)
        structure.water_entry_check(
            impact_speed=4.0,
            deadrise_deg=g.deadrise_deg,
            radius=big_hull.radius,
            wall=hull_wall(big_hull.radius),
            material=big_hull.material,
            report=p.report,
        )
        p.max_entry_speed = _survivable_entry_speed(big_hull, g.deadrise_deg)
    else:
        p.max_entry_speed = 2.0

    # Buoyancy envelope.  Both of these are stated as volumes rather than as
    # yes/no, so that a design which is close to working reads as *close*.  A
    # binary check here is what makes an entire first generation score zero with
    # no gradient to climb, which is exactly the failure mode this whole report
    # is built to avoid.
    rho_w = SEAWATER.rho
    need_displacement = p.budget.dry / rho_w  # m^3 to float at all
    have_displacement = p.buoyant_volume + p.gas_volume
    p.report.add(
        structure.Check(
            "positive_buoyancy_available",
            applied=need_displacement,
            allowable=have_displacement,
            unit="m^3",
            note=(
                f"needs {need_displacement*1e3:.1f} L, has {have_displacement*1e3:.1f} L "
                f"(net {p.buoyancy_state(0.0).net_buoyancy(0.0):+.0f} N)"
            ),
        )
    )
    # To submerge, the ballast system must be able to shed the excess.
    excess = max(have_displacement - need_displacement, 0.0)
    p.report.add(
        structure.Check(
            "can_submerge",
            applied=excess,
            allowable=p.ballast_volume,
            unit="m^3",
            note=(
                f"must shed {excess*1e3:.1f} L, tanks hold {p.ballast_volume*1e3:.1f} L "
                f"(net {p.buoyancy_state(1.0).net_buoyancy(0.0):+.0f} N flooded)"
            ),
        )
    )
    # Depth stability: a machine whose buoyancy falls with depth will run away
    # downward unless it pumps continuously.  Not a failure, but the power cost
    # of holding depth is charged against it in the mission.
    p.depth_instability = -p.buoyancy_state(0.5).depth_stability(5.0)
    # Battery must be able to deliver flight power without exceeding its C rate
    # -- but only for designs that are actually trying to fly.  A medusa has no
    # lifting surface and will never leave the water; that is a *performance*
    # outcome, scored as zero air competence, not a structural impossibility.
    # Treating it as one made every non-flying body plan infeasible before it
    # was ever simulated, which is precisely how a search space gets quietly
    # narrowed back to the one plan its author had in mind.
    # Wing loading above ~450 N/m^2 puts the stall speed past 25 m/s, which no
    # machine in this class reaches.  Such a design is not a flyer, so judging
    # its battery on flight power is judging it on a mission it never flies.
    if p.is_plausible_flyer:
        p.report.add(
            structure.Check(
                "battery_power_rating",
                applied=max(_flight_power_estimate(p), 1.0),
                allowable=max(p.battery.p_max, 1.0),
                unit="W",
                note=f"{p.battery.cell.name} {p.genome.battery_wh:.0f}Wh",
            )
        )


def _survivable_entry_speed(hull: Segment, deadrise_deg: float) -> float:
    """Fastest water entry this hull survives, m/s.

    Inverts the plate-bending check: slam pressure scales as v^2, so the
    allowable speed scales as sqrt(allowable stress).  The dynamic scorer
    compares the peak entry speed actually flown against this number, which
    turns "does it break on landing" into a controller problem rather than a
    geometry problem -- which is what it is.
    """
    wall = hull_wall(hull.radius)
    allow = hull.material.allowable_stress(cycles=1e3)
    dr = math.radians(max(deadrise_deg, 1.0))
    k = min((math.pi / math.tan(dr)) ** 2, 250.0)
    # Curved shell: sigma = p * r / t with p = 0.5 * rho * v^2 * k.
    denom = 0.5 * SEAWATER.rho * k * hull.radius / max(wall, 1e-4)
    return float(np.clip(math.sqrt(allow / max(denom, 1e-9)), 0.2, 30.0))


def _flight_power_estimate(p: Phenotype) -> float:
    from ..physics.energy import cruise_power_air

    v = max(6.0, math.sqrt(2 * p.mass * GRAVITY / (1.225 * max(p.wing_area, 1e-3) * 1.2)))
    power, _ = cruise_power_air(
        mass=p.mass, span=p.max_span, wing_area=p.wing_area, speed=min(v, 30.0)
    )
    return float(min(power, 1e6))


# --------------------------------------------------------------------------
# Fluid discretisation
# --------------------------------------------------------------------------


def build_panels(p: Phenotype, model, body_name_prefix: str = "") -> PanelSet:
    """Discretise the phenotype into the strips the fluid solver integrates.

    Every segment contributes: surfaces become spanwise wing strips, everything
    else becomes a single bluff element carrying its displaced volume.  The
    bluff elements are what give the hull its buoyancy and its drag, so leaving
    them out would make the machine both weightless in water and frictionless.
    """
    import mujoco

    sets: list[PanelSet] = []
    for s in p.segments:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name_prefix + s.name)
        if bid < 0:
            continue
        if s.is_surface and s.surface is not None:
            sf = s.surface
            n = len(sf.u)
            if n < 2:
                continue
            # Strip widths from the station spacing.
            edges = np.concatenate([[sf.u[0]], 0.5 * (sf.u[1:] + sf.u[:-1]), [sf.u[-1]]])
            dr = np.diff(edges)
            keep = dr > 1e-5
            if not keep.any():
                continue
            u, chord, twist = sf.u[keep], sf.chord[keep], sf.twist[keep]
            thick, dr = sf.thickness[keep], dr[keep]
            m = len(u)

            pos = np.zeros((m, 3))
            pos[:, 0] = u  # along the span
            # Strip centroid sits at the quarter chord, aft along local -Y.
            pos[:, 1] = -0.25 * chord

            span_ax = np.tile(np.array([1.0, 0.0, 0.0]), (m, 1))
            # Chord axis is local -Y, rotated by the local geometric twist about
            # the span axis.  This is how the CPPN's twist field becomes real.
            ct, st = np.cos(twist), np.sin(twist)
            chord_ax = np.stack([np.zeros(m), -ct, -st], axis=1)

            ar = s.span**2 / max(float(np.trapezoid(sf.chord, sf.u)), 1e-6)
            sets.append(
                PanelSet(
                    body_id=np.full(m, bid),
                    pos_local=pos,
                    span_local=span_ax,
                    chord_local=chord_ax,
                    chord=chord,
                    dr=dr,
                    volume=0.785 * thick * chord**2 * dr,
                    volume_buoyant=0.785 * thick * chord**2 * dr
                    * (s.volume_buoyant / max(s.volume, 1e-9)),
                    half_height=np.maximum(0.5 * thick * chord, 2e-3),
                    kind=np.full(m, WING),
                    aspect_ratio=np.full(m, np.clip(ar, 0.5, 25.0)),
                    cd_bluff=np.zeros(m),
                    pitch_axis=np.full(m, 0.25),
                )
            )
        else:
            cd = {HULL: 0.20, BALLAST: 0.25, STRUT: 0.9, FOOT: 1.1}.get(s.kind, 0.6)
            sets.append(
                PanelSet(
                    body_id=np.array([bid]),
                    pos_local=np.array([[0.5 * s.length, 0.0, 0.0]]),
                    span_local=np.array([[1.0, 0.0, 0.0]]),
                    chord_local=np.array([[0.0, -1.0, 0.0]]),
                    chord=np.array([max(2 * s.radius, 0.01)]),
                    dr=np.array([max(s.length, 0.01)]),
                    volume=np.array([s.volume]),
                    volume_buoyant=np.array([s.volume_buoyant]),
                    half_height=np.array([max(s.radius, 5e-3)]),
                    kind=np.array([BLUFF]),
                    aspect_ratio=np.array([1.0]),
                    cd_bluff=np.array([cd]),
                    pitch_axis=np.array([0.5]),
                )
            )
    return PanelSet.concat(sets)
