"""The morphology genome: a recursive module graph plus implicit surface fields.

Representation
--------------
Two levels, chosen so that structural search stays productive:

* **Module graph** (Sims 1994).  A small directed graph of ``Part`` specs joined
  by ``Edge`` specs, expanded with per-edge recursion limits and reflection.
  This guarantees a valid kinematic tree no matter what mutation does, which
  matters because an invalid tree is an evaluation that costs time and returns
  nothing.  It also gives free modularity: mutating one part changes every copy
  of it, so the search can discover "four of these legs" in one step.

* **Surface fields** (CPPN, see ``cppn.py``).  Every lifting or paddling surface
  carries an implicit field for chord, twist, camber, thickness and dihedral.
  This is where shapes come from that nobody designed.

Mutation operators are *named*.  Each returns whether it changed anything, and
the name is recorded on the child's lineage, so the curator can run a bandit
over operators and spend its mutation budget on whatever is currently paying
(see ``evolution/curator.py``).  This is the difference between a search that
gets faster as it learns about itself and one that does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .cppn import CPPN, new_surface_cppn

# Part kinds.  Each behaves differently in the fluid solver and the mass budget.
HULL = "hull"  # dry pressure volume: carries battery, avionics, buoyancy
STRUT = "strut"  # structural member, no surface -- a bone, or a digit
WING = "wing"  # stiff lifting surface with its own spar: bird, insect forewing
PADDLE = "paddle"  # high-drag surface tuned for water; still generates lift
FOOT = "foot"  # ground contact element
BALLAST = "ballast"  # variable buoyancy volume
#: A tension-only skin carried by the strut it hangs from, with no spar of its
#: own.  This is the bat and pterosaur architecture: the load path runs through
#: articulated digits rather than a single cantilever, which is why a bat can
#: change its wing's camber and area continuously and a bird cannot.  Expressing
#: it needs a part kind that has area but no bending member.
MEMBRANE = "membrane"
#: A contracting cavity that ejects fluid: medusa bell, squid mantle.  Thrust is
#: momentum flux, rho * Q^2 / A_orifice, so it works underwater and does almost
#: nothing in air.  Nothing in a wing-shaped parameterisation can express this.
BELL = "bell"
#: A slender surface meant to be driven as a travelling wave rather than flapped.
#: Combined with a phase gradient along a serial chain this gives anguilliform
#: swimming and the reverse Karman street that produces its thrust.
FIN = "fin"
PART_KINDS = [HULL, STRUT, WING, PADDLE, FOOT, BALLAST, MEMBRANE, BELL, FIN]

#: Kinds that present an aerodynamic/hydrodynamic surface.
SURFACE_KINDS = (WING, PADDLE, MEMBRANE, FIN)

JOINT_KINDS = ["none", "hinge", "universal"]

MATERIAL_CHOICES = ["petg", "petg_cf", "cfrp", "alu", "membrane", "foam"]
MOTOR_CHOICES = ["bldc", "geared", "coreless"]


@dataclass
class Part:
    """One module type.  May be instantiated many times by the expansion."""

    kind: str = STRUT
    length: float = 0.20  # m, along the part's own +X
    radius: float = 0.02  # m, tube radius (STRUT/HULL) or thickness scale
    span: float = 0.40  # m, for surfaces
    root_chord: float = 0.15  # m, for surfaces
    material: str = "petg"
    surface_cppn: int = -1  # index into Genome.cppns

    joint: str = "hinge"
    joint_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    joint_range: tuple[float, float] = (-1.0, 1.0)  # rad
    actuated: bool = True
    motor_class: str = "bldc"
    motor_mass: float = 0.08  # kg
    gear_ratio: float = 4.0
    sealed: bool = True  # does this axis cross the pressure boundary?

    #: Fraction of this part's internal volume that is sealed gas rather than
    #: free-flooding.  1.0 is a sealed float, 0.0 is an open fairing that fills
    #: with water.  Applies to every kind: a sealed wing is a buoyancy tank, and
    #: whether that helps or hurts depends on which domain the machine is in.
    dry_fraction: float = 0.0

    #: Phase offset of this part's oscillator, radians.  With a serial chain and
    #: a per-part offset the pattern generator can produce a travelling wave
    #: along the body, which is how anguilliform swimmers make thrust.  Fixing
    #: the phase to a linspace, as the first version did, put every undulatory
    #: gait outside the search space.
    phase_offset: float = 0.0
    #: For BELL: orifice area as a fraction of the bell's frontal area.  Jet
    #: thrust goes as 1/A, so a small orifice trades flow rate for velocity.
    jet_area_ratio: float = 0.18
    #: For BELL: fraction of enclosed volume expelled per contraction.
    stroke_fraction: float = 0.35

    def copy(self) -> "Part":
        p = replace(self)
        p.joint_axis = np.array(self.joint_axis, dtype=float)
        return p

    @property
    def is_surface(self) -> bool:
        return self.kind in SURFACE_KINDS

    @property
    def has_own_spar(self) -> bool:
        """Whether this surface carries its own bending member.

        A membrane does not: it is tension-only skin, and its loads pass into
        whatever strut it hangs from.  That distinction is the whole reason to
        have a membrane kind at all.
        """
        return self.kind in (WING, PADDLE, FIN)


@dataclass
class Edge:
    """An attachment of ``child`` onto ``parent``."""

    parent: int
    child: int
    pos_u: float = 1.0  # station along the parent, 0 = base, 1 = tip
    azimuth: float = 0.0  # rad, rotation about the parent's long axis
    elevation: float = 0.0  # rad, tilt away from the parent's long axis
    roll: float = 0.0  # rad, twist of the child about its own axis
    scale: float = 0.8  # size multiplier applied to the child
    reflect: bool = False  # also emit a mirrored copy across the XZ plane
    #: Replicate the child this many times evenly around the parent's long axis.
    #: One gene away from a medusa, a radial limb array, or a ring of fins --
    #: none of which bilateral mirroring can reach at any mutation distance.
    radial: int = 1
    recursion: int = 1  # how many times this edge may be followed

    def copy(self) -> "Edge":
        return replace(self)


@dataclass
class Genome:
    """A complete design.

    Beyond the graph, the global genes are the ones that set the *scale of the
    problem*: how big the machine is, how much energy it carries, how fast it
    beats.  Because the user asked the system to discover its own specification
    rather than be handed one, these are free parameters and mass is an outcome,
    not a constraint.
    """

    parts: list[Part] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    cppns: list[CPPN] = field(default_factory=list)
    root: int = 0

    # --- global design genes ------------------------------------------------
    scale: float = 1.0  # overall linear scale multiplier
    battery_wh: float = 150.0
    battery_chem: str = "liion"
    flap_frequency: float = 4.0  # Hz, base rate of the central pattern generator
    #: Volume of gas carried at surface pressure, m^3.  The diving beetle's
    #: elytral bubble.  Buys buoyancy in air and instability at depth.
    gas_volume: float = 0.004
    #: Fraction of displaced volume that the ballast system can flood.
    ballast_fraction: float = 0.25
    #: Deadrise angle of the underside, degrees.  Sets water-entry survival.
    deadrise_deg: float = 20.0

    # --- bookkeeping --------------------------------------------------------
    lineage: list[str] = field(default_factory=list)
    generation: int = 0
    parent_id: str | None = None
    genome_id: str = ""

    # ------------------------------------------------------------------ basics

    def copy(self) -> "Genome":
        g = Genome(
            parts=[p.copy() for p in self.parts],
            edges=[e.copy() for e in self.edges],
            cppns=[c.copy() for c in self.cppns],
            root=self.root,
            scale=self.scale,
            battery_wh=self.battery_wh,
            battery_chem=self.battery_chem,
            flap_frequency=self.flap_frequency,
            gas_volume=self.gas_volume,
            ballast_fraction=self.ballast_fraction,
            deadrise_deg=self.deadrise_deg,
            lineage=list(self.lineage),
            generation=self.generation,
            parent_id=self.parent_id,
            genome_id=self.genome_id,
        )
        return g

    @property
    def complexity(self) -> int:
        return len(self.parts) + len(self.edges) + sum(c.complexity for c in self.cppns)


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------


def random_genome(rng: np.random.Generator, *, target_scale: float = 1.0) -> Genome:
    """A random but structurally sensible starting design.

    Seeded with a body-plan prior rather than pure noise: a hull, at least one
    reflected surface pair, and one reflected limb pair.  Pure random graphs
    almost never produce anything that moves, and starting from a prior costs
    a little diversity but saves thousands of wasted evaluations.  The prior is
    weak enough that mutation can dismantle it -- surfaces can become paddles,
    limbs can become wings, the whole thing can grow a tail.
    """
    g = Genome(scale=target_scale)

    # A CPPN per surface family.
    g.cppns = [new_surface_cppn(rng) for _ in range(2)]

    hull = Part(
        kind=HULL,
        length=float(rng.uniform(0.35, 0.75)),
        radius=float(rng.uniform(0.06, 0.13)),
        material="petg",
        joint="none",
        actuated=False,
        sealed=False,
        dry_fraction=float(rng.uniform(0.55, 0.9)),
    )
    wing = Part(
        kind=WING,
        span=float(rng.uniform(0.5, 1.4)),
        root_chord=float(rng.uniform(0.12, 0.35)),
        radius=float(rng.uniform(0.008, 0.02)),
        material=str(rng.choice(["cfrp", "petg_cf"])),
        surface_cppn=0,
        dry_fraction=float(rng.uniform(0.4, 0.95)),
        joint="hinge",
        joint_axis=np.array([1.0, 0.0, 0.0]),
        joint_range=(-float(rng.uniform(0.5, 1.2)), float(rng.uniform(0.5, 1.2))),
        actuated=True,
        motor_class="geared",
        motor_mass=float(rng.uniform(0.06, 0.35)),
        gear_ratio=float(rng.uniform(3.0, 20.0)),
        sealed=True,
    )
    limb = Part(
        kind=rng.choice([PADDLE, FOOT]),
        span=float(rng.uniform(0.12, 0.35)),
        root_chord=float(rng.uniform(0.04, 0.12)),
        length=float(rng.uniform(0.10, 0.28)),
        radius=float(rng.uniform(0.006, 0.016)),
        material="petg",
        surface_cppn=1,
        dry_fraction=float(rng.uniform(0.0, 0.7)),
        joint="hinge",
        joint_axis=np.array([0.0, 0.0, 1.0]),
        joint_range=(-0.9, 0.9),
        actuated=True,
        motor_class=str(rng.choice(["geared", "bldc"])),
        motor_mass=float(rng.uniform(0.03, 0.15)),
        gear_ratio=float(rng.uniform(4.0, 30.0)),
        sealed=True,
    )
    g.parts = [hull, wing, limb]
    g.root = 0
    g.edges = [
        Edge(parent=0, child=1, pos_u=float(rng.uniform(0.3, 0.7)),
             azimuth=float(rng.uniform(-0.4, 0.4)), elevation=float(rng.uniform(-0.3, 0.3)),
             scale=1.0, reflect=True, recursion=1),
        Edge(parent=0, child=2, pos_u=float(rng.uniform(0.1, 0.9)),
             azimuth=float(rng.uniform(1.0, 2.2)), elevation=float(rng.uniform(-0.6, 0.2)),
             scale=1.0, reflect=True, recursion=int(rng.integers(1, 3))),
    ]

    g.battery_wh = float(rng.uniform(60.0, 400.0))
    g.battery_chem = str(rng.choice(["liion", "lipo"]))
    g.flap_frequency = float(rng.uniform(1.5, 8.0))
    g.gas_volume = float(rng.uniform(0.0, 0.02))
    g.ballast_fraction = float(rng.uniform(0.1, 0.9))
    g.deadrise_deg = float(rng.uniform(5.0, 50.0))
    g.lineage = ["seed"]
    return g


# --------------------------------------------------------------------------
# Mutation operators
#
# Each takes (genome, rng) and returns True if it changed the genome.  They are
# registered in MUTATION_OPERATORS so the curator can select among them by name
# and attribute outcomes back to the operator that produced them.
# --------------------------------------------------------------------------


def _jitter(x: float, rng: np.random.Generator, rel: float = 0.15, lo: float = 1e-4,
            hi: float = 1e6) -> float:
    return float(np.clip(x * math.exp(rng.normal(0.0, rel)), lo, hi))


def mut_part_dimensions(g: Genome, rng: np.random.Generator) -> bool:
    if not g.parts:
        return False
    p = g.parts[int(rng.integers(len(g.parts)))]
    p.length = _jitter(p.length, rng, 0.18, 0.02, 2.0)
    p.radius = _jitter(p.radius, rng, 0.18, 0.002, 0.30)
    if p.is_surface:
        p.span = _jitter(p.span, rng, 0.20, 0.03, 2.5)
        p.root_chord = _jitter(p.root_chord, rng, 0.20, 0.02, 0.9)
    p.dry_fraction = float(np.clip(p.dry_fraction + rng.normal(0, 0.10), 0.0, 1.0))
    return True


def mut_joint(g: Genome, rng: np.random.Generator) -> bool:
    cand = [p for p in g.parts if p.joint != "none"]
    if not cand:
        return False
    p = cand[int(rng.integers(len(cand)))]
    axis = np.array(p.joint_axis, float) + rng.normal(0, 0.35, 3)
    n = np.linalg.norm(axis)
    p.joint_axis = axis / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
    lo, hi = p.joint_range
    lo = float(np.clip(lo + rng.normal(0, 0.12), -2.6, -0.05))
    hi = float(np.clip(hi + rng.normal(0, 0.12), 0.05, 2.6))
    p.joint_range = (lo, hi)
    return True


def mut_actuator(g: Genome, rng: np.random.Generator) -> bool:
    cand = [p for p in g.parts if p.joint != "none"]
    if not cand:
        return False
    p = cand[int(rng.integers(len(cand)))]
    roll = rng.random()
    if roll < 0.35:
        p.motor_mass = _jitter(p.motor_mass, rng, 0.25, 0.001, 3.0)
    elif roll < 0.6:
        p.gear_ratio = _jitter(p.gear_ratio, rng, 0.30, 1.0, 200.0)
    elif roll < 0.8:
        p.motor_class = str(rng.choice(MOTOR_CHOICES))
    elif roll < 0.9:
        p.actuated = not p.actuated
    else:
        # Moving an axis outside the pressure boundary trades seal drag for
        # having to make the actuator itself pressure tolerant.
        p.sealed = not p.sealed
    return True


def mut_edge_placement(g: Genome, rng: np.random.Generator) -> bool:
    if not g.edges:
        return False
    e = g.edges[int(rng.integers(len(g.edges)))]
    e.pos_u = float(np.clip(e.pos_u + rng.normal(0, 0.12), 0.0, 1.0))
    e.azimuth = float(e.azimuth + rng.normal(0, 0.35))
    e.elevation = float(np.clip(e.elevation + rng.normal(0, 0.25), -1.5, 1.5))
    e.roll = float(e.roll + rng.normal(0, 0.3))
    e.scale = float(np.clip(e.scale * math.exp(rng.normal(0, 0.15)), 0.2, 1.6))
    return True


def mut_radial_symmetry(g: Genome, rng: np.random.Generator) -> bool:
    """Change how many times an attachment is replicated around its parent.

    The single highest-leverage structural gene in the encoding.  Going from
    ``radial=1`` to ``radial=6`` turns a bilateral machine into a medusa in one
    step; a search that can only mirror will never find one, no matter how long
    it runs, because every intermediate is a worse bilateral machine.
    """
    if not g.edges:
        return False
    e = g.edges[int(rng.integers(len(g.edges)))]
    if rng.random() < 0.45:
        e.radial = int(np.clip(e.radial + rng.choice([-2, -1, 1, 2]), 1, 8))
    else:
        e.radial = int(rng.choice([1, 2, 3, 4, 5, 6, 8]))
    if e.radial > 1:
        e.reflect = False  # radial subsumes mirroring
    return True


def mut_phase_gradient(g: Genome, rng: np.random.Generator) -> bool:
    """Shift the oscillator phases, which is what turns flapping into swimming.

    A serial chain of segments whose phases advance along the body produces a
    travelling wave, and a travelling wave sheds a reverse Karman street, which
    is thrust.  With a fixed phase pattern this whole family of gaits is
    unreachable; with this operator it is one mutation away.
    """
    if not g.parts:
        return False
    roll = rng.random()
    if roll < 0.5:
        # A coherent gradient across every part: the travelling-wave move.
        step = float(rng.uniform(-1.4, 1.4))
        for i, part in enumerate(g.parts):
            part.phase_offset = float((part.phase_offset + i * step) % (2 * math.pi))
    else:
        part = g.parts[int(rng.integers(len(g.parts)))]
        part.phase_offset = float((part.phase_offset + rng.normal(0, 0.9)) % (2 * math.pi))
    return True


def mut_jet(g: Genome, rng: np.random.Generator) -> bool:
    """Tune a bell's nozzle and stroke.

    Thrust goes as ``Q^2 / A``, so a smaller orifice is better at fixed flow but
    costs pressure and therefore actuator work.  There is a real optimum and it
    is not obvious, which makes it worth a dedicated operator.
    """
    bells = [p for p in g.parts if p.kind == BELL]
    if not bells:
        return False
    part = bells[int(rng.integers(len(bells)))]
    if rng.random() < 0.5:
        part.jet_area_ratio = float(np.clip(part.jet_area_ratio * math.exp(rng.normal(0, 0.35)),
                                            0.02, 0.9))
    else:
        part.stroke_fraction = float(np.clip(part.stroke_fraction + rng.normal(0, 0.12),
                                             0.05, 0.8))
    return True


def mut_edge_topology(g: Genome, rng: np.random.Generator) -> bool:
    """Rewire, reflect, or change the recursion depth of an attachment.

    Recursion depth is the highest-leverage single gene in the whole encoding:
    raising it by one can turn a two-limb machine into a six-limb one in a
    single mutation, which is the kind of jump that gradient-free search needs
    to escape a body plan that has stopped improving.
    """
    if not g.edges:
        return False
    e = g.edges[int(rng.integers(len(g.edges)))]
    roll = rng.random()
    if roll < 0.35:
        e.recursion = int(np.clip(e.recursion + rng.choice([-1, 1]), 1, 4))
    elif roll < 0.6:
        e.reflect = not e.reflect
    else:
        e.parent = int(rng.integers(len(g.parts)))
        if e.parent == e.child:
            e.child = (e.child + 1) % len(g.parts)
    return True


def mut_add_part(g: Genome, rng: np.random.Generator) -> bool:
    """Grow a new module and attach it somewhere."""
    if len(g.parts) >= 8:
        return False
    kind = str(rng.choice(
        PART_KINDS,
        p=[0.05, 0.13, 0.15, 0.15, 0.10, 0.06, 0.14, 0.10, 0.12],
    ))
    p = Part(
        kind=kind,
        length=float(rng.uniform(0.06, 0.4)),
        radius=float(rng.uniform(0.005, 0.05)),
        span=float(rng.uniform(0.08, 0.9)),
        root_chord=float(rng.uniform(0.03, 0.28)),
        material=str(rng.choice(MATERIAL_CHOICES)),
        joint=str(rng.choice(JOINT_KINDS, p=[0.15, 0.7, 0.15])),
        joint_axis=rng.normal(0, 1, 3),
        joint_range=(-float(rng.uniform(0.3, 1.4)), float(rng.uniform(0.3, 1.4))),
        actuated=bool(rng.random() < 0.75),
        motor_class=str(rng.choice(MOTOR_CHOICES)),
        motor_mass=float(rng.uniform(0.01, 0.3)),
        gear_ratio=float(rng.uniform(1.0, 40.0)),
        sealed=bool(rng.random() < 0.7),
        dry_fraction=float(rng.uniform(0.0, 0.9)),
        phase_offset=float(rng.uniform(0.0, 2 * math.pi)),
        jet_area_ratio=float(rng.uniform(0.05, 0.5)),
        stroke_fraction=float(rng.uniform(0.15, 0.6)),
    )
    n = np.linalg.norm(p.joint_axis)
    p.joint_axis = p.joint_axis / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
    if p.is_surface:
        if rng.random() < 0.5 and g.cppns:
            p.surface_cppn = int(rng.integers(len(g.cppns)))
        else:
            g.cppns.append(new_surface_cppn(rng))
            p.surface_cppn = len(g.cppns) - 1
    g.parts.append(p)
    g.edges.append(
        Edge(
            parent=int(rng.integers(len(g.parts) - 1)),
            child=len(g.parts) - 1,
            pos_u=float(rng.uniform(0.0, 1.0)),
            azimuth=float(rng.uniform(-math.pi, math.pi)),
            elevation=float(rng.uniform(-1.0, 1.0)),
            scale=float(rng.uniform(0.5, 1.1)),
            reflect=bool(rng.random() < 0.45),
            radial=int(rng.choice([1, 1, 1, 2, 3, 4, 6])),
            recursion=int(rng.integers(1, 3)),
        )
    )
    return True


def mut_remove_part(g: Genome, rng: np.random.Generator) -> bool:
    """Delete a module and everything attached below it.

    Removal matters as much as addition: without it, complexity ratchets upward
    forever and every design ends up carrying vestigial mass it cannot shed.
    """
    removable = [i for i in range(len(g.parts)) if i != g.root]
    if len(g.parts) <= 2 or not removable:
        return False
    victim = removable[int(rng.integers(len(removable)))]
    g.edges = [e for e in g.edges if e.parent != victim and e.child != victim]
    g.parts.pop(victim)
    for e in g.edges:
        if e.parent > victim:
            e.parent -= 1
        if e.child > victim:
            e.child -= 1
    if g.root > victim:
        g.root -= 1
    return bool(g.edges)


def mut_part_kind(g: Genome, rng: np.random.Generator) -> bool:
    """Reinterpret a module as a different kind.

    A paddle becoming a wing is a large functional jump at almost no genotypic
    distance, which is exactly the kind of move a triphibian search needs.
    """
    cand = [i for i in range(len(g.parts)) if i != g.root]
    if not cand:
        return False
    p = g.parts[cand[int(rng.integers(len(cand)))]]
    p.kind = str(rng.choice(PART_KINDS))
    if p.is_surface and p.surface_cppn < 0:
        if g.cppns and rng.random() < 0.6:
            p.surface_cppn = int(rng.integers(len(g.cppns)))
        else:
            g.cppns.append(new_surface_cppn(rng))
            p.surface_cppn = len(g.cppns) - 1
    return True


def mut_material(g: Genome, rng: np.random.Generator) -> bool:
    if not g.parts:
        return False
    p = g.parts[int(rng.integers(len(g.parts)))]
    p.material = str(rng.choice(MATERIAL_CHOICES))
    return True


def mut_cppn_weights(g: Genome, rng: np.random.Generator) -> bool:
    if not g.cppns:
        return False
    g.cppns[int(rng.integers(len(g.cppns)))].mutate_weights(rng)
    return True


def mut_cppn_structure(g: Genome, rng: np.random.Generator) -> bool:
    if not g.cppns:
        return False
    c = g.cppns[int(rng.integers(len(g.cppns)))]
    roll = rng.random()
    if roll < 0.35:
        return c.mutate_add_node(rng)
    if roll < 0.7:
        return c.mutate_add_connection(rng)
    if roll < 0.9:
        return c.mutate_activation(rng)
    return c.mutate_toggle(rng)


def mut_global_energy(g: Genome, rng: np.random.Generator) -> bool:
    roll = rng.random()
    if roll < 0.5:
        g.battery_wh = _jitter(g.battery_wh, rng, 0.25, 10.0, 2000.0)
    elif roll < 0.75:
        g.battery_chem = str(rng.choice(["liion", "lipo", "lis"]))
    else:
        g.flap_frequency = _jitter(g.flap_frequency, rng, 0.22, 0.3, 20.0)
    return True


def mut_global_buoyancy(g: Genome, rng: np.random.Generator) -> bool:
    roll = rng.random()
    if roll < 0.4:
        g.gas_volume = float(np.clip(g.gas_volume * math.exp(rng.normal(0, 0.3)), 0.0, 0.15))
    elif roll < 0.75:
        g.ballast_fraction = float(np.clip(g.ballast_fraction + rng.normal(0, 0.07), 0.0, 0.95))
    else:
        g.deadrise_deg = float(np.clip(g.deadrise_deg + rng.normal(0, 6.0), 2.0, 65.0))
    return True


def mut_scale(g: Genome, rng: np.random.Generator) -> bool:
    """Scale the whole machine.

    Kept as its own operator because uniform scaling moves a design a long way
    through behaviour space while preserving everything that already works --
    the classic way out of a local optimum that is merely the wrong size.
    """
    g.scale = float(np.clip(g.scale * math.exp(rng.normal(0, 0.12)), 0.25, 3.0))
    return True


MUTATION_OPERATORS: dict[str, callable] = {
    "radial_symmetry": mut_radial_symmetry,
    "phase_gradient": mut_phase_gradient,
    "jet": mut_jet,
    "part_dimensions": mut_part_dimensions,
    "joint": mut_joint,
    "actuator": mut_actuator,
    "edge_placement": mut_edge_placement,
    "edge_topology": mut_edge_topology,
    "add_part": mut_add_part,
    "remove_part": mut_remove_part,
    "part_kind": mut_part_kind,
    "material": mut_material,
    "cppn_weights": mut_cppn_weights,
    "cppn_structure": mut_cppn_structure,
    "global_energy": mut_global_energy,
    "global_buoyancy": mut_global_buoyancy,
    "scale": mut_scale,
}

#: Operators that change the graph rather than a value.  The curator throttles
#: these separately: structural moves have much higher variance, so they are
#: worth more early and less once the archive is dense.
STRUCTURAL_OPERATORS = {
    "edge_topology", "add_part", "remove_part", "part_kind", "cppn_structure",
    "radial_symmetry",
}


def mutate(
    g: Genome,
    rng: np.random.Generator,
    *,
    operators: list[str] | None = None,
    n_ops: int = 1,
) -> tuple[Genome, list[str]]:
    """Apply ``n_ops`` named operators.  Returns the child and what was applied."""
    child = g.copy()
    child.parent_id = g.genome_id
    child.genome_id = ""
    names = list(operators or MUTATION_OPERATORS)
    applied: list[str] = []
    for _ in range(n_ops):
        name = names[int(rng.integers(len(names)))]
        if MUTATION_OPERATORS[name](child, rng):
            applied.append(name)
    child.lineage = (g.lineage + applied)[-24:]
    return child, applied


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    """Blend two designs.

    Graph crossover between arbitrary topologies is mostly destructive, so this
    stays conservative: the child takes one parent's graph wholesale and imports
    the other's global genes and CPPN weights.  That transfers the two things
    that genuinely are interchangeable between designs -- energy strategy and
    surface shape -- without scrambling a working kinematic tree.
    """
    child = a.copy()
    child.parent_id = a.genome_id
    child.genome_id = ""
    if rng.random() < 0.5:
        child.battery_wh = b.battery_wh
        child.battery_chem = b.battery_chem
    if rng.random() < 0.5:
        child.flap_frequency = b.flap_frequency
    if rng.random() < 0.5:
        child.gas_volume, child.ballast_fraction = b.gas_volume, b.ballast_fraction
    if rng.random() < 0.4:
        child.scale = float(np.sqrt(a.scale * b.scale))
    for i in range(min(len(child.cppns), len(b.cppns))):
        if rng.random() < 0.5:
            child.cppns[i] = CPPN.crossover(child.cppns[i], b.cppns[i], rng)
    child.lineage = (a.lineage + ["crossover"])[-24:]
    return child
