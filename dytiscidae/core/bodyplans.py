"""Body-plan archetypes: the seeds the search starts from.

Why more than one
-----------------
The first version of this project seeded every run from a single plan -- a hull
with one mirrored wing pair and one mirrored limb pair -- and every design it
produced for hundreds of evaluations was a variation on that plan.  That is not
a search failing to find alternatives; it is a search that was never given a
path to them.  Body plans are separated by deep valleys: every intermediate
between a bilateral flapper and a radial medusa is worse than both, so gradient
and mutation alike stay where they started.

Quality-diversity search is supposed to fix exactly this, but only along the
axes it is given, and only between regions it can reach.  Seeding several
distinct plans is how the archive gets footholds in basins that mutation would
otherwise never cross into.

The five here are chosen because they solve the *same* problem -- move through
a fluid -- with mechanically incompatible strategies, and because each one is
strong in a different domain of this mission:

    beetle   bilateral flapper + swimming paddles.  The namesake.  Good in air,
             adequate in water, walks.
    medusa   radial bell, pulsed jet.  Excellent in water, cannot fly at all,
             and is here to occupy the far end of the density axis.
    bat      membrane on articulated digits.  Continuously variable camber and
             area, so it can be a wing in air and a paddle in water without
             changing hardware.
    eel      serial chain of fins driven as a travelling wave.  No wings at all;
             thrust comes from the reverse Karman street the wave sheds.
    ray      wide undulating pectoral membrane.  The batoid solution -- flies
             underwater on the same surfaces it swims with.

None of them is expected to complete the mission as given.  They are starting
points, and the interesting result is which parts of which plans survive when
the archive mixes them.
"""

from __future__ import annotations

import math

import numpy as np

from .cppn import CPPN, SURFACE_INPUTS, SURFACE_OUTPUTS, Connection, Node, new_surface_cppn
from .genome import (
    BALLAST,
    BELL,
    FIN,
    FOOT,
    HULL,
    MEMBRANE,
    PADDLE,
    STRUT,
    WING,
    Edge,
    Genome,
    Part,
)


def _cppn(weights: dict[tuple[int, int], float]) -> CPPN:
    """A hand-wired surface CPPN.  Keys are (input_index, output_index)."""
    c = CPPN(inputs=list(SURFACE_INPUTS), outputs=list(SURFACE_OUTPUTS))
    for _ in SURFACE_INPUTS:
        c.nodes.append(Node(id=c._new_id(), activation="identity", layer=0))
    for _ in SURFACE_OUTPUTS:
        c.nodes.append(Node(id=c._new_id(), activation="tanh", bias=0.0, layer=2))
    c.connections = [Connection(a, b, w) for (a, b), w in weights.items()]
    return c


# Input indices: 0=u (spanwise station), 1=d (lateral), 2=r (radial), 3=bias
# Output indices: 4=chord, 5=twist, 6=camber, 7=thickness, 8=dihedral
U, D, R, BIAS = 0, 1, 2, 3
CHORD, TWIST, CAMBER, THICK, DIHED = 4, 5, 6, 7, 8


# --------------------------------------------------------------------------


def beetle() -> Genome:
    """Bilateral flapper with swimming paddles.  The diving beetle."""
    g = Genome()
    g.cppns = [
        # Tapered, washed-out wing: chord falls outboard, twist negative.
        _cppn({(U, CHORD): -1.30, (BIAS, CHORD): 0.55, (U, TWIST): -0.85,
               (BIAS, TWIST): 0.10, (BIAS, CAMBER): 0.60, (U, THICK): -0.70,
               (BIAS, THICK): 0.30, (U, DIHED): 0.35}),
        # Rowing blade: widens outboard, blunt, untwisted.
        _cppn({(U, CHORD): 0.55, (BIAS, CHORD): 0.35, (BIAS, CAMBER): 0.25,
               (BIAS, THICK): -0.40}),
    ]
    hull = Part(kind=HULL, length=0.62, radius=0.078, material="petg", joint="none",
                actuated=False, sealed=False, dry_fraction=0.88)
    wing = Part(kind=WING, span=1.15, root_chord=0.34, radius=0.013, material="cfrp",
                surface_cppn=0, joint="hinge", joint_axis=np.array([1.0, 0.0, 0.0]),
                joint_range=(-0.62, 0.62), motor_class="geared", motor_mass=0.42,
                gear_ratio=12.0, sealed=True, dry_fraction=0.25)
    paddle = Part(kind=PADDLE, span=0.30, root_chord=0.10, length=0.30, radius=0.011,
                  material="cfrp", surface_cppn=1, joint="hinge",
                  joint_axis=np.array([0.0, 0.0, 1.0]), joint_range=(-1.0, 1.0),
                  motor_class="geared", motor_mass=0.10, gear_ratio=20.0, sealed=True,
                  phase_offset=math.pi)
    foot = Part(kind=FOOT, length=0.13, radius=0.010, material="petg", joint="hinge",
                joint_axis=np.array([0.0, 0.0, 1.0]), joint_range=(-0.8, 0.8),
                motor_class="geared", motor_mass=0.05, gear_ratio=25.0, sealed=True)
    g.parts = [hull, wing, paddle, foot]
    g.edges = [
        Edge(parent=0, child=1, pos_u=0.42, azimuth=0.16, roll=0.10, reflect=True),
        Edge(parent=0, child=2, pos_u=0.78, azimuth=-0.55, elevation=-0.25, reflect=True),
        Edge(parent=2, child=3, pos_u=1.0, elevation=0.9),
    ]
    g.battery_wh, g.battery_chem = 260.0, "lipo"
    g.flap_frequency, g.gas_volume = 2.2, 0.0035
    g.ballast_fraction, g.deadrise_deg = 0.92, 34.0
    g.lineage = ["beetle"]
    return g


def medusa() -> Genome:
    """Radial bell with pulsed jet propulsion, plus a ring of trailing fins.

    Deliberately incapable of flight.  It is here to hold down the dense,
    water-optimised corner of the archive, and to donate its jet and its radial
    symmetry to whatever the search crosses it with.
    """
    g = Genome()
    g.cppns = [
        # Tentacle-like trailing fin: narrow at the root, broad and floppy aft.
        _cppn({(U, CHORD): 0.40, (BIAS, CHORD): -0.10, (BIAS, THICK): -0.75,
               (BIAS, CAMBER): 0.15}),
    ]
    bell = Part(kind=BELL, length=0.34, radius=0.24, material="petg", joint="none",
                actuated=False, sealed=False, dry_fraction=0.35,
                jet_area_ratio=0.16, stroke_fraction=0.42)
    # The contracting segments *are* the muscle: eight radial flaps that squeeze
    # the cavity.  Each is a BELL so each contributes jet thrust.
    muscle = Part(kind=BELL, length=0.22, radius=0.10, material="petg", joint="hinge",
                  joint_axis=np.array([0.0, 0.0, 1.0]), joint_range=(-0.9, 0.15),
                  motor_class="geared", motor_mass=0.10, gear_ratio=30.0, sealed=True,
                  dry_fraction=0.1, jet_area_ratio=0.14, stroke_fraction=0.5)
    tentacle = Part(kind=FIN, span=0.42, root_chord=0.06, radius=0.008,
                    material="cfrp", surface_cppn=0, joint="hinge",
                    joint_axis=np.array([0.0, 0.0, 1.0]), joint_range=(-0.7, 0.7),
                    motor_class="coreless", motor_mass=0.012, gear_ratio=8.0,
                    sealed=False, phase_offset=0.8)
    g.parts = [bell, muscle, tentacle]
    g.edges = [
        # Eight-fold radial symmetry: one gene, and mirroring could never do it.
        Edge(parent=0, child=1, pos_u=0.85, elevation=-0.5, radial=8, scale=1.0),
        Edge(parent=1, child=2, pos_u=1.0, elevation=0.2, scale=1.0),
    ]
    g.battery_wh, g.battery_chem = 120.0, "liion"
    g.flap_frequency, g.gas_volume = 1.1, 0.0
    g.ballast_fraction, g.deadrise_deg = 0.95, 55.0
    g.lineage = ["medusa"]
    return g


def bat() -> Genome:
    """Membrane stretched across articulated digits.

    The load path runs through the digits rather than one cantilever spar, so
    the surface can change camber and area continuously.  That is the property
    worth having here: the same hardware is a wing in air and a paddle in water,
    with no mechanism to switch between them.
    """
    g = Genome()
    g.cppns = [
        _cppn({(U, CHORD): -0.35, (BIAS, CHORD): 0.75, (U, TWIST): -0.55,
               (BIAS, CAMBER): 0.85, (BIAS, THICK): -0.85}),
    ]
    body = Part(kind=HULL, length=0.44, radius=0.062, material="petg", joint="none",
                actuated=False, sealed=False, dry_fraction=0.9)
    # Three digits per side, each shorter than the last, each carrying membrane.
    digit = Part(kind=STRUT, length=0.46, radius=0.010, material="cfrp", joint="hinge",
                 joint_axis=np.array([1.0, 0.0, 0.0]), joint_range=(-0.9, 0.55),
                 motor_class="geared", motor_mass=0.16, gear_ratio=14.0, sealed=True,
                 dry_fraction=0.0)
    web = Part(kind=MEMBRANE, span=0.44, root_chord=0.30, radius=0.002,
               material="membrane", surface_cppn=0, joint="hinge",
               joint_axis=np.array([0.0, 1.0, 0.0]), joint_range=(-0.5, 0.5),
               actuated=False, sealed=False, dry_fraction=0.0)
    foot = Part(kind=FOOT, length=0.11, radius=0.008, material="petg", joint="hinge",
                joint_axis=np.array([0.0, 0.0, 1.0]), joint_range=(-0.7, 0.7),
                motor_class="geared", motor_mass=0.04, gear_ratio=25.0, sealed=True)
    g.parts = [body, digit, web, foot]
    g.edges = [
        Edge(parent=0, child=1, pos_u=0.4, azimuth=0.1, scale=1.0, reflect=True),
        # A chain needs a *self* edge: an edge from part 0 to part 1 fires only
        # from part 0, so it produces one digit however high its recursion is.
        # This one runs digit -> digit, which is what builds the phalanx.
        Edge(parent=1, child=1, pos_u=1.0, elevation=0.25, scale=0.78, recursion=2),
        Edge(parent=1, child=2, pos_u=0.05, azimuth=0.0, elevation=0.0, scale=1.0),
        Edge(parent=0, child=3, pos_u=0.9, azimuth=-0.7, elevation=-0.4, reflect=True),
    ]
    g.battery_wh, g.battery_chem = 200.0, "lipo"
    g.flap_frequency, g.gas_volume = 3.1, 0.0025
    g.ballast_fraction, g.deadrise_deg = 0.8, 28.0
    g.lineage = ["bat"]
    return g


def eel() -> Genome:
    """A serial chain driven as a travelling wave.  No wings anywhere.

    Thrust comes from the reverse Karman street the undulation sheds.  In air it
    can do nothing at all, which is the point: it marks the opposite corner from
    the bat and forces the archive to represent the trade.
    """
    g = Genome()
    g.cppns = [
        _cppn({(BIAS, CHORD): 0.30, (U, CHORD): -0.15, (BIAS, THICK): -0.55}),
    ]
    head = Part(kind=HULL, length=0.28, radius=0.055, material="petg", joint="none",
                actuated=False, sealed=False, dry_fraction=0.85)
    # One body segment; recursion 5 builds the chain, and the phase offset makes
    # the wave travel down it rather than the whole body flapping in unison.
    segment = Part(kind=FIN, span=0.22, root_chord=0.14, radius=0.009,
                   material="petg_cf", surface_cppn=0, joint="hinge",
                   joint_axis=np.array([0.0, 0.0, 1.0]), joint_range=(-0.75, 0.75),
                   motor_class="geared", motor_mass=0.07, gear_ratio=22.0, sealed=True,
                   dry_fraction=0.15, phase_offset=1.05)
    g.parts = [head, segment]
    g.edges = [
        Edge(parent=0, child=1, pos_u=1.0, elevation=math.pi / 2, scale=1.0),
        # Self edge: each body segment carries the next, so recursion actually
        # lengthens the animal instead of doing nothing.
        Edge(parent=1, child=1, pos_u=1.0, elevation=math.pi / 2, scale=0.92,
             recursion=4),
    ]
    g.battery_wh, g.battery_chem = 150.0, "liion"
    g.flap_frequency, g.gas_volume = 1.8, 0.0008
    g.ballast_fraction, g.deadrise_deg = 0.7, 45.0
    g.lineage = ["eel"]
    return g


def ray() -> Genome:
    """Wide pectoral membranes driven as a spanwise travelling wave.

    The batoid solution, and the most interesting one for this mission: the same
    surfaces that fly the animal underwater are large enough to fly it in air, so
    it is the plan most likely to actually close all three domains.
    """
    g = Genome()
    g.cppns = [
        # Broad, thin, strongly cambered: a disc rather than a wing.
        _cppn({(U, CHORD): -0.55, (BIAS, CHORD): 0.90, (U, TWIST): -0.45,
               (BIAS, CAMBER): 0.55, (BIAS, THICK): -0.80, (U, DIHED): 0.25}),
    ]
    disc = Part(kind=HULL, length=0.40, radius=0.070, material="petg", joint="none",
                actuated=False, sealed=False, dry_fraction=0.86)
    # Four fin rays per side, phase-offset so the wave runs fore to aft.
    # 22 mm carbon: the membranes these carry are half a metre across, and in
    # water that is a serious cantilever no matter how compliant the skin is.
    ray_strut = Part(kind=STRUT, length=0.52, radius=0.015, material="cfrp",
                     joint="hinge", joint_axis=np.array([1.0, 0.0, 0.0]),
                     joint_range=(-0.8, 0.8), motor_class="geared", motor_mass=0.13,
                     gear_ratio=16.0, sealed=True, phase_offset=0.9)
    web = Part(kind=MEMBRANE, span=0.50, root_chord=0.42, radius=0.002,
               material="membrane", surface_cppn=0, joint="none", actuated=False,
               sealed=False, dry_fraction=0.0)
    tail = Part(kind=FIN, span=0.30, root_chord=0.05, radius=0.008, material="cfrp",
                surface_cppn=0, joint="hinge", joint_axis=np.array([0.0, 0.0, 1.0]),
                joint_range=(-0.6, 0.6), motor_class="geared", motor_mass=0.05,
                gear_ratio=20.0, sealed=True, phase_offset=2.4)
    g.parts = [disc, ray_strut, web, tail]
    g.edges = [
        Edge(parent=0, child=1, pos_u=0.35, azimuth=0.0, scale=1.0, reflect=True),
        Edge(parent=1, child=1, pos_u=0.9, azimuth=0.0, elevation=0.35, scale=0.86,
             recursion=3),
        Edge(parent=1, child=2, pos_u=0.1, scale=1.0),
        Edge(parent=0, child=3, pos_u=1.0, elevation=math.pi / 2, scale=1.0),
    ]
    g.battery_wh, g.battery_chem = 230.0, "lipo"
    g.flap_frequency, g.gas_volume = 1.6, 0.003
    g.ballast_fraction, g.deadrise_deg = 0.85, 40.0
    g.lineage = ["ray"]
    return g


BODY_PLANS = {
    "beetle": beetle,
    "medusa": medusa,
    "bat": bat,
    "eel": eel,
    "ray": ray,
}


def sample_body_plan(rng: np.random.Generator, *, perturb: int = 0) -> Genome:
    """Draw one archetype and optionally mutate it."""
    from .genome import mutate

    name = str(rng.choice(list(BODY_PLANS)))
    g = BODY_PLANS[name]()
    g.genome_id = name
    for _ in range(perturb):
        g, _ = mutate(g, rng, n_ops=2)
    return g


def seed_population(rng: np.random.Generator, n: int) -> list[Genome]:
    """A seed set that covers every archetype before it repeats any.

    Round-robin rather than random draw, so a short seeding phase cannot leave a
    whole body plan unrepresented purely by luck -- which with five plans and a
    dozen seeds happens more often than intuition suggests.
    """
    names = list(BODY_PLANS)
    out: list[Genome] = []
    for i in range(n):
        name = names[i % len(names)]
        g = BODY_PLANS[name]()
        g.genome_id = f"{name}{i // len(names)}"
        # The first pass is the archetypes untouched; later passes are perturbed
        # so the archive gets both the clean plan and its neighbourhood.
        if i >= len(names):
            from .genome import mutate

            g, _ = mutate(g, rng, n_ops=int(rng.integers(1, 4)))
        out.append(g)
    return out
