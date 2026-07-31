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

The six here are chosen because they solve the *same* problem -- move through
a fluid -- with mechanically incompatible strategies, and because each one is
strong in a different domain of this mission:

    beetle   bilateral flapper + swimming paddles.  The namesake.  Adequate in
             water, walks, and does not fly -- see below.
    medusa   radial bell, pulsed jet.  Excellent in water, cannot fly at all,
             and is here to occupy the far end of the density axis.
    bat      membrane on articulated digits.  Continuously variable camber and
             area, so it can be a wing in air and a paddle in water without
             changing hardware.
    eel      serial chain of fins driven as a travelling wave.  No wings at all;
             thrust comes from the reverse Karman street the wave sheds.
    ray      wide undulating pectoral membrane.  The batoid solution -- flies
             underwater on the same surfaces it swims with.
    gannet   fixed high-aspect wing, lifting tail, folding shoulder.  The only
             one that flies.

None of them is expected to complete the mission as given.  They are starting
points, and the interesting result is which parts of which plans survive when
the archive mixes them.

A correction, because it was load-bearing and it was wrong.  This list used to
say the beetle was "good in air".  Measured on the air segment it scores 0.118,
and so does everything else that was here: 0.116 to 0.129 across all five,
against 1.000 for a plain fixed-wing machine of the same mass.  Every one of
the original plans is a flapping or undulating solution and not one of them
flies, so the search was being asked to cross from 0.12 to 1.0 with no foothold
anywhere on the far side -- which is exactly the deep valley this file exists to
bridge, and there was nothing to bridge from.  The gannet is that foothold.

Two other things had to be true before it could be one, and neither was.  A
mirrored hinge needs a mirrored axis or a symmetric command rolls the machine
instead of flapping it; and a surface had to be able to be *told* to hold still,
which took a gene, because the pattern generator beat every joint of every
design at 45% of its travel.  Until both were fixed, a fixed wing was not a
thing this representation could express.

Hinge axes
----------
``joint_axis`` is written in the *part's own* frame, whose +X is that limb's
span or length.  On a wing, therefore:

    [1, 0, 0]   about its own span      -> feathering: changes incidence
    [0, 1, 0]   about the fore-aft axis -> the stroke, or a fold
    [0, 0, 1]   about the vertical      -> sweep

This is easy to get backwards and for a long time two plans had it backwards in
opposite directions.  The beetle -- named for flapping, described above as a
flapper -- was given ``[1,0,0]``, so its wings changed incidence in place and
never beat; the stroke axis is worth air 0.118 -> 0.166 and water 0.467 ->
0.658.  The gannet's tailplane had the mirror-image mistake, a fold axis where
an elevator belongs, which gave a tail whose entire range moved glide ratio by
0.02 instead of from 0.26 to 4.91.

The other plans were checked against the same question and left alone.  The
eel's and the ray's axes are not the ones that score best, and that is on
purpose: these plans are here to be *different from each other*, and retuning
an anguilliform swimmer into a feathering one because it gains 0.1 in water
spends the diversity they exist to provide.  A plan gets corrected when its
mechanism and its axis disagree, not when a number goes up.
"""

from __future__ import annotations

import math

import numpy as np

from .cppn import CPPN, SURFACE_INPUTS, SURFACE_OUTPUTS, Connection, Node, new_surface_cppn
from .sdf import BODY_INPUTS, BODY_OUTPUTS
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
# Body volumes
#
# Every archetype's body used to be a capsule, because ``body_cppn`` defaulted
# to -1 and only the ``body_field`` mutation ever set it.  A seeded run
# therefore started with five plans whose bodies were all the same rod, and the
# free-form representation only entered the population if that one operator
# happened to fire and happened to survive selection.  That is why the renders
# came out as sticks: the machinery was there and nothing was using it.
#
# The fields below are hand-wired rather than random, for the same reason the
# surface CPPNs are: a prior that is recognisably a beetle, a bell, a ray costs
# a little diversity and saves thousands of evaluations spent rediscovering that
# a body should enclose a volume.  Mutation dismantles them freely -- these are
# ordinary CPPNs with no protected status.
#
# Coordinates are the part's own normalised frame: x runs base to tip (-1..1),
# y and z are lateral, r = sqrt(y^2+z^2), d = |x|.  Occupancy is ``solid > 0``
# intersected with r <= 1, so writing a shape means writing an inequality.
# --------------------------------------------------------------------------

# Input indices for a body CPPN: x, y, z, r, d, bias.  Outputs: solid, shell.
BX, BY, BZ, BR, BD, BB = 0, 1, 2, 3, 4, 5
SOLID, SHELL = 6, 7


def _body(
    weights: dict[tuple[int, int], float],
    *,
    bias: dict[int, float] | None = None,
    hidden: list[tuple[str, float]] | None = None,
) -> CPPN:
    """A hand-wired body CPPN.

    ``hidden`` adds nodes (activation, bias) after the outputs, numbered from 8
    upward, so a shape that needs a ring or a crease can have one.  Keys in
    ``weights`` are (src_index, dst_index) over inputs 0-5, outputs 6-7 and
    hidden 8+.
    """
    c = CPPN(inputs=list(BODY_INPUTS), outputs=list(BODY_OUTPUTS))
    for _ in BODY_INPUTS:
        c.nodes.append(Node(id=c._new_id(), activation="identity", layer=0))
    bias = bias or {}
    for i, _ in enumerate(BODY_OUTPUTS):
        c.nodes.append(
            Node(id=c._new_id(), activation="identity", bias=bias.get(6 + i, 0.0), layer=2)
        )
    for act, b in hidden or []:
        c.nodes.append(Node(id=c._new_id(), activation=act, bias=b, layer=1))
    c.connections = [Connection(a, b, w) for (a, b), w in weights.items()]
    return c


def _fusiform(taper: float = 0.85, flatten: float = 0.0) -> CPPN:
    """A spindle: fat amidships, fine at both ends.

    ``solid = 1.05 - r - taper*|x| - flatten*|z|``, so occupancy is a body of
    revolution whose radius falls off toward the ends, optionally squashed in z.
    This is the shape a hull wants for the same reason every fish and every
    airship has it: minimum wetted area for a given volume at low drag.
    """
    w = {(BR, SOLID): -1.0, (BD, SOLID): -taper, (BB, SOLID): 1.05}
    hid = []
    if flatten > 0.0:
        hid = [("abs", 0.0)]
        w[(BZ, 8)] = 1.0
        w[(8, SOLID)] = -flatten
    return _body(w, hidden=hid)


def _dome_shell(curve: float = 0.55, radius: float = 0.72, thickness: float = 5.0) -> CPPN:
    """A hollow bell, open at one end.

    A single linear unit cannot express an annulus -- it can only cut space with
    a plane -- so this uses one gaussian hidden node to place a *band* at
    ``r + curve*x = radius``.  Everything inside and outside the band is empty,
    which is what makes it a shell with a cavity rather than a lump, and the
    cavity is what the jet model needs to have somewhere to push from.
    """
    return _body(
        {
            (BR, 8): thickness,
            (BX, 8): thickness * curve,
            (8, SOLID): 1.0,
            (BR, SHELL): 1.0,
        },
        bias={SOLID: -0.42},
        hidden=[("gauss", -thickness * radius)],
    )


def _oblate(squash: float = 3.0, taper: float = 0.5) -> CPPN:
    """A flattened disc: wide in y, thin in z, tapering fore and aft."""
    return _body(
        {
            (BZ, 8): 1.0,
            (8, SOLID): -squash,
            (BD, SOLID): -taper,
            (BB, SOLID): 1.2,
            (BR, SHELL): 0.5,
        },
        hidden=[("abs", 0.0)],
    )


def _cone(base: float = 0.75, slope: float = 0.6) -> CPPN:
    """A tapered nose: broad at the base, fine at the tip."""
    return _body({(BR, SOLID): -1.0, (BX, SOLID): -slope, (BB, SOLID): base})


# --------------------------------------------------------------------------


def beetle() -> Genome:
    """Bilateral flapper with swimming paddles.  The diving beetle."""
    g = Genome()
    g.cppns = [
        # Tapered wing with mild washout: +8 deg at the root falling to +1 at
        # the tip, so the root stalls first.  The twist weights here used to be
        # four times that, which was tuned against an inverted twist sign and
        # left the tip at -22 deg -- the plan only trimmed by pitching the whole
        # machine up 34 deg to compensate.
        _cppn({(U, CHORD): -1.30, (BIAS, CHORD): 0.55, (U, TWIST): -0.20,
               (BIAS, TWIST): 0.23, (BIAS, CAMBER): 0.60, (U, THICK): -0.70,
               (BIAS, THICK): 0.30, (U, DIHED): 0.35}),
        # Rowing blade: widens outboard, blunt, untwisted.
        _cppn({(U, CHORD): 0.55, (BIAS, CHORD): 0.35, (BIAS, CAMBER): 0.25,
               (BIAS, THICK): -0.40}),
    ]
    g.body_cppns = [_fusiform(taper=0.80, flatten=1.6)]
    hull = Part(kind=HULL, length=0.62, radius=0.078, material="petg", joint="none",
                actuated=False, sealed=False, dry_fraction=0.88, body_cppn=0)
    # Axis about the fore-aft direction, so the shoulder *strokes*.  It was
    # along the wing's own span, which is a feathering hinge: the wings changed
    # incidence in place and never beat, so the plan documented here as a
    # bilateral flapper could not flap.  Measured, the stroke axis is worth air
    # 0.118 -> 0.166 and water 0.467 -> 0.658.
    wing = Part(kind=WING, span=1.15, root_chord=0.34, radius=0.013, material="cfrp",
                surface_cppn=0, joint="hinge", joint_axis=np.array([0.0, 1.0, 0.0]),
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
    g.body_cppns = [_dome_shell(curve=0.55, radius=0.72), _fusiform(taper=1.05)]
    bell = Part(kind=BELL, length=0.34, radius=0.24, material="petg", joint="none",
                actuated=False, sealed=False, dry_fraction=0.35,
                jet_area_ratio=0.16, stroke_fraction=0.42, body_cppn=0)
    # The contracting segments *are* the muscle: eight radial flaps that squeeze
    # the cavity.  Each is a BELL so each contributes jet thrust.
    muscle = Part(kind=BELL, length=0.22, radius=0.10, material="petg", joint="hinge",
                  joint_axis=np.array([0.0, 0.0, 1.0]), joint_range=(-0.9, 0.15),
                  motor_class="geared", motor_mass=0.10, gear_ratio=30.0, sealed=True,
                  dry_fraction=0.1, jet_area_ratio=0.14, stroke_fraction=0.5,
                  body_cppn=1)
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
        _cppn({(U, CHORD): -0.35, (BIAS, CHORD): 0.75, (U, TWIST): -0.17,
               (BIAS, TWIST): 0.20, (BIAS, CAMBER): 0.85, (BIAS, THICK): -0.85}),
    ]
    g.body_cppns = [_fusiform(taper=0.95, flatten=0.8)]
    body = Part(kind=HULL, length=0.44, radius=0.062, material="petg", joint="none",
                actuated=False, sealed=False, dry_fraction=0.9, body_cppn=0)
    # Three digits per side, each shorter than the last, each carrying membrane.
    # The digit sweeps fore and aft rather than twisting about its own length,
    # which is what changes the membrane's area and camber -- the property this
    # plan exists to contribute.  Twisting the digit does nothing of the sort;
    # measured, sweeping is worth water 0.503 -> 0.623.
    digit = Part(kind=STRUT, length=0.46, radius=0.010, material="cfrp", joint="hinge",
                 joint_axis=np.array([0.0, 0.0, 1.0]), joint_range=(-0.9, 0.55),
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
    g.body_cppns = [_cone(base=0.80, slope=0.55)]
    head = Part(kind=HULL, length=0.28, radius=0.055, material="petg", joint="none",
                actuated=False, sealed=False, dry_fraction=0.85, body_cppn=0)
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
        _cppn({(U, CHORD): -0.55, (BIAS, CHORD): 0.90, (U, TWIST): -0.14,
               (BIAS, TWIST): 0.17, (BIAS, CAMBER): 0.55, (BIAS, THICK): -0.80,
               (U, DIHED): 0.25}),
    ]
    g.body_cppns = [_oblate(squash=2.6, taper=0.45)]
    disc = Part(kind=HULL, length=0.40, radius=0.070, material="petg", joint="none",
                actuated=False, sealed=False, dry_fraction=0.86, body_cppn=0)
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


def gannet() -> Genome:
    """Fixed high-aspect wing with a lifting tail, folding at the shoulder.

    Added because the five plans above are all flapping or undulating solutions
    and *none of them flies*.  Measured on the air segment: beetle 0.113, bat
    0.135, ray 0.138, against 1.000 for a fixed-wing machine of the same mass.
    The module docstring claimed the beetle was "good in air", and it was not
    true of anything here -- the search was being asked to cross from 0.12 to
    1.0 with no foothold anywhere on the far side, which is exactly the deep
    valley this file exists to bridge.

    The wing does not flap.  It hinges at the root instead, which is a shoulder
    for folding rather than a stroke: extended it is a soaring wing, swept back
    it is a diving one.  That is the gannet's actual trick, and it is what makes
    this a triphibian donor rather than a model aeroplane -- an aerial diver
    needs a wing it can get out of the way at entry, and none of the other plans
    has one.  The tail is the pitch control the fixed-wing configuration needs,
    since a glider with no elevator holds trim only until the first disturbance.
    """
    g = Genome()
    g.cppns = [
        # Soaring wing: tapered, cambered, mild washout so the root stalls first.
        _cppn({(BIAS, CHORD): 0.20, (U, CHORD): -0.50, (BIAS, TWIST): 0.20,
               (U, TWIST): -0.17, (BIAS, CAMBER): 0.70, (BIAS, THICK): -0.40}),
        # Lifting tail on a long arm, small and thin.  Its incidence and the
        # wing station below are a matched pair: they set where this thing
        # trims, and off that pair it does not glide at all -- 4.9 at the
        # balance below, 0.6 six centimetres of wing station away.
        _cppn({(BIAS, CHORD): -0.20, (BIAS, TWIST): 0.60, (BIAS, THICK): -0.60}),
        # Webbed foot: broad, blunt, untwisted.  Paddles and takes weight.
        _cppn({(U, CHORD): 0.45, (BIAS, CHORD): 0.30, (BIAS, THICK): -0.30}),
    ]
    g.body_cppns = [_fusiform(taper=0.85, flatten=1.2)]
    fuselage = Part(kind=HULL, length=1.00, radius=0.070, material="petg",
                    joint="none", actuated=False, sealed=True, dry_fraction=0.90,
                    body_cppn=0)
    # Shoulder hinge with a wide range: this is a fold, not a flap, so the
    # range is asymmetric about extended and the gearing is slow and strong.
    wing = Part(kind=WING, span=1.50, root_chord=0.42, radius=0.015,
                material="cfrp", surface_cppn=0, joint="hinge",
                joint_axis=np.array([0.0, 1.0, 0.0]), joint_range=(-1.5, 0.25),
                motor_class="geared", motor_mass=0.34, gear_ratio=40.0,
                sealed=True, dry_fraction=0.25,
                # A trim surface, not an oscillator: held extended, and folded
                # only when something decides to fold it.
                stroke_amplitude=0.0, neutral=1.0)
    # Axis along the tail's own span, so the hinge changes *incidence*: this is
    # an elevator.  Written as (0,1,0) first, which is the fold axis the
    # shoulder wants and gives a tailplane with dihedral authority and no pitch
    # authority at all -- measured, the whole range moved glide ratio by 0.02.
    # About the span it moves it from 0.26 to 4.91.
    tail = Part(kind=WING, span=0.60, root_chord=0.22, radius=0.010,
                material="cfrp", surface_cppn=1, joint="hinge",
                joint_axis=np.array([1.0, 0.0, 0.0]), joint_range=(-0.35, 0.35),
                motor_class="geared", motor_mass=0.08, gear_ratio=24.0,
                sealed=True, dry_fraction=0.3,
                stroke_amplitude=0.0, neutral=0.5)
    leg = Part(kind=STRUT, length=0.16, radius=0.009, material="cfrp",
               joint="hinge", joint_axis=np.array([0.0, 1.0, 0.0]),
               joint_range=(-0.9, 0.9), motor_class="geared", motor_mass=0.06,
               gear_ratio=25.0, sealed=True, neutral=0.75)
    foot = Part(kind=PADDLE, span=0.18, root_chord=0.13, length=0.18, radius=0.007,
                material="cfrp", surface_cppn=2, joint="hinge",
                joint_axis=np.array([0.0, 0.0, 1.0]), joint_range=(-0.8, 0.8),
                motor_class="geared", motor_mass=0.05, gear_ratio=22.0,
                sealed=True, phase_offset=math.pi)
    g.parts = [fuselage, wing, tail, leg, foot]
    g.edges = [
        Edge(parent=0, child=1, pos_u=0.34, reflect=True),
        Edge(parent=0, child=2, pos_u=1.00, reflect=True),
        Edge(parent=0, child=3, pos_u=0.70, azimuth=-1.1, reflect=True),
        Edge(parent=3, child=4, pos_u=1.0, elevation=0.8),
    ]
    g.battery_wh, g.battery_chem = 210.0, "lipo"
    # Low flap frequency: the shoulder folds, it does not beat.
    g.flap_frequency, g.gas_volume = 0.6, 0.0025
    g.ballast_fraction, g.deadrise_deg = 0.88, 52.0
    g.lineage = ["gannet"]
    return g


BODY_PLANS = {
    "beetle": beetle,
    "medusa": medusa,
    "bat": bat,
    "eel": eel,
    "ray": ray,
    "gannet": gannet,
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
