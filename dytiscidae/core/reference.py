"""A hand-designed reference machine.

This exists for two reasons.

First, **falsification**.  A generative pipeline whose constraints admit no
solution at all is indistinguishable, from the outside, from one that is merely
searching badly: both report zero feasible designs forever.  Building one
machine by hand and showing it passes proves the feasible set is non-empty, and
if it ever stops passing, a physics change has broken the space rather than the
search having got worse.

Second, **seeding**.  The reference and its perturbations give MAP-Elites a
foothold in the feasible region, which is worth thousands of random evaluations.

The design itself is a diving beetle rendered as a machine, which is the body
plan the repository is named after: a sealed streamlined hull, a pair of large
low-frequency flapping wings for cruise, a pair of hair-fringed swimming paddles
that double as landing legs, and a compressible gas store for surface buoyancy
that is vented to dive.
"""

from __future__ import annotations

import numpy as np

from .genome import BALLAST, FOOT, HULL, PADDLE, STRUT, WING, Edge, Genome, Part
from .cppn import CPPN, Connection, Node, SURFACE_INPUTS, SURFACE_OUTPUTS


def _tapered_wing_cppn() -> CPPN:
    """A CPPN hand-wired to produce a tapered, washed-out wing.

    Written out explicitly rather than evolved so the reference is reproducible.
    Chord falls toward the tip, twist is negative (washout, so the tip stalls
    last), thickness falls outboard.  Evolution is free to destroy all of this.
    """
    c = CPPN(inputs=list(SURFACE_INPUTS), outputs=list(SURFACE_OUTPUTS))
    for _ in SURFACE_INPUTS:
        c.nodes.append(Node(id=c._new_id(), activation="identity", layer=0))
    for _ in SURFACE_OUTPUTS:
        c.nodes.append(Node(id=c._new_id(), activation="tanh", bias=0.0, layer=2))
    u, d, r, bias = 0, 1, 2, 3
    chord, twist, camber, thick, dihedral = 4, 5, 6, 7, 8
    c.connections = [
        Connection(u, chord, -1.30),  # taper toward the tip
        Connection(bias, chord, 0.55),
        Connection(u, twist, -0.85),  # washout
        Connection(bias, twist, 0.10),
        Connection(bias, camber, 0.60),  # modest constant camber
        Connection(u, thick, -0.70),
        Connection(bias, thick, 0.30),
        Connection(u, dihedral, 0.35),  # slight upward bend outboard
        Connection(bias, dihedral, 0.0),
    ]
    return c


def _paddle_cppn() -> CPPN:
    """A broad, blunt, untwisted surface: a rowing blade, not a wing."""
    c = CPPN(inputs=list(SURFACE_INPUTS), outputs=list(SURFACE_OUTPUTS))
    for _ in SURFACE_INPUTS:
        c.nodes.append(Node(id=c._new_id(), activation="identity", layer=0))
    for _ in SURFACE_OUTPUTS:
        c.nodes.append(Node(id=c._new_id(), activation="tanh", bias=0.0, layer=2))
    u, bias = 0, 3
    chord, twist, camber, thick, dihedral = 4, 5, 6, 7, 8
    c.connections = [
        Connection(u, chord, 0.55),  # widens outboard, like a swimming blade
        Connection(bias, chord, 0.35),
        Connection(bias, twist, 0.0),
        Connection(bias, camber, 0.25),
        Connection(bias, thick, -0.40),
        Connection(bias, dihedral, 0.0),
    ]
    return c


def reference_genome() -> Genome:
    """The hand-designed diving beetle.

    Sizing rationale, in the order the constraints bite:

    * **Flap frequency 2.2 Hz.**  Inertial root moment scales with f^2, and at
      this span anything above ~3 Hz breaks the spar before it flies.  Large
      flapping animals are all slow for the same reason.
    * **Carbon spars, 26 mm diameter -- on the paddles too.**  Printed polymer
      cannot carry the root moment at this span.  The paddles are the
      counter-intuitive part: they are short, so a light printed tube looks
      adequate, but they work in water where the dynamic pressure of a given
      sweep is 840x its value in air.  An 8 mm PETG-CF paddle spar limits the
      sweep to 0.7 m/s at the tip, which is too slow to swim with.
    * **Small hull, big free-flooding fairing.**  The hull is sized for the
      battery and avionics only.  Everything else floods, because sealed volume
      is buoyancy that then has to be carried, pumped out, and structurally
      supported against 1 bar.
    * **Li-Po, not Li-ion.**  Li-ion has the endurance but cannot deliver the
      flight power peak; the pack is sized by power, not by energy, which is the
      usual and slightly disappointing answer for flapping flight.
    """
    g = Genome()
    g.cppns = [_tapered_wing_cppn(), _paddle_cppn()]

    hull = Part(
        kind=HULL,
        length=0.62,
        radius=0.078,
        material="petg",
        joint="none",
        actuated=False,
        sealed=False,
        dry_fraction=0.88,
    )
    wing = Part(
        kind=WING,
        span=1.15,
        root_chord=0.34,
        radius=0.013,  # 26 mm spar
        material="cfrp",
        surface_cppn=0,
        joint="hinge",
        joint_axis=np.array([1.0, 0.0, 0.0]),  # flap about the fore-aft axis
        joint_range=(-0.62, 0.62),
        actuated=True,
        motor_class="geared",
        motor_mass=0.42,
        gear_ratio=12.0,
        sealed=True,
        dry_fraction=0.25,  # mostly free-flooding: cheap to dive with
    )
    paddle = Part(
        kind=PADDLE,
        span=0.30,
        root_chord=0.10,
        length=0.30,
        # Carbon, and thicker than instinct suggests. A paddle works in water,
        # where dynamic pressure is 840x what the same motion costs in air, so
        # it needs a spar in the same class as the wing's -- not the light
        # printed tube that looks adequate for a 300 mm limb.
        radius=0.011,
        material="cfrp",
        surface_cppn=1,
        joint="hinge",
        joint_axis=np.array([0.0, 0.0, 1.0]),  # sweep in the horizontal plane
        joint_range=(-1.0, 1.0),
        actuated=True,
        motor_class="geared",
        motor_mass=0.10,
        gear_ratio=20.0,
        sealed=True,
        dry_fraction=0.0,
    )
    foot = Part(
        kind=FOOT,
        length=0.13,
        radius=0.010,
        material="petg",
        joint="hinge",
        joint_axis=np.array([0.0, 0.0, 1.0]),
        joint_range=(-0.8, 0.8),
        actuated=True,
        motor_class="geared",
        motor_mass=0.05,
        gear_ratio=25.0,
        sealed=True,
        dry_fraction=0.0,
    )

    g.parts = [hull, wing, paddle, foot]
    g.root = 0
    g.edges = [
        # Wings, reflected, mounted mid-hull and slightly above the centreline.
        Edge(parent=0, child=1, pos_u=0.42, azimuth=0.16, elevation=0.0,
             roll=0.10, scale=1.0, reflect=True, recursion=1),
        # Swimming paddles, reflected, mounted aft and low.
        Edge(parent=0, child=2, pos_u=0.78, azimuth=-0.55, elevation=-0.25,
             roll=0.0, scale=1.0, reflect=True, recursion=1),
        # A distal segment on each paddle: the tarsus that touches the ground.
        Edge(parent=2, child=3, pos_u=1.0, azimuth=0.0, elevation=0.9,
             roll=0.0, scale=1.0, reflect=False, recursion=1),
    ]

    g.scale = 1.0
    g.battery_wh = 260.0
    g.battery_chem = "lipo"  # sized by power, not energy
    g.flap_frequency = 2.2  # inertial reversal is the binding constraint
    g.gas_volume = 0.0035  # small vented store for surface trim
    g.ballast_fraction = 0.92
    g.deadrise_deg = 34.0
    g.lineage = ["reference"]
    g.genome_id = "reference"
    return g


def reference_variants(rng: np.random.Generator, n: int = 24) -> list[Genome]:
    """Perturbations of the reference, for seeding the archive.

    Deliberately wide: the point is to scatter feasible-ish designs across the
    behaviour space, not to stay near a single optimum.
    """
    from .genome import mutate

    out = [reference_genome()]
    for _ in range(n - 1):
        g = reference_genome()
        child, _ = mutate(g, rng, n_ops=int(rng.integers(1, 4)))
        out.append(child)
    return out
