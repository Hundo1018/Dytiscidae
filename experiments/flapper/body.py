"""The flapping triphibian's body, built by hand (ROADMAP 2026-11-12, the goal
redirected: one wing for air, water and land).

`flapper(...)` is a genome the existing pipeline builds: a hull and one pair
of wings on universal shoulders (stroke about the body's long axis, feather
about the span).  Everything is a keyword so the capability probes can sweep
it; the defaults are the current design and change as the probes say.
"""
import math

import numpy as np

from dytiscidae.core.bodyplans import _fusiform
from dytiscidae.core.genome import HULL, WING, Edge, Genome, Part

DEFAULTS = dict(
    hull_length=0.16, hull_radius=0.018, hull_dry=0.9,
    wing_length=0.25,        # m, shoulder to tip (one wing)
    wing_chord=0.085,        # m
    wing_material="membrane",
    wing_radius=0.0015,      # m, thickness scale
    shoulder_u=0.65,         # station along the hull, 0 tail, 1 nose
    stroke_range=1.4,        # rad either way
    motor_mass=0.004, gear_ratio=6.0, motor_class="coreless",
    battery_wh=0.6, battery_chem="lipo",
    hand_fraction=0.0,       # share of the wing's length that is the outer panel (0: one panel)
    wrist_motor_mass=0.002,
    hand_compliance=0.0,     # > 0: the wrist is a passive spring hinge of this compliance
)


def flapper(**over) -> Genome:
    e = dict(DEFAULTS)
    e.update(over)
    g = Genome()
    g.cppns = []
    g.body_cppns = [_fusiform(taper=0.9)]
    hull = Part(kind=HULL, length=float(e["hull_length"]), radius=float(e["hull_radius"]),
                material="petg", joint="none", actuated=False, sealed=True,
                dry_fraction=float(e["hull_dry"]), body_cppn=0)
    wing = Part(kind=WING, length=float(e["wing_length"]), span=float(e["wing_length"]),
                root_chord=float(e["wing_chord"]), radius=float(e["wing_radius"]),
                material=str(e["wing_material"]), joint="universal",
                joint_axis=np.array([0.0, 1.0, 0.0]),
                joint_range=(-float(e["stroke_range"]), float(e["stroke_range"])),
                actuated=True, motor_class=str(e["motor_class"]),
                motor_mass=float(e["motor_mass"]), gear_ratio=float(e["gear_ratio"]),
                sealed=False, stroke_amplitude=0.0, neutral=0.5,
                feather_lead=math.pi / 2)
    g.parts = [hull, wing]
    g.edges = [Edge(parent=0, child=1, pos_u=float(e["shoulder_u"]), elevation=0.0, roll=0.0,
                    scale=1.0, reflect=True)]
    if float(e["hand_fraction"]) > 0.0:
        # Two panels: the inner "arm" keeps the shoulder, the outer "hand" turns
        # about the span on a wrist at the arm's tip.  The wing's length and
        # chord are unchanged, so the area is the one-panel wing's.
        h = float(e["hand_fraction"]) * float(e["wing_length"])
        wing.length = wing.span = float(e["wing_length"]) - h
        hand = Part(kind=WING, length=h, span=h, root_chord=float(e["wing_chord"]),
                    radius=float(e["wing_radius"]), material=str(e["wing_material"]),
                    joint="hinge", joint_axis=np.array([1.0, 0.0, 0.0]),
                    joint_range=(-1.0, 1.0), actuated=True, motor_class=str(e["motor_class"]),
                    motor_mass=float(e["wrist_motor_mass"]), gear_ratio=float(e["gear_ratio"]),
                    sealed=False, stroke_amplitude=0.0, neutral=0.5)
        if float(e["hand_compliance"]) > 0.0:
            # A passive wrist: no motor, the hand pitches about its leading edge
            # against a spring (`Part.pitch_compliance`, the package's own
            # pseudo-rigid-body hinge).
            hand.joint, hand.actuated = "none", False
            hand.pitch_compliance = float(e["hand_compliance"])
            hand.motor_mass = 0.0
        g.parts.append(hand)
        # Elevation pi/2 continues along the arm's own axis and roll pi keeps the
        # chord on the arm's side of the spar (checked on the built model: the
        # hand's centre is at the arm's chordwise station, its hinge along the span).
        g.edges.append(Edge(parent=1, child=2, pos_u=1.0, elevation=math.pi / 2, roll=math.pi,
                            scale=1.0, reflect=False))
    g.battery_wh, g.battery_chem = float(e["battery_wh"]), str(e["battery_chem"])
    g.flap_frequency = 1.0
    g.lineage = ["flapper"]
    return g
