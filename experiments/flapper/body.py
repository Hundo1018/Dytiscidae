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
    g.battery_wh, g.battery_chem = float(e["battery_wh"]), str(e["battery_chem"])
    g.flap_frequency = 1.0
    g.lineage = ["flapper"]
    return g
