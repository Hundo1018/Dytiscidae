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
from dytiscidae.core.genome import HULL, PADDLE, STRUT, WING, Edge, Genome, Part

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
    # -- F2 (land): all off by default
    thumb_length=0.0,        # m; > 0: a fixed strut under each wing, pointing down
    thumb_radius=0.002, thumb_u=0.25,      # station along the wing, 0 shoulder, 1 tip
    tail_length=0.0,         # m; > 0: a paddle on an actuated hinge at the rear of the hull
    tail_chord=0.06, tail_radius=0.002, tail_motor_mass=0.003, tail_range=1.0,
    tail_el=-math.pi / 2, tail_roll=0.0,
    hind_u=None,             # a hull station: a second wing pair there (the same wing part)
    # -- pass 1m (pitch): off by default
    stab_length=0.0,         # m; > 0: an all-moving horizontal tail, this long each side,
    stab_chord=0.12,         # on a boom behind the hull, turning about its own span
    stab_boom=0.3, boom_radius=0.004, stab_motor_mass=0.009, stab_range=0.6,
    wing_seat=0.0,           # m; `built` moves each main wing's root this far out from the
                             # hull's axis (the hull's radius: shoulders on its sides)
    wing_shape=None,         # `built`: dict(taper=tip/root chord, camber=fraction of chord,
                             # twist_tip_deg=tip incidence against the root, nose up positive);
                             # the area stays the rectangle's.  None: a flat rectangle.
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
    if e["hind_u"] is not None:
        g.edges.append(Edge(parent=0, child=1, pos_u=float(e["hind_u"]), elevation=0.0, roll=0.0,
                            scale=1.0, reflect=True))
    if float(e["thumb_length"]) > 0.0:
        # One fixed strut through each wing, half below and half above.  The
        # mirrored wing is the first turned half a turn about the body's long
        # axis, so a single edge pointing down on one side points up on the
        # other (checked on the built model); two edges, azimuth +-pi/2, give
        # every wing a thumb below it and an idle one above.
        thumb = Part(kind=STRUT, length=float(e["thumb_length"]), radius=float(e["thumb_radius"]),
                     material="cfrp", joint="none", actuated=False)
        g.parts.append(thumb)
        for az in (math.pi / 2, -math.pi / 2):
            g.edges.append(Edge(parent=1, child=len(g.parts) - 1, pos_u=float(e["thumb_u"]),
                                azimuth=az, elevation=0.0, roll=0.0, scale=1.0, reflect=False))
    if float(e["tail_length"]) > 0.0:
        tail = Part(kind=PADDLE, length=float(e["tail_length"]), span=float(e["tail_length"]),
                    root_chord=float(e["tail_chord"]), radius=float(e["tail_radius"]),
                    material="cfrp", joint="hinge", joint_axis=np.array([0.0, 1.0, 0.0]),
                    joint_range=(-float(e["tail_range"]), float(e["tail_range"])), actuated=True,
                    motor_class=str(e["motor_class"]), motor_mass=float(e["tail_motor_mass"]),
                    gear_ratio=float(e["gear_ratio"]), sealed=False, stroke_amplitude=0.0,
                    neutral=0.5)
        g.parts.append(tail)
        g.edges.append(Edge(parent=0, child=len(g.parts) - 1, pos_u=0.0,
                            azimuth=0.0, elevation=float(e["tail_el"]),
                            roll=float(e["tail_roll"]), scale=1.0, reflect=False))
    if float(e["stab_length"]) > 0.0:
        boom = Part(kind=STRUT, length=float(e["stab_boom"]), radius=float(e["boom_radius"]),
                    material="cfrp", joint="none", actuated=False)
        stab = Part(kind=WING, length=float(e["stab_length"]), span=float(e["stab_length"]),
                    root_chord=float(e["stab_chord"]), radius=float(e["wing_radius"]),
                    material=str(e["wing_material"]), joint="hinge",
                    joint_axis=np.array([1.0, 0.0, 0.0]),
                    joint_range=(-float(e["stab_range"]), float(e["stab_range"])), actuated=True,
                    motor_class=str(e["motor_class"]), motor_mass=float(e["stab_motor_mass"]) / 2,
                    gear_ratio=float(e["gear_ratio"]), sealed=False, stroke_amplitude=0.0,
                    neutral=0.5)
        g.parts += [boom, stab]
        nb = len(g.parts) - 2
        g.edges.append(Edge(parent=0, child=nb, pos_u=0.0, azimuth=0.0, elevation=-math.pi / 2,
                            roll=0.0, scale=1.0, reflect=False))
        g.edges.append(Edge(parent=nb, child=nb + 1, pos_u=1.0, elevation=0.0, roll=math.pi,
                            scale=1.0, reflect=True))
    g.battery_wh, g.battery_chem = float(e["battery_wh"]), str(e["battery_chem"])
    g.flap_frequency = 1.0
    g.lineage = ["flapper"]
    return g


def built(body: dict):
    """The phenotype of ``flapper(**body)``.  The genome can only attach a part
    on its parent's axis, so the main wings' roots, and with them the stroke
    and feather axes, sit on the hull's centre line: the span is short of the
    design's by the hull's diameter (found on the drawing, pass S1).
    ``wing_seat`` moves each root out sideways by that much before the model
    is compiled."""
    from dytiscidae.core.phenotype import build
    p = build(flapper(**body))
    seat = float(dict(DEFAULTS, **body)["wing_seat"])
    if seat:
        for s in p.segments:
            if s.part is p.genome.parts[1] and s.parent == 0:
                s.offset = s.offset + np.array([0.0, -seat if s.mirrored else seat, 0.0])
    shape = dict(DEFAULTS, **body)["wing_shape"]
    if shape:
        from dataclasses import replace
        lam = float(shape.get("taper", 1.0))
        for s in p.segments:
            if s.part is not p.genome.parts[1] or s.parent != 0:
                continue
            sf = s.surface
            x = sf.u / max(float(sf.u[-1]), 1e-9)
            sign = -1.0 if s.mirrored else 1.0           # as `phenotype._surface_of`
            s.surface = replace(
                sf, chord=sf.chord * 2.0 / (1.0 + lam) * (1.0 - (1.0 - lam) * x),
                camber=sign * np.full(len(x), float(shape.get("camber", 0.0))),
                twist=sign * np.radians(float(shape.get("twist_tip_deg", 0.0))) * x)
    return p
