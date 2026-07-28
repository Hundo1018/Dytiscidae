"""Phenotype -> MJCF.  Builds the MuJoCo model and the triphibian scene.

MuJoCo supplies rigid-body dynamics, joints and contacts.  It supplies *no*
fluid behaviour here -- ``opt/density`` and ``opt/viscosity`` are deliberately
zero, because the entire fluid interaction comes from ``physics/fluid.py``.
Leaving MuJoCo's own fluid model switched on would double-count drag and would
reintroduce the buoyancy-free water that made it unusable in the first place.

The scene is a single continuous world rather than three separate environments:
a sloping seabed that rises through the waterline into a beach, open water to
one side, open air above.  The shoreline is therefore a place, not a mode, and a
machine can be scored on crossing it.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from ..physics.medium import SEAWATER
from .genome import BALLAST, FOOT, HULL, PADDLE, STRUT
from .genome import WING as WING_KIND
from .phenotype import AVIONICS_MASS, Phenotype, Segment


def mat2quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix to (w, x, y, z), MuJoCo's convention."""
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return (w / n, x / n, y / n, z / n)


def _fmt(v) -> str:
    if isinstance(v, (int, float, np.floating, np.integer)):
        return f"{float(v):.6g}"
    return " ".join(f"{float(x):.6g}" for x in np.asarray(v).ravel())


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------


def scene_xml(
    *,
    seabed_depth: float = 18.0,
    beach_slope: float = 0.12,
    shore_x: float = 12.0,
    timestep: float = 0.002,
) -> ET.Element:
    """Root ``<mujoco>`` element with the triphibian world already in it.

    ``z = 0`` is the still waterline.  The beach is a long thin box rotated by
    ``beach_slope`` whose upper end clears the surface near ``x = shore_x``, so a
    machine travelling in +X runs out of water and has to walk.
    """
    root = ET.Element("mujoco", {"model": "dytiscidae"})

    ET.SubElement(
        root,
        "option",
        {
            "timestep": _fmt(timestep),
            "gravity": "0 0 -9.80665",
            # Fluid effects are computed externally; see module docstring.
            "density": "0",
            "viscosity": "0",
            "integrator": "implicitfast",
            "cone": "elliptic",
        },
    )
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})

    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        {"friction": "0.9 0.02 0.001", "condim": "4", "solref": "0.004 1", "margin": "0.001"},
    )
    ET.SubElement(default, "joint", {"damping": "0.05", "armature": "0.002", "limited": "true"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {"type": "skybox", "builtin": "gradient", "rgb1": "0.5 0.7 0.9",
         "rgb2": "0.05 0.1 0.2", "width": "256", "height": "256"},
    )
    ET.SubElement(
        asset,
        "texture",
        {"name": "grid", "type": "2d", "builtin": "checker", "rgb1": "0.25 0.28 0.24",
         "rgb2": "0.3 0.34 0.29", "width": "300", "height": "300"},
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "seabed", "texture": "grid", "texrepeat": "20 20", "reflectance": "0.05"},
    )
    ET.SubElement(asset, "material", {"name": "beach", "rgba": "0.72 0.66 0.48 1"})
    # The water surface is a visual cue only; it has no collision geometry,
    # because the free surface is handled analytically by MediumField.
    ET.SubElement(asset, "material", {"name": "water", "rgba": "0.15 0.4 0.55 0.25"})

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 20", "dir": "0 0 -1", "diffuse": "0.9 0.9 0.9"})

    # Seabed.
    ET.SubElement(
        wb,
        "geom",
        {"name": "seabed", "type": "plane", "pos": f"0 0 {-seabed_depth}",
         "size": "60 60 0.5", "material": "seabed"},
    )
    # Beach: a ramp that emerges through the waterline.
    ramp_len = 40.0
    ang = math.atan(beach_slope)
    # Place it so the surface crossing lands at x = shore_x.
    cx = shore_x + 0.5 * ramp_len * math.cos(ang)
    cz = 0.5 * ramp_len * math.sin(ang)
    q = mat2quat(
        np.array(
            [
                [math.cos(ang), 0.0, math.sin(ang)],
                [0.0, 1.0, 0.0],
                [-math.sin(ang), 0.0, math.cos(ang)],
            ]
        )
    )
    ET.SubElement(
        wb,
        "geom",
        {"name": "beach", "type": "box", "pos": f"{_fmt(cx)} 0 {_fmt(cz)}",
         "quat": _fmt(q), "size": f"{_fmt(0.5*ramp_len)} 30 0.5", "material": "beach"},
    )
    # Visual water plane (no contact).
    ET.SubElement(
        wb,
        "geom",
        {"name": "waterplane", "type": "box", "pos": "-14 0 0", "size": "26 30 0.005",
         "material": "water", "contype": "0", "conaffinity": "0", "group": "2"},
    )
    return root


# --------------------------------------------------------------------------
# Body tree
# --------------------------------------------------------------------------


def _add_geoms(body: ET.Element, s: Segment, extra_mass: float) -> None:
    """Attach collision/visual geometry and pin down the mass explicitly.

    Masses come from the phenotype's budget, not from geom density, because the
    budget already accounts for walls, skins, motors and seals.  Letting MuJoCo
    infer mass from a solid geom would roughly triple every design.
    """
    mass = max(s.mass + extra_mass, 1e-4)
    L = max(s.axis_length, 0.01)

    if s.is_surface and s.surface is not None:
        c_mean = float(np.mean(s.surface.chord))
        t_mean = float(np.mean(s.surface.thickness)) * c_mean
        # A thin box spanning +X, chord along Y, thickness along Z.
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{s.name}_g",
                "type": "box",
                "pos": f"{_fmt(0.5*L)} {_fmt(-0.25*c_mean)} 0",
                "size": f"{_fmt(0.5*L)} {_fmt(0.5*c_mean)} {_fmt(max(0.5*t_mean, 0.002))}",
                "mass": _fmt(mass),
                "rgba": "0.85 0.5 0.2 0.9" if s.kind == WING_KIND else "0.3 0.6 0.8 0.9",
            },
        )
    elif s.kind in (HULL, BALLAST):
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{s.name}_g",
                "type": "capsule",
                "fromto": f"0 0 0 {_fmt(L)} 0 0",
                "size": _fmt(max(s.radius, 0.01)),
                "mass": _fmt(mass),
                "rgba": "0.25 0.28 0.32 1" if s.kind == HULL else "0.5 0.35 0.6 1",
            },
        )
    else:
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{s.name}_g",
                "type": "capsule",
                "fromto": f"0 0 0 {_fmt(L)} 0 0",
                "size": _fmt(max(s.radius, 0.004)),
                "mass": _fmt(mass),
                "rgba": "0.4 0.4 0.45 1" if s.kind == STRUT else "0.6 0.3 0.3 1",
            },
        )


def build_model_xml(
    p: Phenotype,
    *,
    spawn: tuple[float, float, float] = (-6.0, 0.0, 1.5),
    scene: ET.Element | None = None,
) -> tuple[str, list[str]]:
    """Compile a phenotype into MJCF.

    Returns the XML string and the list of actuator names in the same order as
    ``phenotype.actuators``, so the energy model can be indexed consistently.
    """
    root = scene if scene is not None else scene_xml()
    wb = root.find("worldbody")
    assert wb is not None

    segs = p.segments
    if not segs:
        # Degenerate genome: emit a bare inert body so the caller still gets a
        # valid model and can score it as failed rather than crashing.
        b = ET.SubElement(wb, "body", {"name": "seg0_null", "pos": _fmt(spawn)})
        ET.SubElement(b, "freejoint", {"name": "root"})
        ET.SubElement(b, "geom", {"type": "sphere", "size": "0.05", "mass": "0.1"})
        return ET.tostring(root, encoding="unicode"), []

    children: dict[int, list[Segment]] = {}
    for s in segs:
        children.setdefault(s.parent, []).append(s)

    actuator_names: list[str] = []
    act_specs: list[tuple[str, str, float]] = []  # (name, joint, forcerange)

    # Non-structural mass (battery, avionics, seals) rides in the root segment.
    extra_root = p.budget.battery + p.budget.avionics + p.budget.seals

    def add_segment(s: Segment, parent_el: ET.Element) -> None:
        attrs = {"name": s.name}
        if s.parent < 0:
            attrs["pos"] = _fmt(spawn)
        else:
            attrs["pos"] = _fmt(s.offset)
            attrs["quat"] = _fmt(mat2quat(s.rotation))
        body = ET.SubElement(parent_el, "body", attrs)

        if s.parent < 0:
            ET.SubElement(body, "freejoint", {"name": "root"})
        elif s.part.joint != "none":
            jname = f"{s.name}_j"
            lo, hi = s.part.joint_range
            axis = np.asarray(s.part.joint_axis, float)
            n = np.linalg.norm(axis)
            axis = axis / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
            ET.SubElement(
                body,
                "joint",
                {
                    "name": jname,
                    "type": "hinge",
                    "axis": _fmt(axis),
                    "range": f"{_fmt(min(lo, hi))} {_fmt(max(lo, hi))}",
                    "damping": _fmt(0.02 + 0.5 * s.mass),
                    "armature": _fmt(max(1e-4, 0.01 * s.mass)),
                },
            )
            if s.actuator is not None:
                aname = f"{s.name}_a"
                actuator_names.append(aname)
                act_specs.append((aname, jname, s.actuator.stall_torque))

        _add_geoms(body, s, extra_root if s.parent < 0 else 0.0)

        for c in children.get(s.index, ()):
            add_segment(c, body)

    root_seg = next(s for s in segs if s.parent < 0)
    add_segment(root_seg, wb)

    if act_specs:
        act = ET.SubElement(root, "actuator")
        for name, joint, frange in act_specs:
            fr = float(np.clip(frange, 0.02, 400.0))
            # Position servo: the controller commands angles, MuJoCo produces the
            # torque, and the energy model reads that torque back.  kp is scaled
            # to the actuator's own authority so a small motor cannot fake a
            # large one through an unrealistically stiff servo.
            ET.SubElement(
                act,
                "position",
                {
                    "name": name,
                    "joint": joint,
                    "kp": _fmt(max(2.0 * fr, 0.5)),
                    "kv": _fmt(max(0.15 * fr, 0.02)),
                    "forcerange": f"{_fmt(-fr)} {_fmt(fr)}",
                    "ctrlrange": "-3.2 3.2",
                },
            )

    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "framepos", {"name": "root_pos", "objtype": "body",
                                       "objname": root_seg.name})
    ET.SubElement(sensor, "framequat", {"name": "root_quat", "objtype": "body",
                                        "objname": root_seg.name})
    ET.SubElement(sensor, "framelinvel", {"name": "root_linvel", "objtype": "body",
                                          "objname": root_seg.name})
    ET.SubElement(sensor, "frameangvel", {"name": "root_angvel", "objtype": "body",
                                          "objname": root_seg.name})

    return ET.tostring(root, encoding="unicode"), actuator_names


def compile_phenotype(p: Phenotype, **kw):
    """Build and compile.  Returns ``(model, data, actuator_names, panels)``."""
    import mujoco

    from .phenotype import build_panels

    xml, names = build_model_xml(p, **kw)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    panels = build_panels(p, model)
    return model, data, names, panels
