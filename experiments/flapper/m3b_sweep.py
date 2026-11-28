"""Pass M3b, arm K's machine: the M0 machine with a third hinge per shoulder,
a fore-aft sweep about the hull's vertical axis, on its own witness motor.

Built in a copy of the compile step (the env's `compile_phenotype` is swapped
for `compile_with_sweep` only while `SweepMachine` is constructed); `mission.py`
is not edited.  What it adds, all stated in PREREG_M3b_land.md:

- per shoulder, a hinge `segN_wing_js` placed first in the wing body (so it acts
  before the stroke and the feather), axis the hull's z, range +-SWEEP_RANGE;
  joint damping and armature as the stroke's built joint;
- its actuator `segN_wing_s`, which `mission.Machine` gives the stroke's witness
  motor (tau_s 1.0 N m, omega_0 1000 rad/s, J_r 2e-5, gear 24; the reflected rotor
  as armature); this class then labels its role "sweep";
- per shoulder a 30 g point mass (TYPED: a motor and gearhead of the witness's
  class) at the shoulder, in the hull body, with no contact: +0.060 kg, all-up
  0.660 kg (the 0.6 kg machine's ballast is unchanged).

    from m3b_sweep import SweepMachine
    mc = SweepMachine("land", pose=dict(x=20, z=5))
"""
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SWEEP_RANGE_DEG = 35.0
MOTOR_MASS = 0.030
MASS_K = 0.60 + 2 * MOTOR_MASS


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def compile_with_sweep(p, **kw):
    """`compile_phenotype`'s signature and result, with a sweep hinge at each shoulder."""
    import mujoco
    from dytiscidae.core.mjcf import build_model_xml
    from dytiscidae.core.phenotype import build_panels
    xml, names = build_model_xml(p, **kw)
    root = ET.fromstring(xml)
    hull = next(b for b in root.iter("body") if b.get("name") == "seg0_hull")
    act = root.find("actuator")
    added = []
    for nm in ("seg1_wing", "seg2_wing"):
        wb = next(b for b in hull.findall("body") if b.get("name") == nm)
        R = _quat_mat([float(v) for v in wb.get("quat").split()])       # wing frame in the hull's
        axis = R.T @ np.array([0.0, 0.0, 1.0])                            # hull z in the wing frame
        stroke = wb.find("joint")
        j = ET.Element("joint", name=f"{nm}_js", type="hinge", axis=" ".join(f"{a:.9g}" for a in axis),
                       range=f"{-math.radians(SWEEP_RANGE_DEG):.6f} {math.radians(SWEEP_RANGE_DEG):.6f}",
                       damping=stroke.get("damping"), armature=stroke.get("armature"))
        wb.insert(list(wb).index(stroke), j)
        ref = next(a for a in act if a.get("joint") == f"{nm}_j")
        a = ET.SubElement(act, "position", name=f"{nm}_s", joint=f"{nm}_js")
        for k in ("kp", "kv", "forcerange", "ctrlrange"):
            a.set(k, ref.get(k))
        ET.SubElement(hull, "geom", name=f"{nm}_sweep_motor", type="sphere", size="0.012",
                      pos=wb.get("pos"), mass=f"{MOTOR_MASS}", contype="0", conaffinity="0",
                      rgba="0.8 0.1 0.1 1")
        added.append(f"{nm}_s")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    panels = build_panels(p, model)
    return model, data, list(names) + added, panels


class SweepMachine:
    """`mission.Machine` built through `compile_with_sweep`; the sweep actuators get role 'sweep'."""

    def __new__(cls, domain="land", **kw):
        import dytiscidae.envs.triphibian as TR
        from mission import Machine
        kw.setdefault("mass", MASS_K)
        orig = TR.compile_phenotype
        TR.compile_phenotype = compile_with_sweep
        try:
            mc = Machine(domain, **kw)
        finally:
            TR.compile_phenotype = orig
        names = list(mc.env.act_names)
        role = mc.role.astype(object)
        for n, a in enumerate(names):
            if a.endswith("_s"):
                role[n] = "sweep"
        mc.role = np.array(role, dtype=str)
        mc.sweep_range = math.radians(SWEEP_RANGE_DEG)
        return mc
