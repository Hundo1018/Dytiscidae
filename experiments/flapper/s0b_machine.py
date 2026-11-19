"""S0b probes (DESIGN_S0b_bold.md): the machine T4 on `mission.Machine`.

T4 = C1's body (`c1_pilot.body()`: hull, W6 wing, the all-moving tail) with
- a second wing pair at hull station HIND_U (`body.flapper(hind_u=...)`),
- a fore-aft sweep hinge on each of the four shoulders (the hinge of
  `m3b_sweep.compile_with_sweep`, here on seg1..seg4), role "sweep",
- every hinge on the ideal motor (tau_s 1e4 N m, omega_0 1e7 rad/s; rotor and
  gear as mission.MOTORS),
- ballast as one point mass in the hull body at compile time (no contact, no
  volume), sized so that weight = b x the solver's buoyancy of the fully
  submerged machine, placed KEEL_FRAC x hull radius below the hull's axis and
  at the station that puts the machine's centre of mass under its centre of
  buoyancy.

Differences from the design note, stated: no mass is added for the sweep drives
(M3b typed 30 g each; here the all-up mass is fixed by b and no drive is
chosen); the hull's inertia is the built one (S0's A0 overrides it to 0.012).

Nothing in mission.py, m3b_sweep.py or s0_machine.py is edited or rebound.
Why HIND_U = 0.11: the wing's chord runs from 0.065 m ahead of its shoulder to
0.193 m behind it (0.2576 m chord, spar at the quarter chord; read on the built
model).  The fore shoulder is C1's (station 0.65, x = 0.325 m), so its trailing
edge is at x = 0.132 m.  The hind shoulder must be at x <= 0.067 m for the two
wings not to share any hull station (they stroke about the same axis); 0.055 m
(station 0.11) leaves 12 mm.  It is inside the hull (0..0.5 m).
"""
from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
G = 9.80665
IDEAL = dict(tau_s=1e4, omega_0=1e7)
#: a bounded, generous motor (the lead, 2026-11-19): mission's witness curve x4 in torque and in speed.
X4 = dict(stroke=dict(tau_s=4.0, omega_0=4000.0), feather=dict(tau_s=4.0, omega_0=4000.0),
          tail=dict(tau_s=0.4, omega_0=4000.0))
#: witness stall torque at the hinge (N m): tau_s x gear.  A free rollout whose peak hinge torque is above
#: 10x this is flagged an artifact row (sweep uses the stroke's motor).
WITNESS_STALL = dict(stroke=24.0, feather=16.0, tail=5.0, sweep=24.0)
HIND_U = 0.11
KEEL_FRAC = 0.8
SWEEP_RANGE_DEG = 35.0
WINGS = ("seg1_wing", "seg2_wing", "seg3_wing", "seg4_wing")      # fore left (+y), fore right, hind left, hind right
TAG = dict(seg1_wing="FL", seg2_wing="FR", seg3_wing="HL", seg4_wing="HR")
CALIB = HERE / "results" / "S0b_calib.json"


def body_t4() -> dict:
    import c1_pilot
    return dict(c1_pilot.body(), hind_u=HIND_U)


def _compile(keel_mass: float, keel_x: float, keel_z: float, sweep: bool = True):
    import mujoco
    from dytiscidae.core.mjcf import build_model_xml
    from dytiscidae.core.phenotype import build_panels
    import m3b_sweep as SW

    def comp(p, **kw):
        xml, names = build_model_xml(p, **kw)
        root = ET.fromstring(xml)
        hull = next(b for b in root.iter("body") if b.get("name") == "seg0_hull")
        act = root.find("actuator")
        added = []
        if sweep:
            for nm in WINGS:
                wb = next(b for b in hull.findall("body") if b.get("name") == nm)
                R = SW._quat_mat([float(v) for v in wb.get("quat").split()])
                axis = R.T @ np.array([0.0, 0.0, 1.0])          # the hull's z in the wing frame
                stroke = wb.find("joint")
                j = ET.Element("joint", name=f"{nm}_js", type="hinge", axis=" ".join(f"{a:.9g}" for a in axis),
                               range=f"{-math.radians(SWEEP_RANGE_DEG):.6f} {math.radians(SWEEP_RANGE_DEG):.6f}",
                               damping=stroke.get("damping"), armature=stroke.get("armature"))
                wb.insert(list(wb).index(stroke), j)
                ref = next(a for a in act if a.get("joint") == f"{nm}_j")
                a = ET.SubElement(act, "position", name=f"{nm}_s", joint=f"{nm}_js")
                for k in ("kp", "kv", "forcerange", "ctrlrange"):
                    a.set(k, ref.get(k))
                added.append(f"{nm}_s")
        if keel_mass > 0.0:
            ET.SubElement(hull, "geom", name="s0b_keel", type="sphere", size="0.008",
                          pos=f"{keel_x:.6f} 0 {keel_z:.6f}", mass=f"{keel_mass:.9g}",
                          contype="0", conaffinity="0", rgba="0.1 0.1 0.9 1")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        return model, mujoco.MjData(model), list(names) + added, build_panels(p, model)
    return comp


def _machine(domain, keel=(0.0, 0.25, 0.0), sweep=True, **kw):
    import dytiscidae.envs.triphibian as TR
    from mission import Machine
    mot = {r: dict(IDEAL) for r in ("stroke", "feather", "tail")}
    for r, v in (kw.pop("motors", None) or {}).items():
        mot[r].update(v)
    kw.setdefault("body", body_t4())
    kw["mass"] = None                               # no ballast at the hull's centre: the keel carries it
    orig = TR.compile_phenotype
    TR.compile_phenotype = _compile(*keel, sweep=sweep)
    try:
        mc = Machine(domain, motors=mot, **kw)
    finally:
        TR.compile_phenotype = orig
    role = mc.role.astype(object)
    names = list(mc.env.act_names)
    for n, a in enumerate(names):
        if a.endswith("_s"):
            role[n] = "sweep"
    mc.role = np.array(role, dtype=str)
    ix = {}
    for n, a in enumerate(names):
        for w in WINGS:
            if a.startswith(w):
                ix[(TAG[w], dict(a="stroke", f="feather", s="sweep")[a[-1]])] = n
    mc.ix = ix
    mc.tail_ix = np.nonzero(mc.role == "tail")[0]
    return mc


def fluid_wrench(mc):
    """(F, M about the dry CoM), world frame, of what the solver applied in the
    last step, less the weight of the added mass it wrote into body_mass."""
    m, d = mc.m, mc.d
    F = d.xfrc_applied[:, :3].copy()
    F[:, 2] -= (m.body_mass - mc.env.solver._dry_mass) * G
    c = mc.com()
    M = d.xfrc_applied[:, 3:].sum(0) + np.cross(d.xipos - c, F).sum(0)
    return F.sum(0), M


def static_buoyancy(mc):
    """Buoyancy (N) and its centre (world) of the machine as posed, at rest."""
    mc.d.qvel[:] = 0.0
    mc.mj.mj_forward(mc.m, mc.d)
    mc.d.xfrc_applied[:] = 0.0
    mc.env.solver.reset()
    mc.env.solver.apply(mc.d, 0.0)
    F, M = fluid_wrench(mc)
    c = mc.com()
    # M = (r_b - c) x F with F = (0, 0, B): r_b.x = c.x - M.y / B, r_b.y = c.y + M.x / B
    B = float(F[2])
    rb = np.array([c[0] - M[1] / B, c[1] + M[0] / B, np.nan])
    mc.d.xfrc_applied[:] = 0.0
    return B, rb, float(mc.env.solver.diag.buoyancy), float(mc.env.solver.diag.max_submerged)


def calibrate(force: bool = False) -> dict:
    """Build without ballast, read mass, buoyancy and the centre of buoyancy; solve the keel."""
    if CALIB.exists() and not force:
        return json.loads(CALIB.read_text())
    mc = _machine("water", pose=dict(x=-30.0, y=0.0, z=-5.0, vx=0.0, vz=0.0))
    B, rb, Bdiag, sub = static_buoyancy(mc)
    R = mc.d.xmat[mc.hull].reshape(3, 3)
    o = mc.d.xpos[mc.hull]
    c_h = R.T @ (mc.com() - o)                       # CoM in the hull frame, no ballast
    xb_h = float((R.T @ (np.array([rb[0], rb[1], o[2]]) - o))[0])
    m0 = mc.mass
    import body as BD
    rad = float(body_t4()["hull_radius"])
    out = dict(m0=m0, B=B, B_diag=Bdiag, max_submerged=sub, b0=B / (m0 * G), com0_hull=c_h.tolist(),
               cob_x_hull=xb_h, hull_radius=rad, hind_u=HIND_U)
    CALIB.write_text(json.dumps(out, indent=1))
    return out


def keel_for(b: float) -> tuple:
    c = calibrate()
    mk = c["B"] / (G * b) - c["m0"]
    if mk <= 0:
        return (0.0, 0.25, 0.0)
    M = c["m0"] + mk
    x = (M * c["cob_x_hull"] - c["m0"] * c["com0_hull"][0]) / mk       # CoM x under the centre of buoyancy
    x = min(max(x, 0.03), 0.47)
    return (mk, x, -KEEL_FRAC * c["hull_radius"])


def make(domain="air", b: float | None = 1.0, sweep=True, motor="ideal", **kw):
    """T4.  b None: no ballast (the built mass).  motor: "ideal" or "x4"."""
    if motor == "x4":
        kw["motors"] = {r: dict(v) for r, v in X4.items()}
    keel = keel_for(b) if b else (0.0, 0.25, 0.0)
    mc = _machine(domain, keel=keel, sweep=sweep, **kw)
    mc.t4 = dict(b=b, keel_kg=keel[0], keel_x=keel[1], keel_z=keel[2], motor=motor)
    return mc


def enable_twist4(mc) -> None:
    """The strip-twist stand-in on all four wings: after this
    ``set_twist4(mc, FL, FR, HL, HR)`` (deg at the tip).  mission's own
    `enable_strip_twist` knows seg1 / seg2 only."""
    mc.enable_strip_twist(body_t4())
    pan = mc.env.solver.panels
    wing = np.full(pan.n, -1)
    for i, bid in enumerate(pan.body_id):
        bid = int(bid)
        while bid and int(mc.m.body_parentid[bid]) != mc.hull:
            bid = int(mc.m.body_parentid[bid])
        nm = mc.mj.mj_id2name(mc.m, mc.mj.mjtObj.mjOBJ_BODY, bid) or ""
        if nm in WINGS:
            wing[i] = WINGS.index(nm)
    mc.tw_wing = wing


def set_twist4(mc, fl, fr, hl, hr) -> None:
    tw, pan = mc.tw, mc.env.solver.panels
    val = np.array([fl, fr, hl, hr, 0.0])
    fd = val[mc.tw_wing]                              # -1 -> the trailing 0
    ang = tw["per_deg"] * (tw["tip"] + tw["fsign"] * tw["share"] * fd)
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    pan.chord_local[:] = tw["c0"] * ca + np.cross(tw["s0"], tw["c0"]) * sa
    pan.normal_local[:] = np.cross(pan.span_local, pan.chord_local)


def regain(mc, wn=300.0):
    """The hinges' PD gains re-read from the mass matrix in the present medium (as `m2_common._regain`,
    without the solver reset)."""
    m, d, mj = mc.m, mc.d, mc.mj
    for n, i in enumerate(mc.dof):
        e, col = np.zeros(m.nv), np.zeros(m.nv)
        e[i] = 1.0
        mj.mj_mulM(m, d, col, e)
        J = col[i]
        mc.kp[n], mc.kv[n] = J * wn * wn, 2.0 * J * wn
        a = mc.aid[n]
        m.actuator_gainprm[a, 0] = mc.kp[n]
        m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2] = -mc.kp[n], -mc.kv[n]


def report(mc) -> dict:
    """Numbers of the built model: mass, CoM, inertia about the CoM (hull axes), hinges."""
    m, d, mj = mc.m, mc.d, mc.mj
    mj.mj_forward(m, d)
    dry = mc.env.solver._dry_mass
    R = d.xmat[mc.hull].reshape(3, 3)
    o = d.xpos[mc.hull]
    c = mc.com()
    I = np.zeros((3, 3))
    din = mc.env.solver._dry_inertia
    for bid in range(1, m.nbody):
        Rb = d.ximat[bid].reshape(3, 3)
        I += Rb @ np.diag(din[bid]) @ Rb.T
        r = d.xipos[bid] - c
        I += dry[bid] * ((r @ r) * np.eye(3) - np.outer(r, r))
    Ih = R.T @ I @ R
    hinges = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, int(j)) for j in mc.jid]
    return dict(mass=float(dry.sum()), com_hull=(R.T @ (c - o)).tolist(), inertia_hull_diag=np.diag(Ih).tolist(),
                inertia_hull=Ih.tolist(), n_hinges=int(len(mc.jid)), hinges=hinges, roles=list(mc.role),
                n_shoulder_hinges=int(sum(r in ("stroke", "feather", "sweep") for r in mc.role)),
                keel=mc.t4, nbody=int(m.nbody), panels=int(mc.env.solver.panels.n))


if __name__ == "__main__":
    cal = calibrate(force="--force" in sys.argv)
    print(json.dumps(cal))
    mc = make("water", b=1.0, pose=dict(x=-30.0, y=0.0, z=-5.0, vx=0.0, vz=0.0))
    rep = report(mc)
    B, rb, Bd, sub = static_buoyancy(mc)
    R = mc.d.xmat[mc.hull].reshape(3, 3)
    rep.update(B=B, b=B / (rep["mass"] * G), cob_x_hull=float((R.T @ (np.array([rb[0], rb[1], 0]) - mc.d.xpos[mc.hull]))[0]),
               max_submerged=sub, bad_qacc=mc.bad_qacc)
    print(json.dumps(rep))
