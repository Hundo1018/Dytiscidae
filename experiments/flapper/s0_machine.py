"""Pass S0 (PREREG_S0_continuation.md, frozen dd89844): the machine of each rung.

Every frozen runner of the M series builds its machine with ``from mission
import Machine`` (or `m3b_sweep.SweepMachine`) at call time.  `configure(cfg)`
rebinds those two names, inside the worker process, to `make`, which builds
the S0 machine of the rungs in ``cfg`` and otherwise passes the caller's
arguments through.  Nothing in mission.py, m3b_sweep.py or any frozen file is
edited.  What `make` adds (PREREG_S0 section 2, results/S0_plan.md):

- A0: every hinge on the ideal motor (tau_s 1e4 N m, omega_0 1e7 rad/s; J_r
  and the gear as given); the hull's transverse inertia 0.012 kg m2
  (`m4_common.I0`, "as C1"); a per-step instrument (`Inst`).
- A1 (cfg["sweep"]): the sweep hinge of `m3b_sweep.compile_with_sweep`, its
  two 30 g point masses, role "sweep".
- A2 (cfg["b"]): all-up mass = B / (g b), B the solver's buoyancy of the
  fully submerged machine read on the built model (`buoyancy`).
- A5p.2 (cfg["keel"]): the mass added for b put at a contact-free point
  0.4 x the hull's half-height below the hull's centre instead of at it.
- A6 (cfg["wing"]): "chord", "span" or "tandem" (body variants).
- cfg["solref_x"] (the contact guard), cfg["dt"] (the dt/2 guard).
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
I0 = 0.012
CALIB = HERE / "results" / "S0_calib.json"

CFG: dict = {}
LAST: list = []            # the machines built in this process, newest last
CAPTURE: list = []         # s0_draw: absolute sim times (s) at which Inst stores a pose
SNAPS: list = []
_ORIG: dict = {}


def body_variant(wing=None) -> dict:
    import c1_pilot
    b = dict(_ORIG.get("c1_body", c1_pilot.body)())
    if wing == "chord":                 # tip chord x1.5, root and span unchanged
        b["wing_shape"] = dict(b.get("wing_shape") or {}, taper=1.5)
        b["wing_chord"] = b["wing_chord"] * 1.25
    elif wing == "span":
        b["wing_length"] = b["wing_length"] * 1.25
    elif wing == "tandem":
        # a second pair 0.3 m behind the first (shoulder_u is a hull station, 0 tail 1 nose)
        import body as B
        su = float(b.get("shoulder_u", B.DEFAULTS["shoulder_u"]))
        b["hind_u"] = su - 0.3 / float(b["hull_length"])
    return b


def _compile_hook(sweep: bool, keel_mass: float):
    """A compile step with the sweep hinges and/or a keel point mass."""
    import mujoco
    from dytiscidae.core.mjcf import build_model_xml
    from dytiscidae.core.phenotype import build_panels
    import m3b_sweep as SW

    def comp(p, **kw):
        if sweep and not keel_mass:
            return SW.compile_with_sweep(p, **kw)
        xml, names = build_model_xml(p, **kw)
        root = ET.fromstring(xml)
        if sweep:                               # rebuild the sweep edits on this tree (as compile_with_sweep)
            hull = next(b for b in root.iter("body") if b.get("name") == "seg0_hull")
            act = root.find("actuator")
            added = []
            for nm in ("seg1_wing", "seg2_wing"):
                wb = next(b for b in hull.findall("body") if b.get("name") == nm)
                R = SW._quat_mat([float(v) for v in wb.get("quat").split()])
                axis = R.T @ np.array([0.0, 0.0, 1.0])
                stroke = wb.find("joint")
                j = ET.Element("joint", name=f"{nm}_js", type="hinge", axis=" ".join(f"{a:.9g}" for a in axis),
                               range=f"{-math.radians(SW.SWEEP_RANGE_DEG):.6f} {math.radians(SW.SWEEP_RANGE_DEG):.6f}",
                               damping=stroke.get("damping"), armature=stroke.get("armature"))
                wb.insert(list(wb).index(stroke), j)
                ref = next(a for a in act if a.get("joint") == f"{nm}_j")
                a = ET.SubElement(act, "position", name=f"{nm}_s", joint=f"{nm}_js")
                for k in ("kp", "kv", "forcerange", "ctrlrange"):
                    a.set(k, ref.get(k))
                ET.SubElement(hull, "geom", name=f"{nm}_sweep_motor", type="sphere", size="0.012",
                              pos=wb.get("pos"), mass=f"{SW.MOTOR_MASS}", contype="0", conaffinity="0",
                              rgba="0.8 0.1 0.1 1")
                added.append(f"{nm}_s")
            names = list(names) + added
        hull = next(b for b in root.iter("body") if b.get("name") == "seg0_hull")
        # keel: 0.4 x half-height below the hull's centre (its compiled centre of mass; hull z is dorsal)
        m0 = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        hid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "seg0_hull")
        c = np.array(m0.body_ipos[hid])
        half = float(CFG.get("hull_half", 0.04))
        ET.SubElement(hull, "geom", name="s0_keel", type="sphere", size="0.006",
                      pos=f"{c[0]:.6f} {c[1]:.6f} {c[2] - 0.4 * half:.6f}", mass=f"{keel_mass:.9g}",
                      contype="0", conaffinity="0", rgba="0.1 0.1 0.9 1")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        return model, data, list(names), build_panels(p, model)
    return comp


def calib(key: str):
    if CALIB.exists():
        return json.loads(CALIB.read_text()).get(key)
    return None


def buoyancy(cfg_base: dict) -> float:
    """B (N): the solver's buoyancy of the fully submerged machine, held level at 5 m."""
    key = json.dumps(dict(sweep=bool(cfg_base.get("sweep")), wing=cfg_base.get("wing")), sort_keys=True)
    v = calib(key)
    if v is not None:
        return float(v)
    save = dict(CFG)
    CFG.clear()
    CFG.update(dict(sweep=cfg_base.get("sweep"), wing=cfg_base.get("wing")))
    try:
        mc = make("water", pose=dict(x=-30.0, z=-5.0, vx=0.0, vz=0.0))
        mc.d.qvel[:] = 0.0
        mc.mj.mj_forward(mc.m, mc.d)
        mc.env.solver.apply(mc.d, 0.0)
        B = float(mc.env.solver.diag.buoyancy)
        sub = float(mc.env.solver.diag.max_submerged)
    finally:
        CFG.clear()
        CFG.update(save)
    d = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    d[key] = B
    d[key + "_max_submerged"] = sub
    CALIB.write_text(json.dumps(d, indent=1))
    return B


def make(domain="land", **kw):
    """The S0 machine of the configured rungs (same signature as mission.Machine)."""
    import dytiscidae.envs.triphibian as TR
    Machine = _ORIG["Machine"]
    cfg = CFG
    sweep = bool(cfg.get("sweep"))
    mot = {k: dict(v) for k, v in (kw.pop("motors", None) or {}).items()}
    if not cfg.get("witness"):                  # diagnostics only: the M-series witness motors
        for r in ("stroke", "feather", "tail"):
            mot.setdefault(r, {}).update(IDEAL)
    kw["motors"] = mot
    if cfg.get("dt"):
        kw["dt"] = cfg["dt"]
    base_mass = 0.60 + (0.06 if sweep else 0.0)
    mass = base_mass
    keel = 0.0
    if cfg.get("b"):
        B = buoyancy(cfg)
        mass = B / (G * float(cfg["b"]))
        if cfg.get("keel"):
            keel = mass - base_mass
    if cfg.get("mass_override"):
        mass = float(cfg["mass_override"])
    kw["mass"] = mass
    if cfg.get("wing"):
        kw["body"] = body_variant(cfg["wing"])
    orig = TR.compile_phenotype
    if sweep or keel:
        TR.compile_phenotype = _compile_hook(sweep, keel)
    try:
        mc = Machine(domain, **kw)
    finally:
        TR.compile_phenotype = orig
    if sweep:
        role = mc.role.astype(object)
        for n, a in enumerate(mc.env.act_names):
            if a.endswith("_s"):
                role[n] = "sweep"
        mc.role = np.array(role, dtype=str)
        mc.sweep_range = math.radians(35.0)
    m, d, mj = mc.m, mc.d, mc.mj
    h = mc.hull
    long_axis = int(np.argmin(m.body_inertia[h]))
    for ax in range(3):
        if ax != long_axis and not cfg.get("no_I0"):    # no_I0: diagnostics only
            m.body_inertia[h, ax] = I0
            mc.env.solver._dry_inertia[h, ax] = I0
    if cfg.get("solref_x"):
        m.geom_solref[:, 0] *= float(cfg["solref_x"])
    mj.mj_setConst(m, mj.MjData(m))
    mj.mj_forward(m, d)
    mc.s0 = dict(cfg=dict(cfg), mass=mc.mass, keel_kg=keel, domain=domain)
    Inst(mc)
    LAST.append(mc)
    return mc


class Inst:
    """Wraps mc.step: peaks per hinge (torque, speed, shaft power), peak contact
    (weights, every step with contacts), and every 10th step a sample of
    (t, CoM speed, max |root feather|, wet strips > 0, contact) for the X2 read.
    ``mc.sweep_hold``: when True the sweep targets are forced to 0 (laws written
    for the six-hinge machine index roles positionally, `m2_common.Law`)."""

    def __init__(self, mc):
        self.mc = mc
        n = len(mc.aid)
        self.tq, self.om, self.pw = np.zeros(n), np.zeros(n), np.zeros(n)
        self.contact_max = 0.0
        self.cbins = {}
        self.samples = []
        self.k = 0
        self.orig = mc.step
        self.fe = np.nonzero(mc.role == "feather")[0]
        self.sw = np.nonzero(mc.role == "sweep")[0]
        mc.inst = self
        mc.sweep_hold = not CFG.get("sweep_free")
        mc.sweep_fn = None
        mc.step = self.step

    def step(self, q_ref, qd_ref=None):
        mc = self.mc
        if len(self.sw):
            q_ref = np.array(q_ref, dtype=float)
            qd_ref = np.zeros(len(q_ref)) if qd_ref is None else np.array(qd_ref, dtype=float)
            if mc.sweep_fn is not None:
                q_ref[self.sw], qd_ref[self.sw] = mc.sweep_fn(mc)
            elif mc.sweep_hold:
                q_ref[self.sw], qd_ref[self.sw] = 0.0, 0.0
        self.orig(q_ref, qd_ref)
        d = mc.d
        f = np.abs(d.actuator_force[mc.aid])
        v = d.qvel[mc.dof]
        np.maximum(self.tq, f, out=self.tq)
        np.maximum(self.om, np.abs(v), out=self.om)
        np.maximum(self.pw, np.abs(d.actuator_force[mc.aid] * v), out=self.pw)
        cl = 0.0
        if d.ncon:
            cl = mc.contact_load()
            if cl > self.contact_max:
                self.contact_max = cl
            b = int(mc.t / 0.05)                  # peak contact per 50 ms bin (the gate windows read it)
            if cl > self.cbins.get(b, 0.0):
                self.cbins[b] = cl
        self.k += 1
        while CAPTURE and mc.t >= CAPTURE[0] - 1e-9:
            CAPTURE.pop(0)
            cons = []
            c6 = np.zeros(6)
            for i in range(d.ncon):
                mc.mj.mj_contactForce(mc.m, d, i, c6)
                cons.append((d.contact[i].pos.tolist(), float(abs(c6[0]))))
            SNAPS.append(dict(t=mc.t, qpos=d.qpos.tolist(), com=mc.com().tolist(), contacts=cons, mass=mc.mass,
                              contact_W=cl, tq=np.abs(d.actuator_force[mc.aid]).tolist()))
        if self.k % 10 == 0:
            wet = float(getattr(mc.env.solver.diag, "max_submerged", 0.0))
            self.samples.append((mc.t, float(np.linalg.norm(d.qvel[:3])),
                                 float(np.abs(d.qpos[mc.qad[self.fe]]).max()) if len(self.fe) else 0.0,
                                 wet, cl))

    def peaks(self):
        mc = self.mc
        roles = list(mc.role)
        names = list(mc.env.act_names)
        return dict(hinge=names, role=roles, tq_Nm=self.tq.tolist(), om_rad_s=self.om.tolist(),
                    shaft_W=self.pw.tolist(), contact_max_W=self.contact_max,
                    contact_bins_50ms={str(k): round(v, 2) for k, v in sorted(self.cbins.items())},
                    E_elec_J=mc.E_elec.tolist(), mass=mc.mass, bad_qacc=int(mc.bad_qacc))

    def x2(self, t0, t1):
        """Share of samples in [t0, t1] with speed < 6 m/s and |feather| > 30 deg, dry and out of contact."""
        S = np.array(self.samples) if self.samples else np.zeros((0, 5))
        sel = (S[:, 0] >= t0) & (S[:, 0] <= t1) if len(S) else np.zeros(0, bool)
        if not sel.any():
            return None
        s = S[sel]
        air = (s[:, 3] <= 0.0) & (s[:, 4] <= 0.0)
        hit = air & (s[:, 1] < 6.0) & (s[:, 2] > math.radians(30.0))
        return dict(share=float(hit.mean()), air_share=float(air.mean()), n=int(sel.sum()))


def configure(cfg: dict) -> None:
    """Rebind mission.Machine and m3b_sweep.SweepMachine to `make` with ``cfg`` (this process only)."""
    import c1_pilot
    import mission
    import m3b_sweep
    if "Machine" not in _ORIG:
        _ORIG["Machine"] = mission.Machine
        _ORIG["SweepMachine"] = m3b_sweep.SweepMachine
        _ORIG["c1_body"] = c1_pilot.body
    CFG.clear()
    CFG.update(cfg)
    mission.Machine = make

    def sweep_factory(domain="land", **kw):
        kw.pop("mass", None)
        save = CFG.get("sweep")
        CFG["sweep"] = True
        try:
            return make(domain, **kw)
        finally:
            CFG["sweep"] = save
    m3b_sweep.SweepMachine = sweep_factory
    if cfg.get("wing"):
        c1_pilot.body = lambda: body_variant(cfg["wing"])
    else:
        c1_pilot.body = _ORIG["c1_body"]
