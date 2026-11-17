"""Crank mechanisms that make two stroke amplitudes by a locked adjuster
(pass D4, `PREREG_D4_variable_stroke.md`).  `drive.Drive`'s interface, with
`drive.py` unedited: the motor, the speed loop, the phase and the recording
are inherited; the design, the bodies and the lock are here.

* "VL": a four-bar whose coupler ends on a slider along the wing lever.
* "VR": a four-bar whose crank pin slides along the crank arm.
* "VP": two same-way cranks at a phase, a rod from each to the ends of a beam
  whose middle is pinned on the wing lever.  The second crank turns on a
  carrier hinged on the hull; the carrier is the adjuster.

One generator per candidate; the only argument that differs between the two
operating points is the adjuster's coordinate (``setting``).  The model is
compiled with the adjuster there, every loop closed at stroke 0, and the
adjuster's joint is held by a one-joint equality at solref 2 dt.  That
equality's constraint force is the holding force (N) or torque (N m).
"""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from drive import Drive, _meet

HERE = Path(__file__).resolve().parent
DESIGN_FILE = HERE.parent / "results" / "D4_design.json"
A_T = 0.06                      # the lever at take-off (and at cruise for VR, VP)
STROKES = {"takeoff": 85.0, "cruise": 40.0}
BLOCK_MASS = 0.002              # slider, pin block, carrier: a point mass (PREREG 1)
N_TAB = 720


# ------------------------------------------------------------------ closed form

def _fourbar_beta(O, r, L, a, th):
    th = np.atleast_1d(np.asarray(th, float))
    Q = np.asarray(O) + r * np.stack([np.cos(th), np.sin(th)], -1)
    dq = np.linalg.norm(Q, axis=-1)
    x = (a * a - L * L + dq * dq) / (2.0 * dq)
    if np.any(a * a - x * x < 0.0):
        return None
    P = _meet(Q, L, a, upper=True)
    return np.arctan2(P[..., 1], P[..., 0])


def _vp_solve(xc, y, r, L, w, delta, a=A_T, n=N_TAB):
    """Stroke and beam angle over the first crank's angle (n points from 0).
    Crank A at th about (xc - w, y), crank B at th + delta about (xc + w, y);
    beam ends P -/+ w e(phi).  None if the loop opens anywhere."""
    from scipy.optimize import fsolve
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    OA, OB = np.array([xc - w, y]), np.array([xc + w, y])
    out, x = [], np.array([0.0, 0.0])
    for t in np.concatenate([th[:1], th]):
        QA = OA + r * np.array([math.cos(t), math.sin(t)])
        QB = OB + r * np.array([math.cos(t + delta), math.sin(t + delta)])

        def F(x):
            P = a * np.array([math.cos(x[0]), math.sin(x[0])])
            e = np.array([math.cos(x[1]), math.sin(x[1])])
            return [np.linalg.norm(P - w * e - QA) - L, np.linalg.norm(P + w * e - QB) - L]
        x, info, _, _ = fsolve(F, x, full_output=True, xtol=1e-13)
        if np.abs(info["fvec"]).max() > 1e-9:
            return None
        out.append(x.copy())
    return th, np.array(out[1:])


def _amp_mean(b):
    return float(np.degrees(b.max() - b.min())), float(np.degrees(0.5 * (b.max() + b.min())))


def solve_designs():
    """The frozen conditions (PREREG 2): stroke 85 and 40 deg, mid of the
    extremes at 0 at both settings, by one set of lengths and centres."""
    from scipy.optimize import fsolve
    th = np.linspace(0.0, 2.0 * math.pi, 4096, endpoint=False)
    bT, bC = (math.radians(STROKES[k] / 2.0) for k in ("takeoff", "cruise"))
    out = {}

    def res4(pairs):
        v = []
        for b, want in pairs:
            if b is None:
                return [1e3] * 4
            amp, mean = _amp_mean(b)
            v += [amp - want, mean]
        return v

    # VL: crank centre depth 160 mm; unknowns centre's u, coupler, crank radius, cruise lever
    y = -0.16
    r0 = A_T * math.sin(bT)
    aC0 = r0 / math.sin(bC)
    g = [0.5 * (A_T + aC0), math.sqrt((0.5 * (aC0 - A_T)) ** 2 + y * y + r0 * r0), r0, aC0]
    f = lambda x: res4([(_fourbar_beta((x[0], y), x[2], x[1], A_T, th), 85.0),
                        (_fourbar_beta((x[0], y), x[2], x[1], x[3], th), 40.0)])
    x, info, ier, _ = fsolve(f, g, full_output=True)
    assert np.abs(info["fvec"]).max() < 1e-6, "VL design did not solve"
    out["VL"] = dict(O=[float(x[0]), y], L=float(x[1]), r=float(x[2]),
                     setting=dict(takeoff=A_T, cruise=float(x[3])))
    # VR: lever 60 mm, depth 120 mm; unknowns centre's u, coupler, the two radii
    y = -0.12
    g = [A_T * math.cos(0.5 * (bT + bC)), math.sqrt(y * y + (A_T * math.sin(bT)) ** 2 * 0.6),
         A_T * math.sin(bT), A_T * math.sin(bC)]
    f = lambda x: res4([(_fourbar_beta((x[0], y), x[2], x[1], A_T, th), 85.0),
                        (_fourbar_beta((x[0], y), x[3], x[1], A_T, th), 40.0)])
    x, info, ier, _ = fsolve(f, g, full_output=True)
    assert np.abs(info["fvec"]).max() < 1e-6, "VR design did not solve"
    out["VR"] = dict(O=[float(x[0]), y], L=float(x[1]), a=A_T,
                     setting=dict(takeoff=float(x[2]), cruise=float(x[3])))
    # VP: rods 120 mm, beam half length 50 mm; unknowns the centres' middle, depth, radius, cruise phase
    L, w = 0.12, 0.05

    def f(x):
        sT, sC = _vp_solve(x[0], x[1], x[2], L, w, 0.0, n=360), _vp_solve(x[0], x[1], x[2], L, w, x[3], n=360)
        return res4([(None if sT is None else sT[1][:, 0], 85.0), (None if sC is None else sC[1][:, 0], 40.0)])
    x, info, ier, _ = fsolve(f, [0.0848, -0.112, 0.0378, math.radians(117.4)], full_output=True, xtol=1e-10)
    assert np.abs(info["fvec"]).max() < 1e-6, "VP design did not solve"
    out["VP"] = dict(xc=float(x[0]), y=float(x[1]), r=float(x[2]), L=L, w=w, a=A_T,
                     setting=dict(takeoff=0.0, cruise=float(x[3])))
    return out


def designs():
    if DESIGN_FILE.exists():
        return json.loads(DESIGN_FILE.read_text())
    d = solve_designs()
    DESIGN_FILE.write_text(json.dumps(d, indent=1))
    return d


# ----------------------------------------------------------------------- drive

class VA(Drive):
    """cand: "VL", "VR", "VP".  point: "takeoff" or "cruise" (picks the
    adjuster's setting and the stroke it is meant to make).  setting: the
    adjuster's coordinate, if not the point's (V0 uses it)."""

    def __init__(self, cand, point, f, N, *, lam=0.015, motor=None, dt=1e-4, feather_axis=0.0, setting=None,
                 lock_armature=1e-7, gear_inertia=0.0):
        # lock_armature: inertia on the adjuster's own (locked) coordinate, kg or
        # kg m^2.  An equality in this engine is as stiff as the inertia it acts on;
        # on a coordinate that does not move it changes nothing else (RESULT_D4).
        # gear_inertia: rotating inertia on VP's second crank, kg m^2, for the
        # same reason at its gear mesh; this one is a real flywheel and is disclosed.
        self.lock_armature, self.gear_inertia = float(lock_armature), float(gear_inertia)
        if cand not in ("VL", "VR", "VP"):
            raise ValueError(cand)
        self.cand, self.point = cand, point
        self.dz = designs()[cand]
        self.setting = float(self.dz["setting"][point] if setting is None else setting)
        self.setting_T = float(self.dz["setting"]["takeoff"])
        a = self.setting if cand == "VL" else A_T
        super().__init__("FB", STROKES[point], f, N, a=a, lam=lam, motor=motor, dt=dt, feather_axis=feather_axis)

    # ------------------------------------------------------------ kinematics

    def _design(self):
        dz = self.dz
        if self.cand == "VL":
            self.O2, self.r, self.L = np.array(dz["O"]), dz["r"], dz["L"]
        elif self.cand == "VR":
            self.O2, self.r, self.L = np.array(dz["O"]), self.setting, dz["L"]
        else:
            self.r, self.L, self.wb = dz["r"], dz["L"], dz["w"]
            self.OA = np.array([dz["xc"] - dz["w"], dz["y"]])
            self.OB = np.array([dz["xc"] + dz["w"], dz["y"]])
            self.O2 = self.OA
            sol = _vp_solve(dz["xc"], dz["y"], self.r, self.L, self.wb, self.setting)
            if sol is None:
                raise ValueError("VP does not close at every crank angle at this phase")
            self._tab_th, tab = sol
            self._tab_b, self._tab_p = tab[:, 0], np.unwrap(tab[:, 1])
        th = np.linspace(0.0, 2.0 * math.pi, 4096, endpoint=False)
        b = self.beta(th)
        up = np.flatnonzero((b < 0.0) & (np.roll(b, -1) >= 0.0))
        lo, hi = th[up[0]], th[up[0]] + th[1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if float(self.beta(mid)[0]) < 0.0:
                lo = mid
            else:
                hi = mid
        self.th0 = 0.5 * (lo + hi)
        b = self.beta(self.th0 + th)
        self.amp_deg = float(np.degrees(b.max() - b.min()))
        self.mean_deg = float(np.degrees(0.5 * (b.max() + b.min())))
        c1, s1 = 2.0 * np.mean(b * np.cos(th)), 2.0 * np.mean(b * np.sin(th))
        harm = [math.hypot(2.0 * np.mean(b * np.cos(k * th)), 2.0 * np.mean(b * np.sin(k * th))) for k in (1, 2, 3)]
        self.h2, self.h3 = harm[1] / harm[0], harm[2] / harm[0]
        self.phi = float(math.atan2(-c1, s1))
        self.down_share = float(np.mean(np.diff(np.concatenate([b, b[:1]])) < 0.0))

    def _interp(self, tab, th):
        th = np.atleast_1d(np.asarray(th, float))
        x = np.mod(th, 2.0 * math.pi) / (2.0 * math.pi) * N_TAB
        i = np.floor(x).astype(int) % N_TAB
        fr = x - np.floor(x)
        return tab[i] * (1.0 - fr) + tab[(i + 1) % N_TAB] * fr

    def beta(self, th):
        if self.cand == "VP":
            return self._interp(self._tab_b, th)
        b = _fourbar_beta(self.O2, self.r, self.L, self.a, th)
        if b is None:
            raise ValueError("the four-bar does not close at every crank angle at this setting")
        return b

    def beam_angle(self, th):
        return self._interp(self._tab_p, th)

    # --------------------------------------------------------------- building

    def compile(self, p, **kw):
        import mujoco

        from dytiscidae.core.mjcf import build_model_xml
        from dytiscidae.core.phenotype import build_panels
        xml, names = build_model_xml(p, **kw)
        root = ET.fromstring(xml)
        plain = mujoco.MjModel.from_xml_string(xml)
        hull = next(b for b in root.iter("body") if b.get("name") == "seg0_hull")
        pdat = mujoco.MjData(plain)
        mujoco.mj_forward(plain, pdat)
        pan = build_panels(p, plain)
        hb = plain.body("seg0_hull").id
        Rh = pdat.xmat[hb].reshape(3, 3)
        wings, self.lever_local = {}, {}
        for side, nm in ((1, "seg1_wing"), (-1, "seg2_wing")):       # as `Drive.compile`
            bid = plain.body(nm).id
            assert int(plain.body_parentid[bid]) == hb
            sel = pan.body_id == bid
            cdir = pan.chord_local[sel][0] / np.linalg.norm(pan.chord_local[sel][0])
            if float(pan.pos_local[sel].mean(0) @ cdir) < 0.0:
                cdir = -cdir
            shift = self.feather_axis * float(pan.chord[sel].mean()) * cdir
            Rw = Rh.T @ pdat.xmat[bid].reshape(3, 3)
            wings[side] = (nm, plain.body_pos[bid].copy() + Rw @ shift)
            self.lever_local[side] = shift + np.array([self.a, 0.0, 0.0])
        self.wings = wings
        ref = f"{2.0 * self.dt} 1"
        eq = root.find("equality")
        if eq is None:
            eq = ET.SubElement(root, "equality")
        act = root.find("actuator")
        mass = lambda length: f"{max(self.lam * length, 1e-6)}"
        fmt = lambda v: " ".join(f"{x:.9g}" for x in v)
        self.loops, self.locks = [], []

        def at(side, uv):
            S = wings[side][1]
            return np.array([S[0], S[1] + side * uv[0], S[2] + uv[1]])

        def vec(side, uv):
            return np.array([0.0, side * uv[0], uv[1]])

        def hinge(body, name, side, ref_=None, arm=1e-7):
            j = ET.SubElement(body, "joint", name=name, type="hinge", axis=f"{side} 0 0", limited="false",
                              armature=f"{arm}", damping="0")
            if ref_ is not None:
                j.set("ref", f"{ref_:.9g}")

        def rod(body, p0, p1, size=0.002):
            ET.SubElement(body, "geom", type="capsule", fromto=fmt((*p0, *p1)), size=f"{size}",
                          mass=mass(float(np.linalg.norm(np.subtract(p1, p0)))), contype="0", conaffinity="0",
                          rgba="0.85 0.35 0.1 1")

        def block(body):
            ET.SubElement(body, "geom", type="sphere", size="0.004", mass=f"{BLOCK_MASS}", contype="0",
                          conaffinity="0", rgba="0.1 0.3 0.8 1")

        def lock(name, joint):
            ET.SubElement(eq, "joint", name=name, joint1=joint, polycoef="0 0 0 0 0", solref=ref)
            self.locks.append(name)

        def connect(name, b1, b2, anchor):
            ET.SubElement(eq, "connect", name=name, body1=b1, body2=b2, anchor=fmt(anchor), solref=ref)
            self.loops.append(name)

        e0 = np.array([math.cos(self.th0), math.sin(self.th0)])
        P0 = np.array([self.a, 0.0])
        for side in (1, -1):
            s = f"{side:+d}"
            if self.cand in ("VL", "VR"):
                Q0 = self.O2 + self.r * e0
                cr = ET.SubElement(hull, "body", name=f"crank{s}", pos=fmt(at(side, self.O2)))
                hinge(cr, f"crank{s}_j", side)
                if self.cand == "VL":
                    rod(cr, (0, 0, 0), vec(side, self.r * e0), 0.003)
                    cp = ET.SubElement(cr, "body", name=f"coupler{s}", pos=fmt(vec(side, self.r * e0)))
                else:
                    # the arm as long as the take-off radius; the pin block on a slide along it
                    rod(cr, (0, 0, 0), vec(side, self.setting_T * e0), 0.003)
                    pb = ET.SubElement(cr, "body", name=f"pinblock{s}", pos=fmt(vec(side, self.r * e0)))
                    ET.SubElement(pb, "joint", name=f"adjust{s}_j", type="slide", axis=fmt(vec(side, e0)),
                                  limited="false", armature=f"{self.lock_armature}", damping="0",
                                  ref=f"{self.r - self.setting_T:.9g}")
                    block(pb)
                    lock(f"lock{s}", f"adjust{s}_j")
                    cp = ET.SubElement(pb, "body", name=f"coupler{s}", pos="0 0 0")
                hinge(cp, f"coupler{s}_j", side)
                rod(cp, (0, 0, 0), vec(side, P0 - Q0))
                if self.cand == "VL":
                    wb = next(b for b in root.iter("body") if b.get("name") == wings[side][0])
                    sl = ET.SubElement(wb, "body", name=f"slider{s}", pos=fmt(self.lever_local[side]))
                    ET.SubElement(sl, "joint", name=f"adjust{s}_j", type="slide", axis="1 0 0", limited="false",
                                  armature=f"{self.lock_armature}", damping="0",
                                  ref=f"{self.a - self.setting_T:.9g}")
                    block(sl)
                    lock(f"lock{s}", f"adjust{s}_j")
                    connect(f"loop{s}", f"coupler{s}", f"slider{s}", vec(side, P0 - Q0))
                else:
                    connect(f"loop{s}", f"coupler{s}", wings[side][0], vec(side, P0 - Q0))
            else:
                dl, w = self.setting, self.wb
                eB = np.array([math.cos(self.th0 + dl), math.sin(self.th0 + dl)])
                ph0 = float(self.beam_angle(self.th0)[0])
                eb = np.array([math.cos(ph0), math.sin(ph0)])
                QA, QB = self.OA + self.r * e0, self.OB + self.r * eB
                EA, EB = P0 - w * eb, P0 + w * eb
                cr = ET.SubElement(hull, "body", name=f"crank{s}", pos=fmt(at(side, self.OA)))
                hinge(cr, f"crank{s}_j", side)
                rod(cr, (0, 0, 0), vec(side, self.r * e0), 0.003)
                ra = ET.SubElement(cr, "body", name=f"rodA{s}", pos=fmt(vec(side, self.r * e0)))
                hinge(ra, f"rodA{s}_j", side)
                rod(ra, (0, 0, 0), vec(side, EA - QA))
                bm = ET.SubElement(ra, "body", name=f"beam{s}", pos=fmt(vec(side, EA - QA)))
                hinge(bm, f"beam{s}_j", side)
                rod(bm, (0, 0, 0), vec(side, EB - EA))
                ca = ET.SubElement(hull, "body", name=f"carrier{s}", pos=fmt(at(side, self.OB)))
                hinge(ca, f"adjust{s}_j", side, ref_=dl, arm=self.lock_armature)
                block(ca)
                lock(f"lock{s}", f"adjust{s}_j")
                cb = ET.SubElement(ca, "body", name=f"crankB{s}", pos="0 0 0")
                hinge(cb, f"crankB{s}_j", side, arm=max(self.gear_inertia, 1e-7))
                rod(cb, (0, 0, 0), vec(side, self.r * eB), 0.003)
                rb = ET.SubElement(cb, "body", name=f"rodB{s}", pos=fmt(vec(side, self.r * eB)))
                hinge(rb, f"rodB{s}_j", side)
                rod(rb, (0, 0, 0), vec(side, EB - QB))
                connect(f"mid{s}", f"beam{s}", wings[side][0], vec(side, P0 - EA))
                connect(f"endB{s}", f"rodB{s}", f"beam{s}", vec(side, EB - QB))
                # the second crank turns with the first, relative to its carrier
                ET.SubElement(eq, "joint", name=f"gearB{s}", joint1=f"crankB{s}_j", joint2=f"crank{s}_j",
                              polycoef="0 1 0 0 0", solref=ref)
        ET.SubElement(eq, "joint", name="gearLR", joint1="crank-1_j", joint2="crank+1_j", polycoef="0 1 0 0 0",
                      solref=ref)
        self.input_joint = "crank+1_j"
        mt = self.motor
        ET.SubElement(act, "general", name="drive_motor", joint=self.input_joint, gear=f"{self.N}",
                      ctrllimited="true", ctrlrange="-1 1", gainprm=f"{mt['tau_s']}",
                      biastype="affine", biasprm=f"0 0 {-mt['tau_s'] / mt['omega_0']}")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        model.dof_armature[model.joint(self.input_joint).dofadr[0]] += mt["J_r"] * self.N ** 2
        data = mujoco.MjData(model)
        return model, data, names, build_panels(p, model)

    # ---------------------------------------------------------------- running

    def setup(self, m, d):
        super().setup(m, d)
        self._loop_id = {int(m.eq(n).id): n for n in self.loops}
        self._lock_id = {int(m.eq(n).id): n for n in self.locks}
        self.rec2 = dict(gap={n: 0.0 for n in self.loops}, force={n: 0.0 for n in self.loops},
                         lock={n: [] for n in self.locks}, lock_slip={n: 0.0 for n in self.locks},
                         gear_slip=0.0, gear_force=0.0, trace=[])
        self._gear = None
        if self.cand == "VP":
            self._gear = (int(m.joint("crankB+1_j").qposadr[0]), int(m.joint("crank+1_j").qposadr[0]),
                          int(m.eq("gearB+1").id))

    def init_state(self, m, d):
        super().init_state(m, d)                  # both first cranks at the planned speed
        if self.cand == "VP":
            for s in ("+1", "-1"):
                d.qvel[int(m.joint(f"crankB{s}_j").dofadr[0])] = self.w

    def after_step(self, m, d, keep):
        import mujoco
        out = super().after_step(m, d, keep)
        if keep:
            eqr = d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
            ids, pos, frc = d.efc_id[: d.nefc][eqr], d.efc_pos[: d.nefc][eqr], d.efc_force[: d.nefc][eqr]
            r2 = self.rec2
            for i, n in self._loop_id.items():
                sel = ids == i
                r2["gap"][n] = max(r2["gap"][n], float(np.linalg.norm(pos[sel])))
                r2["force"][n] = max(r2["force"][n], float(np.linalg.norm(frc[sel])))
            for i, n in self._lock_id.items():
                sel = ids == i
                r2["lock"][n].append(float(frc[sel][0]))
                r2["lock_slip"][n] = max(r2["lock_slip"][n], abs(float(pos[sel][0])))
            if self._gear is not None:
                qb, qa, gi = self._gear
                r2["gear_slip"] = max(r2["gear_slip"], abs(float(d.qpos[qb] - d.qpos[qa])))
                r2["gear_force"] = max(r2["gear_force"], abs(float(frc[ids == gi][0])))
            if getattr(self, "trace_every", 0) and len(self.rec["th"]) % self.trace_every == 0:
                r2["trace"].append([float(d.time), self.rec["th"][-1], self.rec["stroke"][-1],
                                    float(self.beta(self.rec["th"][-1])[0]), self.rec["floop"][-1],
                                    float(d.qpos[self._gear[0]] - d.qpos[self._gear[1]]) if self._gear else 0.0,
                                    self.rec["tau"][-1], self.rec["om"][-1]])
        return out

    def summary(self):
        out = super().summary()
        r2 = self.rec2
        lk = {n: np.asarray(v) for n, v in r2["lock"].items()}
        out.update(kind=self.cand, point=self.point, setting=self.setting, setting_takeoff=self.setting_T,
                   lock_armature=self.lock_armature, gear_inertia=self.gear_inertia,
                   th0=self.th0, phi=self.phi,
                   loop_gap_mm={n: 1e3 * v for n, v in r2["gap"].items()},
                   loop_force_N=dict(r2["force"]),
                   lock_peak={n: float(np.abs(v).max()) for n, v in lk.items()},
                   lock_mean={n: float(v.mean()) for n, v in lk.items()},
                   lock_slip_max={n: float(v) for n, v in r2["lock_slip"].items()},
                   hold_peak=float(max(np.abs(v).max() for v in lk.values())),
                   hold_unit="N m" if self.cand == "VP" else "N",
                   gear_slip_max_rad=float(r2["gear_slip"]), gear_torque_peak=float(r2["gear_force"]))
        if r2["trace"]:
            out["trace"] = r2["trace"]
        return out
