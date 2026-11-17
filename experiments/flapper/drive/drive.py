"""Stroke drives built as mechanisms in the model (pass D1,
`PREREG_D1_stroke_drive.md`).

Each candidate is bodies, joints and constraints added to the compiled
flapper, driven by one motor, so the stroke is what the mechanism makes of
the motor's torque, not a prescribed angle:

* "DD": the motor on the stroke hinge through a gear; the other wing by a
  joint coupling.  A position loop on the sinusoid commands it.
* "FB": a crank-rocker four-bar per wing, the two cranks geared to turn
  together (mirrored), designed for a time ratio of 1.
* "SY": one crank on the centre line driving a yoke on a vertical slide (pin
  in a slot), and a link from the yoke to each wing.
* "CT": the four-bar of "FB" with a cable (a one-sided tendon) in place of
  the coupler, and a spring at each shoulder at rest at the top of the stroke.

The motor is a witness model: shaft torque ``tau_s (u - omega/omega_0)`` with
``u`` in [-1, 1], a rotor of inertia ``J_r`` behind an ideal gear ``N``, copper
loss ``tau^2 omega_0 / tau_s``.  It is not a part chosen.

Geometry is laid out in each wing's stroke plane: ``u`` outward from the
shoulder along the span at rest, ``v`` up.  The lever is the point of the
wing's span line (its feather axis) ``a`` from the shoulder, so the drive does
not load the feather hinge.  Every loop is closed at the built pose (stroke 0),
which MuJoCo's ``connect`` takes as the point to keep together.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

MOTOR = dict(tau_s=1.0, omega_0=1000.0, J_r=2e-5)


def _meet(Q, L, a, upper=True):
    """The point at distance ``a`` from the shoulder (the origin) and ``L``
    from ``Q``: the lever's place for a link of length ``L`` from ``Q``.
    ``upper`` picks the intersection to the left of the line from the
    origin to ``Q`` (counter-clockwise), which is the one near (a, 0) when
    ``Q`` is below it."""
    Q = np.asarray(Q, float)
    dq = np.linalg.norm(Q, axis=-1)
    x = (a * a - L * L + dq * dq) / (2.0 * dq)
    h = np.sqrt(np.maximum(a * a - x * x, 0.0))
    e = Q / dq[..., None]
    perp = np.stack([-e[..., 1], e[..., 0]], -1)
    return x[..., None] * e + (1.0 if upper else -1.0) * h[..., None] * perp


class Drive:
    """One candidate for one operating point.

    kind: "DD", "FB", "SY", "CT".  Phi_deg: the stroke it is designed for
    (peak to peak).  N: gear ratio from motor to crank (or to the hinge for
    "DD").  a: lever length, m.  lam: link mass per length, kg/m.  k_spring:
    N m/rad per wing ("CT").  dt: the step the loops' stiffness is set for
    (solref = 2 dt)."""

    def __init__(self, kind, Phi_deg, f, N, *, a=0.03, lam=0.015, k_spring=0.0, motor=None,
                 dt=1e-4, wn=150.0, feather_axis=0.0):
        if kind not in ("DD", "FB", "SY", "CT"):
            raise ValueError(kind)
        self.kind, self.N, self.a, self.lam, self.k = kind, float(N), float(a), float(lam), float(k_spring)
        self.bm = math.radians(Phi_deg) / 2.0
        self.w = 2.0 * math.pi * f
        self.motor = dict(MOTOR, **(motor or {}))
        self.dt, self.wn = dt, wn
        # `freerig.run_free`'s feather_axis: the feather hinge moved this
        # share of the chord towards the leading edge; the lever goes with it
        self.feather_axis = float(feather_axis)
        self._design()

    # ------------------------------------------------------------ kinematics

    def _design(self):
        a, bm = self.a, self.bm
        if self.kind in ("FB", "CT"):
            # crank centre on the chord through the lever's two ends; time ratio 1
            self.O2 = np.array([a * math.cos(bm), -2.0 * a])
            self.r, self.L = a * math.sin(bm), 2.0 * a
            th = np.linspace(0.0, 2.0 * math.pi, 20001)
            P0 = np.array([a, 0.0])
            gap = np.linalg.norm(self.O2 + self.r * np.stack([np.cos(th), np.sin(th)], -1) - P0, axis=1) - self.L
            roots = th[:-1][np.sign(gap[:-1]) != np.sign(gap[1:])]
            # the root at which the lever rises as the crank turns positive,
            # refined to where the built pose (stroke 0) closes the loop
            for t0 in roots:
                if float(self.beta(t0 + 0.01)[0] - self.beta(t0 - 0.01)[0]) > 0.0:
                    lo, hi = t0 - 0.01, t0 + 0.01
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        if float(self.beta(mid)[0]) < 0.0:
                            lo = mid
                        else:
                            hi = mid
                    self.th0 = 0.5 * (lo + hi)
                    break
        elif self.kind == "SY":
            # the yoke's mid height 2a below the shoulder, links vertical there;
            # the crank sized for the stroke's whole swing (the swing is not
            # quite symmetric about the built pose: the link leans either way)
            lo, hi = 0.0, a
            for _ in range(60):
                r = 0.5 * (lo + hi)
                if float(self._beta_sy(math.pi / 2.0, r)[0] - self._beta_sy(-math.pi / 2.0, r)[0]) < 2.0 * bm:
                    lo = r
                else:
                    hi = r
            self.r, self.L, self.th0 = 0.5 * (lo + hi), 2.0 * a, 0.0
        else:
            self.r, self.L, self.th0 = 0.0, 0.0, 0.0
        if self.kind != "DD":
            th = np.linspace(0.0, 2.0 * math.pi, 4096, endpoint=False)
            b = self.beta(self.th0 + th)
            self.amp_deg = float(np.degrees(b.max() - b.min()))
            self.mean_deg = float(np.degrees(0.5 * (b.max() + b.min())))
            c1, s1 = 2.0 * np.mean(b * np.cos(th)), 2.0 * np.mean(b * np.sin(th))
            harm = [math.hypot(2.0 * np.mean(b * np.cos(k * th)), 2.0 * np.mean(b * np.sin(k * th)))
                    for k in (1, 2, 3)]
            self.h2, self.h3 = harm[1] / harm[0], harm[2] / harm[0]
            # stroke ~ A sin(th - phi): phi where the fundamental crosses zero rising
            self.phi = float(math.atan2(-c1, s1))
            self.down_share = float(np.mean(np.diff(np.concatenate([b, b[:1]])) < 0.0))
        else:
            self.amp_deg, self.mean_deg, self.h2, self.h3, self.phi, self.down_share = (
                math.degrees(2 * self.bm), 0.0, 0.0, 0.0, 0.0, 0.5)

    def _beta_sy(self, th, r):
        Q = np.stack([np.full_like(np.atleast_1d(th), self.a, dtype=float),
                      -2.0 * self.a + r * np.sin(np.atleast_1d(th))], -1)
        P = _meet(Q, 2.0 * self.a, self.a, upper=True)
        return np.arctan2(P[..., 1], P[..., 0])

    def beta(self, th):
        """The stroke angle, rad, at crank angle ``th`` (absolute), closed form."""
        th = np.atleast_1d(np.asarray(th, float))
        if self.kind == "SY":
            return self._beta_sy(th, self.r)
        Q = self.O2 + self.r * np.stack([np.cos(th), np.sin(th)], -1)
        P = _meet(Q, self.L, self.a, upper=True)
        return np.arctan2(P[..., 1], P[..., 0])

    # --------------------------------------------------------------- building

    def compile(self, p, **kw):
        """`compile_phenotype`'s signature and result, with the mechanism in."""
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
        for side, nm in ((1, "seg1_wing"), (-1, "seg2_wing")):
            bid = plain.body(nm).id
            assert int(plain.body_parentid[bid]) == hb
            # the feather hinge's place on the wing, as freerig moves it
            sel = pan.body_id == bid
            cdir = pan.chord_local[sel][0] / np.linalg.norm(pan.chord_local[sel][0])
            if float(pan.pos_local[sel].mean(0) @ cdir) < 0.0:
                cdir = -cdir
            shift = self.feather_axis * float(pan.chord[sel].mean()) * cdir
            Rw = Rh.T @ pdat.xmat[bid].reshape(3, 3)          # wing frame in the hull's
            wings[side] = (nm, plain.body_pos[bid].copy() + Rw @ shift)
            self.lever_local[side] = shift + np.array([self.a, 0.0, 0.0])
        self.wings = wings
        ref = f"{2.0 * self.dt} 1"
        eq = root.find("equality")
        if eq is None:
            eq = ET.SubElement(root, "equality")
        act = root.find("actuator")
        mass = lambda length: f"{max(self.lam * length, 1e-6)}"
        self.loops = []

        def at(side, uv):
            """Hull-frame point of stroke-plane (u, v) for a wing."""
            S = wings[side][1]
            return np.array([S[0], S[1] + side * uv[0], S[2] + uv[1]])

        def joint(body, name, side, kind="hinge"):
            ET.SubElement(body, "joint", name=name, type=kind, axis=f"{side} 0 0" if kind == "hinge" else "0 0 1",
                          limited="false", armature="1e-7", damping="0")

        def rod(body, p0, p1, size=0.002):
            ET.SubElement(body, "geom", type="capsule", fromto=" ".join(f"{v:.6g}" for v in (*p0, *p1)),
                          size=f"{size}", mass=mass(float(np.linalg.norm(np.subtract(p1, p0)))),
                          contype="0", conaffinity="0", rgba="0.85 0.35 0.1 1")

        if self.kind == "DD":
            ET.SubElement(eq, "joint", joint1="seg2_wing_j", joint2="seg1_wing_j", polycoef="0 1 0 0 0",
                          solref=ref)
            self.input_joint = "seg1_wing_j"
        if self.kind in ("FB", "CT"):
            Q0 = self.O2 + self.r * np.array([math.cos(self.th0), math.sin(self.th0)])
            P0 = np.array([self.a, 0.0])
            for side in (1, -1):
                O, Qh, Ph = at(side, self.O2), at(side, Q0), at(side, P0)
                cr = ET.SubElement(hull, "body", name=f"crank{side:+d}", pos=" ".join(f"{v:.6g}" for v in O))
                joint(cr, f"crank{side:+d}_j", side)
                rod(cr, (0, 0, 0), Qh - O, 0.003)
                if self.kind == "FB":
                    cp = ET.SubElement(cr, "body", name=f"coupler{side:+d}",
                                       pos=" ".join(f"{v:.6g}" for v in Qh - O))
                    joint(cp, f"coupler{side:+d}_j", side)
                    rod(cp, (0, 0, 0), Ph - Qh)
                    e = Ph - Qh
                    ET.SubElement(eq, "connect", body1=f"coupler{side:+d}", body2=wings[side][0],
                                  anchor=" ".join(f"{v:.6g}" for v in e), solref=ref)
                    self.loops.append(f"coupler{side:+d}")
                else:
                    ET.SubElement(cr, "site", name=f"pin{side:+d}", pos=" ".join(f"{v:.6g}" for v in Qh - O))
                    wb = next(b for b in root.iter("body") if b.get("name") == wings[side][0])
                    ET.SubElement(wb, "site", name=f"lever{side:+d}",
                                  pos=" ".join(f"{v:.6g}" for v in self.lever_local[side]))
            ET.SubElement(eq, "joint", joint1="crank-1_j", joint2="crank+1_j", polycoef="0 1 0 0 0", solref=ref)
            if self.kind == "CT":
                ten = root.find("tendon")
                if ten is None:
                    ten = ET.SubElement(root, "tendon")
                for side in (1, -1):
                    sp = ET.SubElement(ten, "spatial", name=f"cable{side:+d}", limited="true",
                                       range=f"0 {self.L}", solreflimit=ref, width="0.001")
                    ET.SubElement(sp, "site", site=f"pin{side:+d}")
                    ET.SubElement(sp, "site", site=f"lever{side:+d}")
            self.input_joint = "crank+1_j"
        if self.kind == "SY":
            S = wings[1][1]
            Oc = np.array([S[0], 0.0, S[2] - 2.0 * self.a])
            cr = ET.SubElement(hull, "body", name="crank", pos=" ".join(f"{v:.6g}" for v in Oc))
            joint(cr, "crank_j", 1)
            rod(cr, (0, 0, 0), (0, self.r, 0), 0.003)
            yk = ET.SubElement(hull, "body", name="yoke", pos=" ".join(f"{v:.6g}" for v in Oc))
            joint(yk, "yoke_j", 1, kind="slide")
            half = abs(S[1]) + self.a
            rod(yk, (0, -half, 0), (0, half, 0), 0.003)
            fo = ET.SubElement(yk, "body", name="follower", pos=f"0 {self.r} 0")
            ET.SubElement(fo, "joint", name="follower_j", type="slide", axis="0 1 0", limited="false",
                          armature="1e-7")
            ET.SubElement(fo, "geom", type="sphere", size="0.003", mass=mass(0.01), contype="0", conaffinity="0")
            ET.SubElement(eq, "connect", body1="follower", body2="crank", anchor="0 0 0", solref=ref)
            self.loops.append("follower")
            for side in (1, -1):
                Y = np.array([0.0, side * half, 0.0])
                ln = ET.SubElement(yk, "body", name=f"link{side:+d}", pos=" ".join(f"{v:.6g}" for v in Y))
                joint(ln, f"link{side:+d}_j", side)
                e = np.array([0.0, 0.0, 2.0 * self.a])
                rod(ln, (0, 0, 0), e)
                ET.SubElement(eq, "connect", body1=f"link{side:+d}", body2=wings[side][0],
                              anchor=" ".join(f"{v:.6g}" for v in e), solref=ref)
                self.loops.append(f"link{side:+d}")
            self.input_joint = "crank_j"
        mt = self.motor
        ET.SubElement(act, "general", name="drive_motor", joint=self.input_joint, gear=f"{self.N}",
                      ctrllimited="true", ctrlrange="-1 1", gainprm=f"{mt['tau_s']}",
                      biastype="affine", biasprm=f"0 0 {-mt['tau_s'] / mt['omega_0']}")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        jin = model.joint(self.input_joint)
        model.dof_armature[jin.dofadr[0]] += mt["J_r"] * self.N ** 2
        if self.kind == "CT":
            for nm in ("seg1_wing_j", "seg2_wing_j"):
                j = model.joint(nm)
                model.jnt_stiffness[j.id] = self.k
                model.qpos_spring[j.qposadr[0]] = self.bm
        data = mujoco.MjData(model)
        return model, data, names, build_panels(p, model)

    # ---------------------------------------------------------------- running

    def _ids(self, m):
        self.aid = m.actuator("drive_motor").id
        jin = m.joint(self.input_joint)
        self.q_in, self.v_in = int(jin.qposadr[0]), int(jin.dofadr[0])
        self.q_st, self.v_st = int(m.joint("seg1_wing_j").qposadr[0]), int(m.joint("seg1_wing_j").dofadr[0])

    def setup(self, m, d):
        import mujoco
        self._ids(m)
        e, col = np.zeros(m.nv), np.zeros(m.nv)
        e[self.v_in] = 1.0
        mujoco.mj_mulM(m, d, col, e)
        self.J_in = float(col[self.v_in])
        if self.kind == "DD":
            e2 = np.zeros(m.nv)
            e2[int(m.joint("seg2_wing_j").dofadr[0])] = 1.0
            mujoco.mj_mulM(m, d, col, e2)
            self.J_in += float(col[int(m.joint("seg2_wing_j").dofadr[0])])
        self.u_i = 0.0
        self.rec = dict(Pm=[], Pe=[], u=[], tau=[], om=[], gap=[], floop=[], stroke=[], th=[], slack=[])

    def init_state(self, m, d):
        self._ids(m)
        if self.kind == "DD":
            d.qvel[self.v_st] = self.bm * self.w
            d.qvel[int(m.joint("seg2_wing_j").dofadr[0])] = self.bm * self.w
        else:
            d.qvel[self.v_in] = self.w
            if self.kind in ("FB", "CT"):
                d.qvel[int(m.joint("crank-1_j").dofadr[0])] = self.w

    def phase(self, m, d, t):
        """The feather's phase: the stroke ~ sin(phase)."""
        if self.kind == "DD":
            return self.w * t
        return self.th0 + float(d.qpos[self.q_in]) - self.phi

    def control(self, m, d, t):
        mt = self.motor
        if self.kind == "DD":
            s, c = math.sin(self.w * t), math.cos(self.w * t)
            q_r, qd_r, qdd_r = self.bm * s, self.bm * self.w * c, -self.bm * self.w ** 2 * s
            tq = self.J_in * (qdd_r + self.wn ** 2 * (q_r - d.qpos[self.q_st]) + 2 * self.wn * (qd_r - d.qvel[self.v_st]))
            u = qd_r * self.N / mt["omega_0"] + tq / (self.N * mt["tau_s"])
        else:
            e = self.w - float(d.qvel[self.v_in])
            kp = self.J_in * 2 * math.pi * 10.0 / (self.N * mt["tau_s"])
            u0 = self.w * self.N / mt["omega_0"] + self.u_i + kp * e
            if abs(u0) < 1.0:
                self.u_i += kp * 2 * math.pi * 2.0 * e * self.dt
            u = u0
        d.ctrl[self.aid] = min(max(u, -1.0), 1.0)

    def after_step(self, m, d, keep):
        import mujoco
        mt = self.motor
        tau, om = float(d.actuator_force[self.aid]), float(d.actuator_velocity[self.aid])
        if not keep:
            return tau * om
        Pm = tau * om
        Pe = Pm + tau * tau * mt["omega_0"] / mt["tau_s"]
        r = self.rec
        r["Pm"].append(Pm)
        r["Pe"].append(Pe)
        r["u"].append(float(d.ctrl[self.aid]))
        r["tau"].append(tau)
        r["om"].append(om)
        r["stroke"].append(float(d.qpos[self.q_st]))
        r["th"].append(self.th0 + float(d.qpos[self.q_in]) if self.kind != "DD" else 0.0)
        eqr = d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        gap, fl = 0.0, 0.0
        if eqr.any():
            ids, pos, frc = d.efc_id[: d.nefc][eqr], d.efc_pos[: d.nefc][eqr], d.efc_force[: d.nefc][eqr]
            for i in np.unique(ids):
                if int(m.eq_type[i]) == mujoco.mjtEq.mjEQ_CONNECT:
                    sel = ids == i
                    gap = max(gap, float(np.linalg.norm(pos[sel])))
                    fl = max(fl, float(np.linalg.norm(frc[sel])))
        if self.kind == "CT":
            lim = d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON
            taut = lim & (d.efc_force[: d.nefc] > 1e-9)
            fl = max(fl, float(np.abs(d.efc_force[: d.nefc][lim]).max()) if lim.any() else 0.0)
            r["slack"].append(0.0 if taut.any() else 1.0)
        r["gap"].append(gap)
        r["floop"].append(fl)
        return Pm

    def summary(self):
        r = {k: np.asarray(v) for k, v in self.rec.items()}
        b = r["stroke"]
        out = dict(kind=self.kind, N=self.N, a=self.a, lam=self.lam, k_spring=self.k,
                   motor=self.motor, r_crank=self.r, L_link=self.L,
                   design_amp_deg=self.amp_deg, design_mean_deg=self.mean_deg,
                   design_down_share=self.down_share, design_h2=self.h2, design_h3=self.h3,
                   amp_deg=float(np.degrees(b.max() - b.min())), mean_deg=float(np.degrees(0.5 * (b.max() + b.min()))),
                   down_share=float(np.mean(np.diff(b) < 0.0)),
                   shaft_W=float(r["Pm"].mean()), shaft_peak_W=float(np.abs(r["Pm"]).max()),
                   elec_W=float(r["Pe"].mean()), elec_peak_W=float(r["Pe"].max()),
                   copper_W=float((r["Pe"] - r["Pm"]).mean()),
                   motor_torque_peak=float(np.abs(r["tau"]).max()), motor_speed_max=float(np.abs(r["om"]).max()),
                   at_limit=float(np.mean(np.abs(r["u"]) >= 0.999)),
                   loop_gap_max_mm=float(1e3 * r["gap"].max()), loop_force_peak_N=float(r["floop"].max()))
        # the beat the mechanism actually made: a crank that slows keeps its
        # amplitude and loses frequency (the input's mean speed over the planned one)
        out["freq_ratio"] = 1.0 if self.kind == "DD" else float(r["om"].mean() / (self.N * self.w))
        if self.kind != "DD":
            out["kin_err_deg"] = float(np.degrees(np.abs(b - self.beta(r["th"])).max()))
        if self.kind == "CT":
            out["slack_share"] = float(r["slack"].mean())
        return out
