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
                 dt=1e-4, wn=150.0, feather_axis=0.0, feather=None):
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
        # pass D2: what turns the root feather.  None: freerig's ideal servo.
        # dict(mode="motor", N=...): a second witness motor on the hinge;
        # dict(mode="crank", c_h=0.03): an arm on the stroke's crankshaft and a
        # ball-ended rod to a horn c_h ahead of the hinge ("FB" only);
        # dict(mode="passive", k=...): a torsion spring at rest at 0.
        # Psi_deg: the feather amplitude the motor follows or the arm is sized for.
        self.feather = dict(feather) if feather else None
        if self.feather and self.feather["mode"] in ("crank", "crank2") and kind != "FB":
            raise ValueError("the feather arm rides on the four-bar's crankshaft")
        self.undriven = {"stroke"} | ({"feather"} if self.feather else set())
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
        fe = self.feather
        if fe and fe["mode"] == "motor":
            ET.SubElement(eq, "joint", joint1="seg2_wing_jf", joint2="seg1_wing_jf", polycoef="0 1 0 0 0",
                          solref=ref)
        if fe and fe["mode"] in ("crank", "crank2"):
            self._feather_crank(root, eq, plain, pdat, Rh, ref, rod, mass)
        mt = self.motor
        if fe and fe["mode"] == "motor":
            ET.SubElement(act, "general", name="feather_motor", joint="seg1_wing_jf", gear=f"{fe['N']}",
                          ctrllimited="true", ctrlrange="-1 1", gainprm=f"{mt['tau_s']}",
                          biastype="affine", biasprm=f"0 0 {-mt['tau_s'] / mt['omega_0']}")
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
        if fe and fe["mode"] == "motor":
            model.dof_armature[model.joint("seg1_wing_jf").dofadr[0]] += mt["J_r"] * fe["N"] ** 2
        if fe and fe["mode"] == "passive":
            for nm in ("seg1_wing_jf", "seg2_wing_jf"):
                j = model.joint(nm)
                model.jnt_stiffness[j.id] = fe["k"]
                model.qpos_spring[j.qposadr[0]] = 0.0
        if fe and fe.get("range_deg"):
            for nm in ("seg1_wing_jf", "seg2_wing_jf"):      # the hinge's stop, if asked
                model.jnt_range[model.joint(nm).id] = np.radians([-fe["range_deg"], fe["range_deg"]])
        if fe and fe["mode"] in ("crank", "crank2"):
            for side in (1, -1):            # the horn, not where the rod's end was at the built pose
                i = model.eq(f"featherrod{side:+d}").id
                model.eq_data[i, 3:6] = self.horn_local[side]
        data = mujoco.MjData(model)
        return model, data, names, build_panels(p, model)

    def _feather_crank(self, root, eq, plain, pdat, Rh, ref, rod, mass):
        """Pass D2's arm: a second arm on each crank's shaft, under a horn on the
        wing's root chord ``c_h`` ahead of the feather hinge, and a rod with a
        ball at each end from the arm's pin to the horn.  The arm's radius,
        angle and the rod's length are fitted (on the model's own kinematics)
        so the feather follows ``-Psi cos(phase)``, the servo's reference."""
        import mujoco
        fe = self.feather
        c_h, Psi = float(fe.get("c_h", 0.03)), math.radians(fe.get("Psi_deg", 50.0))
        hb = plain.body("seg0_hull").id
        self.horn_local, self.arm = {}, {}
        # the horn in each wing's frame: c_h along the chord towards the leading edge
        for side, nm in ((1, "seg1_wing"), (-1, "seg2_wing")):
            shift = self.lever_local[side] - np.array([self.a, 0.0, 0.0])
            cdir = shift / max(np.linalg.norm(shift), 1e-12) if self.feather_axis else None
            if cdir is None:
                raise ValueError("the crank feather needs the hinge's chord direction (feather_axis > 0)")
            self.horn_local[side] = shift + c_h * cdir
        js, jf = plain.joint("seg1_wing_j"), plain.joint("seg1_wing_jf")
        # the feather hinge where freerig will put it (its feather_axis)
        plain.jnt_pos[jf.id] = plain.jnt_pos[jf.id] + (self.lever_local[1] - np.array([self.a, 0.0, 0.0]))
        d = mujoco.MjData(plain)
        bid = plain.body("seg1_wing").id

        def horn(beta, psi):
            d.qpos[:] = plain.qpos0
            d.qpos[js.qposadr[0]], d.qpos[jf.qposadr[0]] = beta, psi
            mujoco.mj_kinematics(plain, d)
            w = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ self.horn_local[1]
            return Rh.T @ (w - d.xpos[hb])                    # hull frame
        two = fe["mode"] == "crank2"
        O = self.wings[1][1] + np.array([0.0, self.O2[0], self.O2[1]])
        H0 = horn(0.0, 0.0)
        dx = H0[0] - O[0]
        if two:
            # crank2 (added after "crank" as frozen would not close, PREREG_D2
            # addendum): a shaft of its own along the span, under the horn by
            # `drop`, its pin turning in the horn's own plane, geared 1:1
            # (bevel) to the stroke crank
            drop = float(fe.get("drop", 0.08))
            O = H0 - np.array([0.0, 0.0, drop])
        th = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
        betas = self.beta(self.th0 + th)
        psis = np.linspace(-1.0, 1.0, 81)
        Htab = np.array([[horn(b, ps) for ps in psis] for b in betas])     # (48, 81, 3)
        self._Htab, self._psis, self._th_tab = Htab, psis, th

        def pin(r2, ph2, t):
            if two:
                return O + np.array([r2 * math.cos(ph2 + t), 0.0, r2 * math.sin(ph2 + t)])
            return O + np.array([dx, r2 * math.cos(ph2 + t), r2 * math.sin(ph2 + t)])

        def feather_of(x):
            r2, ph2, ell = x
            out = []
            for i, t in enumerate(th):
                g = np.linalg.norm(Htab[i] - pin(r2, ph2, t), axis=1) - ell
                k = np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:]))
                if len(k) == 0:
                    return None
                k = k[np.argmin(np.abs(psis[k]))]            # the branch near level
                out.append(psis[k] - g[k] * (psis[k + 1] - psis[k]) / (g[k + 1] - g[k]))
            return np.array(out)

        # The frozen rule (PREREG_D2 2): arm radius c_h sin(Psi); its angle so
        # that, with the rod taken as vertical, the horn's height c_h sin(psi)
        # follows the pin's and the feather is -Psi cos(phase); rod length for
        # a level mean.  The feather the model then makes is measured.
        up = float(np.sign(horn(0.0, 0.05)[2] - H0[2]))           # +psi raises the horn, or lowers it
        r2 = c_h * math.sin(Psi)
        # pin height r2 sin(ph2 + t) = up c_h sin(psi), psi = -Psi cos(t + th0 - phi)
        ph2 = self.th0 - self.phi - up * math.pi / 2.0
        ph2 = math.atan2(math.sin(ph2), math.cos(ph2))
        # the rod length that puts the mean feather at 0 (the frozen rule's
        # words), among the lengths at which the loop closes at every angle
        d0 = float(np.linalg.norm(H0 - pin(r2, ph2, -ph2)))
        mean_of = {}
        for ell_ in np.linspace(0.5 * d0, 1.5 * d0, 201):
            f_ = feather_of((r2, ph2, ell_))
            if f_ is not None:
                mean_of[float(ell_)] = float(f_.mean())
        if not mean_of:
            raise ValueError("the feather arm as frozen does not close at every crank angle at any rod length")
        ks = sorted(mean_of)
        ell = min(ks, key=lambda e_: abs(mean_of[e_]))
        for lo_, hi_ in zip(ks[:-1], ks[1:]):
            if mean_of[lo_] * mean_of[hi_] <= 0.0 and hi_ - lo_ < 0.011 * d0:
                for _ in range(40):
                    mid_ = 0.5 * (lo_ + hi_)
                    f_ = feather_of((r2, ph2, mid_))
                    if f_ is None:
                        break
                    if f_.mean() * mean_of[lo_] > 0.0:
                        lo_ = mid_
                    else:
                        hi_ = mid_
                ell = 0.5 * (lo_ + hi_)
                break
        best = type("fit", (), {})()
        best.x = np.array([r2, ph2, ell])
        f = feather_of(best.x)
        self._design_f = f
        if f is None:
            raise ValueError("the feather arm as frozen does not close at every crank angle")
        best.fun = float(np.mean((f - -Psi * np.cos(th + self.th0 - self.phi)) ** 2))
        self.arm = dict(r2=float(r2), ph2=float(ph2), ell=float(ell), dx=float(dx), c_h=c_h,
                        fit_rms_deg=float(math.degrees(math.sqrt(best.fun))),
                        psi0=float(f[0]), design_amp_deg=float(math.degrees(f.max() - f.min()) / 2.0))
        # build both sides; the rod points at the horn's built (psi 0) place and
        # `init_state` turns it to where the loop closes
        hull = next(b for b in root.iter("body") if b.get("name") == "seg0_hull")
        for side in (1, -1):
            if two:
                Os = O * np.array([1.0, side, 1.0])
                cr = ET.SubElement(hull, "body", name=f"fcrank{side:+d}", pos=" ".join(f"{v:.6g}" for v in Os))
                ET.SubElement(cr, "joint", name=f"fcrank{side:+d}_j", type="hinge", axis="0 -1 0", limited="false",
                              armature="1e-7", damping="0")
                Pl = np.array([r2 * math.cos(ph2), 0.0, r2 * math.sin(ph2)])
                rod(cr, (0, 0, 0), Pl, 0.003)
                ET.SubElement(eq, "joint", joint1=f"fcrank{side:+d}_j", joint2="crank+1_j",
                              polycoef="0 1 0 0 0", solref=ref)
            else:
                cr = next(b for b in root.iter("body") if b.get("name") == f"crank{side:+d}")
                Pl = np.array([dx, side * r2 * math.cos(ph2), r2 * math.sin(ph2)])
                rod(cr, (0, 0, 0), (dx, 0, 0), 0.003)
                rod(cr, (dx, 0, 0), Pl, 0.003)
                Os = self.wings[side][1] + np.array([0.0, side * self.O2[0], self.O2[1]])
            Hs = H0 * np.array([1.0, side, 1.0])
            e0 = Hs - (Os + Pl)
            e0 = ell * e0 / np.linalg.norm(e0)
            rb = ET.SubElement(cr, "body", name=f"featherrod{side:+d}", pos=" ".join(f"{v:.6g}" for v in Pl))
            ET.SubElement(rb, "joint", name=f"featherrod{side:+d}_j", type="ball", limited="false",
                          armature="1e-6", damping="0")
            rod(rb, (0, 0, 0), e0, 0.002)
            ET.SubElement(eq, "connect", name=f"featherrod{side:+d}", body1=f"featherrod{side:+d}",
                          body2=f"seg{1 if side > 0 else 2}_wing",
                          anchor=" ".join(f"{v:.6g}" for v in e0), solref=ref)
            self.loops.append(f"featherrod{side:+d}")
        self._rod_e0 = {side: e for side, e in ((1, None), (-1, None))}

    # ---------------------------------------------------------------- running

    def _ids(self, m):
        self.aid = m.actuator("drive_motor").id
        jin = m.joint(self.input_joint)
        self.q_in, self.v_in = int(jin.qposadr[0]), int(jin.dofadr[0])
        self.q_st, self.v_st = int(m.joint("seg1_wing_j").qposadr[0]), int(m.joint("seg1_wing_j").dofadr[0])
        self.q_fe, self.v_fe = int(m.joint("seg1_wing_jf").qposadr[0]), int(m.joint("seg1_wing_jf").dofadr[0])
        self.fid = m.actuator("feather_motor").id if (self.feather and self.feather["mode"] == "motor") else None

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
        if self.fid is not None:
            j2 = int(m.joint("seg2_wing_jf").dofadr[0])
            self.J_fe = 0.0
            for v in (self.v_fe, j2):
                e3 = np.zeros(m.nv)
                e3[v] = 1.0
                mujoco.mj_mulM(m, d, col, e3)
                self.J_fe += float(col[v])
        self.rec = dict(Pm=[], Pe=[], u=[], tau=[], om=[], gap=[], floop=[], stroke=[], th=[], slack=[],
                        feather=[], Pfm=[], Pfe=[], ufe=[])

    def init_state(self, m, d):
        self._ids(m)
        if self.kind == "DD":
            d.qvel[self.v_st] = self.bm * self.w
            d.qvel[int(m.joint("seg2_wing_j").dofadr[0])] = self.bm * self.w
        else:
            d.qvel[self.v_in] = self.w
            if self.kind in ("FB", "CT"):
                d.qvel[int(m.joint("crank-1_j").dofadr[0])] = self.w
        if self.feather and self.feather["mode"] in ("crank", "crank2"):
            import mujoco
            # the feather where the loop closes at the built crank angle, and each
            # rod turned (ball joint) from its built direction to the horn
            psi0 = self.arm["psi0"]
            for nm in ("seg1_wing_jf", "seg2_wing_jf"):
                d.qpos[m.joint(nm).qposadr[0]] = psi0
            mujoco.mj_kinematics(m, d)
            for side in (1, -1):
                b = m.body(f"featherrod{side:+d}").id
                i = m.eq(f"featherrod{side:+d}").id
                wb = int(m.eq_obj2id[i])
                Hw = d.xpos[wb] + d.xmat[wb].reshape(3, 3) @ m.eq_data[i, 3:6]
                Pw = d.xpos[b]
                e0 = m.eq_data[i, 0:3] / np.linalg.norm(m.eq_data[i, 0:3])
                Rp = d.xmat[int(m.body_parentid[b])].reshape(3, 3)
                e1 = Rp.T @ (Hw - Pw)
                e1 /= np.linalg.norm(e1)
                ax_ = np.cross(e0, e1)
                s_, c_ = np.linalg.norm(ax_), float(e0 @ e1)
                q = np.array([1.0, 0.0, 0.0, 0.0]) if s_ < 1e-12 else np.concatenate(
                    [[math.cos(0.5 * math.atan2(s_, c_))], math.sin(0.5 * math.atan2(s_, c_)) * ax_ / s_])
                d.qpos[m.joint(f"featherrod{side:+d}_j").qposadr[0]: m.joint(f"featherrod{side:+d}_j").qposadr[0] + 4] = q

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
        if self.fid is not None:
            fe, ph = self.feather, self.phase(m, d, t)
            A = math.radians(fe.get("Psi_deg", 50.0))
            q_r, qd_r, qdd_r = -A * math.cos(ph), A * self.w * math.sin(ph), A * self.w ** 2 * math.cos(ph)
            tq = self.J_fe * (qdd_r + self.wn ** 2 * (q_r - d.qpos[self.q_fe]) + 2 * self.wn * (qd_r - d.qvel[self.v_fe]))
            u = qd_r * fe["N"] / mt["omega_0"] + tq / (fe["N"] * mt["tau_s"])
            d.ctrl[self.fid] = min(max(u, -1.0), 1.0)

    def after_step(self, m, d, keep):
        import mujoco
        mt = self.motor
        tau, om = float(d.actuator_force[self.aid]), float(d.actuator_velocity[self.aid])
        Pfm = Pfe = 0.0
        if self.fid is not None:
            tf, of = float(d.actuator_force[self.fid]), float(d.actuator_velocity[self.fid])
            Pfm, Pfe = tf * of, tf * of + tf * tf * mt["omega_0"] / mt["tau_s"]
        if not keep:
            return tau * om + Pfm
        Pm = tau * om
        Pe = Pm + tau * tau * mt["omega_0"] / mt["tau_s"]
        r = self.rec
        r["feather"].append(float(d.qpos[self.q_fe]))
        r["Pfm"].append(Pfm)
        r["Pfe"].append(Pfe)
        r["ufe"].append(float(d.ctrl[self.fid]) if self.fid is not None else 0.0)
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
        return Pm + Pfm

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
        # the feather as made, and its phase against the stroke's (fundamentals
        # over the whole window, which is a whole number of planned beats)
        fz = r["feather"]
        n = len(b)
        k = np.arange(n)
        beats = max(int(round(n * self.dt * self.w / (2.0 * math.pi))), 1)
        e = np.exp(-2j * math.pi * beats * k / n)
        Zs, Zf = np.sum((b - b.mean()) * e), np.sum((fz - fz.mean()) * e)
        out.update(feather_amp_deg=float(np.degrees(fz.max() - fz.min()) / 2.0),
                   feather_mean_deg=float(np.degrees(fz.mean())),
                   feather_lead_deg=float(np.degrees(np.angle(Zf / Zs))),
                   feather_mode=(self.feather or {}).get("mode", "servo"))
        if self.fid is not None:
            out.update(feather_motor_shaft_W=float(r["Pfm"].mean()), feather_motor_elec_W=float(r["Pfe"].mean()),
                       feather_motor_at_limit=float(np.mean(np.abs(r["ufe"]) >= 0.999)))
        out["machine_elec_W"] = out["elec_W"] + float(r["Pfe"].mean())
        if self.feather and self.feather["mode"] in ("crank", "crank2"):
            out["arm"] = self.arm
        return out
