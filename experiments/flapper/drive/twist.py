"""What makes the wing's spanwise twist at cruise (pass D3,
`PREREG_D3_tip_twist.md`).

The wing is W7's chain of panels on span hinges (`body.py wing_panels`), the
stroke is D1's four-bar and witness motor (`drive.Drive("FB", ...)`), and the
twist is made by one of:

* "servo"  : freerig's ideal servo on every hinge (T0, the reference);
* "spring" : a torsion spring at every hinge, nothing else (TP);
* "crank"  : a tube fixed to the tip panel running in along the hinge line to
             the shoulder, a horn on its inboard end, and a ball-ended rod
             from a second arm on the stroke's crankshaft to the horn (TX);
             the hinges are springs;
* "cable"  : a one-sided cable (a spatial tendon with a length limit) from
             the hull, under the shoulder, to a post on the tip panel's
             trailing edge (TC); the hinges are springs.

The springs themselves are freerig's (`chain=dict(mode="passive", k=...)`);
their rest twist (`rest_deg`, the tip against the root, shared equally) is
set here.  `drive.py` is not edited: `Drive.compile` names the second wing
"seg2_wing", which on a chain wing is the first wing's second panel, so the
four-bar is built again here for a chain (same geometry, same names).

No force is typed: the rod, the cable and the springs are bodies,
constraints and joint stiffness of the model.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

from drive import Drive


class TwistDrive(Drive):
    """D1's four-bar on a chain wing, with a twist mechanism.

    twist: dict(mode=..., ...).  Keys by mode:
      servo, spring: rest_deg (spring only);
      crank: c_h (horn length, m), A_deg (the tip twist it is sized for),
             rest_deg; optional r2, ph2, ell to skip the fit;
      cable: post (m), A_deg, rest_deg; optional h, L0 to skip the solve.
    psi_root_deg: the root feather the mechanism is sized at."""

    def __init__(self, Phi_deg, f, N, *, twist=None, psi_root_deg=-2.94, **kw):
        super().__init__("FB", Phi_deg, f, N, **kw)
        self.twist = dict(mode="servo")
        self.twist.update(twist or {})
        if self.twist["mode"] not in ("servo", "spring", "crank", "cable"):
            raise ValueError(self.twist["mode"])
        self.psi_root = math.radians(psi_root_deg)
        self.design = {}

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
        # the two wings' root panels and their chains, by the side they are on
        wings, self.lever_local, self.chain, shift_of = {}, {}, {}, {}
        for bid in range(plain.nbody):
            nm = plain.body(bid).name
            if int(plain.body_parentid[bid]) != hb or not nm.endswith("_wing"):
                continue
            side = 1 if float((Rh.T @ (pdat.xpos[bid] - pdat.xpos[hb]))[1]) > 0.0 else -1
            ch, b = [bid], bid
            while True:
                kids = [c for c in range(plain.nbody) if int(plain.body_parentid[c]) == b
                        and plain.body(c).name.endswith("_wing")]
                if not kids:
                    break
                assert len(kids) == 1
                b = kids[0]
                ch.append(b)
            self.chain[side] = [plain.body(c).name for c in ch]
            for c in ch:
                sel = pan.body_id == c
                cdir = pan.chord_local[sel][0] / np.linalg.norm(pan.chord_local[sel][0])
                if float(pan.pos_local[sel].mean(0) @ cdir) < 0.0:
                    cdir = -cdir
                shift_of[plain.body(c).name] = (self.feather_axis * float(pan.chord[sel].mean()) * cdir, cdir,
                                                float(pan.chord[sel].mean()))
            shift = shift_of[nm][0]
            Rw = Rh.T @ pdat.xmat[bid].reshape(3, 3)
            wings[side] = (nm, plain.body_pos[bid].copy() + Rw @ shift)
            self.lever_local[side] = shift + np.array([self.a, 0.0, 0.0])
        assert sorted(wings) == [-1, 1], "two wings on the hull"
        assert wings[1][0] == "seg1_wing", "Drive reads the stroke from seg1_wing_j"
        self.wings, self.shift_of = wings, shift_of
        self.n_h = len(self.chain[1]) - 1
        if self.n_h < 1 and self.twist["mode"] != "servo":
            raise ValueError("the twist mechanisms need a chain wing (wing_panels > 1)")
        ref = f"{2.0 * self.dt} 1"
        eq = root.find("equality")
        if eq is None:
            eq = ET.SubElement(root, "equality")
        act = root.find("actuator")
        mass = lambda length: f"{max(self.lam * length, 1e-6)}"
        self.loops = []

        def at(side, uv):
            S = wings[side][1]
            return np.array([S[0], S[1] + side * uv[0], S[2] + uv[1]])

        def joint(body, name, side):
            ET.SubElement(body, "joint", name=name, type="hinge", axis=f"{side} 0 0",
                          limited="false", armature="1e-7", damping="0")

        def rod(body, p0, p1, size=0.002, rgba="0.85 0.35 0.1 1"):
            ET.SubElement(body, "geom", type="capsule", fromto=" ".join(f"{v:.6g}" for v in (*p0, *p1)),
                          size=f"{size}", mass=mass(float(np.linalg.norm(np.subtract(p1, p0)))),
                          contype="0", conaffinity="0", rgba=rgba)

        # D1's four-bar, as `Drive.compile` builds it
        Q0 = self.O2 + self.r * np.array([math.cos(self.th0), math.sin(self.th0)])
        P0 = np.array([self.a, 0.0])
        for side in (1, -1):
            O, Qh, Ph = at(side, self.O2), at(side, Q0), at(side, P0)
            cr = ET.SubElement(hull, "body", name=f"crank{side:+d}", pos=" ".join(f"{v:.6g}" for v in O))
            joint(cr, f"crank{side:+d}_j", side)
            rod(cr, (0, 0, 0), Qh - O, 0.003)
            cp = ET.SubElement(cr, "body", name=f"coupler{side:+d}", pos=" ".join(f"{v:.6g}" for v in Qh - O))
            joint(cp, f"coupler{side:+d}_j", side)
            rod(cp, (0, 0, 0), Ph - Qh)
            e = Ph - Qh
            ET.SubElement(eq, "connect", name=f"coupler{side:+d}", body1=f"coupler{side:+d}",
                          body2=wings[side][0], anchor=" ".join(f"{v:.6g}" for v in e), solref=ref)
            self.loops.append(f"coupler{side:+d}")
        ET.SubElement(eq, "joint", joint1="crank-1_j", joint2="crank+1_j", polycoef="0 1 0 0 0", solref=ref)
        self.input_joint = "crank+1_j"

        # the chain's kinematics on the plain model, hinges where freerig puts them
        self._plain = plain
        for side in (1, -1):
            nm0 = self.chain[side][0]
            plain.jnt_pos[plain.joint(nm0 + "_jf").id] += shift_of[nm0][0]
            for nm in self.chain[side][1:]:
                plain.jnt_pos[plain.joint(nm + "_j").id] += shift_of[nm][0]
        self._kd = mujoco.MjData(plain)
        self._Rh, self._hb = Rh, hb

        mode = self.twist["mode"]
        if mode == "crank":
            self._build_crank(root, eq, hull, ref, rod, mass)
        if mode == "cable":
            self._build_cable(root, hull, ref, rod)
        mt = self.motor
        ET.SubElement(act, "general", name="drive_motor", joint=self.input_joint, gear=f"{self.N}",
                      ctrllimited="true", ctrlrange="-1 1", gainprm=f"{mt['tau_s']}",
                      biastype="affine", biasprm=f"0 0 {-mt['tau_s'] / mt['omega_0']}")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        jin = model.joint(self.input_joint)
        model.dof_armature[jin.dofadr[0]] += mt["J_r"] * self.N ** 2
        if mode == "crank":
            for side in (1, -1):        # the horn, not where the rod's end was at the built pose
                model.eq_data[model.eq(f"twistrod{side:+d}").id, 3:6] = self.horn_local[side]
        data = mujoco.MjData(model)
        return model, data, names, build_panels(p, model)

    def tip_point(self, side, beta, psi_r, psi_t, local, model=None, data=None):
        """Hull-frame place of a point of the tip panel (``local``, its frame)
        at stroke ``beta``, root feather ``psi_r`` and tip twist ``psi_t``
        (shared equally between the hinges), on the kinematics alone."""
        import mujoco
        m, d = (self._plain, self._kd) if model is None else (model, data)
        ch = self.chain[side]
        if model is None:
            d.qpos[:] = m.qpos0
        d.qpos[m.joint(ch[0] + "_j").qposadr[0]] = beta
        d.qpos[m.joint(ch[0] + "_jf").qposadr[0]] = psi_r
        for nm in ch[1:]:
            d.qpos[m.joint(nm + "_j").qposadr[0]] = psi_t / self.n_h
        mujoco.mj_kinematics(m, d)
        b, hb = m.body(ch[-1]).id, m.body("seg0_hull").id
        w = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.asarray(local, float)
        return d.xmat[hb].reshape(3, 3).T @ (w - d.xpos[hb])

    # ---- TX: tube, horn, rod, and a second arm on the stroke's crankshaft

    def _build_crank(self, root, eq, hull, ref, rod, mass):
        tw = self.twist
        c_h, A = float(tw.get("c_h", 0.03)), math.radians(tw.get("A_deg", 15.0))
        span_in = {}
        self.horn_local, self.tube_local = {}, {}
        for side in (1, -1):
            tip = self.chain[side][-1]
            shift, cdir, _ = self.shift_of[tip]
            # the tip panel's hinge is at its own origin (+ shift); the shoulder
            # is n_h panel lengths inboard along the panel's span axis (local -x)
            b = self._plain.body(tip).id
            seg = float(np.linalg.norm(self._plain.body_pos[b]))          # one panel's length
            span_in[side] = seg * self.n_h
            inboard = shift + np.array([-span_in[side], 0.0, 0.0])
            self.tube_local[side] = (shift.copy(), inboard)
            self.horn_local[side] = inboard + c_h * cdir
        th = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
        betas = self.beta(self.th0 + th)
        psis = np.linspace(-0.9, 0.9, 181)
        Htab = np.array([[self.tip_point(1, b_, self.psi_root, ps, self.horn_local[1]) for ps in psis]
                         for b_ in betas])
        O = self.wings[1][1] + np.array([0.0, self.O2[0], self.O2[1]])
        H0 = self.tip_point(1, 0.0, 0.0, 0.0, self.horn_local[1])
        dx = float(H0[0] - O[0])
        target = -A * np.cos(th + self.th0 - self.phi)

        def pin(r2, ph2, t):
            return O + np.array([dx, r2 * math.cos(ph2 + t), r2 * math.sin(ph2 + t)])

        def twist_of(x):
            r2, ph2, ell = x
            out = []
            for i, t in enumerate(th):
                g = np.linalg.norm(Htab[i] - pin(r2, ph2, t), axis=1) - ell
                k = np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:]))
                if len(k) == 0:
                    return None
                k = k[np.argmin(np.abs(psis[k] - target[i]))]
                out.append(psis[k] - g[k] * (psis[k + 1] - psis[k]) / (g[k + 1] - g[k]))
            return np.array(out)

        if all(k_ in tw for k_ in ("r2", "ph2", "ell")):
            x = np.array([tw["r2"], tw["ph2"], tw["ell"]], float)
        else:
            from scipy.optimize import least_squares
            # first guess: arm radius c_h sin(A), a quarter turn from the stroke
            # crank so the pin's height follows -cos(phase); rod to the rest gap
            up = float(np.sign(self.tip_point(1, 0.0, 0.0, 0.05, self.horn_local[1])[2] - H0[2]))
            r2 = c_h * math.sin(A)
            ph2 = self.th0 - self.phi - up * math.pi / 2.0
            ell = float(np.linalg.norm(H0 - O - np.array([dx, 0.0, 0.0])))

            def res(x):
                f_ = twist_of(x)
                return np.full(len(th), 1.0) if f_ is None else f_ - target
            x = least_squares(res, [r2, ph2, ell], x_scale=[1e-3, 0.1, 1e-3]).x
        f_ = twist_of(x)
        if f_ is None:
            raise ValueError("the twist arm does not close at every crank angle")
        r2, ph2, ell = (float(v) for v in x)
        ph2 = math.atan2(math.sin(ph2), math.cos(ph2))
        c1, s1 = 2.0 * np.mean(f_ * np.cos(th)), 2.0 * np.mean(f_ * np.sin(th))
        self.design = dict(r2=r2, ph2=ph2, ell=ell, dx=dx, c_h=c_h, tube=span_in[1],
                           fit_rms_deg=float(np.degrees(np.sqrt(np.mean((f_ - target) ** 2)))),
                           amp_deg=float(np.degrees(f_.max() - f_.min()) / 2.0),
                           mean_deg=float(np.degrees(f_.mean())),
                           fund_deg=float(np.degrees(math.hypot(c1, s1))),
                           psi0=float(f_[0]))
        self._design_twist = f_
        for side in (1, -1):
            tip = self.chain[side][-1]
            tb = next(b for b in root.iter("body") if b.get("name") == tip)
            p0, p1 = self.tube_local[side]
            rod(tb, p0, p1, 0.003, rgba="0.2 0.5 0.2 1")                 # the tube
            rod(tb, p1, self.horn_local[side], 0.003, rgba="0.2 0.5 0.2 1")   # its horn
            cr = next(b for b in root.iter("body") if b.get("name") == f"crank{side:+d}")
            Pl = np.array([dx, side * r2 * math.cos(ph2), r2 * math.sin(ph2)])
            rod(cr, (0, 0, 0), (dx, 0, 0), 0.003)
            rod(cr, (dx, 0, 0), Pl, 0.003)
            Os = self.wings[side][1] + np.array([0.0, side * self.O2[0], self.O2[1]])
            Hs = H0 * np.array([1.0, side, 1.0])
            e0 = Hs - (Os + Pl)
            e0 = ell * e0 / np.linalg.norm(e0)
            rb = ET.SubElement(cr, "body", name=f"twistrod{side:+d}", pos=" ".join(f"{v:.6g}" for v in Pl))
            ET.SubElement(rb, "joint", name=f"twistrod{side:+d}_j", type="ball", limited="false",
                          armature="1e-6", damping="0")
            rod(rb, (0, 0, 0), e0, 0.002)
            ET.SubElement(eq, "connect", name=f"twistrod{side:+d}", body1=f"twistrod{side:+d}", body2=tip,
                          anchor=" ".join(f"{v:.6g}" for v in e0), solref=ref)
            self.loops.append(f"twistrod{side:+d}")

    # ---- TC: a cable from the hull to a post on the tip panel's trailing edge

    def _build_cable(self, root, hull, ref, rod):
        tw = self.twist
        post, A = float(tw.get("post", 0.05)), math.radians(tw.get("A_deg", 15.0))
        self.post_local, self.te_local, self.anchor = {}, {}, {}
        for side in (1, -1):
            tip = self.chain[side][-1]
            shift, cdir, chord = self.shift_of[tip]
            b = self._plain.body(tip).id
            seg = float(np.linalg.norm(self._plain.body_pos[b]))
            # the trailing edge at the tip panel's middle; the fluid's leading
            # edge is at the built hinge (the strips sit a quarter chord behind it)
            te = np.array([0.5 * seg, 0.0, 0.0]) + chord * cdir
            self.te_local[side] = te
            # "up" (hull +z at the built pose) in the tip panel's frame
            p0 = self.tip_point(side, 0.0, 0.0, 0.0, te)
            ez = np.array([self.tip_point(side, 0.0, 0.0, 0.0, te + e_)[2] - p0[2] for e_ in np.eye(3)])
            self.post_local[side] = te + post * ez / np.linalg.norm(ez)

        def dist(h, beta, psi):
            T = self.tip_point(1, beta, self.psi_root, psi, self.post_local[1])
            T0 = self.tip_point(1, 0.0, 0.0, 0.0, self.post_local[1])
            S = self.wings[1][1]
            Hh = np.array([T0[0], S[1], S[2] - h])
            return float(np.linalg.norm(T - Hh))

        def psi_max(h, L0, beta):
            """The tip twist at which the cable goes taut (the twist's stop),
            searched upward from -0.6 rad: the first twist at which the
            distance reaches the cable's length."""
            grid = np.linspace(-0.6, 1.2, 91)
            g = np.array([dist(h, beta, ps) - L0 for ps in grid])
            if g[0] >= 0.0:
                return float(grid[0])
            k = np.flatnonzero(g >= 0.0)
            if len(k) == 0:
                return float(grid[-1])
            lo, hi = float(grid[k[0] - 1]), float(grid[k[0]])
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if dist(h, beta, mid) < L0:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        if "h" in tw and "L0" in tw:
            h, L0 = float(tw["h"]), float(tw["L0"])
        else:
            # the anchor's drop under the stroke axis and the cable's length so
            # that the taut cable stops the tip at 0 twist at the top of the
            # stroke and at +A at mid stroke
            lo, hi = 1e-4, post
            for _ in range(50):
                h = 0.5 * (lo + hi)
                L0 = dist(h, self.bm, 0.0)
                if psi_max(h, L0, 0.0) > A:
                    hi = h                          # too much gearing: raise the anchor
                else:
                    lo = h
            h = 0.5 * (lo + hi)
            L0 = dist(h, self.bm, 0.0)
        bs = np.linspace(-self.bm, self.bm, 9)
        self.design = dict(h=h, L0=L0, post=post,
                           psi_max_deg=[float(np.degrees(psi_max(h, L0, b_))) for b_ in bs],
                           at_beta_deg=[float(np.degrees(b_)) for b_ in bs],
                           grows_with_twist=bool(dist(h, 0.0, 0.3) > dist(h, 0.0, 0.0)))
        ten = root.find("tendon")
        if ten is None:
            ten = ET.SubElement(root, "tendon")
        for side in (1, -1):
            tip = self.chain[side][-1]
            tb = next(b for b in root.iter("body") if b.get("name") == tip)
            rod(tb, self.te_local[side], self.post_local[side], 0.002, rgba="0.2 0.5 0.2 1")
            ET.SubElement(tb, "site", name=f"tc_tip{side:+d}", pos=" ".join(f"{v:.6g}" for v in self.post_local[side]),
                          size="0.003")
            T0 = self.tip_point(side, 0.0, 0.0, 0.0, self.post_local[side])
            S = self.wings[side][1]
            Hh = np.array([T0[0], S[1], S[2] - h])
            self.anchor[side] = Hh
            ET.SubElement(hull, "site", name=f"tc_hull{side:+d}", pos=" ".join(f"{v:.6g}" for v in Hh), size="0.003")
            sp = ET.SubElement(ten, "spatial", name=f"twistcable{side:+d}", limited="true",
                               range=f"0 {L0}", solreflimit=ref, width="0.001")
            ET.SubElement(sp, "site", site=f"tc_hull{side:+d}")
            ET.SubElement(sp, "site", site=f"tc_tip{side:+d}")

    # ---------------------------------------------------------------- running

    def _twist_ids(self, m):
        self.tw_q = {s: [int(m.joint(nm + "_j").qposadr[0]) for nm in self.chain[s][1:]] for s in (1, -1)}
        self.tw_a = [m.actuator(nm + "_a").id for s in (1, -1) for nm in self.chain[s][1:]]

    def init_state(self, m, d):
        import mujoco
        super().init_state(m, d)
        self._twist_ids(m)
        mode = self.twist["mode"]
        rest = math.radians(self.twist.get("rest_deg", 0.0))
        if mode in ("spring", "cable", "crank") and self.n_h:
            for s in (1, -1):
                d.qpos[self.tw_q[s]] = rest / self.n_h
        if mode != "crank":
            return
        # mj_setConst (freerig calls it) puts a connect's body2 anchor back
        # where the rod's end was built: write the horn again
        for side in (1, -1):
            m.eq_data[m.eq(f"twistrod{side:+d}").id, 3:6] = self.horn_local[side]
        for side in (1, -1):
            i = m.eq(f"twistrod{side:+d}").id
            b = m.body(f"twistrod{side:+d}").id
            wb = int(m.eq_obj2id[i])
            ell = float(np.linalg.norm(m.eq_data[i, 0:3]))

            def gap(psi):
                d.qpos[self.tw_q[side]] = psi / self.n_h
                mujoco.mj_kinematics(m, d)
                Hw = d.xpos[wb] + d.xmat[wb].reshape(3, 3) @ m.eq_data[i, 3:6]
                return float(np.linalg.norm(Hw - d.xpos[b])) - ell, Hw
            # the tip twist that closes the rod at the built crank angle, near the design's
            p0 = self.design["psi0"]
            lo, hi = p0 - 0.2, p0 + 0.2
            glo, ghi = gap(lo)[0], gap(hi)[0]
            if glo * ghi > 0.0:
                raise ValueError("the twist rod does not close at the start")
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                gm = gap(mid)[0]
                if gm * glo > 0.0:
                    lo, glo = mid, gm
                else:
                    hi = mid
            _, Hw = gap(0.5 * (lo + hi))
            Pw = d.xpos[b]
            e0 = m.eq_data[i, 0:3] / ell
            Rp = d.xmat[int(m.body_parentid[b])].reshape(3, 3)
            e1 = Rp.T @ (Hw - Pw)
            e1 /= np.linalg.norm(e1)
            ax_ = np.cross(e0, e1)
            s_, c_ = np.linalg.norm(ax_), float(e0 @ e1)
            q = np.array([1.0, 0.0, 0.0, 0.0]) if s_ < 1e-12 else np.concatenate(
                [[math.cos(0.5 * math.atan2(s_, c_))], math.sin(0.5 * math.atan2(s_, c_)) * ax_ / s_])
            a0 = int(m.joint(f"twistrod{side:+d}_j").qposadr[0])
            d.qpos[a0: a0 + 4] = q

    def setup(self, m, d):
        super().setup(m, d)
        self._twist_ids(m)
        rest = math.radians(self.twist.get("rest_deg", 0.0))
        if self.twist["mode"] != "servo" and rest and self.n_h:
            # freerig's hinge springs are position actuators, force = k (ctrl - q)
            # - c qdot, with ctrl 0: the rest angle goes in the bias's constant
            for aid in self.tw_a:
                m.actuator_biasprm[aid, 0] = m.actuator_gainprm[aid, 0] * rest / self.n_h
        self.trec = dict(tip=[], hinge=[], tq=[], rod_gap=[], rod_f=[], cp_gap=[], cp_f=[], ten=[], taut=[],
                         length=[])
        self._eq_rod = [m.eq(f"twistrod{s:+d}").id for s in (1, -1)] if self.twist["mode"] == "crank" else []
        self._eq_cp = [m.eq(f"coupler{s:+d}").id for s in (1, -1)]
        self._ten = [m.tendon(f"twistcable{s:+d}").id for s in (1, -1)] if self.twist["mode"] == "cable" else []

    def after_step(self, m, d, keep):
        import mujoco
        P = super().after_step(m, d, keep)
        if not keep:
            return P
        r = self.trec
        q = d.qpos[self.tw_q[1]] if self.n_h else np.zeros(0)
        r["tip"].append(float(q.sum()))
        r["hinge"].append(np.array(q))
        r["tq"].append(float(np.abs(d.actuator_force[self.tw_a]).max()) if self.tw_a else 0.0)
        n = d.nefc
        ty, ids, pos, frc = d.efc_type[:n], d.efc_id[:n], d.efc_pos[:n], d.efc_force[:n]
        eqr = ty == mujoco.mjtConstraint.mjCNSTR_EQUALITY

        def loop(eq_ids):
            g, f_ = 0.0, 0.0
            for i in eq_ids:
                sel = eqr & (ids == i)
                if sel.any():
                    g, f_ = max(g, float(np.linalg.norm(pos[sel]))), max(f_, float(np.linalg.norm(frc[sel])))
            return g, f_
        g, f_ = loop(self._eq_rod)
        r["rod_gap"].append(g)
        r["rod_f"].append(f_)
        g, f_ = loop(self._eq_cp)
        r["cp_gap"].append(g)
        r["cp_f"].append(f_)
        if self._ten:
            lim = ty == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON
            t1 = lim & (ids == self._ten[0])
            r["ten"].append(float(np.abs(frc[lim]).max()) if lim.any() else 0.0)
            r["taut"].append(1.0 if (t1.any() and float(np.abs(frc[t1]).max()) > 1e-9) else 0.0)
            r["length"].append(float(d.ten_length[self._ten[0]]))
        return P

    def summary(self):
        out = super().summary()
        r = {k: np.asarray(v) for k, v in self.trec.items()}
        b, tip = np.asarray(self.rec["stroke"]), r["tip"]
        n = len(b)
        beats = max(int(round(n * self.dt * self.w / (2.0 * math.pi))), 1)
        e = np.exp(-2j * math.pi * beats * np.arange(n) / n)
        Zs, Zt = np.sum((b - b.mean()) * e), np.sum((tip - tip.mean()) * e)
        hg = r["hinge"]
        tw = dict(mode=self.twist["mode"], spec=self.twist, design=self.design,
                  tip_amp_deg=float(np.degrees(tip.max() - tip.min()) / 2.0),
                  tip_fund_deg=float(np.degrees(2.0 * abs(Zt) / n)),
                  tip_mean_deg=float(np.degrees(tip.mean())),
                  tip_min_deg=float(np.degrees(tip.min())), tip_max_deg=float(np.degrees(tip.max())),
                  # the tip twist's fundamental against the stroke's; -90 is the
                  # reference's phase (-A cos against sin), as D2's feather lead
                  tip_lead_deg=float(np.degrees(np.angle(Zt / Zs))) if abs(Zt) > 1e-12 else float("nan"),
                  hinge_swing_deg=[float(x) for x in np.degrees(hg.max(0) - hg.min(0)) / 2.0] if hg.size else [],
                  hinge_mean_deg=[float(x) for x in np.degrees(hg.mean(0))] if hg.size else [],
                  hinge_torque_peak=float(r["tq"].max()),
                  coupler_gap_max_mm=float(1e3 * r["cp_gap"].max()), coupler_force_peak_N=float(r["cp_f"].max()))
        if self._eq_rod:
            tw.update(rod_gap_max_mm=float(1e3 * r["rod_gap"].max()), rod_force_peak_N=float(r["rod_f"].max()))
        if self._ten:
            tw.update(cable_force_peak_N=float(r["ten"].max()), cable_taut_share=float(r["taut"].mean()),
                      cable_length_max=float(r["length"].max()))
        out["twist"] = tw
        return out
