"""S0b probes: one stroke family for the four wings of T4 (`s0b_machine`).

Per wing w (L = +1 left / -1 right, Fo = +1 fore / -1 hind), phase ph (hind
pair shifted by ``hind_phase``, right wings by ``lr_phase``; both pi is the
"diagonal" stroke: FL and HR together, FR and HL half a beat later, so the
wings' reactions on the hull cancel in force, roll and pitch):

    stroke_w  = bias_w + A_w sin(ph_w)          A_w = Phi/2 amp (1 + L amp_lr/2 + Fo amp_fh/2)
    feather_w = psi0_w - Psi_w cos(ph_w)        (K1's quarter-beat lead, as m1_lib.k1_targets)
    sweep_w   = sigma_w, positive = tip toward the nose (hinge angle -L sigma_w)

and every *_lr / *_fh term is a difference: left (fore) gets +half, right
(hind) -half.  K1 is M1's take-off point (4.5 Hz, 85 deg, 50 deg).
"""
import math

import numpy as np

K1 = dict(f=4.5, Phi=math.radians(85.0), Psi=math.radians(50.0))
ZERO = dict(amp=1.0, amp_lr=0.0, amp_fh=0.0, bias=0.0, bias_lr=0.0, bias_fh=0.0, psi0=0.0, psi0_lr=0.0, psi0_fh=0.0,
            Psi_lr=0.0, Psi_fh=0.0, sw=0.0, sw_lr=0.0, sw_fh=0.0, hind_phase=math.pi, lr_phase=0.0, tail=0.0,
            Phi=K1["Phi"], Psi=K1["Psi"])
TAGS = (("FL", 1.0, 1.0), ("FR", -1.0, 1.0), ("HL", 1.0, -1.0), ("HR", -1.0, -1.0))


def targets(mc, ph, w, P, still=False):
    q, qd = np.zeros(len(mc.aid)), np.zeros(len(mc.aid))
    for tag, L, Fo in TAGS:
        p = ph + (P["hind_phase"] if Fo < 0 else 0.0) + (P["lr_phase"] if L < 0 else 0.0)
        A = 0.0 if still else P["Phi"] / 2 * P["amp"] * (1.0 + 0.5 * L * P["amp_lr"] + 0.5 * Fo * P["amp_fh"])
        Ps = 0.0 if still else P["Psi"] + 0.5 * L * P["Psi_lr"] + 0.5 * Fo * P["Psi_fh"]
        n = mc.ix[(tag, "stroke")]
        q[n] = P["bias"] + 0.5 * L * P["bias_lr"] + 0.5 * Fo * P["bias_fh"] + A * math.sin(p)
        qd[n] = A * w * math.cos(p)
        n = mc.ix[(tag, "feather")]
        q[n] = P["psi0"] + 0.5 * L * P["psi0_lr"] + 0.5 * Fo * P["psi0_fh"] - Ps * math.cos(p)
        qd[n] = Ps * w * math.sin(p)
        if (tag, "sweep") in mc.ix:
            q[mc.ix[(tag, "sweep")]] = -L * (P["sw"] + 0.5 * L * P["sw_lr"] + 0.5 * Fo * P["sw_fh"])
    q[mc.tail_ix] = P["tail"]
    return q, qd


def clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Fly4:
    """Axial thrust plus three-axis attitude for T4, nose toward ``nt`` (world).

    As M1's `Fly` (beat means of the true state, a PD on the error rotation in
    the body frame, a PI on climb rate into the common amplitude), with the
    three outputs mixed through a measured matrix: ``mix`` maps a wanted
    moment (N m, hull axes) to (bias, amp_lr, psi0_lr), the inverse of the
    held-body map's columns.  kp, kd per axis from the built inertia:
    kp = I wn^2, kd = 2 zeta I wn.  Untuned: wn and zeta are first guesses.
    """

    def __init__(self, mc, f, mix, inertia, a0, wn=6.0, zeta=0.7, H_t=None, k_h=1.0, v_c=1.5,
                 kp_vz=0.10, ki_vz=0.10, amp_lim=(0.3, 1.25), u_lim=0.3, P=None, height=True, v_axis=None):
        self.mc, self.f, self.mix = mc, f, np.asarray(mix)
        I = np.asarray(inertia)
        self.kp, self.kd = I * wn * wn, 2 * zeta * I * wn
        self.a0, self.H_t, self.k_h, self.v_c, self.kp_vz, self.ki_vz = a0, H_t, k_h, v_c, kp_vz, ki_vz
        self.amp_lim, self.u_lim, self.height = amp_lim, u_lim, height
        self.P = dict(ZERO, **(P or {}))
        self.nb = max(int(round(1.0 / (f * mc.dt) / 10)), 2)
        self.buf = []
        self.out = dict(bias=0.0, amp_lr=0.0, psi0_lr=0.0, amp=a0)
        self.integ = 0.0
        self.k = 0
        self.ph = -math.pi / 2
        self.nt = np.array([0.0, 0.0, 1.0])
        self.sat = 0
        self.n_upd = 0

    def step(self):
        """Call once per sim step, before `targets`."""
        mc = self.mc
        self.k += 1
        self.ph += 2 * math.pi * self.f * mc.dt
        if self.k % 10:
            return
        R = mc.d.xmat[mc.hull].reshape(3, 3).copy()
        self.buf.append((R, mc.d.qvel[3:6].copy(), mc.com().copy()))
        if len(self.buf) > self.nb + 1:
            self.buf.pop(0)
        if len(self.buf) < 3:
            return
        B = self.buf
        Rm = np.mean([b[0] for b in B], 0)
        U, _, Vt = np.linalg.svd(Rm)
        Rm = U @ Vt
        n = Rm[:, 0]
        w = np.mean([b[1] for b in B], 0)
        T = (len(B) - 1) * 10 * mc.dt
        v = (B[-1][2] - B[0][2]) / T
        e = Rm.T @ np.cross(n, self.nt)
        s = np.linalg.norm(e)
        ang = math.asin(clip(s, 0.0, 1.0))
        if n @ self.nt < 0:
            ang = math.pi - ang
        if s > 1e-9:
            e = e / s * ang
        Mw = self.kp * e - self.kd * w                 # no roll reference about the nose axis: e[0] = 0, damping only
        u = self.mix @ Mw
        self.n_upd += 1
        if np.any(np.abs(u) > self.u_lim):
            self.sat += 1
        tgt = dict(bias=clip(u[0], -self.u_lim, self.u_lim), amp_lr=clip(u[1], -self.u_lim, self.u_lim),
                   psi0_lr=clip(u[2], -self.u_lim, self.u_lim))
        if self.height and self.H_t is not None:
            z = float(np.mean([b[2][2] for b in B]))
            err = clip(self.k_h * (self.H_t - z), -self.v_c, self.v_c) - v[2]
            dt_s = 10 * mc.dt
            self.integ = clip(self.integ + err * dt_s, -3.0, 3.0)
            a = self.a0 / math.sqrt(max(float(n[2]), 0.5))
            tgt["amp"] = clip(a + self.kp_vz * err + self.ki_vz * self.integ, *self.amp_lim)
        else:
            tgt["amp"] = self.a0
        al = min(1.0, 10 * mc.dt / (0.5 / self.f))
        for k_, val in tgt.items():
            self.out[k_] += al * (val - self.out[k_])

    def targets(self):
        P = dict(self.P, **self.out)
        return targets(self.mc, self.ph, 2 * math.pi * self.f, P)


# ---------------------------------------------------------------------------------------------------------
# B5-B8 (DESIGN_S0b_bold.md, addendum): the level hover.  The sweep hinge is the stroke.
#
# Per wing w (L, Fo as above), phase p_w = ph + hind_phase (hind) + lr_phase (right):
#
#     sweep_w   = sw0 + Fo sw_fh/2 + A_w sin(p_w)          positive = tip toward the nose (hinge angle -L sweep)
#                 A_w = A amp (1 + L amp_lr/2 + Fo amp_fh/2)
#     stroke_w  = dih + Fo dih_fh/2 + (tilt + L tilt_lr/2) sin(p_w)     the held dihedral (+ = wing raised); tilt
#                 raises the wing at the forward end of the stroke (a stroke-plane tilt)
#     pitch_w   = the leading edge's angle above the stroke plane (hinge angle -pitch):
#         flip "flat":  psi0_w + Psi_w c            the wing passes through flat at each reversal; on the aft
#                                                    half-stroke the trailing edge leads
#         flip "vert":  psi0_w + pi/2 - (pi/2 - Psi_w) c      it passes through vertical, leading edge up, and
#                                                    the leading edge leads on both half-strokes (the insect's flip)
#         c = cos(p_w + adv), sharpened by ``flip_k`` (tanh(k c)/tanh(k)) if > 0
#     psi0_w = psi0 + L psi0_lr/2 + Fo psi0_fh/2          more incidence on the forward half-stroke than the aft
#
# On the forward half-stroke (p = 0) the wing moves toward the nose with its leading edge up by Psi.
LEVEL = dict(A=math.radians(55.0), amp=1.0, amp_lr=0.0, amp_fh=0.0, dih=math.radians(35.0), dih_fh=0.0,
             sw0=0.0, sw_lr=0.0, sw_fh=0.0, Psi=math.radians(45.0), Psi_lr=0.0, Psi_fh=0.0, psi0=0.0, psi0_lr=0.0,
             psi0_fh=0.0, tilt=0.0, tilt_lr=0.0, adv=0.0, flip="vert", flip_k=0.0, hind_phase=0.0, lr_phase=0.0,
             tail=0.0)


def targets_level(mc, ph, w, P, osc=1.0):
    """Targets and rates for the level hover.  ``osc`` in [0, 1] scales every oscillating term (the ramp):
    at 0 the wings are held raised at the dihedral and mean sweep, flat (pitch 0) in both flip modes."""
    q, qd = np.zeros(len(mc.aid)), np.zeros(len(mc.aid))
    k = P["flip_k"]
    for tag, L, Fo in TAGS:
        p = ph + (P["hind_phase"] if Fo < 0 else 0.0) + (P["lr_phase"] if L < 0 else 0.0)
        A = osc * P["A"] * P["amp"] * (1.0 + 0.5 * L * P["amp_lr"] + 0.5 * Fo * P["amp_fh"])
        sp, cp = math.sin(p), math.cos(p)
        n = mc.ix[(tag, "sweep")]
        q[n] = -L * (P["sw0"] + 0.5 * L * P["sw_lr"] + 0.5 * Fo * P["sw_fh"] + A * sp)
        qd[n] = -L * A * w * cp
        tl = osc * (P["tilt"] + 0.5 * L * P["tilt_lr"])
        n = mc.ix[(tag, "stroke")]
        q[n] = P["dih"] + 0.5 * Fo * P["dih_fh"] + tl * sp
        qd[n] = tl * w * cp
        c, cdot = math.cos(p + P["adv"]), -w * math.sin(p + P["adv"])
        if k > 0.0:
            th = math.tanh(k * c)
            c, cdot = th / math.tanh(k), k * (1.0 - th * th) / math.tanh(k) * cdot
        Ps = P["Psi"] + 0.5 * L * P["Psi_lr"] + 0.5 * Fo * P["Psi_fh"]
        p0 = osc * (P["psi0"] + 0.5 * L * P["psi0_lr"] + 0.5 * Fo * P["psi0_fh"])
        if P["flip"] == "vert":
            g = 0.5 * math.pi - Ps
            th_, thd = p0 + osc * (0.5 * math.pi - g * c), -osc * g * cdot
        else:
            th_, thd = p0 + osc * Ps * c, osc * Ps * cdot
        n = mc.ix[(tag, "feather")]
        q[n], qd[n] = -th_, -thd
    q[mc.tail_ix] = P["tail"]
    return q, qd


def _rotvec(R):
    """Rotation vector of R (so3 log)."""
    c = clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    a = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.linalg.norm(v)
    return np.zeros(3) if s < 1e-12 else v / s * a


class Hover4:
    """A plain attitude + height loop for the level hover (B7; hand-set gains).

    Beat means of the true state (as `Fly4`).  Attitude: PD on the rotation vector from the hull's mean
    attitude to "level at the starting heading, tilted by the velocity loop", in the hull frame, mixed into
    three stroke inputs by ``mix`` (3x3: wanted moment (Mx roll, My pitch, Mz yaw), N m -> the inputs named in
    ``chan``).  Velocity loop: tilt = -k_v x horizontal velocity (- k_p x position error if k_p), capped at
    ``tilt_max``.  Height: PI on climb rate into the common amplitude, climb-rate command
    k_h (H_t - z) capped at +-v_c.  ``H_t`` may be changed while it runs (the descent)."""

    def __init__(self, mc, f, mix, chan, inertia, a0, P=None, wn=(8.0, 8.0, 4.0), zeta=0.8, H_t=2.0, k_h=1.0,
                 v_c=0.8, kp_vz=0.10, ki_vz=0.15, amp_lim=(0.3, 1.3), u_lim=(0.4, 0.4, 0.4), k_v=0.15,
                 k_p=0.0, tilt_max=0.25, v_dn=None):
        self.mc, self.f, self.mix, self.chan = mc, f, np.asarray(mix, float), tuple(chan)
        I, wn = np.asarray(inertia, float), np.asarray(wn, float)
        self.kp, self.kd = I * wn * wn, 2 * zeta * I * wn
        self.a0, self.H_t, self.k_h, self.v_c, self.kp_vz, self.ki_vz = a0, H_t, k_h, v_c, kp_vz, ki_vz
        self.v_dn = v_c if v_dn is None else v_dn
        self.amp_lim, self.u_lim, self.k_v, self.k_p, self.tilt_max = amp_lim, np.asarray(u_lim, float), k_v, k_p, tilt_max
        self.P = dict(LEVEL, **(P or {}))
        self.nb = max(int(round(1.0 / (f * mc.dt) / 10)), 2)
        self.buf = []
        self.out = {c: 0.0 for c in self.chan}
        self.out["amp"] = a0
        self.base = {c: self.P[c] for c in self.chan}
        self.integ, self.k, self.ph, self.sat, self.n_upd = 0.0, 0, 0.0, 0, 0
        R = mc.d.xmat[mc.hull].reshape(3, 3)
        self.yaw0 = math.atan2(R[1, 0], R[0, 0])
        self.p0 = mc.com()[:2].copy()
        self.last = {}

    def step(self):
        mc = self.mc
        self.k += 1
        self.ph += 2 * math.pi * self.f * mc.dt
        if self.k % 10:
            return
        R = mc.d.xmat[mc.hull].reshape(3, 3).copy()
        self.buf.append((R, mc.d.qvel[3:6].copy(), mc.com().copy()))
        if len(self.buf) > self.nb + 1:
            self.buf.pop(0)
        if len(self.buf) < 3:
            return
        B = self.buf
        U, _, Vt = np.linalg.svd(np.mean([b[0] for b in B], 0))
        Rm = U @ Vt
        w = np.mean([b[1] for b in B], 0)
        T = (len(B) - 1) * 10 * mc.dt
        v = (B[-1][2] - B[0][2]) / T
        pos = np.mean([b[2] for b in B], 0)
        cy, sy = math.cos(self.yaw0), math.sin(self.yaw0)
        vf, vl = cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1]            # forward, left
        ef = cy * (pos[0] - self.p0[0]) + sy * (pos[1] - self.p0[1])
        el = -sy * (pos[0] - self.p0[0]) + cy * (pos[1] - self.p0[1])
        tm = self.tilt_max
        pitch_dn = clip(-self.k_v * vf - self.k_p * ef, -tm, tm)           # nose down (+) pushes forward
        roll_l = clip(-self.k_v * vl - self.k_p * el, -tm, tm)             # left side down (+) pushes left
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        cp, sp = math.cos(pitch_dn), math.sin(pitch_dn)
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])     # +rotation about y: nose down
        cr, sr = math.cos(-roll_l), math.sin(-roll_l)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])     # -rotation about x: left side down
        Rt = Rz @ Ry @ Rx
        e = _rotvec(Rm.T @ Rt)
        Mw = self.kp * e - self.kd * w
        u = self.mix @ Mw
        self.n_upd += 1
        if np.any(np.abs(u) > self.u_lim):
            self.sat += 1
        tgt = {c: clip(float(u[i]), -self.u_lim[i], self.u_lim[i]) for i, c in enumerate(self.chan)}
        z = float(pos[2])
        err = clip(self.k_h * (self.H_t - z), -self.v_dn, self.v_c) - v[2]
        dt_s = 10 * mc.dt
        self.integ = clip(self.integ + err * dt_s, -3.0, 3.0)
        a = self.a0 / math.sqrt(max(float(Rm[2, 2]), 0.5))
        tgt["amp"] = clip(a + self.kp_vz * err + self.ki_vz * self.integ, *self.amp_lim)
        al = min(1.0, 10 * mc.dt / (0.5 / self.f))
        for k_, val in tgt.items():
            self.out[k_] += al * (val - self.out[k_])
        self.last = dict(e=e, w=w, v=v, z=z, tilt=(pitch_dn, roll_l), up=float(Rm[2, 2]))

    def targets(self, osc=1.0):
        P = dict(self.P)
        for c in self.chan:
            P[c] = self.base[c] + self.out[c]
        P["amp"] = self.out["amp"]
        return targets_level(self.mc, self.ph, 2 * math.pi * self.f, P, osc=osc)
