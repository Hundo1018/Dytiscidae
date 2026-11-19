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
