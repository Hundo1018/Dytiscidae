"""Pass M4: shared pieces (PREREG_M4_cruise_entries.md).

The machine of C1 on `mission.Machine` with all six axes free, as
`m4_probe.py` builds it (copied from it, not imported: that file is the
lead's): BODY_S, W6 wing, the 0.06 m2 tail, hull transverse inertia 0.012
kg m2 (pass 1m's, as C1), every hinge on its witness motor, the tip's feather
by the strips turned per side (`Machine.enable_strip_twist`: D3's stand-in
for its tube and rod, not a mechanism).

Pitch, height and speed: C1's tail PI and pilot, as `m4_probe.fly` writes
them.  Lateral inputs (all differential, right minus left, by the harness's
sides: seg1 and seg4 are the +y side, the harness's "right" in `set_twist`):

    amp     stroke amplitude, deg (Phi + amp, Phi - amp)
    bias    stroke offset, deg (+bias, -bias; + is up on both wings)
    twist   tip feather offset, deg (+twist, -twist)
    twamp   tip feather amplitude, deg (Psi + twamp, Psi - twamp)
    feather root feather offset, deg (psi0 + feather, psi0 - feather; + is LE down)
    tail    tail halves, deg (+tail, -tail)

and one common input, `dihedral` (both strokes' offset, deg, + up).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
I0 = 0.012
TL = dict(i0_deg=0.0, kp=1.0, kd=0.1, ki=2.0, limit_deg=25.0, rate=8.0, lag_beats=1.0)
AP = dict(kh=6.0, kvz=6.0, pitch_lim_deg=12.0, kf=0.08, f_lim=0.2)
#: the lateral law L, frozen in PREREG_M4 section 2 (the gains of probe 3, m4_smoke.LAW, untuned)
LAW_L = dict(kphi=0.1, kp=0.01, ff=-0.09, kpsi=0.5, phimax=15.0, fmax=6.0, kbeta=0.4, kr=0.15, ymax=8.0)
LAT_KEYS = ("amp", "bias", "twist", "twamp", "feather", "tail", "dihedral")


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def build(V=8.0, x0=-40.0, y0=0.0, z0=30.0, yaw0=0.0, roll0=0.0, pitch0=0.0, seed=0):
    """The machine launched level-ish at V along its own heading yaw0 (deg)."""
    import c1_pilot
    from mission import Machine
    mc = Machine("air", body=c1_pilot.body(), pose=dict(x=x0, y=y0, z=z0, pitch_deg=pitch0), seed=seed)
    m, d, mj = mc.m, mc.d, mc.mj
    h = mc.hull
    long_axis = int(np.argmin(m.body_inertia[h]))
    for ax in range(3):
        if ax != long_axis:
            m.body_inertia[h, ax] = I0
            mc.env.solver._dry_inertia[h, ax] = I0
    mj.mj_setConst(m, mj.MjData(m))
    q = d.qpos[3:7].copy()
    if yaw0:
        a = math.radians(yaw0)
        q = quat_mul(np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]), q)
    if roll0:
        a = math.radians(roll0)
        q = quat_mul(q, np.array([math.cos(a / 2), math.sin(a / 2), 0.0, 0.0]))
    d.qpos[3:7] = q
    d.qvel[:] = 0.0
    ya = math.radians(yaw0)
    pa = math.radians(pitch0)
    d.qvel[0], d.qvel[1], d.qvel[2] = V * math.cos(ya) * math.cos(pa), V * math.sin(ya) * math.cos(pa), 0.0
    mj.mj_forward(m, d)
    mc.enable_strip_twist(c1_pilot.body())
    return mc


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class Flight:
    """One flight: C1's longitudinal pilot and a lateral input dict set by the caller each step."""

    def __init__(self, mc, pt, V=8.0, z_ref=None):
        self.mc = mc
        f, Phi_d, Psi_d, psi0_d = pt[:4]
        self.f, self.V = f, V
        self.w0 = 2 * math.pi * f
        self.Phi, self.Psi, self.psi0 = math.radians(Phi_d), math.radians(Psi_d), math.radians(psi0_d)
        r = mc.role
        self.st, self.fe, self.ta = r == "stroke", r == "feather", r == "tail"
        # side of each hinge from where its body is at build (+1: +y)
        yb = [float(mc.d.xipos[int(mc.m.jnt_bodyid[j])][1] - mc.d.xipos[mc.hull][1]) for j in mc.jid]
        self.side = np.where(np.array(yb) > 0.0, 1.0, -1.0)
        self.z_ref = float(mc.d.qpos[2]) if z_ref is None else z_ref
        self.tail_q, self.e_f, self.r_f, self.e_i = 0.0, 0.0, 0.0, 0.0
        self.zf, self.vzf, self.vxf = 0.0, 0.0, V
        self.ph, self.w = 0.0, self.w0
        self.n_beat = int(round(1.0 / (f * mc.dt)))
        self.lat = {k: 0.0 for k in LAT_KEYS}
        self.amp_scale = 1.0         # common stroke amplitude factor (a throttle; 1 = the point's)
        self.v_ref = V
        self.hold = None             # None: flying; else the joint targets held (do-nothing)
        self.k = 0
        self.law = None              # lateral law (dict of gains) or None: no lateral loop
        self.psi_c = None            # commanded track heading, deg
        self.lf = dict(roll=0.0, p=0.0, beta=0.0, r=0.0, trk=None)

    def do_nothing(self):
        """From now on every hinge held still: strokes level, root feathers at
        the point's offset, tail where it is, tips untwisted."""
        n = len(self.mc.aid)
        q = np.zeros(n)
        q[self.fe] = self.psi0
        q[self.ta] = self.tail_q
        self.mc.set_twist(0.0, 0.0)
        self.hold = q

    def track_deg(self):
        d = self.mc.d
        return math.degrees(math.atan2(d.qvel[1], d.qvel[0]))

    def lateral(self):
        """The lateral law (PREREG_M4 section 2), from the true state low-passed
        as the pitch loop's: track to the commanded heading by bank, bank by the
        root feathers' differential offset, sideslip and yaw rate by the tips'
        differential twist amplitude (on beta and its rate: a turn's own yaw
        rate is not opposed)."""
        g = self.law
        if g is None or self.psi_c is None:
            return
        mc, d = self.mc, self.mc.d
        a_f = mc.dt * self.f / TL["lag_beats"] * 2.0 * math.pi
        lf = self.lf
        trk = self.track_deg()
        lf["trk"] = trk if lf["trk"] is None else lf["trk"] + a_f * wrap(trk - lf["trk"])
        lf["roll"] += a_f * (mc.roll_deg() - lf["roll"])
        lf["p"] += a_f * (math.degrees(d.qvel[3]) - lf["p"])
        lf["beta"] += a_f * (wrap(mc.yaw_deg() - trk) - lf["beta"])
        v2 = max(d.qvel[0] ** 2 + d.qvel[1] ** 2, 1e-6)
        trk_rate = math.degrees((d.qvel[0] * d.qacc[1] - d.qvel[1] * d.qacc[0]) / v2)
        lf["r"] += a_f * (math.degrees(self.omega_world()[2]) - trk_rate - lf["r"])    # d(beta)/dt
        e = wrap(self.psi_c - lf["trk"])
        phi_c = min(max(-g["kpsi"] * e, -g["phimax"]), g["phimax"])
        u_f = g.get("ff", 0.0) * phi_c + g["kphi"] * (lf["roll"] - phi_c) + g["kp"] * lf["p"]
        u_y = g["kbeta"] * lf["beta"] + g["kr"] * lf["r"]
        self.lat["feather"] = min(max(u_f, -g["fmax"]), g["fmax"])
        self.lat["twamp"] = min(max(u_y, -g["ymax"]), g["ymax"])

    # -- the true state the laws read
    def omega_world(self):
        mc = self.mc
        R = mc.d.xmat[mc.hull].reshape(3, 3)
        return R @ mc.d.qvel[3:6]

    def step(self):
        mc, d = self.mc, self.mc.d
        if self.hold is not None:
            mc.step(self.hold, None)
            self.k += 1
            return
        self.lateral()
        dt, f = mc.dt, self.f
        a_f = dt * f / TL["lag_beats"] * 2.0 * math.pi
        pitch = math.radians(mc.pitch_deg())
        p_rate = -d.qvel[4]
        z_now = d.qpos[2] - self.z_ref
        self.zf += a_f * (z_now - self.zf)
        self.vzf += a_f * (d.qvel[2] - self.vzf)
        self.vxf += a_f * (math.hypot(d.qvel[0], d.qvel[1]) - self.vxf)
        th_ask = math.radians(min(max(AP["kh"] * (0.0 - self.zf) - AP["kvz"] * self.vzf,
                                      -AP["pitch_lim_deg"]), AP["pitch_lim_deg"]))
        self.w = self.w0 * (1.0 + min(max(AP["kf"] * (self.v_ref - self.vxf), -AP["f_lim"]), AP["f_lim"]))
        self.e_f += a_f * ((pitch - th_ask) - self.e_f)
        self.r_f += a_f * (p_rate - self.r_f)
        lim = math.radians(TL["limit_deg"])
        self.e_i = min(max(self.e_i + TL["ki"] * self.e_f * dt, -lim), lim)
        want = min(max(-(math.radians(TL["i0_deg"]) + TL["kp"] * self.e_f + TL["kd"] * self.r_f + self.e_i),
                       -lim), lim)
        self.tail_q += min(max(want - self.tail_q, -TL["rate"] * dt), TL["rate"] * dt)
        s, c = math.sin(self.ph), math.cos(self.ph)
        L = self.lat
        rad = math.radians
        sd = self.side
        n = len(mc.aid)
        q, qd = np.zeros(n), np.zeros(n)
        A = 0.5 * (self.Phi * self.amp_scale + sd * rad(L["amp"]))       # half amplitude per hinge
        b = rad(L["dihedral"]) + sd * rad(L["bias"])
        q[self.st] = (b + A * s)[self.st]
        qd[self.st] = (A * self.w * c)[self.st]
        q[self.fe] = (self.psi0 + sd * rad(L["feather"]))[self.fe]
        tl = np.clip(self.tail_q + sd * rad(L["tail"]), -0.59, 0.59)
        q[self.ta] = tl[self.ta]
        tip, tw_a = math.degrees(-self.Psi * c), -L["twamp"] * c
        mc.set_twist(tip + tw_a + L["twist"], tip - tw_a - L["twist"])
        mc.step(q, qd)
        self.ph += self.w * dt
        self.k += 1

    def state(self):
        mc, d = self.mc, self.mc.d
        om = self.omega_world()
        return dict(t=mc.t, x=float(d.qpos[0]), y=float(d.qpos[1]), z=float(d.qpos[2]),
                    pitch=mc.pitch_deg(), roll=mc.roll_deg(), yaw=mc.yaw_deg(),
                    V=float(math.hypot(d.qvel[0], d.qvel[1])), vz=float(d.qvel[2]),
                    p=float(d.qvel[3]), r_world=float(om[2]), tail=math.degrees(self.tail_q))
