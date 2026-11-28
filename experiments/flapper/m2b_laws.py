"""Pass M2b (PREREG_M2b_water.md): building, and the three new laws.

Reuses `m2_common` (Recorder, run_law, settle, state, Law, Still, draw) and
`m2_laws.Exit` unchanged.  New here:

- `build2`: `m2_common.build`, but a machine whose gains are set in air is
  built **at rest** (`vx = 0` in its pose).  `Machine("air")` keeps the
  launch speed `TriphibianEnv.reset(AIR)` gives it (7.2 m/s along x) unless
  the pose sets vx, and the PD gains are read from the mass matrix with the
  wings' air added mass projected on that flow: stroke J 0.0179 level at
  7.2 m/s, 0.0200 at rest (M0's), 0.0243 pitched 90 deg at 7.2 m/s.  This is
  M2's C0 mismatch (PREREG_M2b section 0).
- `Stay`: M2's XV exit (strike to -1.0 rad, flight when no main-wing strip is
  wet) with M1's V+ flying law (`m2b_fly`, from 331e737), gains verbatim.
- `Track`: the stroke sinusoid with each wing's feather set every 1 ms so
  that its reference strip meets the flow it sees (true strip velocity,
  recoil included) at a set angle, alternating sign with the stroke.
- `Row`: a surface stroke biased down with a fast downstroke and a slow
  recovery; each wing's feather is held at +-Psi_r (sign per wing), so a
  wing pushes water fore or aft; same signs make thrust either way along
  the hull, opposite signs a yaw couple.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from m2_common import DEPTH0, VB_SCALE, X0, Law, _quat, wrap  # noqa: E402
from m2_laws import Exit, _rad  # noqa: E402

D = math.radians


# ------------------------------------------------------------------ building
def build2(start, dr, *, gains, vb=False, entrainment="wetting", motors=None, dt=1e-4, float_z=0.0):
    from mission import Machine
    from m2_common import _regain
    q = _quat(0.0, dr)
    x, y = X0 + dr["dx"], dr["dy"]
    if gains == "air":
        mc = Machine("air", motors=motors, dt=dt, pose=dict(x=x, y=y, z=5.0, vx=0.0, vz=0.0))
    else:
        mc = Machine("water", motors=motors, dt=dt, pose=dict(x=x, y=y, z=-DEPTH0, vx=0.0, vz=0.0))
    d = mc.d
    d.qpos[3:7] = q
    mc.mj.mj_forward(mc.m, d)
    if start == "depth":
        d.qpos[2] += -DEPTH0 - float(mc.com()[2])
    else:
        d.qpos[2] = float_z
    d.qvel[:] = 0.0
    mc.mj.mj_forward(mc.m, d)
    s = mc.env.solver
    if vb:
        s.panels.volume_buoyant = s.panels.volume_buoyant * VB_SCALE
    s.entrainment = entrainment
    if gains == "here" and start == "float":
        _regain(mc)
    s.reset()
    mc.mj.mj_forward(mc.m, d)
    return mc


# ------------------------------------------------------------------ (i) exit and stay up
class Stay(Exit):
    """XV's exit, then M1's V+ flying law (m2b_fly, 331e737), gains verbatim."""
    name = "stay"

    def defaults(self):
        return dict(super().defaults(), mode="XV", switch_clear=True, switch_s=2.0, q_strike=-1.0)

    def _start_fly(self, mc):
        import m2b_fly as F
        g = dict(F.GAINS, **F.PLUS)
        g.update(self.g["fly"])
        self.fly = F.Fly(mc, g, loop=True)
        self.fly.t0 = mc.t
        self.phase, self.t_ph = "fly", mc.t
        self.log.append(("fly", round(mc.t, 3)))


# ------------------------------------------------------------------ helpers
def wing_strips(mc):
    """Per wing (seg1, seg2): the index of its reference strip (nearest 0.7 of
    the wing's strip span) and the feather actuator index."""
    from dytiscidae.physics.fluid import WING
    p = mc.env.solver.panels
    out = []
    for b in (2, 3):
        idx = np.nonzero((p.body_id == b) & (p.kind == WING))[0]
        r = np.abs(p.pos_local[idx] @ (p.span_local[idx[0]] / np.linalg.norm(p.span_local[idx[0]])))
        k = idx[int(np.argmin(np.abs(r - (r.min() + 0.7 * (r.max() - r.min())))))]
        fe = [n for n, (j, role) in enumerate(zip(mc.jid, mc.role)) if role == "feather" and int(mc.m.jnt_bodyid[j]) == b][0]
        st = [n for n, (j, role) in enumerate(zip(mc.jid, mc.role)) if role == "stroke" and int(mc.m.jnt_bodyid[j]) == b][0]
        out.append((b, int(k), fe, st))
    return out


def strip_aoa(mc, b, k):
    """Angle of the flow the strip meets, in its own section plane:
    atan2(u.n, u.c) with u the fluid velocity relative to the strip (still
    water: minus the strip's own velocity, recoil included)."""
    mj, m, d = mc.mj, mc.m, mc.d
    p = mc.env.solver.panels
    R = d.xmat[b].reshape(3, 3)
    pos = d.xpos[b] + R @ p.pos_local[k]
    v6 = np.zeros(6)
    mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, b, v6, 0)
    v = v6[3:] + np.cross(v6[:3], pos - d.xipos[b])
    s = R @ p.span_local[k]
    c = R @ p.chord_local[k]
    n = R @ p.normal_local[k]
    u = -v
    u = u - (u @ s) * s
    return math.atan2(float(u @ n), float(u @ c)), float(np.linalg.norm(u))


# ------------------------------------------------------------------ (ii) swim by flapping with the angle held
class Track(Law):
    """WT.  stroke_i = b + s_i A/2 sin(phi); feather_i set each 1 ms so that the
    reference strip's flow angle is sgn * a_set * sign(stroke rate) (+ d_i), by
    an incremental correction psi += k_i^-1 (target - measured), k_i read from
    the built model (the flow angle's change per radian of feather).  Depth:
    the angle on the downstroke and the upstroke differ by dz (rad per m of
    depth error, limited); heading: the amplitude share, as WS, travel along
    the nose (or the tail when tailfirst)."""
    name = "WT"

    def defaults(self):
        return dict(f=0.25, A=D(60), b=0.0, a_set=D(45), sgn=1.0, gain=0.5, kz=D(10), zlim=D(15),
                    kh=1.5, smin=-1.0, tailfirst=0, ramp=2.0, e0=0.0, kth=0.0, th_ref=0.0, vmin=0.02)

    def __init__(self, **g):
        super().__init__(**_rad(g))
        self.ws = None
        self.psi = [0.0, 0.0]
        self.kk = [None, None]
        self.dz = 0.0
        self.cnt = 0
        self.log = []

    def _k(self, mc, b, k, fe):
        """d(flow angle)/d(feather) for a fixed flow, from the built model: the
        strip's chord and normal before and after turning the feather 0.01 rad."""
        mj, m, d = mc.mj, mc.m, mc.d
        p = mc.env.solver.panels

        def cn():
            R = d.xmat[b].reshape(3, 3)
            return R @ p.chord_local[k], R @ p.normal_local[k]
        q0 = d.qpos.copy()
        c0, n0 = cn()
        u = c0 + n0
        a0 = math.atan2(float(u @ n0), float(u @ c0))
        d.qpos[mc.qad[fe]] += 0.01
        mj.mj_kinematics(m, d)
        c1, n1 = cn()
        a1 = math.atan2(float(u @ n1), float(u @ c1))
        d.qpos[:] = q0
        mj.mj_kinematics(m, d)
        return wrap(a1 - a0) / 0.01

    def update(self, st, cmd):
        g = self.g
        if "h" in cmd:
            travel = st["yaw"] + (math.pi if g["tailfirst"] else 0.0)
            vh = np.asarray(st["v"][:2], float)
            if g.get("course") and float(np.linalg.norm(vh)) > 0.03:
                travel = math.atan2(vh[1], vh[0])       # the course over the ground, not the nose
            e = wrap(cmd["h"] - travel)
            u = float(np.clip(g["kh"] * e, -1.0, 1.0))
            self.s1 = float(np.clip(1.0 + u, g["smin"], 1.0))
            self.s2 = float(np.clip(1.0 - u, g["smin"], 1.0))
        if "z" in cmd:
            ed = st["depth"] - cmd["z"]                     # + : too deep
            dz = float(np.clip(g["kz"] * ed, -g["zlim"], g["zlim"]))
            # the angle difference pushes along the body's dorsal axis: scale by its
            # vertical share (sign included) when `dorsal` is set
            self.dz = dz * float(st["dors"][2]) if g.get("dorsal") else dz
        th = math.radians(st["pitch"])
        self.e = float(np.clip(g["e0"] + g["kth"] * (th - g["th_ref"]), -0.55, 0.55))

    def targets(self, mc, dt):
        q, qd = super().targets(mc, dt)
        if self.ws is None:
            self.ws = wing_strips(mc)
        g = self.g
        r = min((mc.t - self.t0) / max(self.ramp, 1e-9), 1.0)
        self.cnt += 1
        for i, (b, k, fe, stn) in enumerate(self.ws):
            if self.kk[i] is None:
                self.kk[i] = self._k(mc, b, k, fe)
                self.log.append(("k", i, round(self.kk[i], 3)))
            if self.cnt % 10 == 0:
                rate = float(mc.d.qvel[mc.dof[stn]])
                a, U = strip_aoa(mc, b, k)
                sgn_s = 1.0 if rate >= 0 else -1.0
                # the angle differs by dz between down- and upstroke: dz > 0 (too deep) adds on the upstroke
                tgt = g["sgn"] * sgn_s * g["a_set"] + (0.5 * self.dz * sgn_s)
                if U > g["vmin"] and abs(self.kk[i]) > 1e-3:
                    self.psi[i] += g["gain"] * wrap(tgt - a) / self.kk[i]
                lo, hi = mc.m.jnt_range[mc.jid[fe]]
                self.psi[i] = float(np.clip(self.psi[i], lo + 0.02, hi - 0.02))
            q[fe] = r * self.psi[i]
            qd[fe] = 0.0
        return q, qd


# ------------------------------------------------------------------ (iii) turn on the surface
class Row(Law):
    """ST.  At the surface: stroke_i = b + s_i A/2 w(phi) with a fast downstroke
    (a share `down` of the beat) and a slow recovery; feather_i = sigma_i Psi_r
    held.  sigma_1 = sigma_2 = +1 or -1 pushes the body one way or the other
    along the hull; opposite signs turn it.  The law turns toward the heading
    and, once within `lock` of it (or of its reverse), rows along it, forward
    or backward, whichever is nearer.  Sign of each push read from the model
    before freezing (PREREG_M2b section 0)."""
    name = "ST"

    def defaults(self):
        return dict(f=0.3, A=D(50), b=D(-35), Psi_r=D(45), down=0.3, lock=D(25), kh=2.0,
                    fwd_sign=1.0, turn_sign=1.0, ramp=2.0, allow_back=1)

    def __init__(self, **g):
        super().__init__(**_rad(g))
        self.sig = [1.0, 1.0]
        self.mode = "turn"
        self.log = []

    def update(self, st, cmd):
        g = self.g
        if g.get("fix") is not None:              # open loop (sign probes): the pushes as given
            self.sig = list(g["fix"])
            return
        if "h" not in cmd:
            return
        e_f = wrap(cmd["h"] - st["yaw"])                    # error to swim nose first
        e_b = wrap(cmd["h"] - st["yaw"] - math.pi)          # ... or tail first
        back = bool(g["allow_back"]) and abs(e_b) < abs(e_f)
        e = e_b if back else e_f
        if abs(e) > g["lock"]:
            # turn: opposite pushes, toward the error; e > 0 wants +yaw
            s = g["turn_sign"] * (1.0 if e > 0 else -1.0)
            self.sig = [s, -s]
            self.s1 = self.s2 = 1.0
            mode = "turn"
        else:
            d = -1.0 if back else 1.0
            self.sig = [g["fwd_sign"] * d, g["fwd_sign"] * d]
            u = float(np.clip(g["kh"] * e, -0.8, 0.8))
            self.s1 = float(np.clip(1.0 + u, 0.2, 1.0))
            self.s2 = float(np.clip(1.0 - u, 0.2, 1.0))
            mode = "back" if back else "fwd"
        if mode != self.mode:
            self.log.append((mode, round(st["t"], 2)))
            self.mode = mode

    def _w(self, phi):
        """Normalised stroke in [-1, 1] and its derivative in phi: from +1 to -1
        over `down` of the beat (cosine), back over the rest."""
        g = self.g
        x = (phi / (2 * math.pi)) % 1.0
        dn = g["down"]
        if x < dn:
            u = x / dn
            return math.cos(math.pi * u), -math.pi * math.sin(math.pi * u) / (dn * 2 * math.pi)
        u = (x - dn) / (1.0 - dn)
        return -math.cos(math.pi * u), math.pi * math.sin(math.pi * u) / ((1.0 - dn) * 2 * math.pi)

    def targets(self, mc, dt):
        if self.t0 is None:
            self.t0 = mc.t
        g = self.g
        r = min((mc.t - self.t0) / max(self.ramp, 1e-9), 1.0)
        w = 2 * math.pi * g["f"]
        self.phi += w * dt
        wv, dw = self._w(self.phi)
        q = np.zeros(len(mc.aid))
        qd = np.zeros(len(mc.aid))
        k = 0
        for n, role in enumerate(mc.role):
            if role == "stroke":
                s = self.s1 if k == 0 else self.s2
                q[n] = r * (g["b"] + s * g["A"] / 2 * wv)
                qd[n] = r * s * g["A"] / 2 * dw * w
            elif role == "feather":
                q[n] = r * self.sig[0 if k == 1 else 1] * g["Psi_r"]
            k += 1
        rng = mc.m.jnt_range[mc.jid]
        q = np.minimum(np.maximum(q, rng[:, 0] + 1e-3), rng[:, 1] - 1e-3)
        return q, qd


LAWS2 = dict(stay=Stay, WT=Track, ST=Row)


def make2(spec):
    from m2_laws import make, Frozen
    from m2_common import Still
    kind = spec["law"]
    if kind in LAWS2:
        law = LAWS2[kind](**spec.get("g", {}))
        return Frozen(law) if spec.get("frozen") else law
    return make(spec)
