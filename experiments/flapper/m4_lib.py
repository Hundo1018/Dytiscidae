"""Pass M4, phase 2: the runner's pieces (PREREG_M4_cruise_entries.md, addendum
M4a).  `m4_common.py` is frozen as committed (the build, C1's pilot, law L,
the do-nothing hold); this file adds, without changing those:

- `EntryFlight`: law E (approach on a glide path, flare, contact hold) on top
  of `m4_common.Flight` (its step is copied here with the height reference,
  pitch override, amplitude and frequency of E in place of the cruise pilot's);
- `SensorFlight`: law L and C1's pitch loop reading `sensors.Sensors` (arm LS);
- `build_dt`: `m4_common.build` with the step passed through (the dt arm);
- `Rec`: the per-step bookkeeping of section 1 (power windows, hinge torque
  and limit peaks, loads around an entry) and the 10 Hz trace;
- `run_half(spec)`: one half, from launch to the end of its window.

Entrainment: "wetting" primary (M4a); a spec may set ``ent`` to True (the
solver's default) for the sensitivity.
"""
from __future__ import annotations

import hashlib
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m4_common as C  # noqa: E402

G = 9.80665
RAMP = math.atan(0.12)
TC_AFTER = 5.0


def points():
    import c1_pilot
    pts = sorted(c1_pilot.points(), key=lambda p: p[4])
    return dict(P25=pts[0], P35=[p for p in pts if p[4] > 10.0][0])


def build_dt(dt, V=8.0, x0=-40.0, y0=0.0, z0=30.0, yaw0=0.0, roll0=0.0, seed=0):
    """m4_common.build with ``dt`` (a copy: that file is frozen)."""
    import c1_pilot
    from mission import Machine
    mc = Machine("air", body=c1_pilot.body(), pose=dict(x=x0, y=y0, z=z0, pitch_deg=0.0), seed=seed, dt=dt)
    m, d, mj = mc.m, mc.d, mc.mj
    h = mc.hull
    long_axis = int(np.argmin(m.body_inertia[h]))
    for ax in range(3):
        if ax != long_axis:
            m.body_inertia[h, ax] = C.I0
            mc.env.solver._dry_inertia[h, ax] = C.I0
    mj.mj_setConst(m, mj.MjData(m))
    q = d.qpos[3:7].copy()
    if yaw0:
        a = math.radians(yaw0)
        q = C.quat_mul(np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]), q)
    if roll0:
        a = math.radians(roll0)
        q = C.quat_mul(q, np.array([math.cos(a / 2), math.sin(a / 2), 0.0, 0.0]))
    d.qpos[3:7] = q
    d.qvel[:] = 0.0
    ya = math.radians(yaw0)
    d.qvel[0], d.qvel[1] = V * math.cos(ya), V * math.sin(ya)
    mj.mj_forward(m, d)
    mc.enable_strip_twist(c1_pilot.body())
    return mc


def fix_side(fl):
    """The +y side in the HULL's frame, as PREREG_M4 section 2 defines the inputs.
    `m4_common.Flight` reads the sign of each hinge body's offset in WORLD y at
    build, which is the hull's y only while |launch yaw| < 90 deg: launched at
    yaw 135, -120 or 180 it swaps the sides and every lateral input acts with
    the wrong sign (found on P35 start K3, which rolled over before t_c)."""
    mc = fl.mc
    R = mc.d.xmat[mc.hull].reshape(3, 3)
    yb = [float((mc.d.xipos[int(mc.m.jnt_bodyid[j])] - mc.d.xipos[mc.hull]) @ R[:, 1]) for j in mc.jid]
    fl.side = np.where(np.array(yb) > 0.0, 1.0, -1.0)


# ---------------------------------------------------------------- law E
class EntryFlight(C.Flight):
    """Law E (PREREG_M4 section 2).  ``E`` = dict(gamma, d_f, h_f, theta_f, a_f,
    r_f, b_c, e_c) in deg / m; ``x_e`` the commanded place of first contact;
    ``ground(x)`` the surface height under x (sea 0, beach the ramp)."""

    def __init__(self, mc, pt, E, x_e, ground, V=8.0, z_ref=None):
        super().__init__(mc, pt, V=V, z_ref=z_ref)
        self.E, self.x_e, self.ground = E, x_e, ground
        self.x_f = x_e - E["d_f"]
        self.z_f = ground(self.x_f) + E["h_f"]
        self.tg = math.tan(math.radians(E["gamma"]))
        self.ez = 0.0
        self.phase = "cruise"          # cruise -> approach -> flare -> contact
        self.armed = False             # E acts from t_c on

    def z_path(self, x):
        return min(self.z_ref, self.z_f + (self.x_f - x) * self.tg)

    def contact_hold(self):
        n = len(self.mc.aid)
        q = np.zeros(n)
        q[self.st] = math.radians(self.E["b_c"])
        q[self.fe] = self.psi0
        q[self.ta] = math.radians(self.E["e_c"])
        self.mc.set_twist(0.0, 0.0)
        self.hold = q
        self.phase = "contact"

    def step(self):
        if not self.armed or self.hold is not None:
            return C.Flight.step(self)
        mc, d = self.mc, self.mc.d
        self.lateral()
        dt, f = mc.dt, self.f
        a_f = dt * f / C.TL["lag_beats"] * 2.0 * math.pi
        x = float(d.qpos[0])
        E = self.E
        if x >= self.x_f:
            self.phase = "flare"
        elif self.z_path(x) < self.z_ref:
            self.phase = "approach"
        pitch = math.radians(mc.pitch_deg())
        p_rate = -d.qvel[4]
        V = math.hypot(d.qvel[0], d.qvel[1])
        on_path = self.phase == "approach"
        vz_path = -V * self.tg if on_path else 0.0
        self.ez += a_f * ((d.qpos[2] - self.z_path(x)) - self.ez)
        self.vzf += a_f * ((d.qvel[2] - vz_path) - self.vzf)
        self.vxf += a_f * (V - self.vxf)
        if self.phase == "flare":
            th_ask = math.radians(E["theta_f"])
            self.w = self.w0 * E["r_f"]
            self.amp_scale = E["a_f"]
        else:
            th_ask = math.radians(min(max(C.AP["kh"] * (0.0 - self.ez) - C.AP["kvz"] * self.vzf,
                                          -C.AP["pitch_lim_deg"]), C.AP["pitch_lim_deg"]))
            self.w = self.w0 * (1.0 + min(max(C.AP["kf"] * (self.v_ref - self.vxf), -C.AP["f_lim"]), C.AP["f_lim"]))
        self.e_f += a_f * ((pitch - th_ask) - self.e_f)
        self.r_f += a_f * (p_rate - self.r_f)
        lim = math.radians(C.TL["limit_deg"])
        self.e_i = min(max(self.e_i + C.TL["ki"] * self.e_f * dt, -lim), lim)
        want = min(max(-(math.radians(C.TL["i0_deg"]) + C.TL["kp"] * self.e_f + C.TL["kd"] * self.r_f + self.e_i),
                       -lim), lim)
        self.tail_q += min(max(want - self.tail_q, -C.TL["rate"] * dt), C.TL["rate"] * dt)
        self._drive()

    def _drive(self):
        mc = self.mc
        s, c = math.sin(self.ph), math.cos(self.ph)
        L, rad, sd = self.lat, math.radians, self.side
        n = len(mc.aid)
        q, qd = np.zeros(n), np.zeros(n)
        A = 0.5 * (self.Phi * self.amp_scale + sd * rad(L["amp"]))
        A = np.minimum(A, rad(80.0))
        b = rad(L["dihedral"]) + sd * rad(L["bias"])
        q[self.st] = (b + A * s)[self.st]
        qd[self.st] = (A * self.w * c)[self.st]
        q[self.fe] = (self.psi0 + sd * rad(L["feather"]))[self.fe]
        tl = np.clip(self.tail_q + sd * rad(L["tail"]), -0.59, 0.59)
        q[self.ta] = tl[self.ta]
        tip, tw_a = math.degrees(-self.Psi * c), -L["twamp"] * c
        mc.set_twist(tip + tw_a + L["twist"], tip - tw_a - L["twist"])
        mc.step(q, qd)
        self.ph += self.w * mc.dt
        self.k += 1


# ---------------------------------------------------------------- arm LS
class SensorFlight(C.Flight):
    """Law L and C1's pitch loop reading `sensors.Sensors` for attitude, body
    rates and height (pressure); speeds and track true (no air-data or GPS
    model).  A copy of `Flight.step`/`lateral` with those readings substituted."""

    def __init__(self, mc, pt, V=8.0, z_ref=None, seed=0):
        super().__init__(mc, pt, V=V, z_ref=z_ref)
        from sensors import Sensors
        self.S = Sensors(mc, seed=seed, init_true=True)
        self.S.z0 = self.z_ref

    def _read(self):
        r = self.S.last
        mc, d = self.mc, self.mc.d
        if r is None:
            return mc.pitch_deg(), mc.roll_deg(), mc.yaw_deg(), -d.qvel[4], d.qvel[3], d.qvel[5], d.qpos[2] - self.z_ref
        q = r["quat"]
        yaw = math.degrees(math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))
        g = r["gyro"]
        return r["pitch_deg"], r["roll_deg"], yaw, -g[1], g[0], g[2], (r["height"] if r["height"] is not None else 0.0)

    def step(self):
        if self.hold is not None:
            return C.Flight.step(self)
        mc, d = self.mc, self.mc.d
        pitch_d, roll_d, yaw_d, prate, rrate, yrate_b, z_now = self._read()
        dt, f = mc.dt, self.f
        a_f = dt * f / C.TL["lag_beats"] * 2.0 * math.pi
        g = self.law
        if g is not None and self.psi_c is not None:
            lf = self.lf
            trk = self.track_deg()
            lf["trk"] = trk if lf["trk"] is None else lf["trk"] + a_f * C.wrap(trk - lf["trk"])
            lf["roll"] += a_f * (roll_d - lf["roll"])
            lf["p"] += a_f * (math.degrees(rrate) - lf["p"])
            lf["beta"] += a_f * (C.wrap(yaw_d - trk) - lf["beta"])
            v2 = max(d.qvel[0] ** 2 + d.qvel[1] ** 2, 1e-6)
            trk_rate = math.degrees((d.qvel[0] * d.qacc[1] - d.qvel[1] * d.qacc[0]) / v2)
            # world yaw rate from the gyro through the estimated attitude
            from sensors import _quat_to_R
            r_ = self.S.last
            wz = (_quat_to_R(r_["quat"]) @ r_["gyro"])[2] if r_ is not None else self.omega_world()[2]
            lf["r"] += a_f * (math.degrees(wz) - trk_rate - lf["r"])
            e = C.wrap(self.psi_c - lf["trk"])
            phi_c = min(max(-g["kpsi"] * e, -g["phimax"]), g["phimax"])
            u_f = g.get("ff", 0.0) * phi_c + g["kphi"] * (lf["roll"] - phi_c) + g["kp"] * lf["p"]
            u_y = g["kbeta"] * lf["beta"] + g["kr"] * lf["r"]
            self.lat["feather"] = min(max(u_f, -g["fmax"]), g["fmax"])
            self.lat["twamp"] = min(max(u_y, -g["ymax"]), g["ymax"])
        pitch = math.radians(pitch_d)
        self.zf += a_f * (z_now - self.zf)
        self.vzf += a_f * (d.qvel[2] - self.vzf)
        self.vxf += a_f * (math.hypot(d.qvel[0], d.qvel[1]) - self.vxf)
        th_ask = math.radians(min(max(C.AP["kh"] * (0.0 - self.zf) - C.AP["kvz"] * self.vzf,
                                      -C.AP["pitch_lim_deg"]), C.AP["pitch_lim_deg"]))
        self.w = self.w0 * (1.0 + min(max(C.AP["kf"] * (self.v_ref - self.vxf), -C.AP["f_lim"]), C.AP["f_lim"]))
        self.e_f += a_f * ((pitch - th_ask) - self.e_f)
        self.r_f += a_f * (prate - self.r_f)
        lim = math.radians(C.TL["limit_deg"])
        self.e_i = min(max(self.e_i + C.TL["ki"] * self.e_f * dt, -lim), lim)
        want = min(max(-(math.radians(C.TL["i0_deg"]) + C.TL["kp"] * self.e_f + C.TL["kd"] * self.r_f + self.e_i),
                       -lim), lim)
        self.tail_q += min(max(want - self.tail_q, -C.TL["rate"] * dt), C.TL["rate"] * dt)
        EntryFlight._drive(self)
        self.S.sample()


# ---------------------------------------------------------------- recorder
class Rec:
    def __init__(self, mc, fl):
        self.mc, self.fl = mc, fl
        m, mj = mc.m, mc.mj
        self.names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        self.group = np.array([0 if n in ("seg0_hull", "seg3_strut") else (1 if n in ("seg1_wing", "seg2_wing")
                               else (2 if n in ("seg4_wing", "seg5_wing") else -1)) for n in self.names])
        self.wings = [self.names.index("seg1_wing"), self.names.index("seg2_wing")]
        self.dry = mc.env.solver._dry_mass.copy()
        self.W = mc.mass * G
        self.every = int(round(0.1 / mc.dt))
        self.tr = {k: [] for k in ("t", "x", "y", "z", "pitch", "roll", "yaw", "V", "vz", "trk", "zbz", "om",
                                   "wet", "gz_hull", "gz_wing", "gz_tail", "u_f", "u_y", "tail", "com", "comv",
                                   "qpos", "phase")}
        self.reset_window()
        self.rng = np.array(m.jnt_range[mc.jid])
        self.lim_steps = np.zeros(len(mc.jid))
        self.detail = False
        self.series = []               # per step in the detailed window
        self.first = None              # first contact record

    def reset_window(self):
        mc = self.mc
        self.E0 = mc.E_elec.copy()
        self.Es0 = mc.E_shaft.copy()
        self.lim0 = mc.at_limit.copy()
        self.n0 = mc.n
        self.t0 = mc.t
        self.Eprev = float(mc.E_elec.sum())
        self.win = deque()
        self.win_sum = 0.0
        self.p1 = -1e9
        self.p50 = -1e9
        self.tq = {r: 0.0 for r in ("stroke", "feather", "tail")}
        self.lim_steps = np.zeros(len(mc.jid))

    def com_vel(self):
        mc, m, d, mj = self.mc, self.mc.m, self.mc.d, self.mc.mj
        v = np.zeros(3)
        o = np.zeros(6)
        for b in range(1, m.nbody):
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, b, o, 0)
            v += self.dry[b] * o[3:]
        return v / self.dry.sum()

    def ground_forces(self):
        """Ground contact per group: vertical force (N) and normal-force magnitude (N)."""
        m, d, mj = self.mc.m, self.mc.d, self.mc.mj
        fz, fn = np.zeros(3), np.zeros(3)
        c = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            if (b1 == 0) == (b2 == 0):
                continue
            g = self.group[b2 if b1 == 0 else b1]
            if g < 0:
                continue
            mj.mj_contactForce(m, d, i, c)
            fn[g] += abs(c[0])
            fz[g] += abs(c[0] * con.frame[2])
        return fz, fn

    def after_step(self):
        mc, d, m = self.mc, self.mc.d, self.mc.m
        E = float(mc.E_elec.sum())
        p = (E - self.Eprev) / mc.dt
        self.Eprev = E
        self.p1 = max(self.p1, p)
        self.win.append(p)
        self.win_sum += p
        if len(self.win) > int(round(0.05 / mc.dt)):
            self.win_sum -= self.win.popleft()
            self.p50 = max(self.p50, self.win_sum / len(self.win))
        f = np.abs(d.actuator_force[mc.aid])
        for r in self.tq:
            self.tq[r] = max(self.tq[r], float(f[mc.role == r].max()))
        q = d.qpos[mc.qad]
        self.lim_steps += (q <= self.rng[:, 0] + 1e-3) | (q >= self.rng[:, 1] - 1e-3)
        if self.detail:
            self.series.append(self.loads())
        if mc.n % self.every == 0:
            self.sample()

    def loads(self):
        mc, m, d, mj = self.mc, self.mc.m, self.mc.d, self.mc.mj
        mj.mj_rnePostConstraint(m, d)
        out = [mc.t]
        for b in self.wings:
            f = d.cfrc_int[b]
            j = int(m.body_jntadr[b])
            root = int(m.body_rootid[b])
            mom = f[:3] - np.cross(d.xanchor[j] - d.subtree_com[root], f[3:])
            madd = float(m.body_mass[b] - self.dry[b])
            ff = d.xfrc_applied[b, :3] - np.array([0.0, 0.0, madd * G])
            out += [float(np.linalg.norm(f[3:])) / self.W, float(np.linalg.norm(mom)), float(np.linalg.norm(ff)) / self.W]
        fz, fn = self.ground_forces()
        out += (fn / self.W).tolist()
        lim = 0.0
        for i in range(d.nefc):
            if d.efc_type[i] == mj.mjtConstraint.mjCNSTR_LIMIT_JOINT:
                lim = max(lim, abs(float(d.efc_force[i])))
        out.append(lim)
        return out

    def sample(self):
        mc, d, fl = self.mc, self.mc.d, self.fl
        R = d.xmat[mc.hull].reshape(3, 3)
        T = self.tr
        T["t"].append(mc.t)
        for k, v in zip(("x", "y", "z"), d.qpos[:3]):
            T[k].append(float(v))
        T["pitch"].append(mc.pitch_deg())
        T["roll"].append(mc.roll_deg())
        T["yaw"].append(mc.yaw_deg())
        T["V"].append(float(math.hypot(d.qvel[0], d.qvel[1])))
        T["vz"].append(float(d.qvel[2]))
        T["trk"].append(fl.track_deg())
        T["zbz"].append(float(R[2, 2]))
        T["om"].append(float(np.linalg.norm(d.qvel[3:6])))
        T["wet"].append(float(getattr(mc.env.solver.diag, "max_submerged", 0.0)))
        fz, _ = self.ground_forces()
        T["gz_hull"].append(float(fz[0]) / self.W)
        T["gz_wing"].append(float(fz[1]) / self.W)
        T["gz_tail"].append(float(fz[2]) / self.W)
        T["u_f"].append(fl.lat["feather"])
        T["u_y"].append(fl.lat["twamp"])
        T["tail"].append(math.degrees(fl.tail_q))
        T["com"].append(mc.com().tolist())
        T["comv"].append(self.com_vel().tolist())
        T["qpos"].append(d.qpos.copy())
        T["phase"].append(getattr(fl, "phase", "cruise"))

    def window(self):
        mc = self.mc
        T = max(mc.t - self.t0, 1e-9)
        n = max(mc.n - self.n0, 1)
        dE = mc.E_elec - self.E0
        return dict(T=T, E_elec_J=float(dE.sum()), P_mean_W=float(dE.sum() / T),
                    P_mean_role={r: float(dE[mc.role == r].sum() / T) for r in ("stroke", "feather", "tail")},
                    shaft_W=float((mc.E_shaft - self.Es0).sum() / T),
                    P_peak50_W=float(self.p50), P_peak_step_W=float(self.p1),
                    at_limit={r: float((mc.at_limit - self.lim0)[mc.role == r].mean() / n) for r in ("stroke", "feather", "tail")},
                    tq_peak=dict(self.tq), range_limit_steps=self.lim_steps.tolist())


def snapshot(mc, fl, rec):
    """Everything a figure needs at this instant, from the built model (geoms,
    the fluid strips with their present twist, contacts and their forces)."""
    m, d, mj = mc.m, mc.d, mc.mj
    pan = mc.env.solver.panels
    R = d.xmat.reshape(-1, 3, 3)
    pw = d.xpos[pan.body_id] + np.einsum("nij,nj->ni", R[pan.body_id], pan.pos_local)
    cw = np.einsum("nij,nj->ni", R[pan.body_id], pan.chord_local)
    cons = []
    c = np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        mj.mj_contactForce(m, d, i, c)
        cons.append(dict(pos=con.pos.tolist(), fn=float(abs(c[0])), frame=con.frame[:3].tolist()))
    return dict(t=mc.t, qpos=d.qpos.tolist(), geom_xpos=d.geom_xpos.tolist(), geom_xmat=d.geom_xmat.tolist(),
                strip_pos=pw.tolist(), strip_chord=(cw * pan.chord[:, None]).tolist(), strip_body=pan.body_id.tolist(),
                strip_kind=pan.kind.tolist(), contacts=cons, com=mc.com().tolist(), lat=dict(fl.lat),
                tail=math.degrees(fl.tail_q), phase=getattr(fl, "phase", "cruise"), roll=mc.roll_deg(),
                pitch=mc.pitch_deg(), yaw=mc.yaw_deg(), track=fl.track_deg(),
                wet=float(getattr(mc.env.solver.diag, "max_submerged", 0.0)))


def state_hash(mc, fl):
    h = hashlib.sha256()
    for a in (mc.d.qpos, mc.d.qvel, mc.d.act, mc.d.ctrl):
        h.update(np.ascontiguousarray(a).tobytes())
    st = [fl.tail_q, fl.e_f, fl.r_f, fl.e_i, fl.zf, fl.vzf, fl.vxf, fl.ph, fl.w,
          *[float(v) for k, v in sorted(fl.lf.items()) if v is not None]]
    h.update(np.array(st).tobytes())
    return h.hexdigest()


def _snap_due(spec_t, t, t_c, contact):
    """A snapshot time: a float (s after t_c) or a string 'c+X' (s after first contact)."""
    if isinstance(spec_t, str):
        if contact is None or "x" not in contact:
            return False
        return t >= contact["t"] + float(spec_t[2:]) - 1e-9
    return t >= t_c + float(spec_t) - 1e-9


# ---------------------------------------------------------------- one half
def run_half(spec):
    """spec: dict(kind cruise|water|land, arm, point, start=(roll0, psi0, V0[, dz0]),
    cmd (heading for cruise, x_e for entries), E (entry law), ent, dt, id)."""
    kind, arm = spec["kind"], spec["arm"]
    pt = points()[spec["point"]]
    roll0, psi0, V0 = spec["start"][:3]
    dz0 = spec["start"][3] if len(spec["start"]) > 3 else 0.0
    if kind == "cruise":
        x0, z0 = -40.0, 30.0
    elif kind == "water":
        x0, z0 = -110.0, 3.0 + dz0
    else:
        x0, z0 = -50.0, 5.0 + dz0
    dt = spec.get("dt", 1e-4)
    mc = build_dt(dt, V=V0, x0=x0, z0=z0, yaw0=psi0, roll0=roll0) if dt != 1e-4 else \
        C.build(V=V0, x0=x0, z0=z0, yaw0=psi0, roll0=roll0)
    mc.env.solver.entrainment = spec.get("ent", "wetting")
    env = mc.env
    ground = (lambda x: 0.0) if kind != "land" else (lambda x: max(float(env.ground_height(x)), 0.0))
    psi_hold = psi0 if kind == "cruise" else 0.0
    if kind == "cruise":
        fl = SensorFlight(mc, pt, z_ref=z0, seed=spec.get("seed", 0)) if arm == "LS" else C.Flight(mc, pt, z_ref=z0)
    else:
        E = spec.get("E") or dict(gamma=5, d_f=0, h_f=1, theta_f=0, a_f=1, r_f=1, b_c=0, e_c=0)
        fl = EntryFlight(mc, pt, E, spec.get("cmd", 0.0), ground, z_ref=z0)
    fix_side(fl)
    fl.law = C.LAW_L
    fl.psi_c = psi_hold
    rec = Rec(mc, fl)
    nb = 0
    t_c = None
    snaps, snap_t = [], list(spec.get("snap", []))
    stop_at = spec.get("stop_at")

    def snap_check(t_rel_c=None):
        while snap_t and snap_t[0] is not None and _snap_due(snap_t[0], mc.t, t_c, contact):
            snaps.append(snapshot(mc, fl, rec))
            snap_t.pop(0)

    contact = None
    while t_c is None:                      # identical for every arm and half up to t_c
        fl.step()
        rec.after_step()
        b = int(fl.ph // (2 * math.pi))
        if mc.t >= TC_AFTER and b > nb:
            t_c = mc.t
        nb = b
        if not np.all(np.isfinite(mc.d.qpos)):
            break
    hsh = state_hash(mc, fl)
    pre = rec.window()
    rec.reset_window()
    if arm in ("N", "NW", "NL"):
        fl.do_nothing()
    elif arm == "L0":
        fl.law = None
        for k in fl.lat:
            fl.lat[k] = 0.0
    elif kind == "cruise":
        fl.psi_c = spec["cmd"]
    else:
        fl.armed = True
        fl.ez = fl.zf
    if kind != "cruise" and arm in ("NW", "NL"):
        rec.detail = True
    T_end = t_c + spec.get("win", 60.0 if kind == "cruise" else 45.0)
    end_reason = "time"
    while mc.t < T_end - 0.5 * mc.dt:
        if kind != "cruise" and getattr(fl, "phase", "") == "flare" and contact is None:
            rec.detail = True
        if contact is not None and "x" in contact and mc.t > contact["t"] + 2.0:
            rec.detail = False
        fl.step()
        rec.after_step()
        snap_check()
        if stop_at is not None and not snap_t:
            end_reason = "stopped_for_snap"
            break
        d = mc.d
        if not np.all(np.isfinite(d.qpos)):
            end_reason = "nonfinite"
            break
        if kind == "cruise":
            if d.qpos[2] < 0.5:
                end_reason = "sea"
                break
            continue
        if contact is None:
            hit = False
            if kind == "water":
                hit = float(env.solver.diag.max_submerged) > 0.0
                if not hit and float(rec.ground_forces()[1].sum()) > 0.0:
                    contact = dict(t=mc.t, medium="ground_before_water")
            else:
                fz, fn = rec.ground_forces()
                hit = float(fn.sum()) > 0.0
                if not hit and float(env.solver.diag.max_submerged) > 0.0:
                    contact = dict(t=mc.t, medium="water_before_land")
            if hit:
                rec.detail = True
                v = rec.com_vel()
                com = mc.com()
                if kind == "water":
                    vn = float(v[2])
                    pitch_rel = mc.pitch_deg()
                else:
                    n = np.array([-math.sin(RAMP), 0.0, math.cos(RAMP)])
                    vn = float(v @ n)
                    pitch_rel = mc.pitch_deg() - math.degrees(RAMP)
                contact = dict(t=mc.t, x=float(com[0]), com=com.tolist(), v=v.tolist(), vn=vn,
                               v_h=float(math.hypot(v[0], v[1])), roll=mc.roll_deg(), pitch=mc.pitch_deg(),
                               pitch_rel=pitch_rel, phase=getattr(fl, "phase", ""), qpos=d.qpos.tolist())
                if arm not in ("NW", "NL"):
                    fl.contact_hold()
                T_end = mc.t + (10.0 if kind == "water" else 6.0)
                end_reason = "rest_window"
            elif contact is not None and contact.get("medium") in ("water_before_land", "ground_before_water"):
                end_reason = contact["medium"]
                break
    win = rec.window()
    T = {k: (np.array(v) if k != "phase" else v) for k, v in rec.tr.items()}
    out = dict(id=spec["id"], kind=kind, arm=arm, point=spec["point"], start=list(spec["start"]),
               start_name=spec.get("start_name"), cmd=spec.get("cmd"), E=spec.get("E"), ent=spec.get("ent", "wetting"),
               dt=dt, t_c=t_c, hash=hsh, bad_qacc=mc.bad_qacc, end=end_reason, t_end=mc.t, pre=pre, win=win,
               contact=contact)
    if snaps:
        out["snaps"] = snaps
    if rec.series:
        S = np.array(rec.series)
        out["load_cols"] = ["t", "w1_shoulder_W", "w1_moment_Nm", "w1_fluid_W", "w2_shoulder_W", "w2_moment_Nm",
                            "w2_fluid_W", "g_hull_W", "g_wing_W", "g_tail_W", "limit_F"]
        tc = contact["t"] if contact and "x" in contact else None
        sel = (S[:, 0] >= tc - 0.5) & (S[:, 0] <= tc + 2.0) if tc is not None else np.ones(len(S), bool)
        if sel.any():
            Ss = S[sel]
            k10 = max(int(round(0.01 / mc.dt)), 1)
            def mean10(a):
                return float(np.convolve(a, np.ones(k10) / k10, mode="valid").max()) if len(a) >= k10 else float(a.max())
            out["loads"] = {c: float(Ss[:, i].max()) for i, c in enumerate(out["load_cols"]) if i}
            out["loads_10ms"] = {c: mean10(Ss[:, i]) for i, c in enumerate(out["load_cols"]) if i}
            out["loads_window"] = [float(Ss[0, 0]), float(Ss[-1, 0])]
    return out, T
