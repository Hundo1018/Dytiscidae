"""M0: one harness for the whole mission (`TRIPHIBIAN_SPEC.md`).

The 0.6 kg reference machine (BODY_S, the W6 wing: Goe 417a section,
camber 0.04, feather hinge 0.1 chord behind the leading edge; the 0.06 m2
all-moving tail of pass 1m) in the project's own `TriphibianEnv`: a free
six-axis body over the sea, the beach ramp and the air, ground contact by
MuJoCo, fluid forces by the strip solver in whichever medium each strip is in.

Every hinge is driven through a motor with a torque-speed limit (D1's witness
model): torque on the motor shaft ``tau_s (u - omega/omega_0)``, u in [-1, 1],
behind an ideal gear N, its rotor ``J_r`` reflected as ``J_r N^2`` of
armature, copper loss ``tau^2 omega_0 / tau_s``.  The hinge's own position
actuator does the following (an implicit PD, as `freerig`'s, which is stable
in water where the wing's added mass changes several-fold through a beat);
each step its force range is set to what the motor can give at the hinge's
present speed, so no torque outside the curve is ever applied.

    from mission import Machine
    mc = Machine("land")                          # or "water", "air", pose=...
    for k in range(n): mc.step(targets, rates)    # joint angles and rates, by role
    mc.log                                        # energy, positions, contact, medium

Nothing here decides a gate; passes M1-M5 read what it records.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: the witness motor of D1 per role, and the gear it drives the hinge through.
#: The tail's is a tenth of the others' (a small servo's size; TYPED, stated in M0).
MOTORS = {
    "stroke": dict(tau_s=1.0, omega_0=1000.0, J_r=2e-5, N=24.0),
    "feather": dict(tau_s=1.0, omega_0=1000.0, J_r=2e-5, N=16.0),
    "tail": dict(tau_s=0.1, omega_0=1000.0, J_r=2e-6, N=50.0),
}
MASS = 0.60


def reference_body() -> dict:
    import c1_pilot
    return c1_pilot.body()


class Machine:
    def __init__(self, domain: str = "land", *, body: dict | None = None, mass: float | None = MASS,
                 motors: dict | None = None, dt: float = 1e-4, wn: float = 300.0, seed: int = 0,
                 feather_axis: float = 0.1, pose: dict | None = None):
        import mujoco as mj
        import body as B
        from dytiscidae.envs.triphibian import Domain, TriphibianEnv
        import sections
        self.mj = mj
        body = dict(body or reference_body())
        p = B.built(body)
        env = TriphibianEnv(p, seed=seed, timestep=dt)
        if body.get("wing_section"):
            env.solver.section = sections.load(body["wing_section"])
        env.solver.cop_law = "plate"
        env.reset({"land": Domain.LAND, "water": Domain.WATER, "air": Domain.AIR}[domain], randomise=False)
        m, d = env.model, env.data
        self.env, self.m, self.d, self.dt = env, m, d, dt
        hull = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "seg0_hull")
        self.hull = hull
        if mass is not None:                       # ballast at the hull's centre, as freerig
            add = mass - float(env.solver._dry_mass.sum())
            env.solver._dry_mass[hull] += add
            m.body_mass[hull] += add
            mj.mj_setConst(m, d)
        self.mass = float(env.solver._dry_mass.sum())
        # roles, as freerig reads them
        self.aid, self.jid, self.role = [], [], []
        for a in env.act_names:
            aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, a)
            j = int(m.actuator_trnid[aid, 0])
            par = int(m.body_parentid[int(m.jnt_bodyid[j])])
            self.aid.append(aid)
            self.jid.append(j)
            self.role.append("tail" if par != hull else ("feather" if a.endswith("_f") else "stroke"))
        self.aid, self.jid = np.array(self.aid), np.array(self.jid)
        self.role = np.array(self.role)
        self.dof = np.array([int(m.jnt_dofadr[j]) for j in self.jid])
        self.qad = np.array([int(m.jnt_qposadr[j]) for j in self.jid])
        if feather_axis:                           # freerig's move of the feather hinge
            pan = env.solver.panels
            from dytiscidae.physics.fluid import WING as KIND_WING
            for n, j in enumerate(self.jid):
                if self.role[n] != "feather":
                    continue
                b = int(m.jnt_bodyid[j])
                sel = (pan.body_id == b) & (pan.kind == KIND_WING)
                cdir = pan.chord_local[sel][0] / np.linalg.norm(pan.chord_local[sel][0])
                if float(pan.pos_local[sel].mean(0) @ cdir) < 0.0:
                    cdir = -cdir
                m.jnt_pos[j] = m.jnt_pos[j] + feather_axis * float(pan.chord[sel].mean()) * cdir
        mt = {k: dict(MOTORS[k], **((motors or {}).get(k, {}))) for k in MOTORS}
        self.motor = [mt[r] for r in self.role]
        for n, a in enumerate(self.aid):
            mo = self.motor[n]
            m.dof_armature[self.dof[n]] += mo["J_r"] * mo["N"] ** 2
            m.actuator_forcelimited[a] = 1
            m.actuator_ctrllimited[a] = 0
        if pose:
            self.place(**pose)
        mj.mj_forward(m, d)
        env.solver.apply(d, 0.0)
        mj.mj_forward(m, d)
        J = np.zeros(len(self.dof))
        for n, i in enumerate(self.dof):
            e, col = np.zeros(m.nv), np.zeros(m.nv)
            e[i] = 1.0
            mj.mj_mulM(m, d, col, e)
            J[n] = col[i]
        self.kp, self.kv = J * wn * wn, 2.0 * J * wn
        for n, a in enumerate(self.aid):
            m.actuator_gainprm[a, :] = 0.0
            m.actuator_biasprm[a, :] = 0.0
            m.actuator_gainprm[a, 0] = self.kp[n]
            m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2] = -self.kp[n], -self.kv[n]
        self.t = 0.0
        self.E_elec = np.zeros(len(self.aid))    # J per motor
        self.E_shaft = np.zeros(len(self.aid))
        self.at_limit = np.zeros(len(self.aid))   # steps spent on the curve
        self.n = 0
        self.bad_qacc = 0

    # ------------------------------------------------------------- placement
    def place(self, x=None, y=None, z=None, pitch_deg=None, vx=None, vz=None, q=None):
        d = self.d
        if x is not None:
            d.qpos[0] = x
        if y is not None:
            d.qpos[1] = y
        if z is not None:
            d.qpos[2] = z
        if pitch_deg is not None:                  # nose up positive, as freerig
            a = math.radians(pitch_deg)
            d.qpos[3:7] = (math.cos(-a / 2), 0.0, math.sin(-a / 2), 0.0)
        if vx is not None:
            d.qvel[0] = vx
        if vz is not None:
            d.qvel[2] = vz
        if q is not None:
            d.qpos[self.qad] = q
        self.mj.mj_forward(self.m, d)

    def drop_to_ground(self, clearance=0.002):
        """Lower the machine until its lowest geom is ``clearance`` over what is under it."""
        m, d, mj = self.m, self.d, self.mj
        mj.mj_forward(m, d)
        low = min(float(d.geom_xpos[g][2] - (m.geom_rbound[g] if m.geom_type[g] != 6 else 0.0))
                  for g in range(m.ngeom) if m.geom_bodyid[g] != 0)
        under = float(self.env.ground_height(d.qpos[0]))
        d.qpos[2] += under + clearance - low
        mj.mj_forward(m, d)

    # ------------------------------------------------------------------ step
    def step(self, q_ref: np.ndarray, qd_ref: np.ndarray | None = None) -> None:
        m, d = self.m, self.d
        qd_ref = np.zeros(len(self.aid)) if qd_ref is None else qd_ref
        for n, a in enumerate(self.aid):           # the motor's band at this speed
            mo = self.motor[n]
            om = mo["N"] * float(d.qvel[self.dof[n]])
            lo = mo["N"] * mo["tau_s"] * (-1.0 - om / mo["omega_0"])
            hi = mo["N"] * mo["tau_s"] * (1.0 - om / mo["omega_0"])
            m.actuator_forcerange[a] = (lo, hi)
        d.ctrl[:] = 0.0
        d.ctrl[self.aid] = q_ref + (self.kv / self.kp) * qd_ref
        d.xfrc_applied[:] = 0.0
        self.env.solver.apply(d, d.time)
        qv = d.qvel[self.dof].copy()
        self.mj.mj_step(m, d)
        if not np.all(np.isfinite(d.qacc)) or d.warning[self.mj.mjtWarning.mjWARN_BADQACC].number > self.bad_qacc:
            self.bad_qacc = int(d.warning[self.mj.mjtWarning.mjWARN_BADQACC].number)
        f = d.actuator_force[self.aid]
        for n in range(len(self.aid)):
            mo = self.motor[n]
            tau_m, om_m = f[n] / mo["N"], mo["N"] * qv[n]
            ps = tau_m * om_m
            self.E_shaft[n] += ps * self.dt
            self.E_elec[n] += (ps + tau_m * tau_m * mo["omega_0"] / mo["tau_s"]) * self.dt
            lo, hi = m.actuator_forcerange[self.aid[n]]
            self.at_limit[n] += float(f[n] <= lo + 1e-9 or f[n] >= hi - 1e-9)
        self.t += self.dt
        self.n += 1

    # --------------------------------------------------------------- reading
    def pitch_deg(self) -> float:
        q = self.d.qpos[3:7]
        return math.degrees(math.asin(max(-1.0, min(1.0, -2.0 * (q[0] * q[2] - q[3] * q[1])))))

    def contact_load(self) -> float:
        """Vertical contact force on the machine over its weight (0 airborne)."""
        m, d, mj = self.m, self.d, self.mj
        fz = 0.0
        c = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            if (b1 == 0) == (b2 == 0):
                continue
            mj.mj_contactForce(m, d, i, c)
            n = con.frame[:3]
            fz += abs(c[0] * n[2])
        return fz / (self.mass * 9.80665)

    def summary(self) -> dict:
        roles = sorted(set(self.role))
        T = max(self.t, 1e-12)
        return dict(t=self.t, mass=self.mass, bad_qacc=self.bad_qacc,
                    elec_W={r: float(self.E_elec[self.role == r].sum() / T) for r in roles},
                    shaft_W={r: float(self.E_shaft[self.role == r].sum() / T) for r in roles},
                    at_limit={r: float(self.at_limit[self.role == r].mean() / max(self.n, 1)) for r in roles},
                    pos=self.d.qpos[:3].tolist(), vel=self.d.qvel[:3].tolist(), pitch_deg=self.pitch_deg())
