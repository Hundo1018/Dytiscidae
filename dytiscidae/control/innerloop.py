"""The stabilising inner loop of a rotor-and-limb body, and the action it takes.

The split
---------
A controller for the triphibian mission has two jobs that run at different
rates and need different things.  Keeping a multirotor the right way up is a
kilohertz job that reads attitude and body rates and nothing else; deciding
where to go is a 25 Hz job that reads the command and the map.  Until
2026-11-11 the only law that completes the mission
(`experiments/triphibian_diagnosis/existence/triquad.py`, E0) did both in one
function of the simulator state with thirteen latches.  E0s
(`existence/RESULT_stateless.md`) cut it in two and measured that nothing was
lost: 32, 32 and 31 of 32 missions against 32, 32, 32.

This module is the lower half and the interface between the halves:

    u = law(o)                         # `control.outer`: 25 Hz, held between decisions
    targets = InnerLoop(env, model).act(u)     # every simulation step

``u`` is ten numbers (`N_ACT`): an acceleration to produce, how far the body may
lean to produce it, whether thrust may point down, rotors on or off, a limb
drive and turn, a limb pose, and one memory channel the inner loop ignores.

What the inner loop reads
-------------------------
Attitude, body rates, velocity, each rotor's wetness, the submerged fraction
and the model.  Not position -- only differences of positions on the body --
not the command and not the task; the clock only for the gait's phase.  It
assigns no attribute after construction (`tests/test_physics.py` holds both).

Nothing else in the package mixes rotor thrusts: `control.cpg` drives a
pattern generator through a measured basis and has no attitude loop, and the
cascade this one descends from is `experiments/rotor/fly.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..physics.medium import GRAVITY
from ..physics.rotor import bemt

#: Memory channels the outer law is handed back at its next decision.  One is
#: what the mission needs (E0s prediction 4): the staged exit from the water.
K = 1

#: The action layout.  ``u[U_ACC:U_ACC + 3]`` is a world-frame acceleration.
U_ACC, U_TILT, U_DOWN, U_ON, U_DRIVE, U_TURN, U_POSE, U_MEM = 0, 3, 4, 5, 6, 7, 8, 9
N_ACT = U_MEM + K

#: Attitude loop: natural frequency (rad/s) and damping ratio of roll and
#: pitch, then of yaw.  `experiments/rotor/fly.py`'s cascade, rewritten per
#: axis on the inertia about the centre of mass in E0 (`existence/probes/p03`).
ATTITUDE = dict(wn=13.0, zeta=1.3, wn_yaw=9.0, zeta_yaw=1.0)

#: Axial advance ratio J = V_ax / (Omega R) below which a rotor's thrust is
#: static, and at which it is zero (E0 `existence/probes/p06`, this package's
#: `bemt`: T/T0 = 0.99 at 0.05, 0.86 at 0.10, 0.29 at 0.20, -0.05 at 0.25; the
#: same in both media).
J_FLAT, J_ZERO = 0.07, 0.243

#: The limb gait, set by hand in E0b (`existence/landgait.py`, tuning seeds
#: 100 and up; `RESULT_land.md`).
GAIT = dict(
    f=1.5,            # Hz
    sweep=0.6,        # rad, feather amplitude (foot fore-aft)
    down=0.25,        # rad, stroke in stance (foot pressed down, body carried)
    up=-0.35,         # rad, stroke in swing (foot lifted, body on its belly)
    sharp=1.5,        # stance/swing switch sharpness
    stand=0.2,        # rad, stroke of the pose held off land (feet 1.5 cm lower than neutral)
    rest=-0.5,        # rad, stroke when told to stop: limbs lifted, body on its belly
)


@dataclass(frozen=True, eq=False)
class BodyModel:
    """What either half of the split knows about the body that is not in the
    observation: actuator indices, rotor and servo constants, mass and volume.
    No map and no gain."""

    #: Rotor hub body ids, and actuator indices of the rotors, the limb
    #: strokes and the limb feathers.
    rot: list
    irot: list
    istroke: list
    ifeather: list
    #: +1 for a limb on the body's left, -1 on its right, per stroke actuator.
    side: list
    n_act: int
    #: Static thrust and torque coefficients (T = kt Omega^2) in air and water.
    kt_a: float
    kq_a: float
    kt_w: float
    kq_w: float
    rho_w: float
    rotor_radius: float
    om_max: float
    #: Rotor servo gain and torque limit.
    kv: float
    tau_max: float
    mass: float
    buoyant_volume: float
    #: Whether rotor reaction torque can yaw the body (`_yaw_authority`).
    yaw_ok: bool
    #: Rotor hubs in the root frame.
    hubs_body: np.ndarray
    launch_speed: float
    #: `TriphibianEnv.body_twist` (the observation's velocity and rates) is in
    #: the root body's *inertial* frame, at its centre of mass: the constant
    #: rotation and offset back to the root frame.
    R_i: np.ndarray
    ipos: np.ndarray

    @property
    def n(self) -> int:
        return len(self.rot)


def _yaw_authority(env, rot) -> bool:
    """Whether rotor reaction torque can yaw the body independently of roll
    and pitch.  On the reference quad it cannot: its propellers are handed
    by side (left +, right -), so the yaw row of the mixer is -(kappa/0.2)
    times the roll row and the 4x4 mix has rank 3.  fly.py's least-squares
    mixer then leaks every yaw demand into roll (8.6%).  Without authority
    the yaw demand is dropped and the heading is left free; the cascade is
    omnidirectional and no score reads the heading."""
    m, d = env.model, env.data
    env._mj.mj_forward(m, d)
    R = d.xmat[env.root_body].reshape(3, 3)
    mb = m.body_mass[1:]
    com = (mb[:, None] * d.xipos[1:]).sum(0) / float(mb.sum())
    A = np.zeros((4, len(rot)))
    for i, b in enumerate(rot):
        r = R.T @ (d.xipos[b] - com)
        ax_b = R.T @ (d.xmat[b].reshape(3, 3) @ m.jnt_axis[m.body_jntadr[b]])
        A[:, i] = [1.0, r[1], -r[0], -float(np.sign(ax_b[2]))]
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return int(np.linalg.matrix_rank(A, tol=1e-3)) == 4


def body_model(env) -> BodyModel:
    """Read the model constants off an environment.

    Rotor actuators are the ones named ``*_r``, limb strokes ``*_a`` and limb
    feathers ``*_f`` (`core.mjcf`).  Geometry is read from the pose the
    environment is in (after an ``mj_forward``), so it is exact to rounding
    only: build the model where the controller is built.
    """
    mj, m, d = env._mj, env.model, env.data
    names = env.act_names
    irot = [k for k, a in enumerate(names) if a.endswith("_r")]
    rot = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, names[k][:-2] + "_rot") for k in irot]
    spec = env.rotors.spec[0]
    air, water = env.medium.air, env.medium.water
    Ta, Qa = bemt(spec, 600.0, 0.0, 0.0, air.rho, air.mu)
    Tw, Qw = bemt(spec, 30.0, 0.0, 0.0, water.rho, water.mu)
    aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, names[irot[0]])
    yaw_ok = _yaw_authority(env, rot)
    mj.mj_forward(m, d)
    Rr = d.xmat[env.root_body].reshape(3, 3)
    p0 = d.xpos[env.root_body]
    istroke = [k for k, a in enumerate(names) if a.endswith("_a")]
    side = []
    for k in istroke:
        a = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, names[k])
        b = m.jnt_bodyid[m.actuator_trnid[a, 0]]
        side.append(1.0 if float((Rr.T @ (d.xipos[b] - p0))[1]) > 0 else -1.0)
    R_i = np.zeros(9)
    mj.mju_quat2Mat(R_i, np.asarray(m.body_iquat[env.root_body], float))
    return BodyModel(
        rot=rot, irot=irot, istroke=istroke,
        ifeather=[k for k, a in enumerate(names) if a.endswith("_f")], side=side,
        n_act=len(names),
        kt_a=Ta / 600.0 ** 2, kq_a=Qa / 600.0 ** 2, kt_w=Tw / 30.0 ** 2, kq_w=Qw / 30.0 ** 2,
        rho_w=float(water.rho), rotor_radius=float(spec.radius),
        om_max=float(env.cpg.hi[irot[0]]),
        kv=float(m.actuator_gainprm[aid, 0]), tau_max=float(m.actuator_forcerange[aid, 1]),
        mass=float(env.p.mass), buoyant_volume=float(env.p.buoyant_volume), yaw_ok=yaw_ok,
        hubs_body=np.array([Rr.T @ (d.xipos[b] - p0) for b in rot]),
        launch_speed=float(env.launch_speed),
        R_i=R_i.reshape(3, 3), ipos=np.array(m.body_ipos[env.root_body], float),
    )


class InnerLoop:
    """``act(u) -> actuator targets``, every simulation step."""

    def __init__(self, env, model: BodyModel):
        self.env, self.model = env, model
        #: Scratch for `mj_fullM`, which fills it whole on every call.
        self._M = np.zeros((env.model.nv, env.model.nv))

    def act(self, u) -> np.ndarray:
        md, env, d = self.model, self.env, self.env.data
        out = np.zeros(md.n_act)
        # The limbs: a pose, or the gait on the simulation clock.
        drive, turn = float(u[U_DRIVE]), float(u[U_TURN])
        if drive == 0.0 and turn == 0.0:
            out[md.istroke] = GAIT["rest"] + float(u[U_POSE]) * (GAIT["stand"] - GAIT["rest"])
        else:
            phi = 2.0 * math.pi * GAIT["f"] * float(d.time)
            st = math.tanh(GAIT["sharp"] * math.sin(phi))
            out[md.istroke] = (0.5 * (GAIT["down"] + GAIT["up"])
                               + 0.5 * (GAIT["down"] - GAIT["up"]) * st)
            gains = np.array([np.clip(drive - s * turn, -1.0, 1.0) for s in md.side])
            out[md.ifeather] = -gains * GAIT["sweep"] * math.cos(phi)
        if u[U_ON] <= 0.5:
            return out
        # The rotors.  Every position below is relative to the root origin.
        p = d.xpos[env.root_body]
        R = d.xmat[env.root_body].reshape(3, 3).copy()
        # Not `d.subtree_com`: in the rollout loop it is not from the same pass as
        # `xipos` (E0 probes/p05: the lever arms came out 0.058 m off at 6.6 m/s,
        # about two steps of travel, and the mixer pitched the machine over).
        mb = env.model.body_mass[1:]
        com = (mb[:, None] * d.xipos[1:]).sum(0) / float(mb.sum()) - p
        v = d.qvel[:3].copy()
        w = d.qvel[3:6].copy()                          # body frame
        hubs = d.xipos[md.rot] - p
        s = np.asarray(env.medium.properties(d.xipos[md.rot], np.full(md.n, 0.01),
                                             float(d.time))[2], float)
        sub = float(env.solver.diag.mean_submerged)
        water = bool(s.min() > 0.99)
        out[md.irot] = self._cascade(R, com, v, w, hubs, s, sub, u[U_ACC:U_ACC + 3],
                                     float(u[U_TILT]), water=water,
                                     allow_down=bool(water and u[U_DOWN] > 0.5))
        return out

    def _cascade(self, R, com, v, w, hubs, s, sub, a_fb, tilt_deg, *, water, allow_down):
        """Rotor speed commands that produce acceleration ``a_fb``: a thrust
        direction within the tilt limit, an attitude PD toward it, and a
        mixer that serves the torques before the collective."""
        md, m, d = self.model, self.env.model, self.env.data
        self.env._mj.mj_fullM(m, d, self._M)
        Mt = np.diag(self._M[0:3, 0:3]).copy()
        c = R.T @ com                                   # COM in the root frame
        mt = float(Mt.mean())
        J = self._M[3:6, 3:6] - mt * (float(c @ c) * np.eye(3) - np.outer(c, c))
        w_net = md.mass * GRAVITY - md.rho_w * GRAVITY * md.buoyant_volume * sub
        f = Mt * np.asarray(a_fb, float)
        f[2] += w_net
        if water:
            # Under water the weight is a few percent of the dry weight, so the
            # thrust vector is pointed where the demand points: sideways to
            # travel, and -- while the target is far below (`allow_down`) -- down.
            w0 = max(w_net, 0.5)
            if not allow_down:
                f[2] = max(f[2], 0.05 * w0)
                if tilt_deg > 0.0:
                    lim = math.tan(math.radians(tilt_deg))
                    h = float(np.linalg.norm(f[:2]))
                    if h > f[2] * lim:
                        f[:2] *= f[2] * lim / h
            fdir = f.copy()
            fn = float(np.linalg.norm(f))
            if fn < 0.5 * w0:
                fdir[2] += 0.5 * w0 - fn            # a small demand leans back upright
            z_des = fdir / max(float(np.linalg.norm(fdir)), 1e-9)
        else:
            lim = math.tan(math.radians(tilt_deg))
            h = float(np.linalg.norm(f[:2]))
            f[2] = max(f[2], 0.02 * md.mass * GRAVITY)
            if h > f[2] * lim:
                f[:2] *= f[2] * lim / h
            z_des = f / np.linalg.norm(f)
        if md.yaw_ok:
            # The action carries no heading: a body with yaw authority is held
            # at world heading 0.  Not exercised by the anchor body.
            x_c = np.array([1.0, 0.0, 0.0])
        else:
            # No yaw authority (see `_yaw_authority`): ask for the attitude that
            # points the thrust without twisting, whatever the heading is.
            x_c = R[:, 0] if abs(float(R[:, 0] @ z_des)) < 0.95 else R[:, 1]
        y_d = np.cross(z_des, x_c)
        y_d /= max(np.linalg.norm(y_d), 1e-9)
        x_d = np.cross(y_d, z_des)
        Rd = np.stack([x_d, y_d, z_des], 1)
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        e_r = np.array([E[2, 1], E[0, 2], E[1, 0]])
        # fly.py scales one PD by the mean diagonal of the inertia about the root
        # *origin* (the hull's tail end), which on the reference quad is 3.9x its
        # pitch inertia about the centre of mass (E0 probes/p03): an effective 13.9
        # rad/s, zeta 1.6 in pitch.  Written here per axis on the inertia about
        # the centre of mass, so a longer hull does not change the loop.
        g = ATTITUDE
        kp = np.array([g["wn"] ** 2, g["wn"] ** 2, g["wn_yaw"] ** 2])
        kd = np.array([2.0 * g["zeta"] * g["wn"], 2.0 * g["zeta"] * g["wn"],
                       2.0 * g["zeta_yaw"] * g["wn_yaw"]])
        if not md.yaw_ok:
            kp[2] = kd[2] = 0.0
        tau = J @ (-kp * e_r - kd * w)
        thrust = max(float(f @ R[:, 2]), 0.0)
        # per-rotor coefficients of the medium each rotor is in
        k = (1.0 - s) * md.kt_a + s * md.kt_w
        kq = (1.0 - s) * md.kq_a + s * md.kq_w
        kap = kq / k
        A = np.zeros((4, md.n))
        for i, b in enumerate(md.rot):
            r = R.T @ (hubs[i] - com)
            ax_b = R.T @ (d.xmat[b].reshape(3, 3) @ m.jnt_axis[m.body_jntadr[b]])
            sgn = float(np.sign(ax_b[2]))
            A[:, i] = [1.0, r[1], -r[0], -kap[i] * sgn]
        t_max = np.minimum(0.9 * md.tau_max / kap, k * md.om_max ** 2)
        # torques first, then as much collective as the rotors have left: the
        # same least-squares answer as fly.py's while nothing saturates
        if md.yaw_ok:
            dlt = np.linalg.lstsq(A, np.array([0.0, tau[0], tau[1], tau[2]]), rcond=None)[0]
        else:
            dlt = np.linalg.lstsq(A[:3], np.array([0.0, tau[0], tau[1]]), rcond=None)[0]
        span = float(dlt.max() - dlt.min())
        if span > float(t_max.min()):
            dlt *= float(t_max.min()) / span
        col = float(np.clip(thrust / md.n, -dlt.min(), float(np.min(t_max - dlt))))
        t_i = np.clip(col + dlt, 0.0, t_max)
        return self._rotor_cmd(R, v, w, hubs, t_i, k, kq)

    def _rotor_cmd(self, R, v, w, hubs, t_i, k, kq):
        """Rotor speed commands for thrusts ``t_i``: the static law, corrected
        for the flow each rotor is moving through, plus the servo's droop.

        Under water a rotor turns at tens of rad/s, so a 0.6 m/s climb is
        J = 0.17 and takes two thirds of its thrust away (E0: the first exit
        from the water stalled at exactly its pitch speed)."""
        md = self.model
        w_world = R @ w
        om = np.sqrt(t_i / k)
        out = np.zeros(md.n)
        for i in range(md.n):
            v_hub = v + np.cross(w_world, hubs[i])
            v_ax = float(v_hub @ R[:, 2])
            o = float(om[i])
            if v_ax > 0.0 and t_i[i] > 0.0 and v_ax / max(o * md.rotor_radius, 1e-9) > J_FLAT:
                a_ = k[i] * J_ZERO
                b_ = k[i] * v_ax / md.rotor_radius
                c_ = t_i[i] * (J_ZERO - J_FLAT)
                o = (b_ + math.sqrt(b_ * b_ + 4.0 * a_ * c_)) / (2.0 * a_)
            out[i] = o + kq[i] * o * o / md.kv        # kv (om_cmd - om) = kq om^2
        return np.clip(out, 0.0, md.om_max)
