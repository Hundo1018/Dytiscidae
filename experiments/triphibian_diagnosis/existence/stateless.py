#!/usr/bin/env python3
"""E0s: E0's hand law split into a stabilising inner loop and a stateless outer
law that reads only an observation.  Spec: PREREG_stateless.md (frozen).

    o = observe(env, commanded, prev_memory, since)      # the only thing `teach` sees
    u = Teacher(C).teach(o)                              # pure; 25 Hz; held between decisions
    ctrl = Inner(env, C).act(u)                          # every simulation step

`triquad`, `landgait` and `nulls` are imported, never edited.  The inner loop
is `triquad.Hand._cascade` and `_rotor_cmd` themselves (bound to `Inner`), plus
`landgait`'s limb gait written as a function of the clock.

    cd /home/hundo/Projects/Dytiscidae/dytiscidae; export MUJOCO_GL=disable PYTHONPATH=.
    E=$PWD/experiments/triphibian_diagnosis/existence
    .venv/bin/python $E/stateless.py tests
    .venv/bin/python $E/stateless.py run --kind mission --arm teach --seeds 0-31 --out $E/results_stateless.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mujoco  # noqa: E402
import landgait as lg  # noqa: E402
import nulls as nl  # noqa: E402
import triquad as tq  # noqa: E402
import dytiscidae.envs.transitions as T  # noqa: E402
from dytiscidae.core.mjcf import BEACH_SLOPE, beach_surface_z  # noqa: E402
from dytiscidae.envs import mission as M  # noqa: E402
from dytiscidae.envs.evaluate import Controller  # noqa: E402
from dytiscidae.envs.tasks import TASK_SPEED  # noqa: E402
from dytiscidae.envs.triphibian import DOMAIN_CYCLE, Domain  # noqa: E402
from dytiscidae.physics.medium import GRAVITY  # noqa: E402

BODY = "L1"
K = 1                      # memory channels
N_ENV = 33                 # TriphibianEnv.OBS_DIM
# observation layout: see `observe`
I_HEAD, I_CLR, I_DW, I_BEAR, I_WET, I_SINCE, I_AIRERR, I_MEM = 33, 35, 36, 37, 39, 43, 44, 45
N_OBS = I_MEM + K
# action layout
U_ACC, U_TILT, U_DOWN, U_ON, U_DRIVE, U_TURN, U_POSE, U_MEM = 0, 3, 4, 5, 6, 7, 8, 9
N_ACT = U_MEM + K
SINCE_CLIP = 10.0


def _waterline() -> float:
    lo, hi = 0.0, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if beach_surface_z(mid) < 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


X_WATERLINE = _waterline()          # 7.803: where the ramp meets the still sea


# --------------------------------------------------------------------------
# Constants fixed at construction (the model, the map, the gains)
# --------------------------------------------------------------------------


def _quat2mat(q) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, float))
    return out.reshape(3, 3)


def constants(env) -> dict:
    """Everything either object knows that is not in ``o``: the body's model
    (taken from an E0 `Hand` built once and uninstalled), the map, the gains."""
    hand = lg.LegHand(lg._RotorView(env), record=False)
    env.cpg.command = hand._orig                       # uninstall: only its numbers are kept
    m, d = env.model, env.data
    names = env.act_names
    mujoco.mj_forward(m, d)
    Rr = d.xmat[env.root_body].reshape(3, 3)
    p0 = d.xpos[env.root_body]
    ia = [k for k, a in enumerate(names) if a.endswith("_a")]
    side = []
    for k in ia:
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, names[k])
        b = m.jnt_bodyid[m.actuator_trnid[aid, 0]]
        side.append(1.0 if float((Rr.T @ (d.xipos[b] - p0))[1]) > 0 else -1.0)
    return dict(
        g=dict(hand.g), gait=dict(lg.GAIT), token=hand.token, orig=hand._orig,
        rot=list(hand.rot), n=hand.n, kt_a=hand.kt_a, kq_a=hand.kq_a, kt_w=hand.kt_w,
        kq_w=hand.kq_w, rho_w=hand.rho_w, om_max=hand.om_max, kv=hand.kv, tau_max=hand.tau_max,
        mass=hand.mass, vol_b=hand.vol_b, yaw_ok=hand.yaw_ok,
        irot=[k for k, a in enumerate(names) if a.endswith("_r")], ia=ia,
        i_f=[k for k, a in enumerate(names) if a.endswith("_f")], side=side, n_all=len(names),
        hubs_body=np.array([Rr.T @ (d.xipos[b] - p0) for b in hand.rot]),   # rotor hubs, root frame
        v_launch=float(env.launch_speed),
        # `TriphibianEnv.body_twist` (the observation's velocity and rates) is in the
        # root body's *inertial* frame, at its centre of mass (mj_objectVelocity on
        # mjOBJ_BODY): the constant rotation and offset back to the root frame
        R_i=_quat2mat(m.body_iquat[env.root_body]), ipos=np.array(m.body_ipos[env.root_body], float),
        # the map, relative to the waterline along the shore normal (+ inland)
        dw_station=float(tq.P_STATION[0] - X_WATERLINE), dw_shelf=float(tq.X_SHELF - 0.4 - X_WATERLINE),
        dw_deep=float(-1.0 - X_WATERLINE), dw_land=float(tq.X_LAND - X_WATERLINE),
        dw_ashore=float(tq.X_ASHORE - X_WATERLINE), slope=float(BEACH_SLOPE),
        dive_thr=0.10,
        # no position reference under water: the cruise command is over-asked instead
        cruise_ff=float(os.environ.get("E0S_CRUISE_FF", "3.0")),
    )


# --------------------------------------------------------------------------
# The observation
# --------------------------------------------------------------------------

CHANNELS = (
    [("env.observation[%d]" % i, s) for i, s in enumerate(
        ["body-frame velocity x /5 (clipped +-3): speed log / flow sensor"] * 1
        + ["body-frame velocity y /5", "body-frame velocity z /5",
           "body rate x /4: gyro", "body rate y /4", "body rate z /4",
           "gravity in body frame x: accelerometer", "gravity y", "gravity z",
           "tanh(depth/5): pressure sensor", "mean wetness of the body's panels",
           "command one-hot: air", "command: water", "command: land",
           "tanh((depth - commanded depth)/3)", "ground contact switch", "battery fraction",
           "mean stroke of actuated joints", "mean stroke rate",
           "task: cruise", "task: hold/stop", "task: commanded speed ratio",
           "task: cos(heading error)", "task: sin(heading error)", "task: tanh(height lost/3) (air)"]
        + ["morphology %d" % j for j in range(8)])]
    + [("heading sin", "compass"), ("heading cos", "compass"),
       ("clearance, m", "height of the lowest point over what is underneath: altimeter / sonar"),
       ("distance to the waterline along the shore normal, m (+ inland)", "position fix reduced to one number"),
       ("shore-normal bearing in the yaw frame, sin", "map + compass"), ("shore-normal bearing, cos", "map + compass")]
    + [("rotor %d wetness" % i, "wet switch at the hub") for i in range(4)]
    + [("seconds since the commanded medium changed (clipped 10)", "a clock restarted by the command"),
       ("air task height error, m (reference - clearance; 0 off an air task)", "the task's own reference altitude")]
    + [("memory %d" % i, "this law's memory output at the previous decision; 0 at the first") for i in range(K)]
)
assert len(CHANNELS) == N_OBS


def observe(env, commanded, prev_memory, since: float) -> np.ndarray:
    """The whole of what the outer law is given."""
    o = np.zeros(N_OBS)
    o[:N_ENV] = env.observation(commanded)
    d = env.data
    R = d.xmat[env.root_body].reshape(3, 3)
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    o[I_HEAD], o[I_HEAD + 1] = math.sin(yaw), math.cos(yaw)
    clr = float(env.clearance())
    o[I_CLR] = clr
    o[I_DW] = float(d.xpos[env.root_body][0]) - X_WATERLINE
    o[I_BEAR], o[I_BEAR + 1] = math.sin(-yaw), math.cos(-yaw)        # the normal is world +x
    hubs = d.xipos[env.rotors.body_arr]
    o[I_WET:I_WET + 4] = np.asarray(env.medium.properties(hubs, np.full(len(hubs), 0.01), d.time)[2], float)
    o[I_SINCE] = min(float(since), SINCE_CLIP)
    task = env._active_task
    o[I_AIRERR] = float(env._height_ref - clr) if (task is not None and task.domain == "air") else 0.0
    o[I_MEM:] = np.asarray(prev_memory, float)
    return o


# --------------------------------------------------------------------------
# The outer law: a pure function of o
# --------------------------------------------------------------------------


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Teacher:
    """``teach(o) -> u``.  No attribute is assigned after construction."""

    def __init__(self, C: dict, every_step: bool = False):
        self.C = C

    # -- pieces (all pure)
    @staticmethod
    def _vert(e, vz, kz, kvz, v_dn, v_up, a_dec=None, vz_ff=0.0):
        if a_dec is None:
            v = kz * e
        else:
            e_lin = a_dec / (kz * kz)
            v = kz * e if abs(e) <= e_lin else math.copysign(math.sqrt(2.0 * a_dec * (abs(e) - 0.5 * e_lin)), e)
        return kvz * (float(np.clip(v, -v_dn, v_up)) + vz_ff - vz)

    @staticmethod
    def _approach(along, kpos, vmax, a_brake=None):
        dist = abs(along)
        sp = kpos * dist
        if a_brake is not None and dist > a_brake / (kpos * kpos):
            sp = math.sqrt(2.0 * a_brake * (dist - 0.5 * a_brake / (kpos * kpos)))
        return math.copysign(min(sp, vmax), along)

    def _limbs(self, e):
        """`landgait.LegLaw`'s steering: drive and turn from the heading error."""
        g = self.C["gait"]
        direction = 1.0
        if abs(e) > 0.5 * math.pi:
            e, direction = _wrap(e - math.pi), -1.0
        w = float(np.clip(g["k_turn"] * e, -1.0, 1.0))
        v = direction * float(np.clip(1.0 - abs(e) / g["e_fwd"], 0.0, 1.0))
        return v, w

    def teach(self, o) -> np.ndarray:
        C, g = self.C, self.C["g"]
        o = np.asarray(o, float)
        u = np.zeros(N_ACT)
        cmd = o[11:14]
        if cmd.max() <= 0.5:
            return u
        M_ = int(np.argmax(cmd))                               # 0 air, 1 water, 2 land
        # -- what the channels say
        yaw = math.atan2(o[I_HEAD], o[I_HEAD + 1])
        gb = o[6:9]
        pitch = -math.asin(float(np.clip(gb[0] * -1.0, -1.0, 1.0)))     # third row of R is -gravity_body
        roll = math.atan2(float(-gb[1]), float(-gb[2]))
        cy, sy, cp, sp_, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
        R = np.array([[cy * cp, cy * sp_ * sr - sy * cr, cy * sp_ * cr + sy * sr],
                      [sy * cp, sy * sp_ * sr + cy * cr, sy * sp_ * cr - cy * sr],
                      [-sp_, cp * sr, cp * cr]])
        w_b = C["R_i"] @ (4.0 * o[3:6])
        v = R @ (C["R_i"] @ (5.0 * o[0:3]) - np.cross(w_b, C["ipos"]))      # the root origin, world frame
        w_mag = float(np.linalg.norm(w_b))
        tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))
        depth = 5.0 * math.atanh(float(np.clip(o[9], -0.999999999999, 0.999999999999)))
        contact = o[15] > 0.5
        s = o[I_WET:I_WET + 4]
        wet_all, wet_any = s.min() > 0.99, s.max() > 0.01
        clr, dw = float(o[I_CLR]), float(o[I_DW])
        nb = yaw + math.atan2(o[I_BEAR], o[I_BEAR + 1])         # the shore normal, world frame
        n = np.array([math.cos(nb), math.sin(nb)])
        vn = float(v[:2] @ n)
        vt = v[:2] - vn * n
        spd = float(np.linalg.norm(v[:2]))
        stand = 1.0
        u[U_POSE] = stand

        def out(a_xy, a_z, tilt_lim, down=0.0, mem=0.0):
            u[U_ACC:U_ACC + 2], u[U_ACC + 2] = a_xy, a_z
            u[U_TILT], u[U_DOWN], u[U_ON] = tilt_lim, down, 1.0
            u[U_MEM] = mem
            return u

        cruise, hold = o[19] > 0.5, o[20] > 0.5
        if cruise or hold:                                     # a media segment
            head = yaw + math.atan2(o[23], o[22])
            hvec = np.array([math.cos(head), math.sin(head)])
            if M_ == 0:
                a_xy = g["kv_a"] * (C["v_launch"] * hvec - v[:2])
                return out(a_xy, self._vert(float(o[I_AIRERR]), v[2], g["kz_a"], g["kvz_a"], 1e9, 1e9), g["tilt_a"])
            if M_ == 1:
                if not (s.min() > 0.5 and depth > 0.0):
                    return u
                e = 3.0 * math.atanh(float(np.clip(o[14], -0.999999999999, 0.999999999999)))   # depth - commanded
                a_z = self._vert(e, v[2], g["kz_w"], g["kvz_w"], g["vsink"], g["vrise"], g["adec_w"])
                down = 1.0 if -e > C["dive_thr"] else 0.0
                if hold:
                    return out(g["kv_w"] * (0.0 - v[:2]), a_z, g["tilt_hold_w"], down)
                v_cmd = g["cruise_gain_w"] * float(o[21]) * TASK_SPEED["water"] * hvec
                return out(g["kv_w"] * (C["cruise_ff"] * v_cmd - v[:2]), a_z, 0.0, down)
            if hold:
                u[U_POSE] = 0.0
                return u
            u[U_DRIVE], u[U_TURN] = self._limbs(_wrap(head - yaw))
            return u
        # -- no task: crossings and the mission
        hub_top = -depth + float((C["hubs_body"] @ R[2, :]).max())

        def exit_water(z_abs):
            e = z_abs + depth
            if wet_all:
                go = o[I_MEM] > 0.5 or (hub_top > -(g["stage_depth"] + 0.05) and tilt < g["stage_tilt_deg"]
                                        and w_mag < g["stage_rate"])
                if not go:
                    a_z = self._vert(-g["stage_depth"] - hub_top, v[2], g["kz_w"], g["kvz_w"], g["vsink"], 1.5, g["adec_w"])
                    return out(g["kv_w"] * (0.0 - v[:2]), a_z, 20.0)
                a_z = self._vert(e, v[2], g["kz_w"], g["kvz_w"], g["vsink"], 3.0, g["adec_w"])
                return out(g["kv_w"] * (0.0 - v[:2]), a_z, 10.0, mem=1.0)
            return out(g["kv_a"] * (0.0 - v[:2]), self._vert(e, v[2], g["kz_a"], g["kvz_a"], 2.0, 3.0), 10.0, mem=1.0)

        on_beach = dw > 0.0
        vz_ff = C["slope"] * vn if on_beach else 0.0
        if M_ == 2:                                            # LAND
            dry = depth <= 0.0 and s.max() < 0.01
            if dry and dw >= C["dw_ashore"] and (contact or clr < 0.12) and float(np.linalg.norm(v)) < 1.5:
                u[U_DRIVE], u[U_TURN] = self._limbs(_wrap(nb - yaw))      # landed: walk inland
                return u
            if not dry:
                return exit_water(0.9)
            along = max(C["dw_land"] - dw, 0.0)
            if not (dw >= C["dw_ashore"] + 0.3 and along < 1.0 and spd < 1.2):
                v_cmd = self._approach(along, g["kpos_nav"], g["v_tr"], g["abrake"]) * n
                a_z = self._vert(g["h_tr"] - clr, v[2], g["kz_a"], g["kvz_a"], 1.5, 3.0, vz_ff=vz_ff)
                return out(g["kv_nav"] * (v_cmd - v[:2]), a_z, g["tilt_a"])
            v_cmd = self._approach(along, g["kpos_nav"], 1.0, g["abrake"]) * n
            return out(g["kv_nav"] * (v_cmd - v[:2]), self._vert(-1.0, v[2], g["kz_a"], g["kvz_a"], 1.0, 1.0), g["tilt_a"])
        if M_ == 0:                                            # AIR
            if depth > 0.0 or wet_any:
                return exit_water(g["h_air"])
            if clr < 1.0:
                a_z = self._vert(g["h_air"] - clr, v[2], g["kz_a"], g["kvz_a"], 1.0, 3.0)
                return out(g["kv_a"] * (0.0 - v[:2]), a_z, 20.0)
            along = C["dw_station"] - dw
            v_nav = float(np.clip(max(spd, g["v_nav"]), g["v_nav"], 9.0))
            v_cmd = math.copysign(min(v_nav, g["kpos"] * abs(along)), along) * n
            target = float(np.clip(clr, g["h_air"], g["h_air"] + abs(along)))
            a_z = self._vert(target - clr, v[2], g["kz_a"], g["kvz_a"], 6.0, 3.0, vz_ff=vz_ff)
            return out(g["kv_a"] * (v_cmd - v[:2]), a_z, g["tilt_a"])
        # WATER
        if wet_all and depth > 0.0:
            if dw >= C["dw_shelf"]:
                return u                                       # rest on the shelf
            v_cmd = self._approach(C["dw_station"] - dw, g["kpos_w"], 0.5) * n
            a_z = self._vert(depth - g["depth_nav"], v[2], g["kz_w"], g["kvz_w"], g["vsink"], g["vrise"], g["adec_w"])
            return out(g["kv_w"] * (v_cmd - v[:2]), a_z, 45.0)
        if wet_any or depth > 0.0:
            return u                                           # sink
        if dw > C["dw_deep"]:
            v_cmd = self._approach(C["dw_station"] - dw, g["kpos_nav"], g["v_tr"], g["abrake"]) * n
            a_z = self._vert(1.5 - clr, v[2], g["kz_a"], g["kvz_a"], 2.0, 3.0)
            return out(g["kv_nav"] * (v_cmd - v[:2]), a_z, g["tilt_a"])
        if clr < g["cut_clear"]:
            return u                                           # rotors stopped before the surface
        return out(g["kv_a"] * (0.0 - v[:2]), self._vert(-5.0, v[2], g["kz_a"], g["kvz_a"], g["vdesc"], 1.0), g["tilt_a"])


# --------------------------------------------------------------------------
# The inner loop
# --------------------------------------------------------------------------


class Inner:
    """``act(u) -> actuator targets``, every simulation step.  Reads attitude,
    body rates, velocity, each rotor's wetness, the submerged fraction and the
    model; not position (only differences of positions on the body), not the
    command, not the task; the clock only for the gait's phase.  The mixing is
    E0's own code: `triquad.Hand._cascade` and `_rotor_cmd`, bound here."""

    _cascade = tq.Hand._cascade
    _rotor_cmd = tq.Hand._rotor_cmd
    J_FLAT, J_ZERO = tq.Hand.J_FLAT, tq.Hand.J_ZERO

    def __init__(self, env, C: dict):
        self.env, self.C = env, C
        for k in ("g", "rot", "n", "kt_a", "kq_a", "kt_w", "kq_w", "rho_w", "om_max", "kv",
                  "tau_max", "mass", "vol_b", "yaw_ok"):
            setattr(self, k, C[k])
        self.M = np.zeros((env.model.nv, env.model.nv))       # scratch, filled whole by mj_fullM each call

    def act(self, u) -> np.ndarray:
        C, env, d = self.C, self.env, self.env.data
        out = np.zeros(C["n_all"])
        # limbs
        gt = C["gait"]
        n = len(C["ia"])
        drive, turn = float(u[U_DRIVE]), float(u[U_TURN])
        if drive == 0.0 and turn == 0.0:
            out[C["ia"]] = gt["rest"] + float(u[U_POSE]) * (gt["stand"] - gt["rest"])
        else:
            phi = 2.0 * math.pi * gt["f"] * float(d.time)
            st = math.tanh(gt["sharp"] * math.sin(phi))
            out[C["ia"]] = 0.5 * (gt["down"] + gt["up"]) + 0.5 * (gt["down"] - gt["up"]) * st
            gains = np.array([np.clip(drive - s * turn, -1.0, 1.0) for s in C["side"]])
            out[C["i_f"]] = -gains * gt["sweep"] * math.cos(phi)
        if u[U_ON] <= 0.5:
            return out
        # rotors
        S = type("S", (), {})()
        p = d.xpos[env.root_body]
        S.R = d.xmat[env.root_body].reshape(3, 3).copy()
        S.p = np.zeros(3)
        mb = env.model.body_mass[1:]
        S.com = (mb[:, None] * d.xipos[1:]).sum(0) / float(mb.sum()) - p
        S.v = d.qvel[:3].copy()
        S.w = d.qvel[3:6].copy()
        S.hubs = d.xipos[self.rot] - p
        S.s = np.asarray(env.medium.properties(d.xipos[self.rot], np.full(self.n, 0.01), float(d.time))[2], float)
        S.sub = float(env.solver.diag.mean_submerged)
        water = bool(S.s.min() > 0.99)
        out[C["irot"]] = self._cascade(S, u[U_ACC:U_ACC + 3], 0.0, float(u[U_TILT]), water=water,
                                       allow_down=bool(water and u[U_DOWN] > 0.5))
        return out


# --------------------------------------------------------------------------
# The harness: where the 25 Hz hold and the memory feedback live
# --------------------------------------------------------------------------

U_RANGE = dict(acc=8.0, tilt=45.0)        # the range the teacher's outputs span (null arm "const")


class Split:
    """The environment's side of the interface.  It, not the law, holds the
    previous memory output, the held `u` and the command clock."""

    def __init__(self, env, arm="teach", seed=0, fresh_teach=False, fresh_inner=False):
        self.env, self.arm = env, arm
        self.C = constants(env)
        self.teacher, self.inner = Teacher(self.C), Inner(env, self.C)
        self.fresh_teach, self.fresh_inner = fresh_teach, fresh_inner
        self.const_u = None
        if arm == "const":
            r = np.random.default_rng([int(seed), 0x434F4E53])
            cu = np.zeros(N_ACT)
            cu[U_ACC:U_ACC + 3] = r.uniform(-U_RANGE["acc"], U_RANGE["acc"], 3)
            cu[U_TILT] = r.uniform(0.0, U_RANGE["tilt"])
            cu[U_DOWN], cu[U_ON] = float(r.random() < 0.5), float(r.random() < 0.8)
            cu[U_DRIVE], cu[U_TURN], cu[U_POSE] = r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(0, 1)
            self.const_u = cu
        self.log = None
        self._clear()
        self.policy, self.basis = self, tq.HandBasis(self.C["token"])
        self.ctrl = Controller(params=self.C["token"], policy=self, bases={"air": self.basis, "water": self.basis})
        self._orig = self.C["orig"]
        env.cpg.command = self.command

    def _clear(self):
        self.u = np.zeros(N_ACT)
        self.mem = np.zeros(K)
        self.cmd = self.cmd_first = None
        self.t_change = 0.0
        self.t_last = -1.0
        self.n_dec = 0

    def decide(self):
        env = self.env
        t = float(env.data.time)
        cmd = self.cmd_first if self.arm == "frozen" else self.cmd
        o = observe(env, cmd, np.zeros(K) if self.arm == "nomem" else self.mem, t - self.t_change)
        if self.arm == "zero":
            u = np.zeros(N_ACT)
        elif self.arm == "const":
            u = self.const_u.copy()
        else:
            u = (Teacher(self.C) if self.fresh_teach else self.teacher).teach(o)
        self.u, self.mem = u, u[U_MEM:].copy()
        self.n_dec += 1
        if self.log is not None:
            self.log.append((t, o.copy(), u.copy()))

    def act(self, obs):                                        # the evaluator's 25 Hz call
        t = float(self.env.data.time)
        if t <= 0.0 or t < self.t_last:
            self._clear()
        c = np.asarray(obs[11:14], float)
        cmd = DOMAIN_CYCLE[int(np.argmax(c))] if c.max() > 0.5 else None
        if cmd is not self.cmd:
            self.cmd, self.t_change = cmd, t
            if self.cmd_first is None:
                self.cmd_first = cmd
        self.t_last = t
        if self.arm != "step":
            self.decide()
        return np.zeros(1)

    def command(self, params, t):
        if params is not self.C["token"]:
            return self._orig(params, t)
        if self.arm == "step" and self.cmd is not None:
            self.decide()
        return (Inner(self.env, self.C) if self.fresh_inner else self.inner).act(self.u)

    def close(self):
        self.env.cpg.command = self._orig


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------

_BAD = tq._BAD


def row_mission(arm, start, seed, **kw) -> dict:
    t0 = time.time()
    _, env, _ = tq.make_env(BODY, seed)
    sp = Split(env, arm, seed, **kw)
    row = {"kind": "mission", "body": BODY, "arm": arm, "start": start, "seed": seed}
    try:
        read = M.evaluate_mission(env, sp.ctrl, Domain(start), seed)
        res = read.result
        row.update(legs=[[d.value, round(s, 3)] for d, s in read.schedule], held=read.held,
                   shares=[round(x, 3) for x in read.shares],
                   supports=[None if x is None else round(x, 3) for x in read.supports],
                   completed=bool(read.completed),
                   transitions=[read.transitions_completed, read.transitions_commanded],
                   failure=read.failure, bad_qacc=int(env.data.warning[_BAD].number),
                   energy_wh=float(res.energy_wh), total_time=float(res.total_time),
                   end_qpos=[float(x) for x in env.data.qpos[:7]], decisions=sp.n_dec)
    except Exception as exc:
        row["failure"] = f"exception: {exc!r}"
        row["traceback"] = traceback.format_exc()[-1200:]
    finally:
        sp.close()
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def rows_segments(arm, seed, **kw) -> list:
    """air, water, land on one environment, as `evaluate_tier1` orders them."""
    _, env, _ = tq.make_env(BODY, seed)
    rows = []
    for item in ("air", "water", "land"):
        t0 = time.time()
        sp = Split(env, arm, seed, **kw)
        rec = lg.Recorder(env, None)
        try:
            row = tq.run_seg_item(env, sp.ctrl, Domain(item), seed)
            row.update(rec.summary() if item == "land" else {})
        except Exception as exc:
            row = {"score": None, "failure": f"exception: {exc!r}", "traceback": traceback.format_exc()[-1200:]}
        finally:
            rec.close()
            sp.close()
        rows.append({"kind": "segment", "body": BODY, "arm": arm, "item": item, "seed": seed, **row,
                     "wall_s": round(time.time() - t0, 1)})
    return rows


def rows_crossings(arm, entry, **kw) -> list:
    _, env, _ = tq.make_env(BODY, entry)
    rows = []
    for kind in tq.CROSSINGS:
        t0 = time.time()
        sp = Split(env, arm, entry, **kw)
        try:
            nl.set_entry_state(entry)
            row = tq.run_cross_item(env, sp.ctrl, kind)
        except Exception as exc:
            row = {"score": None, "failure": f"exception: {exc!r}", "traceback": traceback.format_exc()[-1200:]}
        finally:
            T.transition_scatter_seed = nl._TSS
            sp.close()
        rows.append({"kind": "crossing", "body": BODY, "arm": arm, "item": kind, "seed": entry,
                     "gates": os.environ.get("DYTISCIDAE_CROSSING_GATES", ""), **row,
                     "wall_s": round(time.time() - t0, 1)})
    return rows


def brief(r) -> str:
    keys = ("kind", "arm", "start", "item", "seed", "score", "completed", "held", "shares", "supports",
            "crossed", "hold", "s", "failure", "bad_qacc", "wall_s")
    b = {k: r[k] for k in keys if k in r}
    for k in ("score", "hold", "s"):
        if isinstance(b.get(k), float):
            b[k] = round(b[k], 4)
    return json.dumps(b, default=float)


def cmd_run(a) -> int:
    lo, hi = (int(x) for x in a.seeds.split("-")) if "-" in a.seeds else (int(a.seeds), int(a.seeds))
    k, n = (int(x) for x in a.shard.split("/"))
    out = Path(a.out).resolve()
    jobs = []
    for arm in a.arm.split(","):
        for seed in range(lo, hi + 1):
            if a.kind == "mission":
                jobs += [("m", arm, s, seed) for s in a.starts.split(",")]
            elif a.kind == "segment":
                jobs.append(("s", arm, None, seed))
            else:
                jobs.append(("c", arm, None, seed))
    for j, (kind, arm, start, seed) in enumerate(jobs):
        if j % n != k:
            continue
        rows = ([row_mission(arm, start, seed)] if kind == "m" else
                rows_segments(arm, seed) if kind == "s" else rows_crossings(arm, seed))
        for r in rows:
            if a.note:
                r["note"] = a.note
            with open(out, "a") as f:
                f.write(json.dumps(r, default=float) + "\n")
            print(brief(r), flush=True)
    return 0


# --------------------------------------------------------------------------
# Statelessness tests (spec section 4)
# --------------------------------------------------------------------------


def assigned_after_construction(cls_name: str) -> list:
    """Every ``self.x = ...`` (plain, augmented, annotated, or through setattr)
    in a method of the class other than ``__init__``."""
    tree = ast.parse(Path(__file__).read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for fn in node.body:
                if not isinstance(fn, ast.FunctionDef) or fn.name == "__init__":
                    continue
                for sub in ast.walk(fn):
                    targets = []
                    if isinstance(sub, ast.Assign):
                        targets = sub.targets
                    elif isinstance(sub, (ast.AugAssign, ast.AnnAssign)):
                        targets = [sub.target]
                    elif isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "setattr":
                        found.append(f"{fn.name}: setattr at line {sub.lineno}")
                    for tg in targets:
                        for leaf in ast.walk(tg):
                            if (isinstance(leaf, ast.Attribute) and isinstance(leaf.value, ast.Name)
                                    and leaf.value.id == "self"):
                                found.append(f"{fn.name}: self.{leaf.attr} at line {leaf.lineno}")
    return found


def cmd_tests(a) -> int:
    ok = True
    for cls in ("Teacher", "Inner"):
        lst = assigned_after_construction(cls)
        ok &= not lst
        print(f"TEST c  {cls}: attributes assigned after construction: {lst} -> {'PASS' if not lst else 'FAIL'}")
    keys = ("held", "shares", "supports", "completed", "transitions", "failure", "bad_qacc", "energy_wh",
            "total_time", "end_qpos", "decisions", "legs")
    for start in ("air", "water", "land"):
        ref = row_mission("teach", start, a.seed)
        for name, kw in (("a fresh Teacher at every decision", dict(fresh_teach=True)),
                         ("b fresh Inner at every step", dict(fresh_inner=True))):
            alt = row_mission("teach", start, a.seed, **kw)
            diff = [k for k in keys if ref.get(k) != alt.get(k)]
            ok &= not diff
            print(f"TEST {name}, mission from {start}, seed {a.seed}: {len(keys)} fields, "
                  f"{len(diff)} differ {diff}; completed {ref.get('completed')}, held {ref.get('held')}, "
                  f"decisions {ref.get('decisions')}, energy {ref.get('energy_wh')!r} / {alt.get('energy_wh')!r}, "
                  f"end x {ref.get('end_qpos', [None])[0]!r} / {alt.get('end_qpos', [None])[0]!r} "
                  f"-> {'PASS' if not diff else 'FAIL'}", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kind", default="mission", choices=("mission", "segment", "crossing"))
    r.add_argument("--arm", default="teach")
    r.add_argument("--seeds", default="100-103")
    r.add_argument("--starts", default="air,water,land")
    r.add_argument("--shard", default="0/1")
    r.add_argument("--out", default=str(HERE / "trials_stateless.jsonl"))
    r.add_argument("--note", default="")
    t = sub.add_parser("tests")
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    return {"run": cmd_run, "tests": cmd_tests}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
