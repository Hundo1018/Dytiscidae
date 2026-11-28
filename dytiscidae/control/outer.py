"""The outer law of the split controller: an observation in, an action out.

    o = observe(env, commanded, prev_memory, since)    # all the law is given
    u = Teacher(model, shore).teach(o)                 # pure; 25 Hz
    targets = InnerLoop(env, model).act(u)             # `control.innerloop`, every step

`Teacher` is the hand law that completes the mission on the anchor body
(`core.bodyplans.legged_quad`), written as a function of the observation
alone.  It is E0's law (`experiments/triphibian_diagnosis/existence`) with
every latch but one replaced by something the observation carries, measured
2026-11-11 (E0s, `existence/RESULT_stateless.md`): missions 32, 32, 31 of 32
from the air, the water and the land; every segment bar on 8 of 8; gated
crossings 6, 8, 8, 8 (`air_to_water` loses two holds; the latched law keeps 8).
It is a teacher and a null to measure a learner against, not a learner: no
parameter in it was fitted.

The observation is `TriphibianEnv.observation` unchanged, with thirteen
channels appended (`OBS_CHANNELS`).  The law's only state is ``K`` memory
channels that `SplitController` hands back at the next decision: one, the
"go" of the staged exit from the water.  Without it 79 of 96 missions still
complete, and every loss is on the leg entered from the water toward land.

`SplitController` is the environment's side of the interface.  It, not the
law, holds the previous memory, the action held between decisions and the
clock that restarts when the commanded medium changes; and it carries the
null arms the law is judged against (command frozen at its first value,
memory not fed back, a decision at every step).

Nothing in the search imports this module.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from ..core.mjcf import BEACH_SLOPE, beach_surface_z
from ..envs.evaluate import Controller
from ..envs.tasks import TASK_SPEED
from ..envs.triphibian import DOMAIN_CYCLE
from .innerloop import (
    K, N_ACT, U_ACC, U_DOWN, U_DRIVE, U_MEM, U_ON, U_POSE, U_TILT, U_TURN,
    BodyModel, InnerLoop, body_model,
)

#: `TriphibianEnv.observation`'s length; its layout is that method's.
N_ENV = 33
#: Where the appended channels start: heading (2), clearance, waterline
#: distance, shore-normal bearing (2), rotor wetness (4), the command clock,
#: the air task's height error, memory (K).
I_HEAD, I_CLR, I_DW, I_BEAR, I_WET, I_SINCE, I_AIRERR, I_MEM = 33, 35, 36, 37, 39, 43, 44, 45
N_OBS = I_MEM + K
#: The command clock saturates here, seconds.
SINCE_CLIP = 10.0


def _waterline() -> float:
    lo, hi = 0.0, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if beach_surface_z(mid) < 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


#: World x at which the beach ramp meets the still sea: 7.803 m.
X_WATERLINE = _waterline()

#: Every channel of the observation, and the sensor that would supply it.
OBS_CHANNELS = (
    [("env.observation[%d]" % i, s) for i, s in enumerate(
        ["velocity x /5 (clipped +-3), in the root body's inertial frame: speed log / flow sensor",
         "velocity y /5", "velocity z /5",
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
       ("distance to the waterline along the shore normal, m (+ inland)",
        "position fix reduced to one number"),
       ("shore-normal bearing in the yaw frame, sin", "map + compass"),
       ("shore-normal bearing, cos", "map + compass")]
    + [("rotor %d wetness" % i, "wet switch at the hub") for i in range(4)]
    + [("seconds since the commanded medium changed (clipped 10)",
        "a clock restarted by the command"),
       ("air task height error, m (reference - clearance; 0 off an air task)",
        "the task's own reference altitude")]
    + [("memory %d" % i, "this law's memory output at the previous decision; 0 at the first")
       for i in range(K)]
)
assert len(OBS_CHANNELS) == N_OBS


def observe(env, commanded, prev_memory, since: float) -> np.ndarray:
    """The whole of what the outer law is given.

    The only channel that reads where the machine is on the map is
    ``I_DW``, its distance from the waterline along the shore normal; nothing
    reads its position along the shore.  A body with four rotors is assumed
    (`I_WET`).
    """
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
    o[I_WET:I_WET + 4] = np.asarray(
        env.medium.properties(hubs, np.full(len(hubs), 0.01), d.time)[2], float)
    o[I_SINCE] = min(float(since), SINCE_CLIP)
    task = env._active_task
    o[I_AIRERR] = (float(env._height_ref - clr)
                   if (task is not None and task.domain == "air") else 0.0)
    o[I_MEM:] = np.asarray(prev_memory, float)
    return o


@dataclass(frozen=True)
class ShoreMap:
    """Where the mission's places are, as distances from the waterline along
    the shore normal (+ inland), metres.  The places are E0's
    (`existence/triquad.py`): the station 2 m seaward of the shelf, the
    landing target 1 m up the beach."""

    #: The water and air station, in deep water.
    station: float = float(-2.0 - X_WATERLINE)
    #: 0.4 m seaward of the ramp's seaward end (`mjcf.beach_extent`: 0.086).
    shelf: float = float(0.086 - 0.4 - X_WATERLINE)
    #: Seaward of this the water is deep enough to drop into.
    deep: float = float(-1.0 - X_WATERLINE)
    #: The landing target, and the line past which the machine counts as ashore.
    land: float = float(8.8 - X_WATERLINE)
    ashore: float = float(8.1 - X_WATERLINE)
    slope: float = float(BEACH_SLOPE)


#: The teacher's gains, set by hand: the air cascade is
#: `experiments/rotor/fly.py`'s, the rest E0's and E0b's on tuning seeds 100
#: and up (`existence/trials*.jsonl`).  Two are E0s's (`trials_stateless.jsonl`,
#: seeds 100-107): `dive_thr`, one threshold where E0 had a hysteresis pair,
#: and `cruise_ff`, 2.0 and 3.0 both 8 of 8 on water, 3.0 kept.
GAINS = dict(
    # air
    kv_a=1.2, kz_a=2.0, kvz_a=5.0, tilt_a=35.0, kv_nav=2.5, kpos_nav=1.5, abrake=4.0,
    # water
    kv_w=1.5, kpos_w=1.0, kz_w=4.0, kvz_w=8.0, adec_w=1.5, vsink=1.5, vrise=1.0,
    cruise_gain_w=1.15, tilt_hold_w=30.0,
    stage_depth=0.12, stage_tilt_deg=3.0, stage_rate=0.3,
    # navigation
    kpos=0.8, v_nav=6.0, v_tr=9.0, h_air=1.8, h_tr=0.45, depth_nav=0.8,
    cut_clear=0.45, vdesc=2.5,
    # limbs: turn share per rad of heading error, and the heading error at
    # which forward drive has faded to 0
    k_turn=2.0, e_fwd=0.6,
    # m above the commanded depth at which thrust may point down
    dive_thr=0.10,
    # no position reference under water: the cruise command is over-asked instead
    cruise_ff=3.0,
)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Teacher:
    """``teach(o) -> u``.  No attribute is assigned after construction."""

    def __init__(self, model: BodyModel, shore: ShoreMap):
        self.model, self.shore = model, shore

    # -- pieces (all pure)
    @staticmethod
    def _vert(e, vz, kz, kvz, v_dn, v_up, a_dec=None, vz_ff=0.0):
        """Vertical acceleration toward a height error ``e``: a velocity
        command clipped to the allowed sink and climb, braking at ``a_dec``."""
        if a_dec is None:
            v = kz * e
        else:
            e_lin = a_dec / (kz * kz)
            v = (kz * e if abs(e) <= e_lin
                 else math.copysign(math.sqrt(2.0 * a_dec * (abs(e) - 0.5 * e_lin)), e))
        return kvz * (float(np.clip(v, -v_dn, v_up)) + vz_ff - vz)

    @staticmethod
    def _approach(along, kpos, vmax, a_brake=None):
        """Speed toward a point ``along`` metres ahead, braking into it."""
        dist = abs(along)
        sp = kpos * dist
        if a_brake is not None and dist > a_brake / (kpos * kpos):
            sp = math.sqrt(2.0 * a_brake * (dist - 0.5 * a_brake / (kpos * kpos)))
        return math.copysign(min(sp, vmax), along)

    @staticmethod
    def _limbs(e):
        """Limb drive and turn from the heading error (E0b's steering)."""
        g = GAINS
        direction = 1.0
        if abs(e) > 0.5 * math.pi:
            e, direction = _wrap(e - math.pi), -1.0
        w = float(np.clip(g["k_turn"] * e, -1.0, 1.0))
        v = direction * float(np.clip(1.0 - abs(e) / g["e_fwd"], 0.0, 1.0))
        return v, w

    def teach(self, o) -> np.ndarray:
        md, sh, g = self.model, self.shore, GAINS
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
        cy, sy, cp, sp_, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch),
                                   math.cos(roll), math.sin(roll))
        R = np.array([[cy * cp, cy * sp_ * sr - sy * cr, cy * sp_ * cr + sy * sr],
                      [sy * cp, sy * sp_ * sr + cy * cr, sy * sp_ * cr - cy * sr],
                      [-sp_, cp * sr, cp * cr]])
        w_b = md.R_i @ (4.0 * o[3:6])
        v = R @ (md.R_i @ (5.0 * o[0:3]) - np.cross(w_b, md.ipos))      # the root origin, world frame
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
        spd = float(np.linalg.norm(v[:2]))
        u[U_POSE] = 1.0                                        # stand, unless told to stop

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
                a_xy = g["kv_a"] * (md.launch_speed * hvec - v[:2])
                return out(a_xy, self._vert(float(o[I_AIRERR]), v[2], g["kz_a"], g["kvz_a"],
                                            1e9, 1e9), g["tilt_a"])
            if M_ == 1:
                if not (s.min() > 0.5 and depth > 0.0):
                    return u
                # depth - commanded depth
                e = 3.0 * math.atanh(float(np.clip(o[14], -0.999999999999, 0.999999999999)))
                a_z = self._vert(e, v[2], g["kz_w"], g["kvz_w"], g["vsink"], g["vrise"], g["adec_w"])
                down = 1.0 if -e > g["dive_thr"] else 0.0
                if hold:
                    return out(g["kv_w"] * (0.0 - v[:2]), a_z, g["tilt_hold_w"], down)
                v_cmd = g["cruise_gain_w"] * float(o[21]) * TASK_SPEED["water"] * hvec
                return out(g["kv_w"] * (g["cruise_ff"] * v_cmd - v[:2]), a_z, 0.0, down)
            if hold:
                u[U_POSE] = 0.0
                return u
            u[U_DRIVE], u[U_TURN] = self._limbs(_wrap(head - yaw))
            return u
        # -- no task: crossings and the mission
        hub_top = -depth + float((md.hubs_body @ R[2, :]).max())

        def exit_water(z_abs):
            """Rise to just under the surface and settle there (staged), then
            go: the one decision the law has to remember having made."""
            e = z_abs + depth
            if wet_all:
                go = o[I_MEM] > 0.5 or (hub_top > -(g["stage_depth"] + 0.05)
                                        and tilt < g["stage_tilt_deg"] and w_mag < g["stage_rate"])
                if not go:
                    a_z = self._vert(-g["stage_depth"] - hub_top, v[2], g["kz_w"], g["kvz_w"],
                                     g["vsink"], 1.5, g["adec_w"])
                    return out(g["kv_w"] * (0.0 - v[:2]), a_z, 20.0)
                a_z = self._vert(e, v[2], g["kz_w"], g["kvz_w"], g["vsink"], 3.0, g["adec_w"])
                return out(g["kv_w"] * (0.0 - v[:2]), a_z, 10.0, mem=1.0)
            return out(g["kv_a"] * (0.0 - v[:2]),
                       self._vert(e, v[2], g["kz_a"], g["kvz_a"], 2.0, 3.0), 10.0, mem=1.0)

        on_beach = dw > 0.0
        vz_ff = sh.slope * vn if on_beach else 0.0
        if M_ == 2:                                            # LAND
            dry = depth <= 0.0 and s.max() < 0.01
            if (dry and dw >= sh.ashore and (contact or clr < 0.12)
                    and float(np.linalg.norm(v)) < 1.5):
                u[U_DRIVE], u[U_TURN] = self._limbs(_wrap(nb - yaw))      # landed: walk inland
                return u
            if not dry:
                return exit_water(0.9)
            along = max(sh.land - dw, 0.0)
            if not (dw >= sh.ashore + 0.3 and along < 1.0 and spd < 1.2):
                v_cmd = self._approach(along, g["kpos_nav"], g["v_tr"], g["abrake"]) * n
                a_z = self._vert(g["h_tr"] - clr, v[2], g["kz_a"], g["kvz_a"], 1.5, 3.0, vz_ff=vz_ff)
                return out(g["kv_nav"] * (v_cmd - v[:2]), a_z, g["tilt_a"])
            v_cmd = self._approach(along, g["kpos_nav"], 1.0, g["abrake"]) * n
            return out(g["kv_nav"] * (v_cmd - v[:2]),
                       self._vert(-1.0, v[2], g["kz_a"], g["kvz_a"], 1.0, 1.0), g["tilt_a"])
        if M_ == 0:                                            # AIR
            if depth > 0.0 or wet_any:
                return exit_water(g["h_air"])
            if clr < 1.0:
                a_z = self._vert(g["h_air"] - clr, v[2], g["kz_a"], g["kvz_a"], 1.0, 3.0)
                return out(g["kv_a"] * (0.0 - v[:2]), a_z, 20.0)
            along = sh.station - dw
            v_nav = float(np.clip(max(spd, g["v_nav"]), g["v_nav"], 9.0))
            v_cmd = math.copysign(min(v_nav, g["kpos"] * abs(along)), along) * n
            target = float(np.clip(clr, g["h_air"], g["h_air"] + abs(along)))
            a_z = self._vert(target - clr, v[2], g["kz_a"], g["kvz_a"], 6.0, 3.0, vz_ff=vz_ff)
            return out(g["kv_a"] * (v_cmd - v[:2]), a_z, g["tilt_a"])
        # WATER
        if wet_all and depth > 0.0:
            if dw >= sh.shelf:
                return u                                       # rest on the shelf
            v_cmd = self._approach(sh.station - dw, g["kpos_w"], 0.5) * n
            a_z = self._vert(depth - g["depth_nav"], v[2], g["kz_w"], g["kvz_w"], g["vsink"],
                             g["vrise"], g["adec_w"])
            return out(g["kv_w"] * (v_cmd - v[:2]), a_z, 45.0)
        if wet_any or depth > 0.0:
            return u                                           # sink
        if dw > sh.deep:
            v_cmd = self._approach(sh.station - dw, g["kpos_nav"], g["v_tr"], g["abrake"]) * n
            a_z = self._vert(1.5 - clr, v[2], g["kz_a"], g["kvz_a"], 2.0, 3.0)
            return out(g["kv_nav"] * (v_cmd - v[:2]), a_z, g["tilt_a"])
        if clr < g["cut_clear"]:
            return u                                           # rotors stopped before the surface
        return out(g["kv_a"] * (0.0 - v[:2]),
                   self._vert(-5.0, v[2], g["kz_a"], g["kvz_a"], g["vdesc"], 1.0), g["tilt_a"])


class _TokenBasis:
    """A mobility basis that hands back one fixed parameter set: the evaluators
    route an action through a basis, and this controller's action is not a
    pattern generator's."""

    def __init__(self, token):
        self.token = token

    def command_policy(self, params, action, n, policy):
        return self.token, np.zeros(6), None

    def twist_of(self, coeffs):
        return np.zeros(6)


class SplitController:
    """The environment's side of the split: an outer law ``law(o) -> u`` asked
    at the evaluators' 25 Hz, its action held between decisions and turned
    into actuator targets by ``act(u)`` at every simulation step.

    It holds what the law must not: the memory output of the previous
    decision, the held action, and the clock restarted when the commanded
    medium changes.  All three are cleared when a rollout starts.

    ``install(env)`` replaces the environment's pattern-generator command and
    builds ``controller``, the object the evaluators take; ``uninstall(env)``
    restores what was replaced.  Parameters other than this controller's own
    still reach the body's pattern generator, so the body's own gait and the
    still machine run unchanged while it is installed.

    The keyword arguments are the arms the law is measured against:
    ``every_step`` asks the law at every simulation step, ``feed_memory=False``
    hands it zeros for memory, ``freeze_command`` shows it the first commanded
    medium for the whole rollout.
    """

    def __init__(self, law, act, *, every_step: bool = False, feed_memory: bool = True,
                 freeze_command: bool = False):
        self.law, self.inner_act = law, act
        self.every_step, self.feed_memory, self.freeze_command = every_step, feed_memory, freeze_command
        self.env = None
        self.controller = None
        #: When a list, every decision appends ``(time, o, u)``.
        self.log = None
        self._clear()

    def _clear(self) -> None:
        self.u = np.zeros(N_ACT)
        self.mem = np.zeros(K)
        self.cmd = self.cmd_first = None
        self.t_change = 0.0
        self.t_last = -1.0
        self.decisions = 0

    # -- installing
    def install(self, env) -> "SplitController":
        if self.env is not None:
            raise RuntimeError("this SplitController is already installed")
        if isinstance(getattr(env.cpg.command, "__self__", None), SplitController):
            raise RuntimeError("the environment already has a SplitController installed")
        self.env = env
        self._token = env.held_still_params()                  # an identity marker only
        self._shadowed = "command" in vars(env.cpg)
        self._replaced = env.cpg.command
        env.cpg.command = self._command
        basis = _TokenBasis(self._token)
        self.controller = Controller(params=self._token, policy=self,
                                     bases={"air": basis, "water": basis})
        self._clear()
        return self

    def uninstall(self, env) -> None:
        if env is not self.env:
            raise RuntimeError("this SplitController is not installed on that environment")
        if self._shadowed:
            env.cpg.command = self._replaced
        else:
            del env.cpg.command
        self.env = self.controller = None

    @contextmanager
    def installed(self, env):
        self.install(env)
        try:
            yield self
        finally:
            self.uninstall(env)

    # -- running
    def _decide(self) -> None:
        env = self.env
        t = float(env.data.time)
        cmd = self.cmd_first if self.freeze_command else self.cmd
        o = observe(env, cmd, self.mem if self.feed_memory else np.zeros(K), t - self.t_change)
        u = self.law(o)
        self.u, self.mem = u, u[U_MEM:].copy()
        self.decisions += 1
        if self.log is not None:
            self.log.append((t, o.copy(), u.copy()))

    def act(self, obs):
        """The evaluators' policy call.  Reads the commanded medium out of the
        observation they built, restarts the clock when it changes, and (unless
        ``every_step``) asks the law."""
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
        if not self.every_step:
            self._decide()
        return np.zeros(1)

    def _command(self, params, t):
        if params is not self._token:
            return self._replaced(params, t)                   # the body's own pattern generator
        if self.every_step and self.cmd is not None:
            self._decide()
        return self.inner_act(self.u)


def teacher_controller(env, **arms) -> SplitController:
    """The teacher over the inner loop for the body in ``env``, not yet
    installed.  ``arms`` are `SplitController`'s keyword arguments."""
    model = body_model(env)
    return SplitController(Teacher(model, ShoreMap()).teach, InnerLoop(env, model).act, **arms)
