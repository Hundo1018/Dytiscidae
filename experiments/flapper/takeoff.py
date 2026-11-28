"""T1: does the two-pair flapper (layout B, the neutral hull) leave the ground
under its own power?  Spec: PREREG_T1.md (frozen).  Three parts in order: lift
at rest on a held rig (T1a), a free hover (T1b), take-off through the repo's
`land_to_air` probe (T1c).  No learning, no optimiser; tuned by hand on seeds
and entry states 100-103.

    cd /home/hundo/Projects/Dytiscidae/dytiscidae
    export FLAPPER_BODY_B='{"hull_radius": 0.0209}' MUJOCO_GL=disable PYTHONPATH=.
    F=$PWD/experiments/flapper
    .venv/bin/python $F/takeoff.py rig     --out $F/results/takeoff.jsonl
    .venv/bin/python $F/takeoff.py hover   --arm hover,held --seeds 0-7 --out $F/results/takeoff.jsonl
    DYTISCIDAE_CROSSING_GATES=1 .venv/bin/python $F/takeoff.py takeoff --arm go,still,frozen --entries 0-7 --out $F/results/takeoff.jsonl

The harness is F2's (`landwalk.py`); `fly.py` is imported only to hand a hover
to F4's law.  Nothing in either is changed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import landwalk as LW  # noqa: E402  (sets the 5 g avionics at import)
from dytiscidae.envs.triphibian import Domain  # noqa: E402

G = 9.80665
DT = LW.DT
DIAGONAL = (0.0, 0.5, 0.5, 0.0)        # fore-left, fore-right, hind-left, hind-right, beats
NOSE_UP = (math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0)   # body +x to world +z, belly to world +x


def stroke_targets(plan, t, f, A, Psi, phases, n_act, A_i=None, bias_i=None, off_i=None):
    """Joint targets and their rates for the four wings: stroke_i = bias_i +
    A_i sin(th_i), feather_i = off_i - Psi cos(th_i) (leading edge down on the
    downstroke: the force is along the body's nose)."""
    tgt, rate = np.zeros(n_act), np.zeros(n_act)
    w = 2.0 * math.pi * f
    for i, wg in enumerate(plan.wings):
        a = w * t + 2.0 * math.pi * phases[i]
        Ai = A if A_i is None else A_i[i]
        b = 0.0 if bias_i is None else bias_i[i]
        o = 0.0 if off_i is None else off_i[i]
        tgt[wg["stroke"]] = wg["s_stroke"] * (b + Ai * math.sin(a))
        rate[wg["stroke"]] = wg["s_stroke"] * Ai * w * math.cos(a)
        tgt[wg["feather"]] = wg["s_feather"] * (o - Psi * math.cos(a))
        rate[wg["feather"]] = wg["s_feather"] * Psi * w * math.sin(a)
    return tgt, rate


def row_rig(args) -> dict:
    """T1a: the body held upright and still, every wing joint prescribed (as
    `rig.run` does it), the mean aerodynamic force along the body's axis."""
    Phi_deg, Psi_deg, f, name, phases = args
    env = LW.make_env("B", 0)
    plan = LW.Plan(env)
    env.reset(Domain.AIR, randomise=False)
    d, m = env.data, env.model
    q0 = d.qpos[:7].copy()
    q0[:3] = (0.0, 0.0, 30.0)
    q0[3:7] = NOSE_UP
    A, Psi = math.radians(Phi_deg) / 2.0, math.radians(Psi_deg)
    n_beat = int(round(1.0 / (f * DT)))
    weight = float(env.solver._dry_mass.sum() * G)
    qadr, vadr = env._act_qadr, env._act_vadr
    F = []
    for k in range(6 * n_beat):
        t = k * DT
        tgt, rate = stroke_targets(plan, t, f, A, Psi, phases, plan.n_act)
        d.qpos[:7] = q0
        d.qvel[:6] = 0.0
        d.qpos[qadr], d.qvel[vadr] = tgt, rate
        env.step(tgt)
        d.qpos[qadr] = tgt
        if k >= 2 * n_beat:
            Fk = d.xfrc_applied[:, :3].sum(0).copy()
            Fk[2] -= (m.body_mass.sum() - env.solver._dry_mass.sum()) * G
            F.append(Fk)
    F = np.array(F)
    mean = F.mean(0)
    return {"kind": "rig", "phasing": name, "f": f, "Phi_deg": Phi_deg, "Psi_deg": Psi_deg,
            "weight_N": weight, "axial_N": float(mean[2]), "axial_over_weight": float(mean[2] / weight),
            "side_N": [float(mean[0]), float(mean[1])],
            "axial_min_N": float(F[:, 2].min()), "axial_max_N": float(F[:, 2].max()),
            "side_swing_N": float(np.abs(F[:, :2]).max()), "finite": bool(np.isfinite(F).all())}


#: The hover stroke and its loop, set by hand on seeds 100-103 (trials_T1.jsonl).
#: The body stands on its tail; the wings sweep about its long axis and lift
#: along it.  Body axes: x the nose (up in a hover), y to the left wing, z the
#: back (the belly faces -z).
HOVER = dict(
    f=12.0, A=0.653, Psi=0.785, phases=DIAGONAL,     # 0.70 rad lifts 1.15 weights on the rig; lift goes as A^2
    # What turns the body, each measured from an upright release with no loop
    # (trials_T1 "open loop from an upright release"), as angular acceleration
    # per unit of the difference:
    #   about z (the back):  left stroke larger than right     -170 rad/s^2 per unit share of the amplitude
    #   about y (the left):  every stroke biased toward the belly  -250 rad/s^2 per rad
    #   about x (the nose):  left feather offset against right  +180 rad/s^2 per rad
    g_z=170.0, g_y=250.0, g_x=180.0,
    wn=8.0, zeta=0.8,                 # tilt loop: natural frequency (rad/s) and damping
    k_spin=10.0,                      # spin about the nose: damping, 1/s
    k_v=0.35, tilt_max=0.3,           # rad of lean per m/s of drift, and its limit
    k_h=0.6, k_vz=0.17, dA_max=0.3,   # share of the amplitude per m of height error and per m/s of climb
    bias_max=0.4, dz_max=0.3, off_max=0.3,
    servo=dict(stroke=(60.0, 0.10), feather=(30.0, 0.03), tail=(5.0, 0.01)),
)

# what the pilot hands the hover loop
U_ON, U_ZREF, U_LEAN = 0, 1, 2
N_U = 3


class HoverWings:
    """``act(u) -> joint targets``, every step.  Reads the body's attitude,
    rates, the velocity of its centre of mass and its height over what is
    underneath (`env.clearance`, an altimeter); no position over the ground,
    no command.

    The thrust is along the nose, so the nose is pointed where the thrust is
    wanted: straight up, leaned against any drift, and leaned ``u[U_LEAN]``
    toward the belly when asked.  Left-right stroke amplitude turns the body
    about its back axis, a common stroke bias about its left axis, a
    left-right feather offset damps the spin about the nose, and the common
    amplitude holds the height.

    Everything it feeds back is a mean over the last beat.  The wings are 45%
    of the machine, so within a beat the hull is thrown about by their
    reaction (and by the loop's own stroke bias); fed back raw, at every step,
    the loop drove itself unstable in 0.05 s (trials_T1 "hover trials 1").
    The means are this object's only memory and are cleared when the clock
    goes back."""

    def __init__(self, env, plan, h: dict):
        self.env, self.plan, self.h = env, plan, dict(h)
        self.n = max(int(round(1.0 / (h["f"] * DT))), 1)
        self.mass = np.asarray(env.solver._dry_mass, float)[1:]
        self._clear()

    def _clear(self):
        self.buf_w, self.buf_c, self.buf_com, self.t_last = [], [], [], -1.0

    def sense(self):
        """Beat means of the body rates and of where 'up' is in the body frame,
        and the centre of mass's velocity over the last beat."""
        env, d = self.env, self.env.data
        t = float(d.time)
        if t < self.t_last:
            self._clear()
        self.t_last = t
        R = d.xmat[env.root_body].reshape(3, 3)
        com = (self.mass[:, None] * d.xipos[1:]).sum(0) / float(self.mass.sum())
        for buf, x in ((self.buf_w, d.qvel[3:6].copy()), (self.buf_c, R[2, :].copy()), (self.buf_com, com)):
            buf.append(x)
            if len(buf) > self.n + 1:
                buf.pop(0)
        w = np.mean(self.buf_w, axis=0)
        up_b = np.mean(self.buf_c, axis=0)
        up_b /= max(float(np.linalg.norm(up_b)), 1e-9)
        v = ((self.buf_com[-1] - self.buf_com[0]) / ((len(self.buf_com) - 1) * DT)
             if len(self.buf_com) > 1 else np.zeros(3))
        return R, w, up_b, v

    def act(self, u) -> np.ndarray:
        h, pl, env = self.h, self.plan, self.env
        d = env.data
        if u[U_ON] <= 0.5:
            self._clear()
            return np.zeros(pl.n_act)
        R, w, up_b, v = self.sense()
        # where the thrust should point: up, leaned against the drift.  `up_b` is
        # world-up as the body sees it; the lean is added in the same frame.
        lean_w = np.array([-h["k_v"] * float(v[0]), -h["k_v"] * float(v[1]), 0.0])
        mag = float(np.linalg.norm(lean_w))
        if mag > h["tilt_max"]:
            lean_w *= h["tilt_max"] / mag
        c = up_b + R.T @ lean_w
        c /= max(float(np.linalg.norm(c)), 1e-9)
        L = float(u[U_LEAN])
        t_b = np.array([math.cos(L), 0.0, math.sin(L)])
        dlt = np.cross(t_b, c)                               # the small rotation that would put it right
        kp, kd = h["wn"] ** 2, 2.0 * h["zeta"] * h["wn"]
        acc_y = kp * float(dlt[1]) - kd * float(w[1])
        acc_z = kp * float(dlt[2]) - kd * float(w[2])
        bias = float(np.clip(-acc_y / h["g_y"], -h["bias_max"], h["bias_max"]))
        dz = float(np.clip(-acc_z / h["g_z"], -h["dz_max"], h["dz_max"]))
        off = float(np.clip(-h["k_spin"] * float(w[0]) / h["g_x"], -h["off_max"], h["off_max"]))
        col = float(np.clip(h["k_h"] * (float(u[U_ZREF]) - float(env.clearance())) - h["k_vz"] * float(v[2]),
                            -h["dA_max"], h["dA_max"]))
        A_i = [h["A"] * (1.0 + col) * (1.0 + wg["side"] * dz) for wg in pl.wings]
        tgt, _ = stroke_targets(pl, float(d.time), h["f"], h["A"], h["Psi"], h["phases"], pl.n_act, A_i=A_i,
                                bias_i=[bias] * 4, off_i=[wg["side"] * off for wg in pl.wings])
        return tgt


class Recorder:
    """Wraps `env.step` on the instance; reads, changes nothing."""

    def __init__(self, env, plan):
        self.env, self.plan, self.rows = env, plan, []
        self._step = env.step
        env.step = self.step

    def close(self):
        if "step" in self.env.__dict__:
            del self.env.__dict__["step"]

    def step(self, target):
        env = self.env
        alive = self._step(target)
        d = env.data
        R = d.xmat[env.root_body]
        af = np.abs(d.actuator_force[: self.plan.n_act])
        p, v, w = d.xpos[env.root_body], d.qvel[:3], d.qvel[3:6]
        self.rows.append((float(d.time), float(p[0]), float(p[1]), float(p[2]),
                          float(v[0]), float(v[1]), float(v[2]),
                          float(R[6]), float(R[7]), float(R[8]),          # world up in the body frame
                          float(w[0]), float(w[1]), float(w[2]),          # body rates
                          float(af[self.plan.i_stroke].max()), float(af[self.plan.i_feather].max()),
                          float(env.clearance())))
        return alive


def setup(seed, hover=None):
    env = LW.make_env("B", seed)
    plan = LW.Plan(env)
    h = {**HOVER, **(hover or {})}
    env.launch_speed, env.level_margin(), env.thrust_margin(), env.flies_level()   # cached, with the package's servos
    LW.ideal_servos(env, h["servo"])
    return env, plan, h


def release_upright(env, seed=None, z=30.0, tilt=(0.0, 0.0), v0=(0.0, 0.0, 0.0)):
    """Upright and still, 30 m up (T1b).  ``tilt`` (rad about the body's left
    and back axes) and ``v0`` (m/s) disturb the release; tuning probes only."""
    env.reset(Domain.AIR, randomise=seed is not None)
    d = env.data
    d.qpos[2] = z
    q = np.array(NOSE_UP)
    for ax, ang in (((0.0, 1.0, 0.0), tilt[0]), ((0.0, 0.0, 1.0), tilt[1])):
        if ang:
            dq, out = np.array([math.cos(ang / 2), *(math.sin(ang / 2) * np.array(ax))]), np.zeros(4)
            env._mj.mju_mulQuat(out, q, dq)                # a rotation about a body axis
            q = out
    d.qpos[3:7] = q
    d.qvel[:] = 0.0
    d.qvel[:3] = v0
    env._mj.mj_forward(env.model, d)


#: The take-off (T1c), set by hand on entry states 100-103 (trials_T1.jsonl).
TAKEOFF = dict(
    push_rate=3.0, push_max=1.3,     # rad/s and rad: the fore wings swung down against the ground
    hind=0.0,                        # rad the hind wings are held at meanwhile
    engage_pitch=0.75, push_time=0.8,   # rad of nose-up pitch, or s after the command, at which the hover loop takes over
    z_ref=1.5,                       # m over the ground it then climbs to
    lean=0.5, lean_above=1.0,        # rad the nose is leaned toward the belly once this far up (it arrives belly down)
    blend=0.08,                      # s over which the stroke takes over from the push-up's pose
)


class TakeoffController:
    """The law for the `land_to_air` probe, and the environment's side of it.

    Told to stay on land, it stands on its four thumbs.  Told the air, it does
    a push-up: the fore wings swing down, their tips meet the beach and the
    nose rises about the tail.  When the nose is up far enough the hover loop
    takes over, climbs, and once clear leans the nose toward the belly so it
    arrives the right way up.

    It keeps what a law may not: the time of the command and whether the hover
    has been engaged (a latch), both cleared when the clock goes back.  The
    ``frozen`` arm is shown the first command for the whole rollout."""

    def __init__(self, env, plan, h: dict, tk: dict, frozen: bool = False):
        from dytiscidae.envs.evaluate import Controller
        from dytiscidae.envs.triphibian import DOMAIN_CYCLE
        self.env, self.plan, self.h, self.tk, self.frozen = env, plan, h, dict(tk), frozen
        self.cycle = DOMAIN_CYCLE
        self.wings = HoverWings(env, plan, h)
        self.token = env.held_still_params()
        self._orig = env.cpg.command
        env.cpg.command = self.command
        b = LW._Basis(self.token)
        self.controller = Controller(params=self.token, policy=self, bases={"air": b, "water": b})
        self.events = []
        self._clear()

    def _clear(self):
        self.cmd = self.cmd_first = None
        self.t_go, self.t_on, self.hovering, self.t_last = None, None, False, -1.0

    def close(self):
        self.env.cpg.command = self._orig

    def act(self, obs):                                        # the probe's 25 Hz call
        t = float(self.env.data.time)
        c = np.asarray(obs[11:14], float)
        cmd = self.cycle[int(np.argmax(c))] if c.max() > 0.5 else None
        if self.cmd_first is None:
            self.cmd_first = cmd
        if self.frozen:
            cmd = self.cmd_first
        if cmd is not self.cmd:
            self.cmd = cmd
            if cmd is Domain.AIR:
                self.t_go = t
                self.events.append(("told the air", round(t, 3)))
        return np.zeros(1)

    def command(self, params, t):
        if params is not self.token:
            return self._orig(params, t)
        env, pl, tk = self.env, self.plan, self.tk
        d = env.data
        now = float(d.time)
        if now < self.t_last:
            self._clear()
        self.t_last = now
        out = np.zeros(pl.n_act)
        if self.cmd is not Domain.AIR or self.t_go is None:
            return out                                         # standing on the thumbs
        s = now - self.t_go
        pitch = math.asin(float(np.clip(d.xmat[env.root_body][6], -1.0, 1.0)))
        if not self.hovering and (pitch >= tk["engage_pitch"] or s >= tk["push_time"]):
            self.hovering, self.t_on = True, now
            self.events.append(("hover loop on", round(now, 3), round(math.degrees(pitch), 1), round(float(env.clearance()), 3)))
        push_s = s if not self.hovering else self.t_on - self.t_go
        for i, wg in enumerate(pl.wings):
            ang = min(tk["push_rate"] * push_s, tk["push_max"]) if i < 2 else tk["hind"]
            out[wg["stroke"]] = wg["s_stroke"] * ang
        if not self.hovering:
            return out
        u = np.zeros(N_U)
        u[U_ON], u[U_ZREF] = 1.0, tk["z_ref"]
        u[U_LEAN] = tk["lean"] if float(env.clearance()) > tk["lean_above"] else 0.0
        # the stroke is blended in from the push-up's pose over `blend` s, so the
        # servos are not handed a step
        b = float(np.clip((now - self.t_on) / tk["blend"], 0.0, 1.0)) if tk["blend"] > 0.0 else 1.0
        return (1.0 - b) * out + b * self.wings.act(u)


def row_takeoff(args) -> dict:
    """T1c: the repo's `land_to_air` probe from one entry state."""
    arm, entry, hover, takeoff = args
    import dytiscidae.envs.transitions as T
    from dytiscidae.envs.evaluate import Controller
    t0 = time.time()
    env, plan, h = setup(entry, hover)
    tk = {**TAKEOFF, **(takeoff or {})}
    ctl = None
    if arm == "still":
        controller = Controller(params=env.held_still_params(), policy=None, bases=None)
    else:
        ctl = TakeoffController(env, plan, h, tk, frozen=(arm == "frozen"))
        controller = ctl.controller
    base = T.transition_scatter_seed
    # the entry states as E0's probes made them (`nulls.set_entry_state`): the
    # repo's draw plus 7919 x entry; entry 0 is the repo's own
    T.transition_scatter_seed = lambda kind, _s=int(entry): (base(kind) + 7919 * _s) & 0x7FFFFFFF
    rec = Recorder(env, plan)
    row = {"kind": "takeoff", "arm": arm, "entry": entry, "gates": os.environ.get("DYTISCIDAE_CROSSING_GATES", ""),
           "hover": {k: v for k, v in h.items() if k != "servo"}, "servo": h["servo"], "takeoff": tk}
    try:
        r = T.run_transition(env, "land_to_air", controller)
        a = np.array(rec.rows)
        go = a[a[:, 0] > 1.5]
        up = np.flatnonzero(go[:, 15] <= 0.05)
        row.update(crossed=bool(r.crossed), failure=r.failure, bad_qacc=int(r.bad_qacc), hold=float(r.hold),
                   hold_drift_m=float(r.hold_drift), started_in=r.started_in,
                   go_peak_clearance_m=float(r.go_peak_clearance), end_clearance_m=float(env.clearance()),
                   exit_upright=float(r.exit_upright), min_upright=float(r.min_upright),
                   aloft_fraction=float(r.aloft_fraction),
                   off_the_ground_for_good_s=(float(go[up[-1] + 1, 0]) if len(up) and up[-1] + 1 < len(go) else
                                              (float(go[0, 0]) if not len(up) else None)),
                   end_speed_mps=float(np.linalg.norm(a[-83:, 4:7].mean(0))),
                   ground_travel_m=float(np.linalg.norm(a[-1, 1:3] - a[0, 1:3])),
                   axis_end_deg=float(np.degrees(np.arccos(np.clip(a[-1, 7], -1.0, 1.0)))),
                   torque_stroke=float(go[:, 13].max()), torque_feather=float(go[:, 14].max()),
                   torque_stroke_hold=float(a[a[:, 0] <= 1.5][:, 13].max()),
                   events=[] if ctl is None else ctl.events,
                   trace=[[round(float(x[0]), 2), round(float(x[15]), 3), round(float(np.degrees(np.arcsin(np.clip(x[7], -1, 1)))), 1),
                           round(float(x[9]), 2), round(float(math.hypot(x[1] - a[0, 1], x[2] - a[0, 2])), 2)] for x in a[99::100]])
    finally:
        T.transition_scatter_seed = base
        rec.close()
        if ctl is not None:
            ctl.close()
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def hover_reads(a, z0, xy0) -> dict:
    """What the hover bar reads over a recorded stretch."""
    n_beat = 83
    drift = np.linalg.norm(a[:, 1:3] - xy0, axis=1)
    return {"height_min_m": float(a[:, 3].min() - z0), "height_max_m": float(a[:, 3].max() - z0),
            "height_end_m": float(a[-1, 3] - z0), "drift_max_m": float(drift.max()), "drift_end_m": float(drift[-1]),
            "axis_min": float(a[:, 7].min()),                # cosine of the nose's angle from vertical
            "axis_max_deg": float(np.degrees(np.arccos(np.clip(a[:, 7].min(), -1.0, 1.0)))),
            "speed_end_mps": float(np.linalg.norm(a[-n_beat:, 4:7].mean(0))),
            "spin_end_rps": float(a[-n_beat:, 10].mean()),
            "torque_stroke": float(a[a[:, 0] > a[0, 0] + 0.25][:, 13].max()),
            "torque_feather": float(a[a[:, 0] > a[0, 0] + 0.25][:, 14].max()),
            "torque_stroke_all": float(a[:, 13].max()), "torque_feather_all": float(a[:, 14].max())}


def row_hover(args) -> dict:
    """T1b: released upright and still at 30 m and told to stay, 6 s."""
    arm, seed, hover, seconds = args
    env, plan, h = setup(seed, hover)
    release_upright(env, seed, tilt=h.get("tilt0", (0.0, 0.0)), v0=h.get("v0", (0.0, 0.0, 0.0)))
    d = env.data
    z0, xy0 = float(d.qpos[2]), d.qpos[:2].copy()
    wings = HoverWings(env, plan, h)
    u = np.zeros(N_U)
    u[U_ON], u[U_ZREF] = (1.0 if arm == "hover" else 0.0), float(env.clearance())
    rec = Recorder(env, plan)
    try:
        for _ in range(int(seconds / DT)):
            env.step(wings.act(u))
    finally:
        rec.close()
    a = np.array(rec.rows)
    row = {"kind": "hover", "arm": arm, "seed": seed, "hover": {k: v for k, v in h.items() if k != "servo"},
           "servo": h["servo"], "seconds": seconds, **hover_reads(a, z0, xy0),
           "bad_qacc": int(d.warning[env._mj.mjtWarning.mjWARN_BADQACC].number),
           "trace": [[round(float(r[0]), 1), round(float(r[3] - z0), 2), round(float(math.hypot(r[1] - xy0[0], r[2] - xy0[1])), 2),
                      round(float(np.degrees(np.arccos(np.clip(r[7], -1, 1)))), 1)] for r in a[499::500]]}
    row["met"] = bool(row["bad_qacc"] == 0 and max(abs(row["height_min_m"]), abs(row["height_max_m"])) <= 1.0
                      and row["drift_max_m"] <= 2.0 and row["axis_max_deg"] <= 45.0)
    return row


def row_handoff(args) -> dict:
    """PREREG section 2: is F4's flight reachable from the hover?  T1b's hover
    for 6 s, then (``lean_s`` > 0) the hover loop leaned toward the belly for
    that long to gather way, then F4's own law (`fly.Pilot` over `fly.Wings`,
    untouched, given a straight air task) for 4 s, read by F4's level bar."""
    seed, lean_s, hover, lean = args
    import fly
    from dytiscidae.envs.tasks import CRUISE, Phase, TaskSchedule
    env, plan, h = setup(seed, hover)
    release_upright(env, seed)
    d = env.data
    hw = HoverWings(env, plan, h)
    u = np.zeros(N_U)
    u[U_ON], u[U_ZREF] = 1.0, float(env.clearance())
    rec = Recorder(env, plan)
    v_launch = float(env.launch_speed)
    try:
        for _ in range(int(6.0 / DT)):
            env.step(hw.act(u))
        u[U_LEAN] = lean
        for _ in range(int(lean_s / DT)):
            env.step(hw.act(u))
        k0 = len(rec.rows)
        fw = fly.Wings(env, plan, dict(fly.WINGS))
        pilot = fly.Pilot(env, dict(fly.PILOT))
        env.task = TaskSchedule("air", (Phase(CRUISE, 0.0, heading=0.0, speed=v_launch),
                                        Phase(CRUISE, 0.5, heading=0.0, speed=v_launch)))
        env._arm_task(Domain.AIR, 4.0)
        t_on = float(d.time)
        uf = np.zeros(fly.N_U)
        for k in range(int(4.0 / DT)):
            if k % 40 == 0:
                uf = pilot(env.observation(Domain.AIR), float(d.time))
            b = float(np.clip((float(d.time) - t_on) / 0.08, 0.0, 1.0))
            env.step((1.0 - b) * hw.act(u) + b * fw.act(uf) if b < 1.0 else fw.act(uf))
    finally:
        rec.close()
    a = np.array(rec.rows)
    fl = a[k0:]
    end = fl[fl[:, 0] > fl[-1, 0] - 0.5]
    last2 = fl[fl[:, 0] > fl[-1, 0] - 2.0]
    speed_end = float(np.linalg.norm(end[:, 4:7], axis=1).mean())
    row = {"kind": "handoff", "seed": seed, "lean_s": lean_s, "lean": lean,
           "at_handoff": {"speed_mps": float(np.linalg.norm(a[k0 - 83:k0, 4:7].mean(0))), "up_z": float(a[k0 - 1, 9]),
                          "nose_from_vertical_deg": float(np.degrees(np.arccos(np.clip(a[k0 - 1, 7], -1, 1)))),
                          "height_m": float(a[k0 - 1, 3] - a[0, 3])},
           "dz_m": float(fl[-1, 3] - fl[0, 3]), "z_min_rel_m": float(fl[:, 3].min() - fl[0, 3]),
           "dz_last2s_m": float(last2[-1, 3] - last2[0, 3]),
           "speed_end_mps": speed_end, "speed_over_launch": speed_end / v_launch, "up_z_min": float(fl[:, 9].min()),
           "up_z_min_last2s": float(last2[:, 9].min()), "up_z_end": float(fl[-1, 9]),
           "torque_stroke": float(fl[:, 13].max()), "torque_feather": float(fl[:, 14].max()),
           "bad_qacc": int(d.warning[env._mj.mjtWarning.mjWARN_BADQACC].number),
           "trace": [[round(float(x[0]), 1), round(float(x[3] - fl[0, 3]), 2), round(float(np.linalg.norm(x[4:7])), 1), round(float(x[9]), 2)]
                     for x in fl[249::250]]}
    row["met_f4_level_bar"] = bool(row["bad_qacc"] == 0 and abs(row["dz_m"]) <= 1.5
                                   and abs(row["speed_over_launch"] - 1.0) <= 0.3 and row["up_z_min"] > 0.5)
    return row


def row_open(args) -> dict:
    """A tuning probe: released upright and still, the stroke with fixed
    per-wing changes and no loop.  Reads the body's rates soon after."""
    mods, hover, seconds = args
    env, plan, h = setup(100, hover)
    release_upright(env)
    rec = Recorder(env, plan)
    A_i = [h["A"] * (1.0 + x) for x in mods.get("dA", (0, 0, 0, 0))]
    try:
        for _ in range(int(seconds / DT)):
            tgt, _ = stroke_targets(plan, float(env.data.time), h["f"], h["A"], h["Psi"], h["phases"], plan.n_act,
                                    A_i=A_i, bias_i=mods.get("bias"), off_i=mods.get("off"))
            env.step(tgt)
    finally:
        rec.close()
    a = np.array(rec.rows)
    n_beat = int(round(1.0 / (h["f"] * DT)))

    def at(t):                                             # one-beat mean ending at t
        k = int(round(t / DT))
        return a[max(k - n_beat, 0):k].mean(0)
    out = {"kind": "open", "mods": mods, "hover": {k: v for k, v in h.items() if k != "servo"}, "seconds": seconds}
    for t in (0.17, 0.33, 0.5):
        if t <= seconds + 1e-9:
            r = at(t)
            out[f"t{t}"] = {"rate_xyz": [round(float(x), 2) for x in r[10:13]], "up_b": [round(float(x), 3) for x in r[7:10]],
                            "vel": [round(float(x), 2) for x in r[4:7]], "dz": round(float(r[3] - a[0, 3]), 3)}
    out["torque_stroke"], out["torque_feather"] = float(a[:, 13].max()), float(a[:, 14].max())
    out["bad_qacc"] = int(env.data.warning[env._mj.mjtWarning.mjWARN_BADQACC].number)
    return out


def _call(job):
    return job[0](job[1])


def _seeds(s):
    lo, hi = (int(x) for x in s.split("-")) if "-" in s else (int(s), int(s))
    return list(range(lo, hi + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rig")
    r.add_argument("--out", required=True)
    r.add_argument("--jobs", type=int, default=6)
    hv = sub.add_parser("hover")
    hv.add_argument("--arm", default="hover")
    hv.add_argument("--seeds", default="100-103")
    hv.add_argument("--hover", default="{}")
    hv.add_argument("--seconds", type=float, default=6.0)
    hv.add_argument("--out", required=True)
    hv.add_argument("--note", default="")
    hv.add_argument("--jobs", type=int, default=6)
    tk = sub.add_parser("takeoff")
    tk.add_argument("--arm", default="go")
    tk.add_argument("--entries", default="100-103")
    tk.add_argument("--hover", default="{}")
    tk.add_argument("--takeoff", default="{}")
    tk.add_argument("--out", required=True)
    tk.add_argument("--note", default="")
    tk.add_argument("--jobs", type=int, default=6)
    ho = sub.add_parser("handoff")
    ho.add_argument("--seeds", default="100-102")
    ho.add_argument("--lean-s", default="[0.0]")
    ho.add_argument("--lean", type=float, default=0.5)
    ho.add_argument("--hover", default="{}")
    ho.add_argument("--out", required=True)
    ho.add_argument("--note", default="")
    ho.add_argument("--jobs", type=int, default=6)
    o = sub.add_parser("open")
    o.add_argument("--mods", default="[{}]")
    o.add_argument("--hover", default="{}")
    o.add_argument("--seconds", type=float, default=0.5)
    o.add_argument("--out", required=True)
    o.add_argument("--note", default="")
    o.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    import multiprocessing as mp
    if a.cmd == "rig":
        jobs = [(row_rig, (Phi, Psi, 12.0, name, ph)) for name, ph in (("diagonal", DIAGONAL), ("in phase", (0.0,) * 4))
                for Phi in (80.0, 130.0) for Psi in (30.0, 45.0)]
    if a.cmd == "hover":
        jobs = [(row_hover, (arm, sd, json.loads(a.hover), a.seconds)) for arm in a.arm.split(",") for sd in _seeds(a.seeds)]
    if a.cmd == "takeoff":
        jobs = [(row_takeoff, (arm, e, json.loads(a.hover), json.loads(a.takeoff))) for arm in a.arm.split(",")
                for e in _seeds(a.entries)]
    if a.cmd == "handoff":
        jobs = [(row_handoff, (sd, ls, json.loads(a.hover), a.lean)) for ls in json.loads(a.lean_s) for sd in _seeds(a.seeds)]
    if a.cmd == "open":
        jobs = [(row_open, (md, json.loads(a.hover), a.seconds)) for md in json.loads(a.mods)]
    with mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
        rows = pool.map(_call, jobs, chunksize=1)
    with open(Path(a.out).resolve(), "a") as fh:
        for row in rows:
            if getattr(a, "note", ""):
                row["note"] = a.note
            fh.write(json.dumps(row, default=float) + "\n")
            if row["kind"] == "hover":
                print(f"hover {row['arm']} {row['seed']}: height {row['height_min_m']:+.2f}..{row['height_max_m']:+.2f} (end {row['height_end_m']:+.2f}) m, drift max {row['drift_max_m']:.2f} end {row['drift_end_m']:.2f} m,"
                      f" axis within {row['axis_max_deg']:.0f} deg, end speed {row['speed_end_mps']:.2f} m/s spin {row['spin_end_rps']:+.1f} rad/s, torque {row['torque_stroke']:.2f} {row['torque_feather']:.2f}, bad {row['bad_qacc']} {'MET' if row['met'] else 'no'}", flush=True)
            if row["kind"] == "takeoff":
                print(f"takeoff {row['arm']} {row['entry']}: crossed {row.get('crossed')} {row.get('failure', '')!r} bad {row.get('bad_qacc')} hold {row.get('hold', 0):.2f} (drift {row.get('hold_drift_m', 0):.2f} m)"
                      f" peak {row.get('go_peak_clearance_m', 0):.2f} end {row.get('end_clearance_m', 0):.2f} m, upright end {row.get('exit_upright', 0):+.2f} min {row.get('min_upright', 0):+.2f},"
                      f" off the ground at {row.get('off_the_ground_for_good_s')}, end speed {row.get('end_speed_mps', 0):.1f} m/s, travel {row.get('ground_travel_m', 0):.1f} m, torque {row.get('torque_stroke', 0):.1f} {row.get('torque_feather', 0):.1f} | {row.get('events')}", flush=True)
            if row["kind"] == "handoff":
                h0 = row["at_handoff"]
                print(f"handoff seed {row['seed']} after {row['lean_s']:g} s of lean: at hand-off speed {h0['speed_mps']:.1f} m/s, nose {h0['nose_from_vertical_deg']:.0f} deg from vertical, up-z {h0['up_z']:+.2f};"
                      f" then 4 s of F4's law: dz {row['dz_m']:+.2f} (low {row['z_min_rel_m']:+.2f}, last 2 s {row['dz_last2s_m']:+.2f}) m, end speed {row['speed_end_mps']:.2f} ({row['speed_over_launch']:.2f} of launch),"
                      f" up-z min {row['up_z_min']:+.2f} (last 2 s {row['up_z_min_last2s']:+.2f}), torque {row['torque_stroke']:.1f} {row['torque_feather']:.1f}, bad {row['bad_qacc']} {'MET' if row['met_f4_level_bar'] else 'not met'}", flush=True)
            if row["kind"] == "open":
                print(json.dumps(row["mods"]), "| " + " | ".join(f"{k}: rates {row[k]['rate_xyz']} up {row[k]['up_b']} vel {row[k]['vel']} dz {row[k]['dz']:+.2f}" for k in ("t0.17", "t0.33", "t0.5") if k in row),
                      f"| tq {row['torque_stroke']:.1f} {row['torque_feather']:.1f} bad {row['bad_qacc']}", flush=True)
            if row["kind"] == "rig":
                print(f"rig {row['phasing']:9s} 12 Hz stroke {row['Phi_deg']:.0f} feather {row['Psi_deg']:.0f}: along the axis {row['axial_N']:.3f} N = {row['axial_over_weight']:.2f} weights"
                      f" (within a beat {row['axial_min_N']:.2f} to {row['axial_max_N']:.2f} N), mean sideways {row['side_N'][0]:+.3f} {row['side_N'][1]:+.3f} N, largest sideways {row['side_swing_N']:.2f} N", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
