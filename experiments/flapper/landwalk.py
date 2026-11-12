"""F2: wing-supported walking, one pair and a tail (A) against two pairs (B).
Spec: PREREG_F2.md (frozen).  No learning, no optimiser; gaits tuned by hand on
seeds 100-103, scored on seeds 0-7.

    cd /home/hundo/Projects/Dytiscidae/dytiscidae; export MUJOCO_GL=disable PYTHONPATH=.
    F=$PWD/experiments/flapper
    .venv/bin/python $F/landwalk.py geometry                       # every added part, on the built model
    .venv/bin/python $F/landwalk.py probes                         # the wing-share and hull-contact readings
    .venv/bin/python $F/landwalk.py run --layout A,B --arm walk,propped,frozen --seeds 0-7 --out $F/results/landwalk.jsonl
    .venv/bin/python $F/landwalk.py speed --layout A,B --out $F/results/landwalk.jsonl
    .venv/bin/python $F/landwalk.py air --out $F/results/landwalk.jsonl

Stage 1: every actuator is an ideal position servo, set on this experiment's
own `env.model` (`ideal_servos`); the largest |actuator force| per joint is
recorded on every rollout.
"""
from __future__ import annotations

import argparse
import json
import os
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dytiscidae.core.phenotype as PH  # noqa: E402

PH.AVIONICS_MASS = 0.005            # the package's 0.30 kg is sized for 15 kg machines (F0)

from body import flapper  # noqa: E402
from dytiscidae.control import innerloop as IL  # noqa: E402
from dytiscidae.envs.evaluate import Controller  # noqa: E402
from dytiscidae.core.phenotype import build  # noqa: E402
from dytiscidae.envs.evaluate import _scatter_seed, run_segment  # noqa: E402
from dytiscidae.envs.tasks import schedule_for, task_seed  # noqa: E402
from dytiscidae.envs.triphibian import Domain, TriphibianEnv  # noqa: E402

G = 9.80665
DT = 0.001
SEGMENT_SECONDS = 8.0

#: The two layouts (PREREG section 0).  B's wing is 0.21 m at aspect ratio 3,
#: F0's map: (0.21 / 0.25)^4 = 0.498 of A's wing each.
LAYOUTS = {
    "A": dict(shoulder_u=0.6, thumb_length=0.045, thumb_u=0.25, tail_length=0.09, tail_chord=0.012),
    "B": dict(wing_length=0.21, wing_chord=0.07, shoulder_u=0.75, hind_u=0.2,
              thumb_length=0.045, thumb_u=0.30),
}
BARE = {"A": dict(), "B": dict(wing_length=0.21, wing_chord=0.07)}

#: Gaits, set by hand on seeds 100-103 (trials_F2.jsonl).  Angles in rad,
#: positive stroke lowers the thumb, positive sweep puts its tip forward.
GAITS = {
    # A crutches (the mudskipper's gait): both thumbs together, the wings held
    # a little raised so the hull rides 5 mm up; in the swing the thumbs lift,
    # the hull sets down for a moment, the thumbs return and press again.
    "A": dict(f=1.5, duty=0.85, down=-0.27, up=-0.45, sweep=0.35, trap=(0.25, 0.75),
              offsets=(0.0, 0.0), stand=-0.27, tail=0.35, tail_swing=0.35,
              e_go=0.35, k_turn=1.5, turn_max=0.5, drive_far=0.0, turn_far=1.0),
    # B crawls: one thumb in the air at a time, three carrying the body.
    "B": dict(f=1.5, duty=0.8, down=0.10, up=-0.25, sweep=0.35, trap=None,
              offsets=(0.0, 0.5, 0.75, 0.25), stand=0.10, tail=0.0, tail_swing=0.0,
              e_go=0.6, k_turn=1.5, turn_max=0.7, drive_far=0.0, turn_far=1.0),
}

#: The fastest setting found by hand for each layout (trials_F2.jsonl, "top
#: speed search"), used only by the top-speed rows.
FAST = {"A": dict(f=2.5, sweep=0.6, duty=0.8),
        # B's fastest is a trot (diagonal thumbs together) that leaves the ground
        "B": dict(f=4.0, sweep=0.8, duty=0.6, up=-0.10, trap=(0.25, 0.75),
                  offsets=(0.0, 0.5, 0.5, 0.0))}

#: Ideal position servos: stiffness (N m/rad) and damping (N m s/rad) per role.
# The built joints carry 0.0226 N m s/rad of drive damping and 1e-4 kg m2 of
# armature; these gains follow a 40 ms flick to within a few hundredths of a rad.
SERVO = dict(stroke=(30.0, 0.10), feather=(20.0, 0.03), tail=(5.0, 0.01))


def make_env(layout: str, seed: int, **over):
    body = dict(LAYOUTS[layout])
    # Body options from the environment (JSON), for a re-read on a changed body
    # without touching the layouts the stored rows were made with: unset, this
    # is the layout as F2 and F3b ran it.  Inherited by spawned workers.
    body.update(json.loads(os.environ.get(f"FLAPPER_BODY_{layout}", "{}")))
    body.update(over)
    env = TriphibianEnv(build(flapper(**body)), seed=seed, timestep=DT)
    # Energy is not counted in stage 1 (the user, 2026-11-12): an ideal servo's
    # draw must not end a rollout as a flat pack.  The mass is the 0.6 Wh pack's.
    env.budget.battery.capacity_j = 1e15
    env.budget.battery.reset()
    # A fore-to-hind wake fed back at this gain (`wakefb.py`), for sensitivity
    # reads; unset or 0 it is not attached and nothing changes.
    if float(os.environ.get("FLAPPER_WAKE_GAIN", "0") or 0.0) != 0.0:
        import wakefb
        wakefb.attach(env, float(os.environ["FLAPPER_WAKE_GAIN"]))
    return env


def ideal_servos(env, servo=None) -> None:
    """Stage 1: a position servo that follows the gait, with no force limit."""
    m, mj = env.model, env._mj
    for a, name in enumerate(env.act_names):
        role = "feather" if name.endswith("_f") else ("stroke" if "wing" in name else "tail")
        kp, kd = (servo or SERVO)[role]
        m.actuator_gainprm[a, 0] = kp
        m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2] = -kp, -kd
        m.actuator_forcelimited[a] = 0
        m.actuator_ctrllimited[a] = 0


class Plan:
    """What the gait and the readings need to know about the built model."""

    def __init__(self, env):
        m, d, mj = env.model, env.data, env._mj
        self.env = env
        name = lambda b: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b)          # noqa: E731
        self.hull = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "seg0_hull")
        self.kind = {}                       # body id -> "hull" | "wing" | "tail"
        for b in range(1, m.nbody):
            n = name(b)
            self.kind[b] = "hull" if b == self.hull else ("tail" if "paddle" in n else "wing")
        # thumbs count with the wing they are fixed to
        env.reset(Domain.AIR, randomise=False)
        mj.mj_forward(m, d)
        R, p0 = d.xmat[self.hull].reshape(3, 3).copy(), d.xpos[self.hull].copy()
        names = env.act_names
        self.wings = []                      # per wing: stroke index, feather index, side, signs
        for k, a in enumerate(names):
            if not (a.endswith("_a") and "wing" in a):
                continue
            j = int(m.actuator_trnid[mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, a), 0])
            wb = int(m.jnt_bodyid[j])
            kf = names.index(a[:-2] + "_f")
            thumbs = [b for b in range(1, m.nbody) if int(m.body_parentid[b]) == wb
                      and "strut" in name(b)]
            low = min(thumbs, key=lambda b: float((R.T @ (d.xipos[b] - p0))[2]))
            tip0 = self._tip(low, R, p0)
            sg = []
            for kk in (k, kf):               # which sign lowers the tip / puts it forward
                q = env._act_qadr[kk]
                d.qpos[q] += 0.2
                mj.mj_forward(m, d)
                tip = self._tip(low, R, p0)
                d.qpos[q] -= 0.2
                sg.append(tip - tip0)
            mj.mj_forward(m, d)
            self.wings.append(dict(stroke=k, feather=kf, body=wb, thumb=low,
                                   side=1.0 if float((R.T @ (d.xipos[wb] - p0))[1]) > 0 else -1.0,
                                   x=float((R.T @ (d.xpos[wb] - p0))[0]),
                                   s_stroke=-1.0 if sg[0][2] > 0 else 1.0,
                                   s_feather=1.0 if sg[1][0] > 0 else -1.0,
                                   tip=tip0.round(4).tolist()))
        # order: fore-left, fore-right, hind-left, hind-right
        self.wings.sort(key=lambda w: (-round(w["x"], 3), -w["side"]))
        self.tail = [k for k, a in enumerate(names) if "paddle" in a]
        self.tail_sign = 1.0
        if self.tail:
            q = env._act_qadr[self.tail[0]]
            tb = [b for b in self.kind if self.kind[b] == "tail"][0]
            z0 = float((R.T @ (d.xipos[tb] - p0))[2])
            d.qpos[q] += 0.2
            mj.mj_forward(m, d)
            self.tail_sign = -1.0 if float((R.T @ (d.xipos[tb] - p0))[2]) > z0 else 1.0
            d.qpos[q] -= 0.2
            mj.mj_forward(m, d)
        self.n_act = len(names)
        self.i_stroke = [w["stroke"] for w in self.wings]
        self.i_feather = [w["feather"] for w in self.wings]
        self.weight = float(np.asarray(env.solver._dry_mass)[1:].sum()) * G

    def _tip(self, b, R, p0):
        m, d = self.env.model, self.env.data
        g = [k for k in range(m.ngeom) if m.geom_bodyid[k] == b][0]
        ax = d.geom_xmat[g].reshape(3, 3)[:, 2] * float(m.geom_size[g][1])
        ends = [R.T @ (d.geom_xpos[g] + s * ax - p0) for s in (1.0, -1.0)]
        return min(ends, key=lambda e: e[2])


class Gait:
    """``act(u) -> joint targets``: stroke presses the thumb down in stance and
    lifts it in swing, the feather sweeps it back in stance and forward in
    swing (E0b arm W's gait, on the wing's own two shoulder axes)."""

    def __init__(self, env, plan: Plan, gait: dict):
        self.env, self.plan, self.g = env, plan, dict(gait)

    def act(self, u) -> np.ndarray:
        g, pl = self.g, self.plan
        out = np.zeros(pl.n_act)
        drive, turn, pose = float(u[IL.U_DRIVE]), float(u[IL.U_TURN]), float(u[IL.U_POSE])
        t = float(self.env.data.time)
        swing_any = 0.0
        for i, w in enumerate(pl.wings):
            if drive == 0.0 and turn == 0.0:
                stroke, sweep = pose * g["stand"], 0.0
            else:
                ph = (g["f"] * t + g["offsets"][i % len(g["offsets"])]) % 1.0
                gain = float(np.clip(drive - w["side"] * turn, -1.0, 1.0))
                if ph < g["duty"]:
                    s = ph / g["duty"]
                    stroke, sweep = g["down"], g["sweep"] * (1.0 - 2.0 * s)
                else:
                    s = (ph - g["duty"]) / (1.0 - g["duty"])
                    if g.get("trap"):
                        # lift, then return, then lower: the thumb is clear of the
                        # ground before it moves and still again before it lands
                        e0, e1 = g["trap"]
                        lift = min(s / e0, 1.0, (1.0 - s) / (1.0 - e1))
                        sweep = -g["sweep"] * math.cos(math.pi * float(np.clip((s - e0) / (e1 - e0), 0.0, 1.0)))
                    else:
                        lift = math.sin(math.pi * s)
                        # the return is over by `early` of the swing, so a thumb that
                        # meets the ground late in it is no longer moving forward
                        sweep = -g["sweep"] * math.cos(math.pi * min(s / g.get("early", 1.0), 1.0))
                    stroke = g["down"] + (g["up"] - g["down"]) * lift
                    swing_any = max(swing_any, lift)
                sweep *= gain
            out[w["stroke"]] = w["s_stroke"] * stroke
            out[w["feather"]] = w["s_feather"] * sweep
        for k in pl.tail:
            out[k] = pl.tail_sign * pose * (g["tail"] + (g["tail_swing"] - g["tail"]) * swing_any)
        return out


#: Largest rate a target may move at, rad/s.  The gait's own fastest motion is
#: the feather's return (about 46 rad/s at sweep 0.35 in 24 ms); the limit only
#: removes the infinite step when the law switches between standing and walking.
SLEW = 60.0


class Slewed:
    """`act(u)` with its targets rate-limited (state of the servo interface,
    cleared when the clock restarts)."""

    def __init__(self, env, act):
        self.env, self.inner, self.prev, self.t = env, act, None, -1.0

    def act(self, u):
        t = float(self.env.data.time)
        tgt = self.inner(u)
        if self.prev is None or t < self.t or t <= 0.0:
            self.prev = np.asarray(self.env.data.qpos[self.env._act_qadr], float).copy()
        self.t = t
        self.prev = self.prev + np.clip(tgt - self.prev, -SLEW * DT, SLEW * DT)
        return self.prev.copy()


def steer(e: float, g: dict | None = None):
    """Limb drive and turn from the heading error, E0b's steering
    (`control.outer.Teacher._limbs`) with the fade made a switch: within
    ``e_go`` of the heading (or of its opposite, walking backward) the gait
    drives at full stride and steers by ``k_turn``; beyond it, it drives at
    ``drive_far`` and turns at ``turn_far``.  The walk phase is 4 s and these
    bodies turn at 10-15 deg/s, so a gait that waits to face the heading
    before it walks never walks."""
    g = g or {}
    direction = 1.0
    if abs(e) > 0.5 * math.pi:
        e, direction = (e - math.pi + math.pi) % (2.0 * math.pi) - math.pi, -1.0
    if abs(e) <= float(g.get("e_go", 10.0)):
        lim = float(g.get("turn_max", 0.7))
        return direction, float(np.clip(float(g.get("k_turn", 1.5)) * e, -lim, lim))
    return direction * float(g.get("drive_far", 1.0)), math.copysign(float(g.get("turn_far", 0.7)), e)


#: Seconds from the start of a rollout before the first step (trials_F2: B,
#: walking from t = 0, was thrown 14 cm up when it landed mid-stride; the drop
#: takes 0.12 s, and 0.5 s cost both layouts a turn they needed).
SETTLE = 0.15


class Walk:
    """Walk toward the commanded heading while the task says walk; stand otherwise."""

    def __init__(self, gait: dict):
        self.g = gait

    def __call__(self, o, t) -> np.ndarray:
        u = np.zeros(IL.N_ACT)
        u[IL.U_POSE] = 1.0
        # the segment drops the body from a few centimetres: no step before it has landed
        if o[19] > 0.5 and t >= SETTLE:
            u[IL.U_DRIVE], u[IL.U_TURN] = steer(math.atan2(o[23], o[22]), self.g)
        return u


def propped(o, t) -> np.ndarray:
    u = np.zeros(IL.N_ACT)
    u[IL.U_POSE] = 1.0
    return u


class FrozenHeading:
    """The gait with its heading frozen at the first one seen (a null arm; the
    state is this arm's, not the law's)."""

    def __init__(self, env, gait):
        self.target, self.env, self.g = None, env, gait

    def yaw(self):
        R = self.env.data.xmat[self.env.root_body].reshape(3, 3)
        return math.atan2(float(R[1, 0]), float(R[0, 0]))

    def __call__(self, o, t):
        u = np.zeros(IL.N_ACT)
        u[IL.U_POSE] = 1.0
        if o[19] > 0.5 and t >= SETTLE:
            yaw = self.yaw()
            if self.target is None:
                self.target = yaw + math.atan2(o[23], o[22])
            e = (self.target - yaw + math.pi) % (2.0 * math.pi) - math.pi
            u[IL.U_DRIVE], u[IL.U_TURN] = steer(e, self.g)
        return u


class _Basis:
    def __init__(self, token):
        self.token = token

    def command_policy(self, params, action, n, policy):
        return self.token, np.zeros(6), None

    def twist_of(self, coeffs):
        return np.zeros(6)


class TokenController:
    """The environment's side, as `control.outer.SplitController` (whose
    `observe` needs four rotors and cannot be used on this body): the law is
    asked at the evaluator's 25 Hz with `TriphibianEnv.observation`, its
    action is held, and `act(u)` gives the joint targets at every step."""

    def __init__(self, env, law_, act):
        self.env, self.law, self.inner = env, law_, act
        self.u = np.zeros(IL.N_ACT)
        self.t_last = -1.0
        self.token = env.held_still_params()
        self._orig = env.cpg.command
        env.cpg.command = self.command
        b = _Basis(self.token)
        self.controller = Controller(params=self.token, policy=self, bases={"air": b, "water": b})

    def act(self, obs):
        self.u = self.law(np.asarray(obs, float), float(self.env.data.time))
        return np.zeros(1)

    def command(self, params, t):
        if params is not self.token:
            return self._orig(params, t)
        # A new rollout (the clock went back): forget the held action.  The
        # evaluator asks for a pose before its first decision; without this the
        # second half of an antipodal pair was posed by the first half's last
        # action and the two halves did not start from one state (found on the
        # frozen-heading arm, 2026-11-12; results/landwalk_before_fix.jsonl).
        now = float(self.env.data.time)
        if now < self.t_last:
            self.u = np.zeros(IL.N_ACT)
        self.t_last = now
        return self.inner(self.u)

    def close(self):
        self.env.cpg.command = self._orig


class Reader:
    """Wraps `env.step` on the instance; reads, changes nothing.  Per step:
    the ground's vertical force by what it enters through (wing and thumb
    bodies, the hull, the tail), whether any hull geometry touches, the
    package's `ground_support`, and every actuator's force."""

    def __init__(self, env, plan: Plan):
        self.env, self.plan = env, plan
        self.rows = []          # (half, t, kind, late, fz_wing, fz_hull, fz_tail, hull_touch, support, x, y)
        self.fmax = np.zeros(plan.n_act)
        self.emax = np.zeros(plan.n_act)
        self.half, self._last_t = -1, 1e18
        self._step = env.step
        env.step = self.step

    def close(self):
        if "step" in self.env.__dict__:
            del self.env.__dict__["step"]

    def contact(self):
        env, pl = self.env, self.plan
        m, d = env.model, env.data
        f6 = np.zeros(6)
        fz = {"wing": 0.0, "hull": 0.0, "tail": 0.0}
        touch = False
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 == 0) == (b2 == 0):
                continue
            env._mj.mj_contactForce(m, d, i, f6)
            f = float(np.asarray(c.frame, float).reshape(3, 3).T[2] @ f6[:3])
            f = f if b1 == 0 else -f
            k = pl.kind[b2 if b1 == 0 else b1]
            fz[k] += f
            if k == "hull" and (c.dist <= 0.0 or f6[0] > 0.0):
                touch = True
        return fz, touch

    def step(self, target):
        env = self.env
        alive = self._step(target)
        d = env.data
        t = float(d.time)
        if t < self._last_t:
            self.half += 1
        self._last_t = t
        kind, late = "", False
        task = env._active_task
        if task is not None:
            frac = (t - env._seg_t0) / env._seg_T
            i = task.index_at(min(frac, 0.999999))
            ph = task.phases[i]
            end = task.phases[i + 1].start if i + 1 < len(task.phases) else 1.0
            kind, late = f"{task.domain}:{ph.kind}", frac >= 0.5 * (ph.start + end)
        fz, touch = self.contact()
        self.fmax = np.maximum(self.fmax, np.abs(d.actuator_force[: self.plan.n_act]))
        self.emax = np.maximum(self.emax, np.abs(np.asarray(target, float) - d.qpos[env._act_qadr]))
        p = d.xpos[env.root_body]
        af = np.abs(d.actuator_force[: self.plan.n_act])
        self.rows.append((self.half, t, kind, late, fz["wing"], fz["hull"], fz["tail"], touch,
                          float(env.ground_support()), float(p[0]), float(p[1]),
                          float(af[self.plan.i_stroke].max()), float(af[self.plan.i_feather].max()),
                          float(af[self.plan.tail].max()) if self.plan.tail else 0.0))
        return alive

    def window(self, pred) -> dict:
        """Readings over the steps ``pred`` selects, per half and the worse half.
        Wing share is missing (None), not 0, when the ground carried nothing."""
        halves = []
        for h in range(self.half + 1):
            sel = [r for r in self.rows if r[0] == h and pred(r)]
            if not sel:
                halves.append(None)
                continue
            w, hu, ta = (float(sum(r[k] for r in sel)) for k in (4, 5, 6))
            tot = w + hu + ta
            halves.append({"n": len(sel), "ground_support": float(np.mean([r[8] for r in sel])),
                           "wing_share": (w / tot) if tot > 1e-9 else None,
                           "tail_share": (ta / tot) if tot > 1e-9 else None,
                           "hull_contact": float(np.mean([r[7] for r in sel])),
                           "airborne": float(np.mean([(r[4] + r[5] + r[6]) <= 1e-9 for r in sel])),
                           "torque_stroke": float(max(r[11] for r in sel)),
                           "torque_feather": float(max(r[12] for r in sel)),
                           "torque_tail": float(max(r[13] for r in sel))})
        ok = [x for x in halves if x is not None]

        def worst(key, fn):
            v = [x[key] for x in ok]
            return None if (not v or any(z is None for z in v)) else fn(v)
        return {"halves": halves, "ground_support": worst("ground_support", min),
                "wing_share": worst("wing_share", min), "hull_contact": worst("hull_contact", max),
                "tail_share": worst("tail_share", max), "airborne": worst("airborne", max),
                "torque_stroke": worst("torque_stroke", max), "torque_feather": worst("torque_feather", max),
                "torque_tail": worst("torque_tail", max)}


def torque_by_joint(env, reader) -> dict:
    out = {}
    for k, a in enumerate(env.act_names):
        role = "feather" if a.endswith("_f") else ("shoulder_stroke" if "wing" in a else "tail")
        out[role] = max(out.get(role, 0.0), float(reader.fmax[k]))
    out["servo_error_max_rad"] = float(reader.emax.max())
    return out


def controller(env, plan, arm, gait):
    g = Gait(env, plan, gait)
    lw = {"walk": Walk(gait), "propped": propped, "frozen": FrozenHeading(env, gait)}[arm]
    return TokenController(env, lw, Slewed(env, g.act).act)


def row_land(layout, arm, seed, gait=None, body=None) -> dict:
    t0 = time.time()
    gait = dict(GAITS[layout], **(gait or {}))
    env = make_env(layout, seed, **(body or {}))
    plan = Plan(env)
    ideal_servos(env, gait.get("servo"))
    ctl = controller(env, plan, arm, gait)
    row = {"kind": "land", "layout": layout, "arm": arm, "seed": seed, "gait": gait,
           "body": body or {}, "mass_kg": float(env.p.mass)}
    rd = Reader(env, plan)
    try:
        dom = Domain.LAND
        sseed = _scatter_seed(seed, dom)
        task = schedule_for(dom, np.random.default_rng(task_seed(sseed)))
        seg = run_segment(env, dom, SEGMENT_SECONDS, ctl.controller, scatter_seed=sseed, task=task)
        mm = seg.measurements
        row.update(score=float(seg.competence), failure=seg.failure, bad_qacc=int(seg.bad_qacc),
                   survived=bool(seg.survived),
                   parts={k: round(float(v), 6) for k, v in dict(seg.parts).items()},
                   m={k: round(float(mm[k]), 6) for k in (
                       "walk_progress", "walk_tracking", "stop_score", "stop_drift", "upright",
                       "contact_fraction", "measured_land_speed", "pair_along_mean") if k in mm},
                   walk=rd.window(lambda r: r[2] == "land:cruise"),
                   walk_late=rd.window(lambda r: r[2] == "land:cruise" and r[3]),
                   stop=rd.window(lambda r: r[2] == "land:stop" and r[3]),
                   torque=torque_by_joint(env, rd),
                   booked_wh=float(env.budget.total_j) / 3600.0)
    finally:
        rd.close()
        ctl.close()
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def row_speed(layout, seed, gait=None, body=None, seconds=8.0) -> dict:
    """The gait on one straight heading along the shore (level), 8 s."""
    gait = dict(GAITS[layout], **(gait or {}))
    env = make_env(layout, seed, **(body or {}))
    plan = Plan(env)
    ideal_servos(env, gait.get("servo"))
    g = Slewed(env, Gait(env, plan, gait).act)
    env.reset(Domain.LAND, randomise=True)
    d = env.data
    # Facing along the shore (level ground), a few degrees off: the heading
    # commanded below is then a straight line, not a turn.
    yaw0 = math.pi / 2 + float(np.random.default_rng([int(seed), 0x59415721]).normal(0.0, 0.1))
    d.qpos[3:7] = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))
    env._mj.mj_forward(env.model, d)
    rd = Reader(env, plan)
    u = np.zeros(IL.N_ACT)
    u[IL.U_POSE] = 1.0
    target = math.pi / 2                                   # world +y: along the shore
    bad = env._mj.mjtWarning.mjWARN_BADQACC
    try:
        for _ in range(int(1.0 / DT)):                     # settle, standing
            env.step(g.act(u))
        p0 = d.xpos[env.root_body].copy()
        t_start = float(d.time)
        for _ in range(int(seconds / DT)):
            R = d.xmat[env.root_body].reshape(3, 3)
            yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
            e = (target - yaw + math.pi) % (2.0 * math.pi) - math.pi
            u[IL.U_DRIVE], u[IL.U_TURN] = steer(e, gait)
            env.step(g.act(u))
        p1 = d.xpos[env.root_body].copy()
        w = rd.window(lambda r: r[1] > t_start + 1.0)      # steady gait: not the start from standing
    finally:
        rd.close()
    v = float(p1[1] - p0[1]) / seconds
    w.pop("halves")
    return {"kind": "speed", "layout": layout, "seed": seed, "gait": gait, "body": body or {}, "speed_mps": v,
            "hull_lengths_per_s": v / 0.16, "across_m": float(p1[0] - p0[0]),
            "bad_qacc": int(d.warning[bad].number), "window": w, "torque": torque_by_joint(env, rd)}


def row_turn(layout, drive, turn, gait=None, body=None, seconds=4.0, uphill=False) -> dict:
    """A tuning probe: the gait at a fixed drive and turn from standing, facing
    along the shore (or uphill), 4 s.  Yaw gained, distance along the nose."""
    gait = dict(GAITS[layout], **(gait or {}))
    env = make_env(layout, 100, **(body or {}))
    plan = Plan(env)
    ideal_servos(env, gait.get("servo"))
    g = Slewed(env, Gait(env, plan, gait).act)
    env.reset(Domain.LAND, randomise=False)
    d = env.data
    if not uphill:
        d.qpos[3:7] = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        env._mj.mj_forward(env.model, d)
    rd = Reader(env, plan)
    u = np.zeros(IL.N_ACT)
    u[IL.U_POSE] = 1.0
    try:
        for _ in range(int(1.0 / DT)):
            env.step(g.act(u))
        u[IL.U_DRIVE], u[IL.U_TURN] = drive, turn
        yaw = lambda: math.atan2(float(d.xmat[env.root_body][3]), float(d.xmat[env.root_body][0]))   # noqa: E731
        prev, tot, p0 = yaw(), 0.0, d.xpos[env.root_body].copy()
        nose = d.xmat[env.root_body].reshape(3, 3)[:2, 0].copy()
        t0 = float(d.time)
        for _ in range(int(seconds / DT)):
            env.step(g.act(u))
            y = yaw()
            tot += (y - prev + math.pi) % (2.0 * math.pi) - math.pi
            prev = y
        w = rd.window(lambda r: r[1] > t0 + 1.0)
    finally:
        rd.close()
    w.pop("halves")
    return {"kind": "turn", "layout": layout, "gait": gait, "body": body or {}, "drive": drive,
            "turn": turn, "uphill": uphill, "yaw_deg": math.degrees(tot),
            "along_nose_m": float((d.xpos[env.root_body] - p0)[:2] @ nose),
            "up_z": float(d.xmat[env.root_body][8]), "window": w,
            "bad_qacc": int(d.warning[env._mj.mjtWarning.mjWARN_BADQACC].number)}


def _turn(args):
    return row_turn(*args)


def rows_air() -> list:
    """What the walking hardware costs on the held rig (`rig.run`)."""
    import rig
    rows = []
    for name, body in (("bare A wing", BARE["A"]), ("A", LAYOUTS["A"]),
                       ("bare B wing, one pair", BARE["B"]), ("B", LAYOUTS["B"])):
        # body upright (nose up): lift is the world-vertical force; of the two
        # feather signs one lifts and the other pushes down (F0 prediction 1)
        hov = [rig.run(dict(body, stroke_range=1.45), 0.0, 90.0, 12.0, 130.0, sg * 45.0, 0.0)
               for sg in (1.0, -1.0)]
        h = max(hov, key=lambda r: r["Fz"])
        cr = rig.trim(dict(body), 8.0, 8.0, 80.0, 20.0)
        rows.append({"kind": "air", "name": name, "body": body, "mass_kg": h["mass_kg"],
                     "hover": {k: h[k] for k in ("Fx", "Fz", "Fy", "aero_power_W", "Psi_deg", "weight", "finite")},
                     "hover_lift_N": h["Fz"], "hover_other_sign_N": min(hov, key=lambda r: r["Fz"])["Fz"],
                     "cruise": {k: cr[k] for k in ("psi0_deg", "lift_over_weight", "Fx", "Fz", "aero_power_W", "trimmed", "evals")}})
    return rows


def cmd_geometry(a) -> int:
    for layout in ("A", "B"):
        env = make_env(layout, 0)
        plan = Plan(env)
        m, d, mj = env.model, env.data, env._mj
        R, p0 = d.xmat[plan.hull].reshape(3, 3), d.xpos[plan.hull]
        print(f"layout {layout}: all-up {env.p.mass * 1000:.2f} g, actuators {env.act_names}")
        for b in range(1, m.nbody):
            n = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b)
            print(f"  {n:14s} {plan.kind[b]:4s} parent {mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.body_parentid[b])):12s}"
                  f" origin {np.round(R.T @ (d.xpos[b] - p0), 3).tolist()} centre {np.round(R.T @ (d.xipos[b] - p0), 3).tolist()}"
                  f" mass {m.body_mass[b] * 1000:.2f} g")
        for j in range(1, m.njnt):
            print(f"  joint {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j):14s} axis {np.round(R.T @ d.xaxis[j], 2).tolist()}"
                  f" anchor {np.round(R.T @ (d.xanchor[j] - p0), 3).tolist()}")
        for w in plan.wings:
            print(f"  wing act {env.act_names[w['stroke']]}: side {w['side']:+.0f} x {w['x']:.3f} thumb tip {w['tip']}"
                  f" stroke sign {w['s_stroke']:+.0f} feather sign {w['s_feather']:+.0f}")
        print(f"  tail sign {plan.tail_sign:+.0f}; com x {float((R.T @ ((m.body_mass[1:, None] * d.xipos[1:]).sum(0) / m.body_mass[1:].sum() - p0))[0]):.4f}")
    return 0


def _hold(env, plan, targets, seconds, reader_pred=None):
    rd = Reader(env, plan)
    try:
        for _ in range(int(seconds / DT)):
            env.step(targets)
        t1 = float(env.data.time)
        w = rd.window(lambda r: r[1] > t1 - 0.5)
    finally:
        rd.close()
    w.pop("halves")
    # which bodies the ground is pushing on at the end, N, and how far the body has rolled
    m, d, f6, by = env.model, env.data, np.zeros(6), {}
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (b1 == 0) == (b2 == 0):
            continue
        env._mj.mj_contactForce(m, d, i, f6)
        n = env._mj.mj_id2name(m, env._mj.mjtObj.mjOBJ_BODY, b2 if b1 == 0 else b1)
        by[n] = round(by.get(n, 0.0) + float(f6[0]), 3)
    R = d.xmat[env.root_body].reshape(3, 3)
    w["touching_N"] = by
    w["roll_deg"] = round(math.degrees(math.atan2(float(R[2, 1]), float(R[2, 2]))), 1)
    return w


def cmd_probes(a) -> int:
    """The readings proved before any walk number is read."""
    out = []
    for layout in ("A", "B"):
        env = make_env(layout, 0)
        plan = Plan(env)
        ideal_servos(env)
        g = Gait(env, plan, GAITS[layout])
        u = np.zeros(IL.N_ACT)
        # 1. resting on the hull: wings raised clear, tail level
        env.reset(Domain.LAND, randomise=False)
        up = np.zeros(plan.n_act)
        for w in plan.wings:
            up[w["stroke"]] = w["s_stroke"] * -0.6
        for k in plan.tail:
            up[k] = plan.tail_sign * -0.6
        out.append({"probe": "resting on the hull (wings and tail raised)", "layout": layout, **_hold(env, plan, up, 2.0)})
        # 2. propped on the thumbs, hull clear
        env.reset(Domain.LAND, randomise=False)
        u[IL.U_POSE] = 1.0
        out.append({"probe": "propped on the thumbs (stance pose)", "layout": layout, **_hold(env, plan, g.act(u), 2.0)})
        # 3. airborne
        env.reset(Domain.AIR, randomise=False)
        out.append({"probe": "airborne (0.3 s of free fall from 30 m)", "layout": layout, **_hold(env, plan, g.act(u), 0.6)})
    for r in out:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    return 0


def _seeds(s):
    lo, hi = (int(x) for x in s.split("-")) if "-" in s else (int(s), int(s))
    return list(range(lo, hi + 1))


def brief(r) -> str:
    b = {k: r.get(k) for k in ("kind", "layout", "arm", "seed", "score", "failure", "bad_qacc", "speed_mps")}
    for k in ("walk", "walk_late", "window"):
        if k in r:
            b[k] = {kk: (None if vv is None else round(vv, 3)) for kk, vv in r[k].items() if kk != "halves"}
    if "m" in r:
        b["m"] = {k: round(v, 3) for k, v in r["m"].items()}
    if "torque" in r:
        b["torque"] = {k: round(v, 4) for k, v in r["torque"].items()}
    return json.dumps({k: v for k, v in b.items() if v is not None})


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("geometry")
    sub.add_parser("probes")
    r = sub.add_parser("run")
    r.add_argument("--layout", default="A,B")
    r.add_argument("--arm", default="walk")
    r.add_argument("--seeds", default="100-103")
    r.add_argument("--gait", default="{}")
    r.add_argument("--body", default="{}")
    r.add_argument("--out", default=str(HERE / "trials_F2.jsonl"))
    r.add_argument("--note", default="")
    r.add_argument("--jobs", type=int, default=4)
    s = sub.add_parser("speed")
    s.add_argument("--layout", default="A,B")
    s.add_argument("--seeds", default="0-2")
    s.add_argument("--gait", default="{}")
    s.add_argument("--body", default="{}")
    s.add_argument("--out", default=str(HERE / "trials_F2.jsonl"))
    s.add_argument("--note", default="")
    s.add_argument("--jobs", type=int, default=3)
    s.add_argument("--fast", action="store_true", help="each layout's FAST setting")
    tn = sub.add_parser("turn")
    tn.add_argument("--layout", default="A")
    tn.add_argument("--gait", default="{}")
    tn.add_argument("--body", default="{}")
    tn.add_argument("--out", default=str(HERE / "trials_F2.jsonl"))
    tn.add_argument("--note", default="")
    tn.add_argument("--jobs", type=int, default=5)
    r2 = sub.add_parser("air")
    r2.add_argument("--out", default=str(HERE / "trials_F2.jsonl"))
    a = ap.parse_args()
    if a.cmd == "geometry":
        return cmd_geometry(a)
    if a.cmd == "probes":
        return cmd_probes(a)
    out = Path(a.out).resolve()
    if a.cmd == "air":
        rows = rows_air()
    else:
        import multiprocessing as mp
        gait = json.loads(a.gait)
        if a.cmd == "turn":
            jobs = [(_turn, (a.layout, dr, tu, gait, json.loads(a.body), 4.0, up))
                    for dr, tu, up in ((0.0, 1.0, False), (0.0, -1.0, False), (1.0, 0.0, False),
                                       (-1.0, 0.0, False), (1.0, 0.0, True))]
        elif a.cmd == "run":
            jobs = [(_land, (L, arm, sd, gait, json.loads(a.body))) for L in a.layout.split(",")
                    for arm in a.arm.split(",") for sd in _seeds(a.seeds)]
        else:
            jobs = [(_speed, (L, sd, dict(FAST[L], **gait) if a.fast else gait, json.loads(a.body)))
                    for L in a.layout.split(",") for sd in _seeds(a.seeds)]
        with mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
            rows = pool.map(_call, jobs, chunksize=1)
    with open(out, "a") as fh:
        for row in rows:
            if getattr(a, "note", ""):
                row["note"] = a.note
            fh.write(json.dumps(row, default=float) + "\n")
            if row["kind"] == "turn":
                w = row["window"]
                print(f"{row['layout']} drive {row['drive']:+.0f} turn {row['turn']:+.0f} {'uphill' if row['uphill'] else 'level '}"
                      f" yaw {row['yaw_deg']:+7.1f} deg along {row['along_nose_m']:+.3f} m  wing {w['wing_share'] or 0:.2f}"
                      f" hull {w['hull_contact']:.2f} air {w['airborne']:.2f} upz {row['up_z']:.2f} tq {w['torque_stroke']:.2f} {w['torque_feather']:.2f}", flush=True)
                continue
            print(brief(row) if row["kind"] != "air" else json.dumps(row, default=float)[:700], flush=True)
    return 0


def _land(args):
    return row_land(*args)


def _speed(args):
    return row_speed(*args)


def _call(job):
    fn, args = job
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
