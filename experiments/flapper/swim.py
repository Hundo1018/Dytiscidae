"""F3b: does the two-pair machine (F2's layout B) swim when its body is free?
Spec: PREREG_F3b.md (frozen).  No learning, no optimiser; strokes and the
steering law tuned by hand on seeds 100-103, scored on seeds 0-7.

    cd /home/hundo/Projects/Dytiscidae/dytiscidae; export MUJOCO_GL=disable PYTHONPATH=.
    F=$PWD/experiments/flapper
    .venv/bin/python $F/swim.py facts                                   # weight, buoyancy, the body left alone
    .venv/bin/python $F/swim.py run --stroke S1,S2,S3 --arm swim,still,frozen --seeds 0-7 --out $F/results/swim.jsonl
    .venv/bin/python $F/swim.py straight --stroke S1,S2,S3 --seeds 0-2 --out $F/results/swim.jsonl

The harness is F2's (`landwalk.py`): the body builder, the ideal servos, the
unlimited pack and the token controller that forgets its held action when the
clock goes back.  Nothing in `landwalk.py` or `body.py` is changed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import landwalk as LW  # noqa: E402  (sets the 5 g avionics at import)
from dytiscidae.envs.evaluate import _scatter_seed, run_segment  # noqa: E402
from dytiscidae.envs.tasks import schedule_for, task_seed  # noqa: E402
from dytiscidae.envs.triphibian import Domain  # noqa: E402

G = 9.80665
DT = LW.DT
SEGMENT_SECONDS = 8.0
HULL = 0.16

# the action the law hands the stroke generator
U_DRIVE, U_TURN, U_LIFT, U_PITCH, U_ROLL, U_BEAT, U_ROW, U_FOLD = 0, 1, 2, 3, 4, 5, 6, 7
N_U = 8

#: Strokes (PREREG section 1), set by hand on seeds 100-103 (trials_F3b.jsonl).
#: Angles in rad.  Positive stroke lowers the wing tip, positive feather raises
#: the leading edge.  Phases are in beats: fore-left, fore-right, hind-left,
#: hind-right.
STROKES = {
    "S1": dict(f=1.0, A=1.0, Psi=0.6, phases=(0.0, 0.0, 0.0, 0.0)),
    "S2": dict(f=1.0, A=1.0, Psi=0.6, phases=(0.0, 0.0, 0.5, 0.5)),
    "S3": dict(f=6.0, A=0.0, Psi=0.4, phases=(0.0, 0.0, 0.5, 0.5)),
    # Not one of the three: fore-left with hind-right against fore-right with
    # hind-left, so the reactions cancel in force, in pitch and in roll.
    "S2x": dict(f=1.0, A=0.5, Psi=0.6, phases=(0.0, 0.5, 0.5, 0.0)),
    # The same at half the beat rate, for the torque it asks.
    "S2x_slow": dict(f=0.5, A=0.5, Psi=0.6, phases=(0.0, 0.5, 0.5, 0.0)),
}
#: The law's gains, shared by the strokes unless a stroke overrides one.
LAW = dict(
    k_turn=1.5, turn_max=1.0, e_go=0.4,        # heading: F2's switch (turn on the spot beyond e_go)
    k_depth=1.0, k_vz=1.0, lift_max=1.0,       # depth error (m, + too deep) and climb rate -> lift command
    c_lift=0.0,                                # share of the lift command given to a feather offset under way
    dive=0.2,                                  # rad of pitch reference per unit lift command under way
    k_pitch=1.0, k_roll=0.0,                   # attitude -> trim commands
    off_lift=0.3, off_pitch=0.3, off_roll=0.2,  # rad of feather / stroke offset per unit command
    hover=0.0, k_row=0.0,                      # rowing for lift in a hold (unused: it tumbles the body, trials_F3b)
    k_brake=2.0,                               # hold: drive against the way still on, per m/s
    fold=1.2,                                  # rad the wings are raised at full fold
    slew=5.0,                                  # rad/s, the fastest a joint target may move
)


def _wrap(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


class Swim:
    """``law(o, t) -> u`` from `TriphibianEnv.observation` alone: the heading
    error steers by the left-right difference, the depth error and the climb
    rate set the lift (a feather offset under way, rowing in a hold), the
    gravity vector levels the body.  Two model constants turn the observed
    velocity into the root frame, as in E0s."""

    def __init__(self, env, law: dict):
        self.g = dict(law)
        m = env.model
        R_i = np.zeros(9)
        env._mj.mju_quat2Mat(R_i, np.asarray(m.body_iquat[env.root_body], float))
        self.R_i = R_i.reshape(3, 3)

    def read(self, o):
        gb = np.asarray(o[6:9], float)                      # gravity in the body frame, unit, down
        v = self.R_i @ (5.0 * np.asarray(o[0:3], float))    # body-frame velocity (the offset term is millimetres)
        return {"pitch": math.asin(float(np.clip(-gb[0], -1.0, 1.0))),
                "roll": math.atan2(float(-gb[1]), float(-gb[2])),
                "vz": float(-(v @ gb)), "vx": float(v[0])}

    def core(self, o, e_head, e_depth, cruise, hold) -> np.ndarray:
        g, r = self.g, self.read(o)
        u = np.zeros(N_U)
        lift = float(np.clip(g["k_depth"] * e_depth - g["k_vz"] * r["vz"], -g["lift_max"], g["lift_max"]))
        # Pitch is held to a reference by the fore-hind feather difference.  The
        # sign is the measured one (trials_F3b "straight trials 3": with it the
        # body swims level, 3.4 deg of swing; with the other it tumbles).
        pitch_ref = 0.0
        u[U_ROLL] = -g["k_roll"] * r["roll"]
        if cruise:
            direction = 1.0
            e = e_head
            if abs(e) > 0.5 * math.pi:
                e, direction = _wrap(e - math.pi), -1.0
            u[U_BEAT] = 1.0
            if abs(e) <= g["e_go"]:
                u[U_DRIVE] = direction
                u[U_TURN] = float(np.clip(g["k_turn"] * e, -g["turn_max"], g["turn_max"]))
                u[U_LIFT] = g["c_lift"] * lift
                # depth under way: point the body where it should go
                pitch_ref = direction * g["dive"] * lift
            else:
                u[U_TURN] = math.copysign(g["turn_max"], e)
        elif hold:
            u[U_BEAT] = 1.0
            u[U_DRIVE] = float(np.clip(-g["k_brake"] * r["vx"], -1.0, 1.0))
            u[U_ROW] = float(np.clip(g["hover"] + g["k_row"] * lift, -1.0, 1.0))
        u[U_PITCH] = float(np.clip(g["k_pitch"] * (r["pitch"] - pitch_ref), -1.0, 1.0))
        return u

    def __call__(self, o, t) -> np.ndarray:
        e_depth = 3.0 * math.atanh(float(np.clip(o[14], -0.999999, 0.999999)))
        return self.core(o, math.atan2(o[23], o[22]), e_depth, o[19] > 0.5, o[20] > 0.5)


class Frozen:
    """The stroke with its heading and depth command frozen at the first seen
    (a null arm; the state is the arm's)."""

    def __init__(self, env, swim: Swim):
        self.env, self.swim, self.head, self.depth = env, swim, None, None

    def __call__(self, o, t):
        env = self.env
        R = env.data.xmat[env.root_body]
        yaw = math.atan2(float(R[3]), float(R[0]))
        e_depth = 3.0 * math.atanh(float(np.clip(o[14], -0.999999, 0.999999)))
        depth = float(env.depth())
        if self.depth is None and (o[19] > 0.5 or o[20] > 0.5):
            self.depth = depth - e_depth
        if self.head is None and o[19] > 0.5:
            self.head = yaw + math.atan2(o[23], o[22])
        e_head = 0.0 if self.head is None else _wrap(self.head - yaw)
        e_d = 0.0 if self.depth is None else depth - self.depth
        return self.swim.core(o, e_head, e_d, o[19] > 0.5, o[20] > 0.5)


class Stroke:
    """``act(u) -> joint targets`` for the four wings, a function of ``u`` and
    the clock.

        stroke_i  = roll trim + A sin(th_i)
        feather_i = -g_i (Psi cos(th_i) + lift and pitch offsets)     g_i = drive -+ turn

    With the leading edge down on the downstroke (g > 0) the wing's force leans
    forward; g < 0 swims backward.  Lift and pitch are feather offsets: all four
    leading edges up to climb, fore up and hind down to raise the nose.
    """

    def __init__(self, env, plan, stroke: dict, law: dict):
        self.env, self.plan, self.s, self.law = env, plan, dict(stroke), dict(law)

    def act(self, u) -> np.ndarray:
        s, lw, pl = self.s, self.law, self.plan
        out = np.zeros(pl.n_act)
        t = float(self.env.data.time)
        beat = float(u[U_BEAT])
        for i, w in enumerate(pl.wings):
            fore = 1.0 if i < 2 else -1.0
            th = 2.0 * math.pi * (s["f"] * t + s["phases"][i])
            g = float(np.clip(u[U_DRIVE] - w["side"] * u[U_TURN], -1.0, 1.0))
            # fold: all four wings raised into a V, to sink faster through the water
            stroke = s.get("stroke0", 0.0) - lw["fold"] * float(u[U_FOLD]) \
                + lw["off_roll"] * w["side"] * float(u[U_ROLL]) + beat * s["A"] * math.sin(th)
            # The lift and pitch offsets ride on the drive: leading edge down to
            # climb when driving forward, up when driving backward.  Both signs are
            # measured, not reasoned (trials_F3b "straight trials 1, 3, 5": asked
            # 1.5 m deeper the body rose 0.81 m with the other lift sign; the pitch
            # loop that levels a forward swim tumbles a backward one).
            feather = -g * (beat * s["Psi"] * math.cos(th) + lw["off_lift"] * float(u[U_LIFT])
                            + lw["off_pitch"] * fore * float(u[U_PITCH]))
            if s.get("row", 0.0) and beat > 0.0:
                # rowing for lift with no way on: the wing is turned edge-on for the
                # half of the beat that would push the wrong way (fore and hind
                # turned opposite ways, so the fore-aft forces cancel)
                up = max(0.0, -math.cos(th)) if u[U_ROW] >= 0.0 else max(0.0, math.cos(th))
                feather += fore * s["row"] * abs(float(u[U_ROW])) * up
            out[w["stroke"]] = w["s_stroke"] * float(np.clip(stroke, -1.3, 1.3))
            out[w["feather"]] = w["s_feather"] * float(np.clip(feather, -0.95, 0.95))
        return out


class Slew:
    """`act(u)` with its targets rate-limited: F2's `landwalk.Slewed` with a
    rate fit for water (the stroke's own fastest motion is about 4 rad/s; at
    F2's 60 rad/s a switch of the drive asked the feather for 10-20 N m)."""

    def __init__(self, env, act, rate: float):
        self.env, self.inner, self.rate, self.prev, self.t = env, act, float(rate), None, -1.0

    def act(self, u):
        t = float(self.env.data.time)
        tgt = self.inner(u)
        if self.prev is None or t < self.t or t <= 0.0:
            self.prev = np.asarray(self.env.data.qpos[self.env._act_qadr], float).copy()
        self.t = t
        self.prev = self.prev + np.clip(tgt - self.prev, -self.rate * DT, self.rate * DT)
        return self.prev.copy()


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
        p = d.xpos[env.root_body]
        self.rows.append((float(d.time), float(p[0]), float(p[1]), float(p[2]),
                          math.asin(float(np.clip(R[6], -1.0, 1.0))),          # nose-up pitch
                          math.atan2(float(R[7]), float(R[8])),                # roll
                          float(af[self.plan.i_stroke].max()), float(af[self.plan.i_feather].max()),
                          math.atan2(float(R[3]), float(R[0])), float(R[8])))
        return alive


def setup(seed, stroke_name, stroke=None, law=None):
    env = LW.make_env("B", seed)
    plan = LW.Plan(env)
    LW.ideal_servos(env, (stroke or {}).get("servo"))
    s = dict(STROKES[stroke_name], **(stroke or {}))
    lw = dict(LAW, **(law or {}))
    return env, plan, s, lw


def row_open(stroke_name, u, stroke=None, law=None, seconds=6.0, seed=100) -> dict:
    """A tuning probe: the stroke at a fixed action from rest, no law."""
    env, plan, s, lw = setup(seed, stroke_name, stroke, law)
    g = Slew(env, Stroke(env, plan, s, lw).act, lw["slew"])
    env.reset(Domain.WATER, randomise=False)
    d = env.data
    rec = Recorder(env, plan)
    uu = np.zeros(N_U)
    uu[: len(u)] = u
    bad = env._mj.mjtWarning.mjWARN_BADQACC
    try:
        for _ in range(int(seconds / DT)):
            env.step(g.act(uu))
    finally:
        rec.close()
    return {"kind": "open", "stroke": stroke_name, "s": s, "law": lw, "u": list(map(float, uu)),
            **beat_reads(rec.rows, s["f"], seconds), "bad_qacc": int(d.warning[bad].number)}


def row_swim(stroke_name, arm, seed, stroke=None, law=None) -> dict:
    """One scored water segment (antipodal pair), unmodified."""
    from dytiscidae.envs.evaluate import Controller
    t0 = time.time()
    env, plan, s, lw = setup(seed, stroke_name, stroke, law)
    row = {"kind": "swim", "stroke": stroke_name, "arm": arm, "seed": seed, "s": s, "law": lw}
    ctl = None
    if arm == "still":
        controller = Controller(params=env.held_still_params(), policy=None, bases=None)
    else:
        swim = Swim(env, lw)
        ctl = LW.TokenController(env, swim if arm == "swim" else Frozen(env, swim),
                                 Slew(env, Stroke(env, plan, s, lw).act, lw["slew"]).act)
        controller = ctl.controller
    rec = Recorder(env, plan)
    try:
        dom = Domain.WATER
        sseed = _scatter_seed(seed, dom)
        task = schedule_for(dom, np.random.default_rng(task_seed(sseed)))
        seg = run_segment(env, dom, SEGMENT_SECONDS, controller, scatter_seed=sseed, task=task)
        mm = seg.measurements
        a = np.array(rec.rows)
        row.update(score=float(seg.competence), failure=seg.failure, bad_qacc=int(seg.bad_qacc),
                   survived=bool(seg.survived),
                   parts={k: round(float(v), 6) for k, v in dict(seg.parts).items()},
                   m={k: round(float(mm[k]), 6) for k in (
                       "cruise_progress", "cruise_tracking", "cruise_depth_hold", "pair_along_mean",
                       "pair_along", "pair_across", "hold_score", "hold_depth_error", "hold_drift",
                       "hold_sink_rate", "cmd_depth", "cmd_heading", "task_hold_first", "upright",
                       "max_depth") if k in mm},
                   torque_stroke=float(a[:, 6].max()), torque_feather=float(a[:, 7].max()),
                   torque_stroke_after_1s=float(a[a[:, 0] > 1.0][:, 6].max()),
                   torque_feather_after_1s=float(a[a[:, 0] > 1.0][:, 7].max()),
                   up_z_min=float(a[:, 9].min()), z_end=float(a[-1, 3]))
    finally:
        rec.close()
        if ctl is not None:
            ctl.close()
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def row_straight(stroke_name, seed, stroke=None, law=None, dz=0.0, back=False, seconds=8.0) -> dict:
    """A straight level swim: the law holding the heading and the depth it starts with."""
    env, plan, s, lw = setup(seed, stroke_name, stroke, law)
    swim = Swim(env, lw)
    g = Slew(env, Stroke(env, plan, s, lw).act, lw["slew"])
    env.reset(Domain.WATER, randomise=True)
    d = env.data
    R = d.xmat[env.root_body]
    yaw0, depth0 = math.atan2(float(R[3]), float(R[0])), float(env.depth())
    if back:                                                  # the heading behind it: the law swims backward
        yaw0 = _wrap(yaw0 + math.pi)
    rec = Recorder(env, plan)
    bad = env._mj.mjtWarning.mjWARN_BADQACC
    u = np.zeros(N_U)
    try:
        for k in range(int(seconds / DT)):
            if k % 40 == 0:                                   # the evaluators' 25 Hz
                R = d.xmat[env.root_body]
                yaw = math.atan2(float(R[3]), float(R[0]))
                u = swim.core(env.observation(Domain.WATER), _wrap(yaw0 - yaw),
                              float(env.depth()) - depth0 - dz, True, False)
            env.step(g.act(u))
    finally:
        rec.close()
    out = beat_reads(rec.rows, s["f"], seconds)
    a = np.array(rec.rows)
    h = np.array([math.cos(yaw0), math.sin(yaw0)])
    late = a[a[:, 0] > seconds / 2.0]
    v = float((late[-1, 1:3] - late[0, 1:3]) @ h) / float(late[-1, 0] - late[0, 0])
    return {"kind": "straight", "stroke": stroke_name, "seed": seed, "s": s, "law": lw, "dz": dz, "back": back,
            "speed_along_heading_mps": v, "hull_lengths_per_s": v / HULL,
            "depth_change_m": float(env.depth()) - depth0, **out, "bad_qacc": int(d.warning[bad].number)}


def beat_reads(rows, f, seconds) -> dict:
    """Net motion over the run, and the hull's swing over a beat in its last third."""
    a = np.array(rows)
    late = a[a[:, 0] > a[-1, 0] - seconds / 3.0]
    n_beat = max(int(round(1.0 / (f * DT))), 1)
    heave, pitch = [], []
    for k in range(0, len(late) - n_beat + 1, n_beat):
        seg = late[k:k + n_beat]
        z = seg[:, 3] - np.linspace(seg[0, 3], seg[-1, 3], len(seg))        # the beat's own swing, drift taken out
        heave.append(float(z.max() - z.min()))
        pitch.append(float(np.degrees(seg[:, 4].max() - seg[:, 4].min())))
    yaw0 = late[0, 8]
    h = np.array([math.cos(yaw0), math.sin(yaw0)])
    dxy = late[-1, 1:3] - late[0, 1:3]
    T = float(late[-1, 0] - late[0, 0])
    return {"speed_mps": float(np.linalg.norm(dxy)) / T, "along_nose_mps": float(dxy @ h) / T,
            "vz_mps": float(late[-1, 3] - late[0, 3]) / T,
            "heave_m": float(np.mean(heave)) if heave else None,
            "pitch_swing_deg": float(np.mean(pitch)) if pitch else None,
            "pitch_end_deg": float(np.degrees(late[-1, 4])), "roll_end_deg": float(np.degrees(late[-1, 5])),
            "up_z_min": float(a[:, 9].min()),
            "yaw_rate_dps": float(np.degrees((late[-1, 8] - late[0, 8] + math.pi) % (2 * math.pi) - math.pi)) / T,
            "torque_stroke": float(late[:, 6].max()), "torque_feather": float(late[:, 7].max()),
            "torque_stroke_all": float(a[:, 6].max()), "torque_feather_all": float(a[:, 7].max())}


def cmd_facts(a) -> int:
    env = LW.make_env("B", 0)
    plan = LW.Plan(env)
    LW.ideal_servos(env)
    w = float(env.solver._dry_mass.sum()) * G
    b = float(env.medium.water.rho) * G * float(env.p.buoyant_volume)
    print(json.dumps({"dry_mass_kg": float(env.solver._dry_mass.sum()), "pipeline_mass_kg": float(env.p.mass),
                      "weight_N": w, "buoyancy_N": b, "net_up_N": b - w, "net_over_weight": (b - w) / w}))
    env.reset(Domain.WATER, randomise=False)
    d = env.data
    rec = Recorder(env, plan)
    for _ in range(6000):
        env.step(np.zeros(plan.n_act))
    rec.close()
    for t in (1.0, 3.0, 6.0):
        r = rec.rows[int(t / DT) - 1]
        print(json.dumps({"left alone, joints at 0, t": t, "dz_m": round(r[3] - rec.rows[0][3], 4),
                          "nose_up_deg": round(math.degrees(r[4]), 1), "up_z": round(r[9], 3)}))
    return 0


def _seeds(s):
    lo, hi = (int(x) for x in s.split("-")) if "-" in s else (int(s), int(s))
    return list(range(lo, hi + 1))


def _open(args):
    return row_open(*args)


def _swim(args):
    return row_swim(*args)


def _straight(args):
    return row_straight(*args)


def _call(job):
    return job[0](job[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("facts")
    o = sub.add_parser("open")
    o.add_argument("--stroke", default="S2")
    o.add_argument("--u", default="[[1,0,0,0,0,1]]", help="a list of actions")
    o.add_argument("--set", default="[{}]", help="a list of stroke overrides")
    o.add_argument("--law", default="{}")
    o.add_argument("--seconds", type=float, default=6.0)
    o.add_argument("--out", default=str(HERE / "trials_F3b.jsonl"))
    o.add_argument("--note", default="")
    o.add_argument("--jobs", type=int, default=6)
    for name in ("run", "straight"):
        r = sub.add_parser(name)
        r.add_argument("--stroke", default="S2x")
        r.add_argument("--arm", default="swim")
        r.add_argument("--seeds", default="100-103")
        r.add_argument("--set", default="{}")
        r.add_argument("--law", default="{}")
        r.add_argument("--out", default=str(HERE / "trials_F3b.jsonl"))
        r.add_argument("--note", default="")
        r.add_argument("--jobs", type=int, default=6)
        r.add_argument("--dz", type=float, default=0.0, help="straight: metres deeper than the start to hold")
        r.add_argument("--back", action="store_true", help="straight: swim backward")
    a = ap.parse_args()
    if a.cmd == "facts":
        return cmd_facts(a)
    import multiprocessing as mp
    if a.cmd == "open":
        jobs = [(_open, (st, u, sv, json.loads(a.law), a.seconds)) for st in a.stroke.split(",")
                for sv in json.loads(a.set) for u in json.loads(a.u)]
    if a.cmd == "run":
        jobs = [(_swim, (st, arm, sd, json.loads(a.set), json.loads(a.law))) for st in a.stroke.split(",")
                for arm in a.arm.split(",") for sd in _seeds(a.seeds)]
    if a.cmd == "straight":
        jobs = [(_straight, (st, sd, json.loads(a.set), json.loads(a.law), a.dz, a.back)) for st in a.stroke.split(",")
                for sd in _seeds(a.seeds)]
    with mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
        rows = pool.map(_call, jobs, chunksize=1)
    with open(Path(a.out).resolve(), "a") as fh:
        for row, job in zip(rows, jobs):
            if a.note:
                row["note"] = a.note
            fh.write(json.dumps(row, default=float) + "\n")
            if row["kind"] == "swim":
                m = row.get("m", {})
                print(f"{row['stroke']} {row['arm']} {row['seed']} score {row.get('score', -1):.3f} {row.get('failure', '')!r} bad {row.get('bad_qacc')}"
                      f" gate {row['parts'].get('gate', 0):.2f} prog {m.get('cruise_progress', -1):.2f} along {m.get('pair_along_mean', -9):+.2f} across {m.get('pair_across', -1):.2f}"
                      f" hold {m.get('hold_score', -1):.2f} (derr {m.get('hold_depth_error', -1):.2f} drift {m.get('hold_drift', -1):.2f} sink {m.get('hold_sink_rate', -1):.3f}) cmd {m.get('cmd_depth', 0):.2f} zend {row['z_end']:.2f}"
                      f" upz>{row['up_z_min']:.2f} tq {row['torque_stroke_after_1s']:.2f} {row['torque_feather_after_1s']:.2f}", flush=True)
            if row["kind"] == "straight":
                print(f"{row['stroke']} seed {row['seed']}: {row['speed_along_heading_mps']:+.3f} m/s = {row['hull_lengths_per_s']:.2f} hull lengths/s, depth change {row['depth_change_m']:+.2f} m,"
                      f" heave {row['heave_m'] * 1000:.1f} mm, pitch swing {row['pitch_swing_deg']:.1f} deg, torque {row['torque_stroke']:.2f} {row['torque_feather']:.2f} (whole run {row['torque_stroke_all']:.2f} {row['torque_feather_all']:.2f}), upz>{row['up_z_min']:.2f} bad {row['bad_qacc']}", flush=True)
            if row["kind"] == "open":
                ov = job[1][2]
                print(f"{row['stroke']} {json.dumps(ov)} u {[round(x, 2) for x in row['u']]}: along {row['along_nose_mps']:+.3f} m/s vz {row['vz_mps']:+.3f}"
                      f" heave {(row['heave_m'] or 0) * 1000:.1f} mm pitch swing {(row['pitch_swing_deg'] or 0):.1f} end pitch {row['pitch_end_deg']:+.0f} roll {row['roll_end_deg']:+.0f}"
                      f" yaw {row['yaw_rate_dps']:+.0f} deg/s upz>{row['up_z_min']:.2f} tq {row['torque_stroke']:.2f} {row['torque_feather']:.2f} bad {row['bad_qacc']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
