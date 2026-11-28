"""F4: free flight of the two-pair flapper (F2's layout B, the neutral hull).
Spec: PREREG_F4.md (frozen).  No learning, no optimiser; the stroke and the law
were set by hand on seeds 100-103, scored on seeds 0-7 (level) and 0-15 (turn).

    cd /home/hundo/Projects/Dytiscidae/dytiscidae
    export FLAPPER_BODY_B='{"hull_radius": 0.0209}' MUJOCO_GL=disable PYTHONPATH=.
    F=$PWD/experiments/flapper
    .venv/bin/python $F/fly.py facts
    .venv/bin/python $F/fly.py rig   --out $F/results/fly.jsonl
    .venv/bin/python $F/fly.py level --arm fly,glide --seeds 0-7 --out $F/results/fly.jsonl
    .venv/bin/python $F/fly.py run   --arm fly,still,frozen --seeds 0-15 --out $F/results/fly.jsonl

The harness is F2's (`landwalk.py`: the body builder, the unlimited pack, the
token controller that forgets its held action when the clock goes back).
Nothing in `landwalk.py`, `swim.py` or `body.py` is changed.

Two loops.  The outer law (`Pilot`, 25 Hz, from `TriphibianEnv.observation`
alone) turns the heading error into a bank, the height lost into a pitch and
the airspeed into how hard to flap.  The inner loop (`Wings.act`, every step)
reads the body's attitude and rates and holds that bank and pitch with feather
offsets: fore against hind for pitch, left against right for roll.
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
from dytiscidae.envs.evaluate import Controller, _scatter_seed, run_segment  # noqa: E402
from dytiscidae.envs.tasks import CRUISE, Phase, TaskSchedule, schedule_for, task_seed  # noqa: E402
from dytiscidae.envs.triphibian import Domain  # noqa: E402

G = 9.80665
DT = LW.DT
SEGMENT_SECONDS = 8.0

# what the pilot hands the wings
U_PITCH, U_BANK, U_BEAT, U_YAW, U_INC = 0, 1, 2, 3, 4
N_U = 5

#: The flight stroke and the inner loop's gains, set by hand on seeds 100-103
#: (trials_F4.jsonl).  Angles in rad.  Positive stroke lowers the wing tip,
#: positive feather raises the leading edge.  Phases in beats: fore-left,
#: fore-right, hind-left, hind-right (the diagonal of F3b, so the wings'
#: inertial reactions cancel in force, pitch and roll).
WINGS = dict(
    f=12.0, A=0.7, Psi=0.5, phases=(0.0, 0.5, 0.5, 0.0),
    dihedral=0.0,                    # rad both wings of a pair are raised
    # The offsets are strong: 0.1 rad of fore-against-hind feather turns the nose
    # at about 49 rad/s^2 and 0.1 rad of left-against-right rolls at about 66
    # (trials_F4 "open loop, wings still"), so these gains close both loops near
    # 15 rad/s.
    k_th=0.6, k_q=0.06, dp_max=0.35,  # pitch: feather offset (fore +, hind -) per rad and per rad/s
    k_ph=0.4, k_p=0.05, dr_max=0.25,  # roll: feather offset (left +, right -)
    # yaw: share of the stroke amplitude, right against left; 0.2 of it turns the
    # nose at about 18 rad/s^2 (trials_F4 "yaw from the left-right stroke difference")
    # Left at 0: fed back at every step it shook the stroke and the flight was
    # lost (trials_F4 "level trials 1, 2"), and a banked body turns its nose
    # into the turn by itself.
    k_r=0.0, k_beta=0.0, dy_max=0.3,
    s_pitch=1.0, s_roll=1.0, s_yaw=1.0,
    servo=dict(stroke=(60.0, 0.10), feather=(30.0, 0.03), tail=(5.0, 0.01)),
)
#: The pilot's gains.
PILOT = dict(
    th0=0.17, inc=0.0,               # pitch held with no error, and the wings' incidence on the body
    k_h=0.05, th_max=0.35, th_min=-0.1,   # rad of pitch per m of height lost
    k_head=1.0, bank_max=0.7,        # rad of bank per rad of heading error
    k_turn_pitch=0.13,               # rad of pitch added per rad of bank asked
    beat0=1.0, k_v=0.0, beat_min=0.5, beat_max=1.0,   # how hard to flap: its share at the launch speed, and per m/s short
)


def _wrap(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


class Pilot:
    """``law(o, t) -> u`` from `TriphibianEnv.observation` alone: the heading
    error asks for a bank (left wing down to turn left), the height lost for
    more pitch, the airspeed for more or less stroke."""

    def __init__(self, env, pilot: dict):
        self.g = dict(pilot)
        self.v_launch = float(env.launch_speed)

    def core(self, o, e_head) -> np.ndarray:
        g = self.g
        u = np.zeros(N_U)
        lost = 3.0 * math.atanh(float(np.clip(o[24], -0.999999, 0.999999)))
        speed = 5.0 * float(np.linalg.norm(o[0:3]))
        bank = float(np.clip(g["k_head"] * e_head, -g["bank_max"], g["bank_max"]))
        u[U_BANK] = -bank                                       # roll is + with the left wing up
        u[U_PITCH] = float(np.clip(g["th0"] + g["k_h"] * lost + g["k_turn_pitch"] * abs(bank),
                                   g["th_min"], g["th_max"]))
        v_ref = self.v_launch * (float(o[21]) if o[19] > 0.5 else 1.0)
        u[U_BEAT] = float(np.clip(g["beat0"] + g["k_v"] * (v_ref - speed), g["beat_min"], g["beat_max"]))
        u[U_INC] = g["inc"]
        return u

    def __call__(self, o, t) -> np.ndarray:
        return self.core(o, math.atan2(o[23], o[22]) if o[19] > 0.5 else 0.0)


class Frozen:
    """The pilot with its command frozen at the first seen (a null arm; the
    state is the arm's): it keeps the first heading it was told."""

    def __init__(self, env, pilot: Pilot):
        self.env, self.pilot, self.head = env, pilot, None

    def __call__(self, o, t):
        _, _, yaw = attitude(self.env)
        if self.head is None and o[19] > 0.5:
            self.head = yaw + math.atan2(o[23], o[22])
        return self.pilot.core(o, 0.0 if self.head is None else _wrap(self.head - yaw))


def body() -> dict:
    b = dict(LW.LAYOUTS["B"])
    b.update(json.loads(os.environ.get("FLAPPER_BODY_B", "{}")))
    return b


def attitude(env):
    d = env.data
    R = d.xmat[env.root_body]
    return (math.asin(float(np.clip(R[6], -1.0, 1.0))),              # nose-up pitch
            math.atan2(float(R[7]), float(R[8])),                    # roll, + left wing up
            math.atan2(float(R[3]), float(R[0])))                    # yaw


class Wings:
    """``act(u) -> joint targets``, every step: the stroke on the clock and the
    attitude loop.  Reads attitude and body rates; no position, no command."""

    def __init__(self, env, plan, w: dict):
        self.env, self.plan, self.w = env, plan, dict(w)

    def act(self, u) -> np.ndarray:
        w, pl, env = self.w, self.plan, self.env
        d = env.data
        out = np.zeros(pl.n_act)
        t = float(d.time)
        th, ph, _ = attitude(env)
        rate = d.qvel[3:6]                                   # body frame: roll, pitch (nose down +), yaw
        R = d.xmat[env.root_body].reshape(3, 3)
        vb = R.T @ d.qvel[:3]
        beta = math.atan2(float(vb[1]), max(float(vb[0]), 1e-3))
        if w.get("open"):                                    # a probe: the commands are the offsets themselves
            dp, dr, dy = float(u[U_PITCH]), float(u[U_BANK]), float(u[U_YAW])
        else:
            dp = float(np.clip(w["k_th"] * (float(u[U_PITCH]) - th) + w["k_q"] * float(rate[1]),
                               -w["dp_max"], w["dp_max"]))
            dr = float(np.clip(w["k_ph"] * (float(u[U_BANK]) - ph) - w["k_p"] * float(rate[0]),
                               -w["dr_max"], w["dr_max"]))
            # yaw: the nose is turned into the wind it feels (beta is the slip, +
            # when the air comes from the left) and its rate is damped
            dy = float(np.clip(float(u[U_YAW]) - w["k_r"] * float(rate[2]) + w["k_beta"] * beta,
                               -w["dy_max"], w["dy_max"]))
        beat = float(u[U_BEAT])
        for i, wg in enumerate(pl.wings):
            fore = 1.0 if i < 2 else -1.0
            a = 2.0 * math.pi * (w["f"] * t + w["phases"][i])
            g = 1.0 - w["s_yaw"] * wg["side"] * dy
            stroke = -w["dihedral"] + beat * g * w["A"] * math.sin(a)
            feather = (float(u[U_INC]) + w["s_pitch"] * fore * dp + w["s_roll"] * wg["side"] * dr
                       - beat * w["Psi"] * math.cos(a))
            out[wg["stroke"]] = wg["s_stroke"] * float(np.clip(stroke, -1.3, 1.3))
            out[wg["feather"]] = wg["s_feather"] * float(np.clip(feather, -0.95, 0.95))
        return out


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
        th, ph, yaw = attitude(env)
        af = np.abs(d.actuator_force[: self.plan.n_act])
        p, v = d.xpos[env.root_body], d.qvel[:3]
        err = np.abs(np.asarray(target, float) - d.qpos[env._act_qadr])
        self.rows.append((float(d.time), float(p[0]), float(p[1]), float(p[2]), th, ph, yaw,
                          float(np.linalg.norm(v)), float(v[2]), float(d.xmat[env.root_body][8]),
                          float(af[self.plan.i_stroke].max()), float(af[self.plan.i_feather].max()),
                          float(err[self.plan.i_stroke].max()), float(err[self.plan.i_feather].max())))
        return alive


def setup(seed, wings=None, pilot=None):
    env = LW.make_env("B", seed)
    plan = LW.Plan(env)
    w = {**WINGS, **(wings or {})}
    # The body's own facts are measured (and cached) with the package's servos,
    # before the ideal ones go on and before anything is recorded: the air score
    # reads `level_margin` on a held rig that steps this same environment.
    env.launch_speed, env.level_margin(), env.thrust_margin(), env.flies_level()
    LW.ideal_servos(env, w["servo"])
    return env, plan, w, {**PILOT, **(pilot or {})}


def trace(rows, every=0.25, until=None) -> list:
    out = []
    k = int(round(every / DT))
    for r in rows[k - 1::k]:
        if until is not None and r[0] > until + 1e-9:
            break
        out.append([round(r[0], 2), round(r[3], 2), round(r[7], 2), round(math.degrees(r[4]), 1),
                    round(math.degrees(r[5]), 1), round(math.degrees(r[6]), 1), round(r[1], 2), round(r[2], 2)])
    return out


def row_probe(u, wings=None, seconds=1.0, seed=100) -> dict:
    """A tuning probe: launched as the air segment launches it (no scatter),
    the wings given fixed commands with the attitude loop open or closed."""
    env, plan, w, _ = setup(seed, wings)
    wg = Wings(env, plan, w)
    env.reset(Domain.AIR, randomise=False)
    d = env.data
    rec = Recorder(env, plan)
    uu = np.zeros(N_U)
    uu[: len(u)] = u
    try:
        for _ in range(int(seconds / DT)):
            env.step(wg.act(uu))
    finally:
        rec.close()
    a = np.array(rec.rows)
    bad = env._mj.mjtWarning.mjWARN_BADQACC
    return {"kind": "probe", "u": list(map(float, uu)), "wings": {k: v for k, v in w.items() if k != "servo"},
            "servo": w["servo"], "seconds": seconds, "trace": trace(rec.rows, 0.1 if seconds <= 2 else 0.5),
            "dz": float(a[-1, 3] - a[0, 3]), "speed_end": float(a[-1, 7]),
            "pitch_end_deg": float(math.degrees(a[-1, 4])), "roll_end_deg": float(math.degrees(a[-1, 5])),
            "yaw_end_deg": float(math.degrees(a[-1, 6])), "up_z_min": float(a[:, 9].min()),
            "torque_stroke": float(a[:, 10].max()), "torque_feather": float(a[:, 11].max()),
            "servo_error_stroke": float(a[:, 12].max()), "servo_error_feather": float(a[:, 13].max()),
            "bad_qacc": int(d.warning[bad].number)}


def controller(env, plan, w, pl, arm):
    """``(evaluator's controller, token controller or None)`` for an arm."""
    if arm in ("still", "glide"):
        return Controller(params=env.held_still_params(), policy=None, bases=None), None
    pilot = Pilot(env, pl)
    ctl = LW.TokenController(env, pilot if arm == "fly" else Frozen(env, pilot), Wings(env, plan, w).act)
    return ctl.controller, ctl


def halves(rows) -> list:
    """The recorder's rows split where the clock went back (the two halves of a pair)."""
    a = np.array(rows)
    cut = [0] + [k for k in range(1, len(a)) if a[k, 0] < a[k - 1, 0]] + [len(a)]
    return [a[i:j] for i, j in zip(cut[:-1], cut[1:])]


def course_deg(h, t0, t1) -> float:
    """Direction of travel over the ground between two times, degrees."""
    seg = h[(h[:, 0] >= t0) & (h[:, 0] <= t1)]
    d = seg[-1, 1:3] - seg[0, 1:3]
    return math.degrees(math.atan2(float(d[1]), float(d[0])))


def flight_reads(h, skip=0.25) -> dict:
    late = h[h[:, 0] > skip]
    end = h[h[:, 0] > h[-1, 0] - 0.5]
    return {"dz_m": float(h[-1, 3] - h[0, 3]), "z_min_rel_m": float(h[:, 3].min() - h[0, 3]),
            "z_max_rel_m": float(h[:, 3].max() - h[0, 3]),
            "speed_start_mps": float(h[0, 7]), "speed_end_mps": float(end[:, 7].mean()),
            "up_z_min": float(h[:, 9].min()), "roll_max_deg": float(np.degrees(np.abs(h[:, 5]).max())),
            "pitch_min_deg": float(np.degrees(h[:, 4].min())), "pitch_max_deg": float(np.degrees(h[:, 4].max())),
            "torque_stroke": float(late[:, 10].max()), "torque_feather": float(late[:, 11].max()),
            "torque_stroke_all": float(h[:, 10].max()), "torque_feather_all": float(h[:, 11].max()),
            "servo_error_stroke": float(late[:, 12].max()), "servo_error_feather": float(late[:, 13].max())}


def row_level(arm, seed, wings=None, pilot=None, seconds=8.0) -> dict:
    """The level bar: launched as the air segment launches it (reset and the
    segment's own scatter draw), told to fly straight for 8 s."""
    env, plan, w, pl = setup(seed, wings, pilot)
    ctrl, ctl = controller(env, plan, w, pl, arm)
    v = float(env.launch_speed)
    straight = TaskSchedule("air", (Phase(CRUISE, 0.0, heading=0.0, speed=v),
                                    Phase(CRUISE, 0.5, heading=0.0, speed=v)))
    rec = Recorder(env, plan)
    row = {"kind": "level", "arm": arm, "seed": seed, "wings": {k: x for k, x in w.items() if k != "servo"},
           "servo": w["servo"], "pilot": pl, "launch_speed_mps": v,
           "launch_pitch_deg": math.degrees(float(env.launch_pitch))}
    try:
        env.reset(Domain.AIR)
        env.scatter(np.random.default_rng(_scatter_seed(seed, Domain.AIR)), pose=ctrl.params)
        env.task = straight
        res = env.rollout(seconds, params=ctrl.params, policy=ctrl.policy, basis=ctrl.basis_for(Domain.AIR),
                          domain=Domain.AIR)
        h = max(halves(rec.rows), key=len)                    # the flight, not a rig the score may step
        row.update(failure=res.failure, bad_qacc=int(res.bad_qacc), survived=bool(res.survived),
                   seconds_flown=float(h[-1, 0]), **flight_reads(h), trace=trace(list(map(tuple, h)), 0.5))
        row["met"] = bool(not res.failure and row["bad_qacc"] == 0 and row["seconds_flown"] > seconds - 0.01
                          and abs(row["dz_m"]) <= 1.5 and abs(row["speed_end_mps"] / v - 1.0) <= 0.3
                          and row["up_z_min"] > 0.5)
    finally:
        rec.close()
        if ctl is not None:
            ctl.close()
    return row


def row_turn(arm, seed, wings=None, pilot=None) -> dict:
    """One scored air segment (the mirrored pair), unmodified."""
    t0 = time.time()
    env, plan, w, pl = setup(seed, wings, pilot)
    ctrl, ctl = controller(env, plan, w, pl, arm)
    rec = Recorder(env, plan)
    row = {"kind": "turn", "arm": arm, "seed": seed, "wings": {k: x for k, x in w.items() if k != "servo"},
           "servo": w["servo"], "pilot": pl}
    try:
        dom = Domain.AIR
        sseed = _scatter_seed(seed, dom)
        task = schedule_for(dom, np.random.default_rng(task_seed(sseed)), trim_speed=float(env.launch_speed))
        seg = run_segment(env, dom, SEGMENT_SECONDS, ctrl, scatter_seed=sseed, task=task)
        mm = seg.measurements
        asked = math.degrees(float(task.phases[1].heading))
        hs = halves(rec.rows)
        made = []
        for k, h in enumerate(hs):
            if h[-1, 0] > 7.9:
                made.append({"asked_deg": asked if k == 0 else -asked,
                             "made_deg": math.degrees(_wrap(math.radians(course_deg(h, 6.0, 8.0) - course_deg(h, 2.0, 4.0)))),
                             **flight_reads(h)})
        row.update(score=float(seg.competence), failure=seg.failure, bad_qacc=int(seg.bad_qacc),
                   survived=bool(seg.survived), parts={k: round(float(x), 6) for k, x in dict(seg.parts).items()},
                   m={k: float(mm[k]) for k in ("turn_response", "pair_turn", "pair_turn_mean", "pair_turn_factor",
                                                "height_hold", "cruise_height_hold", "turn_height_hold", "cmd_turn",
                                                "cmd_speed", "airborne_fraction", "measured_sink_rate", "spin_rate",
                                                "air_gates", "level_margin", "task_score") if k in mm},
                   asked_deg=asked, halves=made)
    finally:
        rec.close()
        if ctl is not None:
            ctl.close()
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def _level(args):
    return row_level(*args)


def _turn(args):
    return row_turn(*args)


def rig_rows() -> list:
    """The held rig for this body (`rig.run`, `rig.trim`): hover lift at 12 Hz,
    trimmed cruise at 6 and 8 m/s and at the launch speed."""
    import rig
    b = body()
    rows = []
    hov = [rig.run(dict(b, stroke_range=1.45), 0.0, 90.0, 12.0, 130.0, sg * 45.0, 0.0) for sg in (1.0, -1.0)]
    h = max(hov, key=lambda r: r["Fz"])
    rows.append({"kind": "rig", "what": "hover", "body": b, **{k: h[k] for k in (
        "f", "Phi_deg", "Psi_deg", "Fz", "Fx", "weight", "lift_over_weight", "aero_power_W", "mass_kg")}})
    return rows


def _trim(args):
    import rig
    V, f, Phi, Psi = args
    r = rig.trim(body(), V, f, Phi, Psi)
    return {"kind": "rig", "what": "cruise", **{k: r[k] for k in (
        "V", "f", "Phi_deg", "Psi_deg", "psi0_deg", "Fz", "Fx", "weight", "lift_over_weight",
        "aero_power_W", "trimmed", "evals")}}


def _probe(args):
    return row_probe(*args)


def _call(job):
    return job[0](job[1])


def _seeds(s):
    lo, hi = (int(x) for x in s.split("-")) if "-" in s else (int(s), int(s))
    return list(range(lo, hi + 1))


def cmd_facts(a) -> int:
    env = LW.make_env("B", 0)
    w = float(env.solver._dry_mass.sum()) * G
    lm = env.level_margin()
    print(json.dumps({"body": body(), "dry_mass_kg": float(env.solver._dry_mass.sum()), "weight_N": w,
                      "launch_speed_mps": float(env.launch_speed),
                      "launch_pitch_deg": math.degrees(float(env.launch_pitch)),
                      "air_gates": list(env.air_gates), "level_margin": None if lm is None else float(lm),
                      "flies_level": bool(env.flies_level()), "lift_margin": float(env.lift_margin)}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("facts")
    r = sub.add_parser("rig")
    r.add_argument("--out", required=True)
    r.add_argument("--jobs", type=int, default=6)
    p = sub.add_parser("probe")
    p.add_argument("--u", default="[[0,0,0,0,0]]")
    p.add_argument("--wings", default="[{}]")
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--out", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--jobs", type=int, default=6)
    for name in ("level", "run"):
        q = sub.add_parser(name)
        q.add_argument("--arm", default="fly")
        q.add_argument("--seeds", default="100-103")
        q.add_argument("--wings", default="{}")
        q.add_argument("--pilot", default="{}")
        q.add_argument("--out", required=True)
        q.add_argument("--note", default="")
        q.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "facts":
        return cmd_facts(a)
    import multiprocessing as mp
    rows = []
    if a.cmd == "rig":
        rows = rig_rows()
        jobs = [(_trim, (V, f, Phi, Psi)) for V in (6.0, 8.0, float(LW.make_env("B", 0).launch_speed))
                for f in (6.0, 8.0, 10.0, 12.0) for Phi in (60.0, 80.0) for Psi in (20.0, 30.0)]
    if a.cmd == "probe":
        jobs = [(_probe, (u, wv, a.seconds)) for wv in json.loads(a.wings) for u in json.loads(a.u)]
    if a.cmd in ("level", "run"):
        fn = _level if a.cmd == "level" else _turn
        jobs = [(fn, (arm, sd, json.loads(a.wings), json.loads(a.pilot))) for arm in a.arm.split(",")
                for sd in _seeds(a.seeds)]
    with mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
        rows += pool.map(_call, jobs, chunksize=1)
    with open(Path(a.out).resolve(), "a") as fh:
        for row in rows:
            if getattr(a, "note", ""):
                row["note"] = a.note
            fh.write(json.dumps(row, default=float) + "\n")
            if row["kind"] == "rig":
                if row["what"] == "hover":
                    print(f"hover 12 Hz stroke 130 feather 45: lift {row['Fz']:.3f} N = {row['lift_over_weight']:.2f} weights ({row['weight']:.3f} N), aero power {row['aero_power_W']:.1f} W", flush=True)
                else:
                    print(f"cruise V {row['V']:.1f} f {row['f']:.0f} Hz stroke {row['Phi_deg']:.0f} feather {row['Psi_deg']:.0f}: incidence {-row['psi0_deg']:.1f} deg, lift/weight {row['lift_over_weight']:.2f}, net force along {row['Fx']:+.3f} N, aero power {row['aero_power_W']:.2f} W, trimmed {row['trimmed']}", flush=True)
            if row["kind"] == "level":
                print(f"level {row['arm']} {row['seed']}: {row.get('failure', '')!r} bad {row.get('bad_qacc')} flown {row.get('seconds_flown', 0):.1f} s, dz {row.get('dz_m', 0):+.2f} m (low {row.get('z_min_rel_m', 0):+.2f} high {row.get('z_max_rel_m', 0):+.2f}),"
                      f" speed {row.get('speed_start_mps', 0):.2f} -> {row.get('speed_end_mps', 0):.2f}, upz>{row.get('up_z_min', 0):.2f}, roll max {row.get('roll_max_deg', 0):.0f}, pitch {row.get('pitch_min_deg', 0):+.0f}..{row.get('pitch_max_deg', 0):+.0f},"
                      f" torque {row.get('torque_stroke', 0):.2f} {row.get('torque_feather', 0):.2f} {'MET' if row.get('met') else 'no'}", flush=True)
            if row["kind"] == "turn":
                m = row.get("m", {})
                hv = " | ".join(f"asked {h['asked_deg']:+.0f} made {h['made_deg']:+.0f} dz {h['dz_m']:+.1f} v {h['speed_end_mps']:.1f} upz>{h['up_z_min']:.2f}" for h in row.get("halves", []))
                print(f"turn {row['arm']} {row['seed']}: score {row.get('score', -1):.3f} {row.get('failure', '')!r} bad {row.get('bad_qacc')} response {m.get('turn_response', float('nan')):.2f} (pair mean {m.get('pair_turn_mean', float('nan')):+.2f}, factor {m.get('pair_turn_factor', float('nan')):.2f})"
                      f" height {m.get('height_hold', float('nan')):.2f} | {hv}", flush=True)
            if row["kind"] == "probe":
                print(f"u {[round(x, 2) for x in row['u']]}: dz {row['dz']:+.2f} m, speed {row['speed_end']:.2f}, pitch {row['pitch_end_deg']:+.0f} roll {row['roll_end_deg']:+.0f} yaw {row['yaw_end_deg']:+.0f}, upz>{row['up_z_min']:.2f}, torque {row['torque_stroke']:.2f} {row['torque_feather']:.2f}, servo error {row['servo_error_stroke']:.2f} {row['servo_error_feather']:.2f}, bad {row['bad_qacc']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
