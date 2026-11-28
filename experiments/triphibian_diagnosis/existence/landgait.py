#!/usr/bin/env python3
"""E0b: land with the weight on the ground.  Spec: PREREG_land.md (frozen).

`triquad` is imported, never edited.  Arm L is B1 with its four fixed legs
replaced by four jointed limbs (existing part kinds: a STRUT on a universal
joint carrying a fixed FOOT), driven through the CPG's own channels by a hand
gait; `Hand` flies, swims and crosses the body as in E0, with the rotors
stopped on land.

    cd /home/hundo/Projects/Dytiscidae/dytiscidae; export MUJOCO_GL=disable PYTHONPATH=.
    E=experiments/triphibian_diagnosis/existence
    .venv/bin/python $E/landgait.py facts --body L1
    .venv/bin/python $E/landgait.py probes
    .venv/bin/python $E/landgait.py run --body L1 --arm walk --items land --seeds 0-7 --out results_land.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mujoco  # noqa: E402
import triquad as tq  # noqa: E402
import nulls as nl  # noqa: E402  (TaskView, task_of; imported, not edited)
import dytiscidae.envs.transitions as T  # noqa: E402
from dytiscidae.core.genome import FOOT, PADDLE, STRUT, Edge, Part  # noqa: E402
from dytiscidae.core.phenotype import build  # noqa: E402
from dytiscidae.envs.evaluate import Controller  # noqa: E402
from dytiscidae.envs.tasks import CRUISE  # noqa: E402
from dytiscidae.envs.triphibian import DOMAIN_CYCLE, Domain  # noqa: E402
from dytiscidae.physics.medium import GRAVITY  # noqa: E402

SEG = ("air", "water", "land")
MISS = ("mission_air", "mission_water", "mission_land")

# --------------------------------------------------------------------------
# Bodies
# --------------------------------------------------------------------------

#: Every edit that turns B1 into the legged body L1.  Listed in RESULT_land.md.
L1_EDITS = dict(
    hull_radius=0.064,           # re-set by hand for the density ratio (see `facts`)
    hip_u=(0.42, 0.58),          # hull stations of the two limb pairs (clear of the rotor discs)
    upper_length=0.09, upper_radius=0.006,   # STRUT, universal joint, out to the side
    lower_length=0.075, lower_radius=0.006,  # FOOT, fixed, straight down from the STRUT's tip
    stroke_range=(-0.7, 0.7),    # rad, raise/lower the foot (the feather hinge is +-1.0 in mjcf)
    motor_class="bldc", motor_mass=0.03, gear_ratio=6.0,
)


def limbs(g, e, lower_part):
    """Replace B1's part 2 (the fixed leg) by ``upper`` + ``lower_part``."""
    hull, arm, _leg, fin = g.parts
    upper = Part(kind=str(e.get("upper_kind", STRUT)), length=float(e["upper_length"]),
                 span=float(e["upper_length"]), root_chord=float(e.get("upper_chord", 0.15)),
                 radius=float(e["upper_radius"]), material="cfrp", joint="universal", joint_axis=np.array([0.0, 0.0, 1.0]),
                 joint_range=tuple(e["stroke_range"]), actuated=True,
                 motor_class=str(e["motor_class"]), motor_mass=float(e["motor_mass"]),
                 gear_ratio=float(e["gear_ratio"]), sealed=False,
                 stroke_amplitude=0.0, neutral=0.5)
    g.parts = [hull, arm, upper, fin, lower_part]
    keep = [ed for ed in g.edges if ed.child != 2]          # arms (child 1) and fin (child 3)
    g.edges = keep + [
        # roll pi/2 puts the limb's local +Y straight down on both sides, so the
        # fixed lower part (in the local XY plane) mirrors properly
        Edge(parent=0, child=2, pos_u=float(u), azimuth=0.0, elevation=0.0, roll=math.pi / 2,
             scale=1.0, reflect=True) for u in e["hip_u"]
    ] + [Edge(parent=2, child=4, pos_u=1.0, azimuth=0.0, elevation=0.0, roll=0.0, scale=1.0,
              reflect=False)]
    return g


def l1_genome(**over):
    e = dict(L1_EDITS)
    e.update(over)
    g = tq.b1_genome(hull_radius=float(e["hull_radius"]))
    lower = Part(kind=FOOT, length=float(e["lower_length"]), radius=float(e["lower_radius"]),
                 material="cfrp", joint="none", actuated=False)
    g = limbs(g, e, lower)
    g.lineage = ["quad", "E0-B1", "E0b-L1"]
    _rethrottle(g)
    return g


def _rethrottle(g, margin=1.05):
    """B1's rule again for the new mass: base-gait rotor throttle = 1.05 x hover."""
    from dytiscidae.core.mjcf import ROTOR_TIP_SPEED
    from dytiscidae.physics.medium import AIR
    from dytiscidae.physics.rotor import bemt
    p = build(g)
    spec = next(s.rotor for s in p.segments if s.rotor is not None)
    Tq, _ = bemt(spec, 600.0, 0.0, 0.0, AIR.rho, AIR.mu)
    om_h = math.sqrt(margin * p.mass * GRAVITY / (4.0 * Tq / 600.0 ** 2))
    g.parts[1].rotor_throttle = float(np.clip(om_h / (ROTOR_TIP_SPEED / spec.radius), 0.05, 0.95))


#: Arm W (exploratory): the same limbs made of flapping surfaces.  The jointed
#: part is a PADDLE on the universal joint and the part it carries to the ground
#: is a fixed PADDLE; everything else is L1.
W1_EDITS = dict(L1_EDITS, upper_kind=PADDLE, upper_chord=0.05, lower_chord=0.04)


def w1_genome(**over):
    e = dict(W1_EDITS)
    e.update(over)
    g = tq.b1_genome(hull_radius=float(e["hull_radius"]))
    lower = Part(kind=PADDLE, span=float(e["lower_length"]), root_chord=float(e["lower_chord"]),
                 length=float(e["lower_length"]), radius=float(e["lower_radius"]), material="cfrp",
                 joint="none", actuated=False)
    g = limbs(g, e, lower)
    g.lineage = ["quad", "E0-B1", "E0b-W1"]
    _rethrottle(g)
    return g


tq.BODIES["L1"] = l1_genome
tq.BODIES["W1"] = w1_genome


def triquad_sha() -> str:
    return hashlib.sha256((HERE / "triquad.py").read_bytes()).hexdigest()


def cmd_facts(args) -> int:
    p = build(tq.BODIES[args.body]())
    from dytiscidae.envs.triphibian import TriphibianEnv
    env = TriphibianEnv(p, seed=0)
    m, d = env.model, env.data
    out = {"body": args.body, "summary": p.summary(), "mass": p.mass,
           "budget": {k: round(v, 4) for k, v in p.budget.as_dict().items()},
           "density_ratio": p.density_ratio, "tier0": tq.tier0_gate(p),
           "act_names": env.act_names, "level_margin": env.level_margin(),
           "flies_level": env.flies_level(), "air_gates": env.air_gates,
           "lo": [round(float(x), 3) for x in env.cpg.lo], "hi": [round(float(x), 3) for x in env.cpg.hi],
           "stall_torque": [round(float(a.stall_torque), 3) for a in p.actuators],
           "kp": [round(float(x), 3) for x in m.actuator_gainprm[:, 0]]}
    env.reset(Domain.LAND, randomise=False)
    out["land_rest_root_z_minus_ground"] = float(d.qpos[2] - env.ground_height(float(d.qpos[0])))
    print(json.dumps(out, indent=1, default=float))
    # kinematics: where each foot goes for +0.3 rad on each leg channel
    feet = [b for b in range(1, m.nbody) if m.body_parentid[b] > 1 and "rot" not in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)]
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 1.0
    mujoco.mj_forward(m, d)

    def tips():
        res = []
        for b in feet:
            g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == b][0]
            R = d.geom_xmat[g].reshape(3, 3)
            half = m.geom_size[g][1]
            ends = [d.geom_xpos[g] + s * half * R[:, 2] for s in (-1, 1)]
            res.append(min(ends, key=lambda q: q[2]) - d.xpos[env.root_body])
        return np.array(res)

    base = tips()
    print("feet (root frame, x y z):", np.round(base, 3).tolist())
    for k, name in enumerate(env.act_names):
        if name.endswith("_r"):
            continue
        d.qpos[env._act_qadr[k]] = 0.3
        mujoco.mj_forward(m, d)
        print(f"{k} {name:18s} +0.3 rad moves feet by", np.round(tips() - base, 3).tolist())
        d.qpos[env._act_qadr[k]] = 0.0
    return 0


# --------------------------------------------------------------------------
# Support share
# --------------------------------------------------------------------------


def ground_fz(env) -> float:
    """Vertical component of the ground's contact force on the machine, N."""
    m, d = env.model, env.data
    f6 = np.zeros(6)
    tot = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 == 0) == (b2 == 0):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        fw = np.asarray(c.frame, float).reshape(3, 3).T @ f6[:3]     # on geom2, world frame
        tot += float(fw[2]) if b1 == 0 else -float(fw[2])
    return tot


class Recorder:
    """Wraps `env.step` on the instance; reads, changes nothing."""

    def __init__(self, env, hand=None):
        self.env, self.hand = env, hand
        self.rows = []          # (half, t, fz, kind, late, rotor_max, x, y, mode)
        self.half = -1
        self._last_t = 1e18
        self._step = env.step
        # dry mass: after a water rollout `model.body_mass` still carries the fluid
        # solver's added mass (a resting body read s = 0.54 until this was fixed)
        self.weight = float(np.asarray(env.solver._dry_mass)[1:].sum()) * GRAVITY
        env.step = self.step

    def close(self):
        if "step" in self.env.__dict__:
            del self.env.__dict__["step"]

    def step(self, target):
        env = self.env
        alive = self._step(target)
        t = float(env.data.time)
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
        rmax = float(np.max(np.abs(env.data.qvel[env.rotors.dof_arr]))) if env.rotors.n else 0.0
        p = env.data.xpos[env.root_body]
        self.rows.append((self.half, t, ground_fz(env), kind, late, rmax, float(p[0]), float(p[1]),
                          "" if self.hand is None else self.hand.mode))
        return alive

    def share(self, pred) -> tuple:
        """``(s, n)`` per half over the steps ``pred(row)`` selects; s is None for n == 0."""
        out = []
        for h in range(self.half + 1):
            sel = [r[2] for r in self.rows if r[0] == h and pred(r)]
            out.append((None if not sel else float(np.mean(sel)) / self.weight, len(sel)))
        return out

    def summary(self) -> dict:
        walk = lambda r: r[3] == "land:cruise" and r[4]              # noqa: E731
        stop = lambda r: r[3] == "land:stop" and r[4]                # noqa: E731
        sw, ss = self.share(walk), self.share(stop)
        rot = [max([r[5] for r in self.rows if r[0] == h and walk(r)], default=None)
               for h in range(self.half + 1)]
        vals = [x[0] for x in sw if x[0] is not None]
        return {"s": (min(vals) if vals else None), "s_halves": [x[0] for x in sw],
                "s_steps": [x[1] for x in sw], "s_stop_halves": [x[0] for x in ss],
                "rotor_max_walk": rot,
                "fz_mean_all_over_mg": float(np.mean([r[2] for r in self.rows])) / self.weight
                if self.rows else None,
                "fz_abs_max_N": float(np.max(np.abs([r[2] for r in self.rows]))) if self.rows else None}


# --------------------------------------------------------------------------
# The law for the limbs
# --------------------------------------------------------------------------


class _RotorView:
    """The environment as `Hand.__init__` should see it: rotors only."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_env"), name)

    @property
    def act_names(self):
        return [a for a in object.__getattribute__(self, "_env").act_names if a.endswith("_r")]


class LegHand(tq.Hand):
    """E0's law, unchanged, except that on land the rotors are off: the limbs walk."""

    def _seg_land(self, S, ph, new, yaw0):
        self.mode = "seg-land-legs"
        return self._off()


GAIT = dict(
    f=1.5,            # Hz
    sweep=0.6,        # rad, feather amplitude (foot fore-aft)
    down=0.25,        # rad, stroke in stance (foot pressed down, body carried)
    up=-0.35,         # rad, stroke in swing (foot lifted, body on its belly)
    sharp=1.5,        # stance/swing switch sharpness
    k_turn=2.0,       # turn share per rad of heading error
    e_fwd=0.6,        # rad: forward drive fades to 0 at this heading error
    stand=0.2,        # rad, stroke of the pose held off land (feet 1.5 cm lower than neutral)
    rest=-0.5,        # rad, stroke when told to stop: limbs lifted, body on its belly
    mission_heading=0.0,
)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class LegLaw:
    def __init__(self, env, hand, gait=None, freeze=False):
        self.env, self.hand = env, hand
        self.g = dict(GAIT)
        if gait:
            self.g.update(gait)
        names = env.act_names
        self.irot = [k for k, a in enumerate(names) if a.endswith("_r")]
        self.ia = [k for k, a in enumerate(names) if a.endswith("_a")]
        self.i_f = [k for k, a in enumerate(names) if a.endswith("_f")]
        assert len(self.ia) == len(self.i_f) and len(self.irot) + 2 * len(self.ia) == len(names)
        m = env.model
        self.side = []                      # +1 left (y > 0), -1 right
        for k in self.ia:
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, names[k])
            b = m.jnt_bodyid[m.actuator_trnid[aid, 0]]
            mujoco.mj_forward(m, env.data)
            Rr = env.data.xmat[env.root_body].reshape(3, 3)
            y = float((Rr.T @ (env.data.xipos[b] - env.data.xpos[env.root_body]))[1])
            self.side.append(1.0 if y > 0 else -1.0)
        self.freeze, self.frozen = freeze, None
        self.t0 = None
        self.walking = False

    def legs(self, t):
        g, env = self.g, self.env
        n = len(self.ia)
        view = self.hand.env                      # the real environment, or a shuffled view of it
        ph = view._phase_now()
        heading = None
        if ph is not None and view._active_task.domain == "land":
            if ph.kind == CRUISE:
                heading = float(ph.heading)
                if self.freeze:
                    if self.frozen is None:
                        self.frozen = heading
                    heading = self.frozen
            else:
                self.walking = False
                return np.full(n, g["rest"]), np.zeros(n)
        elif ph is None and self.hand.mode == "landed":
            heading = float(g["mission_heading"])
        if heading is None:
            self.walking = False
            return np.full(n, g["stand"]), np.zeros(n)
        if not self.walking:
            self.walking, self.t0 = True, float(env.data.time)
        R = env.data.xmat[env.root_body].reshape(3, 3)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
        e = _wrap(heading - yaw)
        direction = 1.0
        if abs(e) > 0.5 * math.pi:
            e, direction = _wrap(e - math.pi), -1.0
        w = float(np.clip(g["k_turn"] * e, -1.0, 1.0))
        v = direction * float(np.clip(1.0 - abs(e) / g["e_fwd"], 0.0, 1.0))
        phi = 2.0 * math.pi * g["f"] * (float(env.data.time) - self.t0)
        st = math.tanh(g["sharp"] * math.sin(phi))                     # +1 stance, -1 swing
        stroke = 0.5 * (g["down"] + g["up"]) + 0.5 * (g["down"] - g["up"]) * st
        a = np.full(n, stroke)
        gains = np.array([np.clip(v - s * w, -1.0, 1.0) for s in self.side])   # left v - w, right v + w
        fthr = -gains * g["sweep"] * math.cos(phi)
        return a, fthr


def controller_for(env, arm: str, gait=None, gains=None, shuffle_task=None):
    """``(ctrl, hand, law)``.  Arms: walk, frozen, shuffled (legged bodies);
    hand (triquad's E0 law, bodies without limbs); still."""
    if arm == "still":
        return Controller(params=env.held_still_params(), policy=None, bases=None), None, None
    if arm == "hand":
        ctrl, hand = tq.controller_for(env, "hand", False, gains)
        return ctrl, hand, None
    hand = LegHand(_RotorView(env), record=False, gains=gains)
    hand.env = env if shuffle_task is None else nl.TaskView(env, shuffle_task)
    law = LegLaw(env, hand, gait, freeze=(arm == "frozen"))
    inner = hand.command

    def command(params, t):
        if params is not hand.token:
            return hand._orig(params, t)
        out = np.zeros(len(env.act_names))
        out[law.irot] = inner(params, t)
        a, fthr = law.legs(t)
        out[law.ia], out[law.i_f] = a, fthr
        return out

    env.cpg.command = command
    return hand.ctrl, hand, law


def run_item(env, body, arm, item, seed, gait=None, gains=None, entry=None) -> dict:
    t0 = time.time()
    shuffle = None
    if arm == "shuffled":
        shuffle = nl.task_of(env, Domain(item), (int(seed) + 1) % 8)
    ctrl, hand, law = controller_for(env, arm, gait, gains, shuffle)
    rec = Recorder(env, hand)
    try:
        if item in SEG:
            row = tq.run_seg_item(env, ctrl, Domain(item), seed)
        elif item in tq.CROSSINGS:
            if entry is not None:
                nl.set_entry_state(entry)
            row = tq.run_cross_item(env, ctrl, item)
        else:
            row = tq.run_mission_item(env, ctrl, Domain(item.split("_", 1)[1]))
    finally:
        T.transition_scatter_seed = nl._TSS
        rec.close()
        if hand is not None:
            env.cpg.command = hand._orig
    row = {"body": body, "arm": arm, "item": item, "seed": int(seed), **row}
    sm = rec.summary()
    if item == "land":
        row.update(sm)
    else:
        row["fz_mean_all_over_mg"], row["s"] = sm["fz_mean_all_over_mg"], sm["s"]
    if item in MISS:
        landed = [r for r in rec.rows if r[8] == "landed"]
        row["landed_steps"] = len(landed)
        if landed:
            row["s_landed"] = float(np.mean([r[2] for r in landed])) / rec.weight
            row["rotor_max_landed_late"] = float(max(r[5] for r in landed[len(landed) // 4:]))
            row["walked_m"] = float(math.hypot(landed[-1][6] - landed[0][6], landed[-1][7] - landed[0][7]))
    if gait:
        row["gait"] = gait
    if gains:
        row["gains"] = gains
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def cmd_run(args) -> int:
    items = args.items.split(",")
    gait = json.loads(args.gait) if args.gait else None
    gains = json.loads(args.gains) if args.gains else None
    out = HERE / args.out
    print(f"triquad.py sha256 {triquad_sha()}", flush=True)
    for seed in tq.parse_seeds(args.seeds):
        p, env, facts = tq.make_env(args.body, seed)
        for item in items:
            try:
                row = run_item(env, args.body, args.arm, item, seed, gait, gains)
            except Exception as exc:
                row = {"body": args.body, "arm": args.arm, "item": item, "seed": seed, "score": None,
                       "failure": f"exception: {type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()[-1500:]}
            if args.note:
                row["note"] = args.note
            with open(out, "a") as f:
                f.write(json.dumps(row, default=float) + "\n")
            b = {k: row.get(k) for k in ("body", "arm", "item", "seed", "score", "failure", "bad_qacc",
                                         "s", "s_halves", "rotor_max_walk", "crossed", "hold",
                                         "transitions", "s_landed", "walked_m", "wall_s") if k in row}
            for k in ("score", "s", "s_landed", "walked_m"):
                if isinstance(b.get(k), float):
                    b[k] = round(b[k], 4)
            if item == "land" and row.get("m"):
                b["m"] = {k: round(v, 3) for k, v in row["m"].items() if k in (
                    "pair_along_mean", "pair_across", "walk_progress", "stop_score", "contact_fraction",
                    "upright", "task_walk_first", "cmd_heading")}
            print(json.dumps(b, default=float), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facts")
    f.add_argument("--body", default="L1")
    r = sub.add_parser("run")
    r.add_argument("--body", default="L1")
    r.add_argument("--arm", default="walk")
    r.add_argument("--items", default="land")
    r.add_argument("--seeds", default="100-103")
    r.add_argument("--out", default="trials_land.jsonl")
    r.add_argument("--gait", default="")
    r.add_argument("--gains", default="")
    r.add_argument("--note", default="")
    args = ap.parse_args()
    return {"facts": cmd_facts, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
