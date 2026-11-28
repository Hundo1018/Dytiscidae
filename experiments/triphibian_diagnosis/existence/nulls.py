#!/usr/bin/env python3
"""E0n: command-ignoring arms on the E0 rig.  Spec: PREREG_nulls_mutants.md (frozen).

Arms, all wrappers around `triquad` (which is imported, never edited; `Hand`
and its gains are untouched):

* ``base``     the genome's own gait, no feedback: ``Controller(params=env.cpg.base)``,
               what `evaluate_tier1` builds when it is handed no controller.
* ``blind``    the hand law, with the commanded medium it sees frozen at the
               first one it is shown (the medium the rollout starts in).
* ``shuffled`` the hand law, reading in *both* halves of a paired segment the
               first-half task of seed ``(s + shift) % 8`` for the same medium
               (shift 1 is the arm; shift 0 is identity test 2).
* ``hand``, ``still``  triquad's own, for the identity tests.

Scores are the repo's, called through triquad's own row functions
(`run_seg_item`, `run_cross_item`, `run_mission_item`).  Crossing rows use the
eight entry states of `probes/x1_entry_states.py` (seed 0 is the repo's draw).

    cd /home/hundo/Projects/Dytiscidae/dytiscidae; export MUJOCO_GL=disable PYTHONPATH=.
    E=experiments/triphibian_diagnosis/existence
    .venv/bin/python $E/nulls.py ident                       # identity tests 1-4
    .venv/bin/python $E/nulls.py table                       # every row -> results_nulls.jsonl
    .venv/bin/python $E/nulls.py row --body B1 --arm base --item land_to_air --seed 3
    .venv/bin/python $E/nulls.py summary
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics as st
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import triquad as tq  # noqa: E402
import dytiscidae.envs.transitions as T  # noqa: E402
from dytiscidae.envs.evaluate import Controller, _scatter_seed  # noqa: E402
from dytiscidae.envs.tasks import TaskSchedule, schedule_for, task_seed  # noqa: E402
from dytiscidae.envs.triphibian import DOMAIN_CYCLE, Domain  # noqa: E402

SEG = ("air", "water", "land")
MISS = ("mission_air", "mission_water", "mission_land")
OUT = HERE / "results_nulls.jsonl"
_TSS = T.transition_scatter_seed


def triquad_sha() -> str:
    return hashlib.sha256((HERE / "triquad.py").read_bytes()).hexdigest()


def set_entry_state(seed: int) -> None:
    """Probe X1's entry states: the repo's draw plus 7919 x seed (seed 0 = the repo's)."""
    T.transition_scatter_seed = (lambda kind, _s=int(seed): (_TSS(kind) + 7919 * _s) & 0x7FFFFFFF)


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


class BlindPolicy(tq.HandPolicy):
    """`HandPolicy`, except that the medium handed to the law is the first one
    the evaluator shows after the rollout begins, for the rest of the rollout."""

    def __init__(self, env):
        self.env = env
        self.frozen = None
        self._commanded = None
        self.env_cmd = []        # (t, medium) each time the environment's command changes
        self.law_cmd = []        # distinct media handed to the law

    @property
    def commanded(self):
        return self._commanded

    @commanded.setter
    def commanded(self, value):
        # `Hand._reset` writes None here at the first step of every rollout.
        self._commanded = value
        if value is None:
            self.frozen = None

    def act(self, obs):
        c = np.asarray(obs[11:14], float)
        seen = DOMAIN_CYCLE[int(np.argmax(c))] if c.max() > 0.5 else None
        name = None if seen is None else seen.value
        if not self.env_cmd or self.env_cmd[-1][1] != name:
            self.env_cmd.append((round(float(self.env.data.time), 3), name))
        if self.frozen is None:
            self.frozen = seen
        self._commanded = self.frozen
        if self.frozen is not None and self.frozen.value not in self.law_cmd:
            self.law_cmd.append(self.frozen.value)
        return np.zeros(1)


class TaskView:
    """The environment as the hand law reads it, with one thing replaced: the
    task.  `_phase_now` and `_active_task` answer from ``fake`` on the real
    segment's own clock; every other attribute is the real environment's."""

    def __init__(self, env, fake: TaskSchedule):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_fake", fake)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_env"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_env"), name, value)

    @property
    def _active_task(self):
        env = object.__getattribute__(self, "_env")
        return None if env._active_task is None else object.__getattribute__(self, "_fake")

    def _phase_now(self):
        env = object.__getattribute__(self, "_env")
        if env._active_task is None:
            return None
        fake = object.__getattribute__(self, "_fake")
        return fake.at((float(env.data.time) - env._seg_t0) / env._seg_T)


def task_of(env, dom: Domain, seed: int) -> TaskSchedule:
    """The first-half task `evaluate_tier1` draws for (seed, medium), with the
    air speed filled in as `TriphibianEnv._arm_task` fills it."""
    sseed = _scatter_seed(seed, dom)
    t = schedule_for(dom, np.random.default_rng(task_seed(sseed)))
    if dom is Domain.AIR:
        v = float(env.launch_speed)
        t = TaskSchedule(t.domain, tuple(dataclasses.replace(ph, speed=v) for ph in t.phases))
    return t


def make_arm(env, arm: str, item: str, seed: int, shift: int = 1):
    """``(controller, hand or None, info, undo)`` for one item."""
    info: dict = {}
    if arm in ("hand", "still"):
        ctrl, hand = tq.controller_for(env, arm, False)
    elif arm == "base":
        ctrl, hand = Controller(params=env.cpg.base, policy=None, bases=None), None
        orig = env.cpg.command
        rec = {"lo": np.inf, "hi": -np.inf, "n": 0}

        def recording(params, t):
            out = orig(params, t)
            rec["lo"], rec["hi"], rec["n"] = min(rec["lo"], float(np.min(out))), max(
                rec["hi"], float(np.max(out))), rec["n"] + 1
            return out

        env.cpg.command = recording
        info["_rec"], info["_orig"] = rec, orig
        info["rest_cmd"] = [float(x) for x in np.asarray(env.cpg.base.offset, float)]
    elif arm == "blind":
        ctrl, hand = tq.controller_for(env, "hand", False)
        hand.policy = BlindPolicy(env)
        hand.ctrl = Controller(params=hand.token, policy=hand.policy,
                               bases={"air": hand.basis, "water": hand.basis})
        ctrl = hand.ctrl
    elif arm == "shuffled":
        assert item in SEG, "shuffled is a media-segment arm"
        ctrl, hand = tq.controller_for(env, "hand", False)
        other = (int(seed) + int(shift)) % 8
        fake = task_of(env, Domain(item), other)
        hand.env = TaskView(env, fake)
        info["fed_task_of_seed"] = other
        info["fed_task"] = [(ph.kind, round(ph.heading, 3), round(ph.speed, 3), ph.depth)
                            for ph in fake.phases]
    else:
        raise SystemExit(f"unknown arm {arm}")
    if hand is not None:
        act, modes = hand._act, []

        def traced():
            out = act()
            if not modes or modes[-1][1] != hand.mode:
                modes.append((round(float(env.data.time), 3), hand.mode))
            return out

        hand._act = traced
        info["_modes"] = modes
    return ctrl, hand, info


def run_row(env, body: str, arm: str, item: str, seed: int, shift: int = 1, keep: dict | None = None) -> dict:
    """One rollout through triquad's own row function, for any arm."""
    t0 = time.time()
    ctrl, hand, info = make_arm(env, arm, item, seed, shift)
    halves: list = []
    roll = env.rollout

    def capture(*a, **k):
        r = roll(*a, **k)
        halves.append(r)
        return r

    try:
        if item in SEG:
            env.rollout = capture                      # both halves of the pair, as scored
            row = tq.run_seg_item(env, ctrl, Domain(item), seed)
        elif item in tq.CROSSINGS:
            set_entry_state(seed)
            row = tq.run_cross_item(env, ctrl, item)
        elif item in MISS:
            row = tq.run_mission_item(env, ctrl, Domain(item.split("_", 1)[1]))
        else:
            raise SystemExit(f"unknown item {item}")
    finally:
        T.transition_scatter_seed = _TSS
        if "rollout" in env.__dict__:
            del env.__dict__["rollout"]
        if hand is not None:
            env.cpg.command = hand._orig
        elif "_orig" in info:
            env.cpg.command = info["_orig"]
    row = {"body": body, "arm": arm, "item": item, "seed": int(seed), **row}
    if arm == "shuffled":
        row["shift"] = int(shift)
    if halves:
        row["halves"] = [{"score": float(h.competence), "failure": h.failure,
                          "bad_qacc": int(h.bad_qacc)} for h in halves]
    if "_rec" in info:
        rec = info["_rec"]
        row["cmd_min"], row["cmd_max"], row["cmd_calls"] = rec["lo"], rec["hi"], rec["n"]
        row["rest_cmd"] = info["rest_cmd"]
    if "_modes" in info:
        row["modes"] = info["_modes"][:40]
    if arm == "blind":
        row["env_cmd"] = hand.policy.env_cmd
        row["law_cmd"] = hand.policy.law_cmd
    for k in ("fed_task_of_seed", "fed_task"):
        if k in info:
            row[k] = info[k]
    row["wall_s"] = round(time.time() - t0, 1)
    if keep is not None:
        keep["halves"] = halves
    return row


def append(row: dict, path: Path = OUT) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def brief(row: dict) -> str:
    b = {k: row[k] for k in ("body", "arm", "item", "seed", "score", "failure", "bad_qacc", "wall_s")
         if k in row}
    if isinstance(b.get("score"), float):
        b["score"] = round(b["score"], 4)
    for k in ("crossed", "hold", "transitions"):
        if k in row:
            b[k] = row[k]
    return json.dumps(b)


# --------------------------------------------------------------------------
# Identity tests (spec section 4)
# --------------------------------------------------------------------------


def ident() -> int:
    ok_all = True
    sha = triquad_sha()
    print(f"triquad.py sha256 {sha}")

    # 1. the hand arm, B1, seed 5, air, from this script's code path
    stored = None
    for line in open(HERE / "results_final.jsonl"):
        r = json.loads(line)
        if (r["body"], r["arm"], r["item"], r["seed"]) == ("B1", "hand", "air", 5):
            stored = r
    p, env, facts = tq.make_env("B1", 5)
    row = run_row(env, "B1", "hand", "air", 5)
    keys = ("score", "failure", "bad_qacc", "survived", "parts", "task", "m", "soc_end",
            "booked_wh", "mean_power_w", "max_overload", "peak_draw_w")
    diff = [k for k in keys if json.loads(json.dumps(row[k], default=float)) != stored[k]]
    n_m = len(stored["m"])
    ok = not diff
    ok_all &= ok
    print(f"IDENT 1 {'PASS' if ok else 'FAIL'}: hand B1 seed 5 air from nulls.run_row: score "
          f"{row['score']!r} vs results_final.jsonl {stored['score']!r}; {len(keys)} fields "
          f"({n_m} measurements) compared, {len(diff)} differ {diff}")

    # 2. shuffled with shift 0 against the hand arm, first halves, seed 0, each medium
    first = {}
    for arm in ("hand", "shuffled"):
        p, env, facts = tq.make_env("B1", 0)
        for item in SEG:
            keep: dict = {}
            row = run_row(env, "B1", arm, item, 0, shift=0, keep=keep)
            h = keep["halves"][0]
            first[(arm, item)] = {"competence": float(h.competence), "failure": h.failure,
                                  "bad_qacc": int(h.bad_qacc), "distance": float(h.distance),
                                  "mean_power": float(h.mean_power), "parts": dict(h.parts),
                                  "m": {k: (float(v) if isinstance(v, (int, float, np.floating))
                                            else v) for k, v in h.measurements.items()},
                                  "pair_score": float(row["score"])}
    parts = []
    ok = True
    for item in SEG:
        a, b = first[("hand", item)], first[("shuffled", item)]
        same = all(a[k] == b[k] for k in ("competence", "failure", "bad_qacc", "distance",
                                          "mean_power", "parts", "m"))
        ok &= same
        parts.append(f"{item}: first-half competence {a['competence']!r} / {b['competence']!r}, "
                     f"{len(a['m'])} measurements {'identical' if same else 'DIFFER'} "
                     f"(pair score hand {a['pair_score']:.4f}, shift-0 {b['pair_score']:.4f})")
    ok_all &= ok
    print(f"IDENT 2 {'PASS' if ok else 'FAIL'}: shuffled shift 0 vs hand, B1 seed 0, first halves: "
          + "; ".join(parts))

    # 3. blind: mode constant while the environment's command changes
    p, env, facts = tq.make_env("B1", 0)
    row = run_row(env, "B1", "blind", "water_to_air", 0)
    change = [t for t, m in row["env_cmd"][1:]]
    after = [m for t, m in row["modes"] if m != "no-command"]
    # the command reaches a controller at 25 Hz, so the change at step 375 (1.5 s)
    # is seen at the next control decision, step 380 (1.52 s)
    ok = (len(row["law_cmd"]) == 1 and len(set(after)) == 1 and len(change) == 1
          and 1.5 <= change[0] <= 1.5 + 0.04 + 1e-9)
    ok_all &= ok
    print(f"IDENT 3 {'PASS' if ok else 'FAIL'}: blind B1 water_to_air: environment commanded "
          f"{row['env_cmd']}; the law was handed {row['law_cmd']}; mode trace {row['modes']} "
          f"(no-command is the first 0.04 s, as in the hand arm); crossed {row['crossed']}")

    # 4. base: commanded rotor speed equals the rest throttle throughout
    p, env, facts = tq.make_env("B1", 0)
    row = run_row(env, "B1", "base", "air", 0)
    rest = row["rest_cmd"]
    thr = [float(q.rotor_throttle) for q in p.genome.parts if q.rotor_radius > 0]
    ok = row["cmd_min"] == row["cmd_max"] == rest[0] and len(set(rest)) == 1
    ok_all &= ok
    print(f"IDENT 4 {'PASS' if ok else 'FAIL'}: base B1 air seed 0: commanded rotor speed min "
          f"{row['cmd_min']!r} max {row['cmd_max']!r} rad/s over {row['cmd_calls']} calls; rest "
          f"throttle {thr[0]!r} x top speed {float(env.cpg.hi[0])!r} = {rest[0]!r}")
    print(f"triquad.py sha256 {triquad_sha()} ({'unchanged' if triquad_sha() == sha else 'CHANGED'})")
    return 0 if ok_all else 1


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------


def plan(bodies=("B1", "B0"), seeds=tq.SEEDS):
    """``(body, seed, arm, [items])`` groups; one fresh environment per group,
    media first and in DOMAIN_CYCLE order, as triquad's table ran them."""
    for body in bodies:
        for seed in seeds:
            yield body, seed, "base", list(SEG) + list(tq.CROSSINGS) + (list(MISS) if seed == 0 else [])
            yield body, seed, "shuffled", list(SEG)
            yield body, seed, "blind", list(tq.CROSSINGS) + (list(MISS) if seed == 0 else [])


def table(args) -> int:
    print(f"triquad.py sha256 {triquad_sha()}", flush=True)
    bodies = args.bodies.split(",")
    t_roll = 0.0
    for body, seed, arm, items in plan(bodies, tq.parse_seeds(args.seeds)):
        p, env, facts = tq.make_env(body, seed)
        for item in items:
            try:
                row = run_row(env, body, arm, item, seed)
            except Exception as exc:  # spec section 5: an item that raises ends the table
                row = {"body": body, "arm": arm, "item": item, "seed": seed, "score": None,
                       "failure": f"exception: {type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
                append(row)
                print(brief(row), flush=True)
                print(row["traceback"], flush=True)
                return 2
            t_roll += row["wall_s"]
            append(row)
            print(brief(row), flush=True)
    print(f"rollout wall time {t_roll:.0f} s; triquad.py sha256 {triquad_sha()}", flush=True)
    return 0


def one_row(args) -> int:
    p, env, facts = tq.make_env(args.body, args.seed)
    order = list(SEG)
    if args.item in SEG:                      # media run in DOMAIN_CYCLE order on one environment
        for it in order[:order.index(args.item)]:
            run_row(env, args.body, args.arm, it, args.seed)
    row = run_row(env, args.body, args.arm, args.item, args.seed)
    print(json.dumps({k: v for k, v in row.items() if k not in ("m", "components", "legs")},
                     default=float))
    return 0


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def failed(r) -> str:
    """Reason a row is not a measurement, or ''."""
    if r.get("score") is None:
        return r.get("failure") or "exception"
    if int(r.get("bad_qacc", 0)) > 0:
        return "bad_qacc"
    f = r.get("failure") or ""
    if r["item"] in SEG and f:
        return f
    if r["item"] in MISS and (f or not r.get("survived", True)):
        return f or "not survived"
    if r["item"] in tq.CROSSINGS and f in ("diverged", "battery exhausted mid-transition"):
        return f
    return ""


def summary(args) -> int:
    by = defaultdict(dict)
    for line in open(args.path or OUT):
        r = json.loads(line)
        by[(r["body"], r["arm"], r["item"])][r["seed"]] = r
    print("| body | arm | item | n | failed rows | result |")
    print("|---|---|---|---|---|---|")
    for body in ("B1", "B0"):
        for arm in ("base", "shuffled", "blind"):
            for item in list(SEG) + list(tq.CROSSINGS) + list(MISS):
                rs = by.get((body, arm, item))
                if not rs:
                    continue
                seeds = sorted(rs)
                bad = [(s, failed(rs[s])) for s in seeds if failed(rs[s])]
                good = [rs[s] for s in seeds if not failed(rs[s])]
                if item in SEG:
                    v = [r["score"] for r in good]
                    res = (f"{sum(x >= 0.012 for x in v)}/{len(seeds)} >= 0.012; "
                           f"{sum(x >= 0.15 for x in v)}/{len(seeds)} >= 0.15; "
                           + (f"min {min(v):.4g} med {st.median(v):.4g} max {max(v):.4g}" if v else "-")
                           + "; per seed " + " ".join(
                               ("F" if failed(rs[s]) else f"{rs[s]['score']:.3f}") for s in seeds))
                elif item in tq.CROSSINGS:
                    res = (f"{sum(bool(r['crossed']) for r in good)}/{len(seeds)} crossed; approach max "
                           f"{max(r['score'] for r in good):.3f}; hold "
                           f"{min(r['hold'] for r in good):.2f}-{max(r['hold'] for r in good):.2f}; "
                           "per entry state " + " ".join(
                               ("F" if failed(rs[s]) else ("X" if rs[s]["crossed"] else "-")
                                + f"{rs[s]['score']:.2f}") for s in seeds))
                else:
                    r = rs[seeds[0]]
                    res = (f"transitions {r['transitions'][0]}/{r['transitions'][1]}; continuous_score "
                           f"{r['score']:.3f}; legs " + ", ".join(
                               f"{l['cmd']} {'entered' if l['entered'] else 'no'} on-task {l['on_task']:.2f}"
                               for l in r.get("legs", [])) + (f"; {r['failure']}" if r["failure"] else ""))
                print(f"| {body} | {arm} | {item} | {len(seeds)} | "
                      f"{'; '.join(f'{s}: {why}' for s, why in bad) or '0'} | {res} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ident")
    t = sub.add_parser("table")
    t.add_argument("--bodies", default="B1,B0")
    t.add_argument("--seeds", default="0-7")
    r = sub.add_parser("row")
    r.add_argument("--body", default="B1")
    r.add_argument("--arm", default="base")
    r.add_argument("--item", default="land_to_air")
    r.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("summary")
    s.add_argument("path", nargs="?", default="")
    args = ap.parse_args()
    return {"ident": lambda a: ident(), "table": table, "row": one_row, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
