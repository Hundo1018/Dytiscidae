"""P0.3 measurement: the mission verdict (`mission.evaluate_mission`) on the
hand-driven bodies and on arms that do nothing or do not read the command.

No learning.  Arms:

    teacher   the E0 hand law (on L1, with the E0b gait on land)
    still     every actuator held still (`held_still_params`)
    base      the body's open-loop base gait, no policy
    frozen    the hand law, told for the whole mission the first medium it saw
    timed     the hand law on a clock: the cycle at the mean leg duration
    otherdraw the hand law on a clock: the leg durations of seed + 1

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python \
        experiments/triphibian_diagnosis/scripts/mission_eval.py OUT.jsonl \
        --bodies B1,L1 --arms teacher,still --seeds 0-31 --shard 0/4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "existence"))

import landgait as lg  # noqa: E402
import nulls as nl  # noqa: E402
import triquad as tq  # noqa: E402
from dytiscidae.envs import mission as M  # noqa: E402
from dytiscidae.envs.evaluate import Controller  # noqa: E402
from dytiscidae.envs.triphibian import DOMAIN_CYCLE, Domain  # noqa: E402

ARMS = ("teacher", "still", "base", "frozen", "timed", "otherdraw")
_BAD = tq._BAD


class ClockPolicy(tq.HandPolicy):
    """`HandPolicy`, except that the medium handed to the law comes from a
    clock: the cycle from ``start`` at fixed leg durations, whatever the
    evaluator commands."""

    def __init__(self, env, start: Domain, seconds):
        self.env, self.i0 = env, DOMAIN_CYCLE.index(start)
        self.edges = np.cumsum(seconds)
        self.t0 = None
        self._commanded = None

    @property
    def commanded(self):
        return self._commanded

    @commanded.setter
    def commanded(self, value):
        # `Hand._reset` writes None here at the first step of every rollout.
        self._commanded = value
        if value is None:
            self.t0 = None

    def act(self, obs):
        t = float(self.env.data.time)
        if self.t0 is None:
            self.t0 = t
        k = int(np.searchsorted(self.edges, t - self.t0, side="right"))
        self._commanded = DOMAIN_CYCLE[(self.i0 + k) % len(DOMAIN_CYCLE)]
        return np.zeros(1)


def controller(env, body: str, arm: str, start: Domain, seed: int, leg_range=None):
    """``(controller, hand or None)``."""
    if arm == "still":
        return Controller(params=env.held_still_params(), policy=None, bases=None), None
    if arm == "base":
        return Controller(params=env.cpg.base, policy=None, bases=None), None
    if body in ("L1", "W1"):
        ctrl, hand, _ = lg.controller_for(env, "walk")
    else:
        ctrl, hand = tq.controller_for(env, "hand", False)
    if arm == "teacher":
        return ctrl, hand
    if arm == "frozen":
        pol = nl.BlindPolicy(env)
    else:
        leg_range = leg_range or M.LEG_SECONDS_RANGE
        mean = sum(leg_range) / 2.0
        secs = ([mean] * 3 if arm == "timed"
                else [s for _, s in M.mission_schedule(start, seed + 1, leg_range=leg_range)])
        pol = ClockPolicy(env, start, secs)
    hand.policy = pol
    hand.ctrl = Controller(params=hand.token, policy=pol,
                           bases={"air": hand.basis, "water": hand.basis})
    return hand.ctrl, hand


def run(body: str, arm: str, start: Domain, seed: int, leg_range=None) -> dict:
    t0 = time.time()
    _, env, _ = tq.make_env(body, seed)
    row = {"body": body, "arm": arm, "start": start.value, "seed": seed}
    hand = None
    try:
        leg_range = leg_range or M.LEG_SECONDS_RANGE
        row["leg_range"] = list(leg_range)
        ctrl, hand = controller(env, body, arm, start, seed, leg_range)
        read = M.evaluate_mission(env, ctrl, start, seed, leg_range=leg_range)
        res = read.result
        row.update({
            "legs": [[d.value, round(s, 3)] for d, s in read.schedule],
            "held": read.held, "shares": [round(x, 3) for x in read.shares],
            "supports": [None if x is None else round(x, 3) for x in read.supports],
            "completed": read.completed,
            "transitions": [read.transitions_completed, read.transitions_commanded],
            "failure": read.failure, "bad_qacc": int(env.data.warning[_BAD].number),
            "old_transitions": [res.transitions_completed, res.transitions_commanded],
            "old_score": float(M.continuous_score(res, M.MissionSpec())),
            "energy_wh": float(res.energy_wh), "total_time": float(res.total_time),
        })
    except Exception as exc:  # a broken rollout is a row, not a crash and not a zero
        row["failure"] = f"exception: {exc!r}"
    finally:
        if hand is not None and hasattr(hand, "_orig"):
            env.cpg.command = hand._orig
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--bodies", default="B1,L1")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0-31")
    ap.add_argument("--starts", default="air,water,land")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--leg-range", default="")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    k, n = (int(x) for x in a.shard.split("/"))
    out = Path(a.out).resolve()
    jobs = [(b, arm, Domain(s), seed) for b in a.bodies.split(",") for arm in a.arms.split(",")
            for s in a.starts.split(",") for seed in range(lo, hi + 1)]
    for j, (b, arm, start, seed) in enumerate(jobs):
        if j % n != k:
            continue
        row = run(b, arm, start, seed,
                  tuple(float(x) for x in a.leg_range.split(",")) if a.leg_range else None)
        with open(out, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        print(json.dumps({x: row.get(x) for x in ("body", "arm", "start", "seed", "held",
                                                  "shares", "supports", "completed", "failure",
                                                  "wall_s")}, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
