"""The mission verdict on the packaged split controller (`control.outer`).

No learning.  The anchor body, the teacher through `SplitController`, and the
arms it is measured against, each a fresh environment with nothing run before
the mission:

    teach    the teacher
    zero     law(o) = 0
    frozen   the teacher shown the first commanded medium for the whole mission
    nomem    the teacher with its memory input forced to zero

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python \
        experiments/triphibian_diagnosis/scripts/split_eval.py OUT.jsonl --shard 0/4
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from dytiscidae.control.innerloop import N_ACT, InnerLoop, body_model
from dytiscidae.control.outer import SplitController, teacher_controller
from dytiscidae.core.bodyplans import ANCHOR_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs import mission as M
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

ARMS = ("teach", "zero", "frozen", "nomem")


def split(env, arm: str) -> SplitController:
    if arm == "zero":
        return SplitController(lambda o: np.zeros(N_ACT), InnerLoop(env, body_model(env)).act)
    return teacher_controller(env, freeze_command=arm == "frozen", feed_memory=arm != "nomem")


def run(arm: str, start: Domain, seed: int, leg_range) -> dict:
    t0 = time.time()
    env = TriphibianEnv(build(ANCHOR_PLANS["legged_quad"]()), seed=seed)
    row = {"body": "legged_quad", "arm": arm, "start": start.value, "seed": seed,
           "leg_range": list(leg_range)}
    try:
        with split(env, arm).installed(env) as sc:
            read = M.evaluate_mission(env, sc.controller, start, seed, leg_range=leg_range)
            row["decisions"] = sc.decisions
        row.update({
            "held": read.held, "shares": [round(x, 3) for x in read.shares],
            "supports": [None if x is None else round(x, 3) for x in read.supports],
            "completed": read.completed,
            "transitions": [read.transitions_completed, read.transitions_commanded],
            "failure": read.failure, "energy_wh": float(read.result.energy_wh),
            "total_time": float(read.result.total_time),
        })
    except Exception as exc:  # a broken rollout is a row, not a crash and not a zero
        row["failure"] = f"exception: {exc!r}"
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0-31")
    ap.add_argument("--starts", default="air,water,land")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--leg-range", default="")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    k, n = (int(x) for x in a.shard.split("/"))
    leg_range = (tuple(float(x) for x in a.leg_range.split(",")) if a.leg_range
                 else M.LEG_SECONDS_RANGE)
    out = Path(a.out).resolve()
    jobs = [(arm, Domain(s), seed) for arm in a.arms.split(",") for s in a.starts.split(",")
            for seed in range(lo, hi + 1)]
    for j, (arm, start, seed) in enumerate(jobs):
        if j % n != k:
            continue
        row = run(arm, start, seed, leg_range)
        with open(out, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        print(json.dumps({x: row.get(x) for x in ("arm", "start", "seed", "held", "completed",
                                                  "failure", "wall_s")}, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
