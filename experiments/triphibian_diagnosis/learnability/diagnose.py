"""L1: where does the student lose the exit from the water?  (PREREG_exit.md.)

Arms S, M, F, C, X fly L0's round-3 student with parts of its action taken
from the teacher; arm W refits it with exit-window rows weighted 10x.

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python \
        experiments/triphibian_diagnosis/learnability/diagnose.py --jobs 8
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402

I_WET, I_DEPTH, I_CMD_WATER = 39, 9, 12
MEM = 9


def in_exit(o) -> bool:
    """Commanded out of the water while still wet (PREREG_exit.md section 1)."""
    o = np.asarray(o)
    commanded = o[11:14].max() > 0.5 and o[I_CMD_WATER] < 0.5
    return bool(commanded and (o[I_WET:I_WET + 4].max() > 0.01 or o[I_DEPTH] > 0.0))


def exit_rows(X) -> np.ndarray:
    X = np.asarray(X)
    return ((X[:, 11:14].max(1) > 0.5) & (X[:, I_CMD_WATER] < 0.5)
            & ((X[:, I_WET:I_WET + 4].max(1) > 0.01) | (X[:, I_DEPTH] > 0.0)))


def fly(args):
    arm, w, start, seed = args
    from dytiscidae.control.innerloop import InnerLoop, body_model
    from dytiscidae.control.outer import ShoreMap, SplitController, Teacher
    from dytiscidae.core.bodyplans import ANCHOR_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import mission as M
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    t0 = time.time()
    env = TriphibianEnv(build(ANCHOR_PLANS["legged_quad"]()), seed=seed)
    model = body_model(env)
    teach = Teacher(model, ShoreMap()).teach
    student = P.student_law(w)
    tr = {"n": 0, "exit": 0, "flag_miss": [0, 0, 0, 0], "cont_max": 0.0, "cont_sum": 0.0,
          "first_miss_since": None}

    def law(o):
        us, ut = student(o), teach(o)
        ex = in_exit(o)
        tr["n"] += 1
        if ex:
            tr["exit"] += 1
            miss = [bool(us[j] != ut[j]) for j in P.FLAG]
            for k, m in enumerate(miss):
                tr["flag_miss"][k] += m
            if any(miss) and tr["first_miss_since"] is None:
                tr["first_miss_since"] = float(o[43])
            e = float(np.max(np.abs(us[P.CONT] - ut[P.CONT]) / w["y_std"]))
            tr["cont_max"] = max(tr["cont_max"], e)
            tr["cont_sum"] += e
        u = us.copy()
        if arm == "M":
            u[MEM] = ut[MEM]
        elif arm == "F":
            u[P.FLAG] = ut[P.FLAG]
        elif arm == "C":
            u[P.CONT] = ut[P.CONT]
        elif arm == "X" and ex:
            u = ut.copy()
        return u

    row = {"arm": arm, "start": start, "seed": seed}
    try:
        with SplitController(law, InnerLoop(env, model).act).installed(env) as sc:
            read = M.evaluate_mission(env, sc.controller, Domain(start), seed)
        row.update({"held": read.held, "completed": read.completed, "failure": read.failure,
                    "transitions": [read.transitions_completed, read.transitions_commanded]})
    except Exception as exc:
        row.update({"completed": False, "failure": f"exception: {exc!r}", "held": [],
                    "transitions": [0, 2]})
    row["trace"] = tr
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    w3 = dict(np.load(HERE / "data" / "weights3.npz"))
    d = np.load(HERE / "data" / "round3.npz")
    X, Y, hold = d["X"], d["Y"], d["hold"]
    ex = exit_rows(X)
    info = {"rows": int(len(X)), "exit_rows": int(ex.sum()),
            "memory_on_rows": int((Y[:, MEM] > 0.5).sum())}
    print(json.dumps(info), flush=True)
    wW, statsW = P.fit(X, Y, hold, weight=np.where(ex, 10.0, 1.0))
    np.savez(HERE / "data" / "weightsW.npz", **wW)
    out = HERE / "results_exit.jsonl"
    summary = {"info": info, "fit_W": statsW}
    seeds = list(range(200, 200 + a.n))
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for arm in ("S", "M", "F", "C", "X", "W"):
            w = wW if arm == "W" else w3
            rows = pool.map(fly, [("S" if arm == "W" else arm, w, s, seed)
                                  for s in P.STARTS for seed in seeds], chunksize=1)
            for r in rows:
                r["arm"] = arm
            with open(out, "a") as f:
                for r in rows:
                    f.write(json.dumps(r, default=float) + "\n")
            summary[arm] = P.tally(rows)
            print(arm, json.dumps({s: [v["completed"], v["failed"]] for s, v in summary[arm].items()}),
                  flush=True)
            if arm in ("S", "W"):
                for done in (True, False):
                    v = [r for r in rows if bool(r["completed"]) == done and r["trace"]["exit"]]
                    if v:
                        summary[f"{arm}_trace_{'completed' if done else 'missed'}"] = {
                            "missions": len(v),
                            "with_any_flag_miss": sum(any(r["trace"]["flag_miss"]) for r in v),
                            "flag_miss_decisions": [sum(r["trace"]["flag_miss"][k] for r in v)
                                                    for k in range(4)],
                            "exit_decisions": sum(r["trace"]["exit"] for r in v),
                            "cont_max_median": float(np.median([r["trace"]["cont_max"] for r in v])),
                            "cont_mean": float(np.mean([r["trace"]["cont_sum"] / r["trace"]["exit"]
                                                        for r in v]))}
            with open(HERE / "summary_exit.json", "w") as f:
                json.dump(summary, f, indent=1)
    for k in summary:
        if "trace" in k:
            print(k, json.dumps(summary[k]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
