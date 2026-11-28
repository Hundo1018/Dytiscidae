#!/usr/bin/env python3
"""Gate 1 of PORT_SPEC_p05_p06.md: the package's split controller reproduces
every row of `existence/results_stateless.jsonl`, field for field.

The body is `core.bodyplans.ANCHOR_PLANS["legged_quad"]`, the controller
`control.outer` over `control.innerloop`.  The row harness (segment, crossing
and entry-state helpers, the support recorder) is the prototype's, imported
and not edited; the prototype's controller (`stateless.py`) is not imported.
Every stored field except `wall_s` must be equal after a JSON round trip.

    cd /home/hundo/Projects/Dytiscidae/dytiscidae; export MUJOCO_GL=disable PYTHONPATH=.
    .venv/bin/python experiments/triphibian_diagnosis/scripts/port_check.py --jobs 4

About 30 minutes on 4 processes.  Prints rows compared and rows that differ;
exit 0 only if every stored row was compared and none differs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXIST = HERE.parent / "existence"
STORED = EXIST / "results_stateless.jsonl"
ARMS = ("teach", "step", "zero", "const", "frozen", "nomem")
STARTS = ("air", "water", "land")
SKIP = ("wall_s",)


def jobs(kind):
    if kind == "mission":
        return [(arm, seed, st) for arm in ARMS for seed in range(32) for st in STARTS]
    return [(arm, seed, None) for arm in ("teach", "step") for seed in range(8)]


def key(r):
    return (r["kind"], r["arm"], r.get("start"), r.get("item"), r["seed"])


def worker(kind: str, k: int, n: int) -> int:
    if kind == "crossing":
        os.environ["DYTISCIDAE_CROSSING_GATES"] = "1"      # as the stored rows were run
    else:
        os.environ.pop("DYTISCIDAE_CROSSING_GATES", None)
    sys.path.insert(0, str(EXIST))
    import numpy as np
    import landgait as lg
    import nulls as nl
    import triquad as tq
    import dytiscidae.envs.transitions as T
    from dytiscidae.control.innerloop import (N_ACT, U_ACC, U_DOWN, U_DRIVE, U_ON, U_POSE, U_TILT,
                                              U_TURN, InnerLoop, body_model)
    from dytiscidae.control.outer import ShoreMap, SplitController, Teacher
    from dytiscidae.core.bodyplans import ANCHOR_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import mission as M
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    def make_env(seed):
        env = TriphibianEnv(build(ANCHOR_PLANS["legged_quad"]()), seed=seed)
        # the body-level facts `triquad.make_env` measures before any controller
        env.launch_speed, env.level_margin(), env.flies_level(), env.air_gates, env.thrust_margin()
        return env

    def controller(env, arm, seed):
        model = body_model(env)
        law = Teacher(model, ShoreMap()).teach
        if arm == "zero":
            law = lambda o: np.zeros(N_ACT)                                  # noqa: E731
        elif arm == "const":                       # the prototype's draw (`stateless.Split`)
            r = np.random.default_rng([int(seed), 0x434F4E53])
            cu = np.zeros(N_ACT)
            cu[U_ACC:U_ACC + 3] = r.uniform(-8.0, 8.0, 3)
            cu[U_TILT] = r.uniform(0.0, 45.0)
            cu[U_DOWN], cu[U_ON] = float(r.random() < 0.5), float(r.random() < 0.8)
            cu[U_DRIVE], cu[U_TURN], cu[U_POSE] = r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(0, 1)
            law = lambda o: cu.copy()                                        # noqa: E731
        return SplitController(law, InnerLoop(env, model).act, every_step=(arm == "step"),
                               feed_memory=(arm != "nomem"),
                               freeze_command=(arm == "frozen")).install(env)

    def row_mission(arm, start, seed):
        env = make_env(seed)
        sp = controller(env, arm, seed)
        row = {"kind": "mission", "body": "L1", "arm": arm, "start": start, "seed": seed}
        try:
            read = M.evaluate_mission(env, sp.controller, Domain(start), seed)
            res = read.result
            row.update(legs=[[d.value, round(s, 3)] for d, s in read.schedule], held=read.held,
                       shares=[round(x, 3) for x in read.shares],
                       supports=[None if x is None else round(x, 3) for x in read.supports],
                       completed=bool(read.completed),
                       transitions=[read.transitions_completed, read.transitions_commanded],
                       failure=read.failure, bad_qacc=int(env.data.warning[tq._BAD].number),
                       energy_wh=float(res.energy_wh), total_time=float(res.total_time),
                       end_qpos=[float(x) for x in env.data.qpos[:7]], decisions=sp.decisions)
        except Exception as exc:
            row["failure"] = f"exception: {exc!r}"
            row["traceback"] = traceback.format_exc()[-1200:]
        finally:
            sp.uninstall(env)
        return [row]

    def rows_segments(arm, seed):
        env = make_env(seed)
        rows = []
        for item in ("air", "water", "land"):
            sp = controller(env, arm, seed)
            rec = lg.Recorder(env, None)
            try:
                row = tq.run_seg_item(env, sp.controller, Domain(item), seed)
                row.update(rec.summary() if item == "land" else {})
            except Exception as exc:
                row = {"score": None, "failure": f"exception: {exc!r}",
                       "traceback": traceback.format_exc()[-1200:]}
            finally:
                rec.close()
                sp.uninstall(env)
            rows.append({"kind": "segment", "body": "L1", "arm": arm, "item": item, "seed": seed, **row})
        return rows

    def rows_crossings(arm, entry):
        env = make_env(entry)
        rows = []
        for kind_ in tq.CROSSINGS:
            sp = controller(env, arm, entry)
            try:
                nl.set_entry_state(entry)
                row = tq.run_cross_item(env, sp.controller, kind_)
            except Exception as exc:
                row = {"score": None, "failure": f"exception: {exc!r}",
                       "traceback": traceback.format_exc()[-1200:]}
            finally:
                T.transition_scatter_seed = nl._TSS
                sp.uninstall(env)
            rows.append({"kind": "crossing", "body": "L1", "arm": arm, "item": kind_, "seed": entry,
                         "gates": os.environ.get("DYTISCIDAE_CROSSING_GATES", ""), **row})
        return rows

    if k == 0 and kind == "mission":
        a, b = ANCHOR_PLANS["legged_quad"](), lg.l1_genome()
        b.body_plan = a.body_plan                 # the stamp is the one intended difference
        same = json.dumps(dataclasses.asdict(a), default=repr) == json.dumps(dataclasses.asdict(b), default=repr)
        print(json.dumps({"genome_equal": bool(same)}), flush=True)

    stored = {key(r): r for r in map(json.loads, STORED.read_text().splitlines())}
    compared, differ = 0, []
    for j, (arm, seed, start) in enumerate(jobs(kind)):
        if j % n != k:
            continue
        rows = (row_mission(arm, start, seed) if kind == "mission" else
                rows_segments(arm, seed) if kind == "segment" else rows_crossings(arm, seed))
        for r in rows:
            r = json.loads(json.dumps(r, default=float))
            ref = stored.get(key(r))
            compared += 1
            if ref is None:
                differ.append({"key": key(r), "fields": ["not stored"]})
                continue
            fields = sorted(f for f in (set(r) | set(ref)) - {"note", *SKIP} if r.get(f) != ref.get(f))
            if fields:
                differ.append({"key": key(r), "fields": fields,
                               "got": {f: r.get(f) for f in fields[:4]},
                               "stored": {f: ref.get(f) for f in fields[:4]}})
    print(json.dumps({"kind": kind, "shard": f"{k}/{n}", "compared": compared, "differ": differ},
                     default=float), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--kinds", default="mission,segment,crossing")
    ap.add_argument("--worker", nargs=3, metavar=("KIND", "K", "N"))
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker[0], int(a.worker[1]), int(a.worker[2]))
    stored = [json.loads(ln) for ln in STORED.read_text().splitlines()]
    compared, differ, genome = 0, [], None
    for kind in a.kinds.split(","):
        procs = [subprocess.Popen([sys.executable, __file__, "--worker", kind, str(k), str(a.jobs)],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                 for k in range(a.jobs)]
        for p in procs:
            out, _ = p.communicate()
            got = [json.loads(ln) for ln in out.splitlines() if ln.startswith("{")]
            for g in got:
                if "genome_equal" in g:
                    genome = g["genome_equal"]
                else:
                    compared += g["compared"]
                    differ += g["differ"]
            if p.returncode or not any("compared" in g for g in got):
                differ.append({"key": [kind, "worker"], "fields": [f"exit {p.returncode}, no summary"]})
        want = sum(r["kind"] == kind for r in stored)
        print(f"{kind}: stored {want}", flush=True)
    for d in differ[:20]:
        print("DIFFERS", json.dumps(d, default=float)[:600])
    total = sum(r["kind"] in a.kinds.split(",") for r in stored)
    print(f"genome equal to the prototype's (body_plan stamp aside): {genome}")
    print(f"rows stored {total}, rows compared {compared}, rows differing {len(differ)}")
    return 0 if (compared == total and not differ and genome is not False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
