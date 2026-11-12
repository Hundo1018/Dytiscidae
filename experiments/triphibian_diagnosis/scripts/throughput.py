"""P0.12: decisions per second of the split controller on the single path.

No learning.  Each worker process flies missions on the anchor body through
`SplitController` for a fixed wall time and counts the outer law's decisions
and the simulation's steps.  The law is the teacher, or the teacher plus one
forward pass of a 64-64 network on the same observation (the cost a learned
policy adds; its output is thrown away).  Measured at several process counts,
because a modelled pool shape lost by 3x here (CLAUDE.md).

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python \
        experiments/triphibian_diagnosis/scripts/throughput.py OUT.json
"""
import json
import multiprocessing as mp
import resource
import sys
import time


def worker(args):
    k, seconds, with_net = args
    import numpy as np

    from dytiscidae.control.innerloop import InnerLoop, body_model
    from dytiscidae.control.outer import N_OBS, ShoreMap, SplitController, Teacher
    from dytiscidae.core.bodyplans import ANCHOR_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import mission as M
    from dytiscidae.envs.triphibian import DOMAIN_CYCLE, TriphibianEnv

    t_build = time.time()
    env = TriphibianEnv(build(ANCHOR_PLANS["legged_quad"]()), seed=k)
    model = body_model(env)
    teach = Teacher(model, ShoreMap()).teach
    law = teach
    if with_net:
        import torch

        torch.set_num_threads(1)
        net = torch.nn.Sequential(torch.nn.Linear(N_OBS, 64), torch.nn.Tanh(),
                                  torch.nn.Linear(64, 64), torch.nn.Tanh(),
                                  torch.nn.Linear(64, 16)).double()

        def law(o):
            with torch.no_grad():
                net(torch.as_tensor(o))
            return teach(o)

    sc = SplitController(law, InnerLoop(env, model).act).install(env)
    t_build = time.time() - t_build
    decisions = steps = missions = failed = 0
    t0 = time.time()
    seed = 1000 * k
    while time.time() - t0 < seconds:
        start = DOMAIN_CYCLE[seed % 3]
        read = M.evaluate_mission(env, sc.controller, start, seed)
        decisions += sc.decisions
        steps += int(round(read.result.total_time / env.timestep))
        missions += 1
        failed += bool(read.failure)
        seed += 1
    wall = time.time() - t0
    return {"decisions": decisions, "steps": steps, "missions": missions, "failed": failed,
            "wall": wall, "build_s": t_build,
            "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}


def main() -> int:
    out = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
    counts = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,4,8,12,16,20").split(",")]
    rows = []
    ctx = mp.get_context("spawn")
    for with_net in (False, True):
        for n in counts:
            if with_net and n not in (1, counts[len(counts) // 2], counts[-1]):
                continue
            t0 = time.time()
            with ctx.Pool(n) as pool:
                res = pool.map(worker, [(k, seconds, with_net) for k in range(n)])
            row = {"processes": n, "with_net": with_net,
                   "decisions_per_s": sum(r["decisions"] / r["wall"] for r in res),
                   "steps_per_s": sum(r["steps"] / r["wall"] for r in res),
                   "per_process_decisions_per_s": [round(r["decisions"] / r["wall"], 1) for r in res],
                   "missions": sum(r["missions"] for r in res),
                   "failed": sum(r["failed"] for r in res),
                   "rss_mb_max": max(r["rss_mb"] for r in res),
                   "rss_mb_sum": sum(r["rss_mb"] for r in res),
                   "build_s": max(r["build_s"] for r in res), "wall_total": time.time() - t0}
            rows.append(row)
            print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v)
                              for k, v in row.items() if k != "per_process_decisions_per_s"}),
                  flush=True)
            with open(out, "w") as f:
                json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
