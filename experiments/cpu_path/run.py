"""What the search does on a machine without the GPU fluid kernel.

Runs a small search through ``run_search`` with the shared policy on and
reports which evaluator ran, what reached PPO, and the wall and CPU time per
generation (CPU time over wall is how many cores stepped machines).

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/cpu_path/run.py <run_dir> \
        [--generations 2] [--batch 4] [--workers 4] [--segment-seconds 1.0]

Measured 2026-10-12 on a 4-core Xeon container with no GPU, 2 generations,
batch 4, 4 workers, 1 s segments, generalist island (ROADMAP "Answers
2026-10-12, questions 4-6"):

- before ``batchroll.HostFluid``: batched 0 calls, single 10; both PPO events
  0 transitions and skipped; 600 s wall, 593 s CPU (one core);
- after: see ``results.json`` beside this file.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--segment-seconds", type=float, default=1.0)
    ap.add_argument("--out", default=str(Path(__file__).with_name("results.json")))
    args = ap.parse_args()

    from dytiscidae.envs import batchroll
    from dytiscidae.envs.triphibian import MissionSpec
    from dytiscidae.evolution import loop

    calls = {"batched": 0, "single": 0}
    _b, _s = loop._batched, loop.evaluate_candidate

    def batched(*a, **k):
        calls["batched"] += 1
        return _b(*a, **k)

    def single(*a, **k):
        calls["single"] += 1
        return _s(*a, **k)

    loop._batched, loop.evaluate_candidate = batched, single
    run = Path(args.run_dir)
    cfg = loop.SearchConfig(
        generations=args.generations, batch=args.batch, workers=args.workers, seed=3,
        segment_seconds=args.segment_seconds, n_reference_seeds=2, n_random_seeds=0,
        islands=("generalist",), tier2_every=999, audit_every=999, migrate_every=999,
        checkpoint_every=999, run_dir=str(run), use_shared_policy=True)
    t0 = time.perf_counter()
    loop.run_search(cfg, MissionSpec())
    wall = time.perf_counter() - t0
    ru_self = resource.getrusage(resource.RUSAGE_SELF)
    ru_kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (ru_self.ru_utime + ru_self.ru_stime + ru_kids.ru_utime + ru_kids.ru_stime)

    ev = [json.loads(line) for line in open(run / "events.jsonl")]
    kinds: dict = {}
    for e in ev:
        k = e.get("kind") or e.get("event")
        kinds[k] = kinds.get(k, 0) + 1
    ppo = [{k: e.get(k) for k in ("gen", "transitions", "trajectories", "kl",
                                  "grad_steps", "skipped")}
           for e in ev if (e.get("kind") or e.get("event")) == "ppo"]
    gens = [json.loads(line) for line in open(run / "generations.jsonl")]
    out = {
        "kernel_usable": bool(batchroll.AVAILABLE),
        "reason": batchroll.UNAVAILABLE_REASON,
        "config": {"generations": args.generations, "batch": args.batch,
                   "workers": args.workers, "segment_seconds": args.segment_seconds},
        "cores": os.cpu_count(),
        "evaluator_calls": calls,
        "event_kinds": kinds,
        "ppo": ppo,
        "generations": [{"gen": g.get("generation"),
                         "evaluate_s": (g.get("cost") or {}).get("seconds", {}).get("evaluate"),
                         "ppo_s": (g.get("cost") or {}).get("seconds", {}).get("ppo"),
                         "steps": (g.get("cost") or {}).get("steps")}
                        for g in gens if "generation" in g],
        "wall_s": round(wall, 1),
        "cpu_s": round(cpu, 1),
        "cores_used": round(cpu / wall, 2),
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
