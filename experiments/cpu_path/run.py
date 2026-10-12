"""What the search does on a machine without the GPU fluid kernel.

Runs two small generations (batch 4, 4 workers, 1 s segments, generalist
island) through run_search with the shared policy on and reports which
evaluator ran and what reached PPO.  Compare wall and user time (`time`) to
see how many cores stepped machines. Re-run: PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/cpu_path/run.py <run_dir>

Measured 2026-10-12 on a 4-core Xeon container with no GPU (ROADMAP
"2026-10-12 -- the CPU path trains nothing"): batched 0 calls, single 10;
2 ppo events, each with 0 transitions and skipped; 600 s wall, 593 s user
(one core).
"""
import json
import sys
from pathlib import Path

from dytiscidae.envs import batchroll
from dytiscidae.envs.triphibian import MissionSpec
from dytiscidae.evolution import loop

run = Path(sys.argv[1])
calls = {"batched": 0, "single": 0}
_b, _s = loop._batched, loop.evaluate_candidate


def batched(*a, **k):
    calls["batched"] += 1
    return _b(*a, **k)


def single(*a, **k):
    calls["single"] += 1
    return _s(*a, **k)


loop._batched, loop.evaluate_candidate = batched, single
cfg = loop.SearchConfig(generations=2, batch=4, workers=4, seed=3, segment_seconds=1.0,
                        n_reference_seeds=2, n_random_seeds=0, islands=("generalist",),
                        tier2_every=999, audit_every=999, migrate_every=999,
                        checkpoint_every=999, run_dir=str(run), use_shared_policy=True)
loop.run_search(cfg, MissionSpec())
ev = [json.loads(l) for l in open(run / "events.jsonl")]
kinds = {}
for e in ev:
    kinds[e.get("kind") or e.get("event")] = kinds.get(e.get("kind") or e.get("event"), 0) + 1
ppo = [e for e in ev if (e.get("kind") or e.get("event")) == "ppo"]
print("kernel usable:", batchroll.AVAILABLE, batchroll.UNAVAILABLE_REASON[:60])
print("evaluator calls:", calls)
print("event kinds:", kinds)
print("ppo events:", len(ppo), [ {k: p.get(k) for k in ("transitions", "n", "samples", "updated", "skipped", "reason")} for p in ppo][:3])
