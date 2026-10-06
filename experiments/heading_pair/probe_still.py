"""Probe: a held-still machine scores exactly zero cruise progress under the
antipodal pair, and the batched and single-machine paths agree (2026-10-06)."""
import sys
import numpy as np
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.evaluate import Controller, evaluate_tier1
from dytiscidae.envs.batchroll import evaluate_tier1_batch
from dytiscidae.envs.triphibian import TriphibianEnv

plans = sys.argv[1].split(",") if len(sys.argv) > 1 else ["eel", "gannet"]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ps = [build(BODY_PLANS[n]()) for n in plans]
still = [Controller(params=TriphibianEnv(p, seed=seed).held_still_params()) for p in ps]
base = [Controller(params=None) for _ in ps]
for name, ctrls in (("still", still), ("base", base)):
    single = [evaluate_tier1(p, controller=c, segment_seconds=8.0, seed=seed,
                             identify_axes=False) for p, c in zip(ps, ctrls)]
    batch = evaluate_tier1_batch(ps, controllers=ctrls, segment_seconds=8.0, seed=seed)
    for n, a, b in zip(plans, single, batch):
        for m in ("water", "land"):
            sa, sb = a.segments[m], b.segments[m]
            key = "cruise_progress" if m == "water" else "walk_progress"
            print(f"{name:5s} {n:7s} {m:5s} comp single {sa.competence:.6f} batch {sb.competence:.6f}"
                  f"  progress {sa.measurements.get(key)!s:>22}  pair_mean {sa.measurements.get("pair_along_mean")!r} batch {sb.measurements.get("pair_along_mean")!r}")
