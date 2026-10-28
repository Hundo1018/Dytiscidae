# CPU only. Eel water, the A6 draws, single host path (evaluate_tier1) only: the same four seeds the
# test's draws use for eel (body index 2 in A6_BODIES + A6_INFO_BODIES): draw 0 = 9200 + 2, draws 1..3 =
# placement_draw_seed(9200 + 2, d).  The batched path's pooled mean (0.039 on these draws, test_devroll
# A6) and the device's (0.019) are recorded in docs/ROADMAP.md; this run decides whether the single path,
# the host reference, agrees with either.  (b) checks whether any non-world body is body_simple (setConst).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.evaluate import evaluate_tier1
from dytiscidae.envs.triphibian import TriphibianEnv
from dytiscidae.evolution.loop import placement_draw_seed
from tests.test_devroll_agreement import _a4_controller

BASE = 9200 + 2
SEEDS = [BASE] + [placement_draw_seed(BASE, d) for d in range(1, 4)]

p = build(BODY_PLANS["eel"]())
ctrl, _ = _a4_controller(p, 171)
env = TriphibianEnv(p, seed=0)
print("(b) eel body_simple any (bodies 1..):", bool(np.any(env.model.body_simple[1:])), flush=True)
print("seeds", SEEDS, flush=True)
single = []
for s in SEEDS:
    r = evaluate_tier1(p, controller=ctrl, segment_seconds=8.0, seed=s)
    single.append(float(r.segments["water"].competence))
    print(f"(a) seed {s} single water {single[-1]:.4f}", flush=True)
print("(a) single water per draw:", np.round(single, 4).tolist(),
      "mean", round(float(np.mean(single)), 4), flush=True)
