# CPU only. Eel water, A6 draws (eval_seed 9200+i, placement seed i for i in 0..3), three host paths:
# single evaluate_tier1 vs batched evaluate_tier1_batch. (a) which path is the outlier against the device
# numbers recorded in docs/ROADMAP.md; (b) whether the host calls mj_setConst (body_simple) for the eel.
import numpy as np
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs import batchroll as br
from dytiscidae.envs.evaluate import evaluate_tier1
from dytiscidae.envs.triphibian import TriphibianEnv
from dytiscidae.envs.triphibian import Domain
from tests.test_devroll_agreement import _a4_controller

p = build(BODY_PLANS["eel"]())
ctrl, _ = _a4_controller(p, 171)
env = TriphibianEnv(p, seed=0)
# body 0 is the world body, which is always simple; the eel's own bodies start at 1.
print("(b) eel body_simple any (bodies 1..):", bool(np.any(env.model.body_simple[1:])), flush=True)
single = []
for i in range(4):
    r = evaluate_tier1(p, controller=ctrl, segment_seconds=8.0, seed=9200 + i)
    single.append(r.segments["water"].competence)
print("(a) single water per draw:", np.round(single, 4).tolist(), "mean", round(float(np.mean(single)), 4), flush=True)
