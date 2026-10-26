# Ray, device rollout (evaluate_tier1_device, float32) with the A5 controller, 8 s segments: which
# segments end with non-finite qacc, and in which medium. Same calls as the A5 test.
import sys
import numpy as np, warp as wp
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.devroll import rollout_wp as R
from tests.test_devroll import _a4_controller

plan = sys.argv[1] if len(sys.argv) > 1 else "ray"
p = build(BODY_PLANS[plan]())
ctrl, _ = _a4_controller(p, 171)
for seed in range(21, 21 + int(sys.argv[2]) if len(sys.argv) > 2 else 24):
    r = R.evaluate_tier1_device([p], controllers=[ctrl], seeds=[seed], segment_seconds=8.0,
                                dtype=wp.float32)[0]
    notes = r.notes
    print(plan, "seed", seed, "notes", str(notes)[:300], flush=True)
    for d in r.draws[:2]:
        for dom in ("air", "water"):
            seg = d[dom]
            print(f"  seed {d['seed']} {dom}: failure={seg.failure} bad_qacc={seg.bad_qacc} "
                  f"survived={seg.survived} peak_slam={seg.peak_slam:.4g} competence={seg.competence:.4g} "
                  f"max_actuator_overload={seg.max_actuator_overload:.4g}", flush=True)
