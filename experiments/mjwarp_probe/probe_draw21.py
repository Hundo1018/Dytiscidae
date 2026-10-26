# Draw-21 "unstable" on the device (2026-10-26). Step 1: is it world 0 or seed 21?
# evaluate_tier1_device takes ONE seed per phenotype (seeds_l[i]) and runs k_mean+k_sampled
# draws; probe_a5_failures passed seeds=range(21,29) with one phenotype, so only seed 21 ran.
# Here: each seed alone, and the draws per device run listed, with the failing world's step count.
# Usage: python experiments/mjwarp_probe/probe_draw21.py [plan ...]
import sys, time
import warp as wp
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.devroll import rollout_wp as R
from tests.test_devroll_agreement import _a4_controller

plans = sys.argv[1:] or ["medusa", "ray"]
for plan in plans:
    p = build(BODY_PLANS[plan]())
    ctrl, _ = _a4_controller(p, 171)
    for seeds in ([21], [22], [23], list(range(21, 29))):
        t = time.time()
        dev = R.evaluate_tier1_device([p], controllers=[ctrl], seeds=seeds, segment_seconds=8.0,
                                      dtype=wp.float32)[0]
        ran = [d["seed"] for d in dev.draws]
        for (k, m, h), rec in sorted(dev.recorded.items()):
            s = dev.draws[k][m + "_halves"][h]
            print(f"RES {plan} arg={seeds[0]}{'..' if len(seeds) > 1 else ''} draws_run={ran} "
                  f"{m} half={h} failure={s.failure or '-'} bad_qacc={s.bad_qacc} "
                  f"steps={rec['steps']} decisions={rec['decisions']} ({time.time() - t:.0f}s)",
                  flush=True)
