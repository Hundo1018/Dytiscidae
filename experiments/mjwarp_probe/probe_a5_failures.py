# Failing segments by body, path, draw and medium for the A5 draws (8 draws per body, 8 s).
# The device takes one eval seed per body and derives its draws (draw 0 the seed, draw d
# placement_draw_seed(seed, d)); the batched path is given the same eight draw seeds.
# (The first version passed seeds=range(21, 29) to the device with k_mean=1, so it ran
# only draw 21 there.)
import warp as wp
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.devroll import rollout_wp as R
from dytiscidae.envs import batchroll as br
from dytiscidae.evolution.loop import placement_draw_seed
from tests.test_devroll_agreement import _a4_controller

for plan in ["medusa", "ray", "bat", "eel", "teal", "beetle", "gannet"]:
    p = build(BODY_PLANS[plan]())
    ctrl, _ = _a4_controller(p, 171)
    seeds = [21] + [placement_draw_seed(21, d) for d in range(1, 8)]
    dev = R.evaluate_tier1_device([p], controllers=[ctrl], seeds=[21], k_mean=8,
                                  segment_seconds=8.0, dtype=wp.float32)[0]
    for d in dev.draws:
        for dom in ("air", "water"):
            s = d[dom]
            if s.failure:
                print(f"{plan:8s} device  draw {d['seed']} {dom:5s} {s.failure} bad_qacc={s.bad_qacc}", flush=True)
    host = br.evaluate_tier1_batch([p] * len(seeds), controllers=[ctrl] * len(seeds), seed=seeds,
                                   segment_seconds=8.0, identify_axes=False)
    for k, res in enumerate(host):
        for dom, s in res.segments.items():
            if getattr(s, "failure", ""):
                print(f"{plan:8s} batched draw {seeds[k]} {dom:5s} {s.failure} bad_qacc={s.bad_qacc}", flush=True)
