"""Does anything fly?  Seed plans, open loop, one 8 s air segment each.

For every seed plan and a sweep of flap frequencies (the plan's own gait with
its frequency replaced), run the real air rollout from the real spawn and
report the score, the airborne time, the measured sink and thrust_margin.
Run it on two trees to compare physics:  python experiments/flight_audit/fly_probe.py
"""
import sys, time
from dataclasses import replace
import numpy as np
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

PLANS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["gannet", "teal", "bat", "beetle"]
FREQS = [None, 4.0, 7.0, 11.0]
for name in PLANS:
    for f in FREQS:
        env = TriphibianEnv(build(BODY_PLANS[name]()), seed=0)
        base = env.cpg.base if f is None else replace(env.cpg.base, frequency=f)
        env.cpg.base = base
        tm = env.thrust_margin()
        env.reset(Domain.AIR, randomise=False)
        t0 = time.time()
        r = env.rollout(8.0, params=base, domain=Domain.AIR)
        m = r.measurements
        print(f"{name:7s} f={base.frequency:5.2f}  score {r.competence:.3f}  "
              f"airborne {m.get('airborne_seconds', m.get('airborne_fraction', float('nan'))):.2f}  "
              f"sink {m.get('measured_sink_rate', m.get('sink_rate', float('nan'))):6.2f}  "
              f"lift {m.get('lift_margin', float('nan')):5.2f}  thrust {tm if tm is not None else float('nan'):+.3f}  "
              f"height_hold {m.get('height_hold', float('nan')):.3f}  ({time.time()-t0:.1f}s)", flush=True)
