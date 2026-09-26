"""ROADMAP AB, dynamic: fly the best quasi-static feathering gaits.

The same random search as feather_probe.py (same rng, so the same gaits), the
top K gaits by thrust_margin kept, and each flown open loop through the real air
segment -- reset, scatter, task, rollout, `_score_segment` -- on three seeds.
Against: the plan's own heave-only gait through the same path.  A quasi-static
thrust that the rollout does not turn into flight is not evidence of flight.
"""
import json
import sys
from pathlib import Path

import numpy as np

from dytiscidae.control.cpg import CPGParams
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.evaluate import _scatter_seed
from dytiscidae.envs.tasks import schedule_for, task_seed
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

N, K = 300, 5
SEEDS = (0, 1, 2)


def feathered(name):
    g = BODY_PLANS[name]()
    for part in g.parts:
        if part.joint == "hinge" and part.actuated and part.is_surface:
            part.joint = "universal"
    return g


def fly(genome, params, seed):
    env = TriphibianEnv(build(genome), seed=seed)
    env.reset(Domain.AIR)
    env.scatter(np.random.default_rng(_scatter_seed(seed, Domain.AIR)))
    env.task = schedule_for(Domain.AIR, np.random.default_rng(task_seed(_scatter_seed(seed, Domain.AIR))))
    seg = env.rollout(8.0, params=params if params is not None else env.cpg.base, domain=Domain.AIR)
    m = seg.measurements
    return {"competence": round(seg.competence, 4),
            "sink": round(float(m.get("measured_sink_rate", m.get("sink_rate", float("nan")))), 3),
            "airborne": round(float(m.get("airborne_fraction", m.get("airborne_seconds", 0.0))), 3),
            "height_hold": round(float(m.get("height_hold", float("nan"))), 3)}


def main():
    rng = np.random.default_rng(20260923)
    out = {}
    for name in ("gannet", "teal", "beetle", "bat"):
        g = feathered(name)
        env = TriphibianEnv(build(g), seed=0)
        lo, hi = env.cpg.lo, env.cpg.hi
        half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
        found = []
        for _ in range(N):
            n = env.cpg.n
            p = CPGParams(rng.uniform(0, 1, n) * half, rng.uniform(0, 2 * np.pi, n),
                          mid + rng.uniform(-0.3, 0.3, n) * half * 2, float(rng.uniform(1.5, 12)))
            env.cpg.base = p
            if hasattr(env.p, "_measured_thrust"):
                del env.p._measured_thrust
            tm = env.thrust_margin()
            if tm is not None:
                found.append((tm, p))
        if name not in ("gannet", "teal"):
            continue
        found.sort(key=lambda x: -x[0])
        base = [fly(BODY_PLANS[name](), None, s) for s in SEEDS]
        print(f"{name} heave-only base gait: {base}", flush=True)
        rows = []
        for tm, p in found[:K]:
            r = [fly(feathered(name), p, s) for s in SEEDS]
            rows.append({"thrust_margin": round(tm, 3), "hz": round(p.frequency, 2), "flights": r})
            print(f"{name} tm {tm:+.3f} @ {p.frequency:.2f} Hz: {r}", flush=True)
        out[name] = {"base": base, "top": rows}
    Path("runs/_logs/feather_fly.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
