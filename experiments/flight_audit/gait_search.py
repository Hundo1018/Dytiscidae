"""Best thrust_margin over random gaits per seed plan (quasi-static, at trim).

Re-runs ROADMAP's "best of 400" (gannet +0.85, teal +0.46, beetle +0.17,
bat +0.12) on whatever tree it is run in.  Gaits: amplitude U(0, 1) of each
joint's half-range, phase U(0, 2pi), offset U(-0.3, 0.3) of range about centre,
frequency U(1.5, 12) Hz.
"""
import sys
import numpy as np
from dytiscidae.control.cpg import CPGParams
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import TriphibianEnv

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(20260923)
for name in ("gannet", "teal", "beetle", "bat"):
    env = TriphibianEnv(build(BODY_PLANS[name]()), seed=0)
    lo, hi = env.cpg.lo, env.cpg.hi
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    best, at = -np.inf, None
    vals = []
    for _ in range(N):
        n = env.cpg.n
        p = CPGParams(rng.uniform(0, 1, n) * half, rng.uniform(0, 2 * np.pi, n),
                      mid + rng.uniform(-0.3, 0.3, n) * half * 2, float(rng.uniform(1.5, 12)))
        env.cpg.base = p
        if hasattr(env.p, "_measured_thrust"):
            del env.p._measured_thrust   # cached per phenotype otherwise
        tm = env.thrust_margin()
        if tm is None:
            continue
        vals.append(tm)
        if tm > best:
            best, at = tm, p.frequency
    v = np.array(vals)
    print(f"{name:7s} n={len(v)} best {best:+.3f} at {at:.2f} Hz  p90 {np.percentile(v,90):+.3f}  share>0 {np.mean(v>0):.2f}", flush=True)
