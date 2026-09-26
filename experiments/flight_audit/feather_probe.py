"""ROADMAP AB probe: does a wing that can feather make the thrust level flight needs?

The seed plans' wing parts are switched to a "universal" joint (stroke plus a
feathering hinge about the span, each with its own motor), and the same random
gait search as gait_search.py runs over both channels -- so the pitch-heave phase
is free.  Pre-registered: if the best thrust_margin does not clear 1.0 on any
plan, the ceiling is elsewhere and feathering alone does not fix flight.
"""
import copy
import sys
import numpy as np
from dytiscidae.control.cpg import CPGParams
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import TriphibianEnv

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = np.random.default_rng(20260923)
for name in ("gannet", "teal", "beetle", "bat"):
    g = BODY_PLANS[name]()
    for part in g.parts:
        if part.joint == "hinge" and part.actuated and part.is_surface:
            part.joint = "universal"
    env = TriphibianEnv(build(g), seed=0)
    nf = sum(a.endswith("_f") for a in env.act_names)
    lo, hi = env.cpg.lo, env.cpg.hi
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    vals, best, at = [], -np.inf, None
    for _ in range(N):
        n = env.cpg.n
        p = CPGParams(rng.uniform(0, 1, n) * half, rng.uniform(0, 2 * np.pi, n),
                      mid + rng.uniform(-0.3, 0.3, n) * half * 2, float(rng.uniform(1.5, 12)))
        env.cpg.base = p
        if hasattr(env.p, "_measured_thrust"):
            del env.p._measured_thrust
        tm = env.thrust_margin()
        if tm is None:
            continue
        vals.append(tm)
        if tm > best:
            best, at = tm, p.frequency
    v = np.array(vals) if vals else np.zeros(1)
    print(f"{name:7s} feather channels {nf}  n={len(vals)} best {best:+.3f} at {at if at else 0:.2f} Hz  "
          f"p90 {np.percentile(v, 90):+.3f}  lift_margin {env.lift_margin:.2f}", flush=True)
