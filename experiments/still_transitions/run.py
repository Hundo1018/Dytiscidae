"""Transition PPO reward with actuators held still vs the body's base gait."""
import numpy as np
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.control.cpg import CPGParams
from dytiscidae.envs.evaluate import Controller
from dytiscidae.envs.transitions import run_transition
from dytiscidae.envs.triphibian import TriphibianEnv

def reward(t):
    return float(t.crossed) * (0.40 + 0.60 * float(np.mean([t.components.get(c, 0.0) for c in ("shock","control","settle","economy","exit_state")])))

rows = {}
for plan in BODY_PLANS:
    try:
        p = build(BODY_PLANS[plan]())
    except Exception as exc:
        print(plan, "build failed", exc); continue
    for seed in (0, 1):
        e = TriphibianEnv(p, seed=seed)
        b = e.cpg.base
        still = CPGParams(amplitude=np.zeros(e.cpg.n), phase=np.asarray(b.phase, float),
                          offset=np.asarray(b.offset, float), frequency=float(b.frequency))
        for kind in ("air_to_water", "water_to_air", "water_to_land", "land_to_air"):
            for name, prm in (("gait", b), ("still", still)):
                try:
                    t = run_transition(e, kind, Controller(params=prm))
                    rows.setdefault((kind, name), []).append((reward(t), float(t.crossed)))
                except Exception as exc:
                    print(plan, kind, name, "err", type(exc).__name__, exc)
for kind in ("air_to_water", "water_to_air", "water_to_land", "land_to_air"):
    for name in ("gait", "still"):
        v = np.array(rows.get((kind, name), [(np.nan, np.nan)]))
        print(f"{kind:14s} {name:5s} n={len(v):2d} reward mean {np.nanmean(v[:,0]):.3f}  crossed {np.nanmean(v[:,1]):.2f}")
