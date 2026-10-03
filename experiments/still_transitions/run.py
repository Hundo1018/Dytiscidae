"""Transition reward and graded approach with actuators held still vs the base gait.

ARCH46_SPEC §8. Run before and after the 2026-10-03 fix:
    PYTHONPATH=. .venv/bin/python experiments/still_transitions/run.py
"""
import numpy as np
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.control.cpg import CPGParams
from dytiscidae.envs.evaluate import Controller
from dytiscidae.envs.transitions import run_transition
from dytiscidae.envs.triphibian import TriphibianEnv

KINDS = ("air_to_water", "water_to_air", "water_to_land", "land_to_air")


def reward(t):
    """The PPO terminal reward for a crossing (batchroll)."""
    return float(t.crossed) * (0.40 + 0.60 * float(np.mean(
        [t.components.get(c, 0.0) for c in ("shock", "control", "settle", "economy", "exit_state")])))


rows = {}
for plan in BODY_PLANS:
    p = build(BODY_PLANS[plan]())
    for seed in (0, 1):
        e = TriphibianEnv(p, seed=seed)
        b = e.cpg.base
        still = CPGParams(amplitude=np.zeros(e.cpg.n), phase=np.asarray(b.phase, float),
                          offset=np.asarray(b.offset, float), frequency=float(b.frequency))
        for kind in KINDS:
            for name, prm in (("gait", b), ("still", still)):
                t = run_transition(e, kind, Controller(params=prm))
                rows.setdefault((kind, name), []).append(
                    (reward(t), float(t.crossed), t.components["crossed"], t.hold))
for kind in KINDS:
    for name in ("gait", "still"):
        v = np.array(rows[(kind, name)])
        print(f"{kind:14s} {name:5s} n={len(v):2d} reward {v[:,0].mean():.3f}  crossed {v[:,1].mean():.2f}"
              f"  graded approach mean {v[:,2].mean():.3f} max {v[:,2].max():.3f}  hold {v[:,3].mean():.2f}")
