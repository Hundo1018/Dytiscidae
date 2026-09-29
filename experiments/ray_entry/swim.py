"""ROADMAP AK: how much the missing entrainment reaction moves a machine that
starts in water.  Each seed plan is released at the water spawn and driven by
its CPG base for 6 s, with and without `entrainment_reaction`; prints distance
covered, peak hull speed and power.  `python experiments/ray_entry/swim.py`.
"""
import numpy as np

from dytiscidae.core import bodyplans
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

for name in ("ray", "gannet", "beetle", "eel"):
    if not hasattr(bodyplans, name):
        continue
    row = []
    for react in (False, True):
        env = TriphibianEnv(build(getattr(bodyplans, name)()))
        env.solver.entrainment = react
        env.reset(Domain.WATER, randomise=False)
        x0 = env.data.qpos[:3].copy()
        vmax = 0.0
        for _ in range(int(6.0 / env.timestep)):
            env.step(env.cpg.command(env.cpg.base, env.data.time))
            vmax = max(vmax, float(np.linalg.norm(env.data.qvel[:3])))
        d = float(np.linalg.norm(env.data.qpos[:3] - x0))
        row.append(f"{'on ' if react else 'off'}: moved {d:5.2f} m, peak {vmax:5.2f} m/s, "
                   f"{env.budget.mean_power:6.1f} W")
    print(f"{name:7s} reaction " + "  |  ".join(row))
