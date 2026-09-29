"""ROADMAP AK: the ray's water entry, flat against nose-first.

Reproduces `test_entry_shock_is_hydrodynamic_not_a_speed_limit` and prints,
per entry, the peak 10 ms-mean slam pressure, when and where it happened, and
which strip carried it.  `python experiments/ray_entry/probe.py`.
"""
import math
import sys

import mujoco
import numpy as np

from dytiscidae.core.bodyplans import ray
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

p = build(ray())
env = TriphibianEnv(p)
print(f"dt {env.timestep}  frontal {p.frontal_area:.3f} m^2  "
      f"capacity {p.slam_pressure_capacity / 1e3:.0f} kPa")
HOLD = "hold" in sys.argv
# `noreact`: the defect, for the before/after -- no `entrainment_reaction`.
env.solver.entrainment = "noreact" not in sys.argv
window = max(int(0.010 / env.timestep), 1)


def enter(pitch: float, speed: float, dz: float = 0.0):
    env.reset(Domain.AIR, randomise=False)
    env.data.qpos[:3] = (-8.0, 0.0, 1.2 + dz)
    a = math.radians(-pitch)
    env.data.qpos[3:7] = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
    env.data.qvel[:] = 0.0
    env.data.qvel[2] = -speed
    env.solver.reset()
    mujoco.mj_forward(env.model, env.data)
    w, peak, tpk, raw = [], 0.0, 0, []
    v_hit, v_after = None, 0.0
    for i in range(int(1.2 / env.timestep)):
        u = env.cpg.command(env.cpg.base, env.data.time)
        env.step(np.zeros_like(u) if HOLD else u)
        s = float(env.solver.diag.slam)
        if v_hit is None and env.solver.diag.max_submerged > 0.0:
            v_hit = -float(env.data.qvel[2])
        elif v_hit is not None:
            v_after = max(v_after, -float(env.data.qvel[2]))
        raw.append((env.data.time, s, float(env.data.qpos[2])))
        w = (w + [s])[-window:]
        if len(w) == window and np.mean(w) > peak:
            peak, tpk = float(np.mean(w)), i
    pr = peak / max(p.frontal_area, 1e-3)
    sc = float(np.clip(1 - (pr / p.slam_pressure_capacity) ** 2, 0, 1))
    print(f"pitch {pitch:4.0f} v {speed:5.1f}: peak {pr / 1e3:8.1f} kPa "
          f"at t={raw[tpk][0]:.3f} z={raw[tpk][2]:+.2f}  score {sc:.3f}  "
          f"sinking {v_hit:.1f} m/s at contact, fastest after {v_after:.1f}")
    return raw


if __name__ == "__main__":
    cases = [(0, 4), (0, 8), (80, 4), (80, 8), (80, 20)]
    args = [a for a in sys.argv[1:] if a not in ("hold", "spread", "noreact")]
    if args:
        cases = [tuple(map(float, a.split(","))) for a in args]
    print("joints held at zero" if HOLD else "joints driven by the CPG base",
          "| entrainment reacted" if env.solver.entrainment else "| NOT reacted")
    dzs = [0.0]
    if "spread" in sys.argv:
        # The noise floor: the same entry from 1-5 mm higher.
        dzs = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005]
    for pit, v in cases:
        for dz in dzs:
            enter(pit, v, dz)
