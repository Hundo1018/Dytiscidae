"""ROADMAP AK: after the nose-first entry the ray *accelerates* down.

`python experiments/ray_entry/plunge.py PITCH SPEED [hold]` prints the machine's
centre-of-mass velocity, the fluid force on it, the added mass and the
entrainment-rate term every 8 ms.  `hold` holds the joints at their base
command instead of driving them (the CPG's base is the same command).
"""
import math
import sys

import mujoco
import numpy as np

from dytiscidae.core.bodyplans import ray
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

pitch, speed = (float(a) for a in sys.argv[1:3])
p = build(ray())
env = TriphibianEnv(p)
m, d, sol = env.model, env.data, env.solver
env.reset(Domain.AIR, randomise=False)
d.qpos[:3] = (-8.0, 0.0, 1.2)
a = math.radians(-pitch)
d.qpos[3:7] = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
d.qvel[:] = 0.0
d.qvel[2] = -speed
sol.reset()
mujoco.mj_forward(m, d)
dry = float(sol._dry_mass.sum())
print(f"dry mass {dry:.2f} kg, bodies {m.nbody}, dofs {m.nv}")
print("   t     z   vcom_z  |v|max   Fz_fluid  m_add  clamped")
for i in range(int(0.6 / env.timestep)):
    env.step(env.cpg.command(env.cpg.base, d.time))
    if i % 2:
        continue
    mass = m.body_mass[1:]
    vcom = np.zeros(3)
    for b in range(1, m.nbody):
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
        vcom += sol._dry_mass[b] * v6[3:] if b < len(sol._dry_mass) else 0
    vcom /= dry
    fz = float(d.xfrc_applied[:, 2].sum())
    print(f"{d.time:.3f} {d.qpos[2]:+.2f} {vcom[2]:+7.2f} {np.abs(d.qvel).max():7.1f} "
          f"{fz:9.1f} {sol.diag.added_mass:6.1f} {sol.diag.clamped}")
