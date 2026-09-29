"""ROADMAP AK: what throws a held membrane joint from 2 to 12 rad/s in one step
at first contact.  Splits each hinge's generalized force into its sources on
the steps around the jump.  `python experiments/ray_entry/whack.py PITCH SPEED`.
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
d.qpos[:3] = (-8.0, 0.0, 1.2 + float(next((a[3:] for a in sys.argv if a.startswith("dz=")), 0.0)))
sol.entrainment = "noreact" not in sys.argv
HOLD = "hold" in sys.argv
a = math.radians(-pitch)
d.qpos[3:7] = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
d.qvel[:] = 0.0
d.qvel[2] = -speed
sol.reset()
mujoco.mj_forward(m, d)
prev = None
for i in range(int(0.5 / env.timestep)):
    u = env.cpg.command(env.cpg.base, d.time)
    qv0 = d.qvel[6:].copy()
    env.step(np.zeros_like(u) if HOLD else u)
    jump = np.abs(d.qvel[6:] - qv0)
    j = int(jump.argmax())
    if jump[j] > 3.0:
        k = 6 + j
        # d.qfrc_* hold the values mj_step used for this step.
        xf = d.qfrc_smooth - (d.qfrc_passive - d.qfrc_bias + d.qfrc_applied
                              + d.qfrc_actuator)
        print(f"t={d.time:.3f} dof {k} ({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[k])})"
              f" dqd={d.qvel[k] - qv0[j]:+.1f} rad/s  M_kk={m.body_mass[m.dof_bodyid[k]]:.3f}kg  "
              f"passive {d.qfrc_passive[k]:+.1f} applied {d.qfrc_applied[k]:+.1f} "
              f"bias {d.qfrc_bias[k]:+.1f} act {d.qfrc_actuator[k]:+.1f} "
              f"xfrc {xf[k]:+.1f} constraint {d.qfrc_constraint[k]:+.1f} "
              f"damp {m.dof_damping[k]:.1f} added_mass {sol.diag.added_mass:.1f}")
        if d.time > 0.27:
            break
