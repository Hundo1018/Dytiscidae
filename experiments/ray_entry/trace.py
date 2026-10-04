"""ROADMAP AK: which strip carries the ray's slam peak, and why.

`python experiments/ray_entry/trace.py PITCH SPEED` prints, for the steps whose
slam is largest, the strip that set it: kind, body, blended density, the change
in its entrained mass, its normal speed and its submerged fraction.
"""
import math
import sys

import mujoco
import numpy as np

import dytiscidae.physics.fluid as fl
from dytiscidae.core.bodyplans import ray
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

pitch, speed = (float(a) for a in sys.argv[1:3])
p = build(ray())
env = TriphibianEnv(p)
sol = env.solver
P = sol.panels
cap = {}
_orig = fl.slam_mass


def spy(m_add, rho, chord, dr, is_wing, scale=1.0):
    cap["rho"] = np.array(rho, float, copy=True)
    cap["m_add"] = np.array(m_add, float, copy=True)
    return _orig(m_add, rho, chord, dr, is_wing, scale)


fl.slam_mass = spy
env.reset(Domain.AIR, randomise=False)
env.data.qpos[:3] = (-8.0, 0.0, 1.2 + float(next((a[3:] for a in sys.argv if a.startswith("dz=")), 0.0)))
a = math.radians(-pitch)
env.data.qpos[3:7] = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
env.data.qvel[:] = 0.0
env.data.qvel[2] = -speed
sol.reset()
mujoco.mj_forward(env.model, env.data)
rows, prev = [], None
for i in range(int(1.2 / env.timestep)):
    u = env.cpg.command(env.cpg.base, env.data.time)
    env.step(np.zeros_like(u) if "hold" in sys.argv else u)
    ms, vn = sol._prev_ma.copy(), sol._prev_vn.copy()
    if prev is not None:
        per = np.abs((ms - prev) / env.timestep * vn)
        k = int(per.argmax())
        rows.append((env.data.time, per[k], k, cap["rho"][k], ms[k] - prev[k], vn[k],
                     float(env.data.qpos[2]), float(sol.diag.slam),
                     float(np.linalg.norm(sol._inflow.w)),
                     float(np.abs(env.data.qvel[6:]).max()),
                     float(np.degrees(np.arcsin(np.clip(-env.data.xmat[1].reshape(3, 3)[2, 0], -1, 1))))))
    prev = ms
N = len(rows)
print(f"panels {P.n}  wing {int((P.kind == fl.WING).sum())}  dt {env.timestep}")
top = sorted(rows, key=lambda r: -r[1])[:12]
for t, s, k, rho, dm, v, z, d, w, qd, pit in sorted(top):
    body = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, int(P.body_id[k]))
    print(f"t={t:.3f} slam {s:6.0f} N k={k:3d} "
          f"{'WING ' if P.kind[k] == fl.WING else 'BLUFF'} {body:>14s} "
          f"rho={rho:6.0f} dm={dm:+.2f} vn={v:+6.2f} z={z:+.2f} |w|={w:5.2f} max|qd|={qd:5.1f} nose-down {pit:+5.1f}")

print("time series (every 4th step, t 0.10-0.40): t |w| max|qd| z slam nose-down-deg")
for r in rows:
    if (0.10 <= r[0] <= 0.40 and int(round(r[0] / env.timestep)) % 4 == 0) or (
            "dense" in sys.argv and 0.225 <= r[0] <= 0.27):
        print(f"  {r[0]:.3f} {r[8]:6.2f} {r[9]:6.1f} {r[6]:+.2f} {r[7]:7.0f} {r[10]:+6.1f}")
