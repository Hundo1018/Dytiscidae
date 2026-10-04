"""ROADMAP AK side effect: `test_physics::test_resonance_needs_a_soft_drive`'s
"a spring under the old hard-wired gain buys little", with and without the
entrainment reaction.  Prints power per radian of motion, and where the ray is.
`python experiments/ray_entry/spring.py`.
"""
import numpy as np

from dytiscidae.core.bodyplans import ray
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv


def probe(stiffness, compliance, react, secs=5.0):
    g = ray()
    for part in g.parts:
        if part.joint != "none" and part.actuated:
            part.series_stiffness = stiffness
            part.drive_compliance = compliance
    env = TriphibianEnv(build(g))
    env.solver.entrainment = react
    env.reset(Domain.AIR, randomise=False)
    q, sub, z = [], [], []
    for _ in range(int(secs / env.timestep)):
        env.step(env.cpg.command(env.cpg.base, env.data.time))
        q.append(env.data.qpos[7:].copy())
        sub.append(env.solver.diag.max_submerged)
        z.append(float(env.data.qpos[2]))
    h = len(q) // 2
    q = np.array(q)[h:]
    swing = float(np.mean(q.max(axis=0) - q.min(axis=0)))
    pw = env.budget.mean_power
    return pw / max(swing, 1e-6), pw, swing, float(np.mean(sub[h:])), min(z), z[-1]


for react in (False, True):
    r = probe(0.0, 1.0, react)
    s = probe(1.0, 1.0, react)
    print(f"reaction {'on ' if react else 'off'}: rigid {r[0]:.0f} W/rad ({r[1]:.0f} W, "
          f"{r[2]:.2f} rad)  stiff spring {s[0]:.0f} W/rad ({s[1]:.0f} W, {s[2]:.2f} rad)"
          f"  ratio {s[0] / r[0]:.2f}  | 2nd-half submerged {r[3]:.2f}, z min {r[4]:+.2f} "
          f"end {r[5]:+.2f}")
