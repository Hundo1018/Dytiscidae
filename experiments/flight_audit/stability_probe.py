"""Largest joint angle over 5 s per seed plan: tensor on/off x implicit damping on/off.

Same protocol as experiments/wing_added_mass section D (reset AIR, not
randomised, driven by the base gait or held still).
"""
import sys
import numpy as np
from dytiscidae.core import bodyplans
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

def travel(plan, dt, drive, tensor, implicit, seconds=5.0, domain=Domain.AIR):
    env = TriphibianEnv(build(getattr(bodyplans, plan)()))
    env.solver.wing_added_mass_tensor = tensor
    env.solver.implicit_damping = implicit
    env.model.opt.timestep = dt
    env.reset(domain, randomise=False)
    q = []
    for _ in range(int(seconds / dt)):
        u = env.cpg.command(env.cpg.base, env.data.time) if drive else np.zeros(env.model.nu)
        env.step(u)
        q.append(env.data.qpos[7:].copy())
    q = np.array(q)
    return float(np.nanmax(np.abs(q))) if q.size else 0.0

if __name__ == "__main__":
    plans = sys.argv[1].split(",") if len(sys.argv) > 1 else ["bat", "beetle", "eel", "gannet", "medusa", "ray", "teal"]
    dom = Domain[sys.argv[2]] if len(sys.argv) > 2 else Domain.AIR
    print(f"{'plan':8s} {'tensor,expl drv':>16} {'tensor,impl drv':>16} {'tensor,impl still':>18} {'scalar,impl drv':>16} {'scalar,expl drv':>16}  ({dom.value})")
    for pl in plans:
        r = [travel(pl, .004, True, True, False, domain=dom), travel(pl, .004, True, True, True, domain=dom),
             travel(pl, .004, False, True, True, domain=dom), travel(pl, .004, True, False, True, domain=dom),
             travel(pl, .004, True, False, False, domain=dom)]
        print(f"{pl:8s} " + " ".join(f"{v:16.2f}" for v in r), flush=True)
