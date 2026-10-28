# CPU. Host half of the settled-body contact measurement (IMPL_1010 A4b): a beetle at rest on the beach
# for 2 s of sim, under the four host configurations {geom margin 0.001 kept, margin 0} x {MULTICCD and
# NATIVECCD on, off}. Reports the settled height, contacts, and the vertical contact force sum.
# The device half (MJWarp on the same configurations) needs the GPU and is not run here.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

SECONDS = 2.0
CCD = mujoco.mjtDisableBit.mjDSBL_MULTICCD | mujoco.mjtDisableBit.mjDSBL_NATIVECCD


def settle(margin: float, ccd_on: bool) -> dict:
    env = TriphibianEnv(build(BODY_PLANS["beetle"]()), seed=0)
    env.reset(Domain.LAND, randomise=False)
    m, d = env.model, env.data
    m.geom_margin[:] = margin
    if ccd_on:
        m.opt.disableflags &= ~int(CCD)
    else:
        m.opt.disableflags |= int(CCD)
    d.ctrl[:] = 0.0
    steps = int(round(SECONDS / m.opt.timestep))
    ncon_max = 0
    for _ in range(steps):
        mujoco.mj_step(m, d)
        ncon_max = max(ncon_max, d.ncon)
    force = np.zeros(6)
    z_force = 0.0
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, force)
        z_force += force[0]
    return {"z": float(d.qpos[2]), "ncon": int(d.ncon), "ncon_max": ncon_max,
            "normal_force_sum": round(float(z_force), 4), "steps": steps}


if __name__ == "__main__":
    print("margin  ccd   z(m)     ncon  ncon_max  normal_force_sum(N)  steps", flush=True)
    for margin in (0.001, 0.0):
        for ccd_on in (True, False):
            r = settle(margin, ccd_on)
            print(f"{margin:<7} {'on ' if ccd_on else 'off'}  {r['z']:.4f}  {r['ncon']:>4}  "
                  f"{r['ncon_max']:>8}  {r['normal_force_sum']:>18}  {r['steps']}", flush=True)
