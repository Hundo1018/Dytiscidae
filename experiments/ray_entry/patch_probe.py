"""ROADMAP AK hypothesis test: MuJoCo keeps qvel when the added mass in the mass
matrix grows, so every step of entrainment *creates* momentum (dm * v) instead
of the fluid decelerating the body.  Patch in the missing -max(dm, 0)/dt * v at
each body's centre of mass (and the matching torque), then re-run the entries.
"""
import sys

import mujoco
import numpy as np

import dytiscidae.physics.fluid as fl

_apply = fl.FluidSolver.apply


def patched(self, data, t):
    m = self.model
    m_old = m.body_mass.copy()
    i_old = m.body_inertia[:, 0].copy()
    d = _apply(self, data, t)
    dt = m.opt.timestep
    dm = np.maximum(m.body_mass - m_old, 0.0)
    di = np.maximum(m.body_inertia[:, 0] - i_old, 0.0)
    v6 = np.zeros(6)
    for b in self._bodies:
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, int(b), v6, 0)
        data.xfrc_applied[b, :3] -= dm[b] / dt * v6[3:]
        data.xfrc_applied[b, 3:] -= di[b] / dt * v6[:3]
    return d


if "--off" not in sys.argv:
    fl.FluidSolver.apply = patched
    print("PATCHED: momentum-conserving entrainment")
sys.argv = [a for a in sys.argv if a != "--off"]
sys.path.insert(0, "experiments/ray_entry")
import runpy  # noqa: E402

runpy.run_path(sys.argv.pop(1), run_name="__main__")
