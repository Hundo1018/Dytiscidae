"""The whole of fluid.apply on the GPU, against FluidSolver itself.

The reference is the real solver running the real MuJoCo state, not a
re-implementation, so a shared misreading cannot pass. What is compared is what
mj_step actually consumes: data.xfrc_applied, and model.body_mass.

It drove `FullPipeline` through a hand-written copy of its descriptor, and that
copy went stale when the layout grew on 2026-09-23; it goes through
`batchroll.BatchedFluid` now, so there is one packing and this tests it.

Run:  PYTHONPATH=. python mojo/tests/test_full_gpu.py
"""
import numpy as np
import mujoco

from dytiscidae.envs.batchroll import BatchedFluid

from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv
from dytiscidae.physics.medium import GRAVITY


def pose(env, seed):
    rng = np.random.default_rng(seed)
    env.reset(Domain.WATER, randomise=False)
    nq = env.model.nq
    if nq >= 7:
        q = rng.normal(size=4)
        env.data.qpos[3:7] = q / np.linalg.norm(q)
        env.data.qpos[0:3] = rng.normal(scale=1.5, size=3)
        env.data.qpos[2] = rng.uniform(-3.0, 1.0)      # straddle the surface
    if nq > 7:
        env.data.qpos[7:] = rng.uniform(-0.7, 0.7, nq - 7)
    nv = env.model.nv
    env.data.qvel[:] = rng.normal(scale=2.0, size=nv)
    mujoco.mj_forward(env.model, env.data)
    return env


def main():
    # Two identical copies of every machine: one stepped by the batched
    # pipeline exactly as a search step runs it, one by FluidSolver.
    batched = [pose(TriphibianEnv(build(p())), hash(k) % 9973)
               for k, p in BODY_PLANS.items()]
    single = [pose(TriphibianEnv(build(p())), hash(k) % 9973)
              for k, p in BODY_PLANS.items()]
    b = BatchedFluid(batched)
    print(f"{len(batched)} machines, {b.n} panels, {b.nb} bodies")

    t = 0.0
    b.apply(t)

    worst_f, worst_m = 0.0, 0.0
    for eb, e in zip(batched, single):
        e.data.xfrc_applied[:] = 0.0
        e.solver.apply(e.data, t)
        ref_f, got_f = e.data.xfrc_applied, eb.data.xfrc_applied
        sc = max(float(np.abs(ref_f).max()), 1e-12)
        worst_f = max(worst_f, float(np.abs(ref_f - got_f).max()) / sc)
        ref_m, got_m = e.model.body_mass, eb.model.body_mass
        sm = max(float(np.abs(ref_m - e.solver._dry_mass).max()), 1e-12)
        worst_m = max(worst_m, float(np.abs(ref_m - got_m).max()) / sm)

    print(f"  xfrc_applied  max relative error {worst_f:.3e}")
    print(f"  body added mass                  {worst_m:.3e}")
    ok = worst_f < 1e-9 and worst_m < 1e-12
    print()
    print("full-pipeline GPU checks passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
