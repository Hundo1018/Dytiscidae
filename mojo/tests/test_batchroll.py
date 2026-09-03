"""Stepping N machines together must give what stepping each alone gives.

Not one call -- a whole trajectory. Errors in a simulator compound, so agreeing
on step 1 says much less than agreeing on step 400.
"""
import numpy as np
import mujoco

from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.batchroll import AVAILABLE, UNAVAILABLE_REASON, BatchedFluid, step_batch
from dytiscidae.envs.evaluate import Controller
from dytiscidae.envs.triphibian import Domain, TriphibianEnv


def make(seed_key, domain=Domain.WATER):
    name, plan = seed_key
    env = TriphibianEnv(build(plan()), seed=hash(name) % 9973)
    env.reset(domain, randomise=False)
    return env


def main():
    if not AVAILABLE:
        print(f"SKIP: GPU extension unavailable ({UNAVAILABLE_REASON})")
        return 0

    items = list(BODY_PLANS.items())
    STEPS = 400

    # Reference: each machine on its own, through the ordinary env.step path.
    solo_states = []
    for it in items:
        e = make(it)
        c = Controller(params=e.cpg.base)
        for _ in range(STEPS):
            if not e.step(e.cpg.command(c.params, e.data.time)):
                break
        solo_states.append((e.data.qpos.copy(), e.data.qvel.copy(),
                            float(e.data.time)))

    # Batched: same machines, same controllers, stepped together.
    envs = [make(it) for it in items]
    ctrls = [Controller(params=e.cpg.base) for e in envs]
    bf = BatchedFluid(envs)
    active = np.ones(len(envs), dtype=bool)
    for _ in range(STEPS):
        angles = [e.cpg.command(c.params, e.data.time)
                  for e, c in zip(envs, ctrls)]
        active = step_batch(envs, angles, bf, active)
        if not active.any():
            break

    print(f"{len(envs)} machines, {bf.n} panels, {STEPS} steps")
    print(f"{'plan':10s} {'qpos':>11s} {'qvel':>11s}")
    worst = 0.0
    for i, (it, (rq, rv, rt)) in enumerate(zip(items, solo_states)):
        gq, gv = envs[i].data.qpos, envs[i].data.qvel
        sq = max(float(np.abs(rq).max()), 1e-9)
        sv = max(float(np.abs(rv).max()), 1e-9)
        eq = float(np.abs(rq - gq).max()) / sq
        ev = float(np.abs(rv - gv).max()) / sv
        worst = max(worst, eq, ev)
        print(f"{it[0]:10s} {eq:11.3e} {ev:11.3e}")

    print()
    # Two separate reasons this is not exact, and the second is the one that
    # matters.
    #
    # First, the reference runs the numpy FluidSolver through env.step while
    # the batch runs the GPU kernels, and those differ in the last bit --
    # numpy's einsum does not sum left-to-right. Four hundred steps of a
    # chaotic system amplifies that.
    #
    # Second, and this is a genuine behavioural change: the GPU path is NOT
    # bit-reproducible run to run. The scatter uses Atomic.fetch_add, which
    # accumulates in whatever order the warps arrive, so identical seeds and
    # identical code give slightly different last bits each time. Measured over
    # five runs of this test: 6.0e-10, 6.7e-10, 6.9e-10, 1.2e-9, 2.4e-9.
    #
    # The bound therefore has to cover that spread rather than pin one run. A
    # transcription error would have diverged visibly by step 50, which is what
    # this is actually guarding against.
    ok = worst < 2e-8
    print("batched rollout matches solo" if ok else f"FAILED (worst {worst:.3e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
