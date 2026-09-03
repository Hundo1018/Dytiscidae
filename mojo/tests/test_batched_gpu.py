"""A batch of machines must give exactly what the same machines give alone.

This is the claim the whole port rests on: panels from different morphologies
concatenate into one launch. If batching changed any number, the speedup would
be worthless.
"""
import sys
import time

import numpy as np
import mujoco

from batched import BatchedPanels
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv


def pose(env, seed):
    rng = np.random.default_rng(seed)
    env.reset(Domain.WATER, randomise=False)
    nq = env.model.nq
    if nq >= 7:
        q = rng.normal(size=4)
        env.data.qpos[3:7] = q / np.linalg.norm(q)
        env.data.qpos[0:3] = rng.normal(scale=2.0, size=3)
    if nq > 7:
        env.data.qpos[7:] = rng.uniform(-0.7, 0.7, nq - 7)
    mujoco.mj_forward(env.model, env.data)
    return env


def flow_for(n, seed):
    rng = np.random.default_rng(seed)
    C = np.ascontiguousarray
    v = C(rng.normal(scale=4.0, size=(n, 3)))
    om = C(rng.normal(scale=6.0, size=(n, 3)))
    rho = C(np.where(rng.random(n) < 0.5, 1.225, 1025.0))
    mu = C(np.where(rho > 500, 1.08e-3, 1.81e-5))
    return v, om, rho, mu


def build_group(reps):
    envs, seeds = [], []
    for r in range(reps):
        for k, plan in BODY_PLANS.items():
            envs.append(pose(TriphibianEnv(build(plan())), seed=hash((k, r)) % 99991))
            seeds.append(hash((k, r)) % 99991)
    return envs, seeds


def main():
    envs, _ = build_group(2)          # 14 machines, 7 morphologies twice
    batch = BatchedPanels(envs)
    n = batch.total_panels
    print(f"{len(envs)} machines, {len(BODY_PLANS)} distinct morphologies, "
          f"{n} panels total")

    # One flow field over the whole batch; each machine's slice of it is what
    # that machine would have seen alone.
    v, om, rho, mu = flow_for(n, 5)
    batch.run(v, om, rho, mu)
    got = {k: getattr(batch, k).copy() for k in
           ("pos_w", "s_hat", "alpha", "cl", "cd", "F_bluff", "m_add", "m_body")}

    # Now each machine on its own, and compare slice by slice.
    worst = {k: 0.0 for k in got}
    worst_mb = 0.0
    for i, e in enumerate(envs):
        solo = BatchedPanels([e])
        sl = batch.slice_of(i)
        solo.run(np.ascontiguousarray(v[sl]), np.ascontiguousarray(om[sl]),
                 np.ascontiguousarray(rho[sl]), np.ascontiguousarray(mu[sl]))
        for k in got:
            if k == "m_body":
                a = got[k][batch.body_off[i]:batch.body_off[i + 1]]
                b = solo.m_body
            else:
                a, b = got[k][sl], getattr(solo, k)
            sc = max(float(np.abs(a).max()), 1e-12)
            worst[k] = max(worst[k], float(np.abs(a - b).max()) / sc)

    print()
    for k, e in worst.items():
        print(f"  {k:10s} max relative difference batched vs alone: {e:.3e}")
    ok = max(worst.values()) < 1e-13

    # And the point of doing it: throughput.
    print()
    for reps in (1, 4, 16):
        group, _ = build_group(reps)
        bp = BatchedPanels(group)
        vv, oo, rr, mm = flow_for(bp.total_panels, 9)
        bp.run(vv, oo, rr, mm)                      # warm
        t0 = time.perf_counter()
        for _ in range(20):
            bp.run(vv, oo, rr, mm)
        batched_us = (time.perf_counter() - t0) / 20 * 1e6

        solos = [BatchedPanels([e]) for e in group]
        for i, s in enumerate(solos):
            sl = bp.slice_of(i)
            s.run(np.ascontiguousarray(vv[sl]), np.ascontiguousarray(oo[sl]),
                  np.ascontiguousarray(rr[sl]), np.ascontiguousarray(mm[sl]))
        t0 = time.perf_counter()
        for _ in range(20):
            for i, s in enumerate(solos):
                sl = bp.slice_of(i)
                s.run(np.ascontiguousarray(vv[sl]), np.ascontiguousarray(oo[sl]),
                      np.ascontiguousarray(rr[sl]), np.ascontiguousarray(mm[sl]))
        solo_us = (time.perf_counter() - t0) / 20 * 1e6
        print(f"  {len(group):4d} machines ({bp.total_panels:6d} panels): "
              f"one at a time {solo_us:9.1f} us, batched {batched_us:8.1f} us"
              f"  -> {solo_us/batched_us:5.2f}x")

    print()
    print("batched GPU checks passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
