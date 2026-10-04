"""Does one batch of 2k machines score exactly what two batches of k did?

    python experiments/perf/merge_check.py [--n 16] [--workers 4] [--min-shard 4]

The premise of ``loop.MERGE_FIRST_REFINE``, checked on the real (batched, GPU)
evaluation path: the same designs and controllers are scored as the search
used to (re-score call, then a trial call) and as it now does (one call with
both), through an actor pool shaped like the run's, with a shared policy on its
mean.  Every segment and transition number is compared to the last bit, and
the wall time of each way is printed.

Needs the GPU extension; refuses to run without it.
"""
import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-shard", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    import torch
    from profile_shard import designs, flatten

    from dytiscidae.control.cpg import TWIST_DIM, Policy
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.actors import ActorPool
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.learning.ppo import SharedPolicy

    ok, why = batchroll.usable()
    if not ok:
        raise SystemExit(f"needs the batched evaluator: {why}")

    torch.manual_seed(0)
    net = SharedPolicy(TriphibianEnv.OBS_DIM, TWIST_DIM + 1)
    net.eval()
    phenos = designs(a.n, a.seed)
    rng = np.random.default_rng(a.seed)

    def ctrl(scale):
        p = Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=6, gain=True)
        p.weights = rng.normal(0.0, scale, size=p.n_weights)
        return Controller(params=None, policy=p)

    ctrls = [ctrl(0.1) for _ in phenos]
    trials = [ctrl(0.1) for _ in phenos]
    kw = dict(spec=MissionSpec(), segment_seconds=a.seconds, identify_axes=False,
              seed=5, shared=net)

    pool = ActorPool(workers=a.workers, min_shard=a.min_shard)
    try:
        pool.evaluate_tier1(phenos[:1], segment_seconds=0.5, spec=MissionSpec(), seed=5)
        t = time.perf_counter()
        r1 = pool.evaluate_tier1(phenos, controllers=ctrls, **kw)
        r2 = pool.evaluate_tier1(phenos, controllers=trials, **kw)
        split_s = time.perf_counter() - t
        t = time.perf_counter()
        both = pool.evaluate_tier1(phenos + phenos, controllers=ctrls + trials, **kw)
        merged_s = time.perf_counter() - t
    finally:
        pool.close()

    k = len(phenos)
    a_split, a_merged = flatten(r1 + r2), flatten(both[:k] + both[k:])
    keys = sorted(set(a_split) | set(a_merged))
    differ = [key for key in keys if a_split.get(key) != a_merged.get(key)]
    print(f"{len(keys)} numbers, {len(differ)} differ")
    for key in differ[:20]:
        print(f"  {key}: split {a_split.get(key)!r}  merged {a_merged.get(key)!r}")
    print(f"wall: split {split_s:.1f} s, merged {merged_s:.1f} s "
          f"({split_s / max(merged_s, 1e-9):.2f}x)")
    raise SystemExit(1 if differ else 0)


if __name__ == "__main__":
    main()
