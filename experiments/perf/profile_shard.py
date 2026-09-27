"""Profile one worker's shard: the batched Tier-1 evaluation of 4 search-like
designs with the shared policy attached, as `ActorPool` hands it to a worker.

    python experiments/perf/profile_shard.py [--n 4] [--identify 1] [--out prof.txt]
"""
import argparse
import cProfile
import pstats
import time

import numpy as np


def designs(n, seed):
    from dytiscidae.core.genome import random_genome
    from dytiscidae.core.phenotype import build
    from dytiscidae.core.bodyplans import BODY_PLANS
    rng = np.random.default_rng(seed)
    out = [build(BODY_PLANS[k]()) for k in ("beetle", "gannet", "teal", "ray")][:n // 2]
    while len(out) < n:
        try:
            out.append(build(random_genome(rng)))
        except Exception:
            pass
    return out


def flatten(results):
    out = {}
    for i, r in enumerate(results):
        for d, seg in r.segments.items():
            for k, v in list(vars(seg).items()) + list(seg.measurements.items()):
                if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                    out[f"{i}.{d}.{k}"] = float(v)
        for d, t in r.transitions.results.items():
            for k, v in vars(t).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                    out[f"{i}.T{d}.{k}"] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--identify", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--noprof", action="store_true")
    ap.add_argument("--dump", default=None, help="write every segment measurement here (json)")
    ap.add_argument("--compare", default=None, help="compare against a --dump file")
    a = ap.parse_args()
    import torch
    from dytiscidae.control.cpg import TWIST_DIM
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.learning.ppo import SharedPolicy
    torch.manual_seed(0)
    torch.set_num_threads(1)
    net = SharedPolicy(TriphibianEnv.OBS_DIM, TWIST_DIM + 1)
    net.eval()
    phenos = designs(a.n, a.seed)
    kw = dict(spec=MissionSpec(), segment_seconds=a.seconds, seed=5,
              identify_axes=bool(a.identify), shared=net)
    batchroll.evaluate_tier1_batch(phenos[:1], segment_seconds=0.5, spec=MissionSpec(),
                                   seed=5, identify_axes=False)      # warm the kernels
    if a.noprof:
        t, c = time.perf_counter(), time.process_time()
        res = batchroll.evaluate_tier1_batch(phenos, **kw)
        print(f"wall {time.perf_counter() - t:.2f} s, cpu {time.process_time() - c:.2f} s "
              f"for {a.n} designs, identify={a.identify}, no profiler")
        flat = flatten(res)
        if a.dump:
            import json
            json.dump(flat, open(a.dump, "w"))
        if a.compare:
            import json
            ref = json.load(open(a.compare))
            keys = sorted(set(ref) & set(flat))
            diffs = sorted(((abs(ref[k] - flat[k]), k) for k in keys), reverse=True)
            print(f"compared {len(keys)} numbers ({len(set(ref) ^ set(flat))} keys differ); "
                  f"worst {diffs[0][0]:.3g} at {diffs[0][1]}; "
                  f"{sum(d > 1e-9 for d, _ in diffs)} above 1e-9")
        return
    pr = cProfile.Profile()
    t = time.perf_counter()
    pr.enable()
    batchroll.evaluate_tier1_batch(phenos, **kw)
    pr.disable()
    wall = time.perf_counter() - t
    print(f"wall {wall:.1f} s for {a.n} designs, identify={a.identify}")
    st = pstats.Stats(pr)
    if a.out:
        st.dump_stats(a.out)
    st.sort_stats("tottime").print_stats(35)
    st.sort_stats("cumulative").print_stats(45)


if __name__ == "__main__":
    main()
