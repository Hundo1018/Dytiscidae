"""Wall time of one generation-shaped batched evaluation (16 designs, axis
identification on, shared policy attached) through the real ActorPool, at
several pool shapes.

    python experiments/perf/pool_sweep.py --shapes 4x4 8x2 6x3
"""
import argparse
import sys
import time

sys.path.insert(0, "experiments/perf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["4x4", "8x2"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=8.0)
    a = ap.parse_args()
    import torch
    from dytiscidae.control.cpg import TWIST_DIM
    from dytiscidae.envs.actors import ActorPool
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.learning.ppo import SharedPolicy
    from profile_shard import designs
    torch.manual_seed(0)
    net = SharedPolicy(TriphibianEnv.OBS_DIM, TWIST_DIM + 1)
    net.eval()
    phenos = designs(a.n, 3)
    for shape in a.shapes:
        w, m = (int(x) for x in shape.split("x"))
        pool = ActorPool(w, min_shard=m)
        try:
            pool.evaluate_tier1(phenos[:w], spec=MissionSpec(), segment_seconds=0.5,
                                seed=5, identify_axes=False)            # warm workers
            t = time.perf_counter()
            pool.evaluate_tier1(phenos, spec=MissionSpec(), segment_seconds=a.seconds,
                                seed=5, identify_axes=True, shared=net)
            print(f"{shape}: {time.perf_counter() - t:.1f} s for {a.n} designs", flush=True)
        finally:
            pool.close()


if __name__ == "__main__":
    main()
