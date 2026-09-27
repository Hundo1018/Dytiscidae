"""Time FullPipeline.step inside the real benchmark, to see whether concurrent
processes contend for the GPU.  Run several copies at once and compare us/call.

    python experiments/perf/gpu_contention.py --n 4
"""
import argparse
import sys
import time

sys.path.insert(0, "experiments/perf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    import torch
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.triphibian import MissionSpec
    from profile_shard import designs
    torch.set_num_threads(1)
    acc = {"t": 0.0, "n": 0}
    real_init = batchroll.BatchedFluid.__init__

    class Timed:
        def __init__(self, pipe):
            self.pipe = pipe

        def step(self, *args):
            t = time.perf_counter()
            self.pipe.step(*args)
            acc["t"] += time.perf_counter() - t
            acc["n"] += 1

        def __getattr__(self, k):
            return getattr(self.pipe, k)

    def init(self, envs):
        real_init(self, envs)
        self.pipe = Timed(self.pipe)
    batchroll.BatchedFluid.__init__ = init
    phenos = designs(a.n, 3)
    t = time.perf_counter()
    batchroll.evaluate_tier1_batch(phenos, spec=MissionSpec(), segment_seconds=8.0,
                                   seed=5, identify_axes=False)
    wall = time.perf_counter() - t
    print(f"n={a.n} wall {wall:.1f} s, pipe.step {acc['t']:.1f} s over {acc['n']} calls "
          f"= {1e6 * acc['t'] / max(acc['n'], 1):.0f} us/call ({100 * acc['t'] / wall:.0f}%)")


if __name__ == "__main__":
    main()
