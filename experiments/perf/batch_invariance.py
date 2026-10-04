"""Does the shared policy give a row the same bits whatever batch it is in?

    python experiments/perf/batch_invariance.py [--trials 200] [--threads 1]

Merging two batched evaluations into one (e.g. the noise-free re-score and the
first refinement step) changes how many rows `SharedPolicy.act_many` sees per
shard.  That merge preserves every score only if a row's output does not
depend on the batch around it.  This draws random observations and compares,
bit for bit, each row computed in batches of 1, 4, 8, 16 and 32.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    import torch
    from dytiscidae.control.cpg import TWIST_DIM
    from dytiscidae.envs.triphibian import TriphibianEnv
    from dytiscidae.learning.ppo import SharedPolicy

    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    net = SharedPolicy(TriphibianEnv.OBS_DIM, TWIST_DIM + 1)
    net.eval()
    rng = np.random.default_rng(1)
    sizes = (1, 4, 8, 16, 32)
    differ = {s: 0 for s in sizes[1:]}
    worst = {s: 0.0 for s in sizes[1:]}
    for _ in range(a.trials):
        obs = rng.normal(size=(32, TriphibianEnv.OBS_DIM)).astype(np.float32)
        ref = np.stack([np.asarray(net.act_many(obs[i:i + 1], deterministic=True)[0])[0]
                        for i in range(32)])
        for s in sizes[1:]:
            got = np.asarray(net.act_many(obs[:s], deterministic=True)[0])
            d = float(np.max(np.abs(got - ref[:s])))
            differ[s] += int(d != 0.0)
            worst[s] = max(worst[s], d)
    print(f"threads {a.threads}, {a.trials} trials, each row vs the same row alone:")
    for s in sizes[1:]:
        print(f"  batch {s:2d}: differs in {differ[s]:4d} trials, max |diff| {worst[s]:.3g}")

    # The comparison a merge makes: the rows of a 4-row shard, against the
    # same rows at the head of an 8-, 16- or 32-row shard.
    print("rows of a batch of 4 vs the same rows in a larger batch:")
    rng = np.random.default_rng(2)
    for big in (8, 16, 32):
        n, w = 0, 0.0
        for _ in range(a.trials):
            obs = rng.normal(size=(big, TriphibianEnv.OBS_DIM)).astype(np.float32)
            four = np.asarray(net.act_many(obs[:4], deterministic=True)[0])
            more = np.asarray(net.act_many(obs, deterministic=True)[0])[:4]
            d = float(np.max(np.abs(four - more)))
            n += int(d != 0.0)
            w = max(w, d)
        print(f"  4 vs {big:2d}: differs in {n:4d} trials, max |diff| {w:.3g}")

    # Which batch sizes give a row the same bits: rows computed in a batch of
    # n, compared with the same rows inside a batch of 64.
    print("batch sizes whose rows match the same rows in a batch of 64:")
    obs = np.random.default_rng(3).normal(size=(64, TriphibianEnv.OBS_DIM)).astype(np.float32)
    ref = np.asarray(net.latent(torch.as_tensor(obs)).mean.detach())
    same = []
    for n in range(1, 65):
        got = np.asarray(net.latent(torch.as_tensor(obs[:n])).mean.detach())
        same.append(n if np.array_equal(got, ref[:n]) else -n)
    print("  " + " ".join(str(v) for v in same))


if __name__ == "__main__":
    main()
