"""Wall time of one generation-shaped batched evaluation (16 designs, axis
identification on, shared policy attached) through the real ActorPool, at
several pool shapes.

    python experiments/perf/pool_sweep.py --shapes 4x4 8x2 6x3
    python experiments/perf/pool_sweep.py --source runs/arch46 --batches 3 \
        --seconds 6 --shapes 4x4 4x4:b 4x2:q2 4x2:q2:b 4x4

A shape is ``WxM`` (workers x min-shard), optionally ``:qP`` (``per_worker``,
a queue of W*P shards) and ``:b`` (``--pool-balance``).  ``--source`` draws each
batch from a run's final archives instead of random genomes, because late
bodies carry rotors and cost twice what random ones do; every shape in a batch
evaluates the same designs, in the order given, so a shape repeated first and
last measures the drift.  Each call's idle fraction comes from ``shard_log``.
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "experiments/perf")


def archive_designs(run, n, seed):
    from pathlib import Path
    from dytiscidae.core.phenotype import build
    from dytiscidae.evolution.archive import Archive
    elites = []
    for p in sorted(Path(run).glob("archive_*.pkl")):
        a = Archive.load(p)
        cells = a.cells.values() if isinstance(a.cells, dict) else a.cells
        elites.extend(e for e in cells if e is not None)
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(elites)):
        try:
            out.append(build(elites[i].genome))
        except Exception:
            continue
        if len(out) == n:
            break
    return out


def parse(shape):
    head, *opts = shape.split(":")
    w, m = (int(x) for x in head.split("x"))
    per = next((float(o[1:]) for o in opts if o.startswith("q")), 1.0)
    return w, m, per, "b" in opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["4x4", "8x2"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--source", default=None, help="run dir to draw genomes from")
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--out", default=None, help="append one json line per call")
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
    for b in range(a.batches):
        phenos = (archive_designs(a.source, a.n, 1000 + b) if a.source
                  else designs(a.n, 3 + b))
        rotors = [sum(1 for s in p.segments if getattr(s, "rotor", None) is not None) for p in phenos]
        for shape in a.shapes:
            w, m, per, bal = parse(shape)
            pool = ActorPool(w, min_shard=m, per_worker=per, balance=bal)
            try:
                pool.evaluate_tier1(phenos[:w], spec=MissionSpec(), segment_seconds=0.5,
                                    seed=5, identify_axes=False)            # warm workers
                pool.shard_log.clear()
                t = time.perf_counter()
                pool.evaluate_tier1(phenos, spec=MissionSpec(), segment_seconds=a.seconds,
                                    seed=5, identify_axes=True, shared=net)
                wall = time.perf_counter() - t
                log = pool.shard_log[-1] if pool.shard_log else {}
                walls = log.get("walls", [])
                idle = (sum(max(walls) - x for x in walls) / (len(walls) * max(walls))
                        if walls else None)
                row = {"batch": b, "shape": shape, "wall": round(wall, 1),
                       "shards": len(walls), "idle": None if idle is None else round(idle, 3),
                       "rotors": rotors}
                print(json.dumps(row), flush=True)
                if a.out:
                    with open(a.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
            finally:
                pool.close()


if __name__ == "__main__":
    main()
