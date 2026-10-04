"""AM in one process: the batched evaluation with the clearance memo and batch
pass on, then off (`clearance` = `_clearance_now`, `clearance_many` a no-op),
alternating, so the load other processes put on the machine falls on both.

    python experiments/perf/am_ab.py --reps 4            # 4 random designs (profile_shard's)
    python experiments/perf/am_ab.py --reps 4 --rotors   # the arch46 rotor-heavy batch
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--rotors", action="store_true")
    ap.add_argument("--seconds", type=float, default=8.0)
    a = ap.parse_args()
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from profile_shard import designs, flatten
    from rotor_cost import rotor_designs
    phenos = rotor_designs(n=4) if a.rotors else designs(4, 3)
    on = (TriphibianEnv.clearance, TriphibianEnv.__dict__["clearance_many"])
    off = (TriphibianEnv._clearance_now, staticmethod(lambda envs, active=None: None))

    def run():
        t = time.perf_counter()
        res = batchroll.evaluate_tier1_batch(phenos, spec=MissionSpec(), segment_seconds=a.seconds,
                                             seed=5, identify_axes=False)
        return time.perf_counter() - t, flatten(res)

    run()                                                    # warm
    walls = {"on": [], "off": []}
    ref = None
    for _ in range(a.reps):
        for side, (c, cm) in (("off", off), ("on", on)):
            TriphibianEnv.clearance, TriphibianEnv.clearance_many = c, cm
            w, flat = run()
            walls[side].append(w)
            if ref is None:
                ref = flat
            assert flat == ref, f"{side}: numbers differ"
    TriphibianEnv.clearance, TriphibianEnv.clearance_many = on
    r = np.array(walls["on"]) / np.array(walls["off"])
    print("off", np.round(walls["off"], 2).tolist(), "on", np.round(walls["on"], 2).tolist())
    print(f"on/off ratio per pair {np.round(r, 3).tolist()}; mean {r.mean():.3f} "
          f"min {r.min():.3f} max {r.max():.3f}; every result identical")


if __name__ == "__main__":
    main()
