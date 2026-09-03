"""Batched Tier-1 must produce the same MissionResult as the unbatched one.

This is the number the search actually selects on, so nothing below it matters
if this disagrees.
"""
import time

import numpy as np

from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.batchroll import AVAILABLE, evaluate_tier1_batch
from dytiscidae.envs.evaluate import evaluate_tier1

SEG = 1.0

# All seven plans now agree to within 1.5e-6 in mission_fraction, and two full
# runs of this test give bit-identical numbers.  Getting there took removing the
# atomic accumulations from the GPU scatter: warp arrival order was selecting
# between two attractors for bat -- 0.01478594, 0.00640494, 0.01473896 on three
# runs of identical code.  A gather that sums each body's panels in index order
# costs nothing measurable (2.65x against 2.68x) and removed that entirely.
#
# What remains is the summation-order difference that this port started with:
# numpy's einsum does not sum left-to-right and the kernels do.  Stepping the two
# paths side by side, the first difference appears at step 0 in xfrc_applied
# alone, at 3.8e-19 -- sub-ULP, the smallest difference two orders can produce.
# It grows through a contact-rich rollout, and teal is the leaping design so it
# spends its rollout in contact, which is where a chaotic system amplifies
# fastest.  1.4e-6 over ~10000 steps is that growth, not an error.
#
# The bound is set an order of magnitude above the worst observed gap.  It still
# catches a transcription error: those do not start at 3.8e-19, they start at the
# magnitude of whatever term was dropped, and the earlier slam, reset and atomic
# bugs showed up here as 6%, 100% and 130% respectively.
TOL = 1e-5


def main():
    if not AVAILABLE:
        print("SKIP: GPU extension unavailable")
        return 0

    phenos = [build(p()) for _, p in BODY_PLANS.items()]
    names = list(BODY_PLANS)

    t0 = time.perf_counter()
    solo = [evaluate_tier1(p, segment_seconds=SEG, seed=3) for p in phenos]
    t_solo = time.perf_counter() - t0

    t0 = time.perf_counter()
    got = evaluate_tier1_batch(phenos, segment_seconds=SEG, seed=3)
    t_batch = time.perf_counter() - t0

    print(f"{'plan':10s} {'mission_fraction':>26s} {'feasible':>10s} "
          f"{'exploit':>9s}")
    worst = 0.0
    bad = []
    for nm, a, b in zip(names, solo, got):
        d = abs(a.mission_fraction - b.mission_fraction)
        worst = max(worst, d)
        same = (a.feasible == b.feasible) and (bool(a.exploit) == bool(b.exploit))
        if d > TOL or not same:
            bad.append(nm)
        print(f"{nm:10s} {a.mission_fraction:11.8f}/{b.mission_fraction:11.8f} "
              f"{str(a.feasible):>4s}/{str(b.feasible):<4s} "
              f"{str(bool(a.exploit)):>4s}/{str(bool(b.exploit)):<4s}"
              f"{'   <-- MISMATCH' if (d > TOL or not same) else ''}")

    print()
    print(f"unbatched {t_solo*1e3:8.0f} ms    batched {t_batch*1e3:8.0f} ms"
          f"    {t_solo/t_batch:5.2f}x")
    print()
    ok = not bad
    print("batched tier-1 matches" if ok else f"FAILED: {bad}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
