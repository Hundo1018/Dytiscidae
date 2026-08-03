"""A batched segment must score what the same segment scores alone.

The competence number is what the search selects on, so agreement in qpos is
not enough -- the whole measurement chain has to land in the same place.
"""
import numpy as np

from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.batchroll import AVAILABLE, BatchedFluid, rollout_batch
from dytiscidae.envs.evaluate import Controller
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

SECONDS = 0.8


def make(name, plan):
    """Constructed once.  A TriphibianEnv compiles MJCF, which dominates this
    test's runtime -- rebuilding per domain cost more than every rollout in it
    put together and hit the timeout."""
    return TriphibianEnv(build(plan()), seed=hash(name) % 9973)


def main():
    if not AVAILABLE:
        print("SKIP: GPU extension unavailable")
        return 0

    items = list(BODY_PLANS.items())
    solo_envs = [make(n, p) for n, p in items]
    batch_envs = [make(n, p) for n, p in items]
    bf = BatchedFluid(batch_envs)
    fails = []
    for domain in (Domain.AIR, Domain.WATER, Domain.LAND):
        solo = []
        for e in solo_envs:
            e.reset(domain, randomise=False)
            c = Controller(params=e.cpg.base)
            solo.append(e.rollout(SECONDS, params=c.params, domain=domain))

        for e in batch_envs:
            e.reset(domain, randomise=False)
        ctrls = [Controller(params=e.cpg.base) for e in batch_envs]
        got = rollout_batch(batch_envs, bf, SECONDS,
                            [c.params for c in ctrls], domain)

        print(f"--- {domain.value} ---")
        print(f"{'plan':10s} {'competence':>22s} {'mean_power':>20s} "
              f"{'distance':>18s}")
        for (name, _), a, b in zip(items, solo, got):
            dc = abs(a.competence - b.competence)
            dp = abs(a.mean_power - b.mean_power) / max(abs(a.mean_power), 1e-9)
            dd = abs(a.distance - b.distance) / max(abs(a.distance), 1e-9)
            flag = ""
            if dc > 1e-6 or dp > 1e-6 or dd > 1e-5:
                flag = "  <-- MISMATCH"
                fails.append((domain.value, name))
            print(f"{name:10s} {a.competence:9.6f}/{b.competence:9.6f} "
                  f"{a.mean_power:9.3f}/{b.mean_power:9.3f} "
                  f"{a.distance:8.4f}/{b.distance:8.4f}{flag}")

    print()
    if fails:
        print(f"FAILED: {fails}")
        return 1
    print("batched segments score the same as solo segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
