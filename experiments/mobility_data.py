"""Collect the raw mobility probes once, so several experiments can share them.

The expensive part of asking anything about the mobility basis is the probing:
24 random CPG-parameter directions, each run with both signs for 1.2 s of
simulated time, per body per medium.  Roughly 15 s of wall clock for one
(body, medium, seed).  The analysis afterwards -- a least-squares, an SVD, a
threshold sweep, a damping sweep -- costs microseconds.

So this module runs the probes and caches `(deltas, responses)`, and the
experiments do their own fitting from that.  Two consequences worth stating:

* The rank sweep and the damping sweep are measured on **identical** data, so
  a difference between them is a difference in the analysis and not in the
  sample.
* Repeat identification with a different probe seed is cheap to ask for, which
  is what makes the *reproducibility* of a mode measurable at all.  A singular
  value only means something if a second independent identification finds it
  again.

Two drift guards, against two different ways the cache can lie.

*The probe loop changed.*  `collect_one` reproduces the probe loop of
`TriphibianEnv.identify`, and `verify_against_env` asserts that refitting the
cached probes gives bit-comparable singular values to the real method.  If the
env's probe loop changes, that assertion fails rather than the cache quietly
describing a different experiment.

*The physics changed underneath a cache that is still on disk.*  The first
guard cannot see this one: it recollects and compares two fresh runs, so both
sides move together and it agrees.  A cached record therefore carries
`sources`, a SHA-256 over the modules whose content decides what a probe
returns (`SOURCE_FILES`).  `collect` treats a record whose stamp does not match
the working tree as missing and recollects it.  Without this, merging a physics
fix and re-running an experiment reproduces the *old* numbers exactly and looks
like evidence that the fix changed nothing.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "_cache"

#: Matches `TriphibianEnv.identify`'s defaults, which in turn match
#: `identify_batch`'s.  Changing these here makes the cache describe a
#: different experiment from the one the search runs.
PROBE_TIME = 1.2
N_PROBES = 24
MAX_MODES = 6

#: Every module whose content can change what `collect_one` returns: the body
#: plan, the model it is built into, the environment's step, the fluid and
#: structural forces, and the CPG that drives the probe.  A record collected
#: under a different set of bytes is a different experiment.
SOURCE_FILES = (
    "dytiscidae/core/bodyplans.py",
    "dytiscidae/core/phenotype.py",
    "dytiscidae/control/cpg.py",
    "dytiscidae/envs/triphibian.py",
    "dytiscidae/physics/fluid.py",
    "dytiscidae/physics/jet.py",
    "dytiscidae/physics/medium.py",
    "dytiscidae/physics/structure.py",
    "dytiscidae/physics/energy.py",
    "dytiscidae/physics/materials.py",
    "dytiscidae/physics/wake.py",
)


def source_stamp() -> str:
    """SHA-256 over `SOURCE_FILES` as they are on disk, 16 hex digits of it.

    Content, not commit: a commit hash goes stale the moment anything is
    edited, and `git_commit`'s `-dirty` suffix would invalidate the cache on an
    unrelated edit to a README.  This moves only when the probe's own physics
    moves.
    """
    h = hashlib.sha256()
    for rel in SOURCE_FILES:
        path = ROOT / rel
        h.update(rel.encode())
        h.update(path.read_bytes() if path.exists() else b"<missing>")
    return h.hexdigest()[:16]


def body_plan(name: str):
    from dytiscidae.core import bodyplans
    return getattr(bodyplans, name)()


def build_env(plan_name: str):
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import TriphibianEnv
    return TriphibianEnv(build(body_plan(plan_name)))


def collect_one(plan_name: str, medium: str, seed: int) -> dict:
    """Run the probe loop and return the raw deltas and responses.

    Mirrors `TriphibianEnv.identify` exactly, minus the final fit.
    """
    from dytiscidae.control.cpg import CPGParams
    from dytiscidae.envs.triphibian import Domain

    env = build_env(plan_name)
    domain = Domain(medium)
    env.reset(domain, randomise=False)
    snap = env.snapshot()
    base = env.cpg.base
    n_params = env.cpg.n_params

    def reset_fn():
        env.restore(snap)
        env.budget.reset()

    def step_fn(delta):
        params = CPGParams.from_flat(base.flat() + delta, env.cpg.n)
        acc = np.zeros(6)
        n = int(PROBE_TIME / env.timestep)
        for _ in range(n):
            env.step(env.cpg.command(params, env.data.time))
            acc += env.body_twist()
        if not np.all(np.isfinite(env.root_pos())):
            return np.zeros(6)
        return acc / max(n, 1)

    rng = np.random.default_rng(seed)
    deltas = rng.normal(0.0, 0.35, size=(N_PROBES, n_params))
    responses = np.zeros((N_PROBES, 6))
    t0 = time.time()
    for k, d in enumerate(deltas):
        reset_fn()
        plus = np.asarray(step_fn(d), float)
        reset_fn()
        minus = np.asarray(step_fn(-d), float)
        responses[k] = 0.5 * (plus - minus)

    return {
        "plan": plan_name,
        "medium": medium,
        "seed": seed,
        "sources": source_stamp(),
        "deltas": deltas,
        "responses": responses,
        "n_joints": env.cpg.n,
        "n_params": n_params,
        "seconds": time.time() - t0,
    }


def verify_against_env(plan_name: str, medium: str, seed: int) -> dict:
    """Assert the cached probe loop and `TriphibianEnv.identify` agree.

    The probes are stochastic only through the seed, and both paths draw from
    `default_rng(seed)` with the same shape, so the singular values must match
    to floating-point summation order -- not approximately.
    """
    from dytiscidae.control.cpg import basis_from_probes
    from dytiscidae.envs.triphibian import Domain

    rec = collect_one(plan_name, medium, seed)
    ours = basis_from_probes(rec["deltas"], rec["responses"],
                             medium=medium, max_modes=MAX_MODES)
    env = build_env(plan_name)
    theirs = env.identify(Domain(medium), probe_time=PROBE_TIME,
                          n_probes=N_PROBES, seed=seed, max_modes=MAX_MODES)
    err = float(np.max(np.abs(ours.authority - theirs.authority)))
    rel = err / max(float(theirs.authority[0]), 1e-30)
    return {"plan": plan_name, "medium": medium, "seed": seed,
            "max_abs_sigma_diff": err, "relative": rel,
            "agrees": rel < 1e-9}


def collect(plans: list[str], media: list[str], seeds: list[int],
            *, cache: Path | None = None, refresh: bool = False,
            keep_stale: bool = False) -> list[dict]:
    """Collect (or load) probes for every (plan, medium, seed) combination.

    A cached record whose `sources` stamp does not match the working tree is
    treated as missing and recollected, so a physics change cannot be masked by
    a cache written before it.  `keep_stale=True` suppresses that, for the one
    case where it is what you want: re-reading an old cache deliberately, to
    compare against a fresh one.
    """
    cache = Path(cache) if cache else CACHE / "probes.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    want = [(p, m, s) for p in plans for m in media for s in seeds]
    stamp = source_stamp()

    have: dict[tuple, dict] = {}
    stale = 0
    if cache.exists() and not refresh:
        blob = np.load(cache, allow_pickle=True)
        for rec in blob["records"]:
            rec = dict(rec.item()) if hasattr(rec, "item") else dict(rec)
            key = (rec["plan"], rec["medium"], rec["seed"])
            if not keep_stale and rec.get("sources") != stamp:
                stale += 1
                continue
            have[key] = rec
    if stale:
        print(f"cache: {stale} record(s) predate the current physics "
              f"(stamp {stamp}); recollecting those", flush=True)

    missing = [k for k in want if k not in have]
    if missing:
        print(f"collecting {len(missing)} probe sets "
              f"(~{len(missing) * 17 / 60:.1f} min)", flush=True)
    for i, (p, m, s) in enumerate(missing, 1):
        rec = collect_one(p, m, s)
        have[(p, m, s)] = rec
        print(f"  [{i}/{len(missing)}] {p:<7} {m:<5} seed={s} "
              f"{rec['seconds']:.1f}s", flush=True)
        np.savez(cache, records=np.array(list(have.values()), dtype=object))

    return [have[k] for k in want]


def fit(rec: dict, max_modes: int = MAX_MODES):
    """Fit the project's own basis from a cached probe set."""
    from dytiscidae.control.cpg import basis_from_probes
    return basis_from_probes(rec["deltas"], rec["responses"],
                             medium=rec["medium"], max_modes=max_modes)


def jacobian(rec: dict) -> np.ndarray:
    """The fitted (P, 6) Jacobian, in the scaled twist space the basis uses."""
    scale = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
    J, *_ = np.linalg.lstsq(rec["deltas"], rec["responses"] * scale, rcond=None)
    return J


# --------------------------------------------------------------------------
# Command line: inspect or rebuild the cache without going through an
# experiment.  `--status` is the one to reach for after merging a physics
# change; it says whether the numbers an experiment is about to print came
# from before or after that change.
# --------------------------------------------------------------------------

def cache_status(cache: Path | None = None) -> dict:
    """What is on disk, and how much of it still describes the current code."""
    cache = Path(cache) if cache else CACHE / "probes.npz"
    stamp = source_stamp()
    if not cache.exists():
        return {"path": str(cache), "exists": False, "stamp": stamp,
                "total": 0, "current": 0, "stale": 0, "stamps": {}}
    blob = np.load(cache, allow_pickle=True)
    stamps: dict[str, int] = {}
    for rec in blob["records"]:
        rec = dict(rec.item()) if hasattr(rec, "item") else dict(rec)
        key = str(rec.get("sources", "unstamped"))
        stamps[key] = stamps.get(key, 0) + 1
    current = stamps.get(stamp, 0)
    total = sum(stamps.values())
    return {"path": str(cache), "exists": True, "stamp": stamp,
            "total": total, "current": current, "stale": total - current,
            "stamps": stamps}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true",
                    help="report how much of the cache matches the working "
                         "tree, and exit non-zero if any of it does not")
    ap.add_argument("--refresh", action="store_true",
                    help="recollect every combination already in the cache")
    ap.add_argument("--verify", nargs=3, metavar=("PLAN", "MEDIUM", "SEED"),
                    help="assert the cached probe loop and TriphibianEnv."
                         "identify still agree on one combination")
    args = ap.parse_args(argv)

    if args.verify:
        plan, medium, seed = args.verify[0], args.verify[1], int(args.verify[2])
        out = verify_against_env(plan, medium, seed)
        print(f"{plan} {medium} seed={seed}: "
              f"max |dsigma| = {out['max_abs_sigma_diff']:.3e} "
              f"({out['relative']:.3e} of sigma_0) -> "
              f"{'agree' if out['agrees'] else 'DISAGREE'}")
        return 0 if out["agrees"] else 1

    st = cache_status()
    print(f"cache {st['path']}")
    print(f"  working-tree stamp {st['stamp']}")
    if not st["exists"]:
        print("  no cache on disk")
    else:
        for key, n in sorted(st["stamps"].items(), key=lambda kv: -kv[1]):
            mark = "current" if key == st["stamp"] else "stale  "
            print(f"  {mark}  {key}  {n} record(s)")
        print(f"  {st['current']}/{st['total']} records describe the "
              f"current code")

    if args.status:
        return 0 if st["stale"] == 0 and st["exists"] else 1

    if args.refresh:
        if not st["exists"]:
            print("nothing to refresh")
            return 1
        blob = np.load(CACHE / "probes.npz", allow_pickle=True)
        keys = set()
        for rec in blob["records"]:
            rec = dict(rec.item()) if hasattr(rec, "item") else dict(rec)
            keys.add((rec["plan"], rec["medium"], rec["seed"]))
        plans = sorted({k[0] for k in keys})
        media = sorted({k[1] for k in keys})
        seeds = sorted({k[2] for k in keys})
        print(f"refreshing {len(plans)}x{len(media)}x{len(seeds)} "
              f"= {len(plans) * len(media) * len(seeds)} combinations")
        collect(plans, media, seeds, refresh=True)
        print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
