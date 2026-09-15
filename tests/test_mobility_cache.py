"""The probe cache must not survive a change to the physics it was measured on.

`experiments/mobility_data.py` caches 84 probe sets so the rank sweep and the
damping sweep can share identical data.  One probe set is ~8 s of simulation,
so the cache is the difference between a 12-minute experiment and a 5-second
one, and it is kept on disk between sessions.

That is a trap with a specific failure mode, and it fired once: after merging
the F-01 added-mass fix, re-running both experiments reproduced the previous
numbers to the digit.  It looked like evidence that the fix does not touch the
mobility identification.  It was the cache.

The module's older guard cannot catch this.  `verify_against_env` recollects
with `collect_one` and compares against `TriphibianEnv.identify` -- both fresh,
both on current code -- so the two move together and it agrees no matter how
old the file on disk is.

So each record carries `sources`, a SHA-256 over the modules that decide what a
probe returns.  These tests pin the three properties that stamp has to have.

Run:  PYTHONPATH=. python tests/test_mobility_cache.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "disable")

from experiments import mobility_data as md  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def write_cache(path: Path, records: list[dict]) -> None:
    np.savez(path, records=np.array(records, dtype=object))


def fake(plan: str, medium: str, seed: int, stamp: str) -> dict:
    """A record with the shape `collect` indexes on and nothing that costs."""
    return {"plan": plan, "medium": medium, "seed": seed, "sources": stamp,
            "deltas": np.zeros((2, 3)), "responses": np.zeros((2, 6)),
            "n_joints": 1, "n_params": 3, "seconds": 0.0}


def test_the_stamp_is_stable_and_covers_the_physics() -> None:
    """Same bytes, same stamp; changed bytes, different stamp."""
    print("\nmobility cache: what the stamp is a function of")
    a = md.source_stamp()
    b = md.source_stamp()
    check("repeated calls on an unchanged tree agree", a == b, f"{a}")

    # Every file it hashes must exist, or the stamp is silently hashing the
    # string "<missing>" and would not move when that module changed.
    missing = [r for r in md.SOURCE_FILES if not (md.ROOT / r).exists()]
    check("every file in SOURCE_FILES is on disk", not missing,
          f"{len(md.SOURCE_FILES)} files" if not missing else str(missing))

    # The modules whose forces a probe integrates have to be in the set.
    must = {"dytiscidae/physics/fluid.py", "dytiscidae/envs/triphibian.py",
            "dytiscidae/control/cpg.py", "dytiscidae/core/bodyplans.py"}
    check("it covers the fluid solver, the env, the CPG and the body plans",
          must <= set(md.SOURCE_FILES),
          f"missing {sorted(must - set(md.SOURCE_FILES))}"
          if not must <= set(md.SOURCE_FILES) else "all four")

    # Perturb one byte of a covered file and the stamp must move.  Restored in
    # a finally, so a failure here cannot leave the tree edited.
    victim = md.ROOT / "dytiscidae/physics/fluid.py"
    original = victim.read_bytes()
    try:
        victim.write_bytes(original + b"\n# stamp probe\n")
        moved = md.source_stamp()
    finally:
        victim.write_bytes(original)
    check("one appended comment in fluid.py moves it", moved != a,
          f"{a} -> {moved}")
    check("and restoring the file restores the stamp", md.source_stamp() == a)


def test_a_stale_record_is_recollected_not_returned() -> None:
    """The property the whole guard exists for."""
    print("\nmobility cache: what happens to a record from another tree")
    stamp = md.source_stamp()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probes.npz"
        write_cache(path, [fake("bat", "air", 1, "0000000000000000")])

        # `collect` would have to run the real probe loop to replace it, which
        # costs 8 s and needs MuJoCo.  What is under test is the decision, so
        # count what it decides to collect rather than letting it collect.
        asked: list[tuple] = []
        real = md.collect_one
        md.collect_one = lambda p, m, s: asked.append((p, m, s)) or fake(
            p, m, s, stamp)
        try:
            out = md.collect(["bat"], ["air"], [1], cache=path)
        finally:
            md.collect_one = real

        check("a record stamped from another tree is recollected",
              asked == [("bat", "air", 1)], f"collected {asked}")
        check("and what comes back carries the current stamp",
              out[0]["sources"] == stamp, out[0]["sources"])

        # Second call, nothing changed: it must now be a pure cache hit.
        asked.clear()
        md.collect_one = lambda p, m, s: asked.append((p, m, s)) or fake(
            p, m, s, stamp)
        try:
            md.collect(["bat"], ["air"], [1], cache=path)
        finally:
            md.collect_one = real
        check("a record from this tree is a cache hit", asked == [],
              "no probe re-run")


def test_keep_stale_is_the_only_way_past_it() -> None:
    """An escape hatch that has to be asked for by name."""
    print("\nmobility cache: the deliberate override")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probes.npz"
        write_cache(path, [fake("bat", "air", 1, "0000000000000000")])
        asked: list[tuple] = []
        real = md.collect_one
        md.collect_one = lambda p, m, s: asked.append((p, m, s)) or fake(
            p, m, s, md.source_stamp())
        try:
            out = md.collect(["bat"], ["air"], [1], cache=path,
                             keep_stale=True)
        finally:
            md.collect_one = real
        check("keep_stale=True returns the old record untouched",
              asked == [] and out[0]["sources"] == "0000000000000000",
              out[0]["sources"])


def test_status_reports_the_split() -> None:
    print("\nmobility cache: what --status sees")
    stamp = md.source_stamp()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probes.npz"
        write_cache(path, [fake("bat", "air", 1, stamp),
                           fake("bat", "air", 2, "0000000000000000"),
                           fake("eel", "water", 1, "0000000000000000")])
        st = md.cache_status(path)
        check("it counts the current records", st["current"] == 1,
              f"{st['current']}/{st['total']}")
        check("and the stale ones", st["stale"] == 2, str(st["stale"]))

        st2 = md.cache_status(Path(tmp) / "absent.npz")
        check("a cache that is not there is reported, not crashed on",
              st2["exists"] is False and st2["total"] == 0)


def main() -> int:
    test_the_stamp_is_stable_and_covers_the_physics()
    test_a_stale_record_is_recollected_not_returned()
    test_keep_stale_is_the_only_way_past_it()
    test_status_reports_the_split()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("mobility cache checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
