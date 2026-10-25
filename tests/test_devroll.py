"""Device evaluation path (IMPL_1010 A0): the guard and its reasons.

The device path is optional.  These checks run wherever the project runs; the
ones that need MuJoCo Warp and a CUDA device skip with a reason, and a skip is
never printed as a pass.  The GPU agreement tests arrive with lane A5.

Run with:  PYTHONPATH=. .venv/bin/python tests/test_devroll.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES: list[str] = []
SKIPPED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def skip(name: str, reason: str) -> None:
    print(f"  [skip] {name}  -- {reason}")
    SKIPPED.append(name)


def test_the_device_path_reports_why_it_cannot_run() -> None:
    from dytiscidae.envs import devroll

    check("AVAILABLE is a bool", isinstance(devroll.AVAILABLE, bool))
    check("UNAVAILABLE_REASON is a str", isinstance(devroll.UNAVAILABLE_REASON, str))
    ok, reason = devroll.usable()
    check("usable() returns (bool, str)", isinstance(ok, bool) and isinstance(reason, str))
    if not devroll.AVAILABLE:
        check("MJWarp not importable: the reason says so",
              devroll.UNAVAILABLE_REASON == "MJWarp not importable",
              devroll.UNAVAILABLE_REASON)
        check("MJWarp not importable: usable() refuses with that reason",
              ok is False and reason == "MJWarp not importable", reason)
    elif not ok:
        check("MJWarp importable but unusable: reason is 'MJWarp not usable: <why>'",
              reason.startswith("MJWarp not usable: ") and len(reason) > len("MJWarp not usable: "),
              reason)
    else:
        check("usable() passes only with an empty reason", reason == "")


def test_the_device_kernels_are_not_built_yet() -> None:
    skip("device evaluation agrees with the batched and single-machine paths",
         "lanes A1a-A5 not built (docs/IMPL_1010_SPEC.md section 1)")


def main() -> int:
    functions = [test_the_device_path_reports_why_it_cannot_run,
                 test_the_device_kernels_are_not_built_yet]
    for fn in functions:
        print(f"{fn.__name__}")
        fn()
    print("\n" + "=" * 68)
    if SKIPPED:
        print(f"{len(SKIPPED)} SKIPPED: not a defect; the device path is not built yet")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all device-path checks passed" + (f", {len(SKIPPED)} skipped" if SKIPPED else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
