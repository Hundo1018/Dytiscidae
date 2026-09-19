#!/usr/bin/env python3
"""Run each test function of a suite on its own, and say which ones can run at all.

Why
---

Every suite here is a script whose ``main()`` calls its test functions in order.
One function that raises ends the run, and everything after it never executes --
but the console shows a traceback, not a list of what was skipped, so "the suite
is red" and "two thirds of the suite did not run" look the same.

That is not hypothetical.  ``tests/test_physics.py`` and ``tests/test_search.py``
both reach ``envs/batchroll.evaluate_tier1_batch``, which needs the Mojo GPU
fluid extension.  On a machine without it each stops at the first function that
touches it.  Measured here: 144 checks of `test_physics` and 133 of `test_search`
run before the abort, out of 37 and 45 test functions respectively -- and this
program is what turns "some of it ran" into a count of which.

It also answers a question the mutation harness raises: when a mutation survives
every suite that *can* run, is the code untested, or is the test merely
unreachable?  Those are different findings and need different fixes.

Usage
-----

    PYTHONPATH=. .venv/bin/python tools/suite_probe.py tests/test_physics.py
    PYTHONPATH=. .venv/bin/python tools/suite_probe.py --all --json out.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).exists():                                # pragma: no cover
    PYTHON = sys.executable

#: Substrings that mark a failure as "this machine lacks the hardware", not "the
#: code is wrong".  Kept explicit: anything not on this list is a real failure,
#: so a new kind of environmental breakage is reported rather than absorbed.
BLOCKED_MARKERS = (
    "GPU fluid extension not importable",
    "No module named 'full_pipeline'",
    "No module named 'mujoco'",
    "No module named 'torch'",
)

DRIVER = """
import importlib.util, sys, traceback
sys.path.insert(0, {root!r})
spec = importlib.util.spec_from_file_location("suite_under_probe", {path!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = getattr(mod, {func!r})
before = len(getattr(mod, "FAILURES", []))
try:
    fn()
except Exception:
    traceback.print_exc()
    raise SystemExit(2)
fails = getattr(mod, "FAILURES", [])[before:]
if fails:
    print("FAILED_CHECKS: " + "; ".join(fails))
    raise SystemExit(1)
raise SystemExit(0)
"""


def test_functions(path: Path) -> list:
    """Top-level ``test_*`` functions, in source order, without importing."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("test_")]


def probe_one(path: Path, func: str, timeout: int = 600) -> tuple:
    """(status, detail) for one function.  Its own interpreter, every time.

    Its own interpreter because these suites keep module-level state -- a
    ``FAILURES`` list, cached models, a torch seed -- and running two functions
    in one process makes the second depend on the first, which is the thing
    checklist item I ("測試彼此獨立") is about.
    """
    env = dict(os.environ, PYTHONPATH=str(ROOT), MUJOCO_GL="disable")
    script = DRIVER.format(root=str(ROOT), path=str(path), func=func)
    try:
        out = subprocess.run([PYTHON, "-c", script], cwd=str(ROOT), env=env,
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f">{timeout}s"
    text = (out.stdout or "") + (out.stderr or "")
    if out.returncode == 0:
        return "pass", ""
    if any(m in text for m in BLOCKED_MARKERS):
        marker = next(m for m in BLOCKED_MARKERS if m in text)
        return "BLOCKED", marker
    line = next((ln for ln in text.splitlines() if "FAILED_CHECKS:" in ln), "")
    if not line:
        line = text.strip().splitlines()[-1] if text.strip() else "no output"
    return "FAIL", line[:160]


def probe(path: Path, timeout: int = 600) -> dict:
    funcs = test_functions(path)
    rows = []
    for f in funcs:
        status, detail = probe_one(path, f, timeout)
        rows.append({"function": f, "status": status, "detail": detail})
        print(f"  [{status:7}] {f}"
              + (f"  -- {detail}" if detail else ""))
    tally: dict = {}
    for r in rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    return {"suite": path.name, "n_functions": len(funcs),
            "tally": tally, "functions": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("suites", nargs="*", help="paths to suite files")
    ap.add_argument("--all", action="store_true", help="every tests/test_*.py")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    paths = ([p for p in sorted((ROOT / "tests").glob("test_*.py"))]
             if args.all else [Path(s) for s in args.suites])
    if not paths:
        ap.error("name at least one suite, or pass --all")

    reports = []
    for p in paths:
        print(f"\n{p.name}  ({len(test_functions(p))} test functions)")
        rep = probe(p, args.timeout)
        reports.append(rep)
        parts = ", ".join(f"{k} {v}" for k, v in sorted(rep["tally"].items()))
        print(f"  => {parts}")

    print("\n" + "=" * 70)
    total: dict = {}
    for rep in reports:
        for k, v in rep["tally"].items():
            total[k] = total.get(k, 0) + v
    print("across " + ", ".join(r["suite"] for r in reports) + ":")
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(total.items())))

    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    # A blocked function is a fact about the machine, not a failure; a real
    # failure is.
    return 1 if total.get("FAIL") or total.get("TIMEOUT") else 0


if __name__ == "__main__":
    raise SystemExit(main())
