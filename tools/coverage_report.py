#!/usr/bin/env python3
"""Line and branch coverage of ``dytiscidae/``, by the suites that can run here.

Why, and what the number does not mean
--------------------------------------

Coverage measures which lines executed, not whether anything checked what they
did.  A suite of ``check(name, True)`` calls reaches 100%.  So this is read
alongside ``tools/mutate.py``, which measures whether a defect in an executed
line is caught, and the two answer different halves of "is this tested":
coverage finds code nothing *touches*, mutation finds code nothing *judges*.

The number that matters most here is not the percentage.  It is the list of
modules at **0%** -- code no suite imports at all -- because that is where a
defect has nothing between it and a run.

Branch coverage is on.  Line coverage counts a two-way ``if`` as covered when
either side runs, which is how a guard that never fires reads as tested.

Usage
-----

    PYTHONPATH=. .venv/bin/python tools/coverage_report.py
    PYTHONPATH=. .venv/bin/python tools/coverage_report.py --suites test_ppo
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).exists():                                # pragma: no cover
    PYTHON = sys.executable

#: Every suite, in the order they are cheapest to run.  `test_search` is last
#: because it is the slowest by an order of magnitude.
DEFAULT_SUITES = (
    "test_index", "test_architecture", "test_domain", "test_application",
    "test_adapters", "test_worker", "test_search_adapter", "test_gpu_mirror",
    "test_math", "test_mobility_cache", "test_stall_blend",
    "test_reduced_frequency", "test_hull_buckling", "test_jet_energy",
    "test_added_mass", "test_ppo", "test_physics", "test_search",
)


def run(suites, data_dir: Path, timeout: int) -> dict:
    """Run each suite under coverage.  A suite that fails still contributes."""
    env = dict(os.environ, PYTHONPATH=str(ROOT), MUJOCO_GL="disable",
               COVERAGE_FILE=str(data_dir / ".coverage"))
    outcomes = {}
    for s in suites:
        print(f"  running {s} ...", flush=True)
        try:
            out = subprocess.run(
                [PYTHON, "-m", "coverage", "run", "--parallel-mode", "--branch",
                 f"--source={ROOT / 'dytiscidae'}", str(ROOT / "tests" / f"{s}.py")],
                cwd=str(ROOT), env=env, capture_output=True, text=True,
                timeout=timeout)
            outcomes[s] = "ok" if out.returncode == 0 else f"rc={out.returncode}"
        except subprocess.TimeoutExpired:
            outcomes[s] = f"timeout>{timeout}s"
        print(f"    {outcomes[s]}")
    return outcomes


def combine_and_report(data_dir: Path) -> dict:
    env = dict(os.environ, COVERAGE_FILE=str(data_dir / ".coverage"))
    subprocess.run([PYTHON, "-m", "coverage", "combine"], cwd=str(ROOT),
                   env=env, capture_output=True, text=True)
    out = subprocess.run([PYTHON, "-m", "coverage", "json", "-o", "-",
                          "--show-contexts"],
                         cwd=str(ROOT), env=env, capture_output=True, text=True)
    if out.returncode != 0:                                  # pragma: no cover
        raise RuntimeError(out.stderr[:400])
    return json.loads(out.stdout)


def summarise(report: dict) -> None:
    files = report["files"]
    by_pkg: dict = {}
    zero = []
    for path, info in sorted(files.items()):
        rel = Path(path)
        parts = rel.parts
        pkg = parts[1] if len(parts) > 2 and parts[0] == "dytiscidae" else "."
        t = info["summary"]
        acc = by_pkg.setdefault(pkg, {"stmts": 0, "miss": 0,
                                      "branches": 0, "partial": 0})
        acc["stmts"] += t["num_statements"]
        acc["miss"] += t["missing_lines"]
        acc["branches"] += t.get("num_branches", 0)
        acc["partial"] += t.get("num_partial_branches", 0)
        if t["num_statements"] and t["covered_lines"] == 0:
            zero.append((path, t["num_statements"]))

    print(f"\n{'package':<16} {'stmts':>7} {'line %':>8} {'branches':>9} {'branch %':>9}")
    for pkg in sorted(by_pkg):
        a = by_pkg[pkg]
        line = 100.0 * (a["stmts"] - a["miss"]) / max(a["stmts"], 1)
        br = 100.0 * (a["branches"] - a["partial"]) / max(a["branches"], 1)
        print(f"{pkg:<16} {a['stmts']:>7} {line:>7.1f}% {a['branches']:>9} "
              f"{br:>8.1f}%")
    tot = report["totals"]
    print(f"{'TOTAL':<16} {tot['num_statements']:>7} "
          f"{tot['percent_covered']:>7.1f}% {tot.get('num_branches', 0):>9} "
          f"{100.0 * (tot.get('num_branches', 0) - tot.get('num_partial_branches', 0)) / max(tot.get('num_branches', 1), 1):>8.1f}%")

    print(f"\n{len(zero)} modules at 0% — no suite imports them at all:")
    for path, n in sorted(zero, key=lambda x: -x[1]):
        print(f"  {n:>5} statements   {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suites", nargs="*", default=list(DEFAULT_SUITES))
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    import tempfile
    data_dir = Path(args.data_dir or tempfile.mkdtemp(prefix="dyt-cov-"))
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(args.suites)} suites, data in {data_dir}")
    outcomes = run(args.suites, data_dir, args.timeout)
    report = combine_and_report(data_dir)
    summarise(report)

    print("\nsuite outcomes: " + ", ".join(f"{k}={v}" for k, v in outcomes.items()))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"outcomes": outcomes, "totals": report["totals"],
             "files": {k: v["summary"] for k, v in report["files"].items()}},
            indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
