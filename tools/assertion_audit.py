#!/usr/bin/env python3
"""Find the assertions that cannot fail, and the ones that barely can.

Why
---

Mutation testing (``tools/mutate.py``) answers "is this suite effective" by
experiment, but it only covers the defects someone thought to write down.  This
is the cheap complement: a scan of every ``check(...)`` call in ``tests/`` for
shapes of condition that are weak *by construction*, regardless of what they are
pointed at.

The categories, in descending order of how much they matter:

``always-true``
    The condition is a literal ``True``, or a comparison of a value with
    itself.  It cannot fail.  In this repository the usual cause is the
    skip-as-pass idiom ``check(name, True, "SKIPPED: ...")``, which reports a
    check that did not run as one that passed.
``not-none``
    The whole condition is ``x is not None``, or a bare ``bool(x)`` with no
    numeric call in it.  It survives any defect that returns the wrong value
    rather than no value.  ``x is None`` is *not* in this category: it asserts a
    specific expected value and counts as substantive.
``shape-only``
    The condition mentions ``.shape`` or ``.size`` and compares nothing
    numeric.  A policy whose gradient is disconnected from its loss produces
    perfectly shaped garbage.  ``len(x) == n`` is *not* in this category: a
    cardinality is a value.
``truthy-call``
    The condition is a bare call, e.g. ``check("...", f(x))``.  Whether that is
    weak depends on ``f``; it is listed so a reader can decide.
``except-branch``
    A literal ``True`` inside an ``except`` handler: the assertion is that
    control reached the handler, with the failing ``check(..., False)`` in the
    ``try``.  Sound, and separated out so it does not inflate ``always-true``.

Everything else is reported as ``substantive`` -- a numeric comparison, a
tolerance, a set or sequence equality, a boolean combination of those.  That is
not a claim that those checks are *good*, only that their shape does not make
them vacuous; mutation testing is what judges the good ones.

Usage
-----

    PYTHONPATH=. .venv/bin/python tools/assertion_audit.py
    PYTHONPATH=. .venv/bin/python tools/assertion_audit.py --show always-true
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

#: The assertion helpers this project uses.  Each suite defines its own; the
#: name is the convention that makes this scan possible at all.
ASSERT_FUNCS = ("check",)

#: Names that mark a numeric comparison even when no operator is present.
NUMERIC_CALLS = frozenset({
    "isclose", "allclose", "abs", "norm", "isfinite", "count_nonzero",
    "array_equal", "std", "mean", "max", "min", "sum", "argmax", "percentile",
})


def _calls(node) -> set:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.attr if isinstance(f, ast.Attribute) else
                    getattr(f, "id", ""))
    return out


def classify(cond) -> str:
    """One label for the shape of a condition expression."""
    if isinstance(cond, ast.Constant) and cond.value is True:
        return "always-true"

    src = ast.unparse(cond)
    calls = _calls(cond)

    # A comparison of something with itself is always true -- unless the two
    # sides construct separate objects, which is how a value type's __eq__ is
    # tested.  `JobId("x") == JobId("x")` is two objects and a real assertion;
    # `x == x` is not.  Without this the scan flags the one test in this repo
    # that checks equality semantics, and a scanner with false positives gets
    # ignored along with its true ones.
    if isinstance(cond, ast.Compare) and len(cond.ops) == 1 and \
            isinstance(cond.ops[0], (ast.Eq, ast.Is)) and \
            ast.unparse(cond.left) == ast.unparse(cond.comparators[0]) and \
            not calls:
        return "always-true"

    has_operator = any(isinstance(n, (ast.Compare,)) for n in ast.walk(cond))
    has_numeric = bool(calls & NUMERIC_CALLS)

    # `x is None` and `x is not None` are not the same kind of claim.  The
    # first asserts a specific expected value -- a lookup that must miss, a
    # cleared channel, an open budget -- and is as substantive as any equality.
    # The second asserts only that *something* came back, and survives any
    # defect that returns the wrong value rather than no value.  Counting them
    # together put 18 sound assertions in the weak column on the first run of
    # this scan.
    is_none_cmp = (isinstance(cond, ast.Compare) and len(cond.ops) == 1
                   and isinstance(cond.ops[0], (ast.Is, ast.IsNot))
                   and isinstance(cond.comparators[0], ast.Constant)
                   and cond.comparators[0].value is None)
    if is_none_cmp:
        return ("substantive" if isinstance(cond.ops[0], ast.Is)
                else "not-none")
    if isinstance(cond, ast.Call) and getattr(cond.func, "id", "") == "bool" \
            and not has_operator and not has_numeric:
        return "not-none"

    # `.shape` / `.size`, not `len(`.  The checklist item is "checks the shape
    # of a numeric result without checking its values"; `len(listing) == 2` is a
    # cardinality assertion and is exactly the right claim for a listing API.
    # Including `len(` put 39 sound assertions in the weak column.
    mentions_shape = any(t in src for t in (".shape", ".size"))
    if mentions_shape and not has_numeric and not _has_value_comparison(cond):
        return "shape-only"

    if isinstance(cond, ast.Call) and not has_operator and not has_numeric:
        return "truthy-call"

    return "substantive"


def _has_value_comparison(cond) -> bool:
    """A comparison whose operands are not all shapes or lengths."""
    for n in ast.walk(cond):
        if isinstance(n, ast.Compare):
            parts = [ast.unparse(n.left)] + [ast.unparse(c) for c in n.comparators]
            if not all(any(t in p for t in (".shape", ".size", "len("))
                       or p.isdigit() or p.startswith("(") for p in parts):
                return True
    return False


def _in_except(tree) -> set:
    """Line numbers of every statement lexically inside an ``except`` handler.

    ``check(name, True)`` inside one is the "it raised, as required" idiom: the
    assertion is that control reached the handler at all, and the matching
    ``check(name, False)`` sits in the ``try``.  That is sound, and counting it
    as vacuous would bury the sites that really are.
    """
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            for sub in ast.walk(node):
                if hasattr(sub, "lineno"):
                    lines.add(sub.lineno)
    return lines


def scan_file(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    in_except = _in_except(tree)
    rows = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", "")
        if name not in ASSERT_FUNCS or len(node.args) < 2:
            continue
        cond = node.args[1]
        label = classify(cond)
        if label == "always-true" and node.lineno in in_except:
            label = "except-branch"
        title = (node.args[0].value
                 if isinstance(node.args[0], ast.Constant) else "<computed>")
        rows.append({"file": path.name, "line": node.lineno, "kind": label,
                     "title": str(title)[:72],
                     "condition": ast.unparse(cond)[:110]})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show", default="",
                    help="print every row of this kind")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    rows = []
    for p in sorted(TESTS.glob("test_*.py")):
        rows.extend(scan_file(p))

    per_file: dict = {}
    for r in rows:
        per_file.setdefault(r["file"], Counter())[r["kind"]] += 1

    order = ("always-true", "not-none", "shape-only", "truthy-call",
             "except-branch", "substantive")
    print(f"{'suite':<28} " + " ".join(f"{k:>13}" for k in order) + "   total")
    for f in sorted(per_file):
        c = per_file[f]
        print(f"{f:<28} " + " ".join(f"{c.get(k, 0):>13}" for k in order)
              + f"   {sum(c.values()):>5}")
    total = Counter()
    for c in per_file.values():
        total.update(c)
    print(f"{'ALL':<28} " + " ".join(f"{total.get(k, 0):>13}" for k in order)
          + f"   {sum(total.values()):>5}")

    weak = sum(total.get(k, 0) for k in order[:3])
    print(f"\n{sum(total.values())} assertions; {weak} in a shape that is weak "
          f"by construction ({weak / max(sum(total.values()), 1):.1%})")

    if args.show:
        print(f"\n--- every {args.show} ---")
        for r in rows:
            if r["kind"] == args.show:
                print(f"{r['file']}:{r['line']}  {r['title']}")
                print(f"    {r['condition']}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
