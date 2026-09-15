"""`fluid.py` and `mojo/src/fluid_gpu.mojo` are the same physics twice.

`batchroll.evaluate_tier1_batch` scores with the Mojo kernels and
`evaluate.evaluate_tier1` scores with the numpy solver.  CLAUDE.md already
records that the two are not interchangeable and have differed by 60x.  What it
does not record is that **nothing asserted they carry the same constants.**

`tests/test_search.py` compares the two numerically, which is the real check --
and it needs the built extension, so it cannot run without a GPU toolchain.  On
a machine that has neither, an edit to `lift_coefficient` that forgets the Mojo
half is invisible.  These checks need neither: they read both files as text.

They catch one class of drift, not all of it: a constant changed on one side
and not the other, and a provenance comment that no longer points at anything.
Agreement of the *formulas* is `test_search.py`'s job and stays there.

The provenance comments were line numbers -- `fluid.py:256-289` -- and six of
the eight had drifted off their target by the time this was written.
`fluid.py:292-305` claimed to be `drag_coefficient` and pointed into the middle
of `lift_coefficient`'s docstring; `fluid.py:428-432` claimed to be the
kinematics block and pointed at two assignments in `__init__`.  A line number
cannot be kept honest by anything, so they are names and block markers now, and
the first check below is that none has come back.

Run:  PYTHONPATH=. python tests/test_gpu_mirror.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY_PATH = ROOT / "dytiscidae/physics/fluid.py"
MJ_PATH = ROOT / "mojo/src/fluid_gpu.mojo"

#: Mojo symbol -> the Python function it mirrors.  The constants of each pair
#: are compared; the pairing itself is the thing a human has to keep right.
FUNCTION_MIRRORS = {
    "_skin_friction_cd": "skin_friction_cd",
    "_lift_coefficient": "lift_coefficient",
    "_drag_coefficient": "drag_coefficient",
}

#: Block markers a Mojo docstring may name.  Each must exist in `fluid.py`.
BLOCK_MARKERS = (
    "--- kinematics",
    "--- strip theory",
    "--- bluff-body drag",
    "--- added mass",
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def strip_text(text: str) -> str:
    """Drop docstrings and line comments, so prose numbers are not constants."""
    text = re.sub(r'""".*?"""', "", text, flags=re.S)
    return re.sub(r"#.*", "", text)


def mojo_body(src: str, name: str) -> str:
    """The `def name(...)` block, signature included, by indentation."""
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if not re.match(rf"\s*def {re.escape(name)}\b", line):
            continue
        indent = len(line) - len(line.lstrip())
        body, j = [line], i
        depth = line.count("(") - line.count(")")
        while depth > 0 or not lines[j].rstrip().endswith(":"):
            j += 1
            body.append(lines[j])
            depth += lines[j].count("(") - lines[j].count(")")
        for nxt in lines[j + 1:]:
            if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                break
            body.append(nxt)
        return "\n".join(body)
    raise KeyError(f"{name} is not in {MJ_PATH.name}")


def mojo_numbers(src: str, name: str) -> set[float]:
    body = strip_text(mojo_body(src, name))
    return {float(m) for m in re.findall(r"(?<![\w.])(\d+\.\d+|\d+)(?![\w.])", body)}


def python_numbers(src: str, name: str) -> set[float]:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return {float(n.value) for n in ast.walk(node)
                    if isinstance(n, ast.Constant)
                    and isinstance(n.value, (int, float))
                    and not isinstance(n.value, bool)}
    raise KeyError(f"{name} is not in {PY_PATH.name}")


# --------------------------------------------------------------------------


def test_no_line_number_references_came_back() -> None:
    print("\ngpu mirror: how the Mojo file points at the Python")
    mj = MJ_PATH.read_text()
    stale = re.findall(r"fluid\.py:\d+", mj)
    check("no `fluid.py:<line>` reference remains", not stale,
          f"found {stale}" if stale else "a line number cannot be kept honest")

    named = re.findall(r"Mirrors `fluid\.([a-z_]+)`", mj)
    check("every function reference names a function that exists",
          all(f"def {n}(" in PY_PATH.read_text() for n in named),
          f"{len(named)} references: {named}")

    marked = set(re.findall(r"marked `(--- [a-z\- ]+?)` in", mj))
    check("every block reference names a marker in fluid.py",
          all(m in PY_PATH.read_text() for m in marked),
          f"{len(marked)} markers: {sorted(marked)}")


def test_each_marker_is_unique() -> None:
    """A marker that appears twice points at two places, so it points at none."""
    print("\ngpu mirror: the block markers")
    py = PY_PATH.read_text()
    for marker in BLOCK_MARKERS:
        n = py.count(marker)
        check(f"`{marker}` appears exactly once in fluid.py", n == 1,
              f"{n} occurrences")


def test_the_pairs_carry_the_same_constants() -> None:
    """The drift that a machine without the extension cannot otherwise see."""
    print("\ngpu mirror: the constants of each mirrored function")
    py_src, mj_src = PY_PATH.read_text(), MJ_PATH.read_text()
    for mojo_name, py_name in FUNCTION_MIRRORS.items():
        a = python_numbers(py_src, py_name)
        b = mojo_numbers(mj_src, mojo_name)
        only_py, only_mj = sorted(a - b), sorted(b - a)
        # Counts legitimately differ -- `1.0 - w` in one language may be
        # spelled with an extra literal in the other -- so this is set
        # equality, which is what catches a value changed on one side.
        check(f"{py_name} and {mojo_name} use the same numbers",
              not only_py and not only_mj,
              f"{len(a & b)} shared" if not (only_py or only_mj)
              else f"python-only {only_py}, mojo-only {only_mj}")


def test_the_mirrored_set_is_the_whole_set() -> None:
    """A Mojo function that mirrors a Python one but is not in the table."""
    print("\ngpu mirror: coverage of the table")
    mj = MJ_PATH.read_text()
    named = set(re.findall(r"Mirrors `fluid\.([a-z_]+)`", mj))
    listed = set(FUNCTION_MIRRORS.values())
    check("every 'Mirrors fluid.<f>' docstring is in FUNCTION_MIRRORS",
          named <= listed, f"unlisted: {sorted(named - listed)}"
          if named - listed else f"{sorted(named)}")
    check("and every pair in FUNCTION_MIRRORS is findable in both files",
          all(f"def {m}(" in mj and f"def {p}(" in PY_PATH.read_text()
              for m, p in FUNCTION_MIRRORS.items()),
          f"{len(FUNCTION_MIRRORS)} pairs")


def main() -> int:
    test_no_line_number_references_came_back()
    test_each_marker_is_unique()
    test_the_pairs_carry_the_same_constants()
    test_the_mirrored_set_is_the_whole_set()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        print("\nThe formulas themselves are tests/test_search.py's job and "
              "need the built extension; these checks do not.")
        return 1
    print("gpu mirror checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
