"""The navigation index, as a gate rather than a document.

`docs/index/` exists so that the question "does this already exist, and where?"
has an answer that is cheaper than reading 30,000 lines.  An index is only worth
consulting if it is true, and a stale index is worse than none: it reads as
authoritative long after it has stopped being accurate, and the thing it is
consulted for is precisely the case where the reader cannot tell.

So there are four gates.

1. **Drift.**  `MODULES.md` and `SYMBOLS.md` are regenerated here and compared
   byte for byte with what is committed.  Any change to the source that moves a
   symbol, renames one, adds an import or edits a first docstring line fails the
   build until `tools/index_gen.py write` has been run.
2. **Determinism.**  The generator is run twice and must produce the same bytes,
   because a generator whose output depends on dictionary or filesystem order
   makes gate 1 fire at random and be ignored within a week.
3. **`FEATURES.yaml` resolves.**  It is the one file in `docs/index/` a person
   writes, so it is the one that can name something that no longer exists.
   Every implementation entry must be a symbol the AST scan found, every test
   entry an existing module (and function, where one is named), every feature
   must have at least one test, and every package of the project must be named
   by at least one feature.
4. **The grammar is strict.**  The restricted YAML reader must *reject* what is
   outside its subset rather than silently reinterpret it, or a mistyped key
   becomes a feature that quietly is not recorded.

And one cross-check: the `third party` column of `MODULES.md` must be empty for
every module inside the hexagon.  `tests/test_architecture.py` is what enforces
the dependency rule; this asserts that the index a reader navigates by tells
them the same thing, so the two cannot drift apart into a document that says
`domain` is clean and a build that knows it is not.

Run:  PYTHONPATH=. python tests/test_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import index_gen  # noqa: E402

INDEX_DIR = ROOT / "docs" / "index"

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def test_the_generated_index_matches_the_source() -> None:
    print("\nindex: the generated files")
    stale = index_gen.drift(INDEX_DIR)
    check("MODULES.md and SYMBOLS.md are what the source generates",
          not stale,
          "clean" if not stale else f"stale: {', '.join(stale)}; run "
                                    "`PYTHONPATH=. python tools/index_gen.py write`")

    generated = index_gen.generate()
    for name, body in generated.items():
        p = INDEX_DIR / name
        check(f"{name} is committed", p.exists(),
              f"{len(body.splitlines())} lines" if p.exists() else "missing")


def test_the_generator_is_deterministic() -> None:
    """Twice, same bytes.  A generator that is not is a gate nobody trusts."""
    print("\nindex: the generator")
    a = index_gen.generate()
    b = index_gen.generate()
    check("two runs produce identical output",
          a == b, f"{len(a)} files, "
                  f"{sum(len(v) for v in a.values()):,} characters")

    mods = index_gen.scan()
    paths = [m.path for m in mods]
    check("and the module list is sorted and unique",
          paths == sorted(paths) and len(set(paths)) == len(paths),
          f"{len(paths)} modules")


def test_the_index_agrees_with_the_dependency_rule() -> None:
    """Nothing inside the hexagon may reach for a third-party package.

    `tests/test_architecture.py` enforces this statically and dynamically.  This
    asserts the *index* says so too, so that a reader navigating by `MODULES.md`
    is told what the build knows.
    """
    print("\nindex: the dependency rule, as the index states it")
    offenders = []
    for m in index_gen.scan():
        if m.package in index_gen.INNER and m.third_party:
            offenders.append(f"{m.dotted} -> {', '.join(m.third_party)}")
    check("the index shows no third-party import inside domain/ports/application",
          not offenders, "clean" if not offenders else "; ".join(offenders))

    table = (INDEX_DIR / "MODULES.md").read_text(encoding="utf-8")
    for pkg in index_gen.INNER:
        check(f"and `{pkg}` is marked as inner in the table",
              f"## `{pkg}`  (inner)" in table, "labelled")


def test_features_resolve_to_things_that_exist() -> None:
    print("\nindex: FEATURES.yaml")
    path = INDEX_DIR / index_gen.FEATURES_FILE
    check("FEATURES.yaml is committed", path.exists(), str(path.name))
    if not path.exists():
        return

    features = index_gen.load_features(path)
    problems = index_gen.check_features(features, index_gen.scan(), ROOT / "tests")
    check("every entry resolves, every feature has a test, every package is named",
          not problems,
          f"{len(features)} features"
          if not problems else f"{len(problems)} problems: " + "; ".join(problems[:6]))

    with_note = sum(1 for v in features.values() if v.get("note"))
    n_impl = sum(len(v.get("implementation") or []) for v in features.values())
    print(f"       {len(features)} features, {n_impl} implementation entries, "
          f"{with_note} with a note")


def test_the_feature_grammar_rejects_what_it_does_not_understand() -> None:
    """The strictness is the point, so it is tested rather than asserted.

    A reader who mistypes `implementations:` should get a build failure naming
    the line, not a feature that silently records nothing.
    """
    print("\nindex: the FEATURES.yaml grammar")
    import tempfile

    bad = {
        "an unknown key": "Feature:\n  implementations:\n    - a.b.C\n",
        "a feature without a colon": "Feature\n  implementation:\n    - a.b.C\n",
        "a list item outside a key": "Feature:\n    - a.b.C\n",
        "a duplicate feature": "F:\n  tests:\n    - tests.test_ppo\nF:\n  tests:\n    - tests.test_ppo\n",
        "an indent outside the grammar": "Feature:\n   implementation:\n    - a.b.C\n",
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in bad.items():
            p = Path(tmp) / "f.yaml"
            p.write_text(text, encoding="utf-8")
            try:
                index_gen.load_features(p)
                check(f"rejects {name}", False, "parsed without complaint")
            except ValueError as exc:
                check(f"rejects {name}", True, str(exc).split(": ", 1)[-1])

        # And it accepts the grammar it documents, including comments and notes.
        p = Path(tmp) / "ok.yaml"
        p.write_text("# a comment\n\nFeature:\n  implementation:\n"
                     "    - a.b.C   # trailing comment\n  tests:\n"
                     "    - tests.test_ppo\n  note: one line\n", encoding="utf-8")
        got = index_gen.load_features(p)
        check("and accepts the documented grammar",
              got == {"Feature": {"implementation": ["a.b.C"],
                                  "tests": ["tests.test_ppo"],
                                  "note": "one line"}},
              str(got))


def main() -> int:
    test_the_generated_index_matches_the_source()
    test_the_generator_is_deterministic()
    test_the_index_agrees_with_the_dependency_rule()
    test_features_resolve_to_things_that_exist()
    test_the_feature_grammar_rejects_what_it_does_not_understand()

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all index checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
