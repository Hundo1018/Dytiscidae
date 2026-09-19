"""Generate ``docs/index/MODULES.md`` and ``docs/index/SYMBOLS.md`` from the AST.

Why this exists
---------------

Ninety-eight modules, 198 top-level classes, 278 functions and 682 methods.  The
failure mode that costs this project real time is not a missing feature, it is a
*second* implementation of one that already exists under a name nobody searched
for -- and the reason is that finding the first one requires knowing what it was
called.  Six times a metric has been re-derived in a probe because the function
that computes it lives in ``envs/`` and the probe author was reading
``evolution/``.

So the index is written by a program rather than by hand.  A hand-written index
is accurate on the day it is written and is a liability after that: it reads as
authoritative long after it has stopped being true, which is worse than having
none.  ``tests/test_index.py`` fails the build when the committed files differ
from what this generator produces, so the index is either correct or the build
is red.

What is generated and what is not
---------------------------------

``MODULES.md`` and ``SYMBOLS.md`` are derived, entirely, from the source.  Do not
edit them; run this.

``FEATURES.yaml`` is **not** generated, because the thing it records -- "these
six functions across four packages together implement one capability" -- is not
recoverable from an import graph.  ``batchroll.evaluate_tier1_batch`` and
``evaluate.evaluate_tier1`` are two implementations of one feature and import
each other not at all; no static analysis will pair them.  It is maintained by
hand and checked by ``tests/test_index.py``: every symbol it names must exist,
and every test module it names must exist.  A stale entry is a build failure,
which is the part a hand-maintained file can still be held to.

Usage
-----

    PYTHONPATH=. .venv/bin/python tools/index_gen.py write    # regenerate
    PYTHONPATH=. .venv/bin/python tools/index_gen.py check    # exit 1 on drift
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "dytiscidae"
OUT_DIR = ROOT / "docs" / "index"

#: Distributions that are not this project.  Everything else an import names at
#: its first segment is either the standard library or one of ours; the split
#: matters because "which modules reach for torch" is the question the
#: dependency rule in ``docs/ARCHITECTURE.md`` is enforced on.
THIRD_PARTY = frozenset({
    "numpy", "scipy", "mujoco", "torch", "matplotlib", "imageio", "ray",
    "pandas", "sklearn", "PIL", "yaml",
})

#: Rendered in the module table so the dependency rule is visible in the index
#: rather than only in the test that enforces it.
INNER = ("domain", "ports", "application")


# --------------------------------------------------------------------- model


@dataclass(frozen=True)
class Symbol:
    """One class, function or method, with where it is and what it says it does."""

    kind: str           # "class" | "function" | "method" | "property"
    name: str           # dotted within its module: ``Archive.add``
    line: int
    signature: str
    doc: str
    decorators: tuple = ()

    @property
    def sort_key(self) -> tuple:
        return (self.name.count("."), self.name.lower(), self.line)


@dataclass
class Module:
    """One source file: what it is for, what it imports, what it defines."""

    dotted: str
    path: str           # repo-relative, posix
    loc: int
    doc: str
    internal: tuple = ()        # dotted modules of this project
    third_party: tuple = ()     # distribution names
    symbols: list = field(default_factory=list)
    has_main: bool = False      # defines ``main()``
    is_script: bool = False     # has ``if __name__ == "__main__"``

    @property
    def package(self) -> str:
        """``dytiscidae.evolution.loop`` -> ``evolution``; top level -> ``.``."""
        parts = self.dotted.split(".")
        return parts[1] if len(parts) > 2 else "."

    @property
    def entry_point(self) -> str:
        if self.is_script and self.has_main:
            return "script + main()"
        if self.is_script:
            return "script"
        if self.has_main:
            return "main()"
        return ""


# ----------------------------------------------------------------- the scan


def _first_line(doc: str | None) -> str:
    """The first non-empty line of a docstring, flattened onto one line.

    One line, not the summary paragraph: the index is read by scanning, and a
    table whose cells wrap is a table nobody scans.
    """
    if not doc:
        return ""
    for raw in doc.splitlines():
        line = raw.strip()
        if line:
            return line.replace("|", "\\|").replace("\n", " ")
    return ""


def _signature(node) -> str:
    args = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"({args}){ret}"


def _decorators(node) -> tuple:
    out = []
    for d in node.decorator_list:
        try:
            out.append(ast.unparse(d))
        except Exception:                                    # pragma: no cover
            out.append("<expr>")
    return tuple(out)


def _resolve(node, dotted: str, is_package: bool) -> list:
    """Absolute dotted targets of one import statement, relative ones resolved.

    ``from ..learning import ppo`` inside ``dytiscidae.evolution.loop`` is
    ``dytiscidae.learning.ppo``.  Getting this wrong would make every relative
    import in the project look like a third-party one, which is most of them.

    ``is_package`` is load-bearing and was wrong in the first version of this
    file.  For a module ``a.b.c``, one leading dot means ``a.b``; for a package
    ``a.b`` written in ``a/b/__init__.py``, one leading dot means ``a.b``
    itself.  With the distinction missed, ``from .control import …`` inside
    ``ports/__init__.py`` resolved to ``dytiscidae.control`` -- the CPG package
    -- and the index reported that `ports`, which is inside the hexagon,
    depends on a package full of numpy.  An index that invents a dependency-rule
    violation is worse than no index.
    """
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if node.level:
        parts = dotted.split(".")
        base = parts if is_package else parts[:-1]
        base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
        prefix = ".".join(base + ([node.module] if node.module else []))
    else:
        prefix = node.module or ""
    if not prefix:
        return []
    # ``from x import y`` may name a module or a symbol; keep the module, and
    # keep ``x.y`` too so a submodule import is not lost.
    out = [prefix]
    for a in node.names:
        if a.name != "*":
            out.append(f"{prefix}.{a.name}")
    return out


def _dotted_of(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def scan_module(path: Path, known: frozenset) -> Module:
    """Parse one file into a :class:`Module`.  Never imports it."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    dotted = _dotted_of(path)
    is_package = path.name == "__init__.py"
    mod = Module(dotted=dotted, path=path.relative_to(ROOT).as_posix(),
                 loc=len(text.splitlines()), doc=_first_line(ast.get_docstring(tree)))

    internal, third = set(), set()
    for node in ast.walk(tree):
        for target in _resolve(node, dotted, is_package):
            head = target.split(".")[0]
            if head == PACKAGE:
                # Keep only targets that name a real module of ours; the rest
                # are symbols inside one and their module is already recorded.
                if target in known:
                    internal.add(target)
                elif target.rsplit(".", 1)[0] in known:
                    internal.add(target.rsplit(".", 1)[0])
            elif head in THIRD_PARTY:
                third.add(head)
    internal.discard(dotted)
    mod.internal = tuple(sorted(internal))
    mod.third_party = tuple(sorted(third))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            mod.symbols.append(Symbol(
                "class", node.name, node.lineno, f"({bases})" if bases else "",
                _first_line(ast.get_docstring(node)), _decorators(node)))
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if m.name.startswith("_") and m.name != "__init__":
                        continue
                    decs = _decorators(m)
                    kind = "property" if "property" in decs else "method"
                    mod.symbols.append(Symbol(
                        kind, f"{node.name}.{m.name}", m.lineno, _signature(m),
                        _first_line(ast.get_docstring(m)), decs))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            mod.symbols.append(Symbol(
                "function", node.name, node.lineno, _signature(node),
                _first_line(ast.get_docstring(node)), _decorators(node)))
            if node.name == "main":
                mod.has_main = True
        elif isinstance(node, ast.If):
            test = ast.unparse(node.test)
            if "__name__" in test and "__main__" in test:
                mod.is_script = True

    mod.symbols.sort(key=lambda s: s.sort_key)
    return mod


def scan(roots=(PACKAGE,)) -> list:
    """Every module under ``roots``, sorted by path.  Deterministic by construction."""
    files: list = []
    for r in roots:
        files.extend(sorted((ROOT / r).rglob("*.py")))
    files = [f for f in files
             if not any(part in {".venv", "__pycache__", "build"} for part in f.parts)]
    known = frozenset(_dotted_of(f) for f in files)
    return [scan_module(f, known) for f in files]


# -------------------------------------------------------------- the renderers


BANNER = ("<!-- Generated by tools/index_gen.py.  Do not edit by hand: "
          "tests/test_index.py fails the build when this file and the source "
          "disagree.  Run `PYTHONPATH=. python tools/index_gen.py write`. -->")


def _short(dotted: str) -> str:
    return dotted[len(PACKAGE) + 1:] if dotted.startswith(PACKAGE + ".") else dotted


def render_modules(mods: list) -> str:
    """The directory-level view: responsibility, size, dependencies."""
    lines = [BANNER, "", "# Modules", "",
             f"{len(mods)} modules, {sum(m.loc for m in mods):,} lines, "
             f"{sum(len(m.symbols) for m in mods):,} indexed symbols.",
             "",
             "`depends on` lists modules of this project only; `third party` is",
             "the outside world.  The three packages named **inner** are the",
             "inside of the hexagon and must show an empty `third party` column",
             "-- `tests/test_architecture.py` is what enforces that, this table",
             "is what makes a violation visible while reading.", ""]

    by_pkg: dict = {}
    for m in mods:
        by_pkg.setdefault(m.package, []).append(m)

    for pkg in sorted(by_pkg):
        mark = "  (inner)" if pkg in INNER else ""
        lines += [f"## `{pkg}`{mark}", "",
                  "| module | responsibility | lines | entry | third party | depends on |",
                  "|---|---|---:|---|---|---|"]
        for m in sorted(by_pkg[pkg], key=lambda x: x.path):
            deps = ", ".join(f"`{_short(d)}`" for d in m.internal) or "—"
            third = ", ".join(f"`{t}`" for t in m.third_party) or "—"
            lines.append(f"| `{_short(m.dotted)}` | {m.doc or '—'} | {m.loc} | "
                         f"{m.entry_point or '—'} | {third} | {deps} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_symbols(mods: list) -> str:
    """The symbol-level view: where every public name is defined."""
    lines = [BANNER, "", "# Symbols", "",
             "Every public class, function, method and property defined at "
             "module or class scope, with the line it starts on.  Private names "
             "(leading underscore) are omitted except `__init__`.", "",
             "Search this file before writing a new function.  The reuse ladder "
             "is in `CLAUDE.md`.", ""]
    for m in sorted(mods, key=lambda x: x.path):
        if not m.symbols:
            continue
        lines += [f"## `{_short(m.dotted)}` — `{m.path}`", ""]
        if m.doc:
            lines += [f"> {m.doc}", ""]
        lines += ["| symbol | kind | line | signature | summary |",
                  "|---|---|---:|---|---|"]
        for s in m.symbols:
            sig = s.signature.replace("|", "\\|")
            sig = f"`{sig}`" if sig else "—"
            lines.append(f"| `{s.name}` | {s.kind} | {s.line} | {sig} | "
                         f"{s.doc or '—'} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------- FEATURES.yaml, checked


#: The one file in ``docs/index/`` that a person writes.  See the module
#: docstring for why it is not generated.
FEATURES_FILE = "FEATURES.yaml"

#: The keys a feature may carry.  Anything else is a typo, and a typo in a
#: hand-maintained index is invisible until somebody trusts it.
FEATURE_KEYS = ("implementation", "tests", "note")


def load_features(path: Path) -> dict:
    """Parse the restricted YAML ``FEATURES.yaml`` is written in.

    Deliberately not PyYAML.  The file is a mapping of feature name to a mapping
    of ``implementation`` / ``tests`` / ``note`` to a list of dotted paths, and
    that is the whole grammar; adding a dependency to read twenty lines of it
    would put a third-party package in the build of a project whose central
    architectural rule is which packages may appear where.

    Being a *strict* subset is the point: anything outside the grammar raises
    with a line number rather than being silently reinterpreted, so a feature
    that was meant to be recorded and was written wrongly fails the build
    instead of quietly not existing.
    """
    out: dict = {}
    feature = section = None
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if indent == 0:
            if not body.endswith(":"):
                raise ValueError(f"{path}:{lineno}: a feature name must end in ':'")
            feature = body[:-1].strip()
            if feature in out:
                raise ValueError(f"{path}:{lineno}: duplicate feature {feature!r}")
            out[feature] = {}
            section = None
        elif indent == 2:
            if feature is None:
                raise ValueError(f"{path}:{lineno}: a key before any feature")
            if body.endswith(":"):
                section = body[:-1].strip()
                if section not in FEATURE_KEYS:
                    raise ValueError(
                        f"{path}:{lineno}: unknown key {section!r}; "
                        f"expected one of {FEATURE_KEYS}")
                out[feature][section] = []
            elif ":" in body:
                key, _, value = body.partition(":")
                key = key.strip()
                if key not in FEATURE_KEYS:
                    raise ValueError(
                        f"{path}:{lineno}: unknown key {key!r}; "
                        f"expected one of {FEATURE_KEYS}")
                out[feature][key] = value.strip()
                section = None
            else:
                raise ValueError(f"{path}:{lineno}: expected 'key:' or 'key: value'")
        elif indent == 4 and body.startswith("- "):
            if feature is None or section is None:
                raise ValueError(f"{path}:{lineno}: a list item outside a key")
            out[feature][section].append(body[2:].strip())
        else:
            raise ValueError(
                f"{path}:{lineno}: indent {indent} is outside the grammar "
                "(0 for a feature, 2 for a key, 4 for '- item')")
    return out


def symbol_table(mods: list) -> set:
    """Every dotted name the source defines: modules, and symbols within them."""
    names = set()
    for m in mods:
        names.add(m.dotted)
        for s in m.symbols:
            names.add(f"{m.dotted}.{s.name}")
    return names


def check_features(features: dict, mods: list, test_dir: Path) -> list:
    """Problems with ``FEATURES.yaml``.  Empty means it still describes the source.

    Four gates, because a hand-maintained file rots in four ways: an entry that
    names something deleted, an entry that names a test that no longer exists, a
    feature recorded with no test at all, and a whole package nobody indexed.
    """
    problems = []
    known = symbol_table(mods)
    packages = {m.package for m in mods} - {"."}
    covered = set()

    for name, body in sorted(features.items()):
        impl = body.get("implementation") or []
        tests = body.get("tests") or []
        if not impl:
            problems.append(f"{name}: names no implementation")
        if not tests:
            problems.append(f"{name}: names no test")
        for dotted in impl:
            full = dotted if dotted.startswith(PACKAGE + ".") else f"{PACKAGE}.{dotted}"
            if full not in known:
                problems.append(f"{name}: implementation {dotted!r} does not exist")
            else:
                parts = full.split(".")
                if len(parts) > 2:
                    covered.add(parts[1])
        for dotted in tests:
            mod, _, func = dotted.partition("::")
            if not mod.startswith("tests."):
                problems.append(f"{name}: test {dotted!r} is not under tests.")
                continue
            path = test_dir / (mod.split(".", 1)[1] + ".py")
            if not path.exists():
                problems.append(f"{name}: test module {dotted!r} does not exist")
            elif func and f"def {func}(" not in path.read_text(encoding="utf-8"):
                problems.append(f"{name}: test {dotted!r} names no such function")

    for pkg in sorted(packages - covered):
        problems.append(f"package {pkg!r} is named by no feature")
    return problems


# ------------------------------------------------------------------- the CLI


def generate(roots=(PACKAGE,)) -> dict:
    """Filename -> content, for the files this program owns."""
    mods = scan(roots)
    return {"MODULES.md": render_modules(mods),
            "SYMBOLS.md": render_symbols(mods)}


def write(out_dir: Path = OUT_DIR) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, body in generate().items():
        (out_dir / name).write_text(body, encoding="utf-8")
        written.append(name)
    return written


def drift(out_dir: Path = OUT_DIR) -> list:
    """Names of generated files that are missing or stale.  Empty means clean."""
    bad = []
    for name, body in generate().items():
        p = out_dir / name
        if not p.exists() or p.read_text(encoding="utf-8") != body:
            bad.append(name)
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("write", "check"), nargs="?", default="write")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    out = Path(args.out)

    if args.action == "write":
        for name in write(out):
            print(f"wrote {(out / name).relative_to(ROOT)}")
        return 0

    rc = 0
    bad = drift(out)
    if bad:
        print("generated index is stale: " + ", ".join(bad))
        print("run: PYTHONPATH=. python tools/index_gen.py write")
        rc = 1
    else:
        print("generated index matches the source")

    features = out / FEATURES_FILE
    if not features.exists():
        print(f"{FEATURES_FILE} is missing")
        return 1
    problems = check_features(load_features(features), scan(), ROOT / "tests")
    if problems:
        print(f"{FEATURES_FILE} no longer describes the source:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"{FEATURES_FILE} resolves")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
