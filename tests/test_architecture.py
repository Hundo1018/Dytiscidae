"""The dependency rule, as a gate rather than a convention.

Every other file in this refactor states that the domain does not depend on
PyTorch, CUDA or SQLite.  This is the file that makes the statement fail the
build when it stops being true, and it is the most important test of the set:
an architecture whose central rule is only written down decays in weeks, and
the decay is invisible until something that was supposed to be testable in
isolation needs a GPU.

Two independent measurements, because they catch different things.

**The static one** walks the AST of every module inside the hexagon and looks
at what it imports.  It catches an import that exists but is never executed --
a top-level ``import torch`` inside a branch, a type-checking import that is not
guarded.  It cannot catch a violation reached at run time.

**The dynamic one** starts a *fresh interpreter with the third-party packages
hidden* and imports the hexagon in it.  It catches everything the static scan
cannot: a lazy import inside a function that the import of the package happens
to trigger, a transitive dependency through a module that looked innocent.  It
is the honest form of the claim -- not "we believe nothing imports numpy" but
"here is an interpreter that cannot import numpy, and the domain loads in it".

The lesson this project keeps re-learning, applied to architecture: when adding
any rule, ask what it reads for a system that is *almost* obeying it.  A scan
that only checks ``domain/`` passes for an application layer that imports torch.
A scan that only checks import statements passes for a lazy import.  So both,
over all three inner packages.

Run:  PYTHONPATH=. python tests/test_architecture.py
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PACKAGE = ROOT / "dytiscidae"

#: The three packages that form the inside of the hexagon.
INNER = ("domain", "ports", "application")

#: Third-party modules the inside must never reach.  ``numpy`` is on the list
#: even though it is a numerics library rather than "infrastructure": the point
#: of the boundary is that the inside runs on a bare interpreter, and a numpy
#: import defeats that just as thoroughly as a torch one.
FORBIDDEN_THIRD_PARTY = frozenset({
    "torch", "mujoco", "numpy", "scipy", "matplotlib", "imageio", "ray",
    "sqlite3", "pandas", "sklearn",
})

#: Packages of this project that the inside must never import.  Adapters,
#: worker and every existing technical package: those are the outside.
FORBIDDEN_INTERNAL = frozenset({
    "adapters", "worker", "evolution", "envs", "physics", "core", "control",
    "learning", "viz", "ops",
})

FAILURES: list[str] = []


SKIPPED: list[str] = []


def skip(name: str, reason: str) -> None:
    """A check that did not run.  Not ``check(name, True, "SKIPPED: ...")``.

    That form printed ``[ok  ]`` and counted as a pass, so a run where the
    search trainer could not be imported reported the same summary as one where
    it was checked.  "Did not run" and "passed" do not share a line.
    """
    print(f"  [skip] {name}  -- {reason}")
    SKIPPED.append(name)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------


def _imports_of(path: Path) -> list[tuple[str, int]]:
    """Every module name this file imports, with the line it is on.

    Relative imports are resolved against the file's own package, so
    ``from ..domain.job import X`` inside ``application/`` comes back as
    ``dytiscidae.domain.job`` rather than as something unrecognisable.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parts = path.relative_to(ROOT).with_suffix("").parts
    # The *package* the file lives in, which is what a relative import counts
    # from.  ``pkg/__init__.py`` and ``pkg/mod.py`` both give ``pkg``, because
    # ``parts`` still ends in ``__init__`` for the first.  Stripping that name
    # before taking the parent -- which is the obvious thing to write -- makes
    # ``from .control import X`` inside ``ports/__init__.py`` resolve to
    # ``dytiscidae.control``, which is a real package in this project and is on
    # the forbidden list.  The false positive was indistinguishable from a true
    # one, which is why it is spelled out here.
    package = parts[:-1]

    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # level 1 is the containing package, 2 its parent, and so on.
                up = node.level - 1
                base = list(package[:len(package) - up] if up <= len(package)
                            else [])
                name = ".".join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ""
            out.append((name, node.lineno))
    return out


def _inner_modules() -> list[Path]:
    files: list[Path] = []
    for package in INNER:
        files.extend(sorted((PACKAGE / package).rglob("*.py")))
    return files


def test_the_inside_imports_no_technology() -> None:
    """Static scan: no module in domain, ports or application names a
    third-party package or an outer package of this project."""
    print("\ntest_the_inside_imports_no_technology")
    modules = _inner_modules()
    check("there is an inside to check", len(modules) >= 12,
          f"{len(modules)} modules across {', '.join(INNER)}")

    offences: list[str] = []
    for path in modules:
        rel = path.relative_to(ROOT)
        for name, lineno in _imports_of(path):
            top = name.split(".")[0]
            if top in FORBIDDEN_THIRD_PARTY:
                offences.append(f"{rel}:{lineno} imports {name}")
                continue
            if top == "dytiscidae":
                second = name.split(".")[1] if "." in name else ""
                if second in FORBIDDEN_INTERNAL:
                    offences.append(f"{rel}:{lineno} imports {name}")
    check("no forbidden import anywhere inside the hexagon", not offences,
          "; ".join(offences) if offences else
          f"{len(modules)} modules clean")


def test_the_inside_loads_without_its_dependencies_installed() -> None:
    """Dynamic proof: a fresh interpreter that cannot import numpy, torch or
    mujoco still imports the whole inside of the hexagon.

    This is the measurement the static scan cannot make.  A blocker is installed
    on ``sys.meta_path`` so the forbidden modules raise ``ImportError`` even
    though they are installed, and then the three packages are imported and a
    job is put through its whole lifecycle.  The lifecycle, not just the import:
    an import that succeeds and a first method call that reaches numpy would
    otherwise pass.
    """
    print("\ntest_the_inside_loads_without_its_dependencies_installed")
    program = r"""
import sys

BLOCKED = {"torch", "mujoco", "numpy", "scipy", "sqlite3", "matplotlib",
           "imageio", "pandas", "sklearn", "ray"}


class Blocker:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(
                f"{name} is hidden by the architecture test: nothing inside the "
                f"hexagon may import it")
        return None


sys.meta_path.insert(0, Blocker())

for name in list(sys.modules):
    if name.split(".")[0] in BLOCKED:
        del sys.modules[name]

import dytiscidae.domain as D
import dytiscidae.ports as P
import dytiscidae.application as A

# Not just the import: the whole lifecycle, so that a method reaching numpy on
# first call is caught too.
plan = D.TrainingPlan(trainer="t", seed=3, hyperparameters={"a": 1, "b": [2, 3]},
                      budget=D.TrainingBudget(max_steps=10))
job = D.TrainingJob(job_id=D.JobId("j1"), experiment_id=D.ExperimentId("e1"),
                    plan=plan, created_at=1.0)
job = job.start(now=2.0)
job = job.request_pause()
job = job.paused(now=3.0)
job = job.prepare_resume(now=4.0, checkpoint="ck1")
job = job.start(now=5.0)
job = job.failed(D.FailureInfo(kind="X", message="m"), now=6.0)
state = D.TrainingState(job_id=D.JobId("j1"), total_steps=10).advanced(step=4)
record = D.CheckpointRecord(checkpoint_id=D.CheckpointId("ck1"),
                            job_id=D.JobId("j1"),
                            experiment_id=D.ExperimentId("e1"), step=4)
experiment = D.Experiment(experiment_id=D.ExperimentId("e1"), name="n", plan=plan)

assert str(job.status) == "failed", job.status
assert state.progress == 0.4, state.progress
assert plan.digest == plan.with_().digest
assert record.step == 4
assert experiment.with_job(D.JobId("j1")).job_ids == (D.JobId("j1"),)
assert A.StartTrainingRequest(experiment_name="n").inline is False
assert isinstance(P.SystemClock().now(), float)
print("INSIDE_OK")
"""
    result = subprocess.run([sys.executable, "-c", program], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=120)
    ok = result.returncode == 0 and "INSIDE_OK" in result.stdout
    detail = "domain + ports + application import and run a full lifecycle"
    if not ok:
        tail = (result.stderr or result.stdout).strip().splitlines()[-4:]
        detail = " | ".join(tail)
    check("the hexagon runs on an interpreter with no numpy, torch or mujoco",
          ok, detail)


def test_the_outside_is_not_imported_by_importing_the_inside() -> None:
    """``import dytiscidae.application`` must not drag in an adapter.

    A resolver that imported every trainer at module load would put MuJoCo and
    torch behind every use case, including ``ListJobs``.  The registry exists to
    stop that, and this is the check that it does.
    """
    print("\ntest_the_outside_is_not_imported_by_importing_the_inside")
    program = r"""
import sys
import dytiscidae.application  # noqa: F401

leaked = sorted(
    name for name in sys.modules
    if name.split(".")[0] in {"torch", "mujoco", "numpy", "scipy"}
    or name.startswith("dytiscidae.adapters")
    or name.startswith("dytiscidae.evolution")
    or name.startswith("dytiscidae.envs"))
print("LEAKED=" + ",".join(leaked))
"""
    result = subprocess.run([sys.executable, "-c", program], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=120)
    line = next((ln for ln in result.stdout.splitlines()
                 if ln.startswith("LEAKED=")), None)
    check("the application layer imports cleanly", line is not None,
          (result.stderr or "").strip()[-200:])
    if line is None:
        return
    leaked = [x for x in line[len("LEAKED="):].split(",") if x]
    check("importing the application pulls in no adapter and no heavy library",
          not leaked, ", ".join(leaked) if leaked else "nothing leaked")


def test_the_registry_defers_every_trainer() -> None:
    """Naming the trainer registry must not import the trainers themselves.

    ``known()`` lists ``search`` without importing MuJoCo.  It is the property
    that lets a CLI print the available trainers on a machine that cannot run
    half of them.
    """
    print("\ntest_the_registry_defers_every_trainer")
    program = r"""
import sys
from dytiscidae.adapters import trainers

names = trainers.known()
heavy = sorted(n for n in sys.modules
               if n.split(".")[0] in {"torch", "mujoco"})
print("NAMES=" + ",".join(names))
print("HEAVY=" + ",".join(heavy))
"""
    result = subprocess.run([sys.executable, "-c", program], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=120)
    out = dict(ln.split("=", 1) for ln in result.stdout.splitlines() if "=" in ln)
    names = [x for x in out.get("NAMES", "").split(",") if x]
    heavy = [x for x in out.get("HEAVY", "").split(",") if x]
    check("the registry knows the search trainer", "search" in names,
          ", ".join(names) or (result.stderr or "").strip()[-200:])
    check("listing trainers imports neither torch nor mujoco", not heavy,
          ", ".join(heavy) if heavy else "nothing heavy loaded")


def test_the_composition_root_is_the_only_place_that_names_adapters() -> None:
    """Exactly one module inside ``dytiscidae`` may import both an adapter and
    the application layer, and it is ``adapters/composition.py``.

    Stated as a count rather than as a prohibition, because the CLI and the
    worker entry point legitimately reach both -- through the composition root.
    What must not exist is a *second* place that wires them, because then
    swapping an adapter means finding every wiring site.
    """
    print("\ntest_the_composition_root_is_the_only_place_that_names_adapters")
    wiring: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(ROOT)
        names = [n for n, _ in _imports_of(path)]
        touches_adapter = any(
            n.startswith("dytiscidae.adapters.") and "composition" not in n
            for n in names)
        touches_application = any(
            n.startswith("dytiscidae.application") for n in names)
        if touches_adapter and touches_application:
            wiring.append(str(rel))
    expected = ["dytiscidae/adapters/composition.py"]
    check("only the composition root wires adapters to use cases",
          wiring == expected, f"found {wiring or 'nothing'}")


def test_every_port_is_a_protocol_with_no_implementation() -> None:
    """A port must be describable without being runnable.

    Checked structurally: each name exported from ``dytiscidae.ports`` is either
    a ``Protocol``, a frozen value object, or ``SystemClock`` -- the one
    deliberate concrete class, because "read the wall clock" has exactly one
    real implementation and a separate adapter module for it would be ceremony.
    A port that grows a method body is a port that has started deciding
    something, and deciding belongs inside.
    """
    print("\ntest_every_port_is_a_protocol_with_no_implementation")
    import dataclasses
    import typing

    import dytiscidae.ports as ports

    concrete: list[str] = []
    protocols: list[str] = []
    for name in ports.__all__:
        obj = getattr(ports, name)
        if not isinstance(obj, type):
            continue
        if getattr(obj, "_is_protocol", False):
            protocols.append(name)
        elif dataclasses.is_dataclass(obj) or issubclass(obj, (str, tuple)):
            continue                      # value objects: StopRequest, WorkerStatus
        elif name == "SystemClock":
            continue                      # the one documented exception
        else:
            concrete.append(name)
    check("no undeclared concrete class is exported as a port", not concrete,
          ", ".join(concrete) if concrete else
          f"{len(protocols)} protocols: {', '.join(sorted(protocols))}")
    check("the four ports the architecture is named for are all present",
          all(hasattr(ports, n) for n in
              ("Trainer", "DatasetRepository", "CheckpointStore", "ExperimentStore")),
          "Trainer, DatasetRepository, CheckpointStore, ExperimentStore")
    assert typing is not None                       # silence the unused import


def test_the_adapters_satisfy_the_ports_they_claim() -> None:
    """Every shipped adapter is accepted by ``isinstance`` against its port.

    The ports are ``runtime_checkable`` protocols, so this is a method-name
    check rather than a signature one -- which is worth saying, because it means
    the check catches a missing method and not a wrong argument.  The argument
    shapes are exercised by the adapter and worker tests, which call them.
    """
    print("\ntest_the_adapters_satisfy_the_ports_they_claim")
    import tempfile

    from dytiscidae.adapters.filesystem import (
        FileCheckpointStore, FileControlChannel, FileDatasetRepository,
        FileEventLog, FileExperimentStore, FileJobStore, FileMetricSink)
    from dytiscidae.adapters.launchers import InlineLauncher, SubprocessLauncher
    from dytiscidae.adapters.sqlite import (
        SqliteExperimentStore, SqliteJobStore, SqliteStore)
    from dytiscidae.adapters.trainers.synthetic import SyntheticTrainer
    from dytiscidae.ports import (
        CheckpointStore, ControlChannel, DatasetRepository, EventLog,
        ExperimentStore, JobLauncher, JobStore, MetricSink, Trainer)

    root = Path(tempfile.mkdtemp(prefix="ports-"))
    sqlite_store = SqliteStore(root / "db.sqlite3")
    pairs = [
        ("FileJobStore", FileJobStore(root), JobStore),
        ("FileExperimentStore", FileExperimentStore(root), ExperimentStore),
        ("FileCheckpointStore", FileCheckpointStore(root), CheckpointStore),
        ("FileDatasetRepository", FileDatasetRepository(root), DatasetRepository),
        ("FileMetricSink", FileMetricSink(root), MetricSink),
        ("FileEventLog", FileEventLog(root), EventLog),
        ("FileControlChannel", FileControlChannel(root), ControlChannel),
        ("SqliteJobStore", SqliteJobStore(sqlite_store), JobStore),
        ("SqliteExperimentStore", SqliteExperimentStore(sqlite_store),
         ExperimentStore),
        ("InlineLauncher", InlineLauncher(lambda *_a: None), JobLauncher),
        ("SubprocessLauncher", SubprocessLauncher(root=root), JobLauncher),
        ("SyntheticTrainer", SyntheticTrainer(), Trainer),
    ]
    for name, instance, port in pairs:
        check(f"{name} satisfies {port.__name__}", isinstance(instance, port))
    sqlite_store.close()

    # The search trainer separately, because importing it costs MuJoCo and the
    # rest of this file must stay runnable without one.
    try:
        from dytiscidae.adapters.trainers.search import SearchTrainer
    except Exception as exc:                                      # noqa: BLE001
        skip("SearchTrainer satisfies Trainer",
             f"not importable here: {type(exc).__name__}: {exc}")
    else:
        check("SearchTrainer satisfies Trainer",
              isinstance(SearchTrainer(), Trainer))


def main() -> int:
    print("=" * 68)
    print("architecture: the dependency rule, checked rather than asserted")
    print("=" * 68)
    test_the_inside_imports_no_technology()
    test_the_inside_loads_without_its_dependencies_installed()
    test_the_outside_is_not_imported_by_importing_the_inside()
    test_the_registry_defers_every_trainer()
    test_the_composition_root_is_the_only_place_that_names_adapters()
    test_every_port_is_a_protocol_with_no_implementation()
    test_the_adapters_satisfy_the_ports_they_claim()

    print("\n" + "=" * 68)
    if SKIPPED:
        print(f"{len(SKIPPED)} SKIPPED, not run on this machine: "
              f"{', '.join(SKIPPED)}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all architecture checks passed" if not SKIPPED
          else f"all architecture checks passed, {len(SKIPPED)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
