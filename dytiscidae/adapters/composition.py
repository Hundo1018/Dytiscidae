"""The composition root: the one place that knows which adapters are real.

Every dependency arrow in this codebase points inward except the ones made
here.  This module names ``FileJobStore``, ``SqliteExperimentStore``,
``SubprocessLauncher`` and the trainer registry, hands them to
``TrainingApplication``, and is imported by exactly two things: the CLI and the
worker's entry point.  Nothing in ``domain``, ``ports`` or ``application`` may
import it, and ``tests/test_architecture.py`` checks that.

Two storage arrangements, chosen with ``--store``:

``files``   everything on disk.  The default, because this project's runs are
            started with ``setsid nohup`` in containers that get reclaimed, and
            a directory of JSON survives that with no service running.
``sqlite``  the record and the lifecycle in a database, checkpoints still on
            disk.  For the queries a directory answers by scanning.  Never the
            checkpoints: they are megabytes of arrays, and a row holding one
            turns every backup of the metadata into a copy of every network the
            run ever wrote.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from ..application.services import TrainingApplication
from ..ports.clock import SystemClock
from .filesystem import (
    FileCheckpointStore,
    FileControlChannel,
    FileDatasetRepository,
    FileEventLog,
    FileExperimentStore,
    FileJobStore,
    FileMetricSink,
)
from .launchers import InlineLauncher, SubprocessLauncher
from .trainers import resolve as resolve_trainer

#: Where everything lives when nothing says otherwise.  Beside ``runs/``, which
#: is what this project already gitignores and already tars to move a machine.
DEFAULT_ROOT = "runs/lab"

STORE_BACKENDS = ("files", "sqlite")


def provenance() -> dict:
    """What code is about to run.

    Collected here rather than in the application layer, because "what commit
    is this" is a question about the environment -- it shells out to git -- and
    the inside of the hexagon must not be the thing that knows how to ask it.

    Every field is best-effort.  A checkout with no git, or a wheel installed
    without one, still produces a usable record; it simply says ``""`` where the
    sha would be, which is an honest "not known" rather than a fabricated value.
    """
    out: dict[str, Any] = {"root": os.getcwd()}
    try:
        out["git"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5, cwd=Path(__file__).resolve().parents[2]
        ).stdout.strip()
    except Exception:                                             # noqa: BLE001
        out["git"] = ""
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            timeout=5, cwd=Path(__file__).resolve().parents[2]).stdout.strip()
        # Recorded, because a checkpoint written from a dirty tree cannot be
        # matched to a commit and a record that does not say so implies it can.
        out["git_dirty"] = bool(dirty)
    except Exception:                                             # noqa: BLE001
        out["git_dirty"] = None
    import platform
    import sys
    out["python"] = sys.version.split()[0]
    out["platform"] = platform.platform()
    out["host"] = platform.node()
    for name in ("numpy", "torch", "mujoco"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "")
        except Exception:                                         # noqa: BLE001
            # Absent, not broken.  The distinction matters: a plan naming the
            # search trainer on a machine with no MuJoCo will fail, and the
            # record should say which of the three was missing.
            out[name] = ""
    return out


class Lab:
    """Every adapter for one root, built once.

    Named for what it is rather than for a pattern.  Hold one, ask it for the
    application or for an individual store; it is cheap to build and the SQLite
    connection it may own is released by ``close``.
    """

    def __init__(self, root: str | Path = DEFAULT_ROOT, *, store: str = "files",
                 metric_sample: int = 1) -> None:
        if store not in STORE_BACKENDS:
            raise ValueError(
                f"unknown store backend {store!r}; known: {', '.join(STORE_BACKENDS)}")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store_kind = store
        self.clock = SystemClock()

        self.checkpoints = FileCheckpointStore(self.root)
        self.datasets = FileDatasetRepository(self.root)
        self.control = FileControlChannel(self.root)
        self.events = FileEventLog(self.root)
        self.metrics = FileMetricSink(self.root, sample=metric_sample)

        self._sqlite = None
        if store == "sqlite":
            from .sqlite import SqliteExperimentStore, SqliteJobStore, SqliteStore
            self._sqlite = SqliteStore(self.root / "lab.sqlite3")
            self.experiments = SqliteExperimentStore(self._sqlite)
            self.jobs = SqliteJobStore(self._sqlite)
        else:
            self.experiments = FileExperimentStore(self.root)
            self.jobs = FileJobStore(self.root)

        self.launcher = SubprocessLauncher(
            root=self.root,
            # Passed to the worker so it opens the same stores.  Without it a
            # worker started against a SQLite lab would open the file stores and
            # write a second, invisible copy of the job.
            extra_args=("--store", store))
        self.inline_launcher = InlineLauncher(self._run_inline)

    # -- the application ---------------------------------------------------

    def application(self) -> TrainingApplication:
        return TrainingApplication(
            jobs=self.jobs, experiments=self.experiments,
            checkpoints=self.checkpoints, control=self.control,
            events=self.events, clock=self.clock,
            launcher=self.launcher, inline_launcher=self.inline_launcher,
            resolve_trainer=resolve_trainer, provenance=provenance,
            workspace_root=str(self.root))

    def worker(self, job_id, *, install_signal_handlers: bool = True):
        """A ``TrainingWorker`` for one job, over these adapters."""
        from ..worker.runtime import TrainingWorker

        return TrainingWorker(
            job_id=job_id, jobs=self.jobs, checkpoints=self.checkpoints,
            control=self.control, events=self.events, metrics=self.metrics,
            resolve_trainer=resolve_trainer, datasets=self.datasets,
            clock=self.clock, provenance=provenance(),
            install_signal_handlers=install_signal_handlers)

    def _run_inline(self, job_id, workspace: str) -> None:
        # Signal handlers are not installed for an inline run: the caller owns
        # this process, and a library that rewrites SIGINT under a caller who
        # did not ask is how Ctrl-C stops working in an interactive session.
        self.worker(job_id, install_signal_handlers=False).run()

    def close(self) -> None:
        self.metrics.close()
        if self._sqlite is not None:
            self._sqlite.close()

    def __enter__(self) -> "Lab":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def build(root: str | Path = DEFAULT_ROOT, *, store: str = "files",
          metric_sample: int = 1) -> tuple[TrainingApplication, Lab]:
    """A ready application and the lab behind it.

    Returns both, because the application deliberately exposes no way to reach
    its adapters -- a caller that needs the dataset repository or the metric
    sink gets it from the lab, and one that only drives use cases can ignore the
    second value.
    """
    lab = Lab(root, store=store, metric_sample=metric_sample)
    return lab.application(), lab
