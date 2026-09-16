"""The ``JobLauncher`` port: starting a training worker, and knowing whether it lives.

Point 5 of the architecture -- long training is isolated from the main process
-- is this port.  Everything behind it is an operating-system decision: a
subprocess, a container, a batch submission.  The application never learns
which.

``is_alive`` is here and is not a convenience.  A job whose store says RUNNING
and whose worker died in ``import torch`` is indistinguishable from a healthy
one by the store alone, and this project's operating notes contain the
consequence: workers surviving their parent for two hours holding a gigabyte,
found by hand with ``pgrep``.  A launcher that can answer "is this still a
thing" makes that a query.

``terminate`` is last, deliberately.  The route to a stop is the
``ControlChannel`` -- cooperative, at a boundary, with a checkpoint.  Killing a
worker costs every step since the last checkpoint, so it belongs to the case
where cooperation failed: a trainer that declares ``cooperative_stop=False``, or
one that was asked and did not answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from ..domain.ids import JobId


class WorkerStatus(str, Enum):
    STARTING = "starting"
    ALIVE = "alive"
    #: Exited.  Whether that was success or failure is the *job's* business, not
    #: the launcher's -- a worker that exits 0 after recording a FAILED job did
    #: its work correctly.
    GONE = "gone"
    #: The launcher cannot tell.  A different machine, a reclaimed container, a
    #: pid that has been recycled.  Distinct from GONE because acting on
    #: "unknown" as though it were "gone" is how two workers end up writing the
    #: same run directory.
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class WorkerHandle:
    """What the application keeps in order to find a worker again.

    Serialisable on purpose: the application that launched a 21-hour run is not
    the process that will check on it.
    """

    job_id: JobId
    #: Which launcher made this, so a handle can be routed back to it.
    kind: str = "unknown"
    pid: int | None = None
    #: Where the worker's own stdout/stderr went.  The first thing anyone wants
    #: when a worker dies before it can record anything.
    log_path: str | None = None
    started_at: float = 0.0
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"job_id": str(self.job_id), "kind": self.kind, "pid": self.pid,
                "log_path": self.log_path, "started_at": self.started_at,
                "detail": dict(self.detail)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "WorkerHandle":
        return cls(job_id=JobId(d["job_id"]), kind=d.get("kind", "unknown"),
                   pid=d.get("pid"), log_path=d.get("log_path"),
                   started_at=float(d.get("started_at", 0.0)),
                   detail=dict(d.get("detail") or {}))


@runtime_checkable
class JobLauncher(Protocol):
    @property
    def kind(self) -> str:
        """A short name, recorded on the handle so it can be routed back."""

    def launch(self, job_id: JobId, *, workspace: str,
               env: Mapping[str, str] | None = None) -> WorkerHandle:
        """Start a worker for this job and return once it has been started.

        Started, not running.  The job goes RUNNING when the *worker* says so,
        because the interval between those two is where ``import torch`` fails.
        """

    def is_alive(self, handle: WorkerHandle) -> WorkerStatus:
        """Whether the worker still exists.  Never raises for a handle it cannot
        interpret -- that is ``UNKNOWN``, which is a fact, not an error."""

    def terminate(self, handle: WorkerHandle, *, grace_seconds: float = 30.0) -> bool:
        """Stop a worker the hard way.  Returns whether it is gone afterwards.

        The last resort: everything since the last checkpoint is lost.  An
        implementation signals the worker's whole process group, because this
        project's search spawns an evaluation pool whose children outlive a
        signal sent to the parent alone -- measured at two hours and a gigabyte.
        """
