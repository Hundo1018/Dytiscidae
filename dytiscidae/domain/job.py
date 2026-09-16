"""``TrainingJob``: one execution attempt, with a lifecycle.

This is the piece that separates an AI training application from a CRUD one.
A job runs for seconds or for days, it can be paused, resumed and cancelled,
and it can fail -- so every one of those has to be a *state someone can read*,
not a Python exception that vanished when the process did.

The lifecycle::

                    worker
                    starts        trainer returns
      PENDING ─────────────► RUNNING ───────────────► SUCCEEDED
         ▲   │                 │  │
         │   │ cancel          │  │ raises
         │   ▼                 │  └──────────────────► FAILED
         │ CANCELLED ◄─────────┤                         │
         │       ▲             │ pause requested         │
         │       │             ▼                         │
         │       │          PAUSING ──► PAUSED           │
         │       │             │           │             │
         │       │ cancel      ▼           │ resume      │ resume
         │       └────── CANCELLING        │             │
         │                                 │             │
         └─────────────────────────────────┴─────────────┘
                     prepare_resume, with a checkpoint

Two facts the diagram hides, both of them load-bearing:

**PAUSING and CANCELLING are real states, not instants.**  The request is made
by one process and honoured by another, at the next checkpoint boundary, which
for this project's search is up to a generation away -- ~74 s in steady state,
~300 s during the opening verification burst.  Collapsing the request and the
acknowledgement into one transition means the caller cannot tell "asked to
stop" from "stopped", and the only way to find out is to look at whether a
process still exists.  That is what the PID-hunting in CLAUDE.md is: a
workaround for a state that was never modelled.

**FAILED is resumable.**  A run that died on an out-of-memory at generation 611
has 611 generations of paid-for work on disk.  Refusing to resume it would
throw that away to keep the state machine tidy.  What is *not* allowed is
resuming without a checkpoint -- that is a restart, and it is a different
operation with a different job id, so that the record does not claim continuity
it does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from .errors import IllegalTransition
from .ids import ExperimentId, JobId
from .plan import TrainingPlan


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    #: A stop has been requested and not yet acknowledged by the runtime.
    PAUSING = "pausing"
    #: Stopped with a current checkpoint.  Resumable, and nothing is running.
    PAUSED = "paused"
    #: A cancel has been requested and not yet acknowledged.
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    def __str__(self) -> str:                       # nicer in f-strings and logs
        return self.value


#: A worker is, or should be, alive for these.
ACTIVE_STATUSES = frozenset({JobStatus.RUNNING, JobStatus.PAUSING, JobStatus.CANCELLING})
#: Nothing further happens on its own for these.
TERMINAL_STATUSES = frozenset({JobStatus.CANCELLED, JobStatus.SUCCEEDED, JobStatus.FAILED})
#: Stopped, but with work on disk that a resume can pick up.
RESUMABLE_STATUSES = frozenset({JobStatus.PAUSED, JobStatus.FAILED})

_ALLOWED: dict[JobStatus, frozenset] = {
    JobStatus.PENDING: frozenset({JobStatus.RUNNING, JobStatus.CANCELLED,
                                  JobStatus.FAILED}),
    JobStatus.RUNNING: frozenset({JobStatus.PAUSING, JobStatus.CANCELLING,
                                  JobStatus.PAUSED, JobStatus.SUCCEEDED,
                                  JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.PAUSING: frozenset({JobStatus.PAUSED, JobStatus.SUCCEEDED,
                                  JobStatus.FAILED, JobStatus.CANCELLING,
                                  JobStatus.CANCELLED}),
    # Back to PENDING, not straight to RUNNING.  A resume is a new attempt and
    # goes through the same door a first attempt does: the application arms the
    # job, and the *worker* declares it running once it is actually alive.  The
    # gap between those two is where ``import torch`` fails.
    JobStatus.PAUSED: frozenset({JobStatus.PENDING, JobStatus.CANCELLED}),
    JobStatus.CANCELLING: frozenset({JobStatus.CANCELLED, JobStatus.FAILED}),
    # Terminal.  A resume of a FAILED job is the one exception, and it is spelled
    # out in ``prepare_resume`` rather than hidden in this table, because the
    # table is what a reader trusts for "can this move".
    JobStatus.CANCELLED: frozenset(),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset({JobStatus.PENDING}),
}


class StopReason(str, Enum):
    """Why a run stopped.  Recorded even when it stopped normally."""

    COMPLETED = "completed"           # the budget was spent
    PAUSE_REQUESTED = "pause_requested"
    CANCEL_REQUESTED = "cancel_requested"
    BUDGET_SECONDS = "budget_seconds"
    BUDGET_MEMORY = "budget_memory"
    SIGNAL = "signal"                 # SIGTERM: the container is going away
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class FailureInfo:
    """What went wrong, in enough detail to act on without the log.

    The traceback is kept because a failure whose traceback is only in a log
    file that is only on a container that has since been reclaimed is a failure
    nobody can diagnose -- which is the normal case in the environment this
    project runs in.
    """

    kind: str                  # the exception class name
    message: str
    traceback: str = ""
    step: int | None = None
    #: True where a retry has a chance: an OOM, a transient device error.  Set
    #: by whoever classifies the failure, defaulting to False so that "unknown"
    #: never silently triggers an automatic retry loop.
    retryable: bool = False

    def as_dict(self) -> dict:
        return {"kind": self.kind, "message": self.message,
                "traceback": self.traceback, "step": self.step,
                "retryable": self.retryable}

    @classmethod
    def from_dict(cls, d: Mapping) -> "FailureInfo":
        return cls(kind=d.get("kind", "Error"), message=d.get("message", ""),
                   traceback=d.get("traceback", ""), step=d.get("step"),
                   retryable=bool(d.get("retryable", False)))

    @classmethod
    def from_exception(cls, exc: BaseException, *, step: int | None = None,
                       retryable: bool = False) -> "FailureInfo":
        import traceback as _tb
        return cls(kind=type(exc).__name__, message=str(exc),
                   traceback="".join(_tb.format_exception(type(exc), exc,
                                                          exc.__traceback__)),
                   step=step, retryable=retryable)


@dataclass(frozen=True)
class TrainingJob:
    """A plan, an experiment it belongs to, and where it has got to.

    Frozen, and every transition returns a new job.  A store that writes the
    returned value has written a complete state; there is no window in which a
    job is half-transitioned, which matters because two processes read this --
    the application and the worker.
    """

    job_id: JobId
    experiment_id: ExperimentId
    plan: TrainingPlan
    status: JobStatus = JobStatus.PENDING
    created_at: float = 0.0
    started_at: float | None = None
    ended_at: float | None = None
    #: How many times this job has entered RUNNING.  A resume is an attempt, so
    #: attempts > 1 means the run has a boundary in it -- and this project has
    #: measured that a boundary crossed without the RNG state costs every score
    #: written before it.
    attempts: int = 0
    #: Where the runtime writes its artefacts.  On the job rather than the plan:
    #: the same plan run twice is two directories and one configuration.
    workspace: str = ""
    #: Throughput settings for *this* execution -- worker count, shard size,
    #: device.  Not in the plan, so the same experiment on two machines still
    #: compares as one configuration.
    resources: Mapping[str, Any] = field(default_factory=dict)
    #: The checkpoint this attempt started from, where it started from one.
    resumed_from: str | None = None
    stop_reason: StopReason | None = None
    failure: FailureInfo | None = None
    #: Set by the runtime once it is alive, so a job claiming RUNNING with no
    #: heartbeat for an hour can be told from one that is working.
    worker: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, JobId):
            object.__setattr__(self, "job_id", JobId(self.job_id))
        if not isinstance(self.experiment_id, ExperimentId):
            object.__setattr__(self, "experiment_id", ExperimentId(self.experiment_id))
        if not isinstance(self.status, JobStatus):
            object.__setattr__(self, "status", JobStatus(self.status))
        object.__setattr__(self, "resources", dict(self.resources))
        object.__setattr__(self, "worker", dict(self.worker))

    # -- queries ----------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def is_resumable(self) -> bool:
        return self.status in RESUMABLE_STATUSES

    @property
    def stop_requested(self) -> bool:
        return self.status in (JobStatus.PAUSING, JobStatus.CANCELLING)

    def can(self, to: JobStatus) -> bool:
        return to in _ALLOWED[self.status]

    def _to(self, to: JobStatus, detail: str = "", **changes) -> "TrainingJob":
        if to not in _ALLOWED[self.status]:
            raise IllegalTransition(self.job_id, str(self.status), str(to), detail)
        return replace(self, status=to, **changes)

    # -- transitions ------------------------------------------------------

    def start(self, *, now: float, worker: Mapping[str, Any] | None = None,
              resumed_from: str | None = None) -> "TrainingJob":
        """PENDING -> RUNNING.  The runtime calls this, not the caller.

        Deliberately distinct from ``StartTraining``: the use case launches a
        worker and the job stays PENDING until that worker says it is alive.
        A job that went RUNNING at launch would report RUNNING for a worker that
        died in ``import torch``, which this project has had happen.
        """
        # Merged, not replaced.  Two parties write this record: the launcher
        # puts the handle on it before the worker exists, and the worker adds
        # its pid and host once it does.  A ``start`` that replaced the dict
        # would drop the handle, and then a forced cancel has no way to find
        # the process it is meant to kill -- which is the one case where
        # finding it matters.
        merged = dict(self.worker or {})
        merged.update(worker or {})
        return self._to(
            JobStatus.RUNNING,
            started_at=self.started_at if self.started_at is not None else now,
            attempts=self.attempts + 1,
            worker=merged,
            resumed_from=resumed_from if resumed_from is not None else self.resumed_from,
            stop_reason=None,
            failure=None,
        )

    def request_pause(self) -> "TrainingJob":
        """RUNNING -> PAUSING.  A request, not an event: see the module docstring."""
        if self.status is JobStatus.PAUSING:
            return self                                   # idempotent on purpose
        return self._to(JobStatus.PAUSING, "only a running job can be paused")

    def request_cancel(self) -> "TrainingJob":
        """-> CANCELLING, or straight to CANCELLED where nothing is running.

        A PENDING or PAUSED job has no runtime to acknowledge anything, so
        making the caller wait for an acknowledgement that can never come would
        strand it.  It goes terminal here.
        """
        if self.status in (JobStatus.CANCELLING, JobStatus.CANCELLED):
            return self
        if self.status in (JobStatus.PENDING, JobStatus.PAUSED):
            return self._to(JobStatus.CANCELLED)
        return self._to(JobStatus.CANCELLING, "only a live job can be cancelled")

    def paused(self, *, now: float,
               reason: StopReason = StopReason.PAUSE_REQUESTED) -> "TrainingJob":
        """The runtime acknowledging: stopped, checkpoint current, resumable."""
        return self._to(JobStatus.PAUSED, ended_at=now, stop_reason=reason,
                        worker={})

    def cancelled(self, *, now: float) -> "TrainingJob":
        return self._to(JobStatus.CANCELLED, ended_at=now,
                        stop_reason=StopReason.CANCEL_REQUESTED, worker={})

    def succeeded(self, *, now: float,
                  reason: StopReason = StopReason.COMPLETED) -> "TrainingJob":
        return self._to(JobStatus.SUCCEEDED, ended_at=now, stop_reason=reason,
                        worker={})

    def failed(self, failure: FailureInfo, *, now: float) -> "TrainingJob":
        return self._to(JobStatus.FAILED, ended_at=now, stop_reason=StopReason.ERROR,
                        failure=failure, worker={})

    def prepare_resume(self, *, now: float, checkpoint: str | None,
                       plan: TrainingPlan | None = None,
                       resources: Mapping[str, Any] | None = None) -> "TrainingJob":
        """PAUSED or FAILED -> PENDING, armed to continue from a checkpoint.

        PENDING rather than RUNNING, so that a resume goes through exactly the
        same door a first start does: the application arms, the worker declares
        itself alive with ``start``.  Anything else would make a resumed job
        report RUNNING for a worker that died before it existed -- and on a
        resume that is worse than on a start, because the job already has hours
        of real work behind it and the false RUNNING hides that it has stopped.

        The checkpoint is required.  A resume without one is a restart wearing
        a resume's name, and it would produce a record claiming 900 generations
        of continuity across a boundary where the network was reinitialised --
        which is the exact defect ``ops/checkpoint.py`` was written to close.
        Callers that want a restart create a new job.

        ``plan`` is the one place a job's configuration may change after it is
        created, and it exists for one case: a resume of a job that spent its
        step budget needs the budget raised or the worker starts and stops
        immediately, which is indistinguishable from a resume that did not
        work.  It is on ``resume`` rather than offered as a general "edit the
        plan", because a plan that can be edited freely is how two runs come to
        share a digest neither of them earned.  The new plan's digest is
        different, and the job records it -- an extended run does not claim to
        be a rerun of the original configuration.

        ``resources`` may change freely: worker count and shard size are
        properties of the machine this attempt runs on, not of the experiment.
        """
        if self.status not in RESUMABLE_STATUSES:
            raise IllegalTransition(self.job_id, str(self.status), "pending",
                                    "only a paused or failed job can be resumed")
        if not checkpoint:
            raise IllegalTransition(
                self.job_id, str(self.status), "pending",
                "no checkpoint to resume from; start a new job instead of "
                "silently restarting this one")
        if plan is not None and plan.trainer != self.plan.trainer:
            raise IllegalTransition(
                self.job_id, str(self.status), "pending",
                f"cannot resume a {self.plan.trainer!r} job as {plan.trainer!r}: "
                f"the checkpoint was written by a different trainer")
        if plan is not None and plan.seed != self.plan.seed:
            raise IllegalTransition(
                self.job_id, str(self.status), "pending",
                f"cannot change the seed on resume ({self.plan.seed} -> "
                f"{plan.seed}): the stored random state is the one the original "
                f"seed produced, so the new seed would have no effect and the "
                f"record would say otherwise")
        return replace(self, status=JobStatus.PENDING, ended_at=None,
                       resumed_from=str(checkpoint),
                       stop_reason=None, failure=None, worker={},
                       plan=self.plan if plan is None else plan,
                       resources=dict(self.resources if resources is None
                                      else resources))

    def heartbeat(self, *, now: float, **info) -> "TrainingJob":
        """Refresh the worker's liveness record without changing status."""
        w = dict(self.worker)
        w.update(info)
        w["heartbeat_at"] = now
        return replace(self, worker=w)

    # -- serialisation ----------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "job_id": str(self.job_id),
            "experiment_id": str(self.experiment_id),
            "plan": self.plan.as_dict(),
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "attempts": self.attempts,
            "workspace": self.workspace,
            "resources": dict(self.resources),
            "resumed_from": self.resumed_from,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "failure": self.failure.as_dict() if self.failure else None,
            "worker": dict(self.worker),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "TrainingJob":
        f = d.get("failure")
        sr = d.get("stop_reason")
        return cls(
            job_id=JobId(d["job_id"]),
            experiment_id=ExperimentId(d["experiment_id"]),
            plan=TrainingPlan.from_dict(d["plan"]),
            status=JobStatus(d.get("status", "pending")),
            created_at=float(d.get("created_at", 0.0)),
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
            attempts=int(d.get("attempts", 0)),
            workspace=d.get("workspace", ""),
            resources=dict(d.get("resources") or {}),
            resumed_from=d.get("resumed_from"),
            stop_reason=StopReason(sr) if sr else None,
            failure=FailureInfo.from_dict(f) if f else None,
            worker=dict(d.get("worker") or {}),
        )

    def __str__(self) -> str:
        return f"{self.job_id} [{self.status}] {self.plan}"
