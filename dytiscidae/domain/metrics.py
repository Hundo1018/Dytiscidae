"""Metrics and events as first-class data.

The rule this file exists to enforce: a number the run produced is data, not a
side effect of a print.  ``ops/telemetry.py`` already applies it -- one JSON
object per line, three streams, readable with ``tail -f`` while the run is
going -- and this is the domain-level shape of the same thing, so that the
telemetry writer becomes one adapter among several rather than the only way to
record anything.

Two records, kept apart for the reason the telemetry streams are kept apart:

``MetricPoint``  a number at a step.  Many per step, sampled where noisy.
``JobEvent``     something that happened to the job.  Few, never sampled.

Sampling a metric loses resolution.  Sampling an event loses the event, and the
events here are the ones that answer "why did this stop" six weeks later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .ids import JobId


@dataclass(frozen=True)
class MetricPoint:
    """One scalar, at one step, with the tags needed to read it correctly.

    ``tags`` is not decoration.  This project's own telemetry has a documented
    trap: ``filled``, ``coverage`` and ``qd_score`` in a generation line belong
    to whichever island that generation visited, so consecutive lines are
    different archives and plotting them as a series is a category error.  A
    metric carrying ``{"island": "amphibian"}`` cannot be mis-plotted that way;
    one that does not, will be.
    """

    name: str
    value: float
    step: int
    job_id: JobId | None = None
    #: Seconds since the run started.  Kept beside ``step`` because steps are
    #: not uniform in time -- this project's generations 0-5 cost ~300 s each
    #: against a steady state of ~74 s.
    wall_seconds: float = 0.0
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("metric name must be non-empty")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "tags", {str(k): str(v) for k, v in self.tags.items()})
        if self.job_id is not None and not isinstance(self.job_id, JobId):
            object.__setattr__(self, "job_id", JobId(self.job_id))

    def as_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "step": self.step,
                "job_id": str(self.job_id) if self.job_id else None,
                "wall_seconds": round(self.wall_seconds, 3),
                "tags": dict(self.tags)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "MetricPoint":
        jid = d.get("job_id")
        return cls(name=d["name"], value=float(d["value"]), step=int(d.get("step", 0)),
                   job_id=JobId(jid) if jid else None,
                   wall_seconds=float(d.get("wall_seconds", 0.0)),
                   tags=dict(d.get("tags") or {}))


class JobEventKind(str, Enum):
    CREATED = "created"
    LAUNCHED = "launched"
    STARTED = "started"
    PROGRESS = "progress"
    CHECKPOINT_WRITTEN = "checkpoint_written"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    RESUMED = "resumed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    #: The worker said something that is not a state change -- a degraded
    #: device, a fallback taken.  Worth recording precisely because it is the
    #: class of thing that otherwise only appears in a log nobody kept.
    NOTE = "note"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class JobEvent:
    """Something that happened to a job, in the order it happened.

    Append-only.  The sequence is the job's history and is the thing that can
    answer why a status is what it is -- ``FAILED`` alone says nothing about
    whether it failed at step 3 or step 611, and the difference is four hours
    of work.
    """

    job_id: JobId
    kind: JobEventKind
    at: float = 0.0
    step: int | None = None
    message: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, JobId):
            object.__setattr__(self, "job_id", JobId(self.job_id))
        if not isinstance(self.kind, JobEventKind):
            object.__setattr__(self, "kind", JobEventKind(self.kind))
        object.__setattr__(self, "detail", dict(self.detail))

    def as_dict(self) -> dict:
        return {"job_id": str(self.job_id), "kind": self.kind.value, "at": self.at,
                "step": self.step, "message": self.message, "detail": dict(self.detail)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "JobEvent":
        return cls(job_id=JobId(d["job_id"]), kind=JobEventKind(d["kind"]),
                   at=float(d.get("at", 0.0)), step=d.get("step"),
                   message=d.get("message", ""), detail=dict(d.get("detail") or {}))

    def __str__(self) -> str:
        at = f"step {self.step}" if self.step is not None else "-"
        return f"[{self.kind}] {self.job_id} {at} {self.message}".rstrip()
