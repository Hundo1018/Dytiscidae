"""The ``EventLog`` port: a job's history, append-only."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..domain.ids import JobId
from ..domain.metrics import JobEvent


@runtime_checkable
class EventLog(Protocol):
    """Append-only.  There is no update and no delete, and that is the point.

    A job's status says where it is.  Only the event sequence says how it got
    there, and an event sequence that can be rewritten answers the question
    "why did this stop" with whatever the last writer preferred.
    """

    def append(self, event: JobEvent) -> None:
        """Record one event.  Must not raise: losing a run to a log is absurd."""

    def read(self, job_id: JobId, *, limit: int | None = None) -> Sequence[JobEvent]:
        """A job's events, oldest first.  ``limit`` keeps the *last* N.

        The last N rather than the first: the interesting end of a run that
        failed at generation 611 is generation 611.
        """
