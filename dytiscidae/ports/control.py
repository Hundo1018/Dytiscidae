"""The ``ControlChannel`` port: how a stop reaches a process that is already running.

The application and the worker are different processes, by design.  So a pause
or a cancel has to cross a process boundary, and how it crosses decides what
happens when things are not tidy.

Three mechanisms were available and the choice is not arbitrary:

**Signals.**  Needs the pid, which means tracking it; does not survive the
application restarting; and in this project's own operating notes, ``pkill -f``
matched the shell that ran it and killed that instead.  Kept as a *secondary*
path -- the worker treats SIGTERM as "the container is going away, pause" --
but not as the way a user's cancel is delivered.

**A queue or a socket.**  Correct, and it adds a broker to a project whose
runs are started with ``setsid nohup``.

**A file the worker reads at its checkpoint boundary.**  Chosen.  It survives
the application exiting, it survives the worker restarting, it needs nothing
running, it is inspectable with ``cat``, and the boundary it is read at is
already the point where the on-disk state is current -- so honouring it costs
nothing beyond the generation in flight.

The cost is latency: a request is honoured at the next boundary, which for this
project's search is up to one generation, ~74 s in steady state and ~300 s
during the opening verification burst.  That is why ``PAUSING`` and
``CANCELLING`` are states in the domain rather than instants.  The latency is
real and the design makes it visible instead of hiding it behind a call that
appears to have taken effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..domain.ids import JobId
from ..domain.job import StopReason


@dataclass(frozen=True)
class ControlSignal:
    """A pending instruction for a running job."""

    reason: StopReason
    requested_at: float = 0.0
    urgent: bool = False
    requested_by: str = ""

    def as_dict(self) -> dict:
        return {"reason": self.reason.value, "requested_at": self.requested_at,
                "urgent": self.urgent, "requested_by": self.requested_by}

    @classmethod
    def from_dict(cls, d) -> "ControlSignal":
        return cls(reason=StopReason(d["reason"]),
                   requested_at=float(d.get("requested_at", 0.0)),
                   urgent=bool(d.get("urgent", False)),
                   requested_by=d.get("requested_by", ""))


@runtime_checkable
class ControlChannel(Protocol):
    def request(self, job_id: JobId, signal: ControlSignal) -> None:
        """Post an instruction.  Returns as soon as it is durable, not honoured.

        A second request replaces the first only when it is stronger: a cancel
        overrides a pending pause, a pause does not override a pending cancel.
        Otherwise a user who asked to stop a run and then asked again, less
        firmly, would have downgraded their own instruction.
        """

    def poll(self, job_id: JobId) -> ControlSignal | None:
        """The pending instruction, or None.  Called by the worker; must be cheap.

        Cheap because it is called at every boundary of a run that may last a
        day.  An implementation that opens a network connection here has made
        every generation depend on that network being up.
        """

    def clear(self, job_id: JobId) -> None:
        """Drop the pending instruction.  Called once it has been honoured.

        Idempotent.  Without it, a job paused by a request and then resumed
        would read the same request again and pause immediately -- which is
        what the first version of this did, and it looked like a resume that
        silently failed.
        """
