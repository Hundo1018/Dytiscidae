"""The ``JobStore`` port: lifecycle state, written by one process and read by another.

This is the port where concurrency is real.  The application asks for a pause;
the worker, in another process, acknowledges it.  Both write the same job.  So
the store has to offer something better than read-modify-write, and it does:

``update`` takes a *function*, applies it to the current stored job, and stores
the result -- atomically with respect to other callers of the same store.  A
caller that reads a job, transitions it and writes it back has a window in
which the worker's heartbeat is lost; a caller that passes the transition in
does not.

``save_state`` is separate from ``update`` because the two have different
frequencies and different durability needs.  The lifecycle status changes a
handful of times in a run and must never be lost.  The runtime state changes
every step, and losing the last one costs nothing because the next arrives in
74 seconds.  Giving them one method would make every progress report pay the
lifecycle's durability cost.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, runtime_checkable

from ..domain.ids import ExperimentId, JobId
from ..domain.job import JobStatus, TrainingJob
from ..domain.state import TrainingState


@runtime_checkable
class JobStore(Protocol):
    def create(self, job: TrainingJob) -> TrainingJob:
        """Store a new job.  Raises if the id exists."""

    def get(self, job_id: JobId) -> TrainingJob:
        """Raises ``JobNotFound`` when it is not there."""

    def update(self, job_id: JobId,
               change: Callable[[TrainingJob], TrainingJob]) -> TrainingJob:
        """Apply ``change`` to the stored job atomically and return the result.

        ``change`` must be a pure function of the job: it may be called more
        than once if the store retries, so anything with a side effect in it --
        an event write, a print -- will happen more than once too.  Put those
        after the call, on the returned job.

        A ``change`` that raises ``IllegalTransition`` leaves the store
        untouched and the exception propagates.  That is the mechanism by which
        two concurrent cancels produce one cancellation.
        """

    def save_state(self, state: TrainingState) -> None:
        """Record the runtime state.  Last write wins; losing one is acceptable."""

    def load_state(self, job_id: JobId) -> TrainingState | None:
        """The last recorded runtime state, or None if none was recorded."""

    def list(self, *, experiment_id: ExperimentId | None = None,
             status: JobStatus | None = None,
             limit: int | None = None) -> Sequence[TrainingJob]:
        """Jobs, newest first."""

    def delete(self, job_id: JobId) -> None:
        """Remove a job and its state.  Idempotent."""
