"""The ``CheckpointStore`` port."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..domain.checkpoint import CheckpointKind, CheckpointPayload, CheckpointRecord
from ..domain.ids import CheckpointId, ExperimentId, JobId


@runtime_checkable
class CheckpointStore(Protocol):
    """Round-trips a checkpoint's bytes, and knows what it has.

    ``save`` takes the payload and the facts about it; the store mints the id,
    computes the digest and the size, and returns the record.  The caller does
    not get to choose the id, because an id chosen by the writer is an id that
    collides across two workers writing the same step of the same job -- which
    is exactly what happens when a launch is retried and the first worker has
    not noticed yet.
    """

    def save(self, *, job_id: JobId, experiment_id: ExperimentId, step: int,
             payload: CheckpointPayload,
             kind: CheckpointKind = CheckpointKind.PERIODIC,
             metrics: Mapping[str, float] | None = None,
             provenance: Mapping[str, Any] | None = None,
             created_at: float = 0.0) -> CheckpointRecord:
        """Store it durably and return its record.

        Durably means: a reader that sees the record can read the payload.  An
        implementation writing to a filesystem does that with a temporary file
        and a rename, because a half-written checkpoint that *looks* complete is
        worse than no checkpoint -- the resume succeeds and the run is wrong.
        """

    def load(self, checkpoint_id: CheckpointId
             ) -> tuple[CheckpointRecord, CheckpointPayload]:
        """Fetch a checkpoint.  Raises ``CheckpointNotFound`` if it is not there.

        Verifies the digest.  A payload that does not hash to its record's
        digest is corruption, and it is raised rather than returned: a resume
        from a truncated checkpoint produces a run that is subtly wrong and
        reports nothing, which is the most expensive failure in this list.
        """

    def get(self, checkpoint_id: CheckpointId) -> CheckpointRecord:
        """The record alone, without fetching the payload."""

    def latest(self, job_id: JobId, *, kind: CheckpointKind | None = None
               ) -> CheckpointRecord | None:
        """The highest-step checkpoint for a job, or None.

        Highest step, not most recently written: a store holding a step-600
        checkpoint and a step-50 one written afterwards by a restart that was
        then abandoned must return the 600.
        """

    def list(self, job_id: JobId | None = None, *,
             experiment_id: ExperimentId | None = None) -> Sequence[CheckpointRecord]:
        """Records for a job or an experiment, ordered by step ascending."""

    def delete(self, checkpoint_id: CheckpointId) -> None:
        """Remove one.  Idempotent: deleting what is not there is not an error."""

    def prune(self, job_id: JobId, *, keep_last: int = 3) -> Sequence[CheckpointId]:
        """Drop old PERIODIC checkpoints, returning what was removed.

        Never touches BEST or FINAL.  "Keep the last N" on its own deletes the
        best one, which is why ``CheckpointKind`` exists.
        """
