"""``LoadCheckpoint`` and ``ExportModel``.

The two uses of a checkpoint that are not "resume the run that wrote it", and
the reason they are use cases rather than a call into the store:

``LoadCheckpoint`` resolves "the latest one of this job, of this kind" -- which
is a policy question, not a storage one -- and it verifies before returning.

``ExportModel`` has to be dispatched back to the trainer.  The payload is a bag
of named blobs and the only thing that knows what they mean is whatever wrote
them; an exporter here would have to guess, and a guess that is wrong produces
a file that loads and is not the model.  So the trainer declares ``exports`` in
its capabilities and implements ``export`` , and this use case's job is to find
the checkpoint, check the destination, and refuse clearly when the trainer
cannot do it.
"""

from __future__ import annotations

from typing import Callable, Protocol

from ..domain.checkpoint import CheckpointPayload, CheckpointRecord
from ..domain.errors import CheckpointNotFound, DomainError
from ..domain.ids import CheckpointId, JobId
from ..ports.checkpoint_store import CheckpointStore
from ..ports.job_store import JobStore
from ..ports.trainer import Trainer
from .dto import (
    ExportModelRequest,
    ExportModelResponse,
    LoadCheckpointRequest,
    LoadCheckpointResponse,
)


class ExportNotSupported(DomainError):
    pass


class ModelExporter(Protocol):
    """What a trainer must also be, to be exportable.

    A separate protocol rather than more methods on ``Trainer``, so that a
    trainer which only trains is not forced to carry a stub that raises.  The
    use case checks for it by ``hasattr``, which is what structural typing
    amounts to at runtime, and says so plainly when it is missing.
    """

    def export(self, record: CheckpointRecord, payload: CheckpointPayload, *,
               destination: str, fmt: str) -> dict:
        """Write the model to ``destination``.  Returns detail for the record."""


class LoadCheckpoint:
    """Fetch a checkpoint by id, or the latest of a job."""

    def __init__(self, checkpoints: CheckpointStore, jobs: JobStore | None = None
                 ) -> None:
        self._checkpoints = checkpoints
        self._jobs = jobs

    def execute(self, request: LoadCheckpointRequest) -> LoadCheckpointResponse:
        record = self._resolve(request)
        if not request.with_payload:
            return LoadCheckpointResponse(record=record)
        record, payload = self._checkpoints.load(record.checkpoint_id)
        return LoadCheckpointResponse(record=record, payload=payload)

    def _resolve(self, request: LoadCheckpointRequest) -> CheckpointRecord:
        if (request.checkpoint_id is None) == (request.job_id is None):
            raise ValueError("give exactly one of checkpoint_id or job_id")
        if request.checkpoint_id is not None:
            return self._checkpoints.get(CheckpointId(request.checkpoint_id))
        record = self._checkpoints.latest(JobId(request.job_id), kind=request.kind)
        if record is None:
            kind = f" of kind {request.kind}" if request.kind else ""
            raise CheckpointNotFound(
                f"job {request.job_id} has no checkpoint{kind}")
        return record


class ExportModel:
    """Turn a checkpoint into an artefact usable outside this codebase."""

    def __init__(self, checkpoints: CheckpointStore, jobs: JobStore,
                 resolve_trainer: Callable[[str], Trainer]) -> None:
        self._checkpoints = checkpoints
        self._jobs = jobs
        # By name, resolved by the composition root.  The application layer
        # importing a trainer would be the application layer importing torch.
        self._resolve_trainer = resolve_trainer

    def execute(self, request: ExportModelRequest) -> ExportModelResponse:
        loader = LoadCheckpoint(self._checkpoints, self._jobs)
        loaded = loader.execute(LoadCheckpointRequest(
            checkpoint_id=request.checkpoint_id, job_id=request.job_id,
            with_payload=True))
        record, payload = loaded.record, loaded.payload or {}

        job = self._jobs.get(record.job_id)
        trainer = self._resolve_trainer(job.plan.trainer)
        caps = trainer.capabilities()
        exporter = getattr(trainer, "export", None)
        if not caps.exports or exporter is None:
            raise ExportNotSupported(
                f"trainer {job.plan.trainer!r} does not export "
                f"(capabilities.exports={caps.exports}, "
                f"export method={'present' if exporter else 'missing'}). "
                f"The checkpoint is still readable with LoadCheckpoint.")

        detail = exporter(record, payload, destination=request.destination,
                          fmt=request.fmt) or {}
        return ExportModelResponse(
            destination=request.destination, record=record, fmt=request.fmt,
            bytes_written=int(detail.get("bytes_written", 0)), detail=detail)
