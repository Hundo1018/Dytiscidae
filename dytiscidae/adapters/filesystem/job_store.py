"""``JobStore`` over a directory tree.

Layout, with the job's own directory doubling as its workspace so that
everything about one attempt is in one place and can be tarred, moved or
deleted as a unit::

    <root>/jobs/<job_id>/job.json        lifecycle state
    <root>/jobs/<job_id>/state.json      runtime state, rewritten every step
    <root>/jobs/<job_id>/job.lock        the update lock
    <root>/jobs/<job_id>/events.jsonl    the event log (a different adapter)
    <root>/jobs/<job_id>/metrics.jsonl   metrics (a different adapter)
    <root>/jobs/<job_id>/checkpoints/    checkpoints (a different adapter)

``update`` takes the lock, re-reads, applies, writes, releases.  The re-read is
the part that matters: the caller's function is applied to what is *stored*, not
to whatever the caller happened to be holding, so a worker heartbeat and an
application pause arriving together compose instead of one overwriting the other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from ...domain.errors import JobNotFound
from ...domain.ids import ExperimentId, JobId
from ...domain.job import JobStatus, TrainingJob
from ...domain.state import TrainingState
from ._io import FileLock, atomic_write_json, read_json


class FileJobStore:
    """The filesystem implementation of ``ports.JobStore``."""

    def __init__(self, root: str | Path, *, lock_timeout: float = 30.0) -> None:
        self.root = Path(root)
        self._lock_timeout = lock_timeout

    # -- layout -----------------------------------------------------------

    def job_dir(self, job_id: JobId) -> Path:
        return self.root / "jobs" / str(job_id)

    def _job_path(self, job_id: JobId) -> Path:
        return self.job_dir(job_id) / "job.json"

    def _state_path(self, job_id: JobId) -> Path:
        return self.job_dir(job_id) / "state.json"

    def _lock(self, job_id: JobId) -> FileLock:
        return FileLock(self.job_dir(job_id) / "job.lock", timeout=self._lock_timeout)

    # -- ports.JobStore ---------------------------------------------------

    def create(self, job: TrainingJob) -> TrainingJob:
        path = self._job_path(job.job_id)
        if path.exists():
            raise FileExistsError(f"job {job.job_id} already exists at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, job.as_dict())
        return job

    def get(self, job_id: JobId) -> TrainingJob:
        d = read_json(self._job_path(JobId(job_id)))
        if d is None:
            raise JobNotFound(f"no job {job_id} under {self.root}")
        return TrainingJob.from_dict(d)

    def update(self, job_id: JobId,
               change: Callable[[TrainingJob], TrainingJob]) -> TrainingJob:
        job_id = JobId(job_id)
        with self._lock(job_id):
            d = read_json(self._job_path(job_id))
            if d is None:
                raise JobNotFound(f"no job {job_id} under {self.root}")
            updated = change(TrainingJob.from_dict(d))
            if updated.job_id != job_id:
                raise ValueError(
                    f"the change returned job {updated.job_id}, not {job_id}")
            atomic_write_json(self._job_path(job_id), updated.as_dict())
            return updated

    def save_state(self, state: TrainingState) -> None:
        # No lock.  One writer -- the worker -- and a lost write costs one step
        # of progress reporting.  Taking the job lock every step would put the
        # progress report on the same contended path as the lifecycle.
        atomic_write_json(self._state_path(state.job_id), state.as_dict())

    def load_state(self, job_id: JobId) -> TrainingState | None:
        d = read_json(self._state_path(JobId(job_id)))
        return TrainingState.from_dict(d) if d else None

    def list(self, *, experiment_id: ExperimentId | None = None,
             status: JobStatus | None = None,
             limit: int | None = None) -> Sequence[TrainingJob]:
        base = self.root / "jobs"
        if not base.is_dir():
            return []
        out: list[TrainingJob] = []
        # Reverse lexical order is newest first, because ``new_id`` puts a
        # UTC timestamp in the middle of the id.  No stat calls, no clock.
        for d in sorted(base.iterdir(), key=lambda p: p.name, reverse=True):
            payload = read_json(d / "job.json")
            if payload is None:
                continue
            try:
                job = TrainingJob.from_dict(payload)
            except Exception:                                     # noqa: BLE001
                # A job written by an older layout is skipped rather than
                # raised on: one unreadable row must not make the whole listing
                # fail, which is what a caller hits when they are trying to
                # find out what went wrong.
                continue
            if experiment_id is not None and job.experiment_id != experiment_id:
                continue
            if status is not None and job.status is not JobStatus(status):
                continue
            out.append(job)
            if limit is not None and len(out) >= limit:
                break
        return out

    def delete(self, job_id: JobId) -> None:
        import shutil
        shutil.rmtree(self.job_dir(JobId(job_id)), ignore_errors=True)
