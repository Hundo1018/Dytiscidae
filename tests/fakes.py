"""In-memory adapters, for testing the application layer without a disk.

These are not simplified versions of the real adapters.  They are the *whole*
implementation of each port, in memory, and keeping them honest is what makes
the use case tests meaningful -- a fake ``JobStore.update`` that does not
re-read before applying the change would let a use case pass here and lose a
heartbeat in production.

They live in ``tests/`` rather than in the package because they are not a
shipped capability.  A caller that wants an in-memory lab in production wants
something durable enough to survive a restart, and these are deliberately not.

The one that carries real weight is ``FakeLauncher``.  It records what it was
asked to do without doing it, which is what lets ``StartTraining`` and
``CancelTraining`` be tested for *the decisions they make* rather than for
whether a process appeared.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dytiscidae.domain.checkpoint import (                         # noqa: E402
    CheckpointKind, CheckpointPayload, CheckpointRecord, digest_payload,
)
from dytiscidae.domain.dataset import Dataset, DatasetKind, DatasetRef  # noqa: E402
from dytiscidae.domain.errors import (                             # noqa: E402
    CheckpointNotFound, DatasetNotFound, ExperimentNotFound, JobNotFound,
)
from dytiscidae.domain.experiment import Experiment                # noqa: E402
from dytiscidae.domain.ids import (                                # noqa: E402
    CheckpointId, ExperimentId, JobId, new_id,
)
from dytiscidae.domain.job import JobStatus, TrainingJob           # noqa: E402
from dytiscidae.domain.metrics import JobEvent, MetricPoint        # noqa: E402
from dytiscidae.domain.state import TrainingState                  # noqa: E402
from dytiscidae.ports.control import ControlSignal                 # noqa: E402
from dytiscidae.ports.launcher import WorkerHandle, WorkerStatus   # noqa: E402


class FakeClock:
    """A clock that only moves when told to.

    Every duration in a job record comes from here, so a test can assert on an
    elapsed time exactly rather than within a tolerance -- and nothing in the
    suite has to sleep to make time pass.
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = float(start)
        self._mono = 0.0

    def now(self) -> float:
        return self.t

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self.t += seconds
        self._mono += seconds


class FakeJobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, TrainingJob] = {}
        self.states: dict[str, TrainingState] = {}
        #: Every version ever stored, for tests that assert on the sequence of
        #: statuses rather than only on the last one.
        self.history: list[TrainingJob] = []

    def create(self, job: TrainingJob) -> TrainingJob:
        if str(job.job_id) in self.jobs:
            raise FileExistsError(f"job {job.job_id} already exists")
        self.jobs[str(job.job_id)] = job
        self.history.append(job)
        return job

    def get(self, job_id: JobId) -> TrainingJob:
        job = self.jobs.get(str(job_id))
        if job is None:
            raise JobNotFound(f"no job {job_id}")
        return job

    def update(self, job_id: JobId,
               change: Callable[[TrainingJob], TrainingJob]) -> TrainingJob:
        # Re-reads before applying, exactly as the real store does.  A fake that
        # applied the change to a job the caller was already holding would hide
        # every lost-update bug the real one is built to prevent.
        updated = change(self.get(job_id))
        if updated.job_id != JobId(job_id):
            raise ValueError("the change returned a different job")
        self.jobs[str(job_id)] = updated
        self.history.append(updated)
        return updated

    def save_state(self, state: TrainingState) -> None:
        self.states[str(state.job_id)] = state

    def load_state(self, job_id: JobId) -> TrainingState | None:
        return self.states.get(str(job_id))

    def list(self, *, experiment_id: ExperimentId | None = None,
             status: JobStatus | None = None,
             limit: int | None = None) -> Sequence[TrainingJob]:
        out = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
        if experiment_id is not None:
            out = [j for j in out if j.experiment_id == experiment_id]
        if status is not None:
            out = [j for j in out if j.status is JobStatus(status)]
        return out[:limit] if limit is not None else out

    def delete(self, job_id: JobId) -> None:
        self.jobs.pop(str(job_id), None)
        self.states.pop(str(job_id), None)

    def statuses(self, job_id: JobId) -> list[str]:
        """The sequence of statuses this job has held, deduplicated in place."""
        out: list[str] = []
        for job in self.history:
            if job.job_id == JobId(job_id) and (not out or out[-1] != job.status.value):
                out.append(job.status.value)
        return out


class FakeExperimentStore:
    def __init__(self) -> None:
        self.experiments: dict[str, Experiment] = {}

    def create(self, experiment: Experiment) -> Experiment:
        if str(experiment.experiment_id) in self.experiments:
            raise FileExistsError(f"experiment {experiment.experiment_id} exists")
        if any(e.name == experiment.name for e in self.experiments.values()):
            raise FileExistsError(f"an experiment named {experiment.name!r} exists")
        self.experiments[str(experiment.experiment_id)] = experiment
        return experiment

    def get(self, experiment_id: ExperimentId) -> Experiment:
        experiment = self.experiments.get(str(experiment_id))
        if experiment is None:
            raise ExperimentNotFound(f"no experiment {experiment_id}")
        return experiment

    def find_by_name(self, name: str) -> Experiment | None:
        for experiment in self.experiments.values():
            if experiment.name == name:
                return experiment
        return None

    def update(self, experiment: Experiment) -> Experiment:
        if str(experiment.experiment_id) not in self.experiments:
            raise ExperimentNotFound(f"no experiment {experiment.experiment_id}")
        self.experiments[str(experiment.experiment_id)] = experiment
        return experiment

    def attach_job(self, experiment_id: ExperimentId, job_id: JobId) -> Experiment:
        updated = self.get(experiment_id).with_job(JobId(job_id))
        self.experiments[str(experiment_id)] = updated
        return updated

    def list(self, *, tag: str | None = None,
             limit: int | None = None) -> Sequence[Experiment]:
        out = sorted(self.experiments.values(), key=lambda e: e.created_at,
                     reverse=True)
        if tag is not None:
            out = [e for e in out if tag in e.tags]
        return out[:limit] if limit is not None else out

    def delete(self, experiment_id: ExperimentId) -> None:
        self.experiments.pop(str(experiment_id), None)


class FakeCheckpointStore:
    def __init__(self) -> None:
        self.records: dict[str, CheckpointRecord] = {}
        self.payloads: dict[str, dict[str, bytes]] = {}

    def save(self, *, job_id: JobId, experiment_id: ExperimentId, step: int,
             payload: CheckpointPayload,
             kind: CheckpointKind = CheckpointKind.PERIODIC,
             metrics: Mapping[str, float] | None = None,
             provenance: Mapping[str, Any] | None = None,
             created_at: float = 0.0) -> CheckpointRecord:
        checkpoint_id = CheckpointId(new_id(f"ck{step:07d}"))
        record = CheckpointRecord(
            checkpoint_id=checkpoint_id, job_id=JobId(job_id),
            experiment_id=ExperimentId(experiment_id), step=int(step),
            kind=CheckpointKind(kind), created_at=created_at,
            size_bytes=sum(len(b) for b in payload.values()),
            digest=digest_payload(payload), metrics=dict(metrics or {}),
            provenance=dict(provenance or {}),
            payload_keys=tuple(sorted(payload)))
        self.records[str(checkpoint_id)] = record
        self.payloads[str(checkpoint_id)] = {k: bytes(v) for k, v in payload.items()}
        return record

    def get(self, checkpoint_id: CheckpointId) -> CheckpointRecord:
        record = self.records.get(str(checkpoint_id))
        if record is None:
            raise CheckpointNotFound(f"no checkpoint {checkpoint_id}")
        return record

    def load(self, checkpoint_id: CheckpointId):
        record = self.get(checkpoint_id)
        payload = self.payloads[str(checkpoint_id)]
        if digest_payload(payload) != record.digest:
            raise CheckpointNotFound(f"checkpoint {checkpoint_id} digest mismatch")
        return record, payload

    def latest(self, job_id: JobId, *, kind: CheckpointKind | None = None
               ) -> CheckpointRecord | None:
        records = [r for r in self.records.values() if r.job_id == JobId(job_id)]
        if kind is not None:
            records = [r for r in records if r.kind is CheckpointKind(kind)]
        return max(records, key=lambda r: (r.step, r.created_at), default=None)

    def list(self, job_id: JobId | None = None, *,
             experiment_id: ExperimentId | None = None) -> Sequence[CheckpointRecord]:
        out = list(self.records.values())
        if job_id is not None:
            out = [r for r in out if r.job_id == JobId(job_id)]
        if experiment_id is not None:
            out = [r for r in out if r.experiment_id == ExperimentId(experiment_id)]
        return sorted(out, key=lambda r: (r.step, r.created_at))

    def delete(self, checkpoint_id: CheckpointId) -> None:
        self.records.pop(str(checkpoint_id), None)
        self.payloads.pop(str(checkpoint_id), None)

    def prune(self, job_id: JobId, *, keep_last: int = 3) -> Sequence[CheckpointId]:
        periodic = [r for r in self.list(job_id) if r.kind is CheckpointKind.PERIODIC]
        doomed = periodic[:-keep_last] if keep_last else list(periodic)
        for record in doomed:
            self.delete(record.checkpoint_id)
        return [r.checkpoint_id for r in doomed]


class FakeDatasetRepository:
    def __init__(self) -> None:
        self.datasets: dict[str, Dataset] = {}
        self.items: dict[str, list[dict]] = {}

    def get(self, ref: DatasetRef) -> Dataset:
        dataset = self.datasets.get(str(ref))
        if dataset is None:
            raise DatasetNotFound(f"no dataset {ref}")
        return dataset

    def open(self, ref: DatasetRef) -> Iterator[Mapping[str, Any]]:
        self.get(ref)
        return iter(list(self.items.get(str(ref), [])))

    def list(self, *, kind: DatasetKind | None = None) -> Sequence[Dataset]:
        out = list(self.datasets.values())
        if kind is not None:
            out = [d for d in out if d.kind is DatasetKind(kind)]
        return out

    def put(self, ref: DatasetRef, items: Iterable[Mapping[str, Any]], *,
            kind: DatasetKind = DatasetKind.GENOMES, description: str = "",
            metadata: Mapping[str, Any] | None = None) -> Dataset:
        if str(ref) in self.datasets:
            raise FileExistsError(f"dataset {ref} already exists")
        rows = [dict(i) for i in items]
        import hashlib
        import json
        h = hashlib.sha256()
        for row in rows:
            h.update(json.dumps(row, sort_keys=True).encode())
        dataset = Dataset(ref=ref, kind=DatasetKind(kind), item_count=len(rows),
                          digest=h.hexdigest()[:32], description=description,
                          metadata=dict(metadata or {}))
        self.datasets[str(ref)] = dataset
        self.items[str(ref)] = rows
        return dataset


class FakeControlChannel:
    _STRENGTH = {"cancel_requested": 2}

    def __init__(self) -> None:
        self.pending: dict[str, ControlSignal] = {}
        self.cleared: list[str] = []

    def request(self, job_id: JobId, signal: ControlSignal) -> None:
        held = self.pending.get(str(job_id))
        if held is not None:
            if (self._STRENGTH.get(signal.reason.value, 1)
                    < self._STRENGTH.get(held.reason.value, 1)):
                return
        self.pending[str(job_id)] = signal

    def poll(self, job_id: JobId) -> ControlSignal | None:
        return self.pending.get(str(job_id))

    def clear(self, job_id: JobId) -> None:
        self.pending.pop(str(job_id), None)
        self.cleared.append(str(job_id))


class FakeMetricSink:
    def __init__(self) -> None:
        self.points: list[MetricPoint] = []
        self.closed = False

    def emit(self, points: Iterable[MetricPoint]) -> None:
        self.points.extend(points)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def series(self, name: str) -> list[float]:
        return [p.value for p in self.points if p.name == name]


class FakeEventLog:
    def __init__(self) -> None:
        self.events: list[JobEvent] = []

    def append(self, event: JobEvent) -> None:
        self.events.append(event)

    def read(self, job_id: JobId, *, limit: int | None = None) -> Sequence[JobEvent]:
        out = [e for e in self.events if e.job_id == JobId(job_id)]
        return out[-limit:] if limit is not None else out

    def kinds(self, job_id: JobId) -> list[str]:
        return [e.kind.value for e in self.read(job_id)]


class FakeLauncher:
    """Records launches without performing them.

    ``alive`` is settable, so a test can construct the case the real world
    produces constantly: a job whose store says RUNNING and whose process is
    gone.
    """

    def __init__(self, *, kind: str = "fake", alive: WorkerStatus = WorkerStatus.ALIVE,
                 fail: Exception | None = None,
                 terminates: bool = True) -> None:
        self.kind = kind
        self.launched: list[JobId] = []
        self.terminated: list[tuple[JobId, float]] = []
        self.alive = alive
        self.fail = fail
        self.terminates = terminates
        self._pid = 4000

    def launch(self, job_id: JobId, *, workspace: str,
               env: Mapping[str, str] | None = None) -> WorkerHandle:
        if self.fail is not None:
            raise self.fail
        self.launched.append(JobId(job_id))
        self._pid += 1
        return WorkerHandle(job_id=JobId(job_id), kind=self.kind, pid=self._pid,
                            started_at=0.0, detail={"workspace": workspace})

    def is_alive(self, handle: WorkerHandle) -> WorkerStatus:
        return self.alive

    def terminate(self, handle: WorkerHandle, *, grace_seconds: float = 30.0) -> bool:
        self.terminated.append((handle.job_id, grace_seconds))
        if self.terminates:
            self.alive = WorkerStatus.GONE
        return self.terminates


class RecordingLauncher(FakeLauncher):
    """A launcher that runs a callback instead of a process.

    Used where a test needs the *worker's* behaviour -- taking the job RUNNING,
    reporting, finishing -- without a real subprocess.  The callback is handed
    the job id, so it can drive the store exactly as a worker would.
    """

    def __init__(self, on_launch: Callable[[JobId], None], **kw) -> None:
        super().__init__(**kw)
        self._on_launch = on_launch

    def launch(self, job_id: JobId, *, workspace: str,
               env: Mapping[str, str] | None = None) -> WorkerHandle:
        handle = super().launch(job_id, workspace=workspace, env=env)
        self._on_launch(JobId(job_id))
        return handle
