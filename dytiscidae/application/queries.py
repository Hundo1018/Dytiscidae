"""The read side: what is going on, without changing anything.

These are use cases too, and separating them from the lifecycle ones is not
ceremony.  ``GetJobStatus`` is called every few seconds by anything watching a
run, and it must never take a lock, post a control signal or write an event --
a status query with a side effect turns a monitoring loop into a participant.

``GetJobStatus`` does one thing beyond reading: it asks the launcher whether
the worker still exists.  That is what makes "RUNNING but the process is gone"
a reported state rather than something found by hand with ``pgrep``.
"""

from __future__ import annotations

from typing import Sequence

from ..domain.experiment import Experiment
from ..domain.ids import ExperimentId, JobId
from ..domain.job import JobStatus, TrainingJob
from ..ports.checkpoint_store import CheckpointStore
from ..ports.clock import Clock
from ..ports.event_log import EventLog
from ..ports.experiment_store import ExperimentStore
from ..ports.job_store import JobStore
from ..ports.launcher import JobLauncher, WorkerHandle, WorkerStatus
from .dto import JobStatusReport


class GetJobStatus:
    def __init__(self, jobs: JobStore, checkpoints: CheckpointStore, clock: Clock, *,
                 launcher: JobLauncher | None = None) -> None:
        self._jobs = jobs
        self._checkpoints = checkpoints
        self._clock = clock
        self._launcher = launcher

    def execute(self, job_id: JobId) -> JobStatusReport:
        job = self._jobs.get(JobId(job_id))
        state = self._jobs.load_state(job.job_id)
        latest = self._checkpoints.latest(job.job_id)

        worker_status = None
        handle_dict = (job.worker or {}).get("handle")
        if self._launcher is not None and handle_dict:
            try:
                worker_status = self._launcher.is_alive(
                    WorkerHandle.from_dict(handle_dict))
            except Exception:                                     # noqa: BLE001
                # A launcher that cannot answer is UNKNOWN, not an error that
                # takes down the thing asking.  A status query is the call
                # someone makes when they already suspect something is wrong.
                worker_status = WorkerStatus.UNKNOWN
        elif job.is_terminal or job.status is JobStatus.PAUSED:
            worker_status = WorkerStatus.GONE

        beat = (job.worker or {}).get("heartbeat_at")
        age = (self._clock.now() - float(beat)) if beat else None
        return JobStatusReport(job=job, state=state, worker_status=worker_status,
                               latest_checkpoint=latest, heartbeat_age=age)


class ListJobs:
    def __init__(self, jobs: JobStore) -> None:
        self._jobs = jobs

    def execute(self, *, experiment_id: ExperimentId | None = None,
                status: JobStatus | None = None,
                limit: int | None = None) -> Sequence[TrainingJob]:
        return self._jobs.list(experiment_id=experiment_id, status=status,
                               limit=limit)


class ListExperiments:
    def __init__(self, experiments: ExperimentStore) -> None:
        self._experiments = experiments

    def execute(self, *, tag: str | None = None,
                limit: int | None = None) -> Sequence[Experiment]:
        return self._experiments.list(tag=tag, limit=limit)


class DescribeExperiment:
    """An experiment with the state of every job under it.

    The query a report is built from, and the reason ``Experiment`` holds job
    ids rather than jobs: a run's record must be readable without loading
    twenty-one hours of telemetry, and *this* is where the two are joined.
    """

    def __init__(self, experiments: ExperimentStore, jobs: JobStore,
                 checkpoints: CheckpointStore, clock: Clock, *,
                 launcher: JobLauncher | None = None,
                 events: EventLog | None = None) -> None:
        self._experiments = experiments
        self._status = GetJobStatus(jobs, checkpoints, clock, launcher=launcher)
        self._events = events

    def execute(self, experiment_id: ExperimentId, *, with_events: bool = False,
                event_limit: int = 20) -> dict:
        experiment = self._experiments.get(ExperimentId(experiment_id))
        reports = []
        for job_id in experiment.job_ids:
            report = self._status.execute(job_id)
            row = {
                "job": report.job.as_dict(),
                "state": report.state.as_dict() if report.state else None,
                "worker_status": (str(report.worker_status)
                                  if report.worker_status else None),
                "latest_checkpoint": (report.latest_checkpoint.as_dict()
                                      if report.latest_checkpoint else None),
                "stale": report.stale,
            }
            if with_events and self._events is not None:
                row["events"] = [e.as_dict() for e in
                                 self._events.read(job_id, limit=event_limit)]
            reports.append(row)
        return {"experiment": experiment.as_dict(), "jobs": reports}
