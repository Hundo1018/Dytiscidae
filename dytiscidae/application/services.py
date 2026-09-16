"""``TrainingApplication``: the use cases, wired together, with the ports injected.

This is the seam.  Everything above it -- a CLI, an HTTP handler, a test -- talks
to this.  Everything below it is a port.  The object is constructed once, by a
composition root that knows which adapters to use (``adapters/composition.py``
for the real ones, a test's own fakes otherwise), and nothing inside it can
name an adapter.

It is a facade over the use case objects rather than a replacement for them.
The use cases stay individually constructible, because a caller that needs only
``CancelTraining`` should not have to supply a dataset repository to get it.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ..domain.experiment import Experiment
from ..domain.ids import ExperimentId, JobId
from ..domain.job import JobStatus, TrainingJob
from ..ports.checkpoint_store import CheckpointStore
from ..ports.clock import Clock, SystemClock
from ..ports.control import ControlChannel
from ..ports.event_log import EventLog
from ..ports.experiment_store import ExperimentStore
from ..ports.job_store import JobStore
from ..ports.launcher import JobLauncher
from ..ports.trainer import Trainer
from .checkpoints import ExportModel, LoadCheckpoint
from .create_experiment import CreateExperiment
from .dto import (
    CancelTrainingRequest,
    CancelTrainingResponse,
    CreateExperimentRequest,
    CreateExperimentResponse,
    ExportModelRequest,
    ExportModelResponse,
    JobStatusReport,
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    PauseTrainingRequest,
    PauseTrainingResponse,
    ResumeTrainingRequest,
    ResumeTrainingResponse,
    StartTrainingRequest,
    StartTrainingResponse,
)
from .queries import DescribeExperiment, GetJobStatus, ListExperiments, ListJobs
from .training import CancelTraining, PauseTraining, ResumeTraining, StartTraining


class TrainingApplication:
    """Every use case, over one set of ports."""

    def __init__(self, *, jobs: JobStore, experiments: ExperimentStore,
                 checkpoints: CheckpointStore, control: ControlChannel,
                 events: EventLog,
                 clock: Clock | None = None,
                 launcher: JobLauncher | None = None,
                 inline_launcher: JobLauncher | None = None,
                 resolve_trainer: Callable[[str], Trainer] | None = None,
                 provenance: Callable[[], Mapping[str, Any]] | None = None,
                 workspace_root: str = "runs") -> None:
        self.clock = clock or SystemClock()
        self.jobs = jobs
        self.experiments = experiments
        self.checkpoints = checkpoints
        self.control = control
        self.events = events
        self.launcher = launcher
        self.inline_launcher = inline_launcher

        lifecycle = dict(jobs=jobs, experiments=experiments, checkpoints=checkpoints,
                         control=control, events=events, clock=self.clock,
                         launcher=launcher, inline_launcher=inline_launcher,
                         workspace_root=workspace_root)

        self.create_experiment = CreateExperiment(experiments, self.clock,
                                                  provenance=provenance)
        self.start_training = StartTraining(**lifecycle)
        self.pause_training = PauseTraining(**lifecycle)
        self.resume_training = ResumeTraining(**lifecycle)
        self.cancel_training = CancelTraining(**lifecycle)
        self.load_checkpoint = LoadCheckpoint(checkpoints, jobs)
        self.export_model = (
            ExportModel(checkpoints, jobs, resolve_trainer)
            if resolve_trainer is not None else None)
        self.get_job_status = GetJobStatus(jobs, checkpoints, self.clock,
                                           launcher=launcher)
        self.list_jobs = ListJobs(jobs)
        self.list_experiments = ListExperiments(experiments)
        self.describe_experiment = DescribeExperiment(
            experiments, jobs, checkpoints, self.clock, launcher=launcher,
            events=events)

    # -- convenience.  Each forwards to the use case object above, so that the
    # -- common call is one line and the uncommon one is still reachable.

    def create(self, request: CreateExperimentRequest) -> CreateExperimentResponse:
        return self.create_experiment.execute(request)

    def start(self, request: StartTrainingRequest) -> StartTrainingResponse:
        return self.start_training.execute(request)

    def pause(self, request: PauseTrainingRequest) -> PauseTrainingResponse:
        return self.pause_training.execute(request)

    def resume(self, request: ResumeTrainingRequest) -> ResumeTrainingResponse:
        return self.resume_training.execute(request)

    def cancel(self, request: CancelTrainingRequest) -> CancelTrainingResponse:
        return self.cancel_training.execute(request)

    def load(self, request: LoadCheckpointRequest) -> LoadCheckpointResponse:
        return self.load_checkpoint.execute(request)

    def export(self, request: ExportModelRequest) -> ExportModelResponse:
        if self.export_model is None:
            from ..domain.errors import DomainError
            raise DomainError(
                "this application was built without a trainer resolver, so it "
                "cannot export. Pass resolve_trainer= when constructing it.")
        return self.export_model.execute(request)

    def status(self, job_id: JobId) -> JobStatusReport:
        return self.get_job_status.execute(job_id)

    def job_list(self, *, experiment_id: ExperimentId | None = None,
                 status: JobStatus | None = None,
                 limit: int | None = None) -> Sequence[TrainingJob]:
        return self.list_jobs.execute(experiment_id=experiment_id, status=status,
                                      limit=limit)

    def experiment_list(self, *, tag: str | None = None,
                        limit: int | None = None) -> Sequence[Experiment]:
        return self.list_experiments.execute(tag=tag, limit=limit)

    def describe(self, experiment_id: ExperimentId, *, with_events: bool = False
                 ) -> dict:
        return self.describe_experiment.execute(experiment_id,
                                                with_events=with_events)
