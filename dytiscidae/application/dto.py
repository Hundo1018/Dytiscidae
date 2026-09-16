"""Request and response objects for the use cases.

Explicit types rather than keyword soup, for one reason that shows up
immediately: a use case with eight optional arguments cannot be called
correctly from a CLI, an HTTP handler and a test without each of them
re-deriving the defaults, and the three will drift.  A request object is the
one place the defaults live.

They are plain dataclasses of scalars and domain values.  Nothing here knows
about argparse, and nothing here knows about a filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..domain.checkpoint import CheckpointKind, CheckpointRecord
from ..domain.experiment import Experiment
from ..domain.ids import CheckpointId, ExperimentId, JobId
from ..domain.job import TrainingJob
from ..domain.plan import TrainingPlan
from ..domain.state import TrainingState
from ..ports.launcher import WorkerHandle, WorkerStatus


# -- create experiment -----------------------------------------------------

@dataclass(frozen=True)
class CreateExperimentRequest:
    name: str
    plan: TrainingPlan
    description: str = ""
    #: What this is predicted to show.  Recorded before the run, or not at all.
    hypothesis: str = ""
    tags: tuple[str, ...] = ()
    #: Extra provenance to merge over what the application collects itself.
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CreateExperimentResponse:
    experiment: Experiment


# -- start / resume --------------------------------------------------------

@dataclass(frozen=True)
class StartTrainingRequest:
    """Start a fresh job under an experiment.

    ``experiment_id`` or ``experiment_name``; exactly one.  The name is there
    because that is how people refer to runs, and making the CLI look up an id
    first would put that lookup in every caller.
    """

    experiment_id: ExperimentId | None = None
    experiment_name: str | None = None
    #: Per-execution settings: worker count, shard size, device.  Not in the
    #: plan, so the same experiment on two machines is still one configuration.
    resources: Mapping[str, Any] = field(default_factory=dict)
    #: Where the job writes.  Defaults to ``<root>/jobs/<job_id>``.
    workspace: str | None = None
    #: Run the trainer in this process instead of launching a worker.  For
    #: tests and for a search short enough to watch.  A long run started this
    #: way dies with its terminal, which is what process isolation is for.
    inline: bool = False
    #: Overrides merged over the experiment's plan hyperparameters for this job
    #: only.  Changes the plan, so the job records its own digest -- a run with
    #: an override is not a rerun of the experiment's plan and does not claim
    #: to be.
    hyperparameter_overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StartTrainingResponse:
    job: TrainingJob
    handle: WorkerHandle | None = None
    #: Present only for an inline start: the terminal state of the run.
    state: TrainingState | None = None


@dataclass(frozen=True)
class ResumeTrainingRequest:
    job_id: JobId
    #: Which checkpoint to continue from.  Defaults to the job's latest.
    checkpoint_id: CheckpointId | None = None
    inline: bool = False
    #: Raise the step budget so the resumed run has somewhere to go.  A resume
    #: of a job that spent its budget otherwise starts a worker that stops
    #: immediately, which looks exactly like a resume that did not work.
    extend_steps: int | None = None
    resources: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ResumeTrainingResponse:
    job: TrainingJob
    resumed_from: CheckpointRecord
    handle: WorkerHandle | None = None
    state: TrainingState | None = None


# -- pause / cancel --------------------------------------------------------

@dataclass(frozen=True)
class PauseTrainingRequest:
    job_id: JobId
    requested_by: str = ""
    #: Ask the trainer to stop at its next boundary rather than its next
    #: natural one.  Advisory: the trainer still picks a consistent point.
    urgent: bool = False


@dataclass(frozen=True)
class PauseTrainingResponse:
    job: TrainingJob
    #: True where the job was already stopped and nothing was requested.
    already_stopped: bool = False


@dataclass(frozen=True)
class CancelTrainingRequest:
    job_id: JobId
    requested_by: str = ""
    #: Kill the worker if it has not acknowledged within ``grace_seconds``.
    #: Everything since the last checkpoint is lost, so it is off by default and
    #: the response says whether it was used.
    force: bool = False
    grace_seconds: float = 0.0


@dataclass(frozen=True)
class CancelTrainingResponse:
    job: TrainingJob
    #: The worker was killed rather than asked.
    forced: bool = False
    worker_status: WorkerStatus | None = None


# -- checkpoints -----------------------------------------------------------

@dataclass(frozen=True)
class LoadCheckpointRequest:
    """By id, or the latest of a job.  Exactly one of the two."""

    checkpoint_id: CheckpointId | None = None
    job_id: JobId | None = None
    kind: CheckpointKind | None = None
    #: False returns the record alone, which is what a listing wants and is
    #: free; True fetches and verifies the payload.
    with_payload: bool = True


@dataclass(frozen=True)
class LoadCheckpointResponse:
    record: CheckpointRecord
    payload: Mapping[str, bytes] | None = None


@dataclass(frozen=True)
class ExportModelRequest:
    """Turn a checkpoint into something usable outside the run that made it."""

    destination: str
    checkpoint_id: CheckpointId | None = None
    job_id: JobId | None = None
    #: A name the trainer understands.  The trainer that wrote the checkpoint is
    #: the only thing that knows how to read it, so the export is dispatched
    #: back to it rather than done here by guessing at the payload.
    fmt: str = "native"
    overwrite: bool = False


@dataclass(frozen=True)
class ExportModelResponse:
    destination: str
    record: CheckpointRecord
    fmt: str
    bytes_written: int = 0
    detail: Mapping[str, Any] = field(default_factory=dict)


# -- queries ---------------------------------------------------------------

@dataclass(frozen=True)
class JobStatusReport:
    """Everything a caller needs to decide what to do about a job.

    ``worker_status`` is beside ``job.status`` and neither is redundant.  A job
    that says RUNNING with a worker that says GONE is the interesting case:
    the process died without recording anything, which is what a container
    reclamation looks like from here.
    """

    job: TrainingJob
    state: TrainingState | None = None
    worker_status: WorkerStatus | None = None
    latest_checkpoint: CheckpointRecord | None = None
    #: Seconds since the worker's last heartbeat, or None if it never sent one.
    heartbeat_age: float | None = None

    @property
    def stale(self) -> bool:
        """Claims to be active, but nothing says it is.

        Not a verdict.  A search generation is ~74 s and the opening burst is
        ~300 s, so a heartbeat written per generation is legitimately minutes
        old; the threshold belongs to whoever knows the cadence.  What this
        answers is the unambiguous half: active, and the launcher says the
        process is gone.
        """
        return (self.job.is_active
                and self.worker_status is WorkerStatus.GONE)
