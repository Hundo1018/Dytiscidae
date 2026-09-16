"""Ports: the interfaces the inside of the hexagon talks through.

Every one is a ``typing.Protocol``, not an ABC.  The reason is practical rather
than doctrinal: a Protocol is satisfied structurally, so a test fake is a
fifteen-line class that inherits from nothing, and an existing object -- an
open file, a ``Telemetry`` -- can satisfy a port without being modified to
declare that it does.  An ABC would force every adapter to import this package
at class-definition time, which is exactly the inward dependency that the
arrangement is meant to keep possible and unnecessary.

The four ports the architecture is named for:

    Trainer             runs the training.  The driven port that matters.
    DatasetRepository   resolves a seed corpus.
    CheckpointStore     round-trips a checkpoint's bytes.
    ExperimentStore     keeps the record.

And the ones a long-running, cancellable, restartable job turns out to need,
each of which would otherwise be an undeclared dependency on the filesystem
inside a use case:

    JobStore            job lifecycle state, read by two processes.
    MetricSink          metrics as data.
    EventLog            job events as data.
    ControlChannel      how a pause or a cancel reaches a running worker.
    JobLauncher         how a worker process is started, and whether it is alive.
    Clock               the only source of "now" inside the hexagon.

``JobStore`` is a deviation from the four-port diagram and the reason is worth
stating: an experiment is a question that may be attempted several times, and a
job is one attempt.  They have different write rates -- an experiment is written
twice in its life, a job's state is written every generation by a different
process -- and folding the second into the first would put a hot, concurrently
written record inside a cold, human-read one.
"""

from .checkpoint_store import CheckpointStore
from .clock import Clock, SystemClock
from .control import ControlChannel, ControlSignal
from .dataset_repository import DatasetRepository
from .event_log import EventLog
from .experiment_store import ExperimentStore
from .job_store import JobStore
from .launcher import JobLauncher, WorkerHandle, WorkerStatus
from .metric_sink import MetricSink
from .trainer import (
    StopRequest,
    Trainer,
    TrainerCapabilities,
    TrainingContext,
    TrainingOutcome,
)

__all__ = [
    "CheckpointStore",
    "Clock",
    "ControlChannel",
    "ControlSignal",
    "DatasetRepository",
    "EventLog",
    "ExperimentStore",
    "JobLauncher",
    "JobStore",
    "MetricSink",
    "StopRequest",
    "SystemClock",
    "Trainer",
    "TrainerCapabilities",
    "TrainingContext",
    "TrainingOutcome",
    "WorkerHandle",
    "WorkerStatus",
]
