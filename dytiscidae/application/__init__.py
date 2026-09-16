"""The application layer: what the system can be asked to do.

One class per use case, each with an ``execute``.  The layer answers "what
happens when someone starts a run", and it answers it without knowing that
PyTorch exists -- the same code drives a MuJoCo search and a two-line synthetic
trainer, because the only thing it knows about either is that it is a
``Trainer``.

The rule this layer keeps: **it decides that something has been asked for, and
the worker decides that it happened.**  ``PauseTraining`` returns with the job
in PAUSING, not PAUSED.  Anything else would be the application reporting an
outcome it is in no position to observe.

Like ``domain`` and ``ports``, this package imports no third-party library.
``tests/test_architecture.py`` fails the build if that stops being true.
"""

from .checkpoints import ExportModel, ExportNotSupported, LoadCheckpoint
from .create_experiment import CreateExperiment, DuplicateExperimentName
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
from .services import TrainingApplication
from .training import (
    CancelTraining,
    LauncherUnavailable,
    NoCheckpointToResume,
    PauseTraining,
    ResumeTraining,
    StartTraining,
)

__all__ = [
    "CancelTraining",
    "CancelTrainingRequest",
    "CancelTrainingResponse",
    "CreateExperiment",
    "CreateExperimentRequest",
    "CreateExperimentResponse",
    "DescribeExperiment",
    "DuplicateExperimentName",
    "ExportModel",
    "ExportModelRequest",
    "ExportModelResponse",
    "ExportNotSupported",
    "GetJobStatus",
    "JobStatusReport",
    "LauncherUnavailable",
    "ListExperiments",
    "ListJobs",
    "LoadCheckpoint",
    "LoadCheckpointRequest",
    "LoadCheckpointResponse",
    "NoCheckpointToResume",
    "PauseTraining",
    "PauseTrainingRequest",
    "PauseTrainingResponse",
    "ResumeTraining",
    "ResumeTrainingRequest",
    "ResumeTrainingResponse",
    "StartTraining",
    "StartTrainingRequest",
    "StartTrainingResponse",
    "TrainingApplication",
]
