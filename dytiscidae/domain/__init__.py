"""The domain: what a training run *is*, with nothing about how it is done.

Nothing in this package may import ``torch``, ``mujoco``, ``sqlite3``,
``numpy`` or any adapter.  That is not a style preference -- it is the property
the rest of the architecture is built on, and ``tests/test_architecture.py``
fails the build if it stops holding.  The measurement that makes it concrete:
on a bare Python 3.11 with no third-party package installed at all,

    python -c "import dytiscidae.domain, dytiscidae.ports, dytiscidae.application"

succeeds.  The same interpreter cannot import ``dytiscidae.evolution.loop``.

The vocabulary, and what each word means *in this project* rather than in
general:

``TrainingPlan``
    the configuration.  Immutable, hashable, and the only thing a rerun needs
    to be the same experiment.  For the design search this carries the
    ``SearchConfig`` fields as an opaque mapping -- opaque because the domain
    must not know that MAP-Elites exists.

``TrainingState``
    the runtime state.  How far the job has got, right now.  Discarded and
    rebuilt on every resume; never the thing anyone cites.

``Experiment``
    the record.  Intent, provenance, and which jobs were run under it.  This is
    what ``runs/archNN_notes.md`` is today, written by hand.

``TrainingJob``
    one execution attempt of a plan, with a lifecycle.  Training is a job, not
    a function call: it runs for hours, it can be paused, resumed, cancelled,
    and it can fail -- and each of those has to be a state someone can read,
    not an exception that vanished with the process.

``Checkpoint``
    the artefact that makes resume, fine-tune and export possible, described
    here and stored elsewhere.

``Dataset``
    the seed corpus a run starts from.  This project has no supervised dataset;
    what plays that role is the collection of genomes a search is seeded with,
    and calling it a dataset is accurate as long as nobody reads it as labelled
    examples.  See ``docs/ARCHITECTURE.md`` §"Dataset, honestly".
"""

from .checkpoint import CheckpointKind, CheckpointRecord
from .dataset import Dataset, DatasetKind, DatasetRef
from .errors import (
    CheckpointNotFound,
    DatasetNotFound,
    DomainError,
    ExperimentNotFound,
    IllegalTransition,
    JobNotFound,
    PlanRejected,
)
from .experiment import Experiment
from .ids import CheckpointId, ExperimentId, JobId, new_id
from .job import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    FailureInfo,
    JobStatus,
    StopReason,
    TrainingJob,
)
from .metrics import JobEvent, JobEventKind, MetricPoint
from .plan import TrainingBudget, TrainingPlan
from .state import TrainingState

__all__ = [
    "ACTIVE_STATUSES",
    "CheckpointId",
    "CheckpointKind",
    "CheckpointNotFound",
    "CheckpointRecord",
    "Dataset",
    "DatasetKind",
    "DatasetNotFound",
    "DatasetRef",
    "DomainError",
    "Experiment",
    "ExperimentId",
    "ExperimentNotFound",
    "FailureInfo",
    "IllegalTransition",
    "JobEvent",
    "JobEventKind",
    "JobId",
    "JobNotFound",
    "JobStatus",
    "MetricPoint",
    "PlanRejected",
    "StopReason",
    "TERMINAL_STATUSES",
    "TrainingBudget",
    "TrainingJob",
    "TrainingPlan",
    "TrainingState",
    "new_id",
]
