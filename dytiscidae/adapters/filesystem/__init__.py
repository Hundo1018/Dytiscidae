"""Filesystem adapters: the default storage for everything.

Chosen as the default rather than the database, because this project's runs are
started with ``setsid nohup`` in containers that get reclaimed, and a directory
of JSON and JSONL survives that without a service being up.  It is also what
``docs/PORTING.md`` already assumes: a run is a directory you can tar and move.

Everything lives under one root, one directory per job, so a job is a unit that
can be copied, archived or deleted whole::

    <root>/experiments/<experiment_id>.json
    <root>/datasets/<name>/<version>/{dataset.json,items.jsonl}
    <root>/jobs/<job_id>/job.json          lifecycle
    <root>/jobs/<job_id>/state.json        runtime state
    <root>/jobs/<job_id>/control.json      pending pause or cancel
    <root>/jobs/<job_id>/events.jsonl      history
    <root>/jobs/<job_id>/metrics.jsonl     metrics
    <root>/jobs/<job_id>/checkpoints/<checkpoint_id>/…
    <root>/jobs/<job_id>/…                 the trainer's own workspace
"""

from .checkpoint_store import CorruptCheckpoint, FileCheckpointStore
from .dataset_repository import FileDatasetRepository
from .experiment_store import FileExperimentStore
from .job_store import FileJobStore
from .observability import FileControlChannel, FileEventLog, FileMetricSink

__all__ = [
    "CorruptCheckpoint",
    "FileCheckpointStore",
    "FileControlChannel",
    "FileDatasetRepository",
    "FileEventLog",
    "FileExperimentStore",
    "FileJobStore",
    "FileMetricSink",
]
