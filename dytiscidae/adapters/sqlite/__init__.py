"""The SQLite DB adapter from the architecture diagram.

Holds the record and the lifecycle -- experiments, jobs, runtime state -- and
never a checkpoint.  See ``store.py`` for the measurement behind that split and
for what SQLite buys over the directory tree.
"""

from .store import (
    SCHEMA_VERSION,
    SqliteExperimentStore,
    SqliteJobStore,
    SqliteStore,
)

__all__ = [
    "SCHEMA_VERSION",
    "SqliteExperimentStore",
    "SqliteJobStore",
    "SqliteStore",
]
