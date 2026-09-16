"""The ``ExperimentStore`` port."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..domain.experiment import Experiment
from ..domain.ids import ExperimentId, JobId


@runtime_checkable
class ExperimentStore(Protocol):
    """Keeps the record: what was asked, under what name, and what came of it."""

    def create(self, experiment: Experiment) -> Experiment:
        """Store a new experiment.  Raises if the id already exists.

        Raises rather than overwrites.  An experiment is a record, and a record
        that can be silently replaced is not one.
        """

    def get(self, experiment_id: ExperimentId) -> Experiment:
        """Raises ``ExperimentNotFound`` when it is not there."""

    def find_by_name(self, name: str) -> Experiment | None:
        """The experiment with this name, or None.

        Names are how people refer to experiments -- "arch38" -- so this exists
        even though the id is the key.  Implementations enforce uniqueness of
        the name, because two experiments called arch38 make every later
        reference ambiguous.
        """

    def update(self, experiment: Experiment) -> Experiment:
        """Replace a stored experiment.  Raises if it does not exist."""

    def attach_job(self, experiment_id: ExperimentId, job_id: JobId) -> Experiment:
        """Record that a job was run under this experiment.  Idempotent."""

    def list(self, *, tag: str | None = None, limit: int | None = None
             ) -> Sequence[Experiment]:
        """Experiments, newest first."""

    def delete(self, experiment_id: ExperimentId) -> None:
        """Remove one.  Idempotent."""
