"""``ExperimentStore`` over a directory of JSON files.

    <root>/experiments/<experiment_id>.json

No name index.  ``find_by_name`` scans, and that is the right trade here: an
index is a second copy of the truth that has to be kept consistent with the
first, and the first -- the directory -- is what someone reads with ``ls`` when
they are trying to work out what happened.  The scan is over a directory whose
size is the number of experiments a person has run, which is hundreds.

``create`` refuses a duplicate name as well as a duplicate id.  The use case
checks it too, and the duplication is deliberate: the use case's check is the
one that produces a good error message, and this one is the one that holds when
two processes create the same name at the same time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ...domain.errors import ExperimentNotFound
from ...domain.experiment import Experiment
from ...domain.ids import ExperimentId, JobId
from ._io import FileLock, atomic_write_json, read_json


class FileExperimentStore:
    """The filesystem implementation of ``ports.ExperimentStore``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def dir(self) -> Path:
        return self.root / "experiments"

    def _path(self, experiment_id: ExperimentId) -> Path:
        return self.dir / f"{experiment_id}.json"

    def _lock(self) -> FileLock:
        return FileLock(self.dir / ".experiments.lock")

    def create(self, experiment: Experiment) -> Experiment:
        with self._lock():
            path = self._path(experiment.experiment_id)
            if path.exists():
                raise FileExistsError(
                    f"experiment {experiment.experiment_id} already exists")
            clash = self._find_by_name_unlocked(experiment.name)
            if clash is not None:
                raise FileExistsError(
                    f"an experiment named {experiment.name!r} already exists "
                    f"({clash.experiment_id})")
            atomic_write_json(path, experiment.as_dict())
        return experiment

    def get(self, experiment_id: ExperimentId) -> Experiment:
        d = read_json(self._path(ExperimentId(experiment_id)))
        if d is None:
            raise ExperimentNotFound(f"no experiment {experiment_id} under {self.root}")
        return Experiment.from_dict(d)

    def find_by_name(self, name: str) -> Experiment | None:
        return self._find_by_name_unlocked(name)

    def _find_by_name_unlocked(self, name: str) -> Experiment | None:
        for experiment in self._scan():
            if experiment.name == name:
                return experiment
        return None

    def update(self, experiment: Experiment) -> Experiment:
        path = self._path(experiment.experiment_id)
        if not path.exists():
            raise ExperimentNotFound(
                f"no experiment {experiment.experiment_id} to update")
        atomic_write_json(path, experiment.as_dict())
        return experiment

    def attach_job(self, experiment_id: ExperimentId, job_id: JobId) -> Experiment:
        experiment_id = ExperimentId(experiment_id)
        # Locked: two jobs started at once under one experiment would otherwise
        # read the same job list and each write back a list containing only its
        # own addition.
        with self._lock():
            d = read_json(self._path(experiment_id))
            if d is None:
                raise ExperimentNotFound(f"no experiment {experiment_id}")
            updated = Experiment.from_dict(d).with_job(JobId(job_id))
            atomic_write_json(self._path(experiment_id), updated.as_dict())
            return updated

    def list(self, *, tag: str | None = None,
             limit: int | None = None) -> Sequence[Experiment]:
        out = []
        for experiment in self._scan(newest_first=True):
            if tag is not None and tag not in experiment.tags:
                continue
            out.append(experiment)
            if limit is not None and len(out) >= limit:
                break
        return out

    def delete(self, experiment_id: ExperimentId) -> None:
        self._path(ExperimentId(experiment_id)).unlink(missing_ok=True)

    def _scan(self, *, newest_first: bool = False):
        if not self.dir.is_dir():
            return
        # Ids carry a UTC timestamp, so reverse lexical order is newest first
        # without a stat call.
        for path in sorted(self.dir.glob("*.json"), key=lambda p: p.name,
                           reverse=newest_first):
            d = read_json(path)
            if d is None:
                continue
            try:
                yield Experiment.from_dict(d)
            except Exception:                                     # noqa: BLE001
                continue
