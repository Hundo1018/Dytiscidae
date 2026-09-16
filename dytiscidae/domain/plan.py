"""The configuration half of "configuration, runtime state, experiment record".

A ``TrainingPlan`` is everything a rerun needs and nothing else.  It is frozen,
it hashes to a stable digest, and it contains no clock reading, no path that
only exists on one machine, and no live object.  Two plans with the same digest
describe the same run; if they do not, something that affects the result is
being carried somewhere else and that is a defect in the plan, not in the
comparison.

What is deliberately *not* here:

* where the output goes.  That is the job's, because the same plan run twice is
  two jobs in two directories and still one plan.
* how many workers to use.  That is a throughput decision about the machine, so
  it lives on the job as ``resources``.  Putting it in the plan would make the
  same experiment on a four-core and a sixteen-core box compare as two
  different configurations -- and this project has measured that pool shape
  changes wall time by 3x and the result by nothing.
* anything mutable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

from .dataset import DatasetRef
from .errors import PlanRejected


def _freeze(value: Any) -> Any:
    """Recursively convert to something hashable, ordered and JSON-writable."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in sorted(value.items())})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise PlanRejected(
        f"hyperparameters must be JSON scalars, lists or mappings; "
        f"got {type(value).__name__} ({value!r})"
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class TrainingBudget:
    """What the job is allowed to spend before it stops on its own.

    ``max_steps`` is in the trainer's own unit.  For the design search a step is
    one generation; for a supervised trainer it would be an optimiser step.  The
    domain does not care which, and the trainer declares it in
    ``TrainerCapabilities.step_unit`` so a reader is never left guessing.

    A budget that is entirely ``None`` is legal and means "until the trainer
    says it is done".  ``max_resident_mb`` is here because this project has
    already had a run stopped by the OOM killer with an hour of work between it
    and the last checkpoint; the search loop honours a ceiling at the checkpoint
    boundary, where stopping costs nothing.
    """

    max_steps: int | None = None
    max_seconds: float | None = None
    max_resident_mb: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_seconds", "max_resident_mb"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise PlanRejected(f"{name} must be positive or None, got {v!r}")

    def as_dict(self) -> dict:
        return {"max_steps": self.max_steps,
                "max_seconds": self.max_seconds,
                "max_resident_mb": self.max_resident_mb}

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "TrainingBudget":
        d = d or {}
        return cls(max_steps=d.get("max_steps"),
                   max_seconds=d.get("max_seconds"),
                   max_resident_mb=d.get("max_resident_mb"))


@dataclass(frozen=True)
class TrainingPlan:
    """An immutable, hashable description of one training configuration."""

    #: Which ``Trainer`` implementation runs this.  A name, resolved by the
    #: composition root -- never an import path into an adapter, or the domain
    #: would be naming infrastructure.
    trainer: str
    #: The trainer's own settings.  Opaque here on purpose: the domain must not
    #: know what ``segment_seconds`` means.
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    #: The seed corpus, if the trainer takes one.
    dataset: DatasetRef | None = None
    #: The one number that decides whether two runs of this plan agree.
    seed: int = 0
    budget: TrainingBudget = field(default_factory=TrainingBudget)
    #: Free-form labels for grouping -- the arm of an A/B, the machine class.
    #: Part of the digest, because an arm label that does not change the digest
    #: makes two arms look like one experiment rerun.
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.trainer).strip():
            raise PlanRejected("trainer must be a non-empty name")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise PlanRejected(f"seed must be an int, got {type(self.seed).__name__}")
        if self.seed < 0:
            raise PlanRejected(f"seed must be non-negative, got {self.seed}")
        object.__setattr__(self, "trainer", str(self.trainer).strip())
        object.__setattr__(self, "hyperparameters", _freeze(self.hyperparameters))
        object.__setattr__(self, "tags", tuple(sorted(str(t) for t in self.tags)))
        if self.dataset is not None and not isinstance(self.dataset, DatasetRef):
            raise PlanRejected(
                f"dataset must be a DatasetRef or None, got {type(self.dataset).__name__}")

    # -- identity ---------------------------------------------------------

    def as_dict(self) -> dict:
        """The canonical form.  Key order is fixed, so the JSON is stable."""
        return {
            "trainer": self.trainer,
            "hyperparameters": _thaw(self.hyperparameters),
            "dataset": self.dataset.as_dict() if self.dataset else None,
            "seed": self.seed,
            "budget": self.budget.as_dict(),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "TrainingPlan":
        ds = d.get("dataset")
        return cls(
            trainer=d["trainer"],
            hyperparameters=d.get("hyperparameters") or {},
            dataset=DatasetRef.from_dict(ds) if ds else None,
            seed=int(d.get("seed", 0)),
            budget=TrainingBudget.from_dict(d.get("budget")),
            tags=tuple(d.get("tags") or ()),
        )

    @property
    def digest(self) -> str:
        """A stable content hash, 16 hex characters.

        ``sort_keys`` on top of the already-sorted mapping, so a plan built from
        a dict in a different insertion order hashes the same.  Truncated to 16
        because it is read by humans out of directory names; the full 64 is
        available from ``hashlib`` if a collision ever matters, and at 16 hex
        characters a birthday collision needs ~4 billion plans.
        """
        blob = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def with_(self, **changes) -> "TrainingPlan":
        """A modified copy.  Named ``with_`` because ``with`` is a keyword."""
        return replace(self, **changes)

    def hyperparameter(self, name: str, default=None):
        return self.hyperparameters.get(name, default)

    def __str__(self) -> str:
        return f"{self.trainer}@{self.digest} seed={self.seed}"
