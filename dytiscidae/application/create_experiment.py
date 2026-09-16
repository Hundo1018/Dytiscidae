"""``CreateExperiment``.

The smallest use case, and the one that decides whether anything later can be
compared.  It does three things: it fixes the plan, it captures the provenance
of the code that will run it, and it refuses a duplicate name.

The refusal is the part worth defending.  Two experiments called "arch38" make
every later reference to arch38 ambiguous -- in a report, in a roadmap, in a
conversation six weeks on -- and the cost of that is paid at the moment someone
tries to reconstruct what a number meant, which is exactly when the information
needed to disambiguate is gone.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..domain.errors import DomainError
from ..domain.experiment import Experiment
from ..domain.ids import ExperimentId, new_id
from ..ports.clock import Clock
from ..ports.experiment_store import ExperimentStore
from .dto import CreateExperimentRequest, CreateExperimentResponse


class DuplicateExperimentName(DomainError):
    def __init__(self, name: str, existing: ExperimentId) -> None:
        self.name, self.existing = name, existing
        super().__init__(
            f"an experiment named {name!r} already exists ({existing}). "
            f"Names are how runs are referred to afterwards; pick another, or "
            f"start a new job under the existing one.")


class CreateExperiment:
    """Record a question before running anything at it."""

    def __init__(self, experiments: ExperimentStore, clock: Clock, *,
                 provenance: Callable[[], Mapping[str, Any]] | None = None) -> None:
        self._experiments = experiments
        self._clock = clock
        # Injected, because "what code is this" is a question about the
        # environment -- a git binary, an installed package's version -- and
        # the application layer must not be the thing that knows how to ask it.
        self._provenance = provenance or (lambda: {})

    def execute(self, request: CreateExperimentRequest) -> CreateExperimentResponse:
        existing = self._experiments.find_by_name(request.name)
        if existing is not None:
            raise DuplicateExperimentName(request.name, existing.experiment_id)

        prov = dict(self._provenance())
        prov.update(request.provenance)
        # The digest goes in the provenance as well as being derivable from the
        # plan, so that a record read without this package still says which
        # configuration it was.
        prov.setdefault("plan_digest", request.plan.digest)

        experiment = Experiment(
            experiment_id=ExperimentId(new_id("exp")),
            name=request.name,
            plan=request.plan,
            created_at=self._clock.now(),
            description=request.description,
            hypothesis=request.hypothesis,
            tags=tuple(request.tags),
            provenance=prov,
        )
        return CreateExperimentResponse(self._experiments.create(experiment))
