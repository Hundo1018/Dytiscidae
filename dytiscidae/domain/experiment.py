"""``Experiment``: the record, as distinct from the configuration and the state.

This is what ``runs/archNN_notes.md`` is in this project today -- written by
hand, after the fact, from memory and a log.  Modelling it means the question
"what was arch38 and what did it show" has an answer that does not depend on
somebody having remembered to write one.

An experiment is not a run.  It is a *question*, under which one or more jobs
are run: a resumed search is two jobs and one experiment, and an A/B is two
experiments whose plans differ in exactly the arm.  Keeping them separate is
what makes "compare the arms" a query rather than an archaeology exercise.

``conclusion`` is deliberately free text and deliberately present.  A record
that holds every number and no statement of what they meant is the state this
project's own roadmap describes as the reason nothing can be compared across
the arch33/arch34 boundary: the numbers survived and the fact that a term had
been redefined did not.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .ids import ExperimentId, JobId
from .plan import TrainingPlan


@dataclass(frozen=True)
class Experiment:
    """A question, its configuration, and the attempts made at it."""

    experiment_id: ExperimentId
    name: str
    plan: TrainingPlan
    created_at: float = 0.0
    description: str = ""
    #: What the experiment is predicted to show, written *before* it runs.
    #: Optional, and it is the field most worth filling: a prediction recorded
    #: afterwards is not a prediction.
    hypothesis: str = ""
    #: What it did show, written after.  Free text on purpose -- the numbers are
    #: in the metrics, and what is missing from a bare number is what it meant.
    conclusion: str = ""
    #: Jobs run under this experiment, oldest first.  Several, because a resume
    #: is a new job and the experiment is still one experiment.
    job_ids: tuple[JobId, ...] = ()
    tags: tuple[str, ...] = ()
    #: git sha, host, library versions, captured when the experiment was
    #: created.  The plan says what was asked for; this says what would run it.
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, ExperimentId):
            object.__setattr__(self, "experiment_id", ExperimentId(self.experiment_id))
        if not str(self.name).strip():
            raise ValueError("experiment name must be non-empty")
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "job_ids",
                           tuple(j if isinstance(j, JobId) else JobId(j)
                                 for j in self.job_ids))
        object.__setattr__(self, "tags", tuple(sorted(str(t) for t in self.tags)))
        object.__setattr__(self, "provenance", dict(self.provenance))

    def with_job(self, job_id: JobId) -> "Experiment":
        """Attach a job.  Idempotent, and order-preserving."""
        jid = job_id if isinstance(job_id, JobId) else JobId(job_id)
        if jid in self.job_ids:
            return self
        return replace(self, job_ids=self.job_ids + (jid,))

    def concluded(self, conclusion: str) -> "Experiment":
        return replace(self, conclusion=conclusion)

    @property
    def latest_job(self) -> JobId | None:
        return self.job_ids[-1] if self.job_ids else None

    def as_dict(self) -> dict:
        return {
            "experiment_id": str(self.experiment_id),
            "name": self.name,
            "plan": self.plan.as_dict(),
            "plan_digest": self.plan.digest,
            "created_at": self.created_at,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "conclusion": self.conclusion,
            "job_ids": [str(j) for j in self.job_ids],
            "tags": list(self.tags),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Experiment":
        return cls(
            experiment_id=ExperimentId(d["experiment_id"]),
            name=d["name"],
            plan=TrainingPlan.from_dict(d["plan"]),
            created_at=float(d.get("created_at", 0.0)),
            description=d.get("description", ""),
            hypothesis=d.get("hypothesis", ""),
            conclusion=d.get("conclusion", ""),
            job_ids=tuple(JobId(j) for j in (d.get("job_ids") or ())),
            tags=tuple(d.get("tags") or ()),
            provenance=dict(d.get("provenance") or {}),
        )

    def __str__(self) -> str:
        return f"{self.experiment_id} {self.name!r} ({len(self.job_ids)} job(s))"
