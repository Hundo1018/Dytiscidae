"""The runtime state: how far a job has got, right now.

Kept apart from the plan and from the experiment record on purpose.  The three
have different lifetimes and different readers:

    plan        immutable, cited in papers, hashes to a digest
    state       overwritten every step, cited by nobody, thrown away on resume
    record      appended to, read months later

Mixing them is the mistake this separation exists to prevent: a config that
carries ``current_generation`` cannot be compared against another config, and a
record that carries the live step count has to be rewritten every 74 seconds.

``TrainingState`` is a value: every mutation returns a new one.  That makes a
state safe to hand to a metric sink, a status file writer and a stop-condition
check in the same step without any of them seeing a half-applied update.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .ids import CheckpointId, JobId


@dataclass(frozen=True)
class TrainingState:
    """Progress through a plan's budget.

    ``step`` is the trainer's unit, matching ``TrainingBudget.max_steps``.  For
    the design search it is the generation number.

    ``samples_seen`` is the finer-grained count that actually costs money -- for
    the search, the number of evaluations.  Both are kept because the ratio
    between them is not constant: a generation that rejects candidates at Tier 0
    costs fewer evaluations than one that does not, and reading progress off
    generations alone has misled this project before.
    """

    job_id: JobId
    step: int = 0
    #: The step the budget ends at, where it is known.  None for an open budget.
    total_steps: int | None = None
    samples_seen: int = 0
    elapsed_seconds: float = 0.0
    #: Wall clock when this state was produced, as a Unix timestamp.  Written by
    #: whoever holds the Clock -- the domain never reads a clock itself.
    updated_at: float = 0.0
    #: The most recent checkpoint, or None if the job has not written one.  A
    #: resume that finds this None starts from step 0, which is correct and is
    #: also why a job that ran for an hour without checkpointing is a
    #: configuration error rather than a crash.
    last_checkpoint: CheckpointId | None = None
    #: The step ``last_checkpoint`` was written at.  The gap between this and
    #: ``step`` is exactly what a crash costs, so it is worth being able to read
    #: without loading the checkpoint.
    last_checkpoint_step: int | None = None
    #: Whatever the trainer wants surfaced live.  Scalars only: this is written
    #: into a status file every step and read by things that must not import a
    #: numeric library.
    metrics: Mapping[str, float] = field(default_factory=dict)
    #: Free-form, for what does not reduce to a float -- the active island, the
    #: current regime.
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, JobId):
            object.__setattr__(self, "job_id", JobId(self.job_id))
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "detail", dict(self.detail))

    # -- derived ----------------------------------------------------------

    @property
    def progress(self) -> float | None:
        """Fraction of the budget spent, or None when the budget is open.

        None rather than 0.0, because "no budget" and "nothing done" are
        different facts and a progress bar that shows 0% forever for the first
        is a bug report waiting to happen.
        """
        if not self.total_steps:
            return None
        return min(1.0, max(0.0, self.step / float(self.total_steps)))

    @property
    def steps_at_risk(self) -> int:
        """Steps done since the last checkpoint -- what a crash would cost."""
        if self.last_checkpoint_step is None:
            return self.step
        return max(0, self.step - self.last_checkpoint_step)

    @property
    def steps_per_second(self) -> float | None:
        if self.elapsed_seconds <= 0 or self.step <= 0:
            return None
        return self.step / self.elapsed_seconds

    def eta_seconds(self) -> float | None:
        """Seconds remaining at the observed rate, or None if unknowable.

        A flat extrapolation, and it is wrong early in a search by design:
        generations 0-5 of a run are a verification burst at ~300 s against a
        steady state of ~74 s, so an ETA taken in the first minutes overstates
        by 4x.  Callers that care should say how many steps the estimate rests
        on; this returns the number, not a promise about it.
        """
        rate = self.steps_per_second
        if rate is None or not self.total_steps:
            return None
        return max(0.0, (self.total_steps - self.step) / rate)

    # -- transitions ------------------------------------------------------

    def advanced(self, *, step: int, samples_seen: int | None = None,
                 elapsed_seconds: float | None = None, now: float | None = None,
                 metrics: Mapping[str, float] | None = None,
                 detail: Mapping[str, Any] | None = None) -> "TrainingState":
        """The state after one step.  Monotonic in ``step`` and ``samples_seen``.

        Monotonic because a trainer that reports a step number lower than the
        last one is either resuming into the wrong job or double-counting, and
        both are worth a loud failure rather than a chart that goes backwards.
        """
        if step < self.step:
            raise ValueError(
                f"step went backwards: {self.step} -> {step} for job {self.job_id}")
        if samples_seen is not None and samples_seen < self.samples_seen:
            raise ValueError(
                f"samples_seen went backwards: {self.samples_seen} -> {samples_seen}")
        return replace(
            self,
            step=step,
            samples_seen=self.samples_seen if samples_seen is None else samples_seen,
            elapsed_seconds=(self.elapsed_seconds if elapsed_seconds is None
                             else elapsed_seconds),
            updated_at=self.updated_at if now is None else now,
            metrics=dict(metrics) if metrics is not None else dict(self.metrics),
            detail=dict(detail) if detail is not None else dict(self.detail),
        )

    def checkpointed(self, checkpoint_id: CheckpointId, *, step: int | None = None
                     ) -> "TrainingState":
        at = self.step if step is None else step
        return replace(self, last_checkpoint=checkpoint_id, last_checkpoint_step=at)

    # -- serialisation ----------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "job_id": str(self.job_id),
            "step": self.step,
            "total_steps": self.total_steps,
            "samples_seen": self.samples_seen,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "updated_at": self.updated_at,
            "last_checkpoint": str(self.last_checkpoint) if self.last_checkpoint else None,
            "last_checkpoint_step": self.last_checkpoint_step,
            "metrics": dict(self.metrics),
            "detail": dict(self.detail),
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "TrainingState":
        ck = d.get("last_checkpoint")
        return cls(
            job_id=JobId(d["job_id"]),
            step=int(d.get("step", 0)),
            total_steps=d.get("total_steps"),
            samples_seen=int(d.get("samples_seen", 0)),
            elapsed_seconds=float(d.get("elapsed_seconds", 0.0)),
            updated_at=float(d.get("updated_at", 0.0)),
            last_checkpoint=CheckpointId(ck) if ck else None,
            last_checkpoint_step=d.get("last_checkpoint_step"),
            metrics=dict(d.get("metrics") or {}),
            detail=dict(d.get("detail") or {}),
        )
