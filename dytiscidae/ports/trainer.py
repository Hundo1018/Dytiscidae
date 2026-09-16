"""The ``Trainer`` port: where PyTorch, MuJoCo and the GPU live, on the far side.

The shape of this port is the whole design decision.  A port of the form

    def train(plan) -> result

is what turns training into a function call, and it cannot express any of the
six things a long run needs: progress, checkpoints, resume, pause, cancel, and
a failure that is recorded rather than lost.  So the port is

    def train(plan, context) -> outcome

where ``context`` is what the runtime hands *inward*: a way to report progress,
a way to save a checkpoint, a resume point, and a stop flag to poll.  Control
is inverted at exactly one place -- the trainer decides *when* it is safe to
stop, and the application decides *whether* it should.  That is the correct
split, because only the trainer knows where its state is consistent, and only
the application knows that the user pressed cancel.

The cooperative-stop contract, stated once and depended on everywhere:

* ``context.stop_requested()`` is polled by the trainer at its own safe points.
* A trainer that sees a stop **finishes the current unit of work, writes a
  checkpoint, and returns** an outcome whose ``stop_reason`` says which.  It
  does not raise.  Raising would be reported as a failure, and a clean pause is
  not a failure.
* A trainer that ignores the flag is not wrong, it is *uncancellable*, and it
  declares that in ``TrainerCapabilities.cooperative_stop``.  The launcher then
  knows its only remaining option is to kill the process, which costs everything
  since the last checkpoint -- so the capability is a promise a caller can plan
  against rather than a hope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable

from ..domain.checkpoint import CheckpointKind, CheckpointPayload, CheckpointRecord
from ..domain.dataset import Dataset
from ..domain.ids import CheckpointId
from ..domain.job import StopReason
from ..domain.plan import TrainingPlan
from ..domain.state import TrainingState


@dataclass(frozen=True)
class StopRequest:
    """A stop the trainer has been asked to honour."""

    reason: StopReason
    #: True where the caller will not accept a long tail: the trainer should
    #: stop at the next boundary it has, rather than at the next *natural* one.
    #: Advisory -- the trainer still chooses a consistent point, because a
    #: checkpoint written mid-update is worse than none.
    urgent: bool = False
    requested_at: float = 0.0

    @property
    def is_cancel(self) -> bool:
        return self.reason is StopReason.CANCEL_REQUESTED

    def __str__(self) -> str:
        return f"{self.reason}{' (urgent)' if self.urgent else ''}"


@dataclass(frozen=True)
class TrainerCapabilities:
    """What a trainer promises, declared rather than discovered at hour four."""

    #: What one ``step`` is.  Free text, and it goes into the record: the
    #: difference between "generation" and "optimiser step" is three orders of
    #: magnitude and every budget is expressed in it.
    step_unit: str = "step"
    #: Honours ``context.stop_requested()``.  False means a pause or a cancel
    #: can only be delivered by killing the process.
    cooperative_stop: bool = False
    #: Writes checkpoints through ``context.save_checkpoint``.
    checkpoints: bool = False
    #: Can start from a checkpoint handed to it in the context.
    resumes: bool = False
    #: Produces something ``export_model`` can turn into a deployable artefact.
    exports: bool = False
    #: True where the same plan, seed and resume point give the same result.
    #: Claimed, not assumed: this project has measured a resume that silently
    #: was not deterministic because the RNG state was not in the checkpoint.
    deterministic: bool = False
    description: str = ""

    def as_dict(self) -> dict:
        return {"step_unit": self.step_unit,
                "cooperative_stop": self.cooperative_stop,
                "checkpoints": self.checkpoints, "resumes": self.resumes,
                "exports": self.exports, "deterministic": self.deterministic,
                "description": self.description}


@dataclass(frozen=True)
class TrainingOutcome:
    """How a training call ended.  Returned, never raised.

    Raising is reserved for failures.  A pause, a cancel and a completion are
    all *results*, and giving them the same channel as an exception is how a
    cancelled run ends up recorded as a crash.
    """

    #: The state as of the last step, so the caller does not have to have been
    #: watching to know where it got to.
    state: TrainingState
    stop_reason: StopReason = StopReason.COMPLETED
    #: The checkpoint the trainer left behind, if any.  What a resume reads.
    final_checkpoint: CheckpointId | None = None
    #: Free-form, for what the trainer wants in the record.
    summary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.stop_reason is StopReason.COMPLETED

    @property
    def cancelled(self) -> bool:
        return self.stop_reason is StopReason.CANCEL_REQUESTED

    @property
    def paused(self) -> bool:
        return self.stop_reason in (
            StopReason.PAUSE_REQUESTED, StopReason.SIGNAL,
            StopReason.BUDGET_SECONDS, StopReason.BUDGET_MEMORY)


@runtime_checkable
class TrainingContext(Protocol):
    """What the runtime hands the trainer.  The trainer's only way out.

    A trainer that writes to a path, reads a clock or prints to stdout is
    reaching around this and will not be testable, pausable or observable.
    Everything it needs is here.
    """

    @property
    def plan(self) -> TrainingPlan:
        """The configuration.  Read-only; the trainer never edits it."""

    @property
    def state(self) -> TrainingState:
        """The state as the runtime last recorded it."""

    @property
    def dataset(self) -> Dataset | None:
        """The resolved seed corpus's metadata, or None where the plan has none."""

    @property
    def resources(self) -> Mapping[str, Any]:
        """Per-execution settings: worker count, shard size, device.

        Separate from ``plan.hyperparameters`` and not part of the plan digest,
        because they are properties of the machine rather than of the
        experiment.  This project measured the same work at 31.7 s on four
        shards of four and 93.3 s on sixteen of one: pool shape changes the wall
        time by 3x and the result by nothing, so two runs that differ only in it
        must still compare as one configuration.
        """

    @property
    def workspace(self) -> str:
        """A directory this job may write into.

        Present because some trainers -- this project's search among them --
        are built around a run directory and will not be rewritten to stream
        every artefact through a port.  It is scoped to the job, so a trainer
        using it is still isolated from every other job; but anything written
        here and not through ``save_checkpoint`` is outside the store's
        retention, its digesting and its provenance, and is therefore not a
        checkpoint no matter what it is called.
        """

    def items(self) -> Iterator[Mapping[str, Any]]:
        """Iterate the seed corpus.  Empty where the plan names no dataset."""

    def report(self, state: TrainingState, *,
               metrics: Mapping[str, float] | None = None) -> None:
        """Publish progress.  Cheap, and safe to call every step."""

    def save_checkpoint(self, payload: CheckpointPayload, *, step: int,
                        kind: CheckpointKind = CheckpointKind.PERIODIC,
                        metrics: Mapping[str, float] | None = None,
                        provenance: Mapping[str, Any] | None = None
                        ) -> CheckpointRecord:
        """Store a checkpoint and return its record."""

    def resume_from(self) -> tuple[CheckpointRecord, CheckpointPayload] | None:
        """The checkpoint this attempt is resuming from, or None for a fresh start.

        Called by the trainer, not pushed to it, because a trainer that cannot
        resume should not be handed a payload it will silently ignore -- it
        should simply never ask, and its ``resumes=False`` capability says so.
        """

    def stop_requested(self) -> StopRequest | None:
        """The pending stop, or None.  Poll at every safe point."""

    def note(self, message: str, **detail) -> None:
        """Record something that is not a metric and not a state change."""


@runtime_checkable
class Trainer(Protocol):
    """Runs the training.  Implemented in ``adapters/trainers/``, never here."""

    def capabilities(self) -> TrainerCapabilities:
        """What this trainer promises.  Called before ``train``, and recorded."""

    def train(self, context: TrainingContext) -> TrainingOutcome:
        """Run until the budget is spent, a stop is honoured, or it raises.

        Everything the trainer needs is on ``context``.  Returning is how it
        reports every ordinary ending, including a pause and a cancel; raising
        is reserved for a real failure and will be recorded as one.
        """
