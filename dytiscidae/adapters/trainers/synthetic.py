"""A trainer that implements the whole contract and computes nothing real.

It exists for three jobs, and the first two are what make it worth keeping in
the shipped package rather than in a test file:

1. **It is the reference implementation of ``ports.Trainer``.**  Every part of
   the contract is exercised here in about a hundred lines -- the stop poll, the
   checkpoint, the resume, the export, the budget -- so "what does a trainer
   have to do" has an answer that runs.

2. **It makes the worker's lifecycle testable in seconds.**  Pause, resume,
   cancel, crash and container-reclamation all have to be tested against a real
   worker process, and testing them against the MuJoCo search would mean minutes
   per case and a machine with MuJoCo on it.  Here a step is a microsecond.

3. It is a load generator: ``step_seconds`` makes a step take as long as
   wanted, which is how the stop latency gets measured rather than asserted.

It uses no third-party library at all -- not even numpy -- so it runs on a bare
interpreter.  The arithmetic is a 64-bit linear congruential generator, which
is not a good random number generator and is not being used as one: it is here
so that the same seed gives the same sequence, and so that a resume that
restores the generator state can be *checked* to have continued rather than
restarted.  That check is the point.  This project has measured a resume that
silently did not restore its random stream, and every score written before the
boundary became unreproducible.
"""

from __future__ import annotations

import json
import time
from typing import Mapping

from ...domain.checkpoint import CheckpointKind, CheckpointPayload, CheckpointRecord
from ...domain.job import StopReason
from ...ports.trainer import TrainerCapabilities, TrainingContext, TrainingOutcome

#: Knuth's constants for a 64-bit LCG.  Not a good generator; a reproducible one.
_A, _C, _M = 6364136223846793005, 1442695040888963407, 1 << 64


class SyntheticTrainer:
    """Descends a quadratic with a reproducible pseudo-random step.

    Hyperparameters, all optional:

    ``checkpoint_every``  steps between checkpoints (default 5; 0 disables)
    ``step_seconds``      wall time to spend per step (default 0.0)
    ``fail_at_step``      raise ``SyntheticFailure`` once this many steps are
                          done, at the top of the next one -- so the reported
                          failure step equals this value and the last
                          checkpoint is at or below it
    ``start_value``       the initial loss (default 1.0)
    ``noise``             amplitude of the pseudo-random part (default 0.05)
    """

    name = "synthetic"

    def capabilities(self) -> TrainerCapabilities:
        return TrainerCapabilities(
            step_unit="step", cooperative_stop=True, checkpoints=True,
            resumes=True, exports=True, deterministic=True,
            description="reference implementation of the Trainer contract; "
                        "computes a quadratic descent with a reproducible LCG")

    # -- the contract -----------------------------------------------------

    def train(self, context: TrainingContext) -> TrainingOutcome:
        plan = context.plan
        hp = plan.hyperparameters
        checkpoint_every = int(hp.get("checkpoint_every", 5))
        step_seconds = float(hp.get("step_seconds", 0.0))
        fail_at = hp.get("fail_at_step")
        noise = float(hp.get("noise", 0.05))

        budget = plan.budget
        total = budget.max_steps
        deadline = (time.monotonic() + budget.max_seconds
                    if budget.max_seconds else None)

        step, value, rng, corpus = self._restore(context, hp)
        state = context.state
        last_record: CheckpointRecord | None = None
        started = time.monotonic()

        while total is None or step < total:
            if fail_at is not None and step == int(fail_at):
                raise SyntheticFailure(
                    f"asked to fail at step {step} (hyperparameter "
                    f"fail_at_step); this is the trainer doing what it was told")

            step += 1
            rng = (_A * rng + _C) % _M
            jitter = ((rng >> 11) / float(1 << 53) - 0.5) * 2.0 * noise
            # Quadratic descent, so the value is monotone apart from the noise
            # and a test can assert that a resumed run continues the curve.
            value = max(0.0, value * 0.97 + jitter * 0.01)
            if step_seconds:
                time.sleep(step_seconds)

            state = state.advanced(step=step, samples_seen=state.samples_seen + 1,
                                   metrics={"loss": round(value, 9),
                                            "rng": float(rng % 1_000_003)},
                                   detail={"phase": "descent",
                                           "corpus": str(corpus)})
            context.report(state, metrics=state.metrics)

            if checkpoint_every and step % checkpoint_every == 0:
                last_record = self._checkpoint(context, step, value, rng,
                                               CheckpointKind.PERIODIC)
                state = context.state

            stop = context.stop_requested()
            if stop is not None:
                # The contract: finish the unit of work, checkpoint unless the
                # run is being discarded, return -- never raise.
                if stop.is_cancel:
                    return TrainingOutcome(
                        state=state, stop_reason=StopReason.CANCEL_REQUESTED,
                        final_checkpoint=(last_record.checkpoint_id
                                          if last_record else None),
                        summary={"stopped_at": step, "reason": str(stop)})
                last_record = self._checkpoint(context, step, value, rng,
                                               CheckpointKind.FINAL)
                return TrainingOutcome(
                    state=context.state, stop_reason=stop.reason,
                    final_checkpoint=last_record.checkpoint_id,
                    summary={"stopped_at": step, "reason": str(stop)})

            if deadline is not None and time.monotonic() >= deadline:
                last_record = self._checkpoint(context, step, value, rng,
                                               CheckpointKind.FINAL)
                return TrainingOutcome(
                    state=context.state, stop_reason=StopReason.BUDGET_SECONDS,
                    final_checkpoint=last_record.checkpoint_id,
                    summary={"stopped_at": step,
                             "seconds": round(time.monotonic() - started, 3)})

        last_record = self._checkpoint(context, step, value, rng,
                                       CheckpointKind.FINAL)
        return TrainingOutcome(
            state=context.state, stop_reason=StopReason.COMPLETED,
            final_checkpoint=last_record.checkpoint_id,
            summary={"steps": step, "loss": round(value, 9),
                     "seconds": round(time.monotonic() - started, 3)})

    def export(self, record: CheckpointRecord, payload: CheckpointPayload, *,
               destination: str, fmt: str) -> dict:
        """Write the checkpoint's model as JSON.

        The trainer exports, not the use case, because the trainer is the only
        thing that knows what the blobs mean.  ``fmt`` is checked rather than
        ignored: an unknown format silently producing the native one is how a
        caller ends up shipping a file that is not what they asked for.
        """
        if fmt not in ("native", "json"):
            raise ValueError(
                f"{self.name} exports 'native' or 'json', not {fmt!r}")
        model = json.loads(payload["model.json"].decode("utf-8"))
        blob = json.dumps({"model": model, "step": record.step,
                           "checkpoint": str(record.checkpoint_id),
                           "provenance": dict(record.provenance)},
                          indent=1).encode("utf-8")
        from pathlib import Path
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        return {"bytes_written": len(blob), "format": fmt, "path": str(path)}

    # -- internals --------------------------------------------------------

    def _restore(self, context: TrainingContext, hp: Mapping):
        """Continue from a checkpoint, or start fresh.

        Counting the corpus here rather than ignoring it is deliberate: it makes
        the dataset port exercised by the reference trainer, so a repository
        that streams nothing is caught by the contract tests rather than by a
        real run that quietly seeded from nothing.
        """
        corpus = 0
        try:
            for _ in context.items():
                corpus += 1
        except Exception:                                         # noqa: BLE001
            corpus = -1              # -1 means "could not read", not "empty"

        resumed = context.resume_from()
        if resumed is None:
            return 0, float(hp.get("start_value", 1.0)), \
                (context.plan.seed * _A + _C) % _M, corpus

        record, payload = resumed
        model = json.loads(payload["model.json"].decode("utf-8"))
        return int(model["step"]), float(model["value"]), int(model["rng"]), corpus

    def _checkpoint(self, context: TrainingContext, step: int, value: float,
                    rng: int, kind: CheckpointKind) -> CheckpointRecord:
        # The generator state goes in the payload.  Without it a resumed run
        # draws a different sequence, which is exactly the defect this project
        # measured on a real run: a stored score of 2.288 re-measured as 0.000.
        payload = {"model.json": json.dumps(
            {"step": step, "value": value, "rng": rng}).encode("utf-8")}
        return context.save_checkpoint(payload, step=step, kind=kind,
                                       metrics={"loss": round(value, 9)})


class SyntheticFailure(RuntimeError):
    """Raised on request, so a failure path can be tested with a real process."""
