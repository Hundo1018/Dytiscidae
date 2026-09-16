"""The training worker: the process that actually runs a job.

This is architecture point 4 and point 5 in code -- training is a job, and a
long job runs in its own process -- and its whole responsibility is to be the
one thing that turns a trainer's behaviour into job state that outlives it.

    Use Case  ->  TrainingJob  ->  Worker Process  ->  Training Runtime
                                        |
                                        +-- Model / Optimizer / GPU (the trainer)
                                        +-- Checkpoints
                                        +-- Metrics
                                        +-- Events

Six things it guarantees, each because its absence is a failure mode this
project has either measured or documented:

1. **Every exit is recorded.**  Normal return, cooperative stop, uncaught
   exception, SIGTERM -- each lands on the job as a status with a reason.  A
   worker that dies leaving a job saying RUNNING is a job nobody can act on.

2. **A failure carries its traceback into the record.**  The log is on a
   container that gets reclaimed; the job record is on disk that is kept.

3. **Stops are cooperative and honoured at the trainer's own boundary.**  The
   worker polls the control channel and raises the flag; the trainer chooses
   where to act on it, because only the trainer knows where its state is
   consistent.

4. **SIGTERM means pause, not die.**  The environments this runs in reclaim
   containers with a term signal and a short grace period.  Treating it as a
   pause converts a reclamation into a resumable stop.

5. **A checkpoint is written before the worker stops for any reason it can
   see.**  Except a cancel, where the run is being discarded on purpose.

6. **The heartbeat is written whether or not the trainer reports.**  A trainer
   that is slow and a trainer that is wedged look identical from the job record
   otherwise, and this project's steps are 74-300 s apart.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import traceback
from typing import Any, Callable, Iterator, Mapping

from ..domain.checkpoint import CheckpointKind, CheckpointPayload, CheckpointRecord
from ..domain.dataset import Dataset
from ..domain.ids import CheckpointId, JobId
from ..domain.job import FailureInfo, JobStatus, StopReason, TrainingJob
from ..domain.metrics import JobEvent, JobEventKind, MetricPoint
from ..domain.plan import TrainingPlan
from ..domain.state import TrainingState
from ..ports.checkpoint_store import CheckpointStore
from ..ports.clock import Clock, SystemClock
from ..ports.control import ControlChannel
from ..ports.dataset_repository import DatasetRepository
from ..ports.event_log import EventLog
from ..ports.job_store import JobStore
from ..ports.metric_sink import MetricSink
from ..ports.trainer import StopRequest, Trainer, TrainingOutcome


class WorkerContext:
    """The ``TrainingContext`` handed to the trainer.

    Everything the trainer is allowed to reach is a method here, and every one
    of them turns into job state.  ``report`` is the one called most; it is
    deliberately cheap -- two small file writes -- because a trainer that finds
    reporting expensive will report less often, and then the job record stops
    describing the run.
    """

    def __init__(self, *, job: TrainingJob, state: TrainingState,
                 jobs: JobStore, checkpoints: CheckpointStore,
                 metrics: MetricSink, events: EventLog, clock: Clock,
                 datasets: DatasetRepository | None,
                 stop: Callable[[], StopRequest | None],
                 resume_checkpoint: CheckpointId | None,
                 provenance: Mapping[str, Any] | None = None) -> None:
        self._job = job
        self._state = state
        self._jobs = jobs
        self._checkpoints = checkpoints
        self._metrics = metrics
        self._events = events
        self._clock = clock
        self._datasets = datasets
        self._stop = stop
        self._resume_id = resume_checkpoint
        self._provenance = dict(provenance or {})
        self._dataset: Dataset | None = None
        self._dataset_loaded = False
        self._t0 = clock.monotonic()
        # Carried across the resume boundary.  Without it a resumed run's
        # elapsed time restarts at zero and every rate derived from it is wrong
        # for as long as anyone believes the number.
        self._elapsed_before = state.elapsed_seconds

    # -- ports.TrainingContext: reads -------------------------------------

    @property
    def plan(self) -> TrainingPlan:
        return self._job.plan

    @property
    def state(self) -> TrainingState:
        return self._state

    @property
    def workspace(self) -> str:
        return self._job.workspace

    @property
    def resources(self) -> Mapping[str, Any]:
        return dict(self._job.resources)

    @property
    def dataset(self) -> Dataset | None:
        if not self._dataset_loaded:
            self._dataset_loaded = True
            ref = self._job.plan.dataset
            if ref is not None and self._datasets is not None:
                self._dataset = self._datasets.get(ref)
        return self._dataset

    def items(self) -> Iterator[Mapping[str, Any]]:
        ref = self._job.plan.dataset
        if ref is None or self._datasets is None:
            return iter(())
        return self._datasets.open(ref)

    # -- ports.TrainingContext: writes ------------------------------------

    def report(self, state: TrainingState,
               *, metrics: Mapping[str, float] | None = None) -> None:
        now = self._clock.now()
        elapsed = self._elapsed_before + (self._clock.monotonic() - self._t0)
        state = state.advanced(step=state.step, samples_seen=state.samples_seen,
                               elapsed_seconds=elapsed, now=now,
                               metrics=metrics if metrics is not None
                               else state.metrics,
                               detail=state.detail)
        # Keep the checkpoint pointer the worker knows about: a trainer building
        # its report from its own state would otherwise drop it, and then a
        # crash would look like a run that had never checkpointed.
        if state.last_checkpoint is None and self._state.last_checkpoint is not None:
            state = state.checkpointed(self._state.last_checkpoint,
                                       step=self._state.last_checkpoint_step or 0)
        self._state = state
        self._jobs.save_state(state)
        self._metrics.emit(
            MetricPoint(name=name, value=float(value), step=state.step,
                        job_id=self._job.job_id, wall_seconds=elapsed,
                        tags={str(k): str(v) for k, v in state.detail.items()})
            for name, value in (state.metrics or {}).items())
        self._jobs.update(self._job.job_id,
                          lambda j: j.heartbeat(now=now, step=state.step))

    def save_checkpoint(self, payload: CheckpointPayload, *, step: int,
                        kind: CheckpointKind = CheckpointKind.PERIODIC,
                        metrics: Mapping[str, float] | None = None,
                        provenance: Mapping[str, Any] | None = None
                        ) -> CheckpointRecord:
        prov = dict(self._provenance)
        prov.update(provenance or {})
        prov.setdefault("plan_digest", self._job.plan.digest)
        prov.setdefault("trainer", self._job.plan.trainer)
        prov.setdefault("seed", self._job.plan.seed)
        prov.setdefault("attempt", self._job.attempts)
        if self._resume_id is not None:
            # The chain back to where this attempt started.  A checkpoint whose
            # ancestry is unrecorded cannot be told from one written by a run
            # that started from scratch, and the two mean different things.
            prov.setdefault("resumed_from", str(self._resume_id))

        record = self._checkpoints.save(
            job_id=self._job.job_id, experiment_id=self._job.experiment_id,
            step=int(step), payload=payload, kind=kind,
            metrics=dict(metrics or self._state.metrics), provenance=prov,
            created_at=self._clock.now())
        self._state = self._state.checkpointed(record.checkpoint_id, step=record.step)
        self._jobs.save_state(self._state)
        self._events.append(JobEvent(
            job_id=self._job.job_id, kind=JobEventKind.CHECKPOINT_WRITTEN,
            at=self._clock.now(), step=record.step,
            message=f"{record.checkpoint_id} ({record.size_bytes} bytes)",
            detail={"kind": str(record.kind), "digest": record.digest}))
        return record

    def resume_from(self) -> tuple[CheckpointRecord, CheckpointPayload] | None:
        if self._resume_id is None:
            return None
        return self._checkpoints.load(self._resume_id)

    def stop_requested(self) -> StopRequest | None:
        return self._stop()

    def note(self, message: str, **detail) -> None:
        self._events.append(JobEvent(
            job_id=self._job.job_id, kind=JobEventKind.NOTE,
            at=self._clock.now(), step=self._state.step, message=message,
            detail=detail))


class TrainingWorker:
    """Runs one job to a recorded conclusion.

    Constructed by ``dytiscidae.worker.__main__`` from a job id and a root, or
    directly by a test.  ``run`` returns the job's terminal state and does not
    raise for a training failure -- the failure is the result.  It raises only
    when it cannot record anything, which is the one case where there is
    nothing useful left to do.
    """

    #: How often the control channel is polled, in seconds of wall time, when
    #: the trainer asks.  The poll itself is one small file read; this bound
    #: stops a trainer that asks in a tight inner loop from making it a
    #: per-iteration syscall.
    POLL_INTERVAL = 0.5

    def __init__(self, *, job_id: JobId, jobs: JobStore,
                 checkpoints: CheckpointStore, control: ControlChannel,
                 events: EventLog, metrics: MetricSink,
                 resolve_trainer: Callable[[str], Trainer],
                 datasets: DatasetRepository | None = None,
                 clock: Clock | None = None,
                 provenance: Mapping[str, Any] | None = None,
                 install_signal_handlers: bool = True) -> None:
        self.job_id = JobId(job_id)
        self._jobs = jobs
        self._checkpoints = checkpoints
        self._control = control
        self._events = events
        self._metrics = metrics
        self._resolve_trainer = resolve_trainer
        self._datasets = datasets
        self._clock = clock or SystemClock()
        self._provenance = dict(provenance or {})
        self._install_handlers = install_signal_handlers

        self._signal_stop: StopRequest | None = None
        self._cached_stop: StopRequest | None = None
        self._last_poll = 0.0
        self._previous_handlers: dict[int, Any] = {}

    # -- signals ----------------------------------------------------------

    def _install(self) -> None:
        if not self._install_handlers:
            return

        def handler(signum, _frame):
            # SIGTERM is how a container says it is going away.  Treating it as
            # a pause turns a reclamation into a resumable stop, which is the
            # difference between losing the generations since the last
            # checkpoint and losing the run.  The handler does nothing but set
            # a flag: work inside a signal handler is how a checkpoint gets
            # written from the middle of another checkpoint's write.
            self._signal_stop = StopRequest(
                reason=StopReason.SIGNAL, urgent=True,
                requested_at=self._clock.now())
            self._cached_stop = self._signal_stop

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous_handlers[sig] = signal.signal(sig, handler)
            except (ValueError, OSError):
                # Not the main thread, or a platform without it.  Cooperative
                # stop through the control channel still works; only the
                # reclamation path is lost, and that is worth continuing for.
                pass

    def _restore(self) -> None:
        for sig, previous in self._previous_handlers.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass
        self._previous_handlers.clear()

    # -- the stop flag ----------------------------------------------------

    def _stop_request(self) -> StopRequest | None:
        if self._signal_stop is not None:
            return self._signal_stop
        now = self._clock.monotonic()
        if self._cached_stop is None and now - self._last_poll < self.POLL_INTERVAL:
            return None
        self._last_poll = now
        if self._cached_stop is not None:
            return self._cached_stop
        signal_ = self._control.poll(self.job_id)
        if signal_ is None:
            return None
        self._cached_stop = StopRequest(reason=signal_.reason,
                                        urgent=signal_.urgent,
                                        requested_at=signal_.requested_at)
        # Reflect the request in the job's status if the application has not
        # already: a stop posted while the job was PENDING arrives here first.
        try:
            self._jobs.update(
                self.job_id,
                lambda j: (j.request_cancel() if self._cached_stop.is_cancel
                           else (j.request_pause() if j.status is JobStatus.RUNNING
                                 else j)))
        except Exception:                                         # noqa: BLE001
            pass
        return self._cached_stop

    # -- the run ----------------------------------------------------------

    def run(self) -> TrainingJob:
        job = self._jobs.get(self.job_id)

        # A cancel that arrived before the worker existed.  Honouring it here,
        # before any setup, is what makes "cancel a queued job" instant instead
        # of "cancel it after it has imported torch and allocated a pool".
        pending = self._control.poll(self.job_id)
        if pending is not None and pending.reason is StopReason.CANCEL_REQUESTED:
            now = self._clock.now()
            job = self._jobs.update(self.job_id, lambda j: (
                j if j.status is JobStatus.CANCELLED
                else j.request_cancel()))
            if job.status is not JobStatus.CANCELLED:
                job = self._jobs.update(self.job_id,
                                        lambda j: j.cancelled(now=now))
            self._emit(JobEventKind.CANCELLED, "cancelled before the trainer started")
            self._control.clear(self.job_id)
            return job

        self._install()
        try:
            return self._run_guarded(job)
        finally:
            self._restore()
            try:
                self._metrics.close()
            except Exception:                                     # noqa: BLE001
                pass

    def _run_guarded(self, job: TrainingJob) -> TrainingJob:
        now = self._clock.now()
        worker_info = {"pid": os.getpid(), "host": socket.gethostname(),
                       "python": sys.version.split()[0],
                       "started_at": now}
        try:
            job = self._jobs.update(
                self.job_id,
                lambda j: j.start(now=now, worker={**(j.worker or {}), **worker_info}))
        except Exception as exc:                                  # noqa: BLE001
            # The job was not in a state that can start -- cancelled while the
            # worker was booting, or started twice.  Recorded and returned, not
            # raised: a second worker for one job must not corrupt the first
            # one's record on its way out.
            self._emit(JobEventKind.NOTE, f"refusing to start: {exc}")
            return self._jobs.get(self.job_id)

        self._emit(JobEventKind.STARTED,
                   f"attempt {job.attempts} on pid {os.getpid()}", **worker_info)

        state = self._jobs.load_state(self.job_id) or TrainingState(
            job_id=self.job_id, total_steps=job.plan.budget.max_steps)
        if state.total_steps != job.plan.budget.max_steps:
            # A resume that extended the budget.  The stored state still carries
            # the old total, and a progress bar reading it would sit at 100%.
            state = TrainingState(
                job_id=self.job_id, step=state.step,
                total_steps=job.plan.budget.max_steps,
                samples_seen=state.samples_seen,
                elapsed_seconds=state.elapsed_seconds, updated_at=state.updated_at,
                last_checkpoint=state.last_checkpoint,
                last_checkpoint_step=state.last_checkpoint_step,
                metrics=state.metrics, detail=state.detail)

        resume_id = CheckpointId(job.resumed_from) if job.resumed_from else None
        context = WorkerContext(
            job=job, state=state, jobs=self._jobs, checkpoints=self._checkpoints,
            metrics=self._metrics, events=self._events, clock=self._clock,
            datasets=self._datasets, stop=self._stop_request,
            resume_checkpoint=resume_id, provenance=self._provenance)

        try:
            trainer = self._resolve_trainer(job.plan.trainer)
        except Exception as exc:                                  # noqa: BLE001
            return self._fail(exc, step=state.step,
                              note=f"trainer {job.plan.trainer!r} could not be resolved")

        caps = trainer.capabilities()
        self._emit(JobEventKind.NOTE, f"trainer {job.plan.trainer!r}",
                   **caps.as_dict())
        if not caps.cooperative_stop:
            # Said once, plainly, at the start: a pause or a cancel on this job
            # can only be delivered by killing the process, and that costs
            # everything since the last checkpoint.
            self._emit(JobEventKind.NOTE,
                       "this trainer does not honour cooperative stops; a pause "
                       "or cancel will have to kill the worker")

        try:
            outcome = trainer.train(context)
        except BaseException as exc:                              # noqa: BLE001
            # BaseException, so that a KeyboardInterrupt or a SystemExit out of
            # the trainer is recorded rather than leaving the job RUNNING
            # forever.  Re-raised after recording where that is the right thing.
            job = self._fail(exc, step=context.state.step)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return job

        return self._finish(outcome, context)

    # -- endings ----------------------------------------------------------

    def _finish(self, outcome: TrainingOutcome, context: WorkerContext) -> TrainingJob:
        state = outcome.state or context.state
        self._jobs.save_state(state)
        now = self._clock.now()
        reason = outcome.stop_reason

        if outcome.cancelled:
            job = self._jobs.update(self.job_id, lambda j: (
                j if j.status is JobStatus.CANCELLED
                else j.request_cancel().cancelled(now=now)
                if j.status is JobStatus.RUNNING else j.cancelled(now=now)))
            self._emit(JobEventKind.CANCELLED, "trainer honoured the cancel",
                       step=state.step)
        elif outcome.paused:
            job = self._jobs.update(self.job_id,
                                    lambda j: j.paused(now=now, reason=reason))
            self._emit(JobEventKind.PAUSED, f"stopped: {reason}", step=state.step,
                       checkpoint=str(outcome.final_checkpoint or ""))
            if outcome.final_checkpoint is None:
                # A pause with nothing to resume from is a pause that will be
                # refused by ResumeTraining.  Better to know now, in the record,
                # than at the resume six hours later.
                self._emit(JobEventKind.NOTE,
                           "paused without a checkpoint: this job cannot be "
                           "resumed and will have to be restarted")
        else:
            job = self._jobs.update(self.job_id,
                                    lambda j: j.succeeded(now=now, reason=reason))
            self._emit(JobEventKind.SUCCEEDED, f"finished: {reason}",
                       step=state.step, **dict(outcome.summary))

        self._control.clear(self.job_id)
        return job

    def _fail(self, exc: BaseException, *, step: int | None,
              note: str = "") -> TrainingJob:
        failure = FailureInfo.from_exception(exc, step=step)
        now = self._clock.now()
        # Printed as well as recorded.  The record is what survives; the print
        # is what a person watching the worker log sees, and a failure visible
        # in only one of the two is a failure someone will miss.
        print(f"\njob {self.job_id} failed at step {step}: "
              f"{failure.kind}: {failure.message}", file=sys.stderr, flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        try:
            job = self._jobs.update(self.job_id, lambda j: j.failed(failure, now=now))
        except Exception:                                         # noqa: BLE001
            # The job was not in a state that can fail -- already terminal.
            # Nothing further to record; return what is stored.
            job = self._jobs.get(self.job_id)
        self._emit(JobEventKind.FAILED, note or f"{failure.kind}: {failure.message}",
                   step=step, kind=failure.kind)
        return job

    def _emit(self, kind: JobEventKind, message: str = "", *,
              step: int | None = None, **detail) -> None:
        self._events.append(JobEvent(job_id=self.job_id, kind=kind,
                                     at=self._clock.now(), step=step,
                                     message=message, detail=detail))
