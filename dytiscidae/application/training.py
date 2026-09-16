"""The four lifecycle use cases: Start, Pause, Resume, Cancel.

They are in one module because they share one invariant, and splitting them
would let it drift: **the application never decides that a job is running or
stopped -- it decides that a job has been asked to.**  The worker decides the
rest, and the domain's PAUSING and CANCELLING states are where the gap between
those two lives.

Read in that light, each use case is short:

``StartTraining``   create a PENDING job, clear stale control state, launch.
``PauseTraining``   move to PAUSING and post a request.
``CancelTraining``  move to CANCELLING and post a request; kill only if asked.
``ResumeTraining``  find a checkpoint, arm the job PENDING against it, launch.

None of the four ever writes RUNNING.  A job becomes RUNNING when its worker
says so, and that symmetry is why a resume whose worker dies in ``import
torch`` is visible as a PENDING job with a dead worker rather than as a RUNNING
job that is doing nothing.

The one place this layer does more than post a request is ``ResumeTraining``,
which refuses to resume without a checkpoint.  That refusal is the reason
``resumed_from`` on a job can be trusted: a resume that silently started from
scratch would produce a record claiming continuity across a boundary where the
network was reinitialised, which is the exact defect this project's portable
checkpoint was written to close.
"""

from __future__ import annotations

from typing import Any, Callable

from ..domain.errors import DomainError, IllegalTransition
from ..domain.ids import ExperimentId, JobId, new_id
from ..domain.job import JobStatus, StopReason, TrainingJob
from ..domain.metrics import JobEvent, JobEventKind
from ..domain.plan import TrainingBudget
from ..ports.checkpoint_store import CheckpointStore
from ..ports.clock import Clock
from ..ports.control import ControlChannel, ControlSignal
from ..ports.event_log import EventLog
from ..ports.experiment_store import ExperimentStore
from ..ports.job_store import JobStore
from ..ports.launcher import JobLauncher, WorkerStatus
from .dto import (
    CancelTrainingRequest,
    CancelTrainingResponse,
    PauseTrainingRequest,
    PauseTrainingResponse,
    ResumeTrainingRequest,
    ResumeTrainingResponse,
    StartTrainingRequest,
    StartTrainingResponse,
)


class NoCheckpointToResume(DomainError):
    def __init__(self, job_id: JobId) -> None:
        super().__init__(
            f"job {job_id} has no checkpoint to resume from. Resuming without "
            f"one is a restart: it would record continuity that does not "
            f"exist. Start a new job instead.")


class LauncherUnavailable(DomainError):
    pass


def _default_workspace(root: str) -> Callable[[JobId], str]:
    # A string join rather than ``pathlib``: the application layer does not
    # touch the filesystem, and the adapter that does is free to interpret this
    # however it stores things.  Forward slashes are accepted by Python's path
    # handling on every platform this runs on.
    root = root.rstrip("/") if root else "runs"
    return lambda job_id: f"{root}/jobs/{job_id}"


class _LifecycleBase:
    """Shared wiring.  Not a use case; it has no ``execute``."""

    def __init__(self, *, jobs: JobStore, experiments: ExperimentStore,
                 checkpoints: CheckpointStore, control: ControlChannel,
                 events: EventLog, clock: Clock,
                 launcher: JobLauncher | None = None,
                 inline_launcher: JobLauncher | None = None,
                 workspace_root: str = "runs",
                 workspace_for: Callable[[JobId], str] | None = None) -> None:
        self._jobs = jobs
        self._experiments = experiments
        self._checkpoints = checkpoints
        self._control = control
        self._events = events
        self._clock = clock
        self._launcher = launcher
        self._inline = inline_launcher
        self._workspace_for = workspace_for or _default_workspace(workspace_root)

    def _pick_launcher(self, inline: bool) -> JobLauncher:
        chosen = self._inline if inline else self._launcher
        if chosen is None:
            which = "inline" if inline else "process"
            raise LauncherUnavailable(
                f"no {which} launcher is configured on this application")
        return chosen

    def _emit(self, job_id: JobId, kind: JobEventKind, message: str = "",
              **detail) -> None:
        self._events.append(JobEvent(job_id=job_id, kind=kind, at=self._clock.now(),
                                     message=message, detail=detail))

    def _launch(self, job: TrainingJob, *, inline: bool) -> tuple[TrainingJob, Any]:
        """Start a worker and record its handle on the job.

        Failure to launch is recorded as a failure *of the job*, not raised into
        the caller's lap alone.  A job created in the store and then abandoned
        because the launcher threw is a PENDING row nobody will ever explain.
        """
        launcher = self._pick_launcher(inline)
        try:
            handle = launcher.launch(job.job_id, workspace=job.workspace)
        except Exception as exc:                                  # noqa: BLE001
            from ..domain.job import FailureInfo
            failure = FailureInfo.from_exception(exc, step=None)
            now = self._clock.now()
            self._jobs.update(job.job_id, lambda j: j.failed(failure, now=now))
            self._emit(job.job_id, JobEventKind.FAILED,
                       f"launch failed: {type(exc).__name__}: {exc}")
            raise

        job = self._jobs.update(
            job.job_id,
            lambda j: j.heartbeat(now=self._clock.now(), **{
                "handle": handle.as_dict(), "launcher": launcher.kind}))
        self._emit(job.job_id, JobEventKind.LAUNCHED,
                   f"launched via {launcher.kind}",
                   handle=handle.as_dict())
        return job, handle


class StartTraining(_LifecycleBase):
    """Create a job under an experiment and start a worker for it.

    The job is PENDING when this returns, not RUNNING.  Between the launch and
    the worker's first heartbeat is where ``import torch`` fails, where a
    container has no memory left, and where a bad ``MUJOCO_GL`` kills the
    process -- and a job that reported RUNNING for any of those would be lying
    for as long as anyone believed it.
    """

    def execute(self, request: StartTrainingRequest) -> StartTrainingResponse:
        experiment = self._resolve_experiment(request)
        plan = experiment.plan
        if request.hyperparameter_overrides:
            merged = dict(plan.hyperparameters)
            merged.update(request.hyperparameter_overrides)
            plan = plan.with_(hyperparameters=merged)

        job_id = JobId(new_id("job", now=self._clock.now()))
        job = TrainingJob(
            job_id=job_id,
            experiment_id=experiment.experiment_id,
            plan=plan,
            status=JobStatus.PENDING,
            created_at=self._clock.now(),
            workspace=request.workspace or self._workspace_for(job_id),
            resources=dict(request.resources),
        )
        job = self._jobs.create(job)
        self._experiments.attach_job(experiment.experiment_id, job_id)
        self._emit(job_id, JobEventKind.CREATED,
                   f"job for experiment {experiment.name!r}",
                   plan_digest=plan.digest, trainer=plan.trainer,
                   workspace=job.workspace)

        # A control file left over from a previous job in a reused workspace
        # would be read by this worker at its first boundary and stop it
        # immediately -- a run that appears to start and then quietly pauses.
        self._control.clear(job_id)

        job, handle = self._launch(job, inline=request.inline)
        state = self._jobs.load_state(job_id) if request.inline else None
        if request.inline:
            job = self._jobs.get(job_id)      # the inline worker has finished
        return StartTrainingResponse(job=job, handle=handle, state=state)

    def _resolve_experiment(self, request: StartTrainingRequest):
        if (request.experiment_id is None) == (request.experiment_name is None):
            raise ValueError(
                "give exactly one of experiment_id or experiment_name")
        if request.experiment_id is not None:
            return self._experiments.get(ExperimentId(request.experiment_id))
        found = self._experiments.find_by_name(request.experiment_name or "")
        if found is None:
            from ..domain.errors import ExperimentNotFound
            raise ExperimentNotFound(
                f"no experiment named {request.experiment_name!r}")
        return found


class PauseTraining(_LifecycleBase):
    """Ask a running job to stop at its next safe point, keeping it resumable.

    Returns with the job in PAUSING.  It reaches PAUSED when the worker gets
    there, which is at most one step away -- up to ~74 s for this project's
    search in steady state and ~300 s during the opening verification burst.
    A caller that needs to know it has actually stopped polls the status; there
    is no way to make that instantaneous that does not involve killing the
    process and losing the step.
    """

    def execute(self, request: PauseTrainingRequest) -> PauseTrainingResponse:
        job = self._jobs.get(JobId(request.job_id))
        if job.is_terminal or job.status is JobStatus.PAUSED:
            return PauseTrainingResponse(job=job, already_stopped=True)
        if job.status is JobStatus.PENDING:
            # Nothing is running to acknowledge anything, and the worker may be
            # seconds from starting.  Posting the request rather than refusing
            # means the worker honours it at its first boundary.
            self._control.request(job.job_id, ControlSignal(
                reason=StopReason.PAUSE_REQUESTED, requested_at=self._clock.now(),
                urgent=request.urgent, requested_by=request.requested_by))
            self._emit(job.job_id, JobEventKind.PAUSE_REQUESTED,
                       "requested before the worker started")
            return PauseTrainingResponse(job=job)

        job = self._jobs.update(job.job_id, lambda j: j.request_pause())
        self._control.request(job.job_id, ControlSignal(
            reason=StopReason.PAUSE_REQUESTED, requested_at=self._clock.now(),
            urgent=request.urgent, requested_by=request.requested_by))
        self._emit(job.job_id, JobEventKind.PAUSE_REQUESTED,
                   request.requested_by or "", urgent=request.urgent)
        return PauseTrainingResponse(job=job)


class CancelTraining(_LifecycleBase):
    """Stop a job for good.

    ``force`` exists and defaults to False.  Killing a worker costs every step
    since its last checkpoint, and this project's pool spawns children that
    outlive a signal sent to the parent alone -- measured at two hours and a
    gigabyte of resident memory.  So the default is to ask, and the response
    says plainly whether asking was enough.
    """

    def execute(self, request: CancelTrainingRequest) -> CancelTrainingResponse:
        job = self._jobs.get(JobId(request.job_id))
        if job.is_terminal:
            return CancelTrainingResponse(job=job)

        was_live = job.is_active
        job = self._jobs.update(job.job_id, lambda j: j.request_cancel())
        self._control.request(job.job_id, ControlSignal(
            reason=StopReason.CANCEL_REQUESTED, requested_at=self._clock.now(),
            urgent=True, requested_by=request.requested_by))
        self._emit(job.job_id, JobEventKind.CANCEL_REQUESTED,
                   request.requested_by or "", force=request.force)

        if job.status is JobStatus.CANCELLED:
            # PENDING or PAUSED: there was nothing to acknowledge and the domain
            # took it terminal directly.
            self._emit(job.job_id, JobEventKind.CANCELLED, "nothing was running")
            self._control.clear(job.job_id)
            return CancelTrainingResponse(job=job)

        if not (request.force and was_live):
            return CancelTrainingResponse(job=job)

        return self._force(job, request)

    def _force(self, job: TrainingJob, request: CancelTrainingRequest
               ) -> CancelTrainingResponse:
        handle_dict = (job.worker or {}).get("handle")
        if not handle_dict:
            self._emit(job.job_id, JobEventKind.NOTE,
                       "force requested but no worker handle was recorded")
            return CancelTrainingResponse(job=job, worker_status=WorkerStatus.UNKNOWN)

        from ..ports.launcher import WorkerHandle
        handle = WorkerHandle.from_dict(handle_dict)
        launcher = self._pick_launcher(inline=False)
        gone = launcher.terminate(handle, grace_seconds=request.grace_seconds)
        status = launcher.is_alive(handle)
        if gone or status is WorkerStatus.GONE:
            now = self._clock.now()
            job = self._jobs.update(job.job_id, lambda j: j.cancelled(now=now))
            self._emit(job.job_id, JobEventKind.CANCELLED,
                       "worker terminated; work since the last checkpoint is lost")
            self._control.clear(job.job_id)
            return CancelTrainingResponse(job=job, forced=True, worker_status=status)

        # Still there.  The job stays CANCELLING, which is the honest state:
        # the request stands and the worker has not honoured it.
        self._emit(job.job_id, JobEventKind.NOTE,
                   "worker did not die within the grace period", worker_status=str(status))
        return CancelTrainingResponse(job=job, forced=True, worker_status=status)


class ResumeTraining(_LifecycleBase):
    """Continue a paused or failed job from a checkpoint.

    A failed job is resumable on purpose.  A run that died at generation 611 has
    611 generations of paid-for simulation on disk, and refusing to continue it
    to keep the state machine tidy would throw that away.  What is refused is a
    resume with nothing to resume from -- see ``NoCheckpointToResume``.
    """

    def execute(self, request: ResumeTrainingRequest) -> ResumeTrainingResponse:
        job = self._jobs.get(JobId(request.job_id))
        if not job.is_resumable:
            raise IllegalTransition(
                job.job_id, str(job.status), "running",
                "only a paused or failed job can be resumed")

        if request.checkpoint_id is not None:
            record = self._checkpoints.get(request.checkpoint_id)
            if record.job_id != job.job_id:
                raise ValueError(
                    f"checkpoint {record.checkpoint_id} belongs to job "
                    f"{record.job_id}, not {job.job_id}")
        else:
            record = self._checkpoints.latest(job.job_id)
        if record is None:
            raise NoCheckpointToResume(job.job_id)

        # A stale control signal is what paused this job in the first place.
        # Clearing it before the worker starts is the difference between a
        # resume and a resume that pauses again at its first boundary.
        self._control.clear(job.job_id)

        plan = job.plan
        if request.extend_steps:
            budget = plan.budget
            # From wherever the run actually reached, not from the old budget.
            # A job stopped at step 611 under a budget of 500 (the budget was
            # lowered, or the checkpoint came from an extended attempt) would
            # otherwise be "extended" to 800 and stop immediately -- which looks
            # exactly like a resume that did not work.
            base = max(budget.max_steps or 0, record.step)
            plan = plan.with_(budget=TrainingBudget(
                max_steps=base + int(request.extend_steps),
                max_seconds=budget.max_seconds,
                max_resident_mb=budget.max_resident_mb))

        now = self._clock.now()
        ck = str(record.checkpoint_id)
        resources = (dict(request.resources) if request.resources is not None
                     else dict(job.resources))

        job = self._jobs.update(
            job.job_id,
            lambda j: j.prepare_resume(now=now, checkpoint=ck, plan=plan,
                                       resources=resources))
        self._emit(job.job_id, JobEventKind.RESUMED,
                   f"from {record.checkpoint_id} at step {record.step}",
                   checkpoint=str(record.checkpoint_id), step=record.step)

        job, handle = self._launch(job, inline=request.inline)
        state = self._jobs.load_state(job.job_id) if request.inline else None
        if request.inline:
            job = self._jobs.get(job.job_id)
        return ResumeTrainingResponse(job=job, resumed_from=record,
                                      handle=handle, state=state)
