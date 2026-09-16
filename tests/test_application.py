"""Use case tests: the application layer against in-memory adapters.

The middle tier.  No disk, no subprocess, no clock that moves on its own -- so
the whole file runs in under a second and every duration in it is exact.

What these pin down is the one rule the layer exists to keep: **the application
decides that something has been asked for; the worker decides that it
happened.**  Most of the checks below are a variation on that.  A use case that
sets a job RUNNING at launch, or PAUSED at the moment a pause is requested,
would pass a naive test and would be reporting a state it is in no position to
observe -- so the tests assert on the *sequence* of statuses a job holds, not
only on where it ends up.

Run:  PYTHONPATH=. python tests/test_application.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dytiscidae.application import (                              # noqa: E402
    CancelTrainingRequest, CreateExperimentRequest, DuplicateExperimentName,
    ExportModelRequest, ExportNotSupported, LoadCheckpointRequest,
    NoCheckpointToResume, PauseTrainingRequest, ResumeTrainingRequest,
    StartTrainingRequest, TrainingApplication,
)
from dytiscidae.domain import (                                   # noqa: E402
    CheckpointKind, ExperimentNotFound, FailureInfo, IllegalTransition, JobId,
    JobStatus, StopReason, TrainingBudget, TrainingPlan,
)
from dytiscidae.ports.launcher import WorkerStatus                # noqa: E402
from dytiscidae.ports.trainer import TrainerCapabilities          # noqa: E402
from tests.fakes import (                                          # noqa: E402
    FakeCheckpointStore, FakeClock, FakeControlChannel, FakeEventLog,
    FakeExperimentStore, FakeJobStore, FakeLauncher, RecordingLauncher,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def _plan(**kw) -> TrainingPlan:
    kw.setdefault("trainer", "synthetic")
    kw.setdefault("seed", 5)
    kw.setdefault("budget", TrainingBudget(max_steps=20))
    return TrainingPlan(**kw)


class _Lab:
    """One application over one set of fakes, with the fakes reachable."""

    def __init__(self, *, launcher=None, resolve_trainer=None) -> None:
        self.clock = FakeClock()
        self.jobs = FakeJobStore()
        self.experiments = FakeExperimentStore()
        self.checkpoints = FakeCheckpointStore()
        self.control = FakeControlChannel()
        self.events = FakeEventLog()
        self.launcher = launcher or FakeLauncher()
        self.app = TrainingApplication(
            jobs=self.jobs, experiments=self.experiments,
            checkpoints=self.checkpoints, control=self.control,
            events=self.events, clock=self.clock, launcher=self.launcher,
            resolve_trainer=resolve_trainer,
            provenance=lambda: {"git": "abc1234"},
            workspace_root="/lab")

    def experiment(self, name: str = "arch39", **kw):
        kw.setdefault("plan", _plan())
        return self.app.create(CreateExperimentRequest(name=name, **kw)).experiment

    def started(self, **kw):
        experiment = self.experiment(**{k: v for k, v in kw.items()
                                        if k in ("name", "plan")})
        return self.app.start(StartTrainingRequest(
            experiment_id=experiment.experiment_id,
            resources=kw.get("resources", {}))).job


# --------------------------------------------------------------------------


def test_create_experiment_fixes_the_question_and_its_provenance() -> None:
    print("\ntest_create_experiment_fixes_the_question_and_its_provenance")
    lab = _Lab()
    plan = _plan(hyperparameters={"batch": 16})
    experiment = lab.experiment(name="arch39", plan=plan,
                                hypothesis="thrust flattens aspect ratio",
                                tags=("air",))
    check("the experiment is stored", lab.experiments.get(experiment.experiment_id)
          == experiment)
    check("the plan is carried whole", experiment.plan.digest == plan.digest)
    check("the provenance the composition root supplied is recorded",
          experiment.provenance.get("git") == "abc1234")
    check("the plan digest is in the provenance too",
          experiment.provenance.get("plan_digest") == plan.digest,
          "so a record read without this package still says which configuration")
    check("the creation time comes from the injected clock",
          experiment.created_at == lab.clock.now())
    check("the hypothesis is kept",
          experiment.hypothesis.startswith("thrust"))

    raised = False
    try:
        lab.experiment(name="arch39")
    except DuplicateExperimentName as exc:
        raised = "already exists" in str(exc)
    check("a duplicate name is refused with a message that names the clash",
          raised, "two experiments called arch39 make every later reference ambiguous")

    raised = False
    try:
        lab.app.start(StartTrainingRequest(experiment_name="nope"))
    except ExperimentNotFound:
        raised = True
    check("starting under an experiment that does not exist raises", raised)


def test_start_leaves_the_job_pending_until_a_worker_says_otherwise() -> None:
    """The load-bearing check of the whole layer.

    Between the launch and the worker's first heartbeat is where ``import
    torch`` fails and where a container runs out of memory.  A job that reported
    RUNNING through that window would be wrong for as long as anyone believed
    it.
    """
    print("\ntest_start_leaves_the_job_pending_until_a_worker_says_otherwise")
    lab = _Lab()
    job = lab.started()
    check("the job is PENDING when start returns",
          job.status is JobStatus.PENDING, str(job.status))
    check("it has never been RUNNING",
          "running" not in lab.jobs.statuses(job.job_id),
          " -> ".join(lab.jobs.statuses(job.job_id)))
    check("a worker was launched", lab.launcher.launched == [job.job_id])
    check("the workspace defaults under the lab root",
          job.workspace == f"/lab/jobs/{job.job_id}", job.workspace)
    check("the handle is recorded on the job so it can be found again",
          (job.worker or {}).get("handle", {}).get("pid") is not None)
    check("the job is attached to its experiment",
          job.job_id in lab.experiments.get(job.experiment_id).job_ids)
    check("the events say created then launched, in that order",
          lab.events.kinds(job.job_id) == ["created", "launched"],
          str(lab.events.kinds(job.job_id)))
    check("stale control state is cleared before the worker can read it",
          str(job.job_id) in lab.control.cleared,
          "a leftover control file would stop the run at its first boundary")

    # Resources stay off the plan: the same experiment on two machines is one
    # configuration.  This project measured pool shape at 3x on wall time and
    # nothing on the result.
    other = _Lab()
    a = other.started(name="a", resources={"workers": 4, "min_shard": 4})
    b = other.started(name="b", resources={"workers": 16, "min_shard": 1})
    check("worker count reaches the job", a.resources["workers"] == 4)
    check("two pool shapes are still one configuration",
          a.plan.digest == b.plan.digest)


def test_a_launch_that_throws_is_recorded_as_a_failed_job() -> None:
    """A job created and then abandoned because the launcher threw is a PENDING
    row nobody will ever explain."""
    print("\ntest_a_launch_that_throws_is_recorded_as_a_failed_job")
    lab = _Lab(launcher=FakeLauncher(fail=OSError("no space left on device")))
    experiment = lab.experiment()
    raised = False
    try:
        lab.app.start(StartTrainingRequest(experiment_id=experiment.experiment_id))
    except OSError:
        raised = True
    check("the exception still reaches the caller", raised)

    jobs = lab.jobs.list()
    check("the job exists rather than being lost", len(jobs) == 1)
    job = jobs[0]
    check("it is FAILED, not PENDING", job.status is JobStatus.FAILED,
          str(job.status))
    check("the failure names the cause",
          job.failure is not None and "no space" in job.failure.message)
    check("the event log says so", "failed" in lab.events.kinds(job.job_id))


def test_pause_posts_a_request_and_does_not_claim_it_happened() -> None:
    print("\ntest_pause_posts_a_request_and_does_not_claim_it_happened")
    lab = _Lab()
    job = lab.started()
    lab.jobs.update(job.job_id, lambda j: j.start(now=lab.clock.now()))  # the worker

    response = lab.app.pause(PauseTrainingRequest(job_id=job.job_id, requested_by="me"))
    check("the job is PAUSING, not PAUSED",
          response.job.status is JobStatus.PAUSING, str(response.job.status))
    check("a request is posted on the control channel",
          lab.control.poll(job.job_id) is not None)
    check("the request carries who asked",
          lab.control.poll(job.job_id).requested_by == "me")
    check("the reason is a pause, not a cancel",
          lab.control.poll(job.job_id).reason is StopReason.PAUSE_REQUESTED)

    again = lab.app.pause(PauseTrainingRequest(job_id=job.job_id))
    check("a second pause is idempotent rather than an error",
          again.job.status is JobStatus.PAUSING)

    # The worker acknowledging is what reaches PAUSED.
    lab.jobs.update(job.job_id, lambda j: j.paused(now=lab.clock.now()))
    settled = lab.app.pause(PauseTrainingRequest(job_id=job.job_id))
    check("pausing an already-paused job reports it rather than raising",
          settled.already_stopped and settled.job.status is JobStatus.PAUSED)

    # A pause of a job whose worker has not started yet still has to land.
    fresh = _Lab()
    pending = fresh.started()
    fresh.app.pause(PauseTrainingRequest(job_id=pending.job_id))
    check("a pause on a PENDING job posts the request anyway",
          fresh.control.poll(pending.job_id) is not None,
          "the worker honours it at its first boundary")
    check("and leaves the status alone, because nothing is running",
          fresh.jobs.get(pending.job_id).status is JobStatus.PENDING)


def test_cancel_asks_first_and_only_kills_when_told_to() -> None:
    print("\ntest_cancel_asks_first_and_only_kills_when_told_to")
    lab = _Lab()
    job = lab.started()
    lab.jobs.update(job.job_id, lambda j: j.start(now=lab.clock.now()))

    response = lab.app.cancel(CancelTrainingRequest(job_id=job.job_id))
    check("a plain cancel leaves the job CANCELLING",
          response.job.status is JobStatus.CANCELLING, str(response.job.status))
    check("no worker was killed", lab.launcher.terminated == [],
          "killing costs everything since the last checkpoint")
    check("the request is on the channel and is urgent",
          lab.control.poll(job.job_id).reason is StopReason.CANCEL_REQUESTED
          and lab.control.poll(job.job_id).urgent)
    check("a cancel overrides a pending pause",
          lab.control.poll(job.job_id).reason is StopReason.CANCEL_REQUESTED)

    # Now with force.
    forced_lab = _Lab()
    forced_job = forced_lab.started()
    forced_lab.jobs.update(forced_job.job_id,
                           lambda j: j.start(now=forced_lab.clock.now()))
    forced = forced_lab.app.cancel(CancelTrainingRequest(
        job_id=forced_job.job_id, force=True, grace_seconds=2.0))
    check("force terminates the worker",
          forced_lab.launcher.terminated == [(forced_job.job_id, 2.0)])
    check("and the job goes terminal", forced.job.status is JobStatus.CANCELLED)
    check("the response says it was forced", forced.forced)
    check("the event log says what it cost",
          any("lost" in e.message for e in forced_lab.events.read(forced_job.job_id)))

    # A worker that will not die keeps the job CANCELLING, which is honest.
    stubborn_lab = _Lab(launcher=FakeLauncher(terminates=False))
    stubborn = stubborn_lab.started()
    stubborn_lab.jobs.update(stubborn.job_id,
                             lambda j: j.start(now=stubborn_lab.clock.now()))
    result = stubborn_lab.app.cancel(CancelTrainingRequest(
        job_id=stubborn.job_id, force=True, grace_seconds=0.0))
    check("a worker that survives the kill leaves the job CANCELLING",
          result.job.status is JobStatus.CANCELLING,
          "the request stands and has not been honoured")

    # A pending job has nothing to acknowledge it, so it goes terminal directly.
    pending_lab = _Lab()
    pending = pending_lab.started()
    done = pending_lab.app.cancel(CancelTrainingRequest(job_id=pending.job_id))
    check("cancelling a job whose worker never started is immediate",
          done.job.status is JobStatus.CANCELLED)
    check("and the stale request is cleared behind it",
          pending_lab.control.poll(pending.job_id) is None)

    check("cancelling a terminal job is a no-op, not an error",
          pending_lab.app.cancel(CancelTrainingRequest(job_id=pending.job_id)
                                 ).job.status is JobStatus.CANCELLED)


def test_resume_refuses_without_a_checkpoint_and_continues_with_one() -> None:
    print("\ntest_resume_refuses_without_a_checkpoint_and_continues_with_one")
    lab = _Lab()
    job = lab.started()
    lab.jobs.update(job.job_id, lambda j: j.start(now=lab.clock.now()))
    lab.jobs.update(job.job_id,
                    lambda j: j.failed(FailureInfo("OOM", "over the ceiling"),
                                       now=lab.clock.now()))

    raised = False
    try:
        lab.app.resume(ResumeTrainingRequest(job_id=job.job_id))
    except NoCheckpointToResume as exc:
        raised = "restart" in str(exc)
    check("a resume with nothing to resume from is refused, and says why",
          raised, "otherwise the record claims continuity that does not exist")

    record = lab.checkpoints.save(
        job_id=job.job_id, experiment_id=job.experiment_id, step=611,
        payload={"model.json": b"{}"}, kind=CheckpointKind.FINAL)
    response = lab.app.resume(ResumeTrainingRequest(job_id=job.job_id))
    check("a failed job resumes from its checkpoint",
          response.resumed_from.checkpoint_id == record.checkpoint_id)
    check("it is armed PENDING, not RUNNING",
          response.job.status is JobStatus.PENDING, str(response.job.status))
    check("the job records what it resumed from",
          response.job.resumed_from == str(record.checkpoint_id))
    check("a second worker was launched",
          lab.launcher.launched.count(job.job_id) == 2)
    check("the control channel is cleared before the worker starts",
          lab.control.poll(job.job_id) is None,
          "a stale request would pause the resumed run at its first boundary")
    check("the event log records the resume",
          "resumed" in lab.events.kinds(job.job_id))

    # Extending the budget is the one plan change a resume may make.
    lab.jobs.update(job.job_id, lambda j: j.start(now=lab.clock.now()))
    lab.jobs.update(job.job_id, lambda j: j.paused(now=lab.clock.now()))
    extended = lab.app.resume(ResumeTrainingRequest(job_id=job.job_id,
                                                    extend_steps=300))
    check("extend_steps counts from where the run reached, not from the budget",
          extended.job.plan.budget.max_steps == 611 + 300,
          f"{extended.job.plan.budget.max_steps} "
          f"(the old budget was {job.plan.budget.max_steps}, "
          f"the checkpoint was at {record.step})")
    check("and that is a different configuration, honestly digested",
          extended.job.plan.digest != job.plan.digest)

    # A checkpoint belonging to another job is refused.  The job is put back
    # into a resumable state first: the resume above left it PENDING, and a
    # PENDING job is refused for a different reason, which would pass the check
    # for the wrong cause.
    lab.jobs.update(job.job_id, lambda j: j.start(now=lab.clock.now()))
    lab.jobs.update(job.job_id, lambda j: j.paused(now=lab.clock.now()))
    other = lab.checkpoints.save(job_id=JobId("elsewhere"),
                                 experiment_id=job.experiment_id, step=1,
                                 payload={"m": b"1"})
    raised = False
    try:
        lab.app.resume(ResumeTrainingRequest(job_id=job.job_id,
                                             checkpoint_id=other.checkpoint_id))
    except ValueError:
        raised = True
    check("resuming from another job's checkpoint is refused", raised)

    # A succeeded job is not resumable.
    done_lab = _Lab()
    done = done_lab.started()
    done_lab.jobs.update(done.job_id, lambda j: j.start(now=done_lab.clock.now()))
    done_lab.jobs.update(done.job_id, lambda j: j.succeeded(now=done_lab.clock.now()))
    raised = False
    try:
        done_lab.app.resume(ResumeTrainingRequest(job_id=done.job_id))
    except IllegalTransition:
        raised = True
    check("a succeeded job cannot be resumed", raised)


def test_status_reports_the_gap_between_the_record_and_the_process() -> None:
    """A job saying RUNNING with a worker that is gone is the case that matters.

    It is what a reclaimed container looks like from here, and before this it
    was found by hand with ``pgrep``.
    """
    print("\ntest_status_reports_the_gap_between_the_record_and_the_process")
    lab = _Lab()
    job = lab.started()
    lab.jobs.update(job.job_id, lambda j: j.start(now=lab.clock.now()))

    report = lab.app.status(job.job_id)
    check("a live job is not stale", not report.stale)
    check("the worker is reported alive",
          report.worker_status is WorkerStatus.ALIVE)

    lab.clock.advance(3_600)
    lab.launcher.alive = WorkerStatus.GONE
    report = lab.app.status(job.job_id)
    check("RUNNING with a dead worker reads as stale", report.stale,
          "which is what a reclaimed container looks like from here")
    check("the heartbeat age is reported",
          report.heartbeat_age is not None and report.heartbeat_age >= 3_600,
          f"{report.heartbeat_age:.0f}s")

    # UNKNOWN is not GONE.  Acting on the first as though it were the second is
    # how two workers end up writing one run directory.
    lab.launcher.alive = WorkerStatus.UNKNOWN
    check("an unknown worker is not reported as stale",
          not lab.app.status(job.job_id).stale)

    lab.jobs.update(job.job_id, lambda j: j.request_cancel().cancelled(
        now=lab.clock.now()))
    check("a terminal job is never stale", not lab.app.status(job.job_id).stale)


def test_the_record_joins_an_experiment_to_every_attempt_under_it() -> None:
    print("\ntest_the_record_joins_an_experiment_to_every_attempt_under_it")
    lab = _Lab()
    experiment = lab.experiment(name="arch39", tags=("air",))
    first = lab.app.start(StartTrainingRequest(
        experiment_id=experiment.experiment_id)).job
    lab.jobs.update(first.job_id, lambda j: j.start(now=lab.clock.now()))
    lab.checkpoints.save(job_id=first.job_id, experiment_id=experiment.experiment_id,
                         step=40, payload={"m": b"x"})
    lab.jobs.update(first.job_id, lambda j: j.paused(now=lab.clock.now()))
    lab.app.resume(ResumeTrainingRequest(job_id=first.job_id))
    second = lab.app.start(StartTrainingRequest(
        experiment_name="arch39",
        hyperparameter_overrides={"batch": 32})).job

    described = lab.app.describe(experiment.experiment_id, with_events=True)
    check("both attempts are under the one experiment",
          len(described["jobs"]) == 2, str(len(described["jobs"])))
    check("a resume is a second attempt on the same job, not a second job",
          lab.jobs.get(first.job_id).attempts == 1,
          "the attempt count rises when the worker starts, not when it is armed")
    check("an override is recorded as its own configuration",
          second.plan.digest != experiment.plan.digest,
          "a run with an override is not a rerun of the experiment's plan")
    check("the experiment's own plan is untouched by the override",
          lab.experiments.get(experiment.experiment_id).plan.digest
          == experiment.plan.digest)
    check("the description carries each job's events when asked",
          all("events" in row for row in described["jobs"]))
    check("listing by tag finds it",
          [e.name for e in lab.app.experiment_list(tag="air")] == ["arch39"])
    check("listing jobs by status works",
          len(lab.app.job_list(status=JobStatus.PENDING)) >= 1)


def test_load_checkpoint_resolves_by_id_or_by_job() -> None:
    print("\ntest_load_checkpoint_resolves_by_id_or_by_job")
    lab = _Lab()
    job = lab.started()
    early = lab.checkpoints.save(job_id=job.job_id, experiment_id=job.experiment_id,
                                 step=10, payload={"model.json": b'{"v":1}'})
    best = lab.checkpoints.save(job_id=job.job_id, experiment_id=job.experiment_id,
                                step=20, payload={"model.json": b'{"v":2}'},
                                kind=CheckpointKind.BEST)
    late = lab.checkpoints.save(job_id=job.job_id, experiment_id=job.experiment_id,
                                step=30, payload={"model.json": b'{"v":3}'})

    by_id = lab.app.load(LoadCheckpointRequest(checkpoint_id=early.checkpoint_id))
    check("by id returns that one", by_id.record.step == 10)
    check("and its payload", by_id.payload["model.json"] == b'{"v":1}')

    by_job = lab.app.load(LoadCheckpointRequest(job_id=job.job_id))
    check("by job returns the highest step, not the last written",
          by_job.record.checkpoint_id == late.checkpoint_id, str(by_job.record))

    by_kind = lab.app.load(LoadCheckpointRequest(job_id=job.job_id,
                                                 kind=CheckpointKind.BEST))
    check("by kind finds the best one even though it is not the latest",
          by_kind.record.checkpoint_id == best.checkpoint_id)

    record_only = lab.app.load(LoadCheckpointRequest(job_id=job.job_id,
                                                     with_payload=False))
    check("a record can be fetched without the bytes",
          record_only.payload is None)

    for bad in (LoadCheckpointRequest(),
                LoadCheckpointRequest(checkpoint_id=early.checkpoint_id,
                                      job_id=job.job_id)):
        raised = False
        try:
            lab.app.load(bad)
        except ValueError:
            raised = True
        check("exactly one of checkpoint_id and job_id is required", raised)


def test_export_is_dispatched_to_the_trainer_that_wrote_the_checkpoint() -> None:
    """An exporter here would have to guess at the payload, and a wrong guess
    produces a file that loads and is not the model."""
    print("\ntest_export_is_dispatched_to_the_trainer_that_wrote_the_checkpoint")
    calls: list[tuple] = []

    class Exporter:
        def capabilities(self):
            return TrainerCapabilities(exports=True)

        def train(self, context):
            raise AssertionError("not called")

        def export(self, record, payload, *, destination, fmt):
            calls.append((record.step, sorted(payload), destination, fmt))
            return {"bytes_written": 42}

    class NonExporter(Exporter):
        def capabilities(self):
            return TrainerCapabilities(exports=False)

    lab = _Lab(resolve_trainer=lambda name: Exporter())
    job = lab.started()
    lab.checkpoints.save(job_id=job.job_id, experiment_id=job.experiment_id,
                         step=900, payload={"model.json": b"{}"})
    response = lab.app.export(ExportModelRequest(destination="/tmp/out",
                                                 job_id=job.job_id, fmt="native"))
    check("the trainer's exporter was called with the payload",
          calls == [(900, ["model.json"], "/tmp/out", "native")], str(calls))
    check("the response reports what was written", response.bytes_written == 42)
    check("and which checkpoint it came from", response.record.step == 900)

    refusing = _Lab(resolve_trainer=lambda name: NonExporter())
    other = refusing.started()
    refusing.checkpoints.save(job_id=other.job_id,
                              experiment_id=other.experiment_id, step=1,
                              payload={"m": b"1"})
    raised = False
    try:
        refusing.app.export(ExportModelRequest(destination="/tmp/x",
                                               job_id=other.job_id))
    except ExportNotSupported as exc:
        raised = "LoadCheckpoint" in str(exc)
    check("a trainer that does not export says so, and names the alternative",
          raised)


def test_the_worker_is_what_moves_a_job_through_its_states() -> None:
    """Drives the launcher's callback as a worker would, and checks the whole
    sequence of statuses the job holds.

    The sequence, not the endpoint: a layer that wrote RUNNING at launch and
    SUCCEEDED at the end would agree on both endpoints and be wrong throughout.
    """
    print("\ntest_the_worker_is_what_moves_a_job_through_its_states")
    lab = None

    def be_a_worker(job_id: JobId) -> None:
        lab.jobs.update(job_id, lambda j: j.start(now=lab.clock.now()))
        lab.clock.advance(74)
        lab.jobs.update(job_id, lambda j: j.heartbeat(now=lab.clock.now(), step=1))
        lab.jobs.update(job_id, lambda j: j.succeeded(now=lab.clock.now()))

    lab = _Lab(launcher=RecordingLauncher(lambda jid: be_a_worker(jid)))
    job = lab.started()
    check("the job ends SUCCEEDED", lab.jobs.get(job.job_id).status
          is JobStatus.SUCCEEDED)
    check("and it went pending -> running -> succeeded, in that order",
          lab.jobs.statuses(job.job_id) == ["pending", "running", "succeeded"],
          " -> ".join(lab.jobs.statuses(job.job_id)))
    check("the recorded duration comes from the injected clock",
          lab.jobs.get(job.job_id).ended_at
          - lab.jobs.get(job.job_id).started_at == 74.0)


def main() -> int:
    print("=" * 68)
    print("application: the use cases, over in-memory ports")
    print("=" * 68)
    test_create_experiment_fixes_the_question_and_its_provenance()
    test_start_leaves_the_job_pending_until_a_worker_says_otherwise()
    test_a_launch_that_throws_is_recorded_as_a_failed_job()
    test_pause_posts_a_request_and_does_not_claim_it_happened()
    test_cancel_asks_first_and_only_kills_when_told_to()
    test_resume_refuses_without_a_checkpoint_and_continues_with_one()
    test_status_reports_the_gap_between_the_record_and_the_process()
    test_the_record_joins_an_experiment_to_every_attempt_under_it()
    test_load_checkpoint_resolves_by_id_or_by_job()
    test_export_is_dispatched_to_the_trainer_that_wrote_the_checkpoint()
    test_the_worker_is_what_moves_a_job_through_its_states()

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all application checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
