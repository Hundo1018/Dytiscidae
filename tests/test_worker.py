"""Training runtime tests: real worker processes, doing the six things.

The third tier the architecture calls for, and the one the other two cannot
replace.  Checkpoint, resume, pause, cancel, failure and container reclamation
are all *cross-process* behaviours: they involve one process asking and another
answering, at a boundary, with the answer landing on disk.  A test that fakes
either side of that proves nothing about it.

So every case here starts a real ``python -m dytiscidae.worker``, over a real
filesystem lab, and reads the result out of the store afterwards.  What makes
that affordable is the synthetic trainer: it implements the whole ``Trainer``
contract and its step is a microsecond, so a pause-and-resume that would take
two generations of MuJoCo -- measured at ~74 s each in steady state, ~300 s
during the opening burst -- takes under a second here.

The most important check in the file is
``test_a_resume_continues_the_run_rather_than_restarting_it``.  A resume that
silently restarts looks identical to one that works: the job says RUNNING, the
steps climb, the log looks right.  The only way to tell is to compare the
trajectory against the one an uninterrupted run produced from the same seed --
which is exactly what this project measured the hard way when a stored
``takeoff_height`` of 2.288 m re-measured as 0.000 because the RNG state was
not in the checkpoint.

Run:  PYTHONPATH=. python tests/test_worker.py
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dytiscidae.adapters.composition import Lab, build             # noqa: E402
from dytiscidae.application import (                               # noqa: E402
    CancelTrainingRequest, CreateExperimentRequest, ExportModelRequest,
    LoadCheckpointRequest, PauseTrainingRequest, ResumeTrainingRequest,
    StartTrainingRequest,
)
from dytiscidae.domain import (                                    # noqa: E402
    CheckpointKind, JobId, JobStatus, StopReason, TrainingBudget, TrainingPlan,
)
from dytiscidae.ports.launcher import WorkerStatus                 # noqa: E402

FAILURES: list[str] = []
TEMPS: list[Path] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def temp_root() -> Path:
    path = Path(tempfile.mkdtemp(prefix="worker-"))
    TEMPS.append(path)
    return path


def _plan(steps: int = 12, **hp) -> TrainingPlan:
    hp.setdefault("checkpoint_every", 3)
    return TrainingPlan(trainer="synthetic", seed=20260901,
                        hyperparameters=hp, budget=TrainingBudget(max_steps=steps))


def _lab(root: Path | None = None, store: str = "files"):
    app, lab = build(root or temp_root(), store=store)
    return app, lab


def _start(app, plan, *, name: str = "run", inline: bool = False, **kw):
    experiment = app.create(CreateExperimentRequest(name=name, plan=plan)).experiment
    return app.start(StartTrainingRequest(experiment_id=experiment.experiment_id,
                                          inline=inline, **kw)).job


def _await(app, job_id: JobId, *, want=None, timeout: float = 90.0):
    """Poll until the job reaches a wanted status, or settles, or times out.

    Polling, because that is what a caller genuinely has to do: a stop is
    honoured at the trainer's next boundary and nothing can make that
    instantaneous.  The default set is every state in which nothing further
    will happen on its own.
    """
    want = want or (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED,
                    JobStatus.PAUSED)
    deadline = time.monotonic() + timeout
    report = app.status(job_id)
    while time.monotonic() < deadline:
        report = app.status(job_id)
        if report.job.status in want:
            return report
        time.sleep(0.05)
    return report


def _trajectory(lab: Lab, job_id: JobId) -> list[tuple[int, float]]:
    """The (step, loss) series a job actually produced, from its metrics file."""
    path = Path(lab.root) / "jobs" / str(job_id) / "metrics.jsonl"
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("name") == "loss":
            out.append((int(row["step"]), float(row["value"])))
    return out


# --------------------------------------------------------------------------


def test_a_worker_records_a_completed_run_end_to_end() -> None:
    print("\ntest_a_worker_records_a_completed_run_end_to_end")
    app, lab = _lab()
    job = _start(app, _plan(steps=12))
    check("start returns with the job PENDING", job.status is JobStatus.PENDING)

    report = _await(app, job.job_id)
    check("the worker takes it to SUCCEEDED",
          report.job.status is JobStatus.SUCCEEDED, str(report.job.status))
    check("with the reason recorded",
          report.job.stop_reason is StopReason.COMPLETED)
    check("the worker is gone afterwards",
          report.worker_status is WorkerStatus.GONE)
    check("the job is not stale: it finished, it did not vanish",
          not report.stale)
    check("the full step budget was spent",
          report.state.step == 12, str(report.state.step))
    check("the attempt count is one", report.job.attempts == 1)

    kinds = [e.kind.value for e in lab.events.read(job.job_id)]
    check("the event history reads start to finish",
          kinds[0] == "created" and kinds[1] == "launched"
          and kinds[2] == "started" and kinds[-1] == "succeeded",
          " -> ".join(kinds[:3] + ["..."] + kinds[-1:]))
    check("checkpoints were written on the cadence and at the end",
          len([k for k in kinds if k == "checkpoint_written"]) == 5,
          "four periodic at 3/6/9/12 plus one final")

    records = lab.checkpoints.list(job.job_id)
    check("the last one is FINAL and at the last step",
          records[-1].kind is CheckpointKind.FINAL and records[-1].step == 12)
    check("every checkpoint carries the plan digest",
          all(r.provenance.get("plan_digest") == job.plan.digest for r in records))
    check("and the commit that wrote it",
          all("git" in r.provenance for r in records),
          "a checkpoint that cannot be matched to a commit cannot be trusted")

    loaded = app.load(LoadCheckpointRequest(job_id=job.job_id))
    check("the latest checkpoint loads and verifies",
          loaded.record.step == 12 and b"step" in loaded.payload["model.json"])

    metrics = _trajectory(lab, job.job_id)
    check("a metric was written for every step",
          [s for s, _ in metrics] == list(range(1, 13)), str(len(metrics)))
    check("the loss went down",
          metrics[-1][1] < metrics[0][1],
          f"{metrics[0][1]:.4f} -> {metrics[-1][1]:.4f}")
    lab.close()


def test_a_pause_stops_at_a_boundary_and_leaves_a_resumable_job() -> None:
    print("\ntest_a_pause_stops_at_a_boundary_and_leaves_a_resumable_job")
    app, lab = _lab()
    # Slow enough that the pause lands mid-run rather than after it.
    job = _start(app, _plan(steps=200, step_seconds=0.02, checkpoint_every=5))

    running = _await(app, job.job_id, want=(JobStatus.RUNNING,), timeout=30)
    check("the worker took the job RUNNING",
          running.job.status is JobStatus.RUNNING, str(running.job.status))

    response = app.pause(PauseTrainingRequest(job_id=job.job_id, requested_by="test"))
    check("the request leaves it PAUSING, not PAUSED",
          response.job.status is JobStatus.PAUSING, str(response.job.status))

    report = _await(app, job.job_id, want=(JobStatus.PAUSED,), timeout=60)
    check("the worker honours it and reaches PAUSED",
          report.job.status is JobStatus.PAUSED, str(report.job.status))
    check("the reason says it was asked for",
          report.job.stop_reason is StopReason.PAUSE_REQUESTED)
    check("it stopped well short of the budget",
          0 < report.state.step < 200, f"step {report.state.step} of 200")
    check("the job is resumable", report.job.is_resumable)
    check("a FINAL checkpoint was written at the stopping step",
          report.latest_checkpoint.step == report.state.step
          and report.latest_checkpoint.kind is CheckpointKind.FINAL,
          f"checkpoint at {report.latest_checkpoint.step}, stopped at "
          f"{report.state.step}")
    check("nothing is at risk: the checkpoint is current",
          report.state.steps_at_risk == 0)
    check("the control request was cleared once honoured",
          lab.control.poll(job.job_id) is None,
          "otherwise the resumed run pauses again at its first boundary")
    check("the worker process is gone", report.worker_status is WorkerStatus.GONE)
    lab.close()


def test_a_resume_continues_the_run_rather_than_restarting_it() -> None:
    """The check a silently-broken resume would pass without.

    Two runs from one seed: one uninterrupted, one paused in the middle and
    resumed.  Their (step, loss) trajectories must agree at every step.  A
    resume that restarted, or that reinitialised the generator, produces a
    curve that is *plausible* -- still descending, still converging -- and
    wrong, which is why the comparison is against the reference series and not
    against a property of the series.
    """
    print("\ntest_a_resume_continues_the_run_rather_than_restarting_it")
    plan = _plan(steps=40, checkpoint_every=4)

    # The reference: one run, start to finish, never interrupted.
    app_ref, lab_ref = _lab()
    reference_job = _start(app_ref, plan, name="reference")
    _await(app_ref, reference_job.job_id)
    reference = _trajectory(lab_ref, reference_job.job_id)
    check("the reference run completed", len(reference) == 40, str(len(reference)))

    # The interrupted one: same seed, same plan, stopped and resumed.
    app, lab = _lab()
    job = _start(app, _plan(steps=40, checkpoint_every=4, step_seconds=0.02),
                 name="interrupted")
    _await(app, job.job_id, want=(JobStatus.RUNNING,), timeout=30)
    app.pause(PauseTrainingRequest(job_id=job.job_id))
    paused = _await(app, job.job_id, want=(JobStatus.PAUSED,), timeout=60)
    stopped_at = paused.state.step
    check("it paused partway", 0 < stopped_at < 40, f"step {stopped_at}")

    resumed = app.resume(ResumeTrainingRequest(job_id=job.job_id))
    check("it resumes from the checkpoint it wrote",
          resumed.resumed_from.step == stopped_at)
    final = _await(app, job.job_id, timeout=90)
    check("and runs to completion", final.job.status is JobStatus.SUCCEEDED,
          str(final.job.status))
    check("as a second attempt on the same job", final.job.attempts == 2,
          "a resume is another attempt, not another job")
    check("the job records where it resumed from",
          final.job.resumed_from == str(resumed.resumed_from.checkpoint_id))

    observed = _trajectory(lab, job.job_id)
    check("the resumed run reached the same final step",
          observed[-1][0] == 40, str(observed[-1][0]))

    # The comparison.  The resumed run re-runs from the checkpoint's step, so
    # its series can repeat steps; compare the last value seen at each step.
    by_step = {}
    for step, value in observed:
        by_step[step] = value
    ref_by_step = dict(reference)
    mismatches = [(s, ref_by_step[s], by_step[s]) for s in sorted(ref_by_step)
                  if s in by_step and abs(ref_by_step[s] - by_step[s]) > 1e-12]
    check("every step after the boundary matches the uninterrupted run",
          not mismatches,
          f"{len(mismatches)} of {len(ref_by_step)} differ; first "
          f"{mismatches[0] if mismatches else ''}")
    after = [s for s in by_step if s > stopped_at]
    check("the comparison actually spans the boundary", len(after) >= 4,
          f"{len(after)} steps observed after step {stopped_at}")
    check("the final losses are identical",
          abs(ref_by_step[40] - by_step[40]) < 1e-12,
          f"{ref_by_step[40]:.12f} vs {by_step[40]:.12f}")

    events = [e.message for e in lab.events.read(job.job_id)]
    check("the resume is in the history",
          any("from ck" in m for m in events))
    lab.close()
    lab_ref.close()


def test_a_cancel_stops_the_run_and_does_not_leave_it_resumable() -> None:
    print("\ntest_a_cancel_stops_the_run_and_does_not_leave_it_resumable")
    app, lab = _lab()
    job = _start(app, _plan(steps=500, step_seconds=0.02, checkpoint_every=5))
    _await(app, job.job_id, want=(JobStatus.RUNNING,), timeout=30)

    response = app.cancel(CancelTrainingRequest(job_id=job.job_id,
                                                requested_by="test"))
    check("the request leaves it CANCELLING",
          response.job.status is JobStatus.CANCELLING, str(response.job.status))
    check("no worker was killed to do it", not response.forced,
          "a cooperative trainer is asked, not killed")

    report = _await(app, job.job_id, want=(JobStatus.CANCELLED,), timeout=60)
    check("the worker honours it and reaches CANCELLED",
          report.job.status is JobStatus.CANCELLED, str(report.job.status))
    check("the reason is recorded",
          report.job.stop_reason is StopReason.CANCEL_REQUESTED)
    check("it stopped short of the budget",
          report.state.step < 500, f"step {report.state.step}")
    check("a cancelled job is not resumable", not report.job.is_resumable)
    check("the process is gone", report.worker_status is WorkerStatus.GONE)
    check("the request was cleared", lab.control.poll(job.job_id) is None)

    # A cancel that arrives before the worker exists must not cost an import.
    early_app, early_lab = _lab()
    early = _start(early_app, _plan(steps=5), name="early")
    early_lab.control.clear(early.job_id)
    from dytiscidae.ports.control import ControlSignal
    early_lab.control.request(early.job_id, ControlSignal(
        reason=StopReason.CANCEL_REQUESTED, requested_at=time.time()))
    worker = early_lab.worker(early.job_id, install_signal_handlers=False)
    finished = worker.run()
    check("a cancel posted before the trainer starts is honoured immediately",
          finished.status is JobStatus.CANCELLED, str(finished.status))
    check("and no checkpoint was written for a run that never began",
          early_lab.checkpoints.list(early.job_id) == [])
    lab.close()
    early_lab.close()


def test_a_failing_trainer_lands_the_job_in_failed_with_its_traceback() -> None:
    """A crash that is recorded is a crash somebody can act on.

    The worker exits 0 here, and that is deliberate: it did its job, which was
    to record what happened.  Whether the *training* succeeded is the job's
    status, which is the point of having one.
    """
    print("\ntest_a_failing_trainer_lands_the_job_in_failed_with_its_traceback")
    app, lab = _lab()
    job = _start(app, _plan(steps=20, checkpoint_every=3, fail_at_step=7))

    report = _await(app, job.job_id, timeout=60)
    check("the job is FAILED", report.job.status is JobStatus.FAILED,
          str(report.job.status))
    check("the failure names the exception class",
          report.job.failure.kind == "SyntheticFailure",
          str(report.job.failure.kind if report.job.failure else None))
    check("the message survives",
          "fail_at_step" in report.job.failure.message)
    check("the traceback survives the process that produced it",
          "SyntheticFailure" in report.job.failure.traceback
          and "synthetic.py" in report.job.failure.traceback,
          "the log is on a container that gets reclaimed; this is not")
    check("the step it failed at is recorded",
          report.job.failure.step == 7,
          f"{report.job.failure.step}: fail_at_step=7 raises once seven steps "
          f"are done, at the top of the eighth")
    check("the reason is an error", report.job.stop_reason is StopReason.ERROR)

    check("the work before the failure is still on disk",
          report.latest_checkpoint is not None
          and report.latest_checkpoint.step == 6,
          "a failed run's finished steps are paid for")
    check("and a failed job is resumable", report.job.is_resumable)

    handle_pid = (report.job.worker or {}).get("pid")
    check("the worker cleared its liveness record on the way out",
          handle_pid is None, str(report.job.worker))

    log = Path(lab.root) / "jobs" / str(job.job_id) / "worker.log"
    check("the worker log also carries the traceback",
          "SyntheticFailure" in log.read_text(),
          "visible in one place only is visible in none")

    # And it can be continued: a failure is not the end of the job.
    app.resume(ResumeTrainingRequest(job_id=job.job_id))
    after = _await(app, job.job_id, timeout=60)
    check("resuming a failed job runs it again from its checkpoint",
          after.job.attempts == 2, str(after.job.attempts))
    check("and it fails again at the same place, because nothing was changed",
          after.job.status is JobStatus.FAILED
          and after.job.failure.step == 7,
          f"{after.job.status} at {after.job.failure.step if after.job.failure else '-'}")
    lab.close()


def test_sigterm_is_a_pause_so_a_reclaimed_container_is_resumable() -> None:
    """The environments this runs in reclaim containers with a term signal.

    Treating SIGTERM as "die" loses everything since the last checkpoint.
    Treating it as "pause" converts a reclamation into a resumable stop, which
    is the difference between losing a generation and losing a run.
    """
    print("\ntest_sigterm_is_a_pause_so_a_reclaimed_container_is_resumable")
    app, lab = _lab()
    job = _start(app, _plan(steps=500, step_seconds=0.02, checkpoint_every=5))
    report = _await(app, job.job_id, want=(JobStatus.RUNNING,), timeout=30)
    check("the worker is running", report.job.status is JobStatus.RUNNING)

    pid = (report.job.worker or {}).get("pid")
    check("the worker recorded its pid", isinstance(pid, int), str(pid))
    # Let it get a few steps in, so there is work to preserve.
    time.sleep(0.6)
    os.kill(int(pid), signal.SIGTERM)

    final = _await(app, job.job_id, timeout=60)
    check("the job ends PAUSED, not FAILED and not RUNNING-forever",
          final.job.status is JobStatus.PAUSED, str(final.job.status))
    check("the reason names the signal",
          final.job.stop_reason is StopReason.SIGNAL,
          str(final.job.stop_reason))
    check("the work up to the signal is checkpointed",
          final.latest_checkpoint is not None
          and final.latest_checkpoint.step == final.state.step,
          f"checkpoint {final.latest_checkpoint.step if final.latest_checkpoint else None}"
          f" vs step {final.state.step}")
    check("nothing is at risk", final.state.steps_at_risk == 0)
    check("and it can be picked up", final.job.is_resumable)

    resumed = app.resume(ResumeTrainingRequest(job_id=job.job_id,
                                               inline=True))
    check("a resume after a reclamation continues",
          resumed.job.attempts == 2 and resumed.job.status in
          (JobStatus.SUCCEEDED, JobStatus.RUNNING, JobStatus.PAUSED),
          f"{resumed.job.status} on attempt {resumed.job.attempts}")
    lab.close()


def test_the_worker_survives_its_parent() -> None:
    """Process isolation, checked by killing the launcher.

    This is the whole reason the training runs somewhere else.  The worker is
    started in its own session, so the process that launched it can die -- a
    terminal closing, a container reclaiming the interactive process -- without
    taking a twenty-one-hour run with it.
    """
    print("\ntest_the_worker_survives_its_parent")
    root = temp_root()
    program = f"""
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
from dytiscidae.adapters.composition import build
from dytiscidae.application import CreateExperimentRequest, StartTrainingRequest
from dytiscidae.domain import TrainingBudget, TrainingPlan

app, lab = build({str(root)!r})
plan = TrainingPlan(trainer="synthetic", seed=1,
                    hyperparameters={{"checkpoint_every": 2, "step_seconds": 0.05}},
                    budget=TrainingBudget(max_steps=60))
exp = app.create(CreateExperimentRequest(name="orphan", plan=plan)).experiment
job = app.start(StartTrainingRequest(experiment_id=exp.experiment_id)).job
print(job.job_id, flush=True)
lab.close()
import time; time.sleep(120)
"""
    import subprocess
    parent = subprocess.Popen([sys.executable, "-c", program],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
    job_id = parent.stdout.readline().strip()
    check("the launching process started a job", bool(job_id), job_id)
    if not job_id:
        parent.kill()
        return

    time.sleep(0.5)
    parent.kill()
    parent.wait(timeout=10)
    check("the launching process is gone", parent.poll() is not None)

    app, lab = build(root)
    report = _await(app, JobId(job_id), timeout=60)
    check("the worker finished anyway",
          report.job.status is JobStatus.SUCCEEDED, str(report.job.status))
    check("having run the whole budget after its parent died",
          report.state.step == 60, str(report.state.step))
    lab.close()


def test_the_worker_refuses_to_run_a_job_that_is_already_running() -> None:
    """Two workers for one job would write one run directory from two processes.

    The domain refuses the transition; this checks the worker reports that
    refusal rather than proceeding or crashing.
    """
    print("\ntest_the_worker_refuses_to_run_a_job_that_is_already_running")
    app, lab = _lab()
    job = _start(app, _plan(steps=4), inline=True)
    check("the inline run finished", app.status(job.job_id).job.status
          is JobStatus.SUCCEEDED)

    # A second worker on the same, now-terminal, job.
    second = lab.worker(job.job_id, install_signal_handlers=False).run()
    check("a second worker leaves the terminal job alone",
          second.status is JobStatus.SUCCEEDED, str(second.status))
    notes = [e.message for e in lab.events.read(job.job_id)
             if e.kind.value == "note"]
    check("and says why in the history",
          any("refusing to start" in m for m in notes),
          str(notes[-1] if notes else "no note"))

    check("the step count was not disturbed",
          app.status(job.job_id).state.step == 4)
    lab.close()


def test_an_unknown_trainer_fails_the_job_instead_of_the_worker() -> None:
    print("\ntest_an_unknown_trainer_fails_the_job_instead_of_the_worker")
    app, lab = _lab()
    plan = TrainingPlan(trainer="no-such-trainer", seed=1,
                        budget=TrainingBudget(max_steps=5))
    job = _start(app, plan, name="unknown")
    report = _await(app, job.job_id, timeout=60)
    check("the job is FAILED", report.job.status is JobStatus.FAILED,
          str(report.job.status))
    check("the failure names what was missing and what is available",
          "no-such-trainer" in report.job.failure.message
          and "synthetic" in report.job.failure.message,
          report.job.failure.message[:110])
    lab.close()


def test_an_out_of_tree_trainer_reaches_the_worker_through_the_environment() -> None:
    """A trainer registered in one process must reach a worker in another.

    In-process registration cannot cross a fork, so ``DYTISCIDAE_TRAINERS``
    exists.  Without it, an embedding application's own trainer would work
    inline and fail the moment the run was isolated -- which is the moment it
    matters.
    """
    print("\ntest_an_out_of_tree_trainer_reaches_the_worker_through_the_environment")
    root = temp_root()
    module_dir = root / "extra"
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "mytrainer.py").write_text('''
from dytiscidae.domain.job import StopReason
from dytiscidae.ports.trainer import TrainerCapabilities, TrainingOutcome


class Marker:
    def capabilities(self):
        return TrainerCapabilities(step_unit="widget", cooperative_stop=True,
                                   checkpoints=True, description="out of tree")

    def train(self, context):
        state = context.state.advanced(step=3, samples_seen=3,
                                       metrics={"widgets": 3.0})
        context.report(state, metrics=state.metrics)
        context.save_checkpoint({"m.bin": b"out-of-tree"}, step=3)
        return TrainingOutcome(state=context.state,
                               stop_reason=StopReason.COMPLETED,
                               summary={"made": "widgets"})
''')
    repo = str(Path(__file__).resolve().parents[1])
    env = {"DYTISCIDAE_TRAINERS": "widgets=mytrainer:Marker",
           "PYTHONPATH": f"{module_dir}{os.pathsep}{repo}"}

    app, lab = build(root)
    lab.launcher._env.update(env)                      # what a deployment would set
    plan = TrainingPlan(trainer="widgets", seed=1, budget=TrainingBudget(max_steps=3))
    job = _start(app, plan, name="out-of-tree")
    report = _await(app, job.job_id, timeout=60)
    check("the worker resolved a trainer it was told about by environment",
          report.job.status is JobStatus.SUCCEEDED, str(report.job.status))
    check("and it ran", report.state is not None and report.state.step == 3,
          str(report.state.step if report.state else None))
    check("its checkpoint is in the store",
          lab.checkpoints.latest(job.job_id) is not None)
    notes = [e.detail.get("step_unit") for e in lab.events.read(job.job_id)
             if e.kind.value == "note" and "step_unit" in e.detail]
    check("its declared step unit is in the record", notes == ["widget"],
          str(notes))
    lab.close()


def test_export_produces_a_file_from_a_finished_run() -> None:
    print("\ntest_export_produces_a_file_from_a_finished_run")
    app, lab = _lab()
    job = _start(app, _plan(steps=6), inline=True)
    _await(app, job.job_id, timeout=60)

    out = Path(lab.root) / "exported.json"
    response = app.export(ExportModelRequest(destination=str(out),
                                             job_id=job.job_id, fmt="json"))
    check("a file was written", out.exists() and response.bytes_written > 0,
          f"{response.bytes_written} bytes")
    payload = json.loads(out.read_text())
    check("it carries the model", payload["model"]["step"] == 6, str(payload["step"]))
    check("and the provenance that says which run made it",
          payload["provenance"].get("plan_digest") == job.plan.digest)

    raised = False
    try:
        app.export(ExportModelRequest(destination=str(out), job_id=job.job_id,
                                      fmt="onnx"))
    except ValueError as exc:
        raised = "'native' or 'json'" in str(exc)
    check("an unknown format is refused rather than quietly producing another",
          raised)
    lab.close()


def test_both_storage_backends_run_a_job_identically() -> None:
    """The same plan, the same seed, two record backends, one trajectory.

    If the storage backend changed the result, the architecture would not be
    doing its job -- storage is an adapter, and an adapter must not be visible
    from inside.
    """
    print("\ntest_both_storage_backends_run_a_job_identically")
    results = {}
    for store in ("files", "sqlite"):
        app, lab = _lab(store=store)
        job = _start(app, _plan(steps=10), name=f"{store}-run")
        report = _await(app, job.job_id, timeout=60)
        results[store] = (report.job.status, report.state.step,
                          _trajectory(lab, job.job_id),
                          lab.checkpoints.latest(job.job_id).digest)
        lab.close()

    files, sqlite = results["files"], results["sqlite"]
    check("both completed", files[0] is sqlite[0] is JobStatus.SUCCEEDED,
          f"{files[0]} / {sqlite[0]}")
    check("both ran the same number of steps", files[1] == sqlite[1] == 10)
    check("the trajectories are identical", files[2] == sqlite[2],
          "the storage backend must not be visible from inside the hexagon")
    check("and so is the final checkpoint's digest", files[3] == sqlite[3],
          f"{files[3][:12]} vs {sqlite[3][:12]}")


def main() -> int:
    print("=" * 68)
    print("training runtime: real worker processes, doing the six things")
    print("=" * 68)
    try:
        test_a_worker_records_a_completed_run_end_to_end()
        test_a_pause_stops_at_a_boundary_and_leaves_a_resumable_job()
        test_a_resume_continues_the_run_rather_than_restarting_it()
        test_a_cancel_stops_the_run_and_does_not_leave_it_resumable()
        test_a_failing_trainer_lands_the_job_in_failed_with_its_traceback()
        test_sigterm_is_a_pause_so_a_reclaimed_container_is_resumable()
        test_the_worker_survives_its_parent()
        test_the_worker_refuses_to_run_a_job_that_is_already_running()
        test_an_unknown_trainer_fails_the_job_instead_of_the_worker()
        test_an_out_of_tree_trainer_reaches_the_worker_through_the_environment()
        test_export_produces_a_file_from_a_finished_run()
        test_both_storage_backends_run_a_job_identically()
    finally:
        for path in TEMPS:
            shutil.rmtree(path, ignore_errors=True)

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all training-runtime checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
