"""Unit tests for the domain: the layer with no IO and no dependencies.

The fastest tier of the three the architecture calls for, and the one that
should stay fast.  Nothing here touches a disk, a clock or a subprocess, so the
whole file runs in well under a second and can be run on every save.

What is being pinned down is the lifecycle and the three-way separation of
configuration, runtime state and record.  Most of it is invariants rather than
examples, in the same spirit as ``tests/test_math.py``: a transition table
checked exhaustively catches the move nobody thought about, where a test of
"pause then resume" only catches the one that was.

Run:  PYTHONPATH=. python tests/test_domain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dytiscidae.domain import (                                   # noqa: E402
    ACTIVE_STATUSES, TERMINAL_STATUSES, CheckpointId, CheckpointKind,
    CheckpointRecord, Dataset, DatasetRef, Experiment, ExperimentId,
    FailureInfo, IllegalTransition, JobEvent, JobEventKind, JobId, JobStatus,
    MetricPoint, PlanRejected, StopReason, TrainingBudget, TrainingJob,
    TrainingPlan, TrainingState, new_id,
)
from dytiscidae.domain.checkpoint import digest_payload            # noqa: E402
from dytiscidae.domain.job import RESUMABLE_STATUSES, _ALLOWED     # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def _plan(**kw) -> TrainingPlan:
    kw.setdefault("trainer", "synthetic")
    kw.setdefault("seed", 1)
    return TrainingPlan(**kw)


def _job(status: JobStatus = JobStatus.PENDING, **kw) -> TrainingJob:
    kw.setdefault("job_id", JobId("j1"))
    kw.setdefault("experiment_id", ExperimentId("e1"))
    kw.setdefault("plan", _plan())
    return TrainingJob(status=status, created_at=1.0, **kw)


# --------------------------------------------------------------------------


def test_identifiers_do_not_compare_across_types() -> None:
    """A ``JobId`` and an ``ExperimentId`` with the same text are not equal.

    The failure this prevents is a silent one: pass a job id where an
    experiment id belongs, and a store keyed on strings finds nothing and
    returns None, which reads as "no such experiment" rather than as a bug.
    """
    print("\ntest_identifiers_do_not_compare_across_types")
    check("same text, different type, not equal",
          JobId("x") != ExperimentId("x"))
    check("same text, same type, equal", JobId("x") == JobId("x"))
    check("a plain string still compares by value, so JSON round-trips",
          JobId("x") == "x" and "x" == JobId("x"))
    check("hashes as its text, so a dict keyed by str finds it",
          {JobId("x"): 1}.get("x") == 1)

    for bad in ("", "has space", "-leading", "a/b", "x" * 200):
        raised = False
        try:
            JobId(bad)
        except ValueError:
            raised = True
        check(f"rejects {bad[:14]!r} as an identifier", raised)

    ids = [new_id("job", now=1_700_000_000 + i) for i in (0, 60, 120)]
    check("generated ids sort in time order", ids == sorted(ids),
          ids[0])
    check("two ids made in the same second still differ",
          new_id("job", now=1.0) != new_id("job", now=1.0))


def test_the_plan_is_the_configuration_and_hashes_to_it() -> None:
    """Two plans describing the same run hash the same; any difference does not.

    The digest is what makes "is this a rerun of that experiment" answerable.
    Anything that changes the result must change it, and anything that does not
    must not -- so both directions are checked, and the second is the one that
    is usually missing.
    """
    print("\ntest_the_plan_is_the_configuration_and_hashes_to_it")
    a = _plan(hyperparameters={"lr": 0.1, "batch": 16}, seed=3,
              budget=TrainingBudget(max_steps=10), tags=("arm-a",))
    # Same content, different insertion order, different tag order.
    b = _plan(hyperparameters={"batch": 16, "lr": 0.1}, seed=3,
              budget=TrainingBudget(max_steps=10), tags=("arm-a",))
    check("key order does not change the digest", a.digest == b.digest, a.digest)
    check("the digest is 16 hex characters",
          len(a.digest) == 16 and all(c in "0123456789abcdef" for c in a.digest))

    for label, changed in (
            ("a hyperparameter", a.with_(hyperparameters={"lr": 0.2, "batch": 16})),
            ("the seed", a.with_(seed=4)),
            ("the trainer", a.with_(trainer="search")),
            ("the budget", a.with_(budget=TrainingBudget(max_steps=11))),
            ("a tag", a.with_(tags=("arm-b",))),
            ("the dataset", a.with_(dataset=DatasetRef("elites", "v1")))):
        check(f"changing {label} changes the digest", a.digest != changed.digest)

    # A tag is part of the digest on purpose: an arm label that did not change
    # it would make two arms of an A/B look like one configuration run twice.
    check("two arms of an A/B are two configurations",
          _plan(tags=("arm-a",)).digest != _plan(tags=("arm-b",)).digest)

    round_tripped = TrainingPlan.from_dict(a.as_dict())
    check("a plan survives a JSON round trip unchanged",
          round_tripped.digest == a.digest and round_tripped == a)

    for bad, why in (({"seed": -1}, "a negative seed"),
                     ({"trainer": "  "}, "an empty trainer"),
                     ({"hyperparameters": {"x": object()}}, "a non-JSON value")):
        raised = False
        try:
            _plan(**bad)
        except PlanRejected:
            raised = True
        check(f"rejects {why} at construction, not at generation one", raised)

    raised = False
    try:
        TrainingBudget(max_steps=0)
    except PlanRejected:
        raised = True
    check("rejects a budget of zero steps", raised)
    check("an entirely open budget is legal",
          TrainingBudget().max_steps is None)


def test_the_lifecycle_table_is_complete_and_enforced() -> None:
    """Every status has a transition row, and every move off it is checked.

    Exhaustive rather than by example: the transition that nobody wrote a test
    for is exactly the one an operator will find at 3 a.m.  This walks the
    product of statuses and asserts that the ones the table forbids raise and
    the ones it allows do not.
    """
    print("\ntest_the_lifecycle_table_is_complete_and_enforced")
    check("every status has a row in the table",
          set(_ALLOWED) == set(JobStatus),
          f"missing {sorted(set(JobStatus) - set(_ALLOWED))}")
    check("the terminal statuses are the ones with no outgoing move",
          {s for s in JobStatus if not _ALLOWED[s]}
          == TERMINAL_STATUSES - {JobStatus.FAILED},
          "FAILED is terminal but resumable, which is deliberate")

    moves = {
        JobStatus.RUNNING: lambda j: j.start(now=2.0),
        JobStatus.PAUSING: lambda j: j.request_pause(),
        JobStatus.PAUSED: lambda j: j.paused(now=3.0),
        JobStatus.CANCELLED: lambda j: j.cancelled(now=3.0),
        JobStatus.SUCCEEDED: lambda j: j.succeeded(now=3.0),
        JobStatus.FAILED: lambda j: j.failed(FailureInfo("E", "m"), now=3.0),
    }
    # The table describes *changes* of status, so it cannot express idempotence:
    # a second pause request on a PAUSING job is not a transition, it is the
    # same request arriving twice, and it must succeed.  Listed here rather
    # than added to the table, because a self-edge in the table would also make
    # ``start`` on a RUNNING job legal -- and starting a job that is already
    # running is a second worker writing one run directory, which must raise.
    IDEMPOTENT_SELF = {JobStatus.PAUSING}

    bad, good = [], 0
    for frm in JobStatus:
        for to, move in moves.items():
            job = _job(frm)
            allowed = (to in _ALLOWED[frm]
                       or (frm is to and to in IDEMPOTENT_SELF))
            try:
                result = move(job)
                actually = result.status is to
            except IllegalTransition:
                actually = False
            except Exception as exc:                              # noqa: BLE001
                bad.append(f"{frm}->{to} raised {type(exc).__name__}")
                continue
            if actually != allowed:
                bad.append(f"{frm}->{to} {'allowed' if actually else 'refused'}"
                           f" but the table says "
                           f"{'allowed' if allowed else 'refused'}")
            else:
                good += 1
    check("every status pair behaves as the table says", not bad,
          "; ".join(bad) if bad else f"{good} pairs agree")

    # The self-edge that must not exist: two workers for one job.
    raised = False
    try:
        _job(JobStatus.RUNNING).start(now=2.0)
    except IllegalTransition:
        raised = True
    check("starting a job that is already running raises", raised,
          "otherwise two workers write one run directory")


def test_pausing_and_cancelling_are_states_not_instants() -> None:
    """A stop request and a stop are different facts and stay different.

    This is the property that makes the PID-hunting in the operating notes
    unnecessary: "asked to stop" is readable without looking at whether a
    process exists.
    """
    print("\ntest_pausing_and_cancelling_are_states_not_instants")
    running = _job().start(now=2.0)
    pausing = running.request_pause()
    check("a pause request leaves the job PAUSING, not PAUSED",
          pausing.status is JobStatus.PAUSING)
    check("PAUSING still counts as active, so the worker is expected alive",
          pausing.status in ACTIVE_STATUSES)
    check("stop_requested reads true while it is pending", pausing.stop_requested)
    check("a repeated pause request is idempotent",
          pausing.request_pause().status is JobStatus.PAUSING)
    paused = pausing.paused(now=3.0)
    check("only the runtime's acknowledgement reaches PAUSED",
          paused.status is JobStatus.PAUSED and paused.ended_at == 3.0)
    check("a paused job is resumable", paused.is_resumable)

    cancelling = running.request_cancel()
    check("a cancel on a running job leaves it CANCELLING",
          cancelling.status is JobStatus.CANCELLING)
    check("a cancel overrides a pending pause",
          pausing.request_cancel().status is JobStatus.CANCELLING)
    check("a pending job cancels straight through: nothing can acknowledge it",
          _job().request_cancel().status is JobStatus.CANCELLED)
    check("a paused job cancels straight through as well",
          paused.request_cancel().status is JobStatus.CANCELLED)
    check("a second cancel is idempotent rather than an error",
          cancelling.request_cancel().status is JobStatus.CANCELLING)


def test_a_resume_needs_a_checkpoint_and_goes_through_pending() -> None:
    """The refusal that makes ``resumed_from`` trustworthy.

    A resume without a checkpoint is a restart, and letting it through would
    produce a record claiming continuity across a boundary where the learned
    state was reinitialised -- the exact defect ``ops/checkpoint.py`` exists to
    close, recreated one layer up.
    """
    print("\ntest_a_resume_needs_a_checkpoint_and_goes_through_pending")
    failed = _job().start(now=2.0).failed(FailureInfo("OOM", "no memory"), now=3.0)
    check("a failed job is resumable: its finished generations are paid for",
          failed.is_resumable and JobStatus.FAILED in RESUMABLE_STATUSES)

    for label, job in (("failed", failed),
                       ("paused", _job().start(now=2.0).request_pause().paused(now=3.0))):
        raised = False
        try:
            job.prepare_resume(now=4.0, checkpoint=None)
        except IllegalTransition:
            raised = True
        check(f"a {label} job refuses to resume without a checkpoint", raised)

        armed = job.prepare_resume(now=4.0, checkpoint="ck-9")
        check(f"a {label} job arms to PENDING, not RUNNING",
              armed.status is JobStatus.PENDING)
        check(f"the {label} job records what it will resume from",
              armed.resumed_from == "ck-9")
        check(f"the {label} job's failure is cleared when it is re-armed",
              armed.failure is None and armed.stop_reason is None)
        check(f"the attempt count rises only when the worker starts a {label} job",
              armed.attempts == job.attempts
              and armed.start(now=5.0).attempts == job.attempts + 1)

    check("a succeeded job is not resumable",
          not _job().start(now=2.0).succeeded(now=3.0).is_resumable)
    check("a cancelled job is not resumable",
          not _job().request_cancel().is_resumable)

    # The budget may be raised on resume; the trainer and the seed may not.
    base = failed.plan
    extended = base.with_(budget=TrainingBudget(max_steps=999))
    check("a resume may raise the step budget",
          failed.prepare_resume(now=4.0, checkpoint="ck", plan=extended
                                ).plan.budget.max_steps == 999)
    for label, bad_plan in (("trainer", base.with_(trainer="other")),
                            ("seed", base.with_(seed=base.seed + 1))):
        raised = False
        try:
            failed.prepare_resume(now=4.0, checkpoint="ck", plan=bad_plan)
        except IllegalTransition:
            raised = True
        check(f"a resume may not change the {label}", raised)


def test_a_failure_survives_the_process_that_produced_it() -> None:
    print("\ntest_a_failure_survives_the_process_that_produced_it")
    try:
        raise ValueError("resident set over the ceiling")
    except ValueError as exc:
        info = FailureInfo.from_exception(exc, step=611)
    check("the exception class is kept", info.kind == "ValueError")
    check("the message is kept", "ceiling" in info.message)
    check("the traceback is kept", "ValueError" in info.traceback
          and "test_a_failure_survives" in info.traceback)
    check("the step it failed at is kept", info.step == 611,
          "611 generations of paid-for work is the thing a resume recovers")
    check("retryable defaults to False, so 'unknown' never auto-retries",
          info.retryable is False)
    check("it round-trips through JSON",
          FailureInfo.from_dict(info.as_dict()) == info)

    job = _job().start(now=2.0).failed(info, now=3.0)
    round_tripped = TrainingJob.from_dict(job.as_dict())
    check("a failed job round-trips with its failure intact",
          round_tripped.failure == info
          and round_tripped.stop_reason is StopReason.ERROR)


def test_runtime_state_is_monotone_and_says_what_a_crash_costs() -> None:
    print("\ntest_runtime_state_is_monotone_and_says_what_a_crash_costs")
    state = TrainingState(job_id=JobId("j1"), total_steps=900)
    check("progress of an unstarted run is 0.0", state.progress == 0.0)
    check("an open budget reports None, not 0.0",
          TrainingState(job_id=JobId("j1")).progress is None,
          "'no budget' and 'nothing done' are different facts")

    state = state.advanced(step=100, samples_seen=1_500, elapsed_seconds=7_400.0)
    check("progress tracks the budget", abs(state.progress - 100 / 900) < 1e-12)
    check("the rate comes from steps and wall time",
          abs(state.steps_per_second - 100 / 7_400.0) < 1e-12)
    eta = state.eta_seconds()
    check("an ETA is a flat extrapolation of that rate",
          eta is not None and abs(eta - 800 * 7_400.0 / 100) < 1e-6,
          f"{eta:.0f}s")

    for label, kwargs in (("a step", {"step": 99}),
                          ("samples", {"step": 101, "samples_seen": 1_400})):
        raised = False
        try:
            state.advanced(**kwargs)
        except ValueError:
            raised = True
        check(f"{label} going backwards raises rather than charting backwards",
              raised)

    check("with no checkpoint, everything done is at risk",
          state.steps_at_risk == 100)
    state = state.checkpointed(CheckpointId("ck1"), step=90)
    check("after a checkpoint, only the gap is at risk",
          state.steps_at_risk == 10,
          "which is exactly what a crash costs")
    check("state round-trips through JSON",
          TrainingState.from_dict(state.as_dict()).as_dict() == state.as_dict())


def test_the_three_records_stay_separate() -> None:
    """Configuration, runtime state and experiment record do not leak into
    each other.

    Checked structurally, because the leak is always by accident: someone adds
    ``current_step`` to the config for convenience, and from then on no two
    configurations compare equal.
    """
    print("\ntest_the_three_records_stay_separate")
    plan_fields = set(TrainingPlan.__dataclass_fields__)
    state_fields = set(TrainingState.__dataclass_fields__)
    experiment_fields = set(Experiment.__dataclass_fields__)

    check("the plan carries no progress",
          not (plan_fields & {"step", "samples_seen", "elapsed_seconds",
                              "last_checkpoint", "status"}),
          f"plan: {sorted(plan_fields)}")
    check("the runtime state carries no configuration",
          not (state_fields & {"trainer", "hyperparameters", "seed", "budget"}),
          f"state: {sorted(state_fields)}")
    check("the experiment record carries no progress",
          not (experiment_fields & {"step", "status", "elapsed_seconds"}))
    check("the plan is frozen", TrainingPlan.__dataclass_params__.frozen)
    check("the runtime state is frozen", TrainingState.__dataclass_params__.frozen)
    check("the job is frozen", TrainingJob.__dataclass_params__.frozen)

    # Pool shape is on the job, not the plan: this project measured the same
    # work at 31.7 s on four shards of four and 93.3 s on sixteen of one, and
    # two runs differing only in that must still be one configuration.
    plan = _plan()
    a = _job(plan=plan, resources={"workers": 4, "min_shard": 4})
    b = _job(plan=plan, resources={"workers": 16, "min_shard": 1})
    check("worker count lives on the job, so it changes no digest",
          a.plan.digest == b.plan.digest and a.resources != b.resources)


def test_an_experiment_is_a_question_with_many_attempts() -> None:
    print("\ntest_an_experiment_is_a_question_with_many_attempts")
    experiment = Experiment(experiment_id=ExperimentId("e1"), name="arch39",
                            plan=_plan(), created_at=1.0,
                            hypothesis="thrust_margin flattens aspect ratio")
    experiment = experiment.with_job(JobId("j1")).with_job(JobId("j2"))
    check("a resume is a second job under one experiment",
          experiment.job_ids == (JobId("j1"), JobId("j2")))
    check("attaching the same job twice is idempotent",
          experiment.with_job(JobId("j2")).job_ids == experiment.job_ids)
    check("the latest job is the last attached", experiment.latest_job == JobId("j2"))
    check("the hypothesis is kept",
          "thrust_margin" in experiment.hypothesis,
          "recorded before the run, or it is not a prediction")
    check("a conclusion can be recorded without touching anything else",
          experiment.concluded("area still flat").conclusion == "area still flat")
    check("it round-trips",
          Experiment.from_dict(experiment.as_dict()).as_dict()
          == experiment.as_dict())

    raised = False
    try:
        Experiment(experiment_id=ExperimentId("e2"), name="  ", plan=_plan())
    except ValueError:
        raised = True
    check("an experiment must have a name", raised)


def test_a_checkpoint_digest_cannot_be_confused_by_its_key_names() -> None:
    """Names and lengths are hashed alongside the bytes.

    The classic defect is concatenating the blobs and hashing that: then
    ``{"ab": b"", "": b"cd"}`` and ``{"a": b"b", "cd": b""}`` collide, and a
    corrupt checkpoint verifies.  Checked with the literal pathological pair.
    """
    print("\ntest_a_checkpoint_digest_cannot_be_confused_by_its_key_names")
    check("the same payload digests the same",
          digest_payload({"a": b"1", "b": b"2"})
          == digest_payload({"b": b"2", "a": b"1"}))
    check("a different value changes the digest",
          digest_payload({"a": b"1"}) != digest_payload({"a": b"2"}))
    check("a different key changes the digest",
          digest_payload({"a": b"1"}) != digest_payload({"z": b"1"}))
    check("boundaries between keys and values are hashed too",
          digest_payload({"ab": b"", "": b"cd"})
          != digest_payload({"a": b"b", "cd": b""}))

    record = CheckpointRecord(
        checkpoint_id=CheckpointId("ck1"), job_id=JobId("j1"),
        experiment_id=ExperimentId("e1"), step=600, kind=CheckpointKind.BEST,
        metrics={"mission_fraction": 0.3474}, payload_keys=("checkpoint.npz",))
    check("a BEST checkpoint is protected from automatic retention",
          record.protected)
    check("a PERIODIC one is not",
          not CheckpointRecord(checkpoint_id=CheckpointId("ck2"),
                               job_id=JobId("j1"),
                               experiment_id=ExperimentId("e1"),
                               step=1).protected)
    check("a FINAL one is protected: it exists because something went wrong",
          CheckpointRecord(checkpoint_id=CheckpointId("ck3"), job_id=JobId("j1"),
                           experiment_id=ExperimentId("e1"), step=1,
                           kind=CheckpointKind.FINAL).protected)
    check("it round-trips",
          CheckpointRecord.from_dict(record.as_dict()).as_dict()
          == record.as_dict())


def test_a_dataset_ref_is_a_name_and_a_version() -> None:
    print("\ntest_a_dataset_ref_is_a_name_and_a_version")
    check("a ref prints as name:version", str(DatasetRef("elites", "arch38"))
          == "elites:arch38")
    check("the version defaults rather than being optional",
          DatasetRef("elites").version == "v1")
    check("two versions of one corpus are two refs",
          DatasetRef("elites", "v1") != DatasetRef("elites", "v2"))
    for bad in ({"name": " "}, {"name": "x", "version": ""}):
        raised = False
        try:
            DatasetRef(**bad)
        except ValueError:
            raised = True
        check(f"rejects {bad}", raised)

    counted = Dataset(ref=DatasetRef("a"), item_count=0)
    unknown = Dataset(ref=DatasetRef("b"), item_count=None)
    check("a measured empty corpus and an uncounted one are different answers",
          counted.item_count == 0 and unknown.item_count is None,
          "this project has been bitten by a metric that used one value for both")


def test_metrics_and_events_carry_what_makes_them_readable() -> None:
    print("\ntest_metrics_and_events_carry_what_makes_them_readable")
    point = MetricPoint(name="coverage", value=0.41, step=447,
                        job_id=JobId("j1"), wall_seconds=33_000.0,
                        tags={"island": "amphibian"})
    check("a metric carries the tag that stops it being mis-plotted",
          point.tags["island"] == "amphibian",
          "consecutive generation lines are different archives")
    check("it round-trips", MetricPoint.from_dict(point.as_dict()) == point)
    raised = False
    try:
        MetricPoint(name=" ", value=1.0, step=0)
    except ValueError:
        raised = True
    check("a metric must be named", raised)

    event = JobEvent(job_id=JobId("j1"), kind=JobEventKind.FAILED, at=5.0,
                     step=611, message="resident set over the ceiling")
    check("an event prints legibly", "[failed]" in str(event) and "611" in str(event))
    check("it round-trips", JobEvent.from_dict(event.as_dict()) == event)


def main() -> int:
    print("=" * 68)
    print("domain: the lifecycle, the plan, the state and the record")
    print("=" * 68)
    test_identifiers_do_not_compare_across_types()
    test_the_plan_is_the_configuration_and_hashes_to_it()
    test_the_lifecycle_table_is_complete_and_enforced()
    test_pausing_and_cancelling_are_states_not_instants()
    test_a_resume_needs_a_checkpoint_and_goes_through_pending()
    test_a_failure_survives_the_process_that_produced_it()
    test_runtime_state_is_monotone_and_says_what_a_crash_costs()
    test_the_three_records_stay_separate()
    test_an_experiment_is_a_question_with_many_attempts()
    test_a_checkpoint_digest_cannot_be_confused_by_its_key_names()
    test_a_dataset_ref_is_a_name_and_a_version()
    test_metrics_and_events_carry_what_makes_them_readable()

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all domain checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
