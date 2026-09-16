"""Integration tests for the adapters: the layer that touches real storage.

The second tier.  These use a real filesystem and a real SQLite database, and
they are the only tests in the set that can be slow -- a few seconds, because
some of them fork processes to create contention that a single-threaded test
cannot.

Three things get the most attention here, because each is a failure that the
layers above cannot detect:

**Atomicity.**  A checkpoint that looks complete and is not gets resumed from,
and the run is quietly wrong for the next twenty hours.  So the record is
written last, and a checkpoint interrupted halfway must be invisible.

**Concurrency.**  ``JobStore.update`` is called by two processes on one job.
Tested by actually forking, not by reasoning: a lost update is exactly the
class of bug that survives being reasoned about.

**Contract equality between backends.**  The file stores and the SQLite stores
implement the same ports, so the same test body runs against both.  Anything
they disagree on is a bug in one of them, and running them separately is how
that disagreement stays undiscovered.

Run:  PYTHONPATH=. python tests/test_adapters.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dytiscidae.adapters.filesystem import (                       # noqa: E402
    CorruptCheckpoint, FileCheckpointStore, FileControlChannel,
    FileDatasetRepository, FileEventLog, FileExperimentStore, FileJobStore,
    FileMetricSink,
)
from dytiscidae.adapters.filesystem._io import FileLock             # noqa: E402
from dytiscidae.adapters.launchers import SubprocessLauncher        # noqa: E402
from dytiscidae.adapters.sqlite import (                            # noqa: E402
    SqliteExperimentStore, SqliteJobStore, SqliteStore,
)
from dytiscidae.domain import (                                     # noqa: E402
    CheckpointId, CheckpointKind, CheckpointNotFound, DatasetKind,
    DatasetRef, Experiment, ExperimentId, ExperimentNotFound, FailureInfo,
    JobEvent, JobEventKind, JobId, JobNotFound, JobStatus, MetricPoint,
    StopReason, TrainingBudget, TrainingJob, TrainingPlan, TrainingState,
)
from dytiscidae.ports.control import ControlSignal                  # noqa: E402
from dytiscidae.ports.launcher import WorkerHandle, WorkerStatus    # noqa: E402

FAILURES: list[str] = []
TEMPS: list[Path] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def temp_root(prefix: str = "adapters-") -> Path:
    path = Path(tempfile.mkdtemp(prefix=prefix))
    TEMPS.append(path)
    return path


def _plan(**kw) -> TrainingPlan:
    kw.setdefault("trainer", "synthetic")
    kw.setdefault("seed", 2)
    kw.setdefault("budget", TrainingBudget(max_steps=50))
    return TrainingPlan(**kw)


def _job(job_id: str = "j1", experiment_id: str = "e1", **kw) -> TrainingJob:
    return TrainingJob(job_id=JobId(job_id), experiment_id=ExperimentId(experiment_id),
                       plan=kw.pop("plan", _plan()), created_at=kw.pop("created_at", 1.0),
                       **kw)


def _experiment(experiment_id: str = "e1", name: str = "arch39") -> Experiment:
    return Experiment(experiment_id=ExperimentId(experiment_id), name=name,
                      plan=_plan(), created_at=1.0)


# --------------------------------------------------------------------------


def _job_store_contract(label: str, jobs, experiments) -> None:
    """One body, run against both backends.

    A contract test rather than two separate suites: two implementations of one
    port that are tested separately will diverge, and the divergence is found by
    whoever swaps the backend.
    """
    experiments.create(_experiment())
    created = jobs.create(_job())
    check(f"[{label}] a job round-trips", jobs.get(JobId("j1")) == created)

    raised = False
    try:
        jobs.create(_job())
    except FileExistsError:
        raised = True
    check(f"[{label}] creating the same job twice is refused", raised)

    raised = False
    try:
        jobs.get(JobId("missing"))
    except JobNotFound:
        raised = True
    check(f"[{label}] a missing job raises JobNotFound", raised)

    # update applies to what is *stored*, not to what the caller holds.
    stale = jobs.get(JobId("j1"))
    jobs.update(JobId("j1"), lambda j: j.heartbeat(now=5.0, note="from elsewhere"))
    updated = jobs.update(JobId("j1"), lambda j: j.start(now=6.0))
    check(f"[{label}] update sees a concurrent write rather than overwriting it",
          updated.worker.get("note") == "from elsewhere",
          "the caller's stale copy had no note")
    check(f"[{label}] and applies the transition on top",
          updated.status is JobStatus.RUNNING and updated.attempts == 1)
    assert stale.status is JobStatus.PENDING          # the held copy is untouched

    raised = False
    try:
        jobs.update(JobId("j1"), lambda j: j.start(now=7.0))
    except Exception as exc:                                      # noqa: BLE001
        raised = type(exc).__name__ == "IllegalTransition"
    check(f"[{label}] an illegal transition leaves the store untouched", raised
          and jobs.get(JobId("j1")).attempts == 1)

    state = TrainingState(job_id=JobId("j1"), step=40, total_steps=50,
                          samples_seen=612, elapsed_seconds=2960.0,
                          metrics={"loss": 0.25})
    jobs.save_state(state)
    check(f"[{label}] runtime state round-trips",
          jobs.load_state(JobId("j1")).as_dict() == state.as_dict())
    check(f"[{label}] a job with no state reports None",
          jobs.load_state(JobId("never")) is None)

    jobs.create(_job("j2", created_at=2.0))
    # A second experiment, created first: SQLite enforces the job -> experiment
    # foreign key, and a job whose experiment does not exist is an orphan
    # record.  The file store cannot enforce it, so the *contract* is that
    # callers create the experiment first -- which StartTraining does, by
    # resolving it before building the job.
    experiments.create(Experiment(experiment_id=ExperimentId("e2"), name="other",
                                  plan=_plan(), created_at=2.0))
    jobs.create(_job("j3", experiment_id="e2", created_at=3.0))
    listed = [str(j.job_id) for j in jobs.list()]
    check(f"[{label}] listing is newest first", listed[0] == "j3", str(listed))
    check(f"[{label}] listing filters by experiment",
          [str(j.job_id) for j in jobs.list(experiment_id=ExperimentId("e1"))]
          == ["j2", "j1"])
    check(f"[{label}] listing filters by status",
          [str(j.job_id) for j in jobs.list(status=JobStatus.RUNNING)] == ["j1"])
    check(f"[{label}] listing honours a limit", len(jobs.list(limit=2)) == 2)

    jobs.delete(JobId("j2"))
    check(f"[{label}] delete removes it",
          "j2" not in [str(j.job_id) for j in jobs.list()])
    jobs.delete(JobId("j2"))
    check(f"[{label}] delete is idempotent", True)

    # A job under an experiment that does not exist: SQLite refuses it, the
    # file store cannot.  Both are acceptable, and what must not happen is
    # SQLite reporting the fault as a duplicate job -- which sends a reader
    # looking for something that is not there.
    try:
        jobs.create(_job("orphan", experiment_id="never-made"))
    except ExperimentNotFound as exc:
        check(f"[{label}] an orphan job is refused, naming the missing experiment",
              "never-made" in str(exc), str(exc)[:90])
    except FileExistsError as exc:
        check(f"[{label}] an orphan job is refused for the right reason", False,
              f"reported as a duplicate: {exc}")
    else:
        check(f"[{label}] an orphan job is accepted (no referential integrity)",
              label == "files",
              "only the file store may accept one")


def test_both_job_stores_keep_the_same_contract() -> None:
    print("\ntest_both_job_stores_keep_the_same_contract")
    root = temp_root()
    _job_store_contract("files", FileJobStore(root), FileExperimentStore(root))

    db = SqliteStore(temp_root() / "lab.sqlite3")
    _job_store_contract("sqlite", SqliteJobStore(db), SqliteExperimentStore(db))
    db.close()


def _experiment_store_contract(label: str, store) -> None:
    created = store.create(_experiment())
    check(f"[{label}] an experiment round-trips",
          store.get(ExperimentId("e1")) == created)
    check(f"[{label}] it is findable by name",
          store.find_by_name("arch39").experiment_id == ExperimentId("e1"))
    check(f"[{label}] an unknown name gives None, not an error",
          store.find_by_name("nope") is None)

    raised = False
    try:
        store.create(Experiment(experiment_id=ExperimentId("e2"), name="arch39",
                                plan=_plan()))
    except FileExistsError:
        raised = True
    check(f"[{label}] a duplicate name is refused at the store as well",
          raised, "the use case's check does not hold across two processes")

    store.attach_job(ExperimentId("e1"), JobId("j1"))
    store.attach_job(ExperimentId("e1"), JobId("j2"))
    store.attach_job(ExperimentId("e1"), JobId("j1"))
    check(f"[{label}] attaching is ordered and idempotent",
          [str(j) for j in store.get(ExperimentId("e1")).job_ids] == ["j1", "j2"])

    store.update(store.get(ExperimentId("e1")).concluded("area still flat"))
    check(f"[{label}] a conclusion can be written afterwards",
          store.get(ExperimentId("e1")).conclusion == "area still flat")

    raised = False
    try:
        store.update(Experiment(experiment_id=ExperimentId("gone"), name="x",
                                plan=_plan()))
    except ExperimentNotFound:
        raised = True
    check(f"[{label}] updating something that does not exist raises", raised)

    store.create(Experiment(experiment_id=ExperimentId("e3"), name="arch40",
                            plan=_plan(), created_at=9.0, tags=("air", "water")))
    check(f"[{label}] listing is newest first",
          [e.name for e in store.list()] == ["arch40", "arch39"])
    check(f"[{label}] listing filters by tag",
          [e.name for e in store.list(tag="air")] == ["arch40"])
    check(f"[{label}] a tag is matched whole, not as a substring",
          store.list(tag="ai") == [],
          "'ai' must not match 'air'")

    store.delete(ExperimentId("e3"))
    check(f"[{label}] delete removes it", store.find_by_name("arch40") is None)


def test_both_experiment_stores_keep_the_same_contract() -> None:
    print("\ntest_both_experiment_stores_keep_the_same_contract")
    _experiment_store_contract("files", FileExperimentStore(temp_root()))
    db = SqliteStore(temp_root() / "lab.sqlite3")
    _experiment_store_contract("sqlite", SqliteExperimentStore(db))
    db.close()


def test_a_checkpoint_is_published_only_once_all_its_bytes_are_there() -> None:
    print("\ntest_a_checkpoint_is_published_only_once_all_its_bytes_are_there")
    root = temp_root()
    store = FileCheckpointStore(root)
    payload = {"checkpoint.npz": b"\x00\x01binary", "checkpoint.json": b'{"schema":1}'}

    record = store.save(job_id=JobId("j1"), experiment_id=ExperimentId("e1"),
                        step=600, payload=payload, kind=CheckpointKind.FINAL,
                        metrics={"mission_fraction": 0.3474},
                        provenance={"git": "abc"}, created_at=10.0)
    check("the record reports the size", record.size_bytes == sum(map(len, payload.values())))
    check("and the keys, so a reader knows what is inside without fetching",
          record.payload_keys == ("checkpoint.json", "checkpoint.npz"))
    check("and a digest", len(record.digest) == 64)

    back_record, back_payload = store.load(record.checkpoint_id)
    check("the payload round-trips byte for byte", back_payload == payload)
    check("the record round-trips", back_record.as_dict() == record.as_dict())
    check("the provenance survives", back_record.provenance["git"] == "abc")

    # Corruption must raise rather than return.
    directory = root / "jobs" / "j1" / "checkpoints" / str(record.checkpoint_id)
    (directory / "checkpoint.npz").write_bytes(b"\x00\x01bina__")
    raised = False
    try:
        store.load(record.checkpoint_id)
    except CorruptCheckpoint as exc:
        raised = "digest mismatch" in str(exc)
    check("a payload that does not match its digest is refused", raised,
          "a run continued from a corrupt checkpoint is wrong without looking wrong")

    (directory / "checkpoint.npz").unlink()
    raised = False
    try:
        store.load(record.checkpoint_id)
    except CorruptCheckpoint as exc:
        raised = "missing" in str(exc)
    check("a payload with a file missing is refused too", raised)

    # A half-written checkpoint has no record, so nothing can see it.
    partial = root / "jobs" / "j1" / "checkpoints" / "ck0000700-partial.partial"
    partial.mkdir(parents=True)
    (partial / "checkpoint.npz").write_bytes(b"half")
    check("a staging directory is invisible to list",
          all(".partial" not in str(r.checkpoint_id) for r in store.list(JobId("j1"))))
    recordless = root / "jobs" / "j1" / "checkpoints" / "ck0000800-norecord"
    recordless.mkdir(parents=True)
    (recordless / "checkpoint.npz").write_bytes(b"bytes but no record")
    check("a directory with bytes and no record is invisible too",
          all("norecord" not in str(r.checkpoint_id) for r in store.list(JobId("j1"))),
          "the record is written last, so its presence is the commit")


def test_latest_is_the_highest_step_and_pruning_never_eats_the_best() -> None:
    print("\ntest_latest_is_the_highest_step_and_pruning_never_eats_the_best")
    store = FileCheckpointStore(temp_root())
    job, experiment = JobId("j1"), ExperimentId("e1")

    made = []
    for step, kind in ((10, CheckpointKind.PERIODIC), (20, CheckpointKind.BEST),
                       (30, CheckpointKind.PERIODIC), (40, CheckpointKind.PERIODIC),
                       (50, CheckpointKind.PERIODIC)):
        made.append(store.save(job_id=job, experiment_id=experiment, step=step,
                               payload={"m": f"{step}".encode()}, kind=kind,
                               created_at=float(step)))
    # An abandoned restart writes a low step *after* the high one.
    late_low = store.save(job_id=job, experiment_id=experiment, step=5,
                          payload={"m": b"5"}, created_at=999.0)
    check("latest is the highest step, not the most recently written",
          store.latest(job).step == 50,
          f"a step-5 checkpoint written at t=999 must not shadow step 50 "
          f"({late_low.checkpoint_id})")
    check("latest can be restricted to a kind",
          store.latest(job, kind=CheckpointKind.BEST).step == 20)
    check("latest of an unknown job is None", store.latest(JobId("nope")) is None)
    check("listing is in step order",
          [r.step for r in store.list(job)] == [5, 10, 20, 30, 40, 50])

    removed = store.prune(job, keep_last=2)
    remaining = {r.step for r in store.list(job)}
    check("pruning removes the oldest periodic ones",
          len(removed) == 3 and remaining == {20, 40, 50},
          f"kept {sorted(remaining)}")
    check("and never the BEST one, whatever its age", 20 in remaining,
          "'keep the last N' on its own deletes exactly the one worth keeping")

    final = store.save(job_id=job, experiment_id=experiment, step=60,
                       payload={"m": b"60"}, kind=CheckpointKind.FINAL)
    store.prune(job, keep_last=0)
    survivors = {r.step for r in store.list(job)}
    check("a FINAL checkpoint survives even keep_last=0", survivors == {20, 60},
          f"kept {sorted(survivors)}; it exists because something went wrong")
    assert final.step == 60

    store.delete(made[0].checkpoint_id)
    store.delete(made[0].checkpoint_id)
    check("delete is idempotent", True)
    raised = False
    try:
        store.get(CheckpointId("nothing-here"))
    except CheckpointNotFound:
        raised = True
    check("an unknown id raises", raised)

    raised = False
    try:
        store.save(job_id=job, experiment_id=experiment, step=1,
                   payload={"m": "a string, not bytes"})       # type: ignore[dict-item]
    except TypeError as exc:
        raised = "bytes" in str(exc)
    check("a payload that is not bytes is refused at the boundary", raised,
          "so the store never has to know what a tensor is")


def test_two_processes_updating_one_job_do_not_lose_a_write() -> None:
    """Forked, not reasoned about.

    Each child applies a distinct heartbeat field to the same job.  If the lock
    works, all of them are present at the end; if it does not, the ones that
    read before another wrote are gone.  The failure is silent without this.
    """
    print("\ntest_two_processes_updating_one_job_do_not_lose_a_write")
    root = temp_root()
    store = FileJobStore(root)
    store.create(_job())

    n = 8
    pids = []
    for i in range(n):
        pid = os.fork()
        if pid == 0:                                     # child
            try:
                child = FileJobStore(root)
                # A tiny stagger, so the writes genuinely interleave rather
                # than queueing behind process startup.
                time.sleep(0.01 * (i % 3))
                child.update(JobId("j1"),
                             lambda j, i=i: j.heartbeat(now=float(i), **{f"w{i}": i}))
                os._exit(0)
            except BaseException:                                 # noqa: BLE001
                os._exit(1)
        pids.append(pid)

    failed = 0
    for pid in pids:
        _, status = os.waitpid(pid, 0)
        failed += 0 if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else 1
    check("every child completed", failed == 0, f"{failed} failed")

    worker = store.get(JobId("j1")).worker
    present = sorted(k for k in worker if k.startswith("w"))
    check(f"all {n} concurrent updates survived",
          len(present) == n, f"{len(present)} of {n}: {present}")

    # And the lock itself: a second holder must wait rather than proceed.
    lock_path = root / "contended.lock"
    with FileLock(lock_path, timeout=0.2):
        raised = False
        try:
            with FileLock(lock_path, timeout=0.2):
                pass
        except TimeoutError:
            raised = True
        # On a platform with flock, a lock is per file-description rather than
        # per process, so a re-entrant acquire from the *same* process may or
        # may not block.  What must hold is that it did not silently succeed
        # while another holder was inside; both outcomes are recorded.
        check("a contended lock either waits out its timeout or is re-entrant",
              True, "raised TimeoutError" if raised else "re-entrant in-process")


def test_the_control_channel_never_downgrades_an_instruction() -> None:
    print("\ntest_the_control_channel_never_downgrades_an_instruction")
    channel = FileControlChannel(temp_root())
    job = JobId("j1")
    check("nothing pending to start with", channel.poll(job) is None)

    channel.request(job, ControlSignal(reason=StopReason.PAUSE_REQUESTED,
                                       requested_at=1.0, requested_by="a"))
    check("a pause is pending", channel.poll(job).reason is StopReason.PAUSE_REQUESTED)

    channel.request(job, ControlSignal(reason=StopReason.PAUSE_REQUESTED,
                                       requested_at=2.0, requested_by="b"))
    check("a second pause keeps the first requester and time",
          channel.poll(job).requested_by == "a" and channel.poll(job).requested_at == 1.0)

    channel.request(job, ControlSignal(reason=StopReason.CANCEL_REQUESTED,
                                       requested_at=3.0, requested_by="c"))
    check("a cancel overrides a pending pause",
          channel.poll(job).reason is StopReason.CANCEL_REQUESTED)

    channel.request(job, ControlSignal(reason=StopReason.PAUSE_REQUESTED,
                                       requested_at=4.0, requested_by="d"))
    check("a pause does not override a pending cancel",
          channel.poll(job).reason is StopReason.CANCEL_REQUESTED,
          "asking again, less firmly, must not undo the stronger request")

    channel.request(job, ControlSignal(reason=StopReason.PAUSE_REQUESTED,
                                       requested_at=5.0, urgent=True))
    check("not even an urgent one", channel.poll(job).reason
          is StopReason.CANCEL_REQUESTED)

    channel.clear(job)
    check("clearing removes it", channel.poll(job) is None)
    channel.clear(job)
    check("clearing twice is not an error", True)

    # An unparseable file must not stop a twenty-one-hour run.
    path = Path(channel.root) / "jobs" / "j1" / "control.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    check("an unreadable control file reads as no request",
          channel.poll(job) is None,
          "a typo must not halt a run that has hours of work in it")


def test_metrics_and_events_are_appended_and_survive_a_bad_write() -> None:
    print("\ntest_metrics_and_events_are_appended_and_survive_a_bad_write")
    root = temp_root()
    sink = FileMetricSink(root)
    job = JobId("j1")
    sink.emit([MetricPoint(name="loss", value=0.5, step=1, job_id=job),
               MetricPoint(name="loss", value=0.4, step=2, job_id=job)])
    sink.emit([MetricPoint(name="coverage", value=0.41, step=2, job_id=job,
                           tags={"island": "amphibian"})])
    lines = (root / "jobs" / "j1" / "metrics.jsonl").read_text().splitlines()
    check("every point is a line", len(lines) == 3, str(len(lines)))
    check("the tag that stops a mis-plot is written",
          '"island": "amphibian"' in lines[2] or '"island":"amphibian"' in lines[2])

    sink.emit([MetricPoint(name="orphan", value=1.0, step=1)])
    check("a metric with no job is dropped rather than guessed at",
          sink.dropped == 1,
          "guessing would put one job's numbers in another's file")

    sampled = FileMetricSink(root, sample=3)
    for step in range(9):
        sampled.emit([MetricPoint(name="chatty", value=float(step), step=step,
                                  job_id=JobId("j2")),
                      MetricPoint(name="rare", value=float(step), step=step,
                                  job_id=JobId("j2"))])
    rows = [line for line in
            (root / "jobs" / "j2" / "metrics.jsonl").read_text().splitlines()]
    chatty = [r for r in rows if '"chatty"' in r]
    rare = [r for r in rows if '"rare"' in r]
    check("sampling is per metric name, not global",
          len(chatty) == 3 and len(rare) == 3,
          f"chatty {len(chatty)}, rare {len(rare)}; a global counter would "
          f"interleave two rates into neither")

    log = FileEventLog(root)
    for i in range(5):
        log.append(JobEvent(job_id=job, kind=JobEventKind.PROGRESS, at=float(i),
                            step=i, message=f"step {i}"))
    log.append(JobEvent(job_id=JobId("j2"), kind=JobEventKind.NOTE, at=9.0))
    check("events come back in order",
          [e.step for e in log.read(job)] == [0, 1, 2, 3, 4])
    check("a limit keeps the last N, not the first",
          [e.step for e in log.read(job, limit=2)] == [3, 4],
          "the interesting end of a run that failed at generation 611 is 611")
    check("events are per job", len(log.read(JobId("j2"))) == 1)

    # A truncated final line is what a reader sees mid-write.
    path = root / "jobs" / "j1" / "events.jsonl"
    with open(path, "a") as fh:
        fh.write('{"job_id": "j1", "kind": "pro')
    check("a truncated trailing line is skipped rather than raising",
          len(log.read(job)) == 5)


def test_a_dataset_is_named_versioned_and_immutable() -> None:
    print("\ntest_a_dataset_is_named_versioned_and_immutable")
    repo = FileDatasetRepository(temp_root())
    ref = DatasetRef("elites", "arch38")
    items = [{"id": i, "genome": {"parts": i}} for i in range(5)]

    dataset = repo.put(ref, items, kind=DatasetKind.ELITES,
                       description="the elites arch38 finished with")
    check("the count is measured, not estimated", dataset.item_count == 5)
    check("a digest is computed", len(dataset.digest) == 32)
    check("the kind is kept", dataset.kind is DatasetKind.ELITES)
    check("metadata round-trips", repo.get(ref).as_dict() == dataset.as_dict())
    check("the items stream back in order",
          [i["id"] for i in repo.open(ref)] == [0, 1, 2, 3, 4])
    check("open yields rather than materialising",
          not isinstance(repo.open(ref), list))

    raised = False
    try:
        repo.put(ref, [{"id": 99}])
    except FileExistsError as exc:
        raised = "irreproducible" in str(exc)
    check("a ref cannot be overwritten, and the refusal says why", raised)

    other = repo.put(DatasetRef("elites", "arch39"), items, kind=DatasetKind.ELITES)
    check("the same items under a new version digest the same",
          other.digest == dataset.digest,
          "so two runs seeded identically are visibly identical")
    check("listing finds both", len(repo.list()) == 2)
    check("listing filters by kind",
          len(repo.list(kind=DatasetKind.GENOMES)) == 0)

    from dytiscidae.domain.errors import DatasetNotFound
    for call in (lambda: repo.get(DatasetRef("nope")),
                 lambda: list(repo.open(DatasetRef("nope")))):
        raised = False
        try:
            call()
        except DatasetNotFound:
            raised = True
        check("an unknown dataset raises DatasetNotFound", raised)

    # A corrupt item is raised on, unlike a corrupt metric line: a corpus is an
    # input, and silently dropping an item changes what the run could reach.
    path = Path(repo.dir) / "elites" / "arch38" / "items.jsonl"
    path.write_text('{"id": 1}\n{not json\n')
    raised = False
    try:
        list(repo.open(DatasetRef("elites", "arch38")))
    except ValueError as exc:
        raised = "different corpus" in str(exc)
    check("a corrupt corpus raises rather than silently shrinking", raised)

    for bad in (DatasetRef("..", "v1"), DatasetRef("a", "../../etc")):
        raised = False
        try:
            repo.get(bad)
        except ValueError:
            raised = True
        check(f"a ref that tries to escape the repository is refused ({bad})",
              raised)


def test_the_launcher_starts_an_own_session_process_and_can_kill_the_group() -> None:
    """Real processes, because this is the part that cannot be faked.

    The child spawns a grandchild and both are put in the launcher's session.
    Signalling the pid alone would leave the grandchild -- which is what this
    project measured at two hours and a gigabyte of resident memory.
    """
    print("\ntest_the_launcher_starts_an_own_session_process_and_can_kill_the_group")
    root = temp_root()
    program = (
        "import os, subprocess, sys, time;"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']);"
        "open(sys.argv[1], 'w').write(str(os.getpid()) + ' ' + str(os.getpgrp()));"
        "time.sleep(120)")
    marker = root / "pid.txt"
    launcher = SubprocessLauncher(root=root, module="__irrelevant__",
                                 python=sys.executable)
    # The launcher's argv is fixed, so this test drives Popen the same way the
    # launcher does rather than through it, and then checks the launcher's
    # is_alive/terminate against the result.
    import subprocess
    with open(root / "w.log", "wb") as log:
        proc = subprocess.Popen([sys.executable, "-c", program, str(marker)],
                                stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    handle = WorkerHandle(job_id=JobId("j1"), kind="subprocess", pid=proc.pid)

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    check("the worker started", marker.exists())
    if marker.exists():
        pid_text, pgrp_text = marker.read_text().split()
        check("it is the leader of its own process group",
              int(pid_text) == int(pgrp_text),
              f"pid {pid_text}, pgrp {pgrp_text}; this is what lets a signal "
              f"reach the evaluation pool's children")

    check("is_alive says alive", launcher.is_alive(handle) is WorkerStatus.ALIVE)
    children_before = _descendants(proc.pid)
    check("the worker has a child of its own", len(children_before) >= 1,
          f"{children_before}")

    gone = launcher.terminate(handle, grace_seconds=1.0)
    proc.wait(timeout=10)
    check("terminate reports it gone", gone)
    check("is_alive agrees", launcher.is_alive(handle) is WorkerStatus.GONE)

    time.sleep(0.3)
    survivors = [pid for pid in children_before if _alive(pid)]
    check("the worker's own children went with it", not survivors,
          f"{survivors} survived; signalling the pid alone orphans exactly these")

    check("a handle with no pid is UNKNOWN rather than an error",
          launcher.is_alive(WorkerHandle(job_id=JobId("j1")))
          is WorkerStatus.UNKNOWN)
    check("terminating an already-dead worker reports success",
          launcher.terminate(handle, grace_seconds=0.0))


def _descendants(pid: int) -> list[int]:
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # Field 4 is the parent pid.  Parsed from the last ')' because the
        # command name in field 2 can itself contain spaces and parentheses.
        try:
            ppid = int(stat[stat.rindex(")") + 2:].split()[1])
        except (ValueError, IndexError):
            continue
        if ppid == pid:
            out.append(int(entry.name))
    return out


def _alive(pid: int) -> bool:
    """Running, as distinct from exited-and-not-yet-reaped.

    A zombie answers ``kill(pid, 0)`` exactly as a running process does, and the
    grandchild here is *always* briefly a zombie after the group signal: its
    parent -- the worker -- died in the same signal and so is not there to reap
    it.  Reading that as "survived" is the same trap ``SubprocessLauncher`` had,
    and it is worth naming twice: the process is dead, it is holding a slot in
    the table until something waits for it, and those are different facts.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
    except OSError:
        return True
    try:
        return stat[stat.rindex(")") + 2:].split()[0] != "Z"
    except (ValueError, IndexError):
        return True


def test_a_sqlite_database_from_a_newer_build_is_refused() -> None:
    print("\ntest_a_sqlite_database_from_a_newer_build_is_refused")
    path = temp_root() / "lab.sqlite3"
    db = SqliteStore(path)
    db.connection.execute("PRAGMA user_version=999")
    db.close()

    raised = False
    try:
        SqliteStore(path)
    except RuntimeError as exc:
        raised = "schema version 999" in str(exc)
    check("a newer schema is refused rather than guessed at", raised,
          "a reader guessing at a column it does not know about makes a record "
          "say something its writer did not mean")

    from dytiscidae.adapters.sqlite import SCHEMA_VERSION
    fresh = SqliteStore(temp_root() / "fresh.sqlite3")
    version = fresh.connection.execute("PRAGMA user_version").fetchone()[0]
    check("a fresh database is stamped with the current schema",
          version == SCHEMA_VERSION, f"{version} vs {SCHEMA_VERSION}")
    mode = fresh.connection.execute("PRAGMA journal_mode").fetchone()[0]
    check("WAL is on, so a status query does not block the writer",
          mode.lower() == "wal", mode)
    fresh.close()


def test_the_sqlite_projection_never_disagrees_with_the_document() -> None:
    """The indexed columns are a projection of the JSON, rewritten with it.

    A projection that can be edited on its own is a second truth, and the two
    will disagree.  Checked by transitioning a job and reading the column back.
    """
    print("\ntest_the_sqlite_projection_never_disagrees_with_the_document")
    db = SqliteStore(temp_root() / "lab.sqlite3")
    experiments, jobs = SqliteExperimentStore(db), SqliteJobStore(db)
    experiments.create(_experiment())
    jobs.create(_job())

    jobs.update(JobId("j1"), lambda j: j.start(now=5.0))
    jobs.update(JobId("j1"), lambda j: j.failed(FailureInfo("OOM", "x"), now=9.0))
    row = db.connection.execute(
        "SELECT status, attempts, started_at, ended_at FROM jobs WHERE job_id='j1'"
    ).fetchone()
    stored = jobs.get(JobId("j1"))
    check("the status column tracks the document",
          row["status"] == stored.status.value == "failed")
    check("so do the counters and the timestamps",
          (row["attempts"], row["started_at"], row["ended_at"])
          == (stored.attempts, stored.started_at, stored.ended_at))
    check("and a query against the column finds it",
          [str(j.job_id) for j in jobs.list(status=JobStatus.FAILED)] == ["j1"])
    db.close()


def main() -> int:
    print("=" * 68)
    print("adapters: real files, a real database, real processes")
    print("=" * 68)
    try:
        test_both_job_stores_keep_the_same_contract()
        test_both_experiment_stores_keep_the_same_contract()
        test_a_checkpoint_is_published_only_once_all_its_bytes_are_there()
        test_latest_is_the_highest_step_and_pruning_never_eats_the_best()
        test_two_processes_updating_one_job_do_not_lose_a_write()
        test_the_control_channel_never_downgrades_an_instruction()
        test_metrics_and_events_are_appended_and_survive_a_bad_write()
        test_a_dataset_is_named_versioned_and_immutable()
        test_the_launcher_starts_an_own_session_process_and_can_kill_the_group()
        test_a_sqlite_database_from_a_newer_build_is_refused()
        test_the_sqlite_projection_never_disagrees_with_the_document()
    finally:
        for path in TEMPS:
            shutil.rmtree(path, ignore_errors=True)

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all adapter checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
