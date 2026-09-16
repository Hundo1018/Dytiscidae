"""The search trainer adapter, tested without running a search.

``SearchTrainer`` is a translator: plan in, ``SearchConfig`` out; generation
report in, job state out; run directory in, checkpoint out.  None of that needs
a physics engine, and testing it against one would mean minutes per case on a
machine with MuJoCo built.  So ``run_search`` is replaced with a stub that
reproduces *the real loop's call ordering*, and the adapter is checked against
it.

The ordering is the point.  The real loop does, per generation:

    report -> on_generation(...)          # progress
    if gen % checkpoint_every == 0:       # the archives and search_state.pkl
        save archives; save_state(gen)    # reach disk here and nowhere else
    should_stop(gen, report)              # and only now is the disk current

An adapter that published its checkpoint from ``on_generation`` copies the
*previous* checkpoint's files and labels them with this generation's number --
a stored checkpoint claiming to be further along than it is, which a resume
would believe.  That is a real defect this adapter had, found by running a real
four-generation search and noticing that generation 0 published nothing; the
stub below reproduces the ordering so it cannot come back.

The parts that genuinely need MuJoCo -- that a generation runs, that the scores
mean anything -- are `tests/test_search.py`'s business and are not duplicated
here.

Run:  PYTHONPATH=. python tests/test_search_adapter.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "disable")

from dytiscidae.domain import (                                    # noqa: E402
    CheckpointKind, JobId, StopReason, TrainingBudget, TrainingPlan,
    TrainingState,
)
from dytiscidae.ports.trainer import StopRequest                   # noqa: E402

FAILURES: list[str] = []
TEMPS: list[Path] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def temp_dir() -> Path:
    path = Path(tempfile.mkdtemp(prefix="search-adapter-"))
    TEMPS.append(path)
    return path


class StubContext:
    """A ``TrainingContext`` that records everything instead of storing it."""

    def __init__(self, plan: TrainingPlan, workspace: Path, *,
                 resources: dict | None = None,
                 stop_at_step: int | None = None,
                 stop_reason: StopReason = StopReason.PAUSE_REQUESTED,
                 resume: tuple | None = None) -> None:
        self._plan = plan
        self._workspace = workspace
        self._resources = dict(resources or {})
        self._state = TrainingState(job_id=JobId("j1"),
                                    total_steps=plan.budget.max_steps)
        self._stop_at = stop_at_step
        self._stop_reason = stop_reason
        self._resume = resume
        self.reports: list[TrainingState] = []
        self.checkpoints: list[tuple[int, CheckpointKind, list[str]]] = []
        self.notes: list[str] = []
        self.resume_calls = 0

    # reads
    @property
    def plan(self): return self._plan

    @property
    def state(self): return self._state

    @property
    def workspace(self): return str(self._workspace)

    @property
    def resources(self): return dict(self._resources)

    @property
    def dataset(self): return None

    def items(self): return iter(())

    # writes
    def report(self, state, *, metrics=None):
        self._state = state
        self.reports.append(state)

    def save_checkpoint(self, payload, *, step, kind=CheckpointKind.PERIODIC,
                        metrics=None, provenance=None):
        from dytiscidae.domain.checkpoint import CheckpointId, CheckpointRecord
        from dytiscidae.domain.ids import ExperimentId
        self.checkpoints.append((int(step), CheckpointKind(kind),
                                 sorted(payload)))
        record = CheckpointRecord(
            checkpoint_id=CheckpointId(f"ck{step:07d}-{len(self.checkpoints)}"),
            job_id=JobId("j1"), experiment_id=ExperimentId("e1"),
            step=int(step), kind=CheckpointKind(kind),
            payload_keys=tuple(sorted(payload)),
            provenance=dict(provenance or {}))
        self._state = self._state.checkpointed(record.checkpoint_id, step=int(step))
        return record

    def resume_from(self):
        self.resume_calls += 1
        return self._resume

    def stop_requested(self):
        if self._stop_at is None or self._state.step < self._stop_at:
            return None
        return StopRequest(reason=self._stop_reason, urgent=False)

    def note(self, message, **detail):
        self.notes.append(message)


def make_stub_loop(*, checkpoint_every: int, generations: int,
                   workspace: Path, written: list[int]):
    """A ``run_search`` that reproduces the real loop's per-generation ordering.

    ``written`` collects the generations at which it wrote ``search_state.pkl``,
    so a test can assert that every published checkpoint corresponds to one.
    """

    def stub(cfg, spec=None, on_generation=None, should_stop=None):
        class _State:
            evaluated = 0
            tier0_rejected = 0
            stopped_at = None
            stop_reason = None

        state = _State()
        for gen in range(generations):
            report = {"generation": gen, "island": "generalist",
                      "regime": "explore", "evaluated": (gen + 1) * 2,
                      "filled": float(gen + 1), "coverage": 0.01 * (gen + 1),
                      "qd_score": 0.5 * (gen + 1), "best_fitness": 0.1 * gen,
                      "mission_best": 0.01 * gen, "elapsed": 74.0 * (gen + 1),
                      "curriculum": {"typical": 1, "reached": 2},
                      "critic": {"calibration": 0.5},
                      "auditor": {"invalidated": 0}}
            state.evaluated = report["evaluated"]

            if on_generation is not None:
                on_generation(state, report)

            if gen % max(checkpoint_every, 1) == 0:
                # What the loop writes, and only here.
                (workspace / "search_state.pkl").write_bytes(
                    f"state at {gen}".encode())
                (workspace / "archive_generalist.pkl").write_bytes(
                    f"archive at {gen}".encode())
                (workspace / "archive_generalist.json").write_bytes(b"{}")
                (workspace / "checkpoint.npz").write_bytes(f"npz {gen}".encode())
                (workspace / "checkpoint.json").write_bytes(b'{"schema":1}')
                written.append(gen)

            if should_stop is not None:
                reason = should_stop(gen, report)
                if reason:
                    (workspace / "search_state.pkl").write_bytes(
                        f"state at {gen}".encode())
                    written.append(gen)
                    state.stopped_at = gen
                    state.stop_reason = str(reason)
                    return state
        return state

    return stub


def _with_stub_loop(stub):
    """Swap ``run_search`` on the loop module for the duration of a call."""
    from dytiscidae.evolution import loop

    original = loop.run_search
    loop.run_search = stub
    return original


def _restore_loop(original) -> None:
    from dytiscidae.evolution import loop
    loop.run_search = original


def _plan(steps: int = 6, **hp) -> TrainingPlan:
    hp.setdefault("islands", ["generalist"])
    return TrainingPlan(trainer="search", seed=20260901, hyperparameters=hp,
                        budget=TrainingBudget(max_steps=steps))


# --------------------------------------------------------------------------


def test_the_plan_translates_into_a_search_config() -> None:
    print("\ntest_the_plan_translates_into_a_search_config")
    from dytiscidae.adapters.trainers.search import SearchTrainer

    trainer = SearchTrainer()
    caps = trainer.capabilities()
    check("a step is a generation, and it says so",
          caps.step_unit == "generation", caps.step_unit)
    check("it promises cooperative stop, checkpoints, resume and export",
          caps.cooperative_stop and caps.checkpoints and caps.resumes
          and caps.exports)

    workspace = temp_dir()
    plan = _plan(steps=9, batch=16, segment_seconds=8.0,
                 controller_refine_steps=2, use_shared_policy=False,
                 checkpoint_every=3, cycles=2, seconds_per_domain=12.0)
    context = StubContext(plan, workspace,
                          resources={"workers": 4, "min_shard": 4})
    cfg = trainer._config(context, workspace, resume=False)

    check("the step budget becomes the generation count", cfg.generations == 9,
          str(cfg.generations))
    check("the seed carries", cfg.seed == 20260901)
    check("the workspace becomes the run directory",
          cfg.run_dir == str(workspace))
    check("hyperparameters reach the config",
          (cfg.batch, cfg.segment_seconds, cfg.controller_refine_steps)
          == (16, 8.0, 2))
    check("islands become a tuple", cfg.islands == ("generalist",),
          str(cfg.islands))
    check("resources reach the config, not the plan",
          (cfg.workers, cfg.min_shard) == (4, 4),
          "pool shape is a property of the machine; this project measured it at "
          "3x on wall time and nothing on the result")
    check("mission settings are routed to the spec, not the config",
          not hasattr(cfg, "cycles"))

    # The budget wins over a stray ``generations`` hyperparameter: two places
    # saying how long a run is would eventually disagree.
    both = _plan(steps=5, generations=900)
    cfg2 = trainer._config(StubContext(both, workspace), workspace, resume=False)
    check("the budget wins over a 'generations' hyperparameter",
          cfg2.generations == 5, str(cfg2.generations))

    open_budget = TrainingPlan(trainer="search", seed=1,
                               hyperparameters={"generations": 7},
                               budget=TrainingBudget())
    cfg3 = trainer._config(StubContext(open_budget, workspace), workspace,
                           resume=False)
    check("without a budget the hyperparameter is used", cfg3.generations == 7)

    raised = False
    try:
        trainer._config(StubContext(
            TrainingPlan(trainer="search", seed=1, budget=TrainingBudget()),
            workspace), workspace, resume=False)
    except ValueError as exc:
        raised = "step budget" in str(exc)
    check("a search with no budget at all is refused", raised)

    raised = False
    try:
        trainer._config(StubContext(_plan(segment_secnods=8.0), workspace),
                        workspace, resume=False)
    except ValueError as exc:
        raised = "segment_secnods" in str(exc) and "Refusing" in str(exc)
    check("a misspelled hyperparameter is refused rather than ignored", raised,
          "a typo that falls back to a default produces a run that is not the "
          "one that was asked for and says nothing about it")

    check("a memory ceiling in the budget reaches the loop",
          trainer._config(StubContext(
              TrainingPlan(trainer="search", seed=1,
                           budget=TrainingBudget(max_steps=3,
                                                 max_resident_mb=4096)),
              workspace), workspace, resume=False).memory_ceiling_mb == 4096)


def test_every_generation_is_reported_with_the_island_it_belongs_to() -> None:
    print("\ntest_every_generation_is_reported_with_the_island_it_belongs_to")
    from dytiscidae.adapters.trainers.search import SearchTrainer

    workspace = temp_dir()
    plan = _plan(steps=4, checkpoint_every=1)
    context = StubContext(plan, workspace)
    written: list[int] = []
    original = _with_stub_loop(make_stub_loop(checkpoint_every=1, generations=4,
                                              workspace=workspace, written=written))
    try:
        outcome = SearchTrainer().train(context)
    finally:
        _restore_loop(original)

    check("one report per generation", len(context.reports) == 4,
          str(len(context.reports)))
    check("the steps are the generation numbers",
          [s.step for s in context.reports] == [0, 1, 2, 3])
    check("the evaluation count is carried as samples",
          [s.samples_seen for s in context.reports] == [2, 4, 6, 8])

    first = context.reports[0]
    check("the island is on the metric's tags",
          first.detail.get("island") == "generalist", str(first.detail),
          )
    check("per-island quantities are published",
          {"filled", "coverage", "qd_score"} <= set(first.metrics),
          str(sorted(first.metrics)))
    check("so are the global ones",
          {"mission_best", "best_fitness"} <= set(first.metrics))
    check("nested report groups are flattened with their prefix",
          "curriculum.typical" in first.metrics and
          "critic.calibration" in first.metrics,
          str([k for k in first.metrics if "." in k]))
    check("the run completed", outcome.stop_reason is StopReason.COMPLETED,
          str(outcome.stop_reason))
    check("the summary carries what the run did",
          outcome.summary.get("generations") == 4
          and outcome.summary.get("evaluated") == 8,
          str(dict(outcome.summary)))


def test_a_checkpoint_is_only_published_where_the_loop_wrote_one() -> None:
    """The ordering defect, made impossible.

    Every published step must be a generation at which the loop actually wrote
    ``search_state.pkl``.  Publishing from ``on_generation`` -- which runs
    *before* that write -- would ship the previous checkpoint under this
    generation's number, and a resume would believe it.
    """
    print("\ntest_a_checkpoint_is_only_published_where_the_loop_wrote_one")
    from dytiscidae.adapters.trainers.search import SearchTrainer

    for checkpoint_every in (1, 2, 3):
        workspace = temp_dir()
        plan = _plan(steps=6, checkpoint_every=checkpoint_every)
        context = StubContext(plan, workspace)
        written: list[int] = []
        original = _with_stub_loop(make_stub_loop(
            checkpoint_every=checkpoint_every, generations=6,
            workspace=workspace, written=written))
        try:
            SearchTrainer().train(context)
        finally:
            _restore_loop(original)

        periodic = [(step, kind) for step, kind, _ in context.checkpoints
                    if kind is CheckpointKind.PERIODIC]
        steps = [step for step, _ in periodic]
        check(f"[every {checkpoint_every}] every publish lands on a write",
              all(s in written for s in steps),
              f"published {steps}, loop wrote at {sorted(set(written))}")
        check(f"[every {checkpoint_every}] and on the loop's own cadence",
              steps == [g for g in range(6) if g % checkpoint_every == 0],
              str(steps))

        final = [(step, kind) for step, kind, _ in context.checkpoints
                 if kind is CheckpointKind.FINAL]
        check(f"[every {checkpoint_every}] exactly one FINAL, at the last step",
              final == [(5, CheckpointKind.FINAL)], str(final))

        keys = context.checkpoints[0][2]
        check(f"[every {checkpoint_every}] the payload is the resumable set",
              keys == ["archive_generalist.pkl", "checkpoint.json",
                       "checkpoint.npz", "search_state.pkl"],
              str(keys))
        check(f"[every {checkpoint_every}] the derived .json export is left out",
              "archive_generalist.json" not in keys,
              "it is regenerated from the .pkl on every checkpoint")

    # And thinning: publish every other checkpoint.
    workspace = temp_dir()
    context = StubContext(_plan(steps=8, checkpoint_every=2, publish_every=2),
                          workspace)
    written = []
    original = _with_stub_loop(make_stub_loop(checkpoint_every=2, generations=8,
                                              workspace=workspace, written=written))
    try:
        SearchTrainer().train(context)
    finally:
        _restore_loop(original)
    steps = [s for s, k, _ in context.checkpoints if k is CheckpointKind.PERIODIC]
    check("publish_every thins the checkpoints, counted in checkpoints",
          steps == [0, 4], str(steps))


def test_nothing_is_published_before_the_loop_has_written_anything() -> None:
    """A checkpoint that exists and cannot be resumed from is worse than none.

    ``search_state.pkl`` is what ``--resume`` reads.  Without it the rest is an
    archive, not a resumable run, and publishing it under a name that promises
    resumability is a lie the resume discovers hours later.
    """
    print("\ntest_nothing_is_published_before_the_loop_has_written_anything")
    from dytiscidae.adapters.trainers.search import SearchTrainer

    workspace = temp_dir()
    context = StubContext(_plan(steps=2), workspace)

    def empty_loop(cfg, spec=None, on_generation=None, should_stop=None):
        class _State:
            evaluated = 0
            tier0_rejected = 0
            stopped_at = None
            stop_reason = None
        # Writes the archives but never ``search_state.pkl``.
        (workspace / "archive_generalist.pkl").write_bytes(b"archive")
        (workspace / "checkpoint.npz").write_bytes(b"npz")
        if on_generation is not None:
            on_generation(_State(), {"generation": 0, "island": "generalist"})
        if should_stop is not None:
            should_stop(0, {"generation": 0})
        return _State()

    original = _with_stub_loop(empty_loop)
    try:
        outcome = SearchTrainer().train(context)
    finally:
        _restore_loop(original)

    check("nothing was published", context.checkpoints == [],
          str(context.checkpoints))
    check("the outcome says there is no checkpoint",
          outcome.final_checkpoint is None)
    check("and a note says the job cannot be resumed",
          any("nothing to resume" in n for n in context.notes),
          str(context.notes))


def test_a_stop_is_honoured_and_mapped_to_its_reason() -> None:
    print("\ntest_a_stop_is_honoured_and_mapped_to_its_reason")
    from dytiscidae.adapters.trainers.search import SearchTrainer

    for reason in (StopReason.PAUSE_REQUESTED, StopReason.CANCEL_REQUESTED,
                   StopReason.SIGNAL):
        workspace = temp_dir()
        context = StubContext(_plan(steps=20, checkpoint_every=1), workspace,
                              stop_at_step=3, stop_reason=reason)
        written: list[int] = []
        original = _with_stub_loop(make_stub_loop(
            checkpoint_every=1, generations=20, workspace=workspace,
            written=written))
        try:
            outcome = SearchTrainer().train(context)
        finally:
            _restore_loop(original)

        check(f"[{reason}] the run stopped early",
              len(context.reports) < 20, f"{len(context.reports)} generations")
        check(f"[{reason}] the outcome carries the reason",
              outcome.stop_reason is reason, str(outcome.stop_reason))
        check(f"[{reason}] a final checkpoint was left behind",
              outcome.final_checkpoint is not None)
        check(f"[{reason}] the summary says where it stopped",
              outcome.summary.get("generations") == len(context.reports),
              str(dict(outcome.summary)))


def test_a_resume_unpacks_the_checkpoint_into_the_run_directory() -> None:
    """``--resume`` reads files, and rewriting that would be rewriting the
    measured part.  So the adapter puts the files back."""
    print("\ntest_a_resume_unpacks_the_checkpoint_into_the_run_directory")
    from dytiscidae.adapters.trainers.search import SearchTrainer
    from dytiscidae.domain.checkpoint import CheckpointRecord
    from dytiscidae.domain.ids import CheckpointId, ExperimentId

    workspace = temp_dir()
    # Something stale in the workspace, which the restore must overwrite.
    (workspace / "search_state.pkl").write_bytes(b"stale")
    record = CheckpointRecord(checkpoint_id=CheckpointId("ck0000600-abc"),
                              job_id=JobId("j1"),
                              experiment_id=ExperimentId("e1"), step=600)
    payload = {"search_state.pkl": b"state at 600",
               "archive_generalist.pkl": b"archive at 600",
               "checkpoint.npz": b"npz 600", "checkpoint.json": b'{"schema":1}'}
    context = StubContext(_plan(steps=2), workspace, resume=(record, payload))

    seen = {}

    def loop(cfg, spec=None, on_generation=None, should_stop=None):
        seen["resume"] = cfg.resume
        class _State:
            evaluated = 0
            tier0_rejected = 0
            stopped_at = None
            stop_reason = None
        return _State()

    original = _with_stub_loop(loop)
    try:
        SearchTrainer().train(context)
    finally:
        _restore_loop(original)

    check("the loop was told to resume", seen.get("resume") is True)
    for name, blob in payload.items():
        check(f"{name} is in the run directory",
              (workspace / name).read_bytes() == blob)
    check("a stale file was overwritten rather than merged with",
          (workspace / "search_state.pkl").read_bytes() == b"state at 600",
          "half from the store and half from a previous attempt is a state "
          "nobody can reason about")
    check("the restore is recorded as a note",
          any("restored 4 file(s)" in n for n in context.notes),
          str(context.notes))

    # And a fresh start does not claim to resume.
    fresh = StubContext(_plan(steps=2), temp_dir())
    original = _with_stub_loop(loop)
    try:
        SearchTrainer().train(fresh)
    finally:
        _restore_loop(original)
    check("a job with no checkpoint starts fresh", seen.get("resume") is False)


def test_the_loop_gained_a_stop_hook_that_is_off_by_default() -> None:
    """The one change to the measured system, checked at the signature.

    ``should_stop=None`` must remain the default, because that is the behaviour
    every stored run was produced under.
    """
    print("\ntest_the_loop_gained_a_stop_hook_that_is_off_by_default")
    import inspect

    from dytiscidae.evolution import loop

    # The stub swap above replaces the attribute; read the source module's own
    # function rather than whatever is currently bound.
    source = inspect.getsource(loop)
    signature = source[source.index("def run_search("):]
    signature = signature[:signature.index(")")]
    check("run_search takes should_stop", "should_stop" in signature, signature)
    check("and it defaults to None", "should_stop=None" in signature)
    check("on_generation is still there and still defaults to None",
          "on_generation=None" in signature)

    body = source[source.index("def run_search("):]
    body = body[:body.index("\ndef _resident_mb")]
    check("the hook is polled after the checkpoint block",
          body.index("save_state(state, gen)") < body.index("should_stop(gen, report)"),
          "stopping before the archives are on disk would leave a checkpoint "
          "that cannot be resumed from")
    check("a stop forces a write before breaking",
          body.index("should_stop(gen, report)")
          < body.index('telemetry.event({"kind": "requested_stop"'),
          "so honouring a stop costs the generation in flight, not the "
          "generations since the last periodic checkpoint")
    check("SearchState carries where and why it stopped",
          hasattr(loop.SearchState, "__dataclass_fields__")
          and {"stopped_at", "stop_reason"}
          <= set(loop.SearchState.__dataclass_fields__),
          "the archives look the same whether a run finished or was paused")


def main() -> int:
    print("=" * 68)
    print("search adapter: translation and ordering, without a physics engine")
    print("=" * 68)
    try:
        # A probe, not a use: the suite is skipped rather than failed on a
        # machine where the search loop cannot be imported at all, because its
        # absence is an environment fact and not a defect in this adapter.
        import numpy

        from dytiscidae.evolution import loop
        assert numpy.__version__ and loop.run_search is not None
    except Exception as exc:                                      # noqa: BLE001
        print(f"\nSKIP: the search loop is not importable here "
              f"({type(exc).__name__}: {exc})")
        return 0

    try:
        test_the_plan_translates_into_a_search_config()
        test_every_generation_is_reported_with_the_island_it_belongs_to()
        test_a_checkpoint_is_only_published_where_the_loop_wrote_one()
        test_nothing_is_published_before_the_loop_has_written_anything()
        test_a_stop_is_honoured_and_mapped_to_its_reason()
        test_a_resume_unpacks_the_checkpoint_into_the_run_directory()
        test_the_loop_gained_a_stop_hook_that_is_off_by_default()
    finally:
        for path in TEMPS:
            shutil.rmtree(path, ignore_errors=True)

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all search-adapter checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
