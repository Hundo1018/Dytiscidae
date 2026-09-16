"""``python -m dytiscidae.ops.run job …`` -- the driving adapter.

A driving adapter, in the hexagonal sense: it turns argv into use case requests
and use case responses into text.  It contains no policy.  Every decision --
whether a job may be resumed, whether a cancel needs a kill, what a pause
means -- is made inside, and if any of it were made here, the same decision
would have to be remade by an HTTP handler or a GUI.

    run job start   --experiment arch39 [--workers 4] [--inline]
    run job list    [--experiment …] [--status running]
    run job status  <job-id> [--events]
    run job pause   <job-id>
    run job resume  <job-id> [--extend 300]
    run job cancel  <job-id> [--force]
    run job export  <job-id> --to DIR [--format native]
    run experiment new  --name arch39 --trainer search --steps 900 [--set k=v]
    run experiment list
    run experiment show <name-or-id>

The one thing it does add is *latency honesty* in the output.  ``pause`` prints
that the request has been posted and will take effect at the trainer's next
boundary, because the alternative -- printing "paused" -- is a claim the
application is in no position to make, and a user who believes it will kill the
process when nothing appears to happen.
"""

from __future__ import annotations

import json
import sys

from ..application import (
    CancelTrainingRequest,
    CreateExperimentRequest,
    ExportModelRequest,
    LoadCheckpointRequest,
    PauseTrainingRequest,
    ResumeTrainingRequest,
    StartTrainingRequest,
)
from ..domain import JobStatus, TrainingBudget, TrainingPlan
from ..domain.dataset import DatasetRef
from ..domain.errors import DomainError


def add_arguments(parser) -> None:
    """Attach the ``job`` and ``experiment`` command groups to ``ops/run.py``."""
    parser.add_argument("--root", default=None,
                        help="the lab root (default: runs/lab, or $DYTISCIDAE_LAB)")
    parser.add_argument("--store", default=None, choices=("files", "sqlite"),
                        help="record backend (default: files, or $DYTISCIDAE_STORE)")


def _lab(args):
    import os

    from .composition import DEFAULT_ROOT, build

    root = args.root or os.environ.get("DYTISCIDAE_LAB") or DEFAULT_ROOT
    store = args.store or os.environ.get("DYTISCIDAE_STORE") or "files"
    return build(root, store=store)


def _parse_setting(text: str):
    """``k=v`` where v is parsed as JSON, falling back to the bare string.

    JSON first so that ``--set batch=16`` is an int and ``--set islands='["a"]'``
    is a list.  A bare word that is not JSON stays a string, which is what
    ``--set trainer=search`` wants.  Getting this wrong is quiet: a batch size
    of ``"16"`` reaches the trainer as a string and fails somewhere else.
    """
    key, sep, raw = text.partition("=")
    if not sep:
        raise ValueError(f"--set needs key=value, got {text!r}")
    try:
        return key.strip(), json.loads(raw)
    except json.JSONDecodeError:
        return key.strip(), raw


# -- experiment ------------------------------------------------------------

def cmd_experiment(args) -> int:
    app, lab = _lab(args)
    try:
        if args.experiment_command == "new":
            return _experiment_new(app, args)
        if args.experiment_command == "list":
            for experiment in app.experiment_list(tag=args.tag, limit=args.limit):
                print(f"{experiment.experiment_id}  {experiment.name:<20s} "
                      f"{experiment.plan.trainer:<10s} "
                      f"digest={experiment.plan.digest}  "
                      f"{len(experiment.job_ids)} job(s)"
                      + (f"  [{', '.join(experiment.tags)}]"
                         if experiment.tags else ""))
            return 0
        if args.experiment_command == "show":
            found = (app.experiments.find_by_name(args.name)
                     or app.experiments.get(args.name))
            described = app.describe(found.experiment_id, with_events=args.events)
            print(json.dumps(described, indent=1, default=str))
            return 0
    except DomainError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        lab.close()
    print("give a subcommand: new, list, show", file=sys.stderr)
    return 2


def _experiment_new(app, args) -> int:
    hyperparameters = dict(_parse_setting(s) for s in (args.set or []))
    plan = TrainingPlan(
        trainer=args.trainer,
        hyperparameters=hyperparameters,
        dataset=DatasetRef(*args.dataset.split(":", 1)) if args.dataset else None,
        seed=args.seed,
        budget=TrainingBudget(max_steps=args.steps, max_seconds=args.max_seconds,
                              max_resident_mb=args.memory_ceiling_mb),
        tags=tuple(args.tag or ()))
    response = app.create(CreateExperimentRequest(
        name=args.name, plan=plan, description=args.description or "",
        hypothesis=args.hypothesis or "", tags=tuple(args.tag or ())))
    experiment = response.experiment
    print(f"created {experiment.experiment_id}  {experiment.name}")
    print(f"  plan    {experiment.plan}")
    print(f"  budget  {experiment.plan.budget.as_dict()}")
    if not experiment.hypothesis:
        # Said once, without nagging.  A prediction recorded after the run is
        # not a prediction, so the only moment it can be asked for is this one.
        print("  (no hypothesis recorded; --hypothesis takes one, and after "
              "the run it is too late for it to be a prediction)")
    return 0


# -- job -------------------------------------------------------------------

def cmd_job(args) -> int:
    app, lab = _lab(args)
    try:
        handler = {
            "start": _job_start, "list": _job_list, "status": _job_status,
            "pause": _job_pause, "resume": _job_resume, "cancel": _job_cancel,
            "export": _job_export,
        }.get(args.job_command)
        if handler is None:
            print("give a subcommand: start, list, status, pause, resume, "
                  "cancel, export", file=sys.stderr)
            return 2
        return handler(app, lab, args)
    except DomainError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        lab.close()


def _resources(args) -> dict:
    out = {}
    for name in ("workers", "min_shard", "memory_ceiling_mb"):
        value = getattr(args, name, None)
        if value is not None:
            out[name] = value
    return out


def _job_start(app, lab, args) -> int:
    response = app.start(StartTrainingRequest(
        experiment_name=args.experiment, resources=_resources(args),
        inline=args.inline,
        hyperparameter_overrides=dict(_parse_setting(s) for s in (args.set or []))))
    job = response.job
    print(f"{job.job_id}  {job.status}")
    print(f"  experiment {job.experiment_id}")
    print(f"  plan       {job.plan}")
    print(f"  workspace  {job.workspace}")
    if response.handle is not None:
        print(f"  worker     {response.handle.kind} pid={response.handle.pid}")
        if response.handle.log_path:
            print(f"  log        {response.handle.log_path}")
    if not args.inline:
        # PENDING, and why, in one line -- otherwise the first reaction to a
        # status that is not RUNNING is to assume something is wrong.
        print("  the job is PENDING until the worker reports itself alive; "
              "check with `run job status`")
    if response.state is not None:
        print(f"  finished at step {response.state.step}")
    return 0


def _job_list(app, lab, args) -> int:
    status = JobStatus(args.status) if args.status else None
    experiment_id = None
    if args.experiment:
        found = app.experiments.find_by_name(args.experiment)
        experiment_id = found.experiment_id if found else args.experiment
    jobs = app.job_list(experiment_id=experiment_id, status=status,
                        limit=args.limit)
    if not jobs:
        print("no jobs")
        return 0
    print(f"{'job':<34s} {'status':<11s} {'trainer':<10s} {'step':>8s}  plan")
    for job in jobs:
        state = lab.jobs.load_state(job.job_id)
        step = (f"{state.step}/{state.total_steps}" if state and state.total_steps
                else (str(state.step) if state else "-"))
        print(f"{str(job.job_id):<34s} {job.status.value:<11s} "
              f"{job.plan.trainer:<10s} {step:>8s}  {job.plan.digest}")
    return 0


def _job_status(app, lab, args) -> int:
    report = app.status(args.job_id)
    job, state = report.job, report.state
    print(f"{job.job_id}")
    print(f"  status      {job.status}"
          + (f" ({job.stop_reason})" if job.stop_reason else ""))
    print(f"  experiment  {job.experiment_id}")
    print(f"  plan        {job.plan}")
    print(f"  attempts    {job.attempts}"
          + (f", resumed from {job.resumed_from}" if job.resumed_from else ""))
    print(f"  workspace   {job.workspace}")
    if report.worker_status is not None:
        age = (f", last heartbeat {report.heartbeat_age:.0f}s ago"
               if report.heartbeat_age is not None else "")
        print(f"  worker      {report.worker_status}{age}")
    if state is not None:
        progress = ("-" if state.progress is None
                    else f"{state.progress * 100:.1f}%")
        print(f"  step        {state.step}"
              + (f" / {state.total_steps} ({progress})" if state.total_steps else ""))
        print(f"  samples     {state.samples_seen}")
        print(f"  elapsed     {state.elapsed_seconds:.0f}s")
        eta = state.eta_seconds()
        if eta is not None:
            print(f"  eta         {eta / 3600:.1f}h at the observed rate "
                  f"over {state.step} step(s)")
        if state.metrics:
            print("  metrics     " + "  ".join(
                f"{k}={v:.4g}" for k, v in sorted(state.metrics.items())))
        if state.detail:
            print("  detail      " + "  ".join(
                f"{k}={v}" for k, v in sorted(state.detail.items())))
    if report.latest_checkpoint is not None:
        record = report.latest_checkpoint
        at_risk = state.steps_at_risk if state else "?"
        print(f"  checkpoint  {record.checkpoint_id} at step {record.step} "
              f"[{record.kind}], {at_risk} step(s) at risk")
    else:
        print("  checkpoint  none -- this job cannot be resumed")
    if report.stale:
        print("  ** the job says it is active and the launcher says the worker "
              "is gone. Resume it, or cancel it. **")
    if job.failure is not None:
        print(f"  failure     {job.failure.kind}: {job.failure.message}"
              + (f" (at step {job.failure.step})" if job.failure.step is not None
                 else ""))
        if args.traceback:
            print(job.failure.traceback)
    if args.events:
        print("  events:")
        for event in lab.events.read(job.job_id, limit=args.event_limit):
            print(f"    {event}")
    return 0


def _job_pause(app, lab, args) -> int:
    response = app.pause(PauseTrainingRequest(job_id=args.job_id,
                                              requested_by=args.by or "",
                                              urgent=args.urgent))
    if response.already_stopped:
        print(f"{response.job.job_id} is already {response.job.status}; "
              f"nothing requested")
        return 0
    print(f"{response.job.job_id}  {response.job.status}")
    print("  the request is posted. The trainer honours it at its next "
          "boundary -- for the design search that is the end of the current "
          "generation -- and the job reaches 'paused' then, with a current "
          "checkpoint.")
    return 0


def _job_resume(app, lab, args) -> int:
    response = app.resume(ResumeTrainingRequest(
        job_id=args.job_id,
        checkpoint_id=args.checkpoint, inline=args.inline,
        extend_steps=args.extend,
        resources=_resources(args) or None))
    print(f"{response.job.job_id}  {response.job.status}")
    print(f"  from      {response.resumed_from.checkpoint_id} "
          f"(step {response.resumed_from.step})")
    print(f"  attempt   {response.job.attempts + (0 if args.inline else 1)}")
    if args.extend:
        print(f"  budget    {response.job.plan.budget.max_steps} step(s); "
              f"this is a different configuration and digests as "
              f"{response.job.plan.digest}")
    if response.handle is not None and response.handle.pid:
        print(f"  worker    {response.handle.kind} pid={response.handle.pid}")
    return 0


def _job_cancel(app, lab, args) -> int:
    response = app.cancel(CancelTrainingRequest(
        job_id=args.job_id, requested_by=args.by or "", force=args.force,
        grace_seconds=args.grace))
    job = response.job
    print(f"{job.job_id}  {job.status}")
    if job.status is JobStatus.CANCELLED:
        if response.forced:
            print("  the worker was terminated; everything since its last "
                  "checkpoint is lost.")
        else:
            print("  nothing was running.")
    else:
        print("  the request is posted. The trainer honours it at its next "
              "boundary. Add --force to kill the worker instead, which costs "
              "everything since the last checkpoint.")
    return 0


def _job_export(app, lab, args) -> int:
    response = app.export(ExportModelRequest(
        destination=args.to, job_id=args.job_id if not args.checkpoint else None,
        checkpoint_id=args.checkpoint, fmt=args.format,
        overwrite=args.overwrite))
    print(f"exported {response.record.checkpoint_id} (step {response.record.step})")
    print(f"  to      {response.destination}")
    print(f"  format  {response.fmt}, {response.bytes_written} bytes")
    for key, value in sorted(response.detail.items()):
        if key not in ("bytes_written", "format", "path"):
            print(f"  {key:<7s} {value}")
    return 0


def build_parsers(subparsers) -> None:
    """Register ``job`` and ``experiment`` on ``ops/run.py``'s subparser set."""
    # -- experiment
    experiment = subparsers.add_parser(
        "experiment", help="create and inspect experiments (the record)")
    add_arguments(experiment)
    experiment.set_defaults(fn=cmd_experiment)
    exp_sub = experiment.add_subparsers(dest="experiment_command")

    new = exp_sub.add_parser("new", help="record a new experiment")
    new.add_argument("--name", required=True)
    new.add_argument("--trainer", default="search",
                     help="which Trainer runs it (default: search)")
    new.add_argument("--steps", type=int, default=None,
                     help="step budget; for the search a step is one generation")
    new.add_argument("--max-seconds", type=float, default=None)
    new.add_argument("--memory-ceiling-mb", type=int, default=None)
    new.add_argument("--seed", type=int, default=0)
    new.add_argument("--dataset", default=None, help="seed corpus, as name:version")
    new.add_argument("--set", action="append", metavar="KEY=VALUE",
                     help="a hyperparameter; the value is parsed as JSON")
    new.add_argument("--description", default=None)
    new.add_argument("--hypothesis", default=None,
                     help="what this is predicted to show, recorded before it runs")
    new.add_argument("--tag", action="append")

    listing = exp_sub.add_parser("list", help="every experiment, newest first")
    listing.add_argument("--tag", default=None)
    listing.add_argument("--limit", type=int, default=None)

    show = exp_sub.add_parser("show", help="an experiment and every job under it")
    show.add_argument("name")
    show.add_argument("--events", action="store_true")

    # -- job
    job = subparsers.add_parser("job", help="start, watch and stop training jobs")
    add_arguments(job)
    job.set_defaults(fn=cmd_job)
    job_sub = job.add_subparsers(dest="job_command")

    start = job_sub.add_parser("start", help="start a job under an experiment")
    start.add_argument("--experiment", required=True, help="the experiment's name")
    start.add_argument("--workers", type=int, default=None)
    start.add_argument("--min-shard", type=int, default=None)
    start.add_argument("--memory-ceiling-mb", type=int, default=None)
    start.add_argument("--inline", action="store_true",
                       help="run in this process instead of a worker. A long "
                            "run started this way dies with the terminal")
    start.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="override a hyperparameter for this job only; it "
                            "becomes a different configuration and says so")

    listing = job_sub.add_parser("list", help="jobs, newest first")
    listing.add_argument("--experiment", default=None)
    listing.add_argument("--status", default=None,
                         choices=[s.value for s in JobStatus])
    listing.add_argument("--limit", type=int, default=20)

    status = job_sub.add_parser("status", help="one job in detail")
    status.add_argument("job_id")
    status.add_argument("--events", action="store_true")
    status.add_argument("--event-limit", type=int, default=20)
    status.add_argument("--traceback", action="store_true",
                        help="print a failed job's traceback")

    pause = job_sub.add_parser("pause", help="stop at the next boundary, resumably")
    pause.add_argument("job_id")
    pause.add_argument("--by", default=None)
    pause.add_argument("--urgent", action="store_true")

    resume = job_sub.add_parser("resume", help="continue from a checkpoint")
    resume.add_argument("job_id")
    resume.add_argument("--checkpoint", default=None,
                        help="which one (default: the job's latest)")
    resume.add_argument("--extend", type=int, default=None,
                        help="raise the step budget by this many")
    resume.add_argument("--inline", action="store_true")
    resume.add_argument("--workers", type=int, default=None)
    resume.add_argument("--min-shard", type=int, default=None)
    resume.add_argument("--memory-ceiling-mb", type=int, default=None)

    cancel = job_sub.add_parser("cancel", help="stop for good")
    cancel.add_argument("job_id")
    cancel.add_argument("--by", default=None)
    cancel.add_argument("--force", action="store_true",
                        help="kill the worker if it does not stop; everything "
                             "since the last checkpoint is lost")
    cancel.add_argument("--grace", type=float, default=30.0)

    export = job_sub.add_parser("export", help="write a checkpoint out as files")
    export.add_argument("job_id", nargs="?", default=None)
    export.add_argument("--to", required=True, metavar="DIR")
    export.add_argument("--checkpoint", default=None)
    export.add_argument("--format", default="native")
    export.add_argument("--overwrite", action="store_true")


def cmd_checkpoints(args) -> int:
    """``run checkpoints <job-id>`` -- what a job has to resume or export from."""
    app, lab = _lab(args)
    try:
        records = lab.checkpoints.list(args.job_id)
        if not records:
            print(f"{args.job_id} has no checkpoints")
            return 0
        for record in records:
            metrics = "  ".join(f"{k}={v:.4g}" for k, v in
                                sorted(record.metrics.items()))
            print(f"{record.checkpoint_id}  step={record.step:<7d} "
                  f"{str(record.kind):<8s} {record.size_bytes:>10d}B  "
                  f"{record.digest[:12]}  {metrics}")
        if args.verify:
            for record in records:
                try:
                    app.load(LoadCheckpointRequest(
                        checkpoint_id=record.checkpoint_id))
                except Exception as exc:                          # noqa: BLE001
                    print(f"  {record.checkpoint_id}: {type(exc).__name__}: {exc}")
                else:
                    print(f"  {record.checkpoint_id}: verified")
        return 0
    finally:
        lab.close()


def default_root() -> str:
    from .composition import DEFAULT_ROOT
    return DEFAULT_ROOT

