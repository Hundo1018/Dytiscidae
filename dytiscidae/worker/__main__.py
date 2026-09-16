"""``python -m dytiscidae.worker`` -- the isolated training process.

Started by ``SubprocessLauncher`` in its own session, so it survives the
terminal that launched it and can be signalled as a group.  It takes a job id
and a root, opens the same stores the application uses, and runs the job to a
recorded conclusion.

    python -m dytiscidae.worker --job-id job-20260916T... --root runs/lab

It is deliberately runnable by hand.  A worker that can only be started by the
application is a worker nobody can reproduce a failure in, and the failures that
matter here happen on a machine that has since been reclaimed.

The exit code is the *worker's*, not the training's:

    0   the job reached a recorded conclusion -- including FAILED and CANCELLED
    1   the worker could not record anything, which is the only real error here
    2   bad arguments

A failed training exiting 0 looks wrong for about a second and then stops:
the worker's job is to record what happened, and it did that.  A supervisor that
wants to know whether the training succeeded reads the job's status, which is
the point of having one.
"""

from __future__ import annotations

import argparse
import sys

from ..domain.ids import JobId


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dytiscidae.worker",
        description="Run one training job to a recorded conclusion.")
    parser.add_argument("--job-id", required=True,
                        help="the job to run; it must already exist in the store")
    parser.add_argument("--root", required=True,
                        help="the lab root the stores live under")
    parser.add_argument("--store", default="files", choices=("files", "sqlite"),
                        help="which record backend the lab uses (default: files)")
    parser.add_argument("--workspace", default=None,
                        help="ignored; the job's own workspace is used. Accepted "
                             "so the launcher's argv is self-describing in ps "
                             "output and in the worker log")
    parser.add_argument("--no-signals", action="store_true",
                        help="do not install SIGTERM/SIGINT handlers. Without "
                             "them a container reclamation kills the run instead "
                             "of pausing it")
    args = parser.parse_args(argv)

    try:
        job_id = JobId(args.job_id)
    except ValueError as exc:
        print(f"bad --job-id: {exc}", file=sys.stderr)
        return 2

    # Imported here rather than at module scope so that ``--help`` works on a
    # machine where the adapters cannot be imported at all -- which is exactly
    # the machine somebody is debugging when they run this by hand.
    from ..adapters.composition import Lab

    lab = Lab(args.root, store=args.store)
    try:
        worker = lab.worker(job_id,
                            install_signal_handlers=not args.no_signals)
        job = worker.run()
    except Exception as exc:                                      # noqa: BLE001
        # Nothing could be recorded: the store was unreachable, the job did not
        # exist.  This is the one case that is the worker's own failure, and it
        # is the one case that exits non-zero.
        import traceback
        print(f"worker for {job_id} could not run: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        lab.close()

    print(f"{job.job_id} finished as {job.status}"
          + (f" ({job.stop_reason})" if job.stop_reason else ""), flush=True)
    if job.failure is not None:
        print(f"  {job.failure.kind}: {job.failure.message}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
