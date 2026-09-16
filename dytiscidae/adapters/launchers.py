"""``JobLauncher`` adapters: in-process, and an isolated worker process.

Two implementations, and the difference between them is the architecture's
point 5 -- long training is isolated from the main process -- made switchable so
that it can be tested without waiting twenty-one hours.

``InlineLauncher`` runs the worker in the calling process, synchronously.  It is
for tests and for a run short enough to watch.  A long run started this way dies
with its terminal, which is exactly what isolation is for, so the class says so
in its docstring rather than leaving it to be discovered.

``SubprocessLauncher`` starts ``python -m dytiscidae.worker`` in its own session
and returns.  Its two decisions:

**``start_new_session=True``.**  The worker gets its own process group and
session, so it survives the terminal that started it and so a signal can be
delivered to the whole group.  This is what ``setsid nohup`` does in this
project's own launch command, and it is here for the same reason: the search
pool spawns evaluation workers that outlive a signal sent to the parent alone.
This project measured those children surviving two hours holding a gigabyte.

**``terminate`` signals the group, not the pid.**  ``os.killpg`` on the worker's
session, SIGTERM then SIGKILL after the grace period.  Signalling the pid alone
orphans exactly the children the previous paragraph is about.

Neither launcher ever matches a process by its command line.  ``pkill -f`` and
``pgrep -f`` match the shell that runs them, which in this project's own
operating notes killed the wrong process; the handle carries a pid and the pid
is what gets signalled.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

from ..domain.ids import JobId
from ..ports.launcher import WorkerHandle, WorkerStatus


class InlineLauncher:
    """Runs the worker in this process, synchronously.

    ``launch`` returns only when the training has finished, so the job is
    already in a terminal or paused state when the use case looks at it.  That
    is why ``StartTrainingResponse.state`` is populated for an inline start and
    not for a launched one.

    Not for a long run: there is no isolation, the training holds the calling
    process for its whole duration, and anything that kills the caller kills the
    run. Use ``SubprocessLauncher`` for that.
    """

    kind = "inline"

    def __init__(self, run_worker) -> None:
        # A callable taking ``(job_id, workspace)``.  Injected rather than
        # imported, so this module does not depend on the worker package and
        # a test can pass its own.
        self._run_worker = run_worker

    def launch(self, job_id: JobId, *, workspace: str,
               env: Mapping[str, str] | None = None) -> WorkerHandle:
        started = time.time()
        self._run_worker(JobId(job_id), workspace)
        return WorkerHandle(job_id=JobId(job_id), kind=self.kind, pid=os.getpid(),
                            started_at=started, detail={"synchronous": True})

    def is_alive(self, handle: WorkerHandle) -> WorkerStatus:
        # ``launch`` only returns once the training is over, so by the time
        # anybody holds a handle there is nothing left running.
        return WorkerStatus.GONE

    def terminate(self, handle: WorkerHandle, *, grace_seconds: float = 30.0) -> bool:
        return True


class SubprocessLauncher:
    """Starts ``python -m dytiscidae.worker`` in its own session."""

    kind = "subprocess"

    def __init__(self, *, root: str | Path, python: str | None = None,
                 module: str = "dytiscidae.worker",
                 extra_args: tuple[str, ...] = (),
                 env: Mapping[str, str] | None = None,
                 cwd: str | Path | None = None) -> None:
        self.root = Path(root)
        self.python = python or sys.executable
        self.module = module
        self.extra_args = tuple(extra_args)
        self._env = dict(env or {})
        self.cwd = str(cwd) if cwd else None

    def _log_path(self, job_id: JobId) -> Path:
        return self.root / "jobs" / str(job_id) / "worker.log"

    def launch(self, job_id: JobId, *, workspace: str,
               env: Mapping[str, str] | None = None) -> WorkerHandle:
        job_id = JobId(job_id)
        log_path = self._log_path(job_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        environment = dict(os.environ)
        environment.update(self._env)
        environment.update(env or {})
        # So a worker started from a checkout rather than an installed package
        # can import it.  Prepended, not replaced: a caller's PYTHONPATH is
        # theirs and overwriting it breaks the case where the trainer lives
        # somewhere else.
        repo_root = str(Path(__file__).resolve().parents[2])
        existing = environment.get("PYTHONPATH", "")
        if repo_root not in existing.split(os.pathsep):
            environment["PYTHONPATH"] = (
                repo_root + (os.pathsep + existing if existing else ""))

        argv = [self.python, "-u", "-m", self.module,
                "--job-id", str(job_id), "--root", str(self.root),
                "--workspace", workspace, *self.extra_args]

        started = time.time()
        # Append, so a relaunch after a crash does not erase the log that says
        # why it crashed.
        with open(log_path, "ab") as log:
            log.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started))} "
                      f"{' '.join(argv)}\n".encode())
            log.flush()
            proc = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=environment, cwd=self.cwd,
                start_new_session=True,   # own session: survives the terminal,
            )                             # and killable as a group
        return WorkerHandle(job_id=job_id, kind=self.kind, pid=proc.pid,
                            log_path=str(log_path), started_at=started,
                            detail={"argv": argv})

    def is_alive(self, handle: WorkerHandle) -> WorkerStatus:
        if handle.pid is None:
            return WorkerStatus.UNKNOWN
        # Reap first, if this process is the parent.  A worker that has exited
        # and not been waited for is a zombie, and a zombie answers ``kill(pid,
        # 0)`` exactly as a running process does -- so without this, a worker
        # that was successfully killed reads as ALIVE for as long as the
        # launching process lives.  Found by the adapter test, which kills a
        # worker and then asks.
        self._reap(handle.pid)
        try:
            os.kill(handle.pid, 0)
        except ProcessLookupError:
            return WorkerStatus.GONE
        except PermissionError:
            # It exists and belongs to someone else.  A pid can be recycled, so
            # this is ALIVE only in the sense that something is there -- and
            # saying UNKNOWN would lose the one fact that is certain.
            return WorkerStatus.ALIVE
        except OSError:
            return WorkerStatus.UNKNOWN
        return WorkerStatus.GONE if self._is_zombie(handle.pid) else WorkerStatus.ALIVE

    @staticmethod
    def _reap(pid: int) -> None:
        """Wait for the worker without blocking, if it is this process's child.

        ``ECHILD`` means it is not ours -- the normal case when the application
        that is asking is not the one that launched -- and that is not an error.
        """
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            return

    @staticmethod
    def _is_zombie(pid: int) -> bool:
        """True where ``/proc`` says the process has exited and not been reaped.

        Linux-specific and deliberately fail-open: on a platform without
        ``/proc`` this returns False and the answer falls back to what
        ``kill(pid, 0)`` said, which is the behaviour it had before.  The state
        is field 3 of ``/proc/<pid>/stat``, read from after the last ``)``
        because the command name in field 2 can itself contain spaces and
        parentheses.
        """
        try:
            stat = open(f"/proc/{pid}/stat", "r", encoding="utf-8",
                        errors="replace").read()
        except OSError:
            return False
        try:
            return stat[stat.rindex(")") + 2:].split()[0] == "Z"
        except (ValueError, IndexError):
            return False

    def terminate(self, handle: WorkerHandle, *, grace_seconds: float = 30.0) -> bool:
        """SIGTERM the worker's process group, then SIGKILL it.

        The group, because the worker's own evaluation pool is its children and
        a signal to the parent alone leaves them running -- measured in this
        project at two hours and a gigabyte of resident memory.

        The worker treats SIGTERM as "the container is going away": it stops at
        its next boundary and checkpoints.  So the grace period is not
        politeness, it is the window in which the run saves its work.
        """
        if handle.pid is None:
            return False
        if self.is_alive(handle) is WorkerStatus.GONE:
            return True

        self._signal_group(handle.pid, signal.SIGTERM)
        deadline = time.monotonic() + max(0.0, grace_seconds)
        while time.monotonic() < deadline:
            if self.is_alive(handle) is WorkerStatus.GONE:
                return True
            time.sleep(0.05)

        self._signal_group(handle.pid, signal.SIGKILL)
        # SIGKILL is not instantaneous from the signaller's point of view: the
        # process has to be reaped.  A short poll rather than one check, so the
        # answer is not "still there" for a process that is already dying.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.is_alive(handle) is WorkerStatus.GONE:
                return True
            time.sleep(0.05)
        return self.is_alive(handle) is WorkerStatus.GONE

    @staticmethod
    def _signal_group(pid: int, sig: int) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except ProcessLookupError:
            return
        except OSError:
            # No process group -- a platform without them, or a pid that is
            # already gone.  Fall back to the pid, which is better than nothing
            # and is exactly the case where there are no children to orphan.
            try:
                os.kill(pid, sig)
            except OSError:
                return
