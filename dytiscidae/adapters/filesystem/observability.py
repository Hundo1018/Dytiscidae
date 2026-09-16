"""``MetricSink``, ``EventLog`` and ``ControlChannel`` over files.

Three small adapters in one module because they share a layout and a discipline:
one JSON object per line for the streams, one small JSON file for the control
signal, all under the job's own directory::

    <root>/jobs/<job_id>/metrics.jsonl
    <root>/jobs/<job_id>/events.jsonl
    <root>/jobs/<job_id>/control.json

JSONL is chosen for the same reason ``ops/telemetry.py`` chose it: a run
interrupted at any instant still leaves a complete record up to that instant,
and it is readable with ``tail -f`` while the run is going -- which in an
environment where containers are reclaimed without warning is the difference
between having the data and not.

The control file is the one piece of the design that carries a visible cost.  A
pause reaches a running worker at its next checkpoint boundary, which for this
project's search is up to one generation: ~74 s in steady state, ~300 s during
the opening six-island verification burst.  ``ports/control.py`` sets out why a
file beat both a signal and a broker here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ...domain.ids import JobId
from ...domain.job import StopReason
from ...domain.metrics import JobEvent, MetricPoint
from ...ports.control import ControlSignal
from ._io import append_jsonl, atomic_write_json, read_json, read_jsonl


class FileMetricSink:
    """Writes ``MetricPoint``s as JSONL under each job's directory.

    ``sample`` drops all but every Nth point *of the same name*, per name rather
    than globally, so sampling a chatty metric does not also thin a rare one --
    which is the mistake that makes a sampled stream unreadable: two metrics at
    different rates, sampled together, come out interleaved at neither rate.
    """

    def __init__(self, root: str | Path, *, sample: int = 1) -> None:
        self.root = Path(root)
        self.sample = max(1, int(sample))
        self._seen: dict[str, int] = {}
        self._dropped = 0
        self._complained = False

    def _path(self, job_id: JobId) -> Path:
        return self.root / "jobs" / str(job_id) / "metrics.jsonl"

    def emit(self, points: Iterable[MetricPoint]) -> None:
        for point in points:
            if self.sample > 1:
                n = self._seen.get(point.name, 0)
                self._seen[point.name] = n + 1
                if n % self.sample:
                    continue
            if point.job_id is None:
                # A metric with nowhere to go is dropped rather than guessed at.
                # Guessing would put one job's numbers in another's file, which
                # is worse than losing them.
                self._dropped += 1
                continue
            try:
                append_jsonl(self._path(point.job_id), point.as_dict())
            except OSError as exc:
                # Never raise: a run of twenty-one hours must not die because
                # the metrics disk went away.  Complain once.
                self._dropped += 1
                if not self._complained:
                    self._complained = True
                    print(f"  (metrics not being written: {exc})", flush=True)

    def flush(self) -> None:
        # ``append_jsonl`` opens, writes and closes per call, so everything
        # emitted is already with the OS.  Nothing to do, and saying so is
        # better than an empty method that looks like an oversight.
        return None

    def close(self) -> None:
        self.flush()

    @property
    def dropped(self) -> int:
        """Points that were not written.  Read by the tests, and by anyone
        wondering why a chart has gaps."""
        return self._dropped


class FileEventLog:
    """Append-only job history as JSONL."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, job_id: JobId) -> Path:
        return self.root / "jobs" / str(job_id) / "events.jsonl"

    def append(self, event: JobEvent) -> None:
        try:
            append_jsonl(self._path(event.job_id), event.as_dict())
        except OSError as exc:
            print(f"  (event not recorded: {exc})", flush=True)

    def read(self, job_id: JobId, *, limit: int | None = None) -> Sequence[JobEvent]:
        rows = list(read_jsonl(self._path(JobId(job_id))))
        if limit is not None:
            rows = rows[-limit:]            # the last N: the end is the interesting end
        out = []
        for row in rows:
            try:
                out.append(JobEvent.from_dict(row))
            except Exception:                                     # noqa: BLE001
                continue
        return out


class FileControlChannel:
    """A pending instruction as one small JSON file per job.

    The strength rule from ``ports/control.py`` is implemented here: a cancel
    replaces a pending pause, a pause does not replace a pending cancel.  A user
    who asked to stop a run and then asked again, less firmly, has not
    downgraded their own instruction.
    """

    #: Higher wins.  Only stop reasons a caller can request appear here.
    _STRENGTH = {
        StopReason.PAUSE_REQUESTED: 1,
        StopReason.SIGNAL: 1,
        StopReason.BUDGET_SECONDS: 1,
        StopReason.BUDGET_MEMORY: 1,
        StopReason.CANCEL_REQUESTED: 2,
    }

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, job_id: JobId) -> Path:
        return self.root / "jobs" / str(job_id) / "control.json"

    def request(self, job_id: JobId, signal: ControlSignal) -> None:
        job_id = JobId(job_id)
        pending = self.poll(job_id)
        if pending is not None:
            have = self._STRENGTH.get(pending.reason, 0)
            want = self._STRENGTH.get(signal.reason, 0)
            if want < have:
                return
            if want == have and not (signal.urgent and not pending.urgent):
                # Same strength: keep the first, which preserves the original
                # requester and timestamp.  An urgent repeat of the same reason
                # does get through, because that is a real escalation.
                return
        atomic_write_json(self._path(job_id), signal.as_dict())

    def poll(self, job_id: JobId) -> ControlSignal | None:
        d = read_json(self._path(JobId(job_id)))
        if not d:
            return None
        try:
            return ControlSignal.from_dict(d)
        except (KeyError, ValueError):
            # An unreadable control file is treated as no request.  The
            # alternative -- stopping on a file nobody can parse -- turns a
            # typo into a halted twenty-one-hour run.
            return None

    def clear(self, job_id: JobId) -> None:
        self._path(JobId(job_id)).unlink(missing_ok=True)
