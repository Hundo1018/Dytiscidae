"""Shared filesystem mechanics for the storage adapters.

Two things, both of which every store below needs and neither of which belongs
in a store:

**Atomic replacement.**  Write to a temporary file in the same directory, fsync
it, rename it over the target.  ``os.replace`` is atomic within a filesystem, so
a reader never sees a partial file -- which matters most for the case this
project already documents: a checkpoint that looks complete and is not, resumed
from, producing a run that is quietly wrong.

**A lock.**  ``JobStore.update`` is called by the application and by the worker,
in different processes, on the same job.  Without a lock, a heartbeat written by
the worker and a pause written by the application interleave as read-read-
write-write and one of them is lost -- and the one that is lost is whichever
was slower, which is not a property anyone can reason about.

``fcntl.flock`` where it exists, an ``O_EXCL`` lock file where it does not.
flock is released by the kernel when the process dies, which is the property
that matters here: a worker killed mid-update must not leave a job permanently
unwritable.  The fallback cannot offer that, so it carries a staleness timeout
and says so.
"""

from __future__ import annotations

import errno
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

try:                                                              # POSIX
    import fcntl
    _HAVE_FLOCK = True
except ImportError:                                               # pragma: no cover
    fcntl = None                                                  # type: ignore
    _HAVE_FLOCK = False


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data``, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any, *, indent: int | None = 1) -> None:
    atomic_write_bytes(
        path, (json.dumps(obj, indent=indent, sort_keys=False,
                          default=_fallback) + "\n").encode("utf-8"))


def _fallback(o):
    # Records reaching here hold scalars, lists and dicts.  Anything else is a
    # bug in the caller, and turning it into its repr would hide that -- so the
    # only conversions are the two that are unambiguous.
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "as_dict"):
        return o.as_dict()
    raise TypeError(f"{type(o).__name__} is not JSON-writable: {o!r}")


def read_json(path: Path) -> Any | None:
    """Parse ``path``, or None if it is absent, empty or truncated.

    Truncated counts as absent on purpose, for the metadata files where a
    missing record and an unreadable one lead to the same recovery.  It does
    **not** apply to checkpoints: those verify a digest and raise, because a
    checkpoint silently treated as absent restarts a run that could have been
    resumed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def append_jsonl(path: Path, obj: Any) -> None:
    """Append one JSON object, line-buffered, opened per call.

    Per call rather than a cached handle: the alternative is a handle held for
    the twenty-one hours of a run, which has to be flushed on every write
    anyway to be useful to ``tail -f``, and which leaks when the writer is a
    short-lived command.  A single line under ``PIPE_BUF`` is written
    atomically by the kernel on an ``O_APPEND`` handle, so concurrent writers
    do not interleave.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, default=_fallback) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield each parseable line.  Skips a truncated final line.

    Skipping rather than raising, because the common reason for one is that the
    file is being read while it is being written -- which is the whole point of
    the format.
    """
    try:
        fh = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


class FileLock:
    """An advisory lock on a path, held for the duration of a ``with`` block."""

    def __init__(self, path: Path, *, timeout: float = 30.0,
                 stale_after: float = 300.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.stale_after = stale_after
        self._fh = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        if _HAVE_FLOCK:
            self._fh = open(self.path, "a+b")
            while True:
                try:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    if time.monotonic() >= deadline:
                        self._fh.close()
                        self._fh = None
                        raise TimeoutError(
                            f"could not lock {self.path} within {self.timeout}s")
                    time.sleep(0.01)
        # No flock: an exclusive create, with a staleness escape so that a
        # worker killed mid-update does not wedge the job forever.
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                self._fh = fd
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.stale_after:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not lock {self.path} within {self.timeout}s")
                time.sleep(0.01)

    def __exit__(self, *_exc) -> None:
        if self._fh is None:
            return
        if _HAVE_FLOCK:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
        else:
            os.close(self._fh)
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self._fh = None


def safe_name(key: str) -> str:
    """A payload key as a file name, reversibly.

    Percent-encoding, so a key containing ``/`` cannot escape its directory and
    two keys differing only in a separator cannot collide.  Reversible because
    ``load`` has to hand back the keys the writer used.
    """
    from urllib.parse import quote
    return quote(key, safe="")


def unsafe_name(name: str) -> str:
    from urllib.parse import unquote
    return unquote(name)
