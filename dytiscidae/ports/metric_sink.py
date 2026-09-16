"""The ``MetricSink`` port."""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from ..domain.metrics import MetricPoint


@runtime_checkable
class MetricSink(Protocol):
    """Where numbers go.  One implementation writes JSONL; another does nothing.

    ``emit`` takes an iterable rather than one point, because a step produces a
    dozen at once and a sink that batches them into one write is the difference
    between one ``fsync`` per generation and twelve.
    """

    def emit(self, points: Iterable[MetricPoint]) -> None:
        """Record metrics.  Must not raise on a metric it does not understand.

        Must not raise at all, in practice: a run of 21 hours must not die
        because a metrics backend went away.  An implementation that cannot
        write drops the points and says so once, rather than on every step.
        """

    def flush(self) -> None:
        """Make everything emitted so far durable."""

    def close(self) -> None:
        """Flush and release.  Idempotent."""
