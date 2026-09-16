"""The ``Clock`` port.

Small, and it earns its place twice over.  A domain object that reads
``time.time()`` cannot be tested for anything involving duration without the
test actually waiting, and an application that reads it directly produces
records whose timestamps are whatever the machine believed -- including a
container whose clock was an hour off, which is a real thing that happens and
which makes two runs' events uninterleavable.

So "now" enters the hexagon in exactly one place.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:
        """Unix time, seconds.  Wall clock: it is what goes in records."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin, never going backwards.

        Separate from ``now`` because durations measured with a wall clock go
        negative when NTP steps it, and an elapsed time of −0.4 s in a record
        is the kind of thing that gets explained away rather than investigated.
        """


class SystemClock:
    """The real one.  The only place this package calls ``time``."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()
