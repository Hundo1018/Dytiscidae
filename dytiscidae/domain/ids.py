"""Identifiers.

Distinct types rather than bare ``str``, because the three of them are passed
through the same call signatures constantly and a transposed pair is otherwise
a silent lookup miss.  ``JobId("x") == ExperimentId("x")`` is False, and the
store that is handed the wrong one raises rather than returning None.
"""

from __future__ import annotations

import re
import time
import uuid

#: Lowercase hex, dashes and underscores.  Identifiers end up in file names and
#: in SQL, so they are constrained at construction rather than escaped at every
#: use.
_ALLOWED = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class Identifier(str):
    """A validated string identifier that does not compare equal across types."""

    __slots__ = ()

    def __new__(cls, value: str):
        if isinstance(value, Identifier) and type(value) is not cls:
            raise TypeError(
                f"{type(value).__name__}({str(value)!r}) is not a {cls.__name__}"
            )
        text = str(value)
        if not _ALLOWED.match(text):
            raise ValueError(
                f"{cls.__name__} must match {_ALLOWED.pattern!r}, got {text!r}"
            )
        return super().__new__(cls, text)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Identifier):
            return type(self) is type(other) and str.__eq__(self, other) is True
        if isinstance(other, str):
            # A plain ``str`` compares by value, so callers reading identifiers
            # out of JSON do not have to wrap them first.  Two *different*
            # Identifier subclasses never compare equal, which is the case this
            # class exists for.
            return str.__eq__(self, other) is True
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    def __hash__(self) -> int:
        return str.__hash__(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"


class JobId(Identifier):
    """One execution attempt of a plan."""

    __slots__ = ()


class ExperimentId(Identifier):
    """A named question, under which jobs are run."""

    __slots__ = ()


class CheckpointId(Identifier):
    """One stored checkpoint."""

    __slots__ = ()


def new_id(prefix: str, *, now: float | None = None, entropy: int = 6) -> str:
    """A sortable identifier: ``prefix-YYYYmmddTHHMMSS-xxxxxx``.

    Sortable because the common operation is "the latest job of this
    experiment", and a lexical sort answering that without a clock lookup means
    a directory listing is already in order.  The random tail is there because
    two jobs started in the same second is normal when a launcher fans out.
    """
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(time.time() if now is None else now))
    tail = uuid.uuid4().hex[:max(0, entropy)]
    return f"{prefix}-{stamp}-{tail}" if tail else f"{prefix}-{stamp}"
