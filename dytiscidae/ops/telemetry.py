"""Structured telemetry.

Everything the run decides or discovers is written as one JSON object per line.
That format is chosen so the run is *inspectable while it is running*, by a
human with ``tail -f`` and by the dashboard, and so that a run interrupted at
any point still leaves a complete record up to that instant -- which matters
here because the environments this is designed to run in are frequently
ephemeral.

Three streams, deliberately separate so that the noisy one can be sampled
without losing the others:

    events.jsonl      every evaluation: what was tried, what happened, why
    generations.jsonl one line per generation: archive state, regime, operators
    exploits.jsonl    every candidate caught beating the simulator
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


def _jsonable(o):
    """Coerce numpy, dataclasses and enums into something json can write."""
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [_jsonable(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if is_dataclass(o) and not isinstance(o, type):
        return _jsonable(asdict(o))
    if hasattr(o, "value") and hasattr(o, "name"):  # Enum
        return o.value
    if o is None or isinstance(o, str):
        return o
    return str(o)


class Telemetry:
    """Append-only JSONL writer with a small in-memory tail for the dashboard."""

    def __init__(self, run_dir: str | Path, *, event_sample: int = 1) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.event_sample = max(1, event_sample)
        self._event_count = 0
        self._files: dict[str, object] = {}
        self.recent: list[dict] = []

    def _f(self, stream: str):
        if stream not in self._files:
            self._files[stream] = open(self.dir / f"{stream}.jsonl", "a", buffering=1)
        return self._files[stream]

    def write(self, stream: str, record: dict) -> None:
        rec = _jsonable(record)
        rec["t"] = round(time.time() - self.t0, 3)
        self._f(stream).write(json.dumps(rec) + "\n")

    def event(self, record: dict) -> None:
        """Per-evaluation record.  Sampled, because there are a lot of these."""
        self._event_count += 1
        if self._event_count % self.event_sample == 0:
            self.write("events", record)

    def generation(self, record: dict) -> None:
        self.write("generations", record)
        self.recent.append(_jsonable(record))
        self.recent = self.recent[-400:]

    def exploit(self, record: dict) -> None:
        self.write("exploits", record)

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._files.clear()

    @staticmethod
    def read(run_dir: str | Path, stream: str) -> list[dict]:
        path = Path(run_dir) / f"{stream}.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a partially written last line during a live run
        return out
