"""``CheckpointStore`` over a directory tree.

Layout::

    <root>/jobs/<job_id>/checkpoints/<checkpoint_id>/record.json
    <root>/jobs/<job_id>/checkpoints/<checkpoint_id>/<percent-encoded key>

One directory per checkpoint, with the payload blobs beside the record.  The
record is written **last**, after every blob is on disk and fsynced, and it is
written atomically -- so the presence of ``record.json`` is the commit, and a
checkpoint interrupted halfway has no record and is invisible to ``latest``.
That ordering is the whole durability argument: a resume can only ever see a
checkpoint whose bytes are all there.

``load`` verifies the digest and raises on a mismatch.  Returning a corrupt
payload would let a run resume from it and be quietly wrong for the next twenty
hours, which is more expensive than any crash.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...domain.checkpoint import (
    CheckpointKind,
    CheckpointPayload,
    CheckpointRecord,
    digest_payload,
)
from ...domain.errors import CheckpointNotFound
from ...domain.ids import CheckpointId, ExperimentId, JobId, new_id
from ._io import atomic_write_json, read_json, safe_name, unsafe_name

RECORD = "record.json"


class CorruptCheckpoint(CheckpointNotFound):
    """The bytes do not hash to what the record says they should.

    A subclass of ``CheckpointNotFound`` so that a caller which only wants to
    fall back to a fresh start can catch one exception -- and a distinct class
    so that a caller which wants to say *why* it is starting fresh can.
    """


class FileCheckpointStore:
    """The filesystem implementation of ``ports.CheckpointStore``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- layout -----------------------------------------------------------

    def _dir(self, job_id: JobId) -> Path:
        return self.root / "jobs" / str(job_id) / "checkpoints"

    def _find(self, checkpoint_id: CheckpointId) -> Path | None:
        """Locate a checkpoint without being told its job.

        A scan over job directories.  Acceptable because the number of jobs is
        in the hundreds and the id itself is sortable, and because the call
        that matters -- a resume -- always knows its job and goes through
        ``latest``.  An index would be a second thing to keep consistent with
        the directory tree, and the directory tree is the truth.
        """
        base = self.root / "jobs"
        if not base.is_dir():
            return None
        for job_dir in base.iterdir():
            candidate = job_dir / "checkpoints" / str(checkpoint_id)
            if (candidate / RECORD).exists():
                return candidate
        return None

    # -- ports.CheckpointStore --------------------------------------------

    def save(self, *, job_id: JobId, experiment_id: ExperimentId, step: int,
             payload: CheckpointPayload,
             kind: CheckpointKind = CheckpointKind.PERIODIC,
             metrics: Mapping[str, float] | None = None,
             provenance: Mapping[str, Any] | None = None,
             created_at: float = 0.0) -> CheckpointRecord:
        job_id, experiment_id = JobId(job_id), ExperimentId(experiment_id)
        kind = CheckpointKind(kind)
        # The step is in the id, so a directory listing is in step order and
        # ``latest`` needs no metadata read to find its candidate.  The random
        # tail keeps two workers writing the same step from colliding, which is
        # what happens when a launch is retried before the first worker notices.
        checkpoint_id = CheckpointId(new_id(f"ck{step:07d}", now=created_at or None))

        target = self._dir(job_id) / str(checkpoint_id)
        staging = target.with_name(target.name + ".partial")
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        total = 0
        for key, blob in payload.items():
            if not isinstance(blob, (bytes, bytearray, memoryview)):
                shutil.rmtree(staging, ignore_errors=True)
                raise TypeError(
                    f"checkpoint payload {key!r} is {type(blob).__name__}, not "
                    f"bytes. The store round-trips bytes so that it never has "
                    f"to know what a tensor is.")
            data = bytes(blob)
            with open(staging / safe_name(key), "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            total += len(data)

        record = CheckpointRecord(
            checkpoint_id=checkpoint_id, job_id=job_id, experiment_id=experiment_id,
            step=int(step), kind=kind, created_at=created_at,
            size_bytes=total, digest=digest_payload(payload),
            metrics=dict(metrics or {}), provenance=dict(provenance or {}),
            payload_keys=tuple(sorted(payload)),
        )
        # Blobs are all durable; the rename publishes them, and the record --
        # written into the published directory, atomically -- is the commit.
        os.replace(staging, target)
        atomic_write_json(target / RECORD, record.as_dict())
        return record

    def get(self, checkpoint_id: CheckpointId) -> CheckpointRecord:
        checkpoint_id = CheckpointId(checkpoint_id)
        found = self._find(checkpoint_id)
        if found is None:
            raise CheckpointNotFound(f"no checkpoint {checkpoint_id} under {self.root}")
        d = read_json(found / RECORD)
        if d is None:
            raise CheckpointNotFound(f"checkpoint {checkpoint_id} has no record")
        return CheckpointRecord.from_dict(d)

    def load(self, checkpoint_id: CheckpointId
             ) -> tuple[CheckpointRecord, CheckpointPayload]:
        record = self.get(checkpoint_id)
        found = self._find(record.checkpoint_id)
        assert found is not None                      # get() just resolved it
        payload: dict[str, bytes] = {}
        for entry in sorted(found.iterdir()):
            if entry.name == RECORD or not entry.is_file():
                continue
            payload[unsafe_name(entry.name)] = entry.read_bytes()

        missing = set(record.payload_keys) - set(payload)
        if missing:
            raise CorruptCheckpoint(
                f"checkpoint {checkpoint_id} is missing {sorted(missing)}; "
                f"the record lists {list(record.payload_keys)}")
        actual = digest_payload(payload)
        if record.digest and actual != record.digest:
            raise CorruptCheckpoint(
                f"checkpoint {checkpoint_id} digest mismatch: stored "
                f"{record.digest}, read {actual}. Refusing to resume from it -- "
                f"a run continued from a corrupt checkpoint is wrong without "
                f"being obviously wrong.")
        return record, payload

    def latest(self, job_id: JobId, *, kind: CheckpointKind | None = None
               ) -> CheckpointRecord | None:
        records = self.list(JobId(job_id))
        if kind is not None:
            kind = CheckpointKind(kind)
            records = [r for r in records if r.kind is kind]
        if not records:
            return None
        # Highest step wins, and among equal steps the most recently created --
        # not the most recently written overall, so a step-50 checkpoint from an
        # abandoned restart never shadows a step-600 one.
        return max(records, key=lambda r: (r.step, r.created_at))

    def list(self, job_id: JobId | None = None, *,
             experiment_id: ExperimentId | None = None) -> Sequence[CheckpointRecord]:
        dirs: list[Path] = []
        if job_id is not None:
            dirs = [self._dir(JobId(job_id))]
        else:
            base = self.root / "jobs"
            if base.is_dir():
                dirs = [d / "checkpoints" for d in base.iterdir()]

        out: list[CheckpointRecord] = []
        for d in dirs:
            if not d.is_dir():
                continue
            for entry in sorted(d.iterdir()):
                if entry.name.endswith(".partial") or not entry.is_dir():
                    continue
                payload = read_json(entry / RECORD)
                if payload is None:
                    continue                        # interrupted mid-write
                try:
                    record = CheckpointRecord.from_dict(payload)
                except Exception:                                 # noqa: BLE001
                    continue
                if experiment_id is not None and record.experiment_id != experiment_id:
                    continue
                out.append(record)
        out.sort(key=lambda r: (r.step, r.created_at))
        return out

    def delete(self, checkpoint_id: CheckpointId) -> None:
        found = self._find(CheckpointId(checkpoint_id))
        if found is not None:
            shutil.rmtree(found, ignore_errors=True)

    def prune(self, job_id: JobId, *, keep_last: int = 3) -> Sequence[CheckpointId]:
        if keep_last < 0:
            raise ValueError(f"keep_last must be non-negative, got {keep_last}")
        periodic = [r for r in self.list(JobId(job_id))
                    if r.kind is CheckpointKind.PERIODIC]
        doomed = periodic[:-keep_last] if keep_last else list(periodic)
        removed = []
        for record in doomed:
            self.delete(record.checkpoint_id)
            removed.append(record.checkpoint_id)
        return removed
