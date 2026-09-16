"""Checkpoints, described here and stored elsewhere.

The domain knows a checkpoint's *identity and provenance*.  It does not know
that the arrays are in an ``.npz``, because the store might hold them in object
storage, and it must not know that some of them are ``torch`` tensors, because
then the domain would depend on torch.

So the payload the store round-trips is a mapping of name to ``bytes``.  That is
the widest interface that stays dependency-free, and it is enough: the search
adapter puts ``checkpoint.npz`` and ``checkpoint.json`` in it, a supervised
adapter would put one ``state_dict`` blob in it, and the store treats both the
same.

``ops/checkpoint.py`` already established what a checkpoint here has to contain
and why -- the network, Adam's moments, the RNG state, the mobility bases and
the commit that wrote it, because each of those was at some point missing and
each absence cost a measurement.  ``CheckpointRecord.provenance`` is where that
lives at the domain level, and ``CheckpointKind`` is why the file is not just
"the latest one".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .ids import CheckpointId, ExperimentId, JobId

#: What a store round-trips.  Named blobs; the trainer that wrote them is the
#: only thing that knows what they mean.
CheckpointPayload = Mapping[str, bytes]


class CheckpointKind(str, Enum):
    """What a checkpoint is for, which decides whether it can be deleted.

    Retention without this is either "keep everything", which fills the disk in
    a 21-hour run that checkpoints every 10 generations, or "keep the last N",
    which deletes the best one.
    """

    #: Written on the periodic cadence.  Prunable.
    PERIODIC = "periodic"
    #: The last one before a clean stop.  This is what a resume reads.
    LATEST = "latest"
    #: Best on the metric the run is judged by.  Never pruned automatically.
    BEST = "best"
    #: Written because the run is about to end -- a signal, a budget, a crash
    #: handler.  Never pruned automatically: it is the one that exists because
    #: something went wrong, which is when it is most wanted.
    FINAL = "final"

    def __str__(self) -> str:
        return self.value


#: Kinds that automatic retention must never delete.
PROTECTED_KINDS = frozenset({CheckpointKind.BEST, CheckpointKind.FINAL})


@dataclass(frozen=True)
class CheckpointRecord:
    """Everything about a checkpoint except its bytes."""

    checkpoint_id: CheckpointId
    job_id: JobId
    experiment_id: ExperimentId
    #: The step it was written at, in the trainer's unit.
    step: int
    kind: CheckpointKind = CheckpointKind.PERIODIC
    created_at: float = 0.0
    size_bytes: int = 0
    #: Content hash over the payload, so "is this the checkpoint I read before"
    #: is answerable without reading it again.
    digest: str = ""
    #: The metrics as of this step.  Copied in rather than referenced, because
    #: the point of a checkpoint is to survive the run that wrote it.
    metrics: Mapping[str, float] = field(default_factory=dict)
    #: git sha, library versions, the plan digest, the schema version of the
    #: payload.  A checkpoint that cannot be matched to the code that wrote it
    #: cannot be trusted to mean what it meant.
    provenance: Mapping[str, Any] = field(default_factory=dict)
    #: The payload's key names, so a reader can tell what is inside without
    #: fetching it.
    payload_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_id, CheckpointId):
            object.__setattr__(self, "checkpoint_id", CheckpointId(self.checkpoint_id))
        if not isinstance(self.job_id, JobId):
            object.__setattr__(self, "job_id", JobId(self.job_id))
        if not isinstance(self.experiment_id, ExperimentId):
            object.__setattr__(self, "experiment_id", ExperimentId(self.experiment_id))
        if not isinstance(self.kind, CheckpointKind):
            object.__setattr__(self, "kind", CheckpointKind(self.kind))
        if self.step < 0:
            raise ValueError(f"checkpoint step must be non-negative, got {self.step}")
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "payload_keys", tuple(self.payload_keys))

    @property
    def protected(self) -> bool:
        return self.kind in PROTECTED_KINDS

    def as_dict(self) -> dict:
        return {
            "checkpoint_id": str(self.checkpoint_id),
            "job_id": str(self.job_id),
            "experiment_id": str(self.experiment_id),
            "step": self.step,
            "kind": self.kind.value,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "digest": self.digest,
            "metrics": dict(self.metrics),
            "provenance": dict(self.provenance),
            "payload_keys": list(self.payload_keys),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "CheckpointRecord":
        return cls(
            checkpoint_id=CheckpointId(d["checkpoint_id"]),
            job_id=JobId(d["job_id"]),
            experiment_id=ExperimentId(d["experiment_id"]),
            step=int(d["step"]),
            kind=CheckpointKind(d.get("kind", "periodic")),
            created_at=float(d.get("created_at", 0.0)),
            size_bytes=int(d.get("size_bytes", 0)),
            digest=d.get("digest", ""),
            metrics=dict(d.get("metrics") or {}),
            provenance=dict(d.get("provenance") or {}),
            payload_keys=tuple(d.get("payload_keys") or ()),
        )

    def __str__(self) -> str:
        return f"{self.checkpoint_id} step={self.step} [{self.kind}]"


def digest_payload(payload: CheckpointPayload) -> str:
    """A stable hash over named blobs.

    Names are hashed alongside the bytes, and their lengths alongside them, so
    that ``{"ab": b"", "": b"cd"}`` and ``{"a": b"b", "cd": b""}`` do not
    collide -- the concatenation without lengths is the classic way to get a
    digest that agrees when the contents do not.
    """
    import hashlib

    h = hashlib.sha256()
    for name in sorted(payload):
        blob = payload[name]
        h.update(len(name).to_bytes(8, "big"))
        h.update(name.encode("utf-8"))
        h.update(len(blob).to_bytes(8, "big"))
        h.update(blob)
    return h.hexdigest()
