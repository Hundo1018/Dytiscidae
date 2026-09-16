"""Datasets, meaning the corpus a run starts from.

This project trains no supervised model on labelled examples, so "dataset" has
to be defined before it is used or it will be read as something this codebase
does not have.  What a design search consumes is a *seed corpus*: the body
plans, the hand-built reference genome, and the elites of an earlier run that a
new one is started from.  That corpus has every property the port exists for --
it is named, it is versioned, it decides what the run can reach, and swapping
it silently makes two runs incomparable -- so it is modelled as a dataset and
the difference is written down here rather than discovered later.

A ``Dataset`` is metadata.  The items live behind ``DatasetRepository.open``,
which yields them one at a time, because a corpus of elites from a finished run
is tens of megabytes and nothing needs all of it resident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class DatasetKind(str, Enum):
    """What the items in a corpus are."""

    #: Genomes, as plain nested dicts.  The common case.
    GENOMES = "genomes"
    #: Whole elites: genome plus the policy weights and mobility basis that
    #: earned its score.  Distinct from GENOMES because a genome without its
    #: control law is a different experiment -- see CLAUDE.md on the showcase.
    ELITES = "elites"
    #: Recorded transitions for an off-policy or distillation trainer.
    TRAJECTORIES = "trajectories"
    #: Anything else, described by ``Dataset.description``.
    OTHER = "other"


@dataclass(frozen=True)
class DatasetRef:
    """A name and a version, and nothing that ties it to a machine.

    The version is mandatory and is a plain string rather than an integer, so
    "arch38-final" and "v3" are both expressible.  It is mandatory because a
    corpus referenced by name alone is the thing that makes two runs silently
    incomparable, which is the failure this whole port is for.
    """

    name: str
    version: str = "v1"

    def __post_init__(self) -> None:
        for field_name in ("name", "version"):
            v = str(getattr(self, field_name)).strip()
            if not v:
                raise ValueError(f"DatasetRef.{field_name} must be non-empty")
            object.__setattr__(self, field_name, v)

    def as_dict(self) -> dict:
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, d: Mapping) -> "DatasetRef":
        return cls(name=d["name"], version=d.get("version", "v1"))

    def __str__(self) -> str:
        return f"{self.name}:{self.version}"


@dataclass(frozen=True)
class Dataset:
    """What a repository knows about a corpus without opening it."""

    ref: DatasetRef
    kind: DatasetKind = DatasetKind.GENOMES
    #: Number of items, or None where the repository cannot say without a scan.
    #: None and 0 are different answers and are kept different: 0 means an empty
    #: corpus was measured, None means nobody counted.  This project has been
    #: bitten once by a metric that used one value for both.
    item_count: int | None = None
    #: Content hash of the items, where the repository computes one.  Two
    #: corpora with the same digest are the same corpus.
    digest: str | None = None
    #: Where it came from, as an opaque locator the repository understands.
    uri: str | None = None
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "ref": self.ref.as_dict(),
            "kind": self.kind.value,
            "item_count": self.item_count,
            "digest": self.digest,
            "uri": self.uri,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Dataset":
        return cls(
            ref=DatasetRef.from_dict(d["ref"]),
            kind=DatasetKind(d.get("kind", "genomes")),
            item_count=d.get("item_count"),
            digest=d.get("digest"),
            uri=d.get("uri"),
            description=d.get("description", ""),
            metadata=dict(d.get("metadata") or {}),
        )
