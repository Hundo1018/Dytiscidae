"""The ``DatasetRepository`` port."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

from ..domain.dataset import Dataset, DatasetKind, DatasetRef


@runtime_checkable
class DatasetRepository(Protocol):
    """Resolves a ``DatasetRef`` to a corpus, and streams its items.

    ``open`` yields, it does not return a list.  A corpus here is the elites of
    a finished run -- thousands of genomes with their policy weights -- and the
    trainer consumes them once, in order, at the start.  Materialising that to
    take the first sixteen is a cost paid for nothing.
    """

    def get(self, ref: DatasetRef) -> Dataset:
        """Metadata for a corpus.  Raises ``DatasetNotFound`` if unknown."""

    def open(self, ref: DatasetRef) -> Iterator[Mapping[str, Any]]:
        """Stream the items.  Raises ``DatasetNotFound`` if unknown."""

    def list(self, *, kind: DatasetKind | None = None) -> Sequence[Dataset]:
        """Every corpus this repository knows about."""

    def put(self, ref: DatasetRef, items: Iterable[Mapping[str, Any]], *,
            kind: DatasetKind = DatasetKind.GENOMES,
            description: str = "",
            metadata: Mapping[str, Any] | None = None) -> Dataset:
        """Store a corpus and return its metadata, digest included.

        Writing a ref that already exists is an error, not an overwrite.  A
        corpus that can change under a fixed ref makes every plan that names it
        irreproducible while still hashing to the same digest -- the failure
        mode that having a version in the ref is meant to prevent, defeated by
        the repository rather than by the caller.
        """
