"""``DatasetRepository`` over a directory of JSONL corpora.

    <root>/datasets/<name>/<version>/dataset.json    metadata, digest included
    <root>/datasets/<name>/<version>/items.jsonl     the items

JSONL again, and for a reason specific to this port: a seed corpus is the elites
of a finished run, which is thousands of genomes each carrying its policy
weights, and the consumer reads it once in order.  A format that has to be
parsed in full before the first item can be handed over would make "seed from
the last run" cost its whole memory footprint to take sixteen designs.

``put`` refuses to overwrite an existing ref.  A corpus that can change under a
fixed ``name:version`` makes every plan naming it irreproducible while still
hashing to the same digest -- the exact failure that having a version in the ref
is meant to prevent, defeated by the repository instead of by the caller.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ...domain.dataset import Dataset, DatasetKind, DatasetRef
from ...domain.errors import DatasetNotFound
from ._io import atomic_write_json, read_json

META = "dataset.json"
ITEMS = "items.jsonl"


class FileDatasetRepository:
    """The filesystem implementation of ``ports.DatasetRepository``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def dir(self) -> Path:
        return self.root / "datasets"

    def _dir(self, ref: DatasetRef) -> Path:
        # The ref's components are path segments, so they are checked rather
        # than trusted: a version of "../../etc" would otherwise write outside
        # the repository.
        for part in (ref.name, ref.version):
            if "/" in part or "\\" in part or part in (".", ".."):
                raise ValueError(f"dataset ref component {part!r} is not a name")
        return self.dir / ref.name / ref.version

    # -- ports.DatasetRepository ------------------------------------------

    def get(self, ref: DatasetRef) -> Dataset:
        d = read_json(self._dir(ref) / META)
        if d is None:
            raise DatasetNotFound(f"no dataset {ref} under {self.dir}")
        return Dataset.from_dict(d)

    def open(self, ref: DatasetRef) -> Iterator[Mapping[str, Any]]:
        path = self._dir(ref) / ITEMS
        if not path.exists():
            raise DatasetNotFound(f"no items for dataset {ref} at {path}")
        return self._stream(path)

    @staticmethod
    def _stream(path: Path) -> Iterator[Mapping[str, Any]]:
        with open(path, "r", encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    # Raised, not skipped -- unlike the event and metric streams.
                    # A corpus is an input to a run, and silently dropping an
                    # item from it changes what the run could reach while
                    # reporting nothing.
                    raise ValueError(
                        f"{path}:{n} is not valid JSON ({exc}); a seed corpus "
                        f"with a dropped item is a different corpus") from exc

    def list(self, *, kind: DatasetKind | None = None) -> Sequence[Dataset]:
        if not self.dir.is_dir():
            return []
        out = []
        for name_dir in sorted(self.dir.iterdir()):
            if not name_dir.is_dir():
                continue
            for version_dir in sorted(name_dir.iterdir()):
                d = read_json(version_dir / META)
                if d is None:
                    continue
                try:
                    dataset = Dataset.from_dict(d)
                except Exception:                                 # noqa: BLE001
                    continue
                if kind is not None and dataset.kind is not DatasetKind(kind):
                    continue
                out.append(dataset)
        return out

    def put(self, ref: DatasetRef, items: Iterable[Mapping[str, Any]], *,
            kind: DatasetKind = DatasetKind.GENOMES,
            description: str = "",
            metadata: Mapping[str, Any] | None = None) -> Dataset:
        target = self._dir(ref)
        if (target / META).exists():
            raise FileExistsError(
                f"dataset {ref} already exists. A corpus that changes under a "
                f"fixed ref makes every plan naming it irreproducible; write a "
                f"new version instead.")
        target.mkdir(parents=True, exist_ok=True)

        # Digest and count as the items stream past, so a corpus is hashed
        # exactly once and never has to be held in memory to be described.
        h = hashlib.sha256()
        count = 0
        tmp = target / (ITEMS + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for item in items:
                line = json.dumps(item, sort_keys=True, separators=(",", ":"))
                fh.write(line + "\n")
                h.update(line.encode("utf-8"))
                h.update(b"\n")
                count += 1
            fh.flush()
            import os
            os.fsync(fh.fileno())
        tmp.replace(target / ITEMS)

        dataset = Dataset(ref=ref, kind=DatasetKind(kind), item_count=count,
                          digest=h.hexdigest()[:32], uri=str(target),
                          description=description, metadata=dict(metadata or {}))
        atomic_write_json(target / META, dataset.as_dict())
        return dataset
