"""Is the built GPU kernel the one its source says it should be?

`mojo/build/*.so` is a compiled mirror of `dytiscidae/physics/fluid.py`, and
nothing compared the two. Measured 2026-09-20: the kernel in `mojo/build` was
built 2026-08-04 and `mojo/src` was changed on 2026-09-15 by F-05, which
replaced the stall blend -- a logistic in the old binary, a compactly supported
smoothstep in the source and in the numpy solver. So the search scored with one
physics and every verification, probe and film used another, for five weeks, and
the test that would have caught it could not run on a machine whose driver was
broken. air and land measurements disagreed by up to 2.0 between the paths;
after a rebuild they agree to 0.000002.

A manifest written at build time records the sha256 of every Mojo source, and
`freshness` compares it with what is on disk now. Stdlib only, so the build step
can call it from inside the pixi environment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "mojo" / "src"
#: ``DYTISCIDAE_KERNEL_DIR`` points every process at a staged build instead:
#: a live run has ``mojo/build/*.so`` mapped, so a new kernel is built and
#: verified elsewhere and installed only once nothing is using the old one.
BUILD = Path(os.environ.get("DYTISCIDAE_KERNEL_DIR") or ROOT / "mojo" / "build")
MANIFEST = BUILD / "BUILD_MANIFEST.json"


def source_hashes(src: Path | None = None) -> dict:
    src = src or SRC
    out = {}
    for p in sorted(src.glob("*.mojo")):
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def write_manifest(src: Path | None = None, build: Path | None = None) -> Path:
    """Record what was compiled.  Called by ``pixi run build-all``."""
    build = build or BUILD
    build.mkdir(parents=True, exist_ok=True)
    path = build / "BUILD_MANIFEST.json"
    path.write_text(json.dumps({"sources": source_hashes(src)}, indent=2, sort_keys=True))
    return path


def freshness(src: Path | None = None, build: Path | None = None) -> tuple:
    """``(state, reason)`` where state is "fresh", "stale" or "unverified".

    "unverified" is a kernel built before manifests existed, or a tree with no
    build at all: it is not a failure, and it is not a pass either, so it says
    so rather than picking one.
    """
    build = build or BUILD
    path = build / "BUILD_MANIFEST.json"
    try:
        recorded = json.loads(path.read_text()).get("sources") or {}
    except Exception:                                             # noqa: BLE001
        return ("unverified", f"no {path.name} in {build}: the kernel cannot be "
                              f"matched to its source. Build with "
                              f"`cd mojo && pixi run build-all`.")
    now = source_hashes(src)
    changed = sorted(k for k in set(now) | set(recorded)
                     if now.get(k) != recorded.get(k))
    if changed:
        return ("stale", "the built GPU kernel is older than its source: "
                         + ", ".join(changed)
                         + ". Rebuild with `cd mojo && pixi run build-all`.")
    return ("fresh", "")
