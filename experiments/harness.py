"""Shared plumbing for the experiments: config, provenance, results, statistics.

Nothing here decides anything.  It exists so that every `run.py` records the
same provenance -- which commit, which seed, which numpy -- and writes results
in the same shape, because a measurement whose conditions are not recorded is
an anecdote.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 -- provenance must never fail a run
        return "unknown"


def load_config(path: Path) -> dict:
    """Read an experiment's config.json.  Missing file is an error, not a default.

    An experiment whose inputs live in the code rather than in a config is one
    whose inputs are invisible in the result.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing.  An experiment's inputs belong in config.json, "
            "not in argv and not in a default argument.")
    return json.loads(path.read_text())


@dataclass
class ExperimentResult:
    """What one run produced, plus enough provenance to repeat it."""

    name: str
    config: dict
    ok: bool = True
    started: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def record(self, key: str, value: Any) -> None:
        self.data[key] = _jsonable(value)

    def to_dict(self) -> dict:
        return {
            "experiment": self.name,
            "ok": self.ok,
            "commit": git_commit(),
            "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "elapsed_s": round(time.time() - self.started, 3),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "config": self.config,
            "data": self.data,
        }

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "result.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")
        print(f"\nwrote {path.relative_to(ROOT)}")
        return path


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


# --------------------------------------------------------------------------
# Statistics.  Small, explicit, and no library that hides the assumption.
# --------------------------------------------------------------------------


def bootstrap_ci(sample, statistic=np.mean, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI.  Returns (point, lo, hi).

    Percentile rather than normal-approximation because nothing here is
    promised to be symmetric, and bootstrap rather than a t interval because
    the statistics of interest (a rank, a ratio of errors) are not means.
    """
    x = np.asarray(sample, float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = statistic(x[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(x)), float(lo), float(hi)


def paired_delta(a, b, seed: int = 0) -> dict:
    """Paired comparison of two measurements of the same units.

    Reports the mean difference, its standard error, a paired t statistic, the
    fraction of pairs where b beats a, and Cohen's dz.  All five, because any
    one of them alone has been misread in this project before.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    point, lo, hi = bootstrap_ci(d, seed=seed)
    return {
        "n": int(n),
        "mean_delta": mean,
        "se": float(se),
        "t": float(mean / se) if se > 0 else float("nan"),
        "ci95": [lo, hi],
        "fraction_better": float((d > 0).mean()),
        "cohens_dz": float(mean / sd) if sd > 0 else float("nan"),
    }


def subspace_angles(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Principal angles between the column spans of two matrices, radians.

    The comparison that matters when asking "did two independent
    identifications find the same control axes".  Mode index and sign are
    private coordinates of an SVD; the subspace is not.
    """
    qu, _ = np.linalg.qr(np.asarray(u, float))
    qv, _ = np.linalg.qr(np.asarray(v, float))
    s = np.linalg.svd(qu.T @ qv, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))
