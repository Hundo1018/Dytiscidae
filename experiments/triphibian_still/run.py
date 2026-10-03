"""What the triphibian island reads for a machine with its actuators held still.

The project's standing check (CLAUDE.md, "The lesson this project keeps
re-learning"): before a score is trusted, ask what it reads for a machine that
does nothing.  The triphibian island pays the weakest of the three media, and a
still machine earns a little in each medium (water ~0.01 after task
conditioning), so the question is whether "a little in all three" is enough to
climb the island's ladder.

Every body plan, two seeds, the real Tier-1 path (``evaluate_tier1``), once
with the body's base gait and once with every amplitude at zero.

    PYTHONPATH=. python experiments/triphibian_still/run.py            # measure + analyse
    PYTHONPATH=. python experiments/triphibian_still/run.py --analyse  # re-read results only
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "result.json"
MEDIA = ("air", "water", "land")


def measure() -> dict:
    from dytiscidae.control.cpg import CPGParams
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.evaluate import Controller, evaluate_tier1
    from dytiscidae.envs.triphibian import TriphibianEnv

    rows = []
    t0 = time.time()
    for plan in BODY_PLANS:
        p = build(BODY_PLANS[plan]())
        for seed in (0, 1):
            b = TriphibianEnv(p, seed=seed).cpg.base
            still = CPGParams(amplitude=np.zeros_like(np.asarray(b.amplitude, float)),
                              phase=np.asarray(b.phase, float),
                              offset=np.asarray(b.offset, float),
                              frequency=float(b.frequency))
            for name, prm in (("gait", b), ("still", still)):
                r = evaluate_tier1(p, controller=Controller(params=prm),
                                   segment_seconds=8.0, seed=seed)
                comps = {m: float(r.segments[m].competence) if m in r.segments else 0.0
                         for m in MEDIA}
                rows.append({"plan": plan, "seed": seed, "drive": name, **comps,
                             "mission_fraction": float(r.mission_fraction)})
                print(f"{plan:10s} s{seed} {name:5s} " +
                      " ".join(f"{m}={comps[m]:.4f}" for m in MEDIA), flush=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=HERE).stdout.strip()
    except Exception:
        commit = ""
    return {"commit": commit, "numpy": np.__version__, "segment_seconds": 8.0,
            "wall_s": round(time.time() - t0, 1), "rows": rows}


def analyse(res: dict) -> None:
    from dytiscidae.evolution.curriculum import triphibian_stage_score
    from dytiscidae.evolution.islands import triphibian_score

    for drive in ("gait", "still"):
        rows = [r for r in res["rows"] if r["drive"] == drive]
        c = np.array([[r[m] for m in MEDIA] for r in rows])
        sc = [triphibian_score([r[m] for m in MEDIA]) for r in rows]
        st0 = [triphibian_stage_score(0, [r[m] for m in MEDIA]) for r in rows]
        st1 = [triphibian_stage_score(1, [r[m] for m in MEDIA]) for r in rows]
        print(f"{drive:5s} n={len(rows):2d}  mean " +
              " ".join(f"{m}={v:.4f}" for m, v in zip(MEDIA, c.mean(0))) +
              "  max " + " ".join(f"{m}={v:.4f}" for m, v in zip(MEDIA, c.max(0))) +
              f"  island max {max(sc):.4f}  stage0 max {max(st0):.4f}"
              f"  stage1 max {max(st1):.4f}")


if __name__ == "__main__":
    if "--analyse" not in sys.argv:
        res = measure()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1))
    res = json.loads(OUT.read_text())
    try:
        analyse(res)
    except ImportError as exc:
        print("analysis needs the triphibian island:", exc)
