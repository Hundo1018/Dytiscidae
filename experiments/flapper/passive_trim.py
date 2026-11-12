"""F1a-2: trimmed level flight with a passive wrist (PREREG_F1a2.md).

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/passive_trim.py OUT.jsonl --jobs 8
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def one(args) -> dict:
    import rig
    c, V, f, Phi, Psi = args
    body = dict(wing_length=0.25, wing_chord=0.25 / 3, stroke_range=1.45, hand_fraction=0.5,
                hand_compliance=c)
    r = rig.trim(body, V, f, Phi, Psi)
    r["hand_compliance"] = c
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    grid = [(c, V, f, Phi, Psi) for c in (0.5, 1.0, 2.0, 4.0) for V in (6.0, 8.0)
            for f in (5.0, 6.0, 8.0) for Phi in (60.0, 80.0, 100.0) for Psi in (0.0, 10.0, 20.0)]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        rows = pool.map(one, grid, chunksize=2)
    with open(Path(a.out).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows), "trimmed", sum(r["trimmed"] for r in rows),
          "non-finite", sum(not r["finite"] for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
