"""F1a: trimmed level flight of a two-panel wing against wrist twist (PREREG_F1a.md).

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/twist_trim.py OUT.jsonl --jobs 8
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BODY = dict(wing_length=0.25, wing_chord=0.25 / 3, stroke_range=1.45, hand_fraction=0.5)


def one(args) -> dict:
    import rig
    V, f, Phi, Psi, T = args
    return rig.trim(BODY, V, f, Phi, Psi, T)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    grid = [(V, f, Phi, Psi, T) for V in (6.0, 8.0) for f in (4.0, 5.0, 6.0, 8.0)
            for Phi in (60.0, 80.0, 100.0) for Psi in (0.0, 10.0, 20.0)
            for T in (-10.0, 0.0, 10.0, 20.0, 30.0)]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        rows = pool.map(one, grid, chunksize=2)
    with open(Path(a.out).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows), "trimmed", sum(r["trimmed"] for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
