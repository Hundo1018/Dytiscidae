"""L1 pass 1c: the reference machine in water with only the inner k of each
wing working (PREREG_L1c_fold.md).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1c_fold.py OUT.jsonl
"""
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FOLDS = (0.6, 0.4, 0.25)
SPEEDS = (0.3, 0.6)
STROKES = (25.0, 40.0, 60.0, 85.0)
FREQS = (0.1, 0.25, 0.5, 1.0, 1.5)
FEATHERS = (0.0, 10.0, 20.0, 30.0)


def job(a):
    from l1a import BODY, LENGTH
    from rig import run
    k, V, Phi, f, Psi = a
    body = dict(BODY, wing_length=k * LENGTH)
    r = run(body, V, 0.0, f, Phi, Psi, 0.0, beats=3, skip=1, medium="water")
    r["fold"] = k
    r["crank_torque"] = r["stroke_torque_peak"] * math.sin(math.radians(Phi) / 2) / math.sin(
        math.radians(42.5))
    return r


def main() -> int:
    grid = [(k, V, 40.0, 0.0, 0.0) for k in FOLDS for V in SPEEDS]
    grid += [(k, V, Phi, f, Psi) for k in FOLDS for V in SPEEDS for f in FREQS
             for Phi in STROKES for Psi in FEATHERS]
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
