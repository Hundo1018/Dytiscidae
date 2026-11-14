"""L1 pass 1e: chord narrowed on one half of the beat (PREREG_L1e_recovery.md).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1e_recovery.py OUT.jsonl
"""
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SPANS = (0.4, 0.6)
ARMS = {"fixed": None, "narrow_pos": (0.4, 1), "narrow_neg": (0.4, -1), "narrow_all": (0.4, 0)}
STROKES = (40.0, 85.0)
FREQS = (0.25, 0.5, 1.0, 1.5)
FEATHERS = (0.0, 10.0, 20.0, 30.0)


def job(a):
    from l1a import BODY, LENGTH
    from rig import run
    ks, arm, Phi, f, Psi = a
    body = dict(BODY, wing_length=ks * LENGTH)
    r = run(body, 0.3, 0.0, f, Phi, Psi, 0.0, beats=3, skip=1, medium="water",
            chord_law=ARMS[arm])
    r.update(fold=ks, arm=arm, crank_torque=r["stroke_torque_peak"] * math.sin(
        math.radians(Phi) / 2) / math.sin(math.radians(42.5)))
    return r


def main() -> int:
    grid = [(ks, "fixed", 40.0, 0.0, 0.0) for ks in SPANS]
    grid += [(ks, arm, Phi, f, Psi) for ks in SPANS for arm in ARMS for f in FREQS
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
