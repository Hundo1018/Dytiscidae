"""L1 pass 1d: water, wings contracted in span and chord (PREREG_L1d_fold2.md).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1d_fold2.py OUT.jsonl
"""
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WINGS = ((0.6, 0.6), (0.4, 0.4), (0.4, 0.7))      # span fraction, chord fraction
SPEEDS = (0.3, 0.6)
STROKES = (25.0, 40.0, 60.0, 85.0)
FREQS = (0.1, 0.25, 0.5, 1.0, 1.5, 2.5)
FEATHERS = (0.0, 10.0, 20.0, 30.0)


def job(a):
    from l1a import BODY, LENGTH
    from rig import run
    (ks, kc), V, Phi, f, Psi = a
    body = dict(BODY, wing_length=ks * LENGTH, wing_chord=kc * BODY["wing_chord"])
    r = run(body, V, 0.0, f, Phi, Psi, 0.0, beats=3, skip=1, medium="water")
    r["fold"], r["chord_fold"] = ks, kc
    r["crank_torque"] = r["stroke_torque_peak"] * math.sin(math.radians(Phi) / 2) / math.sin(
        math.radians(42.5))
    return r


def main() -> int:
    grid = [(w, V, 40.0, 0.0, 0.0) for w in WINGS for V in SPEEDS]
    grid += [(w, V, Phi, f, Psi) for w in WINGS for V in SPEEDS for f in FREQS
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
