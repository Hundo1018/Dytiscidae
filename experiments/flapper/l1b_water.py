"""L1 pass 1b: water map of the L0 reference machine at low frequency and
small feather (PREREG_L1b_water.md).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1b_water.py OUT.jsonl
"""
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STROKES = (15.0, 25.0, 40.0, 60.0, 85.0)
FREQS = (0.05, 0.1, 0.15, 0.25, 0.4, 0.6)
FEATHERS = (0.0, 10.0, 20.0, 30.0, 45.0)
V = 0.3
RHO = 1000.0


def job(a):
    from l1a import BODY, LENGTH
    from rig import run
    Phi, f, Psi = a
    r = run(BODY, V, 0.0, f, Phi, Psi, 0.0, beats=3, skip=1, medium="water")
    chord = BODY["wing_chord"]
    j_added = RHO * math.pi * (chord / 2) ** 2 * LENGTH * LENGTH ** 2 / 3     # one wing, broadside
    r["broadside_added_torque"] = 2 * j_added * math.radians(Phi) / 2 * (2 * math.pi * f) ** 2
    r["crank_torque"] = r["stroke_torque_peak"] * math.sin(math.radians(Phi) / 2) / math.sin(
        math.radians(42.5))
    return r


def main() -> int:
    grid = [(Phi, f, Psi) for f in FREQS for Phi in STROKES for Psi in FEATHERS]
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
