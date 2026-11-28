"""L1 pass 1l: the cam's feather amplitudes in water (PREREG_L1l_camwater.md).
Held body, 0.3 m/s, wing folded to 0.4 and 0.6 of its length, on `rig.run`
(prescribed motion, as passes 1b-1e), with pass 1i's crank metric.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1l_camwater.py OUT.jsonl
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def job(a):
    from l1a import BODY, LENGTH, finish
    from l1i_cam import cam
    from rig import run
    ks, Phi, f, Psi, law, h = a
    body = dict(BODY, wing_length=ks * LENGTH)
    r = run(body, 0.3, 0.0, f, Phi, Psi, 0.0, beats=3, skip=1, medium="water", cop_law=law,
            feather_axis=h)
    if f > 0:
        r = cam(finish(r, "water"))
    else:
        for key in ("time_series", "power_series", "stroke_power_series"):
            r.pop(key)
    r.update(fold=ks, law=law, axis=h)
    return r


def main() -> int:
    grid = [(ks, 40.0, 0.0, 0.0, None, 0.0) for ks in (0.4, 0.6)]
    grid += [(ks, Phi, f, Psi, law, h) for ks in (0.4, 0.6) for Phi in (40.0, 85.0)
             for f in (0.25, 0.5, 0.75, 1.0) for Psi in (20.0, 30.0, 50.0, 60.0)
             for law, h in ((None, 0.0), ("plate", 0.1), ("insect", 0.1))]
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
