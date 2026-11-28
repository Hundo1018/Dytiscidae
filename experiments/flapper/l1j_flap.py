"""L1 pass 1j: the one-way flap as a built panel (PREREG_L1j_flap.md).

Block A repeats pass 1e's grid (up-down stroke, body level, 0.3 m/s) with the
flap welded, free to swing up, free to swing down.  Block B is a rowing
stroke: hull vertical, wings sweeping fore and aft, at rest and at 0.3 m/s.
Block C varies the flap on one rowing point.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1j_flap.py OUT.jsonl
"""
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FLAP = dict(x_h=0.4, open_deg=80.0, k_f=0.005, damp_f=0.002)


def job(a):
    from l1a import BODY, LENGTH
    from rig import run
    block, ks, sign, V, th, Phi, f, Psi, x_h, k_f = a
    body = dict(BODY, wing_length=ks * LENGTH)
    fl = dict(FLAP, sign=sign, x_h=x_h, k_f=k_f)
    r = run(body, V, th, f, Phi, Psi, 0.0, beats=3, skip=1, medium="water", one_way_flap=fl)
    for key in ("time_series", "power_series", "stroke_power_series"):
        r.pop(key)
    r.update(block=block, fold=ks, sign=sign, x_h=x_h, k_f=k_f,
             crank_torque=r["stroke_torque_peak"] * math.sin(math.radians(Phi) / 2)
             / math.sin(math.radians(42.5)))
    return r


def main() -> int:
    X, K = FLAP["x_h"], FLAP["k_f"]
    grid = [("A", ks, sg, 0.3, 0.0, 40.0, 0.0, 0.0, X, K) for ks in (0.4, 0.6) for sg in (0, 1, -1)]
    grid += [("A", ks, sg, 0.3, 0.0, Phi, f, Psi, X, K) for ks in (0.4, 0.6) for sg in (0, 1, -1)
             for f in (0.25, 0.5, 1.0, 1.5) for Phi in (40.0, 85.0) for Psi in (0.0, 10.0, 20.0, 30.0)]
    grid += [("B", ks, sg, 0.3, 90.0, 40.0, 0.0, 0.0, X, K) for ks in (0.4, 0.6) for sg in (0, 1, -1)]
    grid += [("B", ks, sg, V, 90.0, Phi, f, 0.0, X, K) for ks in (0.4, 0.6) for sg in (0, 1, -1)
             for V in (0.0, 0.3) for f in (0.25, 0.5, 0.75, 1.0) for Phi in (40.0, 85.0)]
    grid += [("C", 0.4, 1, 0.0, 90.0, 85.0, 0.5, 0.0, x, k) for x in (0.2, 0.4, 0.7)
             for k in (0.005, 0.05)]
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
