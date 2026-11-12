"""F0c: trimmed level flight of the F0 wing (PREREG_F0c.md).

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/cruise_trim.py OUT.jsonl --jobs 8
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def trim(args) -> dict:
    import flight_modes as M
    R, V, f, Phi, Psi = args
    lo, hi = -18.0, 0.0
    x0, x1 = -8.0, -12.0
    r0 = M.rig((R, V, 0.0, f, Phi, Psi, x0))
    r1 = M.rig((R, V, 0.0, f, Phi, Psi, x1))
    evals = 2
    best = min((r0, r1), key=lambda r: abs(r["lift_over_weight"] - 1.0))
    while abs(best["lift_over_weight"] - 1.0) > 0.03 and evals < 4:
        y0, y1 = r0["lift_over_weight"] - 1.0, r1["lift_over_weight"] - 1.0
        if abs(y1 - y0) < 1e-9:
            break
        x2 = min(max(x1 - y1 * (x1 - x0) / (y1 - y0), lo), hi)
        r2 = M.rig((R, V, 0.0, f, Phi, Psi, x2))
        evals += 1
        x0, r0, x1, r1 = x1, r1, x2, r2
        if abs(r2["lift_over_weight"] - 1.0) < abs(best["lift_over_weight"] - 1.0):
            best = r2
    out = dict(best)
    out["trimmed"] = bool(abs(best["lift_over_weight"] - 1.0) <= 0.03)
    out["evals"] = evals
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    grid = [(R, V, f, Phi, Psi) for R in (0.20, 0.25) for V in (5.0, 6.0, 7.0, 8.0, 9.0)
            for f in (3.0, 4.0, 5.0, 6.0, 8.0) for Phi in (40.0, 60.0, 80.0, 100.0)
            for Psi in (0.0, 10.0, 20.0, 30.0)]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        rows = pool.map(trim, grid, chunksize=2)
    with open(Path(a.out).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows), "trimmed", sum(r["trimmed"] for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
