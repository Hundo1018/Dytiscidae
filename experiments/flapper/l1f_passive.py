"""L1 pass 1f: take-off and cruise of the reference machine with a passive
leading-edge feather (PREREG_L1f_passive.md).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1f_passive.py OUT.jsonl
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ZETA, COM = 0.3, 0.25


def job(a):
    from l1a import BODY, W, finish
    from rig import run
    kind = a[0]
    if kind == "rest":
        _, theta, f, k = a
        r = finish(run(BODY, 0.0, theta, f, 85.0, 0.0, 0.0, passive_feather=(k, ZETA, COM)), kind)
        r["k_spring"] = k
        return r
    _, V, f, Phi, k = a

    def go(x):
        r = run(BODY, V, 0.0, f, Phi, 0.0, x, passive_feather=(k, ZETA, COM))
        r["lift_over_real"] = r["Fz"] / W
        return r
    x0, x1 = -4.0, -10.0
    r0, r1 = go(x0), go(x1)
    best, evals = min((r0, r1), key=lambda r: abs(r["lift_over_real"] - 1.0)), 2
    while abs(best["lift_over_real"] - 1.0) > 0.03 and evals < 6:
        y0, y1 = r0["lift_over_real"] - 1.0, r1["lift_over_real"] - 1.0
        if abs(y1 - y0) < 1e-9:
            break
        x2 = min(max(x1 - y1 * (x1 - x0) / (y1 - y0), -40.0), 20.0)
        r2 = go(x2)
        evals += 1
        x0, r0, x1, r1 = x1, r1, x2, r2
        if abs(r2["lift_over_real"] - 1.0) < abs(best["lift_over_real"] - 1.0):
            best = r2
    out = finish(dict(best), kind)
    out["k_spring"] = k
    out["trimmed"] = bool(abs(best["lift_over_real"] - 1.0) <= 0.03)
    out["sustains"] = bool(out["trimmed"] and out["Fx"] >= 0.0 and out["finite"])
    return out


def main() -> int:
    grid = [("rest", th, f, k) for th in (-45.0, -60.0, -75.0, -90.0)
            for f in (3.5, 4.0, 4.5, 5.0, 5.5) for k in (0.3, 0.6, 1.2, 2.4, 5.0)]
    grid += [("cruise", V, f, Phi, k) for V in (6.0, 8.0) for f in (2.5, 3.0, 3.5, 4.0, 4.5)
             for Phi in (40.0, 60.0, 85.0) for k in (0.6, 1.2, 2.4, 5.0, 10.0)]
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
