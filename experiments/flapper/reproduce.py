"""R0: two machines that flew, at their reported operating points, through the
held-body rig (PREREG_R0_reproduce.md).  Lift is trimmed to the real weight.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/reproduce.py OUT.jsonl
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

G = 9.80665
MACHINES = {
    # mass kg, span m, wing area m2 (both wings), hull length m
    "eflap": dict(mass=0.510, span=1.5, area=0.438, hull=0.55,
                  V=(3.0, 4.0, 5.0), f=(0.0, 2.0, 3.5, 4.0), Phi=(40.0, 60.0, 80.0, 100.0)),
    "raven2": dict(mass=0.340, span=1.33, area=0.25, hull=0.45,   # area TYPED
                   V=(5.6, 6.7), f=(0.0, 2.0, 4.0), Phi=(57.0, 67.0)),
}
PSI = (0.0, 10.0, 20.0, 30.0)


def body(name):
    k = MACHINES[name]
    hull_radius = 0.04
    length = k["span"] / 2 - hull_radius
    return dict(wing_length=length, wing_chord=k["area"] / (2 * length),
                hull_length=k["hull"], hull_radius=hull_radius)


def point(args):
    """Trim the feather offset so the mean lift is the real weight (secant)."""
    from rig import run
    name, V, f, Phi, Psi = args
    W = MACHINES[name]["mass"] * G
    b = body(name)

    def go(x):
        r = run(b, V, 0.0, f, Phi, Psi, x)
        r["lift_over_real"] = r["Fz"] / W
        return r
    lo, hi = -30.0, 5.0
    x0, x1 = -6.0, -12.0
    r0, r1 = go(x0), go(x1)
    evals, best = 2, None
    best = min((r0, r1), key=lambda r: abs(r["lift_over_real"] - 1.0))
    while abs(best["lift_over_real"] - 1.0) > 0.03 and evals < 6:
        y0, y1 = r0["lift_over_real"] - 1.0, r1["lift_over_real"] - 1.0
        if abs(y1 - y0) < 1e-9:
            break
        x2 = min(max(x1 - y1 * (x1 - x0) / (y1 - y0), lo), hi)
        r2 = go(x2)
        evals += 1
        x0, r0, x1, r1 = x1, r1, x2, r2
        if abs(r2["lift_over_real"] - 1.0) < abs(best["lift_over_real"] - 1.0):
            best = r2
    out = dict(best)
    out.update(machine=name, real_weight=W, evals=evals,
               trimmed=bool(abs(best["lift_over_real"] - 1.0) <= 0.03))
    out["sustains"] = bool(out["trimmed"] and out["Fx"] >= 0.0 and out["finite"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    grid = [(n, V, f, Phi, Psi) for n, k in MACHINES.items() for V in k["V"] for f in k["f"]
            for Phi in k["Phi"] for Psi in PSI]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        rows = pool.map(point, grid, chunksize=1)
    with open(Path(a.out).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows), "trimmed", sum(r["trimmed"] for r in rows),
          "sustain", sum(r["sustains"] for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
