"""L1 pass 1a: the L0 reference machine at scale on the held rig
(PREREG_L1a.md): (a) lift from rest, (b) level cruise, (c) water.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1a.py OUT.jsonl
"""
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

G = 9.80665
MASS, SPAN, AREA, HULL_R = 0.60, 1.4, 0.34, 0.04
LENGTH = SPAN / 2 - HULL_R
BODY = dict(wing_length=LENGTH, wing_chord=AREA / (2 * LENGTH), hull_length=0.5,
            hull_radius=HULL_R)
W = MASS * G
J_WING = 0.187 * (AREA / 2) * LENGTH ** 2 / 3        # kg m2, one wing (A13: 187 g/m2)


def finish(r, kind):
    Phi, f = math.radians(r["Phi_deg"]), r["f"]
    inertial = 2 * J_WING * (Phi / 2) * (2 * math.pi * f) ** 2      # both wings, peak
    r.update(kind=kind, real_weight=W, inertial_torque_peak=inertial,
             drive_torque_peak=math.hypot(r["stroke_torque_peak"], inertial))
    return r


def job(a):
    from rig import run
    kind = a[0]
    if kind == "rest":
        _, theta, f, Psi = a
        return finish(run(BODY, 0.0, theta, f, 85.0, Psi, 0.0), kind)
    if kind == "water":
        _, Phi, f, Psi = a
        return finish(run(BODY, 0.3, 0.0, f, Phi, Psi, 0.0, medium="water"), kind)
    _, V, f, Phi, Psi = a                                 # cruise: trim lift to W

    def go(x):
        r = run(BODY, V, 0.0, f, Phi, Psi, x)
        r["lift_over_real"] = r["Fz"] / W
        return r
    x0, x1 = -4.0, -10.0
    r0, r1 = go(x0), go(x1)
    best, evals = min((r0, r1), key=lambda r: abs(r["lift_over_real"] - 1.0)), 2
    while abs(best["lift_over_real"] - 1.0) > 0.03 and evals < 6:
        y0, y1 = r0["lift_over_real"] - 1.0, r1["lift_over_real"] - 1.0
        if abs(y1 - y0) < 1e-9:
            break
        x2 = min(max(x1 - y1 * (x1 - x0) / (y1 - y0), -30.0), 5.0)
        r2 = go(x2)
        evals += 1
        x0, r0, x1, r1 = x1, r1, x2, r2
        if abs(r2["lift_over_real"] - 1.0) < abs(best["lift_over_real"] - 1.0):
            best = r2
    out = finish(dict(best), kind)
    out["trimmed"] = bool(abs(best["lift_over_real"] - 1.0) <= 0.03)
    out["sustains"] = bool(out["trimmed"] and out["Fx"] >= 0.0 and out["finite"])
    return out


def main() -> int:
    grid = [("rest", th, f, Psi) for th in (45.0, 60.0, 75.0, 90.0)
            for f in (0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5) for Psi in (30.0, 40.0, 50.0, 60.0, 70.0)]
    grid += [("cruise", V, f, Phi, Psi) for V in (6.0, 8.0) for f in (2.5, 3.0, 3.5, 4.0, 4.5)
             for Phi in (40.0, 60.0, 85.0) for Psi in (0.0, 10.0, 20.0, 30.0)]
    grid += [("water", Phi, f, Psi) for Phi, f in ((15.0, 1.0), (40.0, 0.38), (85.0, 0.15),
                                                    (40.0, 0.0))
             for Psi in (30.0, 45.0, 60.0)]
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
