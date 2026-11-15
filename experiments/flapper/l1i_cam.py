"""L1 pass 1i: the feather driven from the stroke by a cam, one motor for both
(PREREG_L1i_cam.md).  The wing's motion is pass 1a's prescribed stroke and
feather; what changes is that the crank also pays the feather.

Crank torque, exact for any ideal linkage turning at a steady crank rate w:
the instantaneous power of both axes (fluid and wing inertia) over w.  It is
reported as the shoulder torque a K1 rocker at 85 deg would show for it,
T / sin(42.5 deg), so that the 12 N m gate of passes 1a-1h applies unchanged.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1i_cam.py OUT.jsonl
"""
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

K85 = math.sin(math.radians(42.5))


def cam(r):
    """Adds the crank's torque with and without the feather, from the series."""
    from l1a import J_WING
    f, Phi, Psi = r["f"], math.radians(r["Phi_deg"]), math.radians(r["Psi_deg"])
    w = 2 * math.pi * f
    both, stroke = [], []
    for t, P, Ps in zip(r["time_series"], r["power_series"], r["stroke_power_series"]):
        s, c = math.sin(w * t), math.cos(w * t)
        p_s = 2 * J_WING * (0.5 * Phi * w * c) * (-0.5 * Phi * w * w * s)
        p_f = sum(r["feather_inertia"]) * (Psi * w * s) * (Psi * w * w * c)
        both.append((P + p_s + p_f) / w / K85)
        stroke.append((Ps + p_s) / w / K85)
    for key in ("time_series", "power_series", "stroke_power_series"):
        r.pop(key)
    r["crank_eq_peak"] = max(abs(x) for x in both)
    r["crank_eq_stroke_only_peak"] = max(abs(x) for x in stroke)
    r["crank_eq_min"] = min(both)
    return r


def job(a):
    from l1a import BODY, W, finish
    from rig import run
    kind = a[0]
    if kind == "passive":
        # Pass 1h's cheapest passive points, re-read under the same crank metric
        # (the feather is free, so the crank pays only the stroke).
        _, V, th, f, Phi, k, h, law, psi0 = a
        r = finish(run(BODY, V, th, f, Phi, 0.0, psi0, passive_feather=(k, 0.3, 0.25, h),
                       cop_law=law), kind)
        r.update(axis=h, law=law, k_spring=k)
        return cam(r)
    if kind == "rest":
        _, th, f, Psi, h, law = a
        r = finish(run(BODY, 0.0, th, f, 85.0, Psi, 0.0, feather_axis=h, cop_law=law), kind)
        r.update(axis=h, law=law)
        return cam(r)
    _, V, f, Phi, Psi, h, law = a

    def go(x):
        r = run(BODY, V, 0.0, f, Phi, Psi, x, feather_axis=h, cop_law=law)
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
    out.update(axis=h, law=law)
    out["trimmed"] = bool(abs(best["lift_over_real"] - 1.0) <= 0.03)
    out["sustains"] = bool(out["trimmed"] and out["Fx"] >= 0.0 and out["finite"])
    return cam(out)


def main() -> int:
    AX, LAW = (0.0, 0.1, 0.2), ("plate", "insect")
    grid = [("rest", th, f, Psi, h, z) for th in (45.0, 60.0, 75.0, 90.0)
            for f in (3.0, 3.5, 4.0, 4.5, 5.0) for Psi in (30.0, 40.0, 50.0, 60.0, 70.0)
            for h in AX for z in LAW]
    grid += [("cruise", V, f, Phi, Psi, h, z) for V in (6.0, 8.0) for f in (2.5, 3.0, 3.5, 4.0)
             for Phi in (40.0, 60.0, 85.0) for Psi in (0.0, 10.0, 20.0, 30.0)
             for h in AX for z in LAW]
    grid += [("passive", 0.0, th, f, 85.0, 0.3, h, z, 0.0) for z, h, th, f in (
        ("plate", 0.1, -90.0, 3.5), ("plate", 0.2, -90.0, 3.5), ("plate", 0.2, -75.0, 3.5),
        ("plate", 0.0, -75.0, 4.0), ("insect", 0.1, -90.0, 3.5), ("insect", 0.0, -75.0, 4.0),
        ("insect", 0.0, -90.0, 4.0), ("insect", 0.2, -90.0, 3.5))]
    grid += [("passive", V, 0.0, f, Phi, k, h, z, p0) for z, h, V, f, Phi, k, p0 in (
        ("plate", 0.4, 6.0, 4.0, 60.0, 0.6, -14.7), ("plate", 0.2, 8.0, 3.5, 85.0, 2.4, -11.2),
        ("insect", 0.3, 8.0, 4.0, 60.0, 0.6, -2.9), ("insect", 0.3, 8.0, 4.0, 85.0, 0.6, -6.6))]
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
