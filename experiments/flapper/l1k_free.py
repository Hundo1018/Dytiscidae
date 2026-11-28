"""L1 pass 1k: pass 1i's gate points on the torque-driven rig, body held and
body let go (PREREG_L1k_free.md).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1k_free.py OUT.jsonl
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
G = 9.80665


def points():
    rows = [json.loads(x) for x in open(HERE / "results" / "L1i_cam.jsonl")]
    W = rows[0]["real_weight"]
    to = [r for r in rows if r["kind"] == "rest" and r["Fz"] / W >= 1.6 and r["crank_eq_peak"] <= 12
          and r["aero_power_W"] <= 150]
    cr = [r for r in rows if r["kind"] == "cruise" and r["sustains"] and r["aero_power_W"] <= 41
          and r["crank_eq_peak"] <= 12]
    key = ("V", "theta_deg", "f", "Phi_deg", "Psi_deg", "psi0_deg", "axis", "law")
    return [tuple(r[k] for k in key) for r in to], [tuple(r[k] for k in key) for r in cr]


def job(a):
    import numpy as np
    from freerig import run_free
    from l1a import BODY
    kind, pt, free, damp = a
    V, th, f, Phi, Psi, psi0, h, law = pt

    def go(x):
        return run_free(BODY, V, th, f, Phi, Psi, x, free=free, feather_axis=h, cop_law=law,
                        keep_series=True, joint_damping=damp)
    evals = 1
    r = go(psi0)
    if kind == "cruise" and free == "xz":
        # re-trim the cam's offset so the released body neither climbs nor sinks
        x0, r0 = psi0, r
        x1 = psi0 - 1.0
        r1 = go(x1)
        evals = 2
        best = min((r0, r1), key=lambda q: abs(q["lift_over_weight"] - 1.0))
        while abs(best["lift_over_weight"] - 1.0) > 0.01 and evals < 6:
            y0, y1 = r0["lift_over_weight"] - 1.0, r1["lift_over_weight"] - 1.0
            if abs(y1 - y0) < 1e-9:
                break
            x2 = min(max(x1 - y1 * (x1 - x0) / (y1 - y0), -30.0), 5.0)
            r2 = go(x2)
            evals += 1
            x0, r0, x1, r1 = x1, r1, x2, r2
            if abs(r2["lift_over_weight"] - 1.0) < abs(best["lift_over_weight"] - 1.0):
                best = r2
        r = best
    s = r.pop("series")
    n = int(round(1.0 / (f * 2.5e-4)))
    vel = np.array(s["vel"])
    if "z" in free:                       # the first beat after release, before a climb builds
        r["lift_first_beat"] = 1.0 + float(vel[n - 1, 2] - vel[0, 2]) / (G / f)
    if "p" in free:
        pc = np.degrees(np.array(s["pitch"])) - th
        r["pitch_min_deg"], r["pitch_max_deg"] = float(pc.min()), float(pc.max())
        r["pitch_first_beat_deg"] = float(np.abs(pc[:n]).max())
    r.update(kind=kind, axis=h, law=law, psi0_1i=psi0, evals=evals, joint_damping=damp)
    return r


def main() -> int:
    to, cr = points()
    grid = [(k, pt, fr, 0.0) for k, pts in (("takeoff", to), ("cruise", cr)) for pt in pts
            for fr in ("", "xz", "xzp")]
    grid += [(k, pt, "", None) for k, pts in (("takeoff", to), ("cruise", cr)) for pt in pts]
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows), "takeoff points", len(to), "cruise points", len(cr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
