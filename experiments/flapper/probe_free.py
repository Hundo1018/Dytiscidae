"""Cross-check of `freerig.run_free` against `rig.run` on two pass-1i gate
points, body held, then the same points with the body let go in x and z.
Not a measurement: it informs PREREG_L1k_free.md.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_free.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TAKEOFF = (0.0, 90.0, 4.5, 85.0, 60.0, 0.0, 0.1, "plate")
CRUISE = (8.0, 0.0, 3.5, 60.0, 20.0, -7.4, 0.2, "plate")


def job(a):
    from l1a import BODY, W
    kind, pt = a[0], a[1]
    V, th, f, Phi, Psi, psi0, h, law = pt
    if kind == "old":
        from l1a import finish
        from l1i_cam import cam
        from rig import run
        r = cam(finish(run(BODY, V, th, f, Phi, Psi, psi0, feather_axis=h, cop_law=law), "rest"))
        return a[0], pt[0], "lift/W %.3f Fx %+.2f P %.1f crank %.1f stroke-only %.1f" % (
            r["Fz"] / W, r["Fx"], r["aero_power_W"], r["crank_eq_peak"], r["crank_eq_stroke_only_peak"])
    from freerig import run_free
    r = run_free(BODY, V, th, f, Phi, Psi, psi0, free=a[2], mass=a[3], feather_axis=h, cop_law=law, keep_series=True)
    s = "mass %.3f lift/W %.3f Fx %+.2f P %.1f crank %.1f stroke-only %.1f err %.2f deg" % (
        r["mass_kg"], r["Fz"] / W, r["Fx"], r["power_W"], r["crank_eq_peak"],
        r["crank_eq_stroke_only_peak"], r["track_err_deg"])
    if "pitch_moment_mean" in r:
        s += " pitch moment mean %+.2f peak %.2f N m" % (r["pitch_moment_mean"], r["pitch_moment_peak"])
    if "pitch_end_deg" in r:
        import numpy as np
        pc = np.degrees(np.array(r["series"]["pitch"])) - th
        s += " pitch end %+.0f deg, range %+.0f..%+.0f" % (r["pitch_end_deg"], pc.min(), pc.max())
    if a[2]:
        s += " dx %+.3f dz %+.3f vz_end %+.2f bob %.1f mm" % (r["dx"], r["dz"], r["vz_end"], r["bob_mm"])
    return kind, pt[0], a[2], s, r["finite"]


def main():
    grid = [("old", TAKEOFF), ("old", CRUISE)]
    for pt in (TAKEOFF, CRUISE):
        grid += [("new", pt, "", 0.60), ("new", pt, "xz", 0.60), ("new", pt, "xzp", 0.60)]
    with mp.get_context("spawn").Pool(6) as pool:
        for row in pool.map(job, grid, chunksize=1):
            print(*row)


if __name__ == "__main__":
    main()
