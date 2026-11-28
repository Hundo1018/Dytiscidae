"""Probe for pass 1l: the torque-driven rig in water on pass 1c's point (wing
0.4 of its length, 0.3 m/s, stroke 85 deg, 0.5 Hz, feather 30) against the
held rig, and one flap point with a feather on both rigs.  Not a measurement.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_water.py
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
FL = dict(x_h=0.4, open_deg=80.0, sign=1, k_f=0.005, damp_f=0.002)


def job(a):
    from l1a import BODY, LENGTH
    body = dict(BODY, wing_length=0.4 * LENGTH)
    which, Phi, f, Psi, flap, law, h = a
    t0 = time.time()
    if which == "old":
        from rig import run
        r = run(body, 0.3, 0.0, f, Phi, Psi, 0.0, beats=3, skip=1, medium="water",
                one_way_flap=FL if flap else None, cop_law=law, feather_axis=h)
        return a, "Fx %+.2f P %.2f stroke torque %.1f" % (r["Fx"], r["aero_power_W"], r["stroke_torque_peak"])
    from freerig import run_free
    r = run_free(body, 0.3, 0.0, f, Phi, Psi, 0.0, medium="water", hold_beats=1, beats=2,
                 one_way_flap=FL if flap else None, cop_law=law, feather_axis=h)
    return a, "Fx %+.2f P %.2f (stroke %.2f) stroke torque %.1f crank %.1f err %.2f deg, %.0f s" % (
        r["Fx"], r["power_W"], r["stroke_power_W"], r["stroke_torque_peak"], r["crank_eq_peak"],
        r["track_err_deg"], time.time() - t0)


def main():
    grid = [(w, 85.0, 0.5, Psi, False, law, h) for Psi in (45.0, 60.0)
            for law, h in ((None, 0.0), ("plate", 0.0), ("plate", 0.1)) for w in ("old", "new")]
    with mp.get_context("spawn").Pool(6) as pool:
        for row in pool.map(job, grid, chunksize=1):
            print(*row)


if __name__ == "__main__":
    main()
