"""Probe for pass S1: is the "wetting" entrainment mode stable on the
torque-driven rig in water, and does the reference added-mass load run on the
held rig.  Two points, not a measurement.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_entrain.py
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def job(a):
    from l1a import BODY, LENGTH
    body = dict(BODY, wing_length=0.4 * LENGTH)
    which, Psi, ent, ref = a
    t0 = time.time()
    if which == "held":
        from rig import run
        r = run(body, 0.3, 0.0, 0.5, 85.0, Psi, 0.0, beats=3, skip=1, medium="water",
                solver_flags={"entrainment": ent}, added_mass_ref=ref)
        return a, "Fx %+.2f P %.2f stroke torque %.1f, %.0f s" % (
            r["Fx"], r["aero_power_W"], r["stroke_torque_peak"], time.time() - t0)
    from freerig import run_free
    r = run_free(body, 0.3, 0.0, 0.5, 85.0, Psi, 0.0, medium="water", hold_beats=1, beats=2,
                 solver_flags={"entrainment": ent})
    return a, "Fx %+.2f P %.2f stroke torque %.1f crank %.1f err %.2f deg, %.0f s" % (
        r["Fx"], r["power_W"], r["stroke_torque_peak"], r["crank_eq_peak"],
        r["track_err_deg"], time.time() - t0)


def main():
    grid = [("held", Psi, e, r) for Psi in (30.0, 60.0)
            for e, r in ((True, False), (False, False), (False, True))]
    grid += [("free", Psi, e, False) for Psi in (30.0, 60.0) for e in (False, "wetting")]
    with mp.get_context("spawn").Pool(5) as pool:
        for row in pool.map(job, grid, chunksize=1):
            print(*row)


if __name__ == "__main__":
    main()
