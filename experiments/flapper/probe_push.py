"""How much the stroke's inertial push on a free hinge changes passes 1h and 1j
(it was wiped there).  Pass 1h's cheapest passive take-off point and pass 1j's
still-water rowing point, as run and with the push acting.  Not a measurement.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_push.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def job(a):
    from l1a import BODY, LENGTH, W
    from rig import run
    kind, legacy, coupling = a
    if kind == "hinge":
        r = run(BODY, 0.0, -90.0, 3.5, 85.0, 0.0, 0.0, passive_feather=(0.3, 0.3, 0.25, 0.1),
                cop_law="plate", coupling=coupling, legacy_coupling=legacy)
        return a, "lift/W %.3f P %.1f stroke torque %.2f feather %s" % (
            r["Fz"] / W, r["aero_power_W"], r["stroke_torque_peak"], [round(x) for x in r["feather_taken_deg"]])
    body = dict(BODY, wing_length=0.4 * LENGTH)
    fl = dict(x_h=0.4, open_deg=80.0, sign=1, k_f=0.005, damp_f=0.002)
    th, V, Psi = (90.0, 0.0, 0.0) if kind == "row" else (0.0, 0.3, 30.0)
    r = run(body, V, th, 0.5, 85.0, Psi, 0.0, beats=3, skip=1, medium="water", one_way_flap=fl,
            legacy_coupling=legacy)
    return a, "Fx %+.2f P %.2f stroke torque %.1f flap max %s" % (
        r["Fx"], r["aero_power_W"], r["stroke_torque_peak"], [round(x) for x in r["flap_open_max_deg"]])


def main():
    grid = [("hinge", True, "static"), ("hinge", False, "static"), ("hinge", False, "step"),
            ("row", True, "step"), ("row", False, "step"), ("updown", True, "step"), ("updown", False, "step")]
    with mp.get_context("spawn").Pool(6) as pool:
        for row in pool.map(job, grid, chunksize=1):
            print(*row)


if __name__ == "__main__":
    main()
