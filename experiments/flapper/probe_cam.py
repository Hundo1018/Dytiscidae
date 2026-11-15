"""Probe for pass 1i: pass 1a's 4 Hz take-off point (body 90 deg, feather 60,
in 1a's best family) and its cheapest cruise point, with the feather driven by
the crank.  Also 1a's point with no centre-of-pressure law, which must still
read 1a's numbers.  Not a measurement: it informs PREREG_L1i_cam.md.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_cam.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def job(a):
    from l1a import BODY, W, finish
    from l1i_cam import cam
    from rig import run
    V, th, f, Phi, Psi, psi0, h, law = a
    r = cam(finish(run(BODY, V, th, f, Phi, Psi, psi0, feather_axis=h, cop_law=law), "rest"))
    return a, round(r["Fz"] / W, 2), round(r["Fx"], 2), round(r["aero_power_W"]), \
        round(r["drive_torque_peak"], 1), round(r["crank_eq_stroke_only_peak"], 1), \
        round(r["crank_eq_peak"], 1), round(r["feather_torque_peak"], 2), r["finite"]


def main():
    grid = [(0.0, 90.0, 4.0, 85.0, 60.0, 0.0, 0.0, None)]
    grid += [(0.0, 90.0, 4.0, 85.0, 60.0, 0.0, h, law) for law in ("plate", "insect") for h in (0.0, 0.1, 0.2)]
    grid += [(8.0, 0.0, 2.5, 40.0, 20.0, -8.0, h, law) for law in (None, "plate") for h in (0.0, 0.2)]
    with mp.get_context("spawn").Pool(6) as pool:
        print("args | lift W | Fx | P | shoulder N m (1a) | crank-eq stroke only | crank-eq with cam | feather torque")
        for row in pool.map(job, grid, chunksize=1):
            print(*row)


if __name__ == "__main__":
    main()
