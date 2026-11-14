"""Probe 2 for the hinge-axis sweep, after the hinge damping was found wiped
each step: does stiffness still matter with the axis at 0.2 chord, and does
zeta now matter?  Rest, 4 Hz, stroke 85 deg, body -60 deg.  Not a measurement.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_axis2.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def job(a):
    from l1a import BODY, W
    from rig import run
    k, z, h = a
    r = run(BODY, 0.0, -60.0, 4.0, 85.0, 0.0, 0.0, passive_feather=(k, z, 0.25, h))
    return a, round(r["Fz"] / W, 3), round(r["aero_power_W"], 1), \
        round(r["stroke_torque_peak"], 2), [round(x, 1) for x in r["feather_taken_deg"]], r["finite"]


def main():
    grid = [(k, z, h) for h in (0.0, 0.2) for k in (0.15, 0.6, 2.4, 10.0) for z in (0.0, 0.3)]
    with mp.get_context("spawn").Pool(6) as pool:
        for row in pool.map(job, grid, chunksize=1):
            print(*row)


if __name__ == "__main__":
    main()
