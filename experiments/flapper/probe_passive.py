"""Probe for pass 1f: lift from rest at 4 Hz, stroke 85 deg, body vertical, with
a passive leading-edge feather hinge, across stiffness, damping and the
wing's centre of mass along the chord.  Not a measurement: it informs the
pre-registration and is disclosed there.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_passive.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def job(a):
    from l1a import BODY, W
    from rig import run
    k, z, com = a
    r = run(BODY, 0.0, 90.0, 4.0, 85.0, 0.0, 0.0, passive_feather=(k, z, com))
    return a, round(r["Fz"] / W, 2), round(r["aero_power_W"]), round(r["stroke_torque_peak"], 1), \
        [round(x) for x in r["feather_taken_deg"]], r["finite"]


def main():
    grid = [(k, z, c) for c in (0.25, 0.05) for k in (0.3, 1.2, 5.0, 20.0) for z in (0.1, 0.5)]
    with mp.get_context("spawn").Pool(8) as pool:
        for row in pool.map(job, grid):
            print(*row)


if __name__ == "__main__":
    main()
