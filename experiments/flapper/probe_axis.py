"""Probe for a hinge-axis sweep: is moving the passive feather axis aft the same,
in this model, as a stiffer spring at the leading edge?  Every strip load acts
at the quarter chord, so the aerodynamic arm about an axis at h x chord is
(0.25 - h) x chord, and (k, h) should read like (k * 0.25 / (0.25 - h), 0).
Not a measurement: it decides whether the sweep can tell anything.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_axis.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def job(a):
    from l1a import BODY, W
    from rig import run
    V, theta, f, Phi, psi0, k, h = a
    r = run(BODY, V, theta, f, Phi, 0.0, psi0, passive_feather=(k, 0.3, 0.25, h))
    return a, round(r["Fz"] / W, 3), round(r["Fx"], 3), round(r["aero_power_W"], 1), \
        round(r["stroke_torque_peak"], 2), [round(x, 1) for x in r["feather_taken_deg"]], r["finite"]


def main():
    grid = []
    for V, theta, psi0 in ((0.0, -60.0, 0.0), (8.0, 0.0, -8.0)):
        for k in (0.6, 2.4):
            for h in (0.0, 0.10, 0.15, 0.20):
                grid.append((V, theta, 4.0, 85.0, psi0, k, h))
            for h in (0.10, 0.15, 0.20):                  # the predicted equivalent at h = 0
                grid.append((V, theta, 4.0, 85.0, psi0, round(k * 0.25 / (0.25 - h), 3), 0.0))
        grid.append((V, theta, 4.0, 85.0, psi0, 2.4, 0.35))   # behind the load point
    with mp.get_context("spawn").Pool(6) as pool:
        for row in pool.map(job, grid, chunksize=1):
            print(*row)


if __name__ == "__main__":
    main()
