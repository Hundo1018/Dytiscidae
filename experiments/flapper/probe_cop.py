"""Probe for pass 1h: the passive feather at a few hinge axes with the
centre-of-pressure laws on.  Rest, 4 Hz, stroke 85 deg, body -60 deg; and
8 m/s, psi0 -8 deg.  Not a measurement: it informs PREREG_L1h_cop.md.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_cop.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def job(a):
    from l1a import BODY, W
    from rig import run
    V, th, psi0, k, h, law = a
    r = run(BODY, V, th, 4.0, 85.0, 0.0, psi0, passive_feather=(k, 0.3, 0.25, h), cop_law=law)
    return a, round(r["Fz"] / W, 2), round(r["Fx"], 2), round(r["aero_power_W"]), \
        round(r["stroke_torque_peak"], 1), [round(x) for x in r["feather_taken_deg"]], r["finite"]


def main():
    grid = [(V, th, p0, 1.2, h, law) for (V, th, p0) in ((0.0, -60.0, 0.0), (8.0, 0.0, -8.0))
            for law in (None, "plate", "insect") for h in (0.0, 0.2, 0.3, 0.4)]
    with mp.get_context("spawn").Pool(6) as pool:
        for row in pool.map(job, grid, chunksize=1):
            print(*row)


if __name__ == "__main__":
    main()
