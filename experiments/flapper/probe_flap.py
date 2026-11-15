"""Probe for pass 1j: the one-way flap built as a body, at pass 1e's probe
point (water, wing at 0.4 of its length, 0.3 m/s, stroke 85, 0.5 Hz), and
the two couplings on pass 1h's cheapest passive point.  Not a measurement.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_flap.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def job(a):
    from l1a import BODY, LENGTH, W
    from rig import run
    if a[0] == "water":
        _, Psi, sign = a
        body = dict(BODY, wing_length=0.4 * LENGTH)
        fl = None if sign is None else dict(x_h=0.4, open_deg=80.0, sign=sign, k_f=0.005, damp_f=0.002)
        r = run(body, 0.3, 0.0, 0.5, 85.0, Psi, 0.0, beats=3, skip=1, medium="water", one_way_flap=fl)
        return a, round(r["Fx"], 2), round(r["Fz"], 2), round(r["aero_power_W"], 2), \
            round(r["stroke_torque_peak"], 1), r["flap_open_max_deg"] and [round(x) for x in r["flap_open_max_deg"]], r["finite"]
    _, coupling = a
    r = run(BODY, 0.0, -90.0, 3.5, 85.0, 0.0, 0.0, passive_feather=(0.3, 0.3, 0.25, 0.1),
            cop_law="plate", coupling=coupling)
    return a, round(r["Fz"] / W, 3), round(r["aero_power_W"], 1), round(r["stroke_torque_peak"], 2), \
        [round(x) for x in r["feather_taken_deg"]]


def main():
    grid = [("water", Psi, sg) for Psi in (30.0, 15.0, 0.0) for sg in (None, 0, 1, -1)]
    grid += [("air", "static"), ("air", "step")]
    with mp.get_context("spawn").Pool(6) as pool:
        for row in pool.map(job, grid, chunksize=1):
            print(*row)


if __name__ == "__main__":
    main()
