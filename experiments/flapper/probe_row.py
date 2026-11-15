"""Probe 2 for pass 1j: a rowing stroke.  Hull vertical (body pitch 90 deg), so
the wings sweep fore and aft with their faces to the 0.3 m/s stream; no
feather.  Welded flap against a flap free to swing either way.  Water, wing at
0.4 of its length, stroke 85 deg.  Not a measurement.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_row.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def job(a):
    from l1a import BODY, LENGTH
    from rig import run
    th, f, sign, kf = a
    body = dict(BODY, wing_length=0.4 * LENGTH)
    fl = dict(x_h=0.4, open_deg=80.0, sign=sign, k_f=kf, damp_f=0.002)
    r = run(body, 0.3, th, f, 85.0, 0.0, 0.0, beats=3, skip=1, medium="water", one_way_flap=fl)
    return a, round(r["Fx"], 2), round(r["Fz"], 2), round(r["aero_power_W"], 2), \
        round(r["stroke_torque_peak"], 1), r["flap_open_max_deg"] and [round(x) for x in r["flap_open_max_deg"]], \
        r["flap_open_mean_deg"] and [round(x) for x in r["flap_open_mean_deg"]], r["finite"]


def main():
    grid = [(90.0, f, sg, 0.005) for f in (0.0, 0.5, 1.0) for sg in (0, 1, -1)]
    grid += [(90.0, 0.5, sg, 0.05) for sg in (1, -1)]
    with mp.get_context("spawn").Pool(6) as pool:
        print("args (pitch, f, sign, k_f) | Fx | Fz | power | stroke torque | flap max | flap mean")
        for row in pool.map(job, grid, chunksize=1):
            print(*row)


if __name__ == "__main__":
    main()
