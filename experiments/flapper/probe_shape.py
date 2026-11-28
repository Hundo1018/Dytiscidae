"""Probe: what the fluid model makes of a wing's shape.  Pass 1i's cheapest
cruise point (8 m/s, 3.5 Hz, stroke 60, feather 20, axis 0.2, "plate"), the
cam offset re-trimmed to the weight for each shape, on the held rig.  Not a
measurement.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_shape.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
SHAPES = [("flat rectangle", None), ("camber 4%", dict(camber=0.04)), ("camber 8%", dict(camber=0.08)),
          ("taper 0.5", dict(taper=0.5)), ("taper 0.3", dict(taper=0.3)),
          ("washout 8 deg", dict(twist_tip_deg=-8.0)), ("washin 8 deg", dict(twist_tip_deg=8.0)),
          ("taper 0.5, camber 6%, washout 8", dict(taper=0.5, camber=0.06, twist_tip_deg=-8.0))]


def job(a):
    import l1a
    name, shape, V, f = a
    l1a.BODY = dict(l1a.BODY_S, wing_shape=shape)
    from l1i_cam import job as cruise
    r = cruise(("cruise", V, f, 60.0, 20.0, 0.2, "plate"))
    return "%-34s V %.0f f %.1f: trimmed %s offset %+6.2f deg, net force %+.2f N, power %.1f W, crank %.1f N m" % (
        name, V, f, r["trimmed"], r["psi0_deg"], r["Fx"], r["aero_power_W"], r["crank_eq_peak"])


def main():
    grid = [(n, s, V, f) for V, f in ((8.0, 3.5), (6.0, 3.0)) for n, s in SHAPES]
    with mp.get_context("spawn").Pool(4) as pool:
        for row in pool.map(job, grid, chunksize=1):
            print(row)


if __name__ == "__main__":
    main()
