"""Probe for pass 1m: what the held body's pitching moment is made of.  Pass
1i's cheapest gate points, body held; the holder's force and pitching moment
through the beat, and the moment about a centre of mass moved fore-aft and
up-down.  Not a measurement.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_pitch.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
G = 9.80665


def job(pt):
    from freerig import run_free
    from l1a import BODY
    V, th, f, Phi, Psi, psi0, h, law = pt
    r = run_free(BODY, V, th, f, Phi, Psi, psi0, feather_axis=h, cop_law=law, keep_series=True)
    hold = np.array(r["series"]["hold"])
    m = r["mass_kg"]
    F = hold[:, :3] + np.array([0.0, 0.0, m * G])          # everything but gravity
    com = np.array(r["com0"])
    My = hold[:, 4] - np.cross(com, np.array([0.0, 0.0, -m * G]))[1]   # about the hull origin
    out = "V %.0f th %.0f f %.1f Phi %.0f Psi %.0f h %.1f %s | com (%.3f, %.3f) m | Fz %.2f..%.2f W, Fx %.1f..%.1f N" % (
        V, th, f, Phi, Psi, h, law, com[0], com[2], F[:, 2].min() / m / G, F[:, 2].max() / m / G,
        F[:, 0].min(), F[:, 0].max())
    best = None
    for dx in np.arange(-0.30, 0.3001, 0.01):
        for dz in np.arange(-0.10, 0.1001, 0.02):
            M = My - (dz * F[:, 0] - dx * F[:, 2])
            c = (np.abs(M - M.mean()).max(), dx, dz, M.mean(), np.abs(M).max())
            if best is None or c[0] < best[0]:
                best = c
    M0 = My - (com[2] * F[:, 0] - com[0] * F[:, 2])
    out += "\n    about the built centre of mass: mean %+.2f, peak %.2f N m" % (M0.mean(), np.abs(M0).max())
    out += "\n    least swing about (%.2f, %.2f) m from the hull origin: mean %+.2f, swing %.2f, peak %.2f N m" % (
        best[1], best[2], best[3], best[0], best[4])
    for dx in (-0.1, -0.05, 0.0, 0.05, 0.1):
        M = My - (com[2] * F[:, 0] - (com[0] + dx) * F[:, 2])
        out += "\n    centre of mass moved %+.2f m along x: mean %+.2f, swing %.2f" % (
            dx, M.mean(), np.abs(M - M.mean()).max())
    return out


def main():
    from l1k_free import points
    to, cr = points()
    pts = to[:2] + sorted(cr, key=lambda p: p[0])[:2] + cr[-2:]
    with mp.get_context("spawn").Pool(6) as pool:
        for row in pool.map(job, pts, chunksize=1):
            print(row)


if __name__ == "__main__":
    main()
