"""Probe for pass S1: the stroke torque through one beat on both rigs (wing
0.4, still water and 0.3 m/s, stroke 85 deg, 0.5 Hz, feather 30).  Not a
measurement.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_entrain2.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    from freerig import run_free
    from l1a import BODY, LENGTH
    from rig import run
    body = dict(BODY, wing_length=0.4 * LENGTH)
    for V in (0.0, 0.3):
        rows = {}
        for name, kw in (("held", dict(solver_flags={"entrainment": False})),
                         ("held+ref", dict(added_mass_ref=True))):
            r = run(body, V, 0.0, 0.5, 85.0, 30.0, 0.0, beats=3, skip=2, medium="water", **kw)
            rows[name] = (np.array(r["time_series"]), np.array(r["stroke_torque_series"]),
                          np.array(r["power_series"]), r["Fx"])
        r = run_free(body, V, 0.0, 0.5, 85.0, 30.0, 0.0, medium="water", hold_beats=2, beats=1,
                     solver_flags={"entrainment": "wetting"}, keep_series=True)
        s = r["series"]
        rows["free"] = (np.array(s["t"]), np.array(s["tq"]), np.array(s["P"]), r["Fx"])
        print("V", V)
        for name, (t, tq, P, fx) in rows.items():
            ph = (t * 0.5) % 1.0
            cells = []
            for a in np.arange(0.0, 1.0, 0.125):
                i = int(np.argmin(np.abs(ph - a)))
                cells.append("%5.1f/%+6.1f" % (tq[i], P[i]))
            print("  %-9s Fx %+6.2f P %6.2f peak %5.1f | torque/power at phase 0..7/8: %s" % (
                name, fx, P.mean(), tq.max(), " ".join(cells)))


if __name__ == "__main__":
    main()
