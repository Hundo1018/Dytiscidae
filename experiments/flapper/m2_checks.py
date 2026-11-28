"""Pass M2: checks C0, C8, C9 (PREREG_M2_water.md section 4).

    m2_checks.py c0                  # the exit arms' gains: held take-off lift against M0's 1.840
    (C8, C9 are rows of m2_run.py with an `extra`: m2_run.py checks)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: C8 and C9 run on the best swim arm (one seed pair) and the best exit arm (one seed);
#: filled when those are known from the scored rows (section 4 names them "best").
BEST_SWIM = ("WS", 0)     # the swim arm read first at the surface; set before any scored row
BEST_EXIT = ("XV", 0)


def jobs_checks(only=None):
    J = []
    for arm, seed in [x for x in (BEST_SWIM,) if x]:
        halves = ("A", "B")
        for extra in (dict(dt=5e-5), dict(motors=dict(stroke=dict(J_r=5e-6), feather=dict(J_r=5e-6))),
                      dict(motors=dict(stroke=dict(J_r=1e-6), feather=dict(J_r=1e-6))), dict(entrainment=True)):
            for h in halves:
                J.append((arm, seed, h, extra))
    for arm, seed in [x for x in (BEST_EXIT,) if x]:
        for extra in (dict(motors=dict(stroke=dict(J_r=5e-6), feather=dict(J_r=5e-6))),
                      dict(motors=dict(stroke=dict(J_r=1e-6), feather=dict(J_r=1e-6))), dict(entrainment=True),
                      dict(dt=5e-5)):          # dt 5e-5 on the exit: ADDED (the strike is the fastest event)
            J.append((arm, seed, "X", extra))
    if only:
        J = [j for j in J if j[0] in only]
    return J


def c0():
    import numpy as np
    from m2_common import build, draw
    dr = draw(0)
    mc = build("float", dr, gains="air")
    mc.place(z=30.0, pitch_deg=90.0, vx=0.0)
    mc.d.qvel[:] = 0.0
    mc.env.solver.reset()
    mc.mj.mj_forward(mc.m, mc.d)
    f, Phi, Psi = 4.5, math.radians(85), math.radians(50)
    w = 2 * math.pi * f
    q0 = mc.d.qpos[:7].copy()
    nb = int(round(1 / (f * mc.dt)))
    st, fe = mc.role == "stroke", mc.role == "feather"
    az = []
    for k in range(6 * nb):
        t = mc.t
        q = np.zeros(len(mc.aid)); qd = np.zeros(len(mc.aid))
        q[st] = Phi / 2 * math.sin(w * t); qd[st] = Phi / 2 * w * math.cos(w * t)
        q[fe] = -Psi * math.cos(w * t); qd[fe] = Psi * w * math.sin(w * t)
        vb = mc.d.qvel[:3].copy()
        mc.step(q, qd)
        if k >= 2 * nb:
            az.append((mc.d.qvel[:3] - vb) / mc.dt)
        mc.d.qpos[:7] = q0
        mc.d.qvel[:6] = 0.0
    a = np.mean(az, axis=0)
    lift = (a[2] + 9.80665) / 9.80665
    row = dict(kind="C0", lift_weights=float(lift), m0=1.840, rel=float(lift / 1.840 - 1), ok=bool(abs(lift / 1.840 - 1) <= 0.02),
               kp=mc.kp.tolist(), bad_qacc=mc.bad_qacc)
    with open(HERE / "results" / "M2_checks.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(row)


if __name__ == "__main__":
    if sys.argv[1] == "c0":
        c0()
