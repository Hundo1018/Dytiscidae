"""Probe for pass W3: what the wing strips see through a beat at two gate
points of pass 1n (cruise 8 m/s 3.5 Hz stroke 60 feather 20; take-off 4.5 Hz):
incidence, and the leading-edge-vortex weight that hands a strip to the
separated plate.  Not a measurement.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_lev.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    import dytiscidae.physics.fluid as FL
    from l1a import BODY_S
    from rig import run
    rec = []
    orig = FL.UnsteadyState.update

    def spy(self, alpha, rev, U, chord, omega, s_hat, dt):
        ae, lev = orig(self, alpha, rev, U, chord, omega, s_hat, dt)
        ro = FL.rossby_lev(U, chord, omega, s_hat)
        rec.append((np.degrees(np.abs(alpha[:20])), lev[:20].copy(), ro[:20].copy(), U[:20].copy()))
        return ae, lev
    FL.UnsteadyState.update = spy
    for name, args, kw in (("cruise", (8.0, 0.0, 3.5, 60.0, 20.0, -7.3), dict(feather_axis=0.2, cop_law="plate")),
                           ("cruise slow flap", (8.0, 0.0, 2.5, 40.0, 10.0, -4.0), dict(feather_axis=0.1, cop_law="plate")),
                           ("take-off", (0.0, 60.0, 4.5, 85.0, 50.0, 0.0), dict(feather_axis=0.1, cop_law="plate"))):
        rec.clear()
        r = run(BODY_S, *args, **kw)
        a, lev, ro, U = (np.array([x[i] for x in rec[len(rec) // 2:]]) for i in range(4))
        w = U ** 2                                   # weight by dynamic pressure
        f = lambda m: float((w * m).sum() / w.sum())
        print("%-17s L/W %.2f P %.1f W | share of the dynamic pressure with: lev > 0.9 %.2f, lev < 0.1 %.2f (Rossby part > 0.9: %.2f) |"
              " incidence under 5 deg %.2f, 5-12 %.2f, 12-25 %.2f, over 25 %.2f" % (
                  name, r["lift_over_weight"], r["aero_power_W"], f(lev > 0.9), f(lev < 0.1), f(ro > 0.9),
                  f(a < 5), f((a >= 5) & (a < 12)), f((a >= 12) & (a < 25)), f(a >= 25)))


if __name__ == "__main__":
    main()
