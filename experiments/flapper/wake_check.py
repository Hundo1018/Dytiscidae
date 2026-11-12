"""The wake feedback against a known case: Wagner's impulsive start.

A wing started at once at a fixed incidence sheds a starting vortex whose
downwash holds its lift down at first; the lift then rises toward the steady
value as the vortex is left behind (Wagner 1925: about 0.6 of steady after
half a chord of travel, 0.7 after one, 0.8 after two, 0.9 after five, in two
dimensions).  Here: one wing pair, the feedback acting on the pair itself,
the solver's own unsteady term off.

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/wake_check.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dytiscidae.core.phenotype as PH  # noqa: E402

PH.AVIONICS_MASS = 0.005
import wakefb  # noqa: E402
from body import flapper  # noqa: E402
from dytiscidae.core.phenotype import build  # noqa: E402
from dytiscidae.envs.triphibian import Domain, TriphibianEnv  # noqa: E402

WAGNER = {0.5: 0.62, 1.0: 0.69, 2.0: 0.78, 5.0: 0.89, 14.0: 0.95}


def run(gain: float, sign: float, V: float = 6.0, dt: float = 5e-4, steps: int = 400):
    wakefb.SIGN = sign
    env = TriphibianEnv(build(flapper(wing_length=0.25, wing_chord=0.25 / 3, stroke_range=1.45)),
                        seed=0, timestep=dt)
    env.solver.unsteady = False
    wf = wakefb.attach(env, gain, self_influence=True)
    env.reset(Domain.AIR, randomise=False)
    d = env.data
    q0 = d.qpos[:7].copy()
    q0[:3], q0[3:7] = (0.0, 0.0, 30.0), (1.0, 0.0, 0.0, 0.0)
    is_f = np.array([a.endswith("_f") for a in env.act_names])
    lift, t = [], 0.0
    for _ in range(steps):
        tgt = np.zeros(len(is_f))
        tgt[is_f] = math.radians(-8.0)
        d.qpos[:7] = q0
        d.qpos[0] = q0[0] + V * t
        d.qvel[:6] = 0.0
        d.qvel[0] = V
        d.qpos[env._act_qadr], d.qvel[env._act_vadr] = tgt, 0.0
        env.step(tgt)
        d.qpos[env._act_qadr] = tgt
        t += dt
        lift.append(float(d.xfrc_applied[:, 2].sum()))
    return np.array(lift), len(wf.pos), V * dt / (0.25 / 3)


def main() -> int:
    base, _, c = run(0.0, 1.0)
    at = {k: int(round(k / c)) - 1 for k in WAGNER}
    print("chords travelled:", list(WAGNER), "| Wagner (2-D):", list(WAGNER.values()))
    ok = True
    for sign in (1.0, -1.0):
        f, n, _ = run(1.0, sign)
        ratio = [round(float(f[i] / base[i]), 3) for i in at.values()]
        print(f"sign {sign:+.0f}: lift over lift without feedback {ratio}  ({n} segments kept)")
        if sign > 0:
            ok = ratio[0] < 1.0 and ratio[0] < ratio[2] <= 1.02 and abs(ratio[-1] - 1.0) < 0.02
    print("SIGN = +1 holds the lift down at the start and lets it recover:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
