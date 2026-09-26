"""ROADMAP AH: what would let real actuation deliver the teal's level gait?

The teal's level-flight gait (level_flight.py) on the fixed rig with free
joints, under: its own motors; motors 2x and 3x as heavy (the motor_mass gene:
more torque, more weight -- the margin counts both); a series spring at the
gait's own offset, tuned to the gait frequency; the same with a compliant drive.
Pre-registered: whatever clears margin 1 with its weight counted is a direction
the search can take with genes it already has; nothing else gets built.
"""
import math
import pickle
import sys
import numpy as np
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

sys.path.insert(0, "experiments/flight_audit")
from feather_fly import feathered  # noqa: E402


def rig(motor=1.0, spring=None, compliance=None, seconds=3.0):
    b = pickle.load(open("runs/_logs/level_best_teal_feathering.pkl", "rb"))
    p, v, pitch = b["params"], b["v"], b["pitch"]
    g = feathered("teal")
    for part in g.parts:
        if part.actuated and part.joint != "none":
            part.motor_mass *= motor
            if compliance is not None:
                part.drive_compliance = compliance
            if spring is not None:
                part.series_stiffness = spring
    if spring is not None:
        g.flap_frequency = float(p.frequency)
    env = TriphibianEnv(build(g), seed=0)
    env.reset(Domain.AIR, randomise=False)
    d, m = env.data, env.model
    if spring is not None:
        # rest angle at the gait's own offset, not mid-range
        for k, name in enumerate(env.act_names):
            j = m.actuator_trnid[k, 0]
            m.qpos_spring[m.jnt_qposadr[j]] = p.offset[k]
    q0 = d.qpos[:7].copy()
    q0[3:7] = (math.cos(-pitch / 2), 0.0, math.sin(-pitch / 2), 0.0)
    W = float(env.solver._dry_mass.sum() * 9.80665)
    n = int(seconds / env.timestep)
    F, sat = np.zeros(3), []
    env.cpg.reset()
    for k in range(n):
        d.qpos[:7] = q0; d.qpos[0] = q0[0] + v * d.time
        d.qvel[:6] = 0.0; d.qvel[0] = v
        env.step(env.cpg.command(p, d.time))
        if k > n // 2:
            f = d.xfrc_applied[:, :3].sum(0)
            f[2] -= (m.body_mass.sum() - env.solver._dry_mass.sum()) * 9.80665
            F += f / (n - n // 2 - 1)
            sat.append(np.mean(np.abs(d.actuator_force) >= 0.999 * m.actuator_forcerange[:, 1]))
    return min(F[2] / W, 1 + F[0] / W), F[0] / W, F[2] / W, W, float(np.mean(sat))


if __name__ == "__main__":
    for label, kw in [("own motors", {}), ("motors x2", {"motor": 2.0}), ("motors x3", {"motor": 3.0}),
                      ("spring at offset", {"spring": 1.0}), ("spring at offset, compliant", {"spring": 1.0, "compliance": 0.3}),
                      ("motors x2 + spring at offset", {"motor": 2.0, "spring": 1.0})]:
        mg, fx, fz, W, sat = rig(**kw)
        print(f"{label:30s} margin {mg:+.3f}  Fx/W {fx:+.3f}  Fz/W {fz:+.3f}  W {W:.1f} N  torque-limited {sat:.0%}", flush=True)
