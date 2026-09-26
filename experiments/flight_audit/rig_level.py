"""The teal's level-flight gait (level_flight.py's best) on the fixed rig.

Held at the gait's own speed and attitude; four seconds; the last two averaged.
Prescribed kinematics against free joints, the level-flight margin
min(<Fz>/W, 1 + <Fx>/W), and for each joint the share of time its motor is at
its torque limit and the lag of the reached angle behind the command.
"""
import math
import pickle
import sys
import numpy as np
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

sys.path.insert(0, "experiments/flight_audit")
from feather_fly import feathered  # noqa: E402


def run(name, label, prescribed, seconds=4.0, compliance=None, spring=None):
    b = pickle.load(open(f"runs/_logs/level_best_{label}.pkl", "rb"))
    p, v, pitch = b["params"], b["v"], b["pitch"]
    g = feathered(name) if "feather" in label else __import__("dytiscidae.core.bodyplans", fromlist=["x"]).BODY_PLANS[name]()
    if compliance is not None:
        for part in g.parts:
            part.drive_compliance = compliance
    if spring is not None:
        # Series elasticity tuned to the gait's own frequency (mjcf: k = ratio I w^2
        # with w from genome.flap_frequency).
        g.flap_frequency = float(p.frequency)
        for part in g.parts:
            if part.actuated and part.joint != "none":
                part.series_stiffness = spring
    env = TriphibianEnv(build(g), seed=0)
    env.reset(Domain.AIR, randomise=False)
    d, m = env.data, env.model
    q0 = d.qpos[:7].copy()
    q0[3:7] = (math.cos(-pitch / 2), 0.0, math.sin(-pitch / 2), 0.0)
    W = float(env.solver._dry_mass.sum() * 9.80665)
    n = int(seconds / env.timestep)
    qadr, vadr = env._act_qadr, env._act_vadr
    F, sat, cmd_hist, got_hist = [], [], [], []
    env.cpg.reset()
    for k in range(n):
        d.qpos[:7] = q0; d.qpos[0] = q0[0] + v * d.time
        d.qvel[:6] = 0.0; d.qvel[0] = v
        tgt = env.cpg.command(p, d.time)
        if prescribed:
            rate = env.cpg.pop_rate()
            d.qpos[qadr] = tgt; d.qvel[vadr] = rate
        env.step(tgt)
        if prescribed:
            d.qpos[qadr] = tgt
        if k > n // 2:
            f = d.xfrc_applied[:, :3].sum(0)
            f[2] -= (m.body_mass.sum() - env.solver._dry_mass.sum()) * 9.80665
            F.append(f)
            sat.append(np.abs(d.actuator_force) >= 0.999 * m.actuator_forcerange[:, 1])
            cmd_hist.append(tgt.copy()); got_hist.append(d.qpos[qadr].copy())
    F = np.mean(F, 0)
    margin = min(F[2] / W, 1 + F[0] / W)
    C, G = np.array(cmd_hist), np.array(got_hist)
    lags = []
    w = 2 * math.pi * p.frequency
    t = np.arange(len(C)) * env.timestep
    basis = np.stack([np.sin(w * t), np.cos(w * t)], 1)
    for j in range(C.shape[1]):
        if p.amplitude[j] < 1e-3:
            continue
        a = np.linalg.lstsq(basis, C[:, j] - C[:, j].mean(), rcond=None)[0]
        bb = np.linalg.lstsq(basis, G[:, j] - G[:, j].mean(), rcond=None)[0]
        lag = (math.atan2(a[1], a[0]) - math.atan2(bb[1], bb[0]) + math.pi) % (2 * math.pi) - math.pi
        lags.append((env.act_names[j][-1], round(math.degrees(lag)), round(float(np.hypot(*bb) / max(np.hypot(*a), 1e-9)), 2)))
    print(f"{label} prescribed={prescribed:d} compliance={compliance} spring={spring}: Fx/W {F[0]/W:+.3f} Fz/W {F[2]/W:+.3f} margin {margin:+.3f} | "
          f"torque-limited {np.mean(sat):.0%} | (kind, lag deg, reach) {lags}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for spec in sys.argv[1:]:
            c, k = (float(x) if x != "none" else None for x in spec.split(","))
            run("teal", "teal_feathering", False, compliance=c, spring=k)
    else:
        run("teal", "teal_feathering", True)
        run("teal", "teal_feathering", False)
