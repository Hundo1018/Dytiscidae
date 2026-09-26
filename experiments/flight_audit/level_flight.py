"""Does any gait fly level at all?  An existence check with ideal kinematics.

Quasi-static, prescribed kinematics (as `thrust_margin`): for random gaits,
airspeeds (3-30 m/s) and pitch attitudes (-4-44 deg), the cycle-mean force on
the machine.  Level flight needs both

    <Fz> >= W          and          <Fx> >= 0

so the margin is  min(<Fz>/W, 1 + <Fx>/W)  -- >= 1 is a gait, speed and
attitude at which the machine can hold height and speed.  `thrust_margin` asks
only the second question, and its best gannet gait pulls the machine *down*
with 0.39 of its weight (fixed_rig.py).
"""
import math
import sys
from pathlib import Path
import numpy as np
from dytiscidae.control.cpg import CPGParams
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

sys.path.insert(0, "experiments/flight_audit")
from feather_fly import feathered  # noqa: E402

PHASES = 16


def forces(env, base, v, pitch):
    mj, m, d = env._mj, env.model, env.data
    jid = np.asarray(m.actuator_trnid[:, 0], int)
    qadr = np.asarray(m.jnt_qposadr, int)[jid]
    dadr = np.asarray(m.jnt_dofadr, int)[jid]
    dt = 1.0 / (base.frequency * PHASES)
    env.cpg.reset()
    cmds = [np.asarray(env.cpg.command(base, k * dt), float) for k in range(PHASES)]
    x0, y0, z0 = env.SPAWN[Domain.AIR]
    F = np.zeros(3)
    for k in range(PHASES):
        mj.mj_resetData(m, d)
        d.qpos[:3] = (x0, y0, z0)
        d.qpos[3:7] = (math.cos(-pitch / 2), 0.0, math.sin(-pitch / 2), 0.0)
        d.qvel[0] = v
        d.qpos[qadr] = cmds[k]
        d.qvel[dadr] = (cmds[(k + 1) % PHASES] - cmds[(k - 1) % PHASES]) / (2 * dt)
        env.solver.reset()
        mj.mj_forward(m, d)
        d.xfrc_applied[:] = 0.0
        with env.solver.steady():
            env.solver.apply(d, 0.0)
        f = d.xfrc_applied[:, :3].sum(0)
        f[2] -= env.solver.diag.added_mass * 9.80665
        F += f / PHASES
    return F


def search(genome, n, rng, label):
    env = TriphibianEnv(build(genome), seed=0)
    W = float(env.solver._dry_mass.sum() * 9.80665)
    lo, hi = env.cpg.lo, env.cpg.hi
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    best = (-np.inf, None)
    vals = []
    for _ in range(n):
        nn = env.cpg.n
        p = CPGParams(rng.uniform(0, 1, nn) * half, rng.uniform(0, 2 * np.pi, nn),
                      mid + rng.uniform(-0.3, 0.3, nn) * half * 2, float(rng.uniform(1.5, 12)))
        v = float(rng.uniform(3, 30)); pitch = math.radians(float(rng.uniform(-4, 44)))
        F = forces(env, p, v, pitch)
        mgn = min(F[2] / W, 1 + F[0] / W)
        vals.append(mgn)
        if mgn > best[0]:
            best = (mgn, (p.frequency, v, math.degrees(pitch), F[0] / W, F[2] / W))
            best_p = (p, v, pitch)
    vals = np.array(vals)
    import pickle
    Path("runs/_logs").mkdir(exist_ok=True)
    with open(f"runs/_logs/level_best_{label.replace(' ', '_')}.pkl", "wb") as fh:
        pickle.dump({"params": best_p[0], "v": best_p[1], "pitch": best_p[2], "margin": best[0]}, fh)
    f, v, pd, fx, fz = best[1]
    print(f"{label:22s} W {W:6.1f} N  best margin {best[0]:+.3f} (f {f:.1f} Hz, V {v:.1f} m/s, pitch {pd:.0f} deg, "
          f"Fx/W {fx:+.3f}, Fz/W {fz:+.3f})  p90 {np.percentile(vals, 90):+.3f}  share>=1 {np.mean(vals >= 1):.3f}", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rng = np.random.default_rng(20260926)
    for name in ("gannet", "teal", "beetle"):
        search(BODY_PLANS[name](), n, rng, f"{name} heave-only")
        search(feathered(name), n, rng, f"{name} feathering")
