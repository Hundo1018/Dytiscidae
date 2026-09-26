"""ROADMAP H/AB: the flapping actuator on a fixed rig.

The airframe is held at its own trim speed and attitude (root pose and velocity
rewritten every step); the joints run their real dynamics -- servo, torque
limit, inertia, the unsteady fluid -- through the gait.  Measured over the last
two of four seconds: the mean fluid force on the whole machine along and
across the flight path, against the same machine with its joints held at the
gait's mean angles.  So

    dynamic thrust margin = (<Fx>_flapping - <Fx>_held) / drag_held

is the quasi-static `thrust_margin` with everything the rollout adds except the
airframe's freedom to move.  If it clears 1 and free flight does not, the wall
is stability and control; if it does not, the wall is force.
"""
import sys
import numpy as np
from dytiscidae.control.cpg import CPGParams
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

sys.path.insert(0, "experiments/flight_audit")
from feather_fly import feathered  # noqa: E402


def rig(genome, params, seconds=4.0, hold=False, unsteady=True, inflow=True, kinematic=False):
    env = TriphibianEnv(build(genome), seed=0)
    env.solver.unsteady = unsteady
    env.solver.inflow = env.solver.inflow and inflow
    v, pitch = env._trim()[:2]
    env.reset(Domain.AIR, randomise=False)
    d, m = env.data, env.model
    q0 = d.qpos[:7].copy()
    q0[3:7] = (np.cos(-pitch / 2), 0.0, np.sin(-pitch / 2), 0.0)
    fx, fz, reach = [], [], []
    mean = params.offset.copy()
    n = int(seconds / env.timestep)
    qadr = env._act_qadr
    for k in range(n):
        d.qpos[:7] = q0
        d.qpos[0] = q0[0] + v * d.time
        d.qvel[:6] = 0.0
        d.qvel[0] = v
        tgt = mean if hold else env.cpg.command(params, d.time)
        if kinematic:
            # Prescribed: the joints are where the command says, at its rate.
            rate = env.cpg.pop_rate() if not hold else np.zeros_like(tgt)
            d.qpos[qadr] = tgt
            d.qvel[env._act_vadr] = rate if rate is not None else 0.0
        env.step(tgt)
        if kinematic:
            d.qpos[qadr] = tgt
        if k > n // 2:
            F = d.xfrc_applied[:, :3].sum(0)
            F[2] -= (m.body_mass.sum() - env.solver._dry_mass.sum()) * 9.80665
            fx.append(F[0]); fz.append(F[2])
            reach.append(d.qpos[qadr].copy())
    reach = np.array(reach)
    stroke = (reach.max(0) - reach.min(0)) / np.maximum(2 * params.amplitude, 1e-9)
    weight = float(env.solver._dry_mass.sum() * 9.80665)
    return float(np.mean(fx)), float(np.mean(fz)), weight, float(np.median(stroke[params.amplitude > 1e-3])) if (params.amplitude > 1e-3).any() else 1.0


def best_gaits(name, k=3, n=300):
    rng = np.random.default_rng(20260923)
    order = ("gannet", "teal", "beetle", "bat")
    out = None
    for nm in order:
        g = feathered(nm)
        env = TriphibianEnv(build(g), seed=0)
        lo, hi = env.cpg.lo, env.cpg.hi
        half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
        found = []
        for _ in range(n):
            nn = env.cpg.n
            p = CPGParams(rng.uniform(0, 1, nn) * half, rng.uniform(0, 2 * np.pi, nn),
                          mid + rng.uniform(-0.3, 0.3, nn) * half * 2, float(rng.uniform(1.5, 12)))
            env.cpg.base = p
            if hasattr(env.p, "_measured_thrust"):
                del env.p._measured_thrust
            tm = env.thrust_margin()
            if tm is not None:
                found.append((tm, p))
        if nm == name:
            found.sort(key=lambda x: -x[0])
            return found[:k]
    return out


if __name__ == "__main__":
    for name in sys.argv[1:2] or ["gannet"]:
        for tm, p in best_gaits(name, k=int(sys.argv[2]) if len(sys.argv) > 2 else 3):
            g = feathered(name)
            for us, inf, kin in ((False, False, True), (False, False, False), (True, True, True), (True, True, False)):
                fx_h, fz_h, W, _ = rig(g, p, hold=True, unsteady=us, inflow=inf, kinematic=kin)
                fx_f, fz_f, W, st = rig(g, p, unsteady=us, inflow=inf, kinematic=kin)
                drag = -fx_h
                print(f"{name} qs tm {tm:+.2f} @ {p.frequency:.2f} Hz unsteady={us:d} inflow={inf:d} prescribed={kin:d} | held Fx {fx_h:+.2f} | "
                      f"flapping Fx {fx_f:+.2f} Fz {fz_f:+.2f} N | W {W:.1f} | dynamic tm "
                      f"{(fx_f - fx_h) / max(drag, 1e-9):+.2f} | lift/W {fz_f / W:.2f} | stroke {st:.2f}", flush=True)
