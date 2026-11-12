"""F0b: hover, slow flight, forward flight and glide of one wing (PREREG_F0b.md).

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/flight_modes.py OUT.jsonl --jobs 8
"""
import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
G = 9.80665


def rig(args) -> dict:
    R, V, theta_deg, f, Phi_deg, Psi_deg, psi0_deg = args
    import dytiscidae.core.phenotype as PH
    PH.AVIONICS_MASS = 0.005
    from body import flapper
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    c = R / 3.0
    Phi, Psi, psi0, th = (math.radians(x) for x in (Phi_deg, Psi_deg, psi0_deg, theta_deg))
    flap = f > 0.0
    dt = min(1e-3, 1.0 / (50.0 * f)) if flap else 1e-3
    p = build(flapper(wing_length=R, wing_chord=c, stroke_range=1.45))
    env = TriphibianEnv(p, seed=0, timestep=dt)
    env.reset(Domain.AIR, randomise=False)
    d, m = env.data, env.model
    q0 = d.qpos[:7].copy()
    q0[:3] = (0.0, 0.0, 30.0)
    q0[3:7] = (math.cos(-th / 2), 0.0, math.sin(-th / 2), 0.0)
    qadr, vadr = env._act_qadr, env._act_vadr
    names = env.act_names
    i_s = [k for k, a in enumerate(names) if a.endswith("_a")]
    i_f = [k for k, a in enumerate(names) if a.endswith("_f")]
    jid = [int(m.actuator_trnid[env._mj.mj_name2id(m, env._mj.mjtObj.mjOBJ_ACTUATOR, a), 0])
           for a in names]
    w = 2 * math.pi * f
    n_beat = int(round(1.0 / (f * dt))) if flap else int(round(0.2 / dt))
    weight = float(env.solver._dry_mass.sum() * G)
    F, P = [], []
    t = 0.0
    for k in range(5 * n_beat):
        tgt, rate = np.zeros(len(names)), np.zeros(len(names))
        tgt[i_f] = psi0
        if flap:
            tgt[i_s], rate[i_s] = 0.5 * Phi * math.sin(w * t), 0.5 * Phi * w * math.cos(w * t)
            tgt[i_f], rate[i_f] = psi0 - Psi * math.cos(w * t), Psi * w * math.sin(w * t)
        d.qpos[:7] = q0
        d.qpos[0] = q0[0] + V * t
        d.qvel[:6] = 0.0
        d.qvel[0] = V
        d.qpos[qadr], d.qvel[vadr] = tgt, rate
        env.step(tgt)
        d.qpos[qadr] = tgt
        t += dt
        if k >= 2 * n_beat:
            Fk = d.xfrc_applied[:, :3].sum(0).copy()
            Fk[2] -= (m.body_mass.sum() - env.solver._dry_mass.sum()) * G
            F.append(Fk)
            pw = 0.0
            for a, j in enumerate(jid):
                b = int(m.jnt_bodyid[j])
                tau = float(d.xaxis[j] @ (d.xfrc_applied[b, 3:]
                                          + np.cross(d.xipos[b] - d.xanchor[j], d.xfrc_applied[b, :3])))
                pw -= tau * rate[a]
            P.append(pw)
    F = np.array(F).mean(0)
    Iw = float(sum(s.mass for s in p.segments if s.part.kind == "wing")) / 2.0 * (R ** 2) / 3.0
    return {"R": R, "V": V, "theta_deg": theta_deg, "f": f, "Phi_deg": Phi_deg, "Psi_deg": Psi_deg,
            "psi0_deg": psi0_deg, "Fz": float(F[2]), "Fx": float(F[0]), "Fy": float(F[1]),
            "weight": weight, "lift_over_weight": float(F[2] / weight),
            "aero_power_W": float(np.mean(P)), "mass_kg": float(p.mass),
            "peak_inertial_torque_Nm": float(Iw * 0.5 * Phi * w * w) if flap else 0.0,
            "finite": bool(np.isfinite(F).all())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    grid = [(R, V, th, f, Phi, Psi, psi0) for R in (0.20, 0.25) for V in (0.0, 3.0, 6.0, 9.0)
            for th in (0.0, 20.0, 45.0, 70.0, 90.0) for f in (5.0, 8.0, 12.0)
            for Phi in (80.0, 130.0) for Psi in (20.0, 45.0) for psi0 in (-15.0, -5.0, 5.0, 15.0)]
    grid += [(0.20, 0.0, 90.0, 12.0, 130.0, 45.0, 0.0)]                      # F0's point
    glide = [(R, V, 0.0, 0.0, 0.0, 0.0, s * i) for R in (0.20, 0.25)
             for V in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0) for i in (0.0, 4.0, 8.0, 12.0, 16.0, 20.0)
             for s in (1.0, -1.0) if not (i == 0.0 and s < 0)]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        rows = pool.map(rig, grid + glide, chunksize=4)
    with open(Path(a.out).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows), "non-finite", sum(not r["finite"] for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
