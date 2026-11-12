"""F3: the stroke map of the wing under water (PREREG_F3.md).

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/water_map.py OUT.jsonl --jobs 6
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
BODY = dict(wing_length=0.25, wing_chord=0.25 / 3, stroke_range=1.45)


def run(args) -> dict:
    U, f, Phi_deg, Psi_deg = args
    import dytiscidae.core.phenotype as PH
    PH.AVIONICS_MASS = 0.005
    from body import flapper
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    Phi, Psi = math.radians(Phi_deg), math.radians(Psi_deg)
    flap = f > 0.0
    dt = 1e-3
    p = build(flapper(**BODY))
    env = TriphibianEnv(p, seed=0, timestep=dt)
    env.reset(Domain.WATER, randomise=False)
    d, m, mj = env.data, env.model, env._mj
    q0 = d.qpos[:7].copy()
    q0[:3] = (-30.0, 0.0, -3.0)
    q0[3:7] = (1.0, 0.0, 0.0, 0.0)
    qadr, vadr = env._act_qadr, env._act_vadr
    names = env.act_names
    jid = [int(m.actuator_trnid[mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, a), 0]) for a in names]
    is_f = np.array([a.endswith("_f") for a in names])
    w = 2 * math.pi * f
    n_beat = int(round(1.0 / (f * dt))) if flap else 500
    weight = float(env.solver._dry_mass.sum() * G)
    F, P, TQ = [], [], []
    t = 0.0
    for k in range(5 * n_beat):
        tgt, rate = np.zeros(len(names)), np.zeros(len(names))
        if flap:
            s, c = math.sin(w * t), math.cos(w * t)
            tgt[~is_f], rate[~is_f] = 0.5 * Phi * s, 0.5 * Phi * w * c
            tgt[is_f], rate[is_f] = -Psi * c, Psi * w * s
        d.qpos[:7] = q0
        d.qpos[0] = q0[0] + U * t
        d.qvel[:6] = 0.0
        d.qvel[0] = U
        d.qpos[qadr], d.qvel[vadr] = tgt, rate
        env.step(tgt)
        d.qpos[qadr] = tgt
        t += dt
        if k >= 2 * n_beat:
            Fk = d.xfrc_applied[:, :3].sum(0).copy()
            Fk[2] -= (m.body_mass.sum() - env.solver._dry_mass.sum()) * G
            F.append(Fk)
            taus, pw = [], 0.0
            for a, j in enumerate(jid):
                b = int(m.jnt_bodyid[j])
                tau = float(d.xaxis[j] @ (d.xfrc_applied[b, 3:] + np.cross(
                    d.xipos[b] - d.xanchor[j], d.xfrc_applied[b, :3])))
                taus.append(tau)
                pw -= tau * rate[a]
            TQ.append(taus)
            P.append(pw)
    F, TQ = np.array(F), np.abs(np.array(TQ))
    tip = 2.0 * BODY["wing_length"] * math.sin(0.5 * Phi)
    return {"U": U, "f": f, "Phi_deg": Phi_deg, "Psi_deg": Psi_deg,
            "Fx": float(F[:, 0].mean()), "Fz_mean": float(F[:, 2].mean()),
            "Fz_swing": float(0.5 * (F[:, 2].max() - F[:, 2].min())),
            "Fx_swing": float(0.5 * (F[:, 0].max() - F[:, 0].min())),
            "weight": weight, "power_W": float(np.mean(P)),
            "peak_stroke_torque": float(TQ[:, ~is_f].max()) if flap else 0.0,
            "peak_feather_torque": float(TQ[:, is_f].max()) if flap else 0.0,
            "strouhal": float(f * tip / U) if flap else 0.0,
            "finite": bool(np.isfinite(F).all())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    grid = [(U, 0.0, 0.0, 0.0) for U in (0.25, 0.5, 1.0)]
    grid += [(U, f, Phi, Psi) for U in (0.25, 0.5, 1.0) for f in (0.5, 1.0, 2.0, 3.0)
             for Phi in (30.0, 60.0, 100.0) for Psi in (0.0, 20.0, 40.0, 55.0)]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        rows = pool.map(run, grid, chunksize=1)
    with open(Path(a.out).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows), "non-finite", sum(not r["finite"] for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
