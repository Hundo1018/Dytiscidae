"""F0: the hover force map of a prescribed flapping wing (PREREG_F0.md).

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/hover_map.py OUT.jsonl --jobs 8
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
RHO_AIR = 1.225


def hand_estimate(R, c, f, Phi, Psi, n_t=400, n_r=40) -> float:
    """Blade elements, Dickinson's translational lift only, both wings, N."""
    t = (np.arange(n_t) + 0.5) / n_t / f
    w = 2 * math.pi * f
    dphi = 0.5 * Phi * w * np.cos(w * t)
    alpha = np.pi / 2 - np.abs(Psi * np.cos(w * t))
    cl = 0.225 + 1.58 * np.sin(np.radians(2.13 * np.degrees(alpha) - 7.2))
    r = (np.arange(n_r) + 0.5) / n_r * R
    q = 0.5 * RHO_AIR * (dphi[:, None] * r[None, :]) ** 2
    return float(2.0 * (q * cl[:, None] * c * (R / n_r)).sum(1).mean())


def rig(args) -> dict:
    R, f, Phi_deg, Psi_deg, sign, dt_scale, arm = args
    import dytiscidae.core.phenotype as PH
    PH.AVIONICS_MASS = 0.005
    from body import flapper
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    c = R / 3.0
    Phi, Psi = math.radians(Phi_deg), math.radians(Psi_deg)
    dt = min(1e-3, 1.0 / (60.0 * f)) * dt_scale
    p = build(flapper(wing_length=R, wing_chord=c, stroke_range=1.45))
    env = TriphibianEnv(p, seed=0, timestep=dt)
    env.reset(Domain.AIR, randomise=False)
    d, m = env.data, env.model
    q0 = d.qpos[:7].copy()
    q0[:3] = (0.0, 0.0, 30.0)
    q0[3:7] = (math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0)   # nose up
    qadr, vadr = env._act_qadr, env._act_vadr
    names = env.act_names
    i_s = [k for k, a in enumerate(names) if a.endswith("_a")]
    i_f = [k for k, a in enumerate(names) if a.endswith("_f")]
    w = 2 * math.pi * f
    n_beat = int(round(1.0 / (f * dt)))
    F = []
    t = 0.0
    for k in range(8 * n_beat):
        tgt, rate = np.zeros(len(names)), np.zeros(len(names))
        if arm != "still":
            tgt[i_s], rate[i_s] = 0.5 * Phi * math.sin(w * t), 0.5 * Phi * w * math.cos(w * t)
            if arm != "broadside":
                tgt[i_f], rate[i_f] = sign * Psi * math.cos(w * t), -sign * Psi * w * math.sin(w * t)
        d.qpos[:7] = q0
        d.qvel[:6] = 0.0
        d.qpos[qadr], d.qvel[vadr] = tgt, rate
        env.step(tgt)
        d.qpos[qadr] = tgt
        t += dt
        if k >= 3 * n_beat:
            Fk = d.xfrc_applied[:, :3].sum(0).copy()
            Fk[2] -= (m.body_mass.sum() - env.solver._dry_mass.sum()) * G
            F.append(Fk)
    F = np.array(F)
    mean = F.mean(0)
    wing_mass = float(sum(s.mass for s in p.segments if s.part.kind == "wing"))
    return {"R": R, "chord": c, "f": f, "Phi_deg": Phi_deg, "Psi_deg": Psi_deg, "sign": sign,
            "dt": dt, "arm": arm, "Fz": float(mean[2]), "Fx": float(mean[0]), "Fy": float(mean[1]),
            "Fz_std_over_beats": float(F[:, 2].reshape(5, -1).mean(1).std()),
            "lift_gf": float(mean[2] / G * 1000.0),
            "hand_N": hand_estimate(R, c, f, Phi, Psi),
            "mass_kg": float(p.mass), "wing_pair_mass_kg": wing_mass,
            "finite": bool(np.isfinite(F).all())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    jobs = []
    # the sign, the inert arms and the time step, at one point and at the fastest
    for s in (1.0, -1.0):
        jobs.append((0.20, 12.0, 130.0, 45.0, s, 1.0, "flap"))
    jobs += [(0.20, 12.0, 130.0, 45.0, 1.0, 1.0, "still"), (0.20, 12.0, 130.0, 45.0, 1.0, 1.0, "broadside")]
    for R in (0.10, 0.20, 0.30):
        for s in (1.0, -1.0):
            jobs.append((R, 25.0, 130.0, 45.0, s, 0.5, "flap"))
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        head = pool.map(rig, jobs, chunksize=1)
        up = max((r for r in head[:2]), key=lambda r: r["Fz"])["sign"]
        grid = [(R, f, Phi, Psi, up, 1.0, "flap") for R in (0.10, 0.15, 0.20, 0.25, 0.30)
                for f in (5.0, 8.0, 12.0, 16.0, 20.0, 25.0) for Phi in (100.0, 130.0, 160.0)
                for Psi in (35.0, 45.0, 55.0)]
        rows = head + pool.map(rig, grid, chunksize=2)
    with open(Path(a.out).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("lifting sign", up, "rows", len(rows), "non-finite", sum(not r["finite"] for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
