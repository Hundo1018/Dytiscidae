"""The held-body rig of the flapper experiments, with every wing joint
prescribed (stage 1: no torque limit).

Roles are read off the built model: a shoulder's stroke hinge (`*_j` on a
body whose parent is the hull), its feather hinge (`*_jf`), and a wrist's
twist hinge (`*_j` on a body whose parent is a wing).  Both sides get the
same angles; the model mirrors the axes.

    stroke  = (Phi/2) sin(wt)
    feather = psi0 - Psi cos(wt)
    twist   = tau0 - T   cos(wt)          (added to the feather, outer panel only)
"""
import math

import numpy as np

G = 9.80665


def run(body: dict, V: float, theta_deg: float, f: float, Phi_deg: float, Psi_deg: float,
        psi0_deg: float, T_deg: float = 0.0, tau0_deg: float = 0.0, *, beats: int = 5,
        skip: int = 2, steps_per_beat: int = 50, medium: str = "air") -> dict:
    import dytiscidae.core.phenotype as PH
    PH.AVIONICS_MASS = 0.005
    from body import flapper
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    Phi, Psi, psi0, T, tau0, th = (math.radians(x) for x in
                                   (Phi_deg, Psi_deg, psi0_deg, T_deg, tau0_deg, theta_deg))
    flap = f > 0.0
    dt = min(1e-3, 1.0 / (steps_per_beat * f)) if flap else 1e-3
    p = build(flapper(**body))
    env = TriphibianEnv(p, seed=0, timestep=dt)
    dom = Domain.AIR if medium == "air" else Domain.WATER
    env.reset(dom, randomise=False)
    d, m, mj = env.data, env.model, env._mj
    q0 = d.qpos[:7].copy()
    q0[:3] = (0.0, 0.0, 30.0 if medium == "air" else -3.0)
    q0[3:7] = (math.cos(-th / 2), 0.0, math.sin(-th / 2), 0.0)
    qadr, vadr = env._act_qadr, env._act_vadr
    names = env.act_names
    hull = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "seg0_hull")
    jid, role = [], []
    for a in names:
        j = int(m.actuator_trnid[mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, a), 0])
        jid.append(j)
        if a.endswith("_f"):
            role.append("feather")
        else:
            bj = int(m.jnt_bodyid[j])
            if "wing" not in mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, bj):
                role.append("held")          # F2: a tail hinge is held at 0, not flapped
            else:
                role.append("stroke" if int(m.body_parentid[bj]) == hull else "twist")
    role = np.array(role)
    w = 2 * math.pi * f
    n_beat = int(round(1.0 / (f * dt))) if flap else int(round(0.2 / dt))
    weight = float(env.solver._dry_mass.sum() * G)
    F, P, Tq = [], [], []
    t = 0.0
    for k in range(beats * n_beat):
        tgt, rate = np.zeros(len(names)), np.zeros(len(names))
        s, c = (math.sin(w * t), math.cos(w * t)) if flap else (0.0, 0.0)
        tgt[role == "stroke"], rate[role == "stroke"] = 0.5 * Phi * s, 0.5 * Phi * w * c
        tgt[role == "feather"], rate[role == "feather"] = psi0 - Psi * c, Psi * w * s
        tgt[role == "twist"], rate[role == "twist"] = tau0 - T * c, T * w * s
        d.qpos[:7] = q0
        d.qpos[0] = q0[0] + V * t
        d.qvel[:6] = 0.0
        d.qvel[0] = V
        d.qpos[qadr], d.qvel[vadr] = tgt, rate
        env.step(tgt)
        d.qpos[qadr] = tgt
        t += dt
        if k >= skip * n_beat:
            Fk = d.xfrc_applied[:, :3].sum(0).copy()
            Fk[2] -= (m.body_mass.sum() - env.solver._dry_mass.sum()) * G
            F.append(Fk)
            # Aerodynamic power: every wing body's fluid wrench about each
            # prescribed axis that carries it, times that axis's rate.
            pw, tq = 0.0, 0.0      # tq: fluid torque about the stroke axes, all shoulders
            for a, j in enumerate(jid):
                if rate[a] == 0.0:
                    continue
                b0 = int(m.jnt_bodyid[j])
                tau = 0.0
                for b in range(1, m.nbody):
                    bb = b
                    while bb != 0 and bb != b0:
                        bb = int(m.body_parentid[bb])
                    if bb == b0:
                        tau += float(d.xaxis[j] @ (d.xfrc_applied[b, 3:] + np.cross(
                            d.xipos[b] - d.xanchor[j], d.xfrc_applied[b, :3])))
                pw -= tau * rate[a]
                if role[a] == "stroke":
                    tq += abs(tau)
            P.append(pw)
            Tq.append(tq)
    F = np.array(F).mean(0)
    return {"V": V, "theta_deg": theta_deg, "f": f, "Phi_deg": Phi_deg, "Psi_deg": Psi_deg,
            "psi0_deg": psi0_deg, "T_deg": T_deg, "tau0_deg": tau0_deg, "medium": medium,
            "Fz": float(F[2]), "Fx": float(F[0]), "Fy": float(F[1]), "weight": weight,
            "lift_over_weight": float(F[2] / weight), "aero_power_W": float(np.mean(P)),
            "stroke_torque_mean": float(np.mean(Tq)), "stroke_torque_peak": float(np.max(Tq)),
            "mass_kg": float(p.mass), "roles": sorted(set(role.tolist())),
            "finite": bool(np.isfinite(F).all())}


def trim(body: dict, V: float, f: float, Phi: float, Psi: float, T: float = 0.0,
         lo: float = -18.0, hi: float = 0.0, tol: float = 0.03, max_evals: int = 4) -> dict:
    """The feather offset at which mean lift is the weight, by the secant rule."""
    x0, x1 = -8.0, -12.0
    r0, r1 = run(body, V, 0.0, f, Phi, Psi, x0, T), run(body, V, 0.0, f, Phi, Psi, x1, T)
    evals = 2
    best = min((r0, r1), key=lambda r: abs(r["lift_over_weight"] - 1.0))
    while abs(best["lift_over_weight"] - 1.0) > tol and evals < max_evals:
        y0, y1 = r0["lift_over_weight"] - 1.0, r1["lift_over_weight"] - 1.0
        if abs(y1 - y0) < 1e-9:
            break
        x2 = min(max(x1 - y1 * (x1 - x0) / (y1 - y0), lo), hi)
        r2 = run(body, V, 0.0, f, Phi, Psi, x2, T)
        evals += 1
        x0, r0, x1, r1 = x1, r1, x2, r2
        if abs(r2["lift_over_weight"] - 1.0) < abs(best["lift_over_weight"] - 1.0):
            best = r2
    out = dict(best)
    out["trimmed"] = bool(abs(best["lift_over_weight"] - 1.0) <= tol)
    out["evals"] = evals
    return out
