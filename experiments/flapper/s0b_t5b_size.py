"""Sizing of T5b's dorsal float from the built model: mass, centre of mass, righting stiffness under water
(roll, pitch; fully submerged, wings in the sink pose) at fills 0.95 / 0.5 / 0.0, for a list of dorsal floats
(radius dr, length dL, height dz in the hull frame; dr 0 = T5).  Rows -> results/S0b_T5b_size.json.

    MUJOCO_GL=disable PYTHONPATH=. python experiments/flapper/s0b_t5b_size.py "[[0,0,0],[0.025,0.45,0.10]]"
"""
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0b_machine as T
import s0b_t5_machine as T5
import s0b_t5b_machine as T5B
import s0b_laws as LW

G = 9.80665


def sink_pose(mc):
    q, _ = LW.targets_level(mc, 0.0, 1.0, dict(LW.LEVEL), osc=0.0)
    for tag in ("FL", "FR", "HL", "HR"):
        q[mc.ix[(tag, "feather")]] = -0.5 * math.pi
    mc.d.qpos[mc.qad] = q
    mc.mj.mj_forward(mc.m, mc.d)


def one(dr, dL, dz, extra=None):
    g = dict(T5B.GEOM_B, dr=dr, dL=dL, dz=dz, **(extra or {}))
    c = T5B.calibrate(g)
    out = dict(dr=dr, dL=dL, dz=dz, geom=g, m_dry=c["m_dry"], B_np=c["B_np"], m_p_each=c["m_cap"] + c["m_pt"], x_pt=c["x_pt"],
               xp=c["xp"], resid_mass=c["resid_mass"], resid_xp=c["resid_xp"], no_spare=c.get("no_spare", False))
    for fill in (T5.F_NEUTRAL, 0.5, 0.0):
        mc = T5B.make("water", g=g, fill=fill, pose=dict(x=-30.0, y=0.0, z=-5.0, vx=0.0, vz=0.0))
        sink_pose(mc)
        B, rb, _, sub = T.static_buoyancy(mc)
        R = mc.d.xmat[mc.hull].reshape(3, 3)
        f = T5.facts(mc)
        k = T5B.righting(mc)
        zB = float((R.T @ (np.array([rb[0], rb[1], mc.d.xpos[mc.hull][2]]) - mc.d.xpos[mc.hull]))[0])
        out[f"fill{fill}"] = dict(mass=f["mass"], com_hull=f["com_hull"], inertia=f["inertia_hull_diag"], B=B, b=B / (f["mass"] * G),
                                  K_roll=k["roll"], K_pitch=k["pitch"], hW=(k["roll"] / (f["mass"] * G)))
    return out


if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1])
    res = []
    for c in cfgs:
        r = one(*c)
        res.append(r)
        n = r["fill0.95"]
        print(json.dumps(dict(dr=c[0], dL=c[1], dz=c[2], m_dry=round(r["m_dry"], 3), m_p=round(r["m_p_each"], 3), x_pt=round(r["x_pt"], 3),
                              com_z95=round(n["com_hull"][2], 4), K_roll95=round(n["K_roll"], 3), K_pitch95=round(n["K_pitch"], 3),
                              K_roll_f0=round(r["fill0.0"]["K_roll"], 3), K_roll_f50=round(r["fill0.5"]["K_roll"], 3),
                              I95=[round(v, 4) for v in n["inertia"]], b95=round(n["b"], 4))), flush=True)
    p = HERE / "results" / "S0b_T5b_size.json"
    old = json.loads(p.read_text()) if p.exists() else []
    p.write_text(json.dumps(old + res, indent=1))
