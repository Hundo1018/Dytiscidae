"""S0b B5: wing-wing / wing-hull / wing-tail clearance of the level-hover stroke, kinematic (targets tracked
exactly, `mj_forward` only, no dynamics), on the built T4 model.  The solver has no interference model and the
machine's own geoms do not collide (contype 1 / conaffinity 2), so this is the only read of it.

``cut`` removes the inner ``cut`` metres of each main wing's box ON A SCRATCH COPY, for this geometric read
only (no rollout uses it): how much of the root would have to be absent for the stroke to clear.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/s0b_clear.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0b_laws as LW  # noqa: E402
import s0b_probe2 as P2  # noqa: E402


def scan(mc, P, cut=0.0, n=240):
    m, d, mj = mc.m, mc.d, mc.mj
    size0, pos0 = m.geom_size.copy(), m.geom_pos.copy()
    wg = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in (2, 3, 4, 5)]
    for g in wg:                                             # box along its x from 0 to 2 size_x: keep [cut, 2 size_x]
        full = 2 * size0[g, 0]
        m.geom_size[g, 0] = (full - cut) / 2
        m.geom_pos[g, 0] = cut + (full - cut) / 2
    pairs = P2._pairs(mc)
    names = lambda g: mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g)
    out = {k: (9.0, None, None) for k in pairs}
    ft = np.zeros(6)
    Pf = dict(LW.LEVEL, **P)
    for i in range(n):
        ph = 2 * math.pi * i / n
        q, _ = LW.targets_level(mc, ph, 1.0, Pf)
        d.qpos[mc.qad] = q
        mj.mj_forward(m, d)
        for k, pp in pairs.items():
            for a, b in pp:
                dist = mj.mj_geomDistance(m, d, a, b, 1.0, ft)
                if dist < out[k][0]:
                    out[k] = (float(dist), f"{names(a)}|{names(b)}", round(math.degrees(ph)))
    m.geom_size[:], m.geom_pos[:] = size0, pos0
    return out


if __name__ == "__main__":
    mc = P2._make("air", None, "x4", 1e-4)
    rows = []
    D = P2.D
    cases = [("base", D(flip="vert", A=55, dih=35, Psi=45)),
             ("flat", D(flip="flat", A=55, dih=35, Psi=45)),
             ("vert_swfh40", D(flip="vert", A=55, dih=35, Psi=45, sw_fh=40)),
             ("vert_dihfh30", D(flip="vert", A=55, dih=35, Psi=45, dih_fh=30)),
             ("vert_hindpi", dict(D(flip="vert", A=55, dih=35, Psi=45), hind_phase=math.pi)),
             ("vert_A40", D(flip="vert", A=40, dih=35, Psi=45)),
             ("vert_A70", D(flip="vert", A=70, dih=35, Psi=45)),
             ("vert_dih20", D(flip="vert", A=55, dih=20, Psi=45)),
             ("vert_dih50", D(flip="vert", A=55, dih=50, Psi=45)),
             ("vert_A70_swfh10_dihfh20", D(flip="vert", A=65, dih=35, Psi=45, sw_fh=10, dih_fh=20))]
    for nm, P in cases:
        for cut in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
            o = scan(mc, P, cut)
            rows.append(dict(case=nm, cut=cut, **{k: v for k, v in o.items()}))
            print(nm, cut, {k: (round(v[0], 3), v[1], v[2]) for k, v in o.items()}, flush=True)
    (HERE / "results" / "S0b_clearance.json").write_text(json.dumps(rows, indent=1))
