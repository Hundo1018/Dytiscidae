"""M3b phase-1 probes (disclosed in PREREG_M3b_land.md section 0). Decides nothing.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m3b_probe.py build|step|turn1
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def build():
    from m3b_sweep import SweepMachine
    mc = SweepMachine("air", pose=dict(x=20, z=30))
    m, d, mj = mc.m, mc.d, mc.mj
    print("mass", round(mc.mass, 4), "roles", list(mc.role), "names", list(mc.env.act_names))
    for n, j in enumerate(mc.jid):
        print(" ", mc.env.act_names[n], mc.role[n], "range", np.degrees(m.jnt_range[j]).round(1),
              "armature", round(float(m.dof_armature[mc.dof[n]]), 5), "motor", mc.motor[n])
    w = mc.env.solver._dry_mass
    print("CoM (dry) hull frame x:", round(float(((w[:, None] * d.xipos).sum(0) / w.sum())[0] - d.xpos[mc.hull][0]), 4))
    for n, a in enumerate(mc.env.act_names):
        if not a.endswith("_s"):
            continue
        b = int(m.jnt_bodyid[mc.jid[n]])
        g = [k for k in range(m.ngeom) if m.geom_bodyid[k] == b][0]
        q = np.zeros(len(mc.aid)); mc.place(q=q); p0 = d.geom_xpos[g].copy()
        q[n] = 0.3; mc.place(q=q); p1 = d.geom_xpos[g].copy()
        print(" ", a, "+0.3 rad moves the wing centre by", (p1 - p0).round(4), "(world; nose is +x)")
        mc.place(q=np.zeros(len(mc.aid)))


def keyed(mc, keys, col, sign=None):
    ts = np.array([k[0] for k in keys]); vs = np.radians(np.array([k[1] for k in keys], float))
    out = []
    for k in range(int(ts[-1] / mc.dt)):
        t = k * mc.dt
        i = min(np.searchsorted(ts, t, side="right") - 1, len(ts) - 2)
        s = (t - ts[i]) / (ts[i + 1] - ts[i])
        v = vs[i] + s * (vs[i + 1] - vs[i]); r = (vs[i + 1] - vs[i]) / (ts[i + 1] - ts[i])
        sg = np.ones(len(col)) if sign is None else sign
        mc.step(sg * v[col], sg * r[col])
        if k % 5000 == 0:
            out.append((t, mc.com().copy()))
    return out


def step():
    """Sweep machine: stand, then three strides of a both-wing sweep step, keyframed (stroke, feather, tail, sweep)."""
    from m3b_sweep import SweepMachine
    mc = SweepMachine("land", pose=dict(x=20, z=5)); mc.drop_to_ground()
    col = [{"stroke": 0, "feather": 1, "tail": 2, "sweep": 3}[r] for r in mc.role]
    keys = [(0.0, (0, 0, 0, 0)), (1.0, (0, 0, 0, 0)), (2.0, (-12, 0, -20, 20))]
    t = 2.0
    for i in range(3):   # stance: sweep 20 -> -20 (tips go back); lift; sweep back; plant
        keys += [(t + 1.2, (-12, 0, -20, -20)), (t + 1.5, (5, 0, -20, -20)), (t + 1.8, (5, 0, -20, 20)), (t + 2.1, (-12, 0, -20, 20))]
        t += 2.1
    c0 = None
    # +sweep in the keys = wing tip toward the nose: the left (seg1) joint's sign is the opposite (build probe)
    sign = np.array([-1.0 if (r == "sweep" and a.startswith("seg1")) else 1.0 for r, a in zip(mc.role, mc.env.act_names)])
    for tt, c in keyed(mc, keys, col, sign):
        if tt >= 2.0 and c0 is None:
            c0 = c
        if c0 is not None:
            print("t %.1f com dx %+.3f dy %+.3f pitch %+.1f" % (tt, *(c - c0)[:2], mc.pitch_deg()))
    print("qacc", mc.bad_qacc, "mass", round(mc.mass, 3), "elec", {k: round(v, 1) for k, v in mc.summary()["elec_W"].items()})


if __name__ == "__main__":
    {"build": build, "step": step}[sys.argv[1]]()
