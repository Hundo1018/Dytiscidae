"""Kinematic clearance of a stroke family on the built model (targets tracked exactly, `mj_forward` only, 120
phases per beat): the smallest signed distance (m) of each wing against every other part (pontoon, strut,
dorsal float, hull, tail, other wings), over a list of parameter sets.  Used for T5b's dorsal float and the
under-water stroke's whole input range.

    MUJOCO_GL=disable PYTHONPATH=. python experiments/flapper/s0b_t5b_clear.py '<json: {"geom":{...},"cases":[{"id":..,"P":{..}},..],"fill":0.0}>' [out.json]
"""
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0b_laws as LW
import s0b_t5b_probe as Q
import s0b_t5b_machine as T5B


def scan(mc, P, n=120, osc=1.0, family="level"):
    m, d, mj = mc.m, mc.d, mc.mj
    pairs = Q.clear_pairs(mc)
    names = lambda g: mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g)
    out = {k: [9.0, None, None] for k in pairs}
    ft = np.zeros(6)
    Pf = dict(LW.LEVEL, **P)
    for i in range(n):
        ph = 2 * math.pi * i / n
        if family == "level":
            q, _ = LW.targets_level(mc, ph, 1.0, Pf, osc=osc)
        else:                                               # the stroke-hinge stroke (B2): targets, diagonal phasing in P
            q, _ = LW.targets(mc, ph, 1.0, dict(LW.ZERO, **P))
        d.qpos[mc.qad] = q
        mj.mj_forward(m, d)
        for k, pp in pairs.items():
            for a, b in pp:
                dist = mj.mj_geomDistance(m, d, a, b, 1.0, ft)
                if dist < out[k][0]:
                    out[k] = [float(dist), f"{names(a)}|{names(b)}", round(math.degrees(ph))]
    return out


def main(argv):
    spec = json.loads(argv[1])
    mc = T5B.make("water", g=spec.get("geom"), fill=spec.get("fill", 0.0), pose=dict(x=-30.0, y=0.0, z=0.3, vx=0.0, vz=0.0))
    res = []
    for c in spec["cases"]:
        o = scan(mc, c["P"], n=c.get("n", 120), family=c.get("family", "level"))
        res.append(dict(id=c["id"], P=c["P"], clr={k: v for k, v in o.items()}))
        print(c["id"], {k: (round(v[0], 3), v[1], v[2]) for k, v in o.items()}, flush=True)
    return res


if __name__ == "__main__":
    res = main(sys.argv)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
