"""T5d probe S (DESIGN_S0b_bold.md, Addendum 3), no rollout: the static table and the kinematic clearance of the columns and fin.

    MUJOCO_GL=disable PYTHONPATH=. python experiments/flapper/s0b_t5d_static.py clear '<geom json>'
    ... s0b_t5d_static.py table '<geom json>' [pose ...]      (rows appended to results/S0b_probes6.jsonl, kind "static")

clear: smallest distance (m, negative = through) between every main-wing geom and every column / the fin / the pontoons over
       the strokes and poses the machine uses, hinge targets tracked exactly (the machine's own geoms do not collide in the model,
       so this is the only read).
table: at fill 0, 0.05 ... 1: the floating (or neutral-depth) equilibrium heave with the hull level, then K pitch and K roll from
       the fluid moment about the dry CoM at +-0.01 rad (the `kf` method of B19 step 3).  A fill that sinks (weight over the
       largest buoyancy) is evaluated fully submerged.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0b_laws as LW  # noqa: E402
import s0b_t5b_probe as Q  # noqa: E402
import s0b_t5d_machine as TD  # noqa: E402

OUT = HERE / "results" / "S0b_probes6.jsonl"
MOD = "s0b_t5d_machine"
TAGS = (("FL", 1.0), ("FR", -1.0), ("HL", 1.0), ("HR", -1.0))


def build(g=None, z=0.2, fill=0.0):
    return Q._make("water", fill, "x4", 1e-4, z=z, g=g, mod=MOD)


def poses(mc):
    """Named wing poses (hinge targets)."""
    n = len(mc.aid)
    Pl = dict(LW.LEVEL, **Q.law_cfg_b()["P"])
    raised = LW.targets_level(mc, 0.0, 1.0, Pl, osc=0.0)[0]
    sink = raised.copy()
    flat = np.zeros(n)
    flat45 = np.zeros(n)
    for tag, L in TAGS:
        sink[mc.ix[(tag, "feather")]] = -0.5 * math.pi
        flat45[mc.ix[(tag, "sweep")]] = -L * 0.45
    return dict(raised=raised, sink=sink, flat=flat, flat45=flat45)


def _pairs(mc):
    m, mj = mc.m, mc.mj
    bname = lambda g_: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g_])) or ""
    gname = lambda g_: mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g_) or ""
    wing = [g_ for g_ in range(m.ngeom) if bname(g_) in ("seg1_wing", "seg2_wing", "seg3_wing", "seg4_wing")]
    tail = [g_ for g_ in range(m.ngeom) if bname(g_) in ("seg6_wing", "seg7_wing")]
    col = [g_ for g_ in range(m.ngeom) if gname(g_).startswith("t5d_col_")]
    fin = [g_ for g_ in range(m.ngeom) if gname(g_) == "t5d_fin"]
    pont = [g_ for g_ in range(m.ngeom) if bname(g_) in ("t5_pont_L", "t5_pont_R")]
    out = dict(wing_col=[(a, b) for a in wing for b in col], wing_fin=[(a, b) for a in wing for b in fin],
               tail_fin=[(a, b) for a in tail for b in fin], tail_col=[(a, b) for a in tail for b in col],
               wing_pont=[(a, b) for a in wing for b in pont])
    return {k: v for k, v in out.items() if v}


def _dist(mc, pairs, cur):
    ft = np.zeros(6)
    for k, pp in pairs.items():
        dmin = min(mc.mj.mj_geomDistance(mc.m, mc.d, a, b, 1.0, ft) for a, b in pp)
        cur[k] = min(cur.get(k, 9.0), float(dmin))
    return cur


def clearance(g=None, nph=72):
    mc = build(g)
    m, d = mc.m, mc.d
    pairs = _pairs(mc)
    out = {}
    cfg = Q.law_cfg_b()
    Pl = dict(LW.LEVEL, **cfg["P"])

    def setq(q):
        d.qpos[mc.qad] = q
        mc.mj.mj_forward(m, d)
    # hover stroke
    for amp in (0.713, 0.9):
        for sw0 in (Pl["sw0"], Pl["sw0"] + 0.3, Pl["sw0"] - 0.3):
            cur = {}
            for i in range(nph):
                q, _ = LW.targets_level(mc, 2 * math.pi * i / nph, 1.0, dict(Pl, amp=amp, sw0=sw0))
                setq(q)
                _dist(mc, pairs, cur)
            out[f"hover amp {amp} sw0 {sw0:+.2f}"] = cur
    # swim stroke (B20's: stroke 20 deg full, feather 50 deg, diagonal; pitch input on the mean dihedral +-0.12)
    for bias in (0.0, 0.12, -0.12):
        cur = {}
        Pw = dict(LW.ZERO, Phi=math.radians(20.0), Psi=math.radians(50.0), lr_phase=math.pi, amp=1.0, bias=bias)
        for i in range(nph):
            q, _ = LW.targets(mc, 2 * math.pi * i / nph, 1.0, Pw)
            setq(q)
            _dist(mc, pairs, cur)
        out[f"swim stroke bias {bias:+.2f}"] = cur
    # static poses and the surfacing sweep range
    P = poses(mc)
    for nm in ("raised", "sink", "flat"):
        setq(P[nm])
        out[f"pose {nm}"] = _dist(mc, pairs, {})
    for sw in (-0.5, 0.0, 0.45, 0.9):
        q = np.zeros(len(mc.aid))
        for tag, L in TAGS:
            q[mc.ix[(tag, "sweep")]] = -L * sw
        setq(q)
        out[f"flat sweep {sw:+.2f}"] = _dist(mc, pairs, {})
    return out


def _quat(roll, pitch_up):
    h, hr = pitch_up / 2, roll / 2
    a = np.array([math.cos(h), 0.0, math.sin(-h), 0.0])
    b = np.array([math.cos(hr), math.sin(hr), 0.0, 0.0])
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def eval_at(mc, q, zdry, roll=0.0, pitch=0.0):
    d, m, s = mc.d, mc.m, mc.env.solver
    d.qpos[mc.qad] = q
    d.qpos[:3] = (0.0, 0.0, 0.0)
    d.qpos[3:7] = _quat(roll, pitch)
    mc.mj.mj_forward(m, d)
    d.qpos[2] += zdry - float(TD.dry_com(mc)[2])
    mc.mj.mj_forward(m, d)
    d.qvel[:] = 0.0
    d.xfrc_applied[:] = 0.0
    s.reset()
    s.apply(d, 0.0)
    F, M = TD.wrench_about_dry_com(mc)
    d.xfrc_applied[:] = 0.0
    return F, M


def table(g=None, pose_names=("flat45", "raised"), fills=None, eps=0.02, write=True, tag=""):
    mc = build(g)
    s = mc.env.solver
    P = poses(mc)
    fills = list(np.round(np.arange(0.0, 1.0001, 0.05), 2)) if fills is None else fills
    rows = []
    for pn in pose_names:
        q = P[pn]
        prev = None
        for fill in fills:
            TD.set_fill(mc, float(fill))
            W = float(s._dry_mass.sum() * TD.G)
            Fdeep = eval_at(mc, q, -3.0)[0][2]
            if Fdeep < W - 1e-9:                         # sinks: evaluated fully submerged
                z, afloat = -3.0, False
            else:
                lo, hi = -1.2, 0.8
                for _ in range(44):
                    mid = 0.5 * (lo + hi)
                    if eval_at(mc, q, mid)[0][2] > W:
                        lo = mid
                    else:
                        hi = mid
                z, afloat = 0.5 * (lo + hi), True
            Kp = (eval_at(mc, q, z, pitch=+eps / 2)[1][1] - eval_at(mc, q, z, pitch=-eps / 2)[1][1]) / eps
            Kr = -(eval_at(mc, q, z, roll=+eps / 2)[1][0] - eval_at(mc, q, z, roll=-eps / 2)[1][0]) / eps
            F0, M0 = eval_at(mc, q, z)
            hull_z = float(mc.d.xpos[mc.hull][2])
            row = dict(kind="static", id=f"S_{tag}_{pn}_f{int(round(fill * 100)):03d}", geom=mc.t5["geom"], pose=pn, fill=float(fill),
                       fill_pont=mc.fill_p, fill_col=mc.fill_c, weight=W, afloat=afloat, hull_z=hull_z if afloat else None,
                       b=float(Fdeep / W), K_pitch=float(Kp), K_roll=float(Kr), trim_My=float(M0[1]),
                       d_draft=(None if (prev is None or not afloat) else float(hull_z - prev)), mass_dry=float(mc.t5["m_dry"]))
            prev = hull_z if afloat else None
            rows.append(row)
    if write:
        with open(OUT, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
    return rows, mc


def show(rows):
    for r in rows:
        print(f"{r['pose']:7s} fill {r['fill']:.2f} (p {r['fill_pont']:.2f} c {r['fill_col']:.2f}) W {r['weight']:6.2f} b {r['b']:.4f} "
              f"{'hull z %+.3f' % r['hull_z'] if r['afloat'] else 'submerged   '} d {('%+.3f' % r['d_draft']) if r['d_draft'] is not None else '   -  '} "
              f"Kp {r['K_pitch']:+6.2f} Kr {r['K_roll']:+6.2f} My0 {r['trim_My']:+.2f}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    g = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    if what == "clear":
        res = clearance(g)
        for k, v in res.items():
            print(f"{k:28s} " + "  ".join(f"{a} {b:+.3f}" for a, b in v.items()), flush=True)
    elif what == "table":
        rows, mc = table(g, pose_names=tuple(sys.argv[3:]) or ("flat45", "raised"), tag=(sys.argv[2][:0] or "T5d"))
        print("dry mass", mc.t5["m_dry"], {k: round(v, 4) for k, v in mc.t5.items() if k in ("m_col_shell_each", "m_fin", "B_all", "B_columns", "v_flood", "xp", "resid_mass")},
              "free-surface bound", {k: round(v, 4) for k, v in TD.free_surface_bound(mc).items()})
        show(rows)
