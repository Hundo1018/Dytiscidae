"""T5e probe S (DESIGN_S0b_bold.md, Addenda 3 and 4), no rollout: the command ranges, the clearance and the static table.

    MUJOCO_GL=disable PYTHONPATH=. python experiments/flapper/s0b_t5e_static.py envelope '<geom json>'   (results/S0b_T5e_envelope_<tag>.npz)
    ... s0b_t5e_static.py clear '<geom json>'       smallest distance of every wing / tail geom to every fixed geom, and where
    ... s0b_t5e_static.py table '<geom json>' tag   the static table (rows -> results/S0b_probes7.jsonl)

**Command ranges** (`LIM`): what the T5e laws may command; the laws in `s0b_t5e_probe.py` are built from this dict, so the
clearance below covers every hinge target they can produce.  Targets are clipped to the hinge ranges, as the hinges do.
clear: `mj_geomDistance` over the grid of those ranges (every phase of the strokes, every corner of the inputs, the pose blends).
envelope: the same grid, the wing and tail surfaces sampled, per 1 cm cell of the hull's xy plane the lowest and highest point
       reached: where a vertical part may stand (used to place the columns; the `clear` scan is the check that is reported).
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0b_laws as LW  # noqa: E402
import s0b_t5b_probe as Q  # noqa: E402
import s0b_t5e_machine as TE  # noqa: E402

OUT = HERE / "results" / "S0b_probes7.jsonl"
MOD = "s0b_t5e_machine"
TAGS = (("FL", 1.0), ("FR", -1.0), ("HL", 1.0), ("HR", -1.0))
X0, X1, Y0, Y1, DX = -0.95, 1.35, 0.0, 0.95, 0.01
#: the T5e laws' command ranges.  hover: Hover4 on `targets_level` (amp limits, mean sweep = trim + u, |u| <= sw_u; left-right
#: amplitude difference; left-right feather offset).  swim / U-turn: `targets` (throttle a, stroke 20 a deg, feather 50 a deg;
#: yaw by Psi_lr; differential thrust by amp_lr; tail).  surfacing: wings level at a mean sweep, one side raised; or the raised pose.
#: hover: the mean sweep is clipped to ``sw0_lo`` (absolute, rad) while the amplitude is at most ``amp_split`` and to ``sw0`` above
#: it; the left-right amplitude difference to ``amp_lr_lo`` / ``amp_lr_hi`` (found on the T5b body itself: at amp 0.9 the wing roots come
#: within 0.03 m of the hull or the dorsal float outside these, `results/t5e/hover_limits.log`).
LIM = dict(hover=dict(amp=(0.2, 0.9), sw0=(-0.05, 0.25), sw0_lo=(-0.05, 0.25), amp_split=0.85, amp_lr_lo=0.15, amp_lr_hi=0.08, psi0_lr=0.3),
           swim=dict(a=(0.0, 1.0), Psi_lr=0.3, amp_lr=1.0, tail=0.5, Phi_deg=20.0, Psi_deg=50.0),
           surface=dict(sweep=(-0.3, 0.75), dih=0.6),
           tail=0.5)


def build(g=None, z=0.2, fill=0.0):
    return Q._make("water", fill, "x4", 1e-4, z=z, g=g, mod=MOD)


def poses(mc):
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


def command_grid(mc, lim=LIM, nph=48):
    """Yield (label, q) over the laws' command ranges (hinge targets, clipped to the hinge ranges)."""
    m = mc.m
    jr = m.jnt_range[m.dof_jntid[mc.dof]]
    clip = lambda q: np.clip(q, jr[:, 0], jr[:, 1])
    n = len(mc.aid)
    Pl = dict(LW.LEVEL, **Q.law_cfg_b()["P"])
    H = lim["hover"]
    P = poses(mc)
    for amp in (H["amp"][0], 0.55, 0.7, 0.8, H["amp_split"], H["amp_split"] + 1e-6, H["amp"][1]):
        al = H["amp_lr_lo"] if amp <= H["amp_split"] else H["amp_lr_hi"]
        sr = H["sw0_lo"] if amp <= H["amp_split"] else H["sw0"]
        for sw0 in (sr[0], 0.5 * (sr[0] + sr[1]), sr[1]):
            for alr in (-al, 0.0, al):
                for plr in (-H["psi0_lr"], 0.0, H["psi0_lr"]):
                    Pv = dict(Pl, amp=amp, sw0=sw0, amp_lr=alr, psi0_lr=plr)
                    for i in range(nph):
                        yield f"hover amp {amp:.2f} sw0 {sw0:+.2f} amp_lr {alr:+.2f} psi0_lr {plr:+.1f}", clip(LW.targets_level(mc, 2 * math.pi * i / nph, 1.0, Pv)[0])
    for osc in np.linspace(0.0, 1.0, 6):                        # the ramp in and out of the stroke
        for amp, sw0r in ((0.713, H["sw0_lo"][0]), (0.713, H["sw0_lo"][1]), (H["amp"][1], H["sw0"][0]), (H["amp"][1], H["sw0"][1])):
            for i in range(0, nph, 2):
                yield "hover ramp", clip(LW.targets_level(mc, 2 * math.pi * i / nph, 1.0, dict(Pl, amp=amp, sw0=sw0r), osc=osc)[0])
    Sw = lim["swim"]
    for a in (0.3, 0.6, Sw["a"][1]):
        for plr in (-Sw["Psi_lr"], 0.0, Sw["Psi_lr"]):
            for alr in (-Sw["amp_lr"], 0.0, Sw["amp_lr"]):
                for tl in (-Sw["tail"], 0.0, Sw["tail"]):
                    Pw = dict(LW.ZERO, Phi=math.radians(Sw["Phi_deg"]), Psi=math.radians(Sw["Psi_deg"]) * a, lr_phase=math.pi, amp=a,
                              Psi_lr=plr, amp_lr=alr, tail=tl)
                    for i in range(nph):
                        yield f"swim a {a} Psi_lr {plr:+.1f} amp_lr {alr:+.1f} tail {tl:+.1f}", clip(LW.targets(mc, 2 * math.pi * i / nph, 1.0, Pw)[0])
    Su = lim["surface"]
    for sw in np.linspace(Su["sweep"][0], Su["sweep"][1], 13):
        for dl, dr_ in ((0.0, 0.0), (Su["dih"], 0.0), (0.0, Su["dih"]), (Su["dih"], Su["dih"])):
            for tl in (-lim["tail"], 0.0, lim["tail"]):
                q = np.zeros(n)
                for tag, L in TAGS:
                    q[mc.ix[(tag, "sweep")]] = -L * sw
                    q[mc.ix[(tag, "stroke")]] = dl if L > 0 else dr_
                q[mc.tail_ix] = tl
                yield f"surface sweep {sw:+.2f} dih {dl}/{dr_}", clip(q)
    for a_, b_ in (("raised", "sink"), ("sink", "flat"), ("flat", "raised"), ("flat45", "raised"), ("flat", "flat45")):
        for r in np.linspace(0.0, 1.0, 11):
            yield f"blend {a_}>{b_}", clip(P[a_] + (P[b_] - P[a_]) * r)


def _geoms(mc):
    m, mj = mc.m, mc.mj
    bname = lambda g_: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g_])) or ""
    gname = lambda g_: mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g_) or ""
    wing = [g_ for g_ in range(m.ngeom) if bname(g_) in ("seg1_wing", "seg2_wing", "seg3_wing", "seg4_wing")]
    tail = [g_ for g_ in range(m.ngeom) if bname(g_) in ("seg6_wing", "seg7_wing")]
    fixed = {}
    for g_ in range(m.ngeom):
        bn, gn = bname(g_), gname(g_)
        if gn.endswith(("_k", "tank_L_g", "tank_R_g")) or "ctank" in gn:
            continue
        if bn in ("t5_pont_L", "t5_pont_R"):
            fixed.setdefault("pontoon", []).append(g_)
        elif bn == "seg0_hull":
            k = ("column" if gn.startswith("t5e_col") else "boom" if gn.startswith("t5e_boom") else "post" if gn.startswith("t5e_post")
                 else "fin" if gn == "t5e_fin" else "dorsal" if gn == "t5b_dorsal" else "strut" if gn.startswith("t5_strut") else "hull")
            fixed.setdefault(k, []).append(g_)
    return wing, tail, fixed


def clearance(g=None, lim=LIM, nph=48, mc=None):
    """{(surface, part): (distance, label)}: the smallest distance and the command at which it occurs."""
    mc = build(g) if mc is None else mc
    m, d = mc.m, mc.d
    wing, tail, fixed = _geoms(mc)
    ft = np.zeros(6)
    best = {}
    for label, q in command_grid(mc, lim, nph):
        d.qpos[mc.qad] = q
        mc.mj.mj_forward(m, d)
        for sn, sg in (("wing", wing), ("tail", tail)):
            for pn, pg in fixed.items():
                if sn == "tail" and pn in ("hull",):           # the tail is carried by its own boom from the hull's stern
                    continue
                dm = min(mc.mj.mj_geomDistance(m, d, a, b, 1.0, ft) for a in sg for b in pg)
                if dm < best.get((sn, pn), (9.0, ""))[0]:
                    best[(sn, pn)] = (float(dm), label)
    return best


def envelope(g=None, lim=LIM, nph=48, tag="A"):
    mc = build(g)
    m, d, mj = mc.m, mc.d, mc.mj
    names = ("seg1_wing_g", "seg3_wing_g", "seg6_wing_g")       # left wings and the left tail surface; right by mirror
    gid = [g_ for g_ in range(m.ngeom) if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g_) or "") in names]
    u, v = np.linspace(-1, 1, 31), np.linspace(-1, 1, 61)
    nx, ny = int(round((X1 - X0) / DX)), int(round((Y1 - Y0) / DX))
    zmin, zmax = np.full((nx, ny), 9.0), np.full((nx, ny), -9.0)
    UV = np.stack(np.meshgrid(u, v, indexing="ij"), -1).reshape(-1, 2)
    for label, q in command_grid(mc, lim, nph):
        d.qpos[mc.qad] = q
        mj.mj_forward(m, d)
        o, R = d.xpos[mc.hull], d.xmat[mc.hull].reshape(3, 3)
        for g_ in gid:
            s = m.geom_size[g_]
            c, Rg = d.geom_xpos[g_], d.geom_xmat[g_].reshape(3, 3)
            for sz in (-1.0, 1.0):
                P = np.concatenate([UV * s[:2], np.full((len(UV), 1), sz * s[2])], 1)
                W = ((c + P @ Rg.T) - o) @ R
                ix = np.floor((W[:, 0] - X0) / DX).astype(int)
                iy = np.floor((np.abs(W[:, 1]) - Y0) / DX).astype(int)
                ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
                np.minimum.at(zmin, (ix[ok], iy[ok]), W[ok, 2])
                np.maximum.at(zmax, (ix[ok], iy[ok]), W[ok, 2])
    np.savez(HERE / "results" / f"S0b_T5e_envelope_{tag}.npz", zmin=zmin, zmax=zmax)
    return zmin, zmax


def free_distance(zmin, zmax, z0, z1, c=0.03):
    """Per cell: horizontal distance (m) to the nearest cell where a wing or tail surface comes within ``c`` vertically of the
    height range [z0, z1] (0 inside such a cell)."""
    from scipy.ndimage import distance_transform_edt
    occ = (zmin < 8.0) & (zmin - c < z1) & (zmax + c > z0)
    full = np.concatenate([occ[:, ::-1], occ], axis=1)          # mirror over the centre line
    dist = distance_transform_edt(~full) * DX
    return dist[:, occ.shape[1]:], occ


def _quat(roll, pitch_up):
    h, hr = pitch_up / 2, roll / 2
    w1, x1, y1, z1 = math.cos(h), 0.0, math.sin(-h), 0.0
    w2, x2, y2, z2 = math.cos(hr), math.sin(hr), 0.0, 0.0
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def eval_at(mc, q, zdry, roll=0.0, pitch=0.0):
    d, m, s = mc.d, mc.m, mc.env.solver
    d.qpos[mc.qad] = q
    d.qpos[:3] = (0.0, 0.0, 0.0)
    d.qpos[3:7] = _quat(roll, pitch)
    mc.mj.mj_forward(m, d)
    d.qpos[2] += zdry - float(TE.dry_com(mc)[2])
    mc.mj.mj_forward(m, d)
    d.qvel[:] = 0.0
    d.xfrc_applied[:] = 0.0
    s.reset()
    s.apply(d, 0.0)
    F, M = TE.wrench_about_dry_com(mc)
    d.xfrc_applied[:] = 0.0
    return F, M


def table(g=None, pose_names=("flat45", "raised", "flat"), fills=None, eps=0.02, write=True, tag="A", mc=None):
    mc = build(g) if mc is None else mc
    s = mc.env.solver
    P = poses(mc)
    fills = list(np.round(np.arange(0.0, 1.0001, 0.05), 2)) if fills is None else fills
    rows = []
    for pn in pose_names:
        q = P[pn]
        prev = None
        for fill in fills:
            TE.set_fill(mc, float(fill))
            W = float(s._dry_mass.sum() * TE.G)
            Fdeep = eval_at(mc, q, -3.0)[0][2]
            if Fdeep < W * (1 + 2e-4):                    # sinks or neutral: evaluated fully submerged
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
            rows.append(dict(kind="static", id=f"S_{tag}_{pn}_f{int(round(fill * 100)):03d}", layout=tag, geom=mc.t5["geom"], pose=pn,
                             fill=float(fill), fill_pont=mc.fill_p, fill_col=mc.fill_c, weight=W, afloat=afloat,
                             hull_z=hull_z if afloat else None, b=float(Fdeep / W), K_pitch=float(Kp), K_roll=float(Kr),
                             trim_My=float(M0[1]), trim_pitch_deg=float(np.degrees(M0[1] / Kp)) if Kp > 0 else None,
                             d_draft=(None if (prev is None or not afloat) else float(hull_z - prev)), mass_dry=float(mc.t5["m_dry"])))
            prev = hull_z if afloat else None
    TE.set_fill(mc, 0.0)
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


def summary(rows):
    af = [r for r in rows if r["afloat"]]
    return dict(K_pitch_min=min(r["K_pitch"] for r in rows), K_roll_min=min(r["K_roll"] for r in rows),
                d_draft_max=max(abs(r["d_draft"]) for r in af if r["d_draft"] is not None),
                b_at_095=[r["b"] for r in rows if abs(r["fill"] - 0.95) < 1e-6][0], mass_dry=rows[0]["mass_dry"],
                trim_pitch_max_deg=max(abs(r["trim_pitch_deg"] or 0.0) for r in af))


if __name__ == "__main__":
    what = sys.argv[1]
    g = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    tag = sys.argv[3] if len(sys.argv) > 3 else "A"
    if what == "envelope":
        zmin, zmax = envelope(g, tag=tag)
        print("saved", tag, float(zmin[zmin < 8].min()), float(zmax[zmax > -8].max()))
    elif what == "clear":
        res = clearance(g)
        for (sn, pn), (dm, lab) in sorted(res.items(), key=lambda kv: kv[1][0]):
            print(f"{sn:4s} - {pn:8s} {dm:+.3f} m  at  {lab}", flush=True)
    elif what == "table":
        rows, mc = table(g, tag=tag, write=(len(sys.argv) > 4 and sys.argv[4] == "w"))
        print(tag, "dry mass", round(mc.t5["m_dry"], 4), {k: round(v, 4) for k, v in mc.t5.items() if k in ("m_col_shells", "m_booms", "m_posts", "m_fin", "B_all", "B_columns", "v_flood", "xp", "x_pt", "resid_mass")},
              "fsb", {k: round(v, 4) for k, v in TE.free_surface_bound(mc).items()}, "summary", {k: round(v, 4) for k, v in summary(rows).items()})
        show(rows)
