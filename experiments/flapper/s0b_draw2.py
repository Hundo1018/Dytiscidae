"""S0b B6 / B7 figures, drawn from the built model at poses read from results/S0b_probes2.jsonl.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/s0b_draw2.py float
    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/s0b_draw2.py liftoff [row id]

float:   figures/S0b_T4_float.png (+ .json with the numbers printed on it).  World frame, the waterline at
         z = 0.  Rows: (a) wings flat, settled; (b) wings raised to 35 deg and swept 45 deg forward, settled
         10 s; (c) the hover stroke at its forward reversal (chord vertical) on the hull pose of (b).
         Fluid strips are dots: red = the solver counts it wet at that pose, black = dry.
liftoff: figures/S0b_T4_liftoff.png.  The trace of one B7 row, and the built model placed at the recorded
         position and attitude at six instants (wings drawn at the law's rest pose, not at their phase).
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

COL = {"seg0_hull": "0.45", "seg1_wing": "tab:blue", "seg2_wing": "tab:orange", "seg3_wing": "tab:purple",
       "seg4_wing": "tab:pink", "seg5_strut": "0.3", "seg6_wing": "tab:green", "seg7_wing": "tab:olive"}


def rows():
    return [json.loads(x) for x in open(HERE / "results" / "S0b_probes2.jsonl")]


def outline(mc, ax, i, j, alpha=0.3, off=np.zeros(3), only=None, lw=1.0):
    from scipy.spatial import ConvexHull
    m, d, mj = mc.m, mc.d, mc.mj
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b == 0:
            continue
        bn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b)
        if only and bn not in only:
            continue
        gn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
        if gn == "s0b_keel":
            continue
        c, Rg, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        ty = int(m.geom_type[g])
        if ty == 6:
            cs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * s
        elif ty == 7:
            mid = int(m.geom_dataid[g])
            a0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            cs = np.array(m.mesh_vert[a0:a0 + nv])
        elif ty in (3, 5):
            cs = np.array([[0, 0, -s[1]], [0, 0, s[1]]])
        else:
            cs = np.zeros((1, 3))
        Pt = np.array([c + Rg @ q for q in cs]) + off
        col = COL.get(bn, "k")
        if ty in (6, 7):
            try:
                hv = ConvexHull(Pt[:, [i, j]]).vertices
                ax.fill(Pt[hv, i], Pt[hv, j], color=col, alpha=alpha, ec=col, lw=lw)
            except Exception:
                ax.plot(Pt[:, i], Pt[:, j], color=col, lw=2)
        else:
            ax.plot(Pt[:, i], Pt[:, j], color=col, lw=2)


def strips(mc):
    p, d = mc.env.solver.panels, mc.d
    R = d.xmat.reshape(-1, 3, 3)[p.body_id]
    pos = d.xpos[p.body_id] + np.einsum("nij,nj->ni", R, p.pos_local)
    sub = mc.env.medium.submerged_fraction(pos, p.half_height, 0.0)
    return pos, sub, p.body_id


def float_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R_ = {r["id"]: r for r in rows() if r.get("kind") == "float"}
    ra, rb = R_["settle_bNone_flat_default"], R_["rest_bNone_d35_sw45_default"]
    mc = P2._make("water", None, "x4", 1e-4, z=0.0)
    m, d, mj = mc.m, mc.d, mc.mj
    w = 2 * math.pi * 4.5
    Pb = dict(LW.LEVEL, **P2.D(dih=35, sw0=45))
    q_flat = np.zeros(len(mc.aid))
    q_b, _ = LW.targets_level(mc, 0.0, w, Pb, osc=0.0)
    Pc = dict(LW.LEVEL, **P2.D(flip="vert", A=55, dih=35, Psi=45, sw0=5.9), amp=0.42)
    q_c, _ = LW.targets_level(mc, math.pi / 2, w, Pc, osc=1.0)
    states = [("(a) wings flat, settled 8 s (id settle_bNone_flat_default)", ra, q_flat),
              ("(b) wings raised 35 deg, swept 45 deg forward, 10 s (id rest_bNone_d35_sw45_default)", rb, q_b),
              ("(c) the hover stroke at its forward reversal (chord vertical), on the hull pose of (b)", rb, q_c)]
    fig, axs = plt.subplots(3, 2, figsize=(15, 13), gridspec_kw=dict(width_ratios=[1.25, 1.75]))
    out = {}
    for k, (title, r, q) in enumerate(states):
        d.qpos[:3] = r["hull_pos_end"]
        d.qpos[3:7] = r["quat_end"]
        d.qpos[mc.qad] = q
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)
        pos, sub, bid = strips(mc)
        Rh = d.xmat[mc.hull].reshape(3, 3)
        o = d.xpos[mc.hull]
        ax_z = [float((o + Rh @ np.array([x, 0, 0]))[2]) for x in (0.0, 0.25, 0.5)]
        per = [int(((sub > 0) & (bid == b)).sum()) for b in (2, 3, 4, 5)]
        low = P2.wing_low(mc)
        att = P2._att(Rh)
        tail = sub[np.isin(bid, [7, 8])]
        hull = sub[bid == 1]
        num = dict(hull_axis_z_tail_mid_nose=[round(v, 4) for v in ax_z], roll_pitch_yaw=[round(v, 2) for v in att],
                   wet_strips_FL_FR_HL_HR=per, wing_lowest_corner_z=round(low, 4),
                   tail_strips_wet=int((tail > 0).sum()), tail_strips=int(len(tail)),
                   hull_panel_sub=[round(float(v), 2) for v in hull], com_z=round(float(mc.com()[2]), 4))
        out[title[:3]] = num
        for ax, (i, j, lab) in zip(axs[k], ((0, 2, "side: world x (nose to the right) against z, m"),
                                             (1, 2, "front: world y against z, m (seen from the nose)"))):
            x0 = float(o[i]) if i == 0 else 0.0
            off = np.array([-x0 if i == 0 else 0.0, 0.0, 0.0])
            ax.axhspan(-0.5, 0.0, color="tab:cyan", alpha=0.18, zorder=0)
            ax.axhline(0.0, color="tab:blue", lw=1.5, zorder=1)
            outline(mc, ax, i, j, off=off)
            pp = pos + off
            wet = sub > 0
            ax.plot(pp[~wet, i], pp[~wet, j], ".", color="k", ms=5, zorder=6)
            ax.plot(pp[wet, i], pp[wet, j], ".", color="red", ms=8, zorder=7)
            c = mc.com() + off
            ax.plot(c[i], c[j], "k+", ms=14, mew=2.2, zorder=8)
            ax.set_aspect("equal")
            ax.set_xlim((-0.5, 0.75) if i == 0 else (-0.8, 0.8))
            ax.set_ylim(-0.3, 0.5)
            ax.set_xlabel(lab)
            ax.grid(alpha=0.3)
        axs[k][0].set_title(title, fontsize=9, loc="left")
        axs[k][1].set_title(f"hull axis z at tail end / mid / nose {ax_z[0]:+.3f} / {ax_z[1]:+.3f} / {ax_z[2]:+.3f} m; "
                            f"pitch {att[1]:+.1f}, roll {att[0]:+.1f} deg\nwet main-wing strips FL FR HL HR {per} "
                            f"(red dots), lowest wing corner z {low:+.3f} m, tail strips wet {int((tail > 0).sum())}/{len(tail)}",
                            fontsize=9, loc="left")
    fig.suptitle("S0b B6: T4 on the water, ballast out (0.918 kg), default mode.  Waterline z = 0 (blue).  "
                 "Dots: the solver's strips (red = wet); + the dry CoM.", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "figures" / "S0b_T4_float.png", dpi=105)
    (HERE / "figures" / "S0b_T4_float.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def liftoff_fig(rid="beach_T4_x4"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = [x for x in rows() if x.get("id") == rid and x.get("probe") in ("B7", "B7tune")][-1]
    tr = r["trace"]
    t = np.array([a[0] for a in tr])
    xyz = np.array([a[1:4] for a in tr])
    att = np.array([a[5:8] for a in tr])
    amp = np.array([a[8] for a in tr])
    con = np.array([a[10] for a in tr])
    ph = [a[11] for a in tr]
    start = r["start"]
    cut = r.get("cut", 0.0)
    dom = dict(float="water", beach="land", air="air")[start]
    mc = P2._make(dom, None, r.get("motor", "x4"), 1e-4, z=5.0, x=20.0 if start == "beach" else -30.0, cut=cut)
    m, d, mj = mc.m, mc.d, mc.mj
    P = dict(LW.LEVEL, **r["law"]["P"])
    q_rest, _ = LW.targets_level(mc, 0.0, 1.0, P, osc=0.0)
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    ev = r["events"]
    marks = {"rest": ev["start"]["t"] - 0.6}
    t_up = ev["start"]["t"]
    marks.update({"lift-off +0.6 s": t_up + 0.6, "climb": t_up + 2.0})
    if "at_height" in ev:
        marks["hold"] = ev["at_height"]["t"] + 5.0
    if "descend" in ev:
        marks["descent"] = ev["descend"]["t"] + 3.0
    if "touch" in ev:
        marks["after touch-down +3 s"] = min(ev["touch"]["t"] + 3.0, t[-1])
    xs = np.linspace(xyz[:, 0].min() - 1.2, xyz[:, 0].max() + 1.2, 50)
    if start == "beach":
        pass                                         # the ground is drawn under each snapshot, at its own offset
    else:
        ax.axhspan(xyz[:, 2].min() - 0.6, 0.0, color="tab:cyan", alpha=0.2)
        ax.axhline(0.0, color="tab:blue")
    ax.plot(xyz[:, 0], xyz[:, 2], color="0.6", lw=0.8, zorder=1)
    for n, (lab, tm) in enumerate(marks.items()):
        k = int(np.argmin(np.abs(t - tm)))
        rl, pt, yw = (math.radians(v) for v in att[k])
        cy, sy, cp, sp, cr, sr = math.cos(yw), math.sin(yw), math.cos(-pt), math.sin(-pt), math.cos(rl), math.sin(rl)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Rm = Rz @ Ry @ Rx
        w_ = math.sqrt(max(1.0 + Rm[0, 0] + Rm[1, 1] + Rm[2, 2], 1e-12)) / 2
        quat = np.array([w_, (Rm[2, 1] - Rm[1, 2]) / (4 * w_), (Rm[0, 2] - Rm[2, 0]) / (4 * w_), (Rm[1, 0] - Rm[0, 1]) / (4 * w_)])
        d.qpos[3:7] = quat
        d.qpos[mc.qad] = q_rest if ph[k] not in ("settle",) or tm > t_up - 0.3 else np.zeros(len(mc.aid))
        if lab.startswith("after"):
            d.qpos[mc.qad] = 0.0
        d.qpos[:3] = 0.0
        mj.mj_forward(m, d)
        d.qpos[:3] = xyz[k] - mc.com()
        mj.mj_forward(m, d)
        dx = 0.9 * (n - (len(marks) - 1) / 2) if start != "beach" else 0.0      # spread the snapshots sideways
        if start == "beach":                         # the beach under this snapshot, moved with it
            gx = np.linspace(xyz[k, 0] - 0.5, xyz[k, 0] + 0.5, 12)
            gz = np.array([mc.env.ground_height(float(x)) for x in gx])
            ax.fill_between(gx + 1.1 * (n - 2.5), gz, gz - 0.08, color="tan", alpha=0.8, zorder=0)
        outline(mc, ax, 0, 2, alpha=0.35, off=np.array([dx if start != "beach" else 1.1 * (n - 2.5), 0, 0]))
        ax.annotate(f"{lab}\nt {t[k]:.1f} s, z {xyz[k, 2] - r['z_start']:+.2f} m\npitch {att[k, 1]:+.0f}, roll {att[k, 0]:+.0f}",
                    (xyz[k, 0] + (dx if start != "beach" else 1.1 * (n - 2.5)), xyz[k, 2] + 0.45), fontsize=8, ha="center")
    ax.set_aspect("equal")
    ax.set_xlabel("world x, m (snapshots are spread sideways by 1.1 m each, each with the beach under its true place; "
                  "the grey line is the CoM path at its true x)")
    ax.set_ylabel("z, m")
    ax.grid(alpha=0.3)
    a1 = fig.add_subplot(gs[1, 0])
    a1.plot(t, xyz[:, 2] - r["z_start"], label="CoM height over the start, m")
    a1.plot(t, amp, label="amplitude (of 55 deg)")
    for k_, v in ev.items():
        if isinstance(v, dict) and "t" in v:
            a1.axvline(v["t"], color="0.5", ls=":", lw=0.8)
            a1.text(v["t"], 2.25, k_, fontsize=7, rotation=90, va="top")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    a1.set_xlabel("t, s")
    a2 = fig.add_subplot(gs[1, 1])
    a2.plot(t, att[:, 0], label="roll, deg")
    a2.plot(t, att[:, 1], label="pitch, deg (sampled at 50 Hz: the 4.5 Hz stroke wobble shows)")
    a2.plot(t, att[:, 2], label="yaw, deg")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    a2.set_xlabel("t, s")
    a3 = fig.add_subplot(gs[2, 0])
    a3.plot(t, con, color="tab:red")
    a3.set_ylabel("contact, weights (sampled at 50 Hz)")
    a3.set_xlabel("t, s")
    a3.grid(alpha=0.3)
    a4 = fig.add_subplot(gs[2, 1])
    a4.axis("off")
    hz = r.get("hold_z_rel")
    txt = (f"row {r['probe']} / {rid}: start {start}, motor {r.get('motor', 'x4')}, mass {r['mass']:.3f} kg"
           + (f", root cut {cut} m" if cut else "") + "\n"
           f"reached {r['z_max_rel']:.2f} m; hold {r['hold_s_done']:.1f} s, z {hz[0]:.2f}..{hz[1]:.2f} m, "
           f"drift {r['hold_drift_m']:.2f} m, tilt max {r['hold_tilt_max_deg']:.1f} deg\n"
           f"touch-down vz {ev.get('touch', {}).get('vz', float('nan')):+.2f} m/s (beat mean "
           f"{(ev.get('touch', {}).get('v_beat_mean') or [0, 0, float('nan')])[2]:+.2f}); end attitude roll/pitch/yaw "
           f"{[round(v, 1) for v in r['att_end']]}\n"
           f"peak torque N m {({k: round(v, 1) for k, v in r['tq_peak'].items()})}\n"
           f"contact peak, weights, by phase {({k: round(v, 1) for k, v in r.get('contact_peak_W_by_phase', {}).items()})}\n"
           f"wing contact by phase {({k: round(v, 1) for k, v in r.get('contact_peak_W_wing_by_phase', {}).items()})}\n"
           f"QACC {r['bad_qacc']}, limiter steps {r['clamped_steps']}, not evidence: {r['not_evidence']}, "
           f"X2 share {r['x2_share']:.2f}, shaft work + {r['E_pos_J']:.0f} J")
    a4.text(0.0, 1.0, txt, fontsize=9, va="top", family="monospace")
    fig.suptitle(f"S0b B7: lift-off, hold and descent ({rid}).  Model drawn at the recorded CoM and attitude; "
                 "wings at the law's raised rest pose, not at their stroke phase.", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "figures" / "S0b_T4_liftoff.png", dpi=105)
    print("ok", rid, list(marks.items()))


if __name__ == "__main__":
    if sys.argv[1] == "float":
        float_fig()
    else:
        liftoff_fig(*sys.argv[2:3])
