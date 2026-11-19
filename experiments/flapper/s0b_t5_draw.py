"""S0b B9-B11 figures of T5, drawn from the built model and from rows of results/S0b_probes3.jsonl.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/s0b_t5_draw.py built
    ... s0b_t5_draw.py float
    ... s0b_t5_draw.py mission <row id>

built:   figures/S0b_T5_built.png (+ .json): side, top and front views in the hull frame, wings at the float
         pose (raised 35 deg); CoM dry and at the neutral fill; centre of buoyancy at the float waterline and
         fully submerged; masses.
float:   figures/S0b_T5_float.png (+ .json): world frame, waterline z = 0, at the settled float pose of each
         mode: at rest, and at the phase of the hover stroke where a wing is lowest.  Dots = the solver's
         strips and slices (red = counted wet).
mission: figures/S0b_T5_mission.png: the traces of one mission row and the model at instants along it.
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
import s0b_t5_machine as T5  # noqa: E402
import s0b_t5_probe as Q  # noqa: E402
import s0b_machine as T  # noqa: E402

G = 9.80665
COL = {"seg0_hull": "0.45", "seg1_wing": "tab:blue", "seg2_wing": "tab:orange", "seg3_wing": "tab:purple",
       "seg4_wing": "tab:pink", "seg5_strut": "0.3", "seg6_wing": "tab:green", "seg7_wing": "tab:olive",
       "t5_pont_L": "tab:red", "t5_pont_R": "tab:brown", "t5_tank_L": None, "t5_tank_R": None}


def rows():
    return [json.loads(x) for x in open(HERE / "results" / "S0b_probes3.jsonl")]


def outline(mc, ax, i, j, alpha=0.3, off=np.zeros(3), lw=1.0, frame=None):
    """Every geom of the machine projected on axes (i, j).  ``frame``: (origin, R) to draw in (the hull's)."""
    from scipy.spatial import ConvexHull
    m, d, mj = mc.m, mc.d, mc.mj
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b == 0:
            continue
        bn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b)
        col = COL.get(bn, "k")
        gn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
        if col is None or gn.endswith("_k"):
            continue
        c, Rg, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        ty = int(m.geom_type[g])
        if ty == 6:
            cs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * s
        elif ty == 7:
            mid = int(m.geom_dataid[g])
            a0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            cs = np.array(m.mesh_vert[a0:a0 + nv])
        elif ty == 3 and s[0] > 0.01:                     # a pontoon: the capsule's surface, sampled
            th = np.linspace(0, 2 * math.pi, 24, endpoint=False)
            ring = np.stack([s[0] * np.cos(th), s[0] * np.sin(th), np.zeros(24)], 1)
            cap = [np.array([0, 0, sg * (s[1] + s[0])]) for sg in (-1, 1)]
            cs = np.concatenate([ring + [0, 0, -s[1]], ring + [0, 0, s[1]], cap])
        elif ty in (3, 5):
            cs = np.array([[0, 0, -s[1]], [0, 0, s[1]]])
        else:
            cs = np.zeros((1, 3))
        Pt = np.array([c + Rg @ q for q in cs]) + off
        if frame is not None:
            Pt = (Pt - frame[0]) @ frame[1]
        if len(Pt) > 2:
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


def _cob(mc):
    """Centre of buoyancy (world) and buoyancy (N) from the solver's own elements at the present pose."""
    p = mc.env.solver.panels
    pos, sub, _ = strips(mc)
    f = mc.rho_w * G * p.volume_buoyant * sub
    return (f[:, None] * pos).sum(0) / max(f.sum(), 1e-12), float(f.sum())


def built_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R_ = {r["id"]: r for r in rows() if r.get("kind") == "float"}
    rf = R_.get("float_default")
    mc = Q._make("water", 0.0, "x4", 1e-4, z=Q.Z_FLOAT)
    m, d, mj = mc.m, mc.d, mc.mj
    Pl = dict(LW.LEVEL, **Q.law_cfg()["P"])
    q_r, _ = LW.targets_level(mc, 0.0, 1.0, Pl, osc=0.0)
    d.qpos[mc.qad] = q_r
    if rf:
        d.qpos[:3], d.qpos[3:7] = rf["hull_pos_end"], rf["quat_end"]
    d.qvel[:] = 0.0
    mj.mj_forward(m, d)
    o, Rh = d.xpos[mc.hull].copy(), d.xmat[mc.hull].reshape(3, 3).copy()
    hf = lambda p: (np.asarray(p) - o) @ Rh
    com_dry = hf(mc.com())
    cob_float, B_float = _cob(mc)
    cob_float = hf(cob_float)
    wl = [float(hf(o + np.array([x, 0, -o[2] - 0.0]) * 0 + (np.array([o[0] + x, 0, 0]) - o))[2]) for x in (0.0, 0.5)]
    f0 = T5.facts(mc)
    # fully submerged, neutral fill
    T5.set_fill(mc, T5.F_NEUTRAL)
    z_keep = d.qpos[2]
    d.qpos[2] = -5.0
    mj.mj_forward(m, d)
    o2, R2 = d.xpos[mc.hull].copy(), d.xmat[mc.hull].reshape(3, 3).copy()
    cob_sub, B_sub = _cob(mc)
    cob_sub = (cob_sub - o2) @ R2
    com_wet = (mc.com() - o2) @ R2
    f1 = T5.facts(mc)
    T5.set_fill(mc, 0.0)
    d.qpos[2] = z_keep
    mj.mj_forward(m, d)
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.6), gridspec_kw=dict(width_ratios=[1.25, 1.45, 1.45]))
    views = ((0, 2, "side: hull x (nose right) against hull z, m"), (0, 1, "top: hull x against hull y, m"),
             (1, 2, "front: hull y against hull z, m"))
    for ax, (i, j, lab) in zip(axs, views):
        outline(mc, ax, i, j, frame=(o, Rh))
        pos, sub, bid = strips(mc)
        ph = (pos - o) @ Rh
        wet = sub > 0
        ax.plot(ph[~wet, i], ph[~wet, j], ".", color="k", ms=3, zorder=6)
        ax.plot(ph[wet, i], ph[wet, j], ".", color="red", ms=6, zorder=7)
        ax.plot(com_dry[i], com_dry[j], "k+", ms=15, mew=2.4, zorder=9, label="CoM, ballast out")
        ax.plot(com_wet[i], com_wet[j], "+", color="tab:blue", ms=15, mew=2.4, zorder=9, label=f"CoM, fill {T5.F_NEUTRAL}")
        ax.plot(cob_float[i], cob_float[j], "o", mfc="none", color="k", ms=11, mew=1.8, zorder=9, label="centre of buoyancy at the float")
        ax.plot(cob_sub[i], cob_sub[j], "o", mfc="none", color="tab:blue", ms=11, mew=1.8, zorder=9, label="centre of buoyancy, submerged")
        if j == 2:
            # the float waterline in the hull frame: world z = 0
            xs = np.array([-0.5, 0.9]) if i == 0 else np.array([-0.8, 0.8])
            pts = []
            for x in xs:
                # solve (o + Rh @ [x or y on axis i, 0, zz])_z = 0 for zz
                e = np.zeros(3)
                e[i] = x
                zz = -(o[2] + Rh[2] @ e) / Rh[2, 2]
                pts.append(zz)
            ax.plot(xs, pts, color="tab:blue", lw=1.5, label="waterline at the float (settled row)")
        ax.set_aspect("equal")
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3)
    axs[0].set_xlim(-0.5, 0.75)
    axs[0].set_ylim(-0.3, 0.55)
    axs[2].set_ylim(-0.3, 0.55)
    axs[0].legend(fontsize=7, loc="upper left")
    bm = f0["body_mass"]
    g_ = mc.t5["geom"]
    txt = (f"T5 as built.  Dry {f0['mass']:.3f} kg (hull {bm['seg0_hull']:.3f} with struts, wings 4 x {bm['seg1_wing']:.4f}, "
           f"boom+tail {bm['seg5_strut'] + 2 * bm['seg6_wing']:.4f}, pontoons 2 x {bm['t5_pont_L']:.4f}); "
           f"fill {T5.F_NEUTRAL}: {f1['mass']:.3f} kg (water 2 x {f1['body_mass']['t5_tank_L']:.3f}).\n"
           f"Pontoons: r {g_['r']}, length {g_['L']}, at hull y +-{g_['ys']}, z {g_['zp']}, x {mc.t5['xp']:.3f} m; "
           f"volume 2 x {mc.v_pont * 1e3:.3f} L.  Buoyancy submerged {B_sub:.2f} N (b {B_sub / (f1['mass'] * G):.4f} at fill {T5.F_NEUTRAL}, "
           f"{B_sub / (f0['mass'] * G):.3f} dry); at the float {B_float:.2f} N = weight {f0['mass'] * G:.2f} N.\n"
           f"CoM dry ({com_dry[0]:.3f}, {com_dry[1]:.3f}, {com_dry[2]:.3f}), fill {T5.F_NEUTRAL} ({com_wet[0]:.3f}, {com_wet[1]:.3f}, {com_wet[2]:.3f}); "
           f"CoB float ({cob_float[0]:.3f}, {cob_float[2]:.3f}), submerged ({cob_sub[0]:.3f}, {cob_sub[2]:.3f}) (x, z; hull frame, m).  "
           f"Dots: fluid strips and slices (red = wet at the float).")
    fig.suptitle(txt, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(HERE / "figures" / "S0b_T5_built.png", dpi=100)
    out = dict(facts_dry=f0, facts_neutral=f1, com_dry_hull=com_dry.tolist(), com_neutral_hull=com_wet.tolist(),
               cob_float_hull=cob_float.tolist(), cob_submerged_hull=cob_sub.tolist(), B_float=B_float, B_submerged=B_sub,
               geom=g_, xp=mc.t5["xp"], calib=mc.t5, float_row="float_default" if rf else None)
    (HERE / "figures" / "S0b_T5_built.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("calib", "facts_dry", "facts_neutral")}, default=float))


def float_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R_ = {r["id"]: r for r in rows() if r.get("kind") == "float"}
    mc = Q._make("water", 0.0, "x4", 1e-4, z=Q.Z_FLOAT)
    m, d, mj = mc.m, mc.d, mc.mj
    cfg = Q.law_cfg()
    Pl = dict(LW.LEVEL, **cfg["P"])
    q_rest, _ = LW.targets_level(mc, 0.0, 1.0, Pl, osc=0.0)
    fig, axs = plt.subplots(4, 2, figsize=(15, 17), gridspec_kw=dict(width_ratios=[1.25, 1.75]))
    out = {}
    k = 0
    for mode in ("default", "wetting"):
        r = R_[f"float_{mode}"]
        mc.env.solver.entrainment = True if mode == "default" else "wetting"
        # the phase of the hover stroke (largest amplitude the law may use) at which a wing is lowest
        Pmax = dict(Pl, amp=cfg["gains"]["amp_lim"][1])
        best = (9e9, 0.0)
        for i in range(240):
            ph = 2 * math.pi * i / 240
            q, _ = LW.targets_level(mc, ph, 1.0, Pmax)
            d.qpos[:3], d.qpos[3:7], d.qpos[mc.qad] = r["hull_pos_end"], r["quat_end"], q
            mj.mj_forward(m, d)
            lo = P2.wing_low(mc)
            if lo < best[0]:
                best = (lo, ph)
        q_low, _ = LW.targets_level(mc, best[1], 1.0, Pmax)
        for title, q in ((f"({'ab'[k % 2] if mode == 'default' else 'cd'[k % 2]}) {mode}: at rest, settled 12 s (id float_{mode})", q_rest),
                         (f"({'ab'[(k + 1) % 2] if mode == 'default' else 'cd'[(k + 1) % 2]}) {mode}: the hover stroke at its lowest wing point (amp {Pmax['amp']}, phase {math.degrees(best[1]):.0f} deg)", q_low)):
            d.qpos[:3], d.qpos[3:7], d.qpos[mc.qad] = r["hull_pos_end"], r["quat_end"], q
            d.qvel[:] = 0.0
            mj.mj_forward(m, d)
            pos, sub, bid = strips(mc)
            Rh, o = d.xmat[mc.hull].reshape(3, 3), d.xpos[mc.hull]
            per = [int(((sub > 0) & (bid == b)).sum()) for b in (2, 3, 4, 5)]
            low = P2.wing_low(mc)
            att = P2._att(Rh)
            tail = sub[np.isin(bid, [7, 8])]
            tail_low = min(float((d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ (np.array([sx, sy, sz]) * m.geom_size[g]))[2])
                           for g in range(m.ngeom) if int(m.geom_bodyid[g]) in (7, 8)
                           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1))
            pont = sub[np.isin(bid, mc.pont)]
            hull = sub[bid == 1]
            num = dict(roll_pitch_yaw=[round(v, 2) for v in att], wet_strips_FL_FR_HL_HR=per, wing_lowest_corner_z=round(low, 4),
                       tail_lowest_corner_z=round(tail_low, 4), tail_strips_wet=int((tail > 0).sum()),
                       hull_and_strut_elements_wet=int((hull > 0).sum()), hull_elements=int(len(hull)),
                       pontoon_slice_sub=[round(float(v), 2) for v in pont], hull_origin_z=round(float(o[2]), 4),
                       com_z=round(float(mc.com()[2]), 4))
            out[title[:3] + " " + mode] = num
            for ax, (i, j, lab) in zip(axs[k], ((0, 2, "side: world x (nose to the right) against z, m"),
                                                 (1, 2, "front: world y against z, m"))):
                off = np.array([-float(o[0]) if i == 0 else 0.0, 0.0, 0.0])
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
                ax.set_ylim(-0.12, 0.6)
                ax.set_xlabel(lab)
                ax.grid(alpha=0.3)
            axs[k][0].set_title(title, fontsize=9, loc="left")
            axs[k][1].set_title(f"hull origin z {o[2]:+.3f} m, pitch {att[1]:+.1f}, roll {att[0]:+.1f} deg; wet main-wing strips {per}; "
                                f"lowest wing corner z {low:+.3f} m, lowest tail corner {tail_low:+.3f} m\n"
                                f"wet: tail strips {int((tail > 0).sum())}/{len(tail)}, hull+strut elements {int((hull > 0).sum())}/{len(hull)}, "
                                f"pontoon slices {int((pont > 0).sum())}/{len(pont)} (mean submerged {pont.mean():.2f})", fontsize=9, loc="left")
            k += 1
    fig.suptitle(f"S0b B9: T5 on the water, ballast out ({mc.mass:.3f} kg), both entrainment modes.  Waterline z = 0 (blue).  "
                 "Dots: the solver's strips and slices (red = wet); + the dry CoM.", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(HERE / "figures" / "S0b_T5_float.png", dpi=100)
    (HERE / "figures" / "S0b_T5_float.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def mission_fig(rid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = [x for x in rows() if x.get("id") == rid and x.get("kind") == "mission"][-1]
    a = np.array(r["trace"], dtype=float)
    t = a[:, 0]
    fig = plt.figure(figsize=(17, 15))
    gs = fig.add_gridspec(6, 1, height_ratios=[2.4, 1.3, 1.1, 0.9, 0.9, 0.9])
    ax0 = fig.add_subplot(gs[0])
    ph = r["phases_out"]
    # the model at instants: the end of each phase (and the middle of long ones)
    mc = Q._make("water", 0.0, "x4", 1e-4, z=Q.Z_FLOAT)
    m, d, mj = mc.m, mc.d, mc.mj
    Pl = dict(LW.LEVEL, **(r.get("law") or Q.law_cfg())["P"])
    q_r, _ = LW.targets_level(mc, 0.0, 1.0, Pl, osc=0.0)
    q_sink = q_r.copy()
    for tag in ("FL", "FR", "HL", "HR"):
        q_sink[mc.ix[(tag, "feather")]] = -0.5 * math.pi
    inst = []
    for p in ph:
        ts = [p["t1"]] if (p["t1"] - p["t0"]) < 8 else [0.5 * (p["t0"] + p["t1"]), p["t1"]]
        inst += [(tt, p) for tt in ts]
    span = 1.6
    for n_, (tt, p) in enumerate(inst):
        i = int(np.argmin(np.abs(t - tt)))
        z, roll, pitch = a[i, 3], a[i, 5], a[i, 6]
        d.qpos[:3] = (0.0, 0.0, 0.0)
        d.qpos[3:7] = P2._quat(roll, pitch)
        d.qpos[mc.qad] = q_sink if (a[i, 8] > 0.5 and z < -0.3) else q_r
        mj.mj_forward(m, d)
        d.qpos[:3] += np.array([0.0, 0.0, z]) - mc.com()
        mj.mj_forward(m, d)
        off = np.array([n_ * span + 0.6 - 0.235, 0.0, 0.0])
        outline(mc, ax0, 0, 2, off=off, alpha=0.35)
        ax0.text(n_ * span + 0.6, -3.75, f"{tt:.0f} s\n{p['kind']}", ha="center", fontsize=7)
    ax0.axhspan(-4.2, 0.0, color="tab:cyan", alpha=0.15, zorder=0)
    ax0.axhline(0.0, color="tab:blue", lw=1.2)
    ax0.set_xlim(-0.3, len(inst) * span + 0.2)
    ax0.set_ylim(-4.0, 2.8)
    ax0.set_aspect("equal")
    ax0.set_ylabel("z, m (side view at the recorded CoM height,\nroll and pitch; wings at a rest pose)")
    ax0.set_xticks([])
    axs = [fig.add_subplot(gs[k]) for k in range(1, 6)]
    axs[0].plot(t, a[:, 3], "k", label="CoM z")
    axs[0].axhline(0.0, color="tab:blue", lw=1)
    axs[0].set_ylabel("CoM z, m")
    axs[1].plot(t, a[:, 5], label="roll")
    axs[1].plot(t, a[:, 6], label="pitch (nose up +)")
    axs[1].plot(t, a[:, 7], label="yaw", lw=0.8)
    axs[1].set_ylabel("attitude, deg")
    axs[1].legend(fontsize=7, ncol=3, loc="upper right")
    axs[2].plot(t, a[:, 8], "tab:blue", label="pontoon fill")
    axs[2].plot(t, a[:, 13], "tab:red", lw=0.8, label="stroke amplitude (law)")
    axs[2].set_ylabel("fill; amp")
    axs[2].legend(fontsize=7, ncol=2, loc="upper right")
    axs[3].plot(t, a[:, 11], "tab:brown", label="contact, weights")
    axs[3].plot(t, a[:, 10], "tab:red", lw=0.8, label="wet main-wing strips (of 32)")
    axs[3].set_ylabel("contact W; wet strips")
    axs[3].legend(fontsize=7, ncol=2, loc="upper right")
    axs[4].plot(t, a[:, 12], "tab:purple")
    axs[4].set_ylabel("limiter steps,\ncumulative")
    axs[4].set_xlabel("t, s")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.set_xlim(0, t[-1])
        for p in ph:
            ax.axvline(p["t0"], color="0.5", lw=0.6, ls=":")
    for p in ph:
        axs[0].text(0.5 * (p["t0"] + p["t1"]), axs[0].get_ylim()[1], p["kind"], ha="center", va="bottom", fontsize=7)
    cl = r["clamped_by_phase"]
    fig.suptitle(f"S0b B11: T5 mission row {rid} ({r.get('mode')}, motor {r.get('motor', 'x4')}): "
                 f"{' > '.join(p['kind'] for p in ph)}\n"
                 f"limiter steps by phase {cl} (total {r['clamped_steps']}), QACC {r['bad_qacc']}, peak torque N m "
                 f"{ {k: round(v, 1) for k, v in r['tq_peak'].items()} }, contact peak {r['contact_peak_W']:.1f} W "
                 f"(wing {r['contact_peak_W_wing']:.1f}), not evidence: {r['not_evidence']}, abort: {r['abort']}",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(HERE / "figures" / "S0b_T5_mission.png", dpi=95)
    print("drawn", rid)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "built":
        built_fig()
    elif what == "float":
        float_fig()
    else:
        mission_fig(sys.argv[2])
