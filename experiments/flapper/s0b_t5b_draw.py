"""Figures of T5b, drawn from the built model and from rows of results/S0b_probes4.jsonl.

    MUJOCO_GL=disable PYTHONPATH=. python experiments/flapper/s0b_t5b_draw.py built
    ... s0b_t5b_draw.py water <fwd row id> <antipode row id>      (figures/S0b_T5b_water.png)
    ... s0b_t5b_draw.py mission <row id>                          (figures/S0b_T5b_mission.png)
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
import s0b_t5_draw as D5  # noqa: E402   (outline, strips, _cob; the T5 figures' helpers, unchanged)
import s0b_t5b_machine as T5B  # noqa: E402
import s0b_t5b_probe as Q  # noqa: E402
import s0b_machine as T  # noqa: E402

G = 9.80665
D5.COL.update({"t5_pont_L": "tab:red", "t5_pont_R": "tab:brown"})
MOD = "s0b_t5b_machine"


def rows():
    return [json.loads(x) for x in open(HERE / "results" / "S0b_probes4.jsonl")]


def _outline_dorsal(mc, ax, i, j, frame):
    """The dorsal float (a hull-body geom) is drawn by D5.outline as a capsule of the hull body; nothing to add."""


def built_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rf = {r["id"]: r for r in rows()}.get("b14z_float_default")
    mc = Q._make("water", 0.0, "x4", 1e-4, z=0.16, mod=MOD)
    D5.T5 = T5B                                       # D5.outline needs only geometry; facts and fills come from T5b
    m, d, mj = mc.m, mc.d, mc.mj
    Pl = dict(LW.LEVEL, **Q.law_cfg_b()["P"])
    q_r, _ = LW.targets_level(mc, 0.0, 1.0, Pl, osc=0.0)
    d.qpos[mc.qad] = q_r
    if rf:
        d.qpos[:3], d.qpos[3:7] = rf["hull_pos_end"], rf["quat_end"]
    d.qvel[:] = 0.0
    mj.mj_forward(m, d)
    o, Rh = d.xpos[mc.hull].copy(), d.xmat[mc.hull].reshape(3, 3).copy()
    hf = lambda p: (np.asarray(p) - o) @ Rh
    com_dry = hf(mc.com())
    cob_float, B_float = D5._cob(mc)
    cob_float = hf(cob_float)
    f0 = T5B.facts(mc)
    T5B.set_fill(mc, T5B.F_NEUTRAL)
    z_keep = d.qpos[2]
    d.qpos[2] = -5.0
    mj.mj_forward(m, d)
    o2, R2 = d.xpos[mc.hull].copy(), d.xmat[mc.hull].reshape(3, 3).copy()
    cob_sub, B_sub = D5._cob(mc)
    cob_sub = (cob_sub - o2) @ R2
    com_wet = (mc.com() - o2) @ R2
    f1 = T5B.facts(mc)
    Kb = T5B.righting(mc)
    T5B.set_fill(mc, 0.0)
    d.qpos[2] = z_keep
    mj.mj_forward(m, d)
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.6), gridspec_kw=dict(width_ratios=[1.25, 1.45, 1.45]))
    views = ((0, 2, "side: hull x (nose right) against hull z, m"), (0, 1, "top: hull x against hull y, m"),
             (1, 2, "front: hull y against hull z, m"))
    for ax, (i, j, lab) in zip(axs, views):
        D5.outline(mc, ax, i, j, frame=(o, Rh))
        pos, sub, bid = D5.strips(mc)
        ph = (pos - o) @ Rh
        wet = sub > 0
        ax.plot(ph[~wet, i], ph[~wet, j], ".", color="k", ms=3, zorder=6)
        ax.plot(ph[wet, i], ph[wet, j], ".", color="red", ms=6, zorder=7)
        ax.plot(com_dry[i], com_dry[j], "k+", ms=15, mew=2.4, zorder=9, label="CoM, ballast out")
        ax.plot(com_wet[i], com_wet[j], "+", color="tab:blue", ms=15, mew=2.4, zorder=9, label=f"CoM, fill {T5B.F_NEUTRAL}")
        ax.plot(cob_float[i], cob_float[j], "o", mfc="none", color="k", ms=11, mew=1.8, zorder=9, label="centre of buoyancy at the float")
        ax.plot(cob_sub[i], cob_sub[j], "o", mfc="none", color="tab:blue", ms=11, mew=1.8, zorder=9, label="centre of buoyancy, submerged")
        if j == 2:
            xs = np.array([-0.5, 0.9]) if i == 0 else np.array([-0.8, 0.8])
            pts = []
            for x in xs:
                e = np.zeros(3)
                e[i] = x
                pts.append(-(o[2] + Rh[2] @ e) / Rh[2, 2])
            ax.plot(xs, pts, color="tab:blue", lw=1.5, label="waterline at the float (b14z_float_default)")
        ax.set_aspect("equal")
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3)
    axs[0].set_xlim(-0.5, 0.75)
    axs[0].set_ylim(-0.3, 0.6)
    axs[2].set_ylim(-0.3, 0.6)
    axs[0].legend(fontsize=7, loc="upper left")
    bm, g_ = f0["body_mass"], mc.t5["geom"]
    txt = (f"T5b as built = T5 + a dorsal float.  Dry {f0['mass']:.3f} kg (T5 1.336); at fill {T5B.F_NEUTRAL} {f1['mass']:.3f} kg, "
           f"buoyancy submerged {B_sub:.2f} N (b {B_sub / (f1['mass'] * G):.4f}).  Pontoons r {g_['r']}, L {g_['L']}, y +-{g_['ys']}, z {g_['zp']}, "
           f"x {mc.t5['xp']:.3f}; shells 2 x {bm['t5_pont_L']:.3f} kg (derived by neutrality).\n"
           f"Dorsal float: radius {g_['dr']}, length {g_['dL']}, centre line, hull z {g_['dz']}, x {mc.t5['xp']:.3f}; volume "
           f"{(math.pi * g_['dr'] ** 2 * (g_['dL'] - 2 * g_['dr']) + 4 / 3 * math.pi * g_['dr'] ** 3) * 1e3:.2f} L.   Righting stiffness fully submerged at fill {T5B.F_NEUTRAL}: "
           f"roll {Kb['roll']:.2f}, pitch {Kb['pitch']:.2f} N m/rad (T5 0.34).\n"
           f"CoM dry ({com_dry[0]:.3f}, {com_dry[1]:.3f}, {com_dry[2]:.3f}), fill {T5B.F_NEUTRAL} ({com_wet[0]:.3f}, {com_wet[1]:.3f}, {com_wet[2]:.3f}); "
           f"CoB float ({cob_float[0]:.3f}, {cob_float[2]:.3f}), submerged ({cob_sub[0]:.3f}, {cob_sub[2]:.3f}) (x, z; hull frame, m).  "
           f"Dots: fluid strips and slices (red = wet at the float).")
    fig.suptitle(txt, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(HERE / "figures" / "S0b_T5b_built.png", dpi=100)
    out = dict(mass_dry=f0["mass"], mass_neutral=f1["mass"], com_dry_hull=com_dry.tolist(), com_neutral_hull=com_wet.tolist(),
               cob_float_hull=cob_float.tolist(), cob_submerged_hull=cob_sub.tolist(), B_float=B_float, B_submerged=B_sub,
               K=Kb, geom=g_, inertia_dry=f0["inertia_hull_diag"], inertia_neutral=f1["inertia_hull_diag"], body_mass=f0["body_mass"])
    (HERE / "figures" / "S0b_T5b_built.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float))


def water_fig(*rids):
    """Swim rows (heading and antipode, both modes: four columns by default): displacement along the commanded heading, speed,
    depth, yaw error, attitude.  A row id ending _rev_ is the antipode (reverse thrust), _fwd_ the heading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = {r["id"]: r for r in rows()}
    nc = len(rids)
    fig, axs = plt.subplots(5, nc, figsize=(5.2 * nc, 15), sharex="col", squeeze=False)
    sp = {}
    for c, rid in enumerate(rids):
        r = R[rid]
        ph = [p for p in r["phases_out"] if p["kind"] == "swim"][0]
        sp[rid] = ph
        a = np.array(ph["swim_trace"])                        # t v_cmd x_cmd z yaw_err_deg x_h0
        t = a[:, 0]
        tr = np.array(r["trace"])
        sel = tr[tr[:, 9] == ph["i"]]
        axs[0][c].plot(t, a[:, 2], "k")
        axs[0][c].set_ylabel("displacement along\nthe commanded direction, m")
        axs[1][c].plot(t, a[:, 1], "k", lw=0.6)
        axs[1][c].axhline(0.3, color="tab:green", ls="--", lw=1, label="target 0.3 m/s")
        axs[1][c].set_ylabel("speed along the command,\nsampled each 0.1 s, m/s")
        axs[1][c].legend(fontsize=7)
        axs[2][c].plot(t, a[:, 3], "tab:blue")
        axs[2][c].set_ylabel("CoM depth z, m")
        axs[3][c].plot(t, a[:, 4], "tab:red")
        axs[3][c].set_ylabel("yaw error to the\ncommanded heading, deg")
        if len(sel):
            axs[4][c].plot(sel[:, 0] - ph["t0"], sel[:, 5], label="roll")
            axs[4][c].plot(sel[:, 0] - ph["t0"], sel[:, 6], label="pitch (nose up +)")
            axs[4][c].legend(fontsize=7)
        axs[4][c].set_ylabel("attitude, deg")
        axs[4][c].set_xlabel("t in the swim phase, s")
        axs[0][c].set_title(f"{rid} ({r['mode']})\nwhole-run speed {ph['disp_along_cmd'] / a[-1, 0]:+.3f}, last window {ph['v_mean_along_cmd']:+.3f} m/s; disp {ph['disp_along_cmd']:+.2f} m\n"
                            f"limiter {r['clamped_steps']}, QACC {r['bad_qacc']}, peak N m { {k: round(v) for k, v in r['tq_peak'].items()} }", fontsize=7, loc="left")
        for ax in axs[:, c]:
            ax.grid(alpha=0.3)
    msg = []
    for mode in ("default", "wetting"):
        f = [i for i in rids if i.endswith("_fwd_" + mode)]
        v = [i for i in rids if i.endswith("_rev_" + mode)]
        if f and v:
            whole = [sp[i]["disp_along_cmd"] / sp[i]["swim_trace"][-1][0] for i in (f[0], v[0])]
            last = [sp[i]["v_mean_along_cmd"] for i in (f[0], v[0])]
            msg.append(f"{mode}: mean signed speed {0.5 * sum(whole):+.3f} (whole run), {0.5 * sum(last):+.3f} (last window)")
    fig.suptitle("S0b B16: T5b free swim, heading (fwd) and antipode (rev, wings turned over), same start; " + "; ".join(msg) + "; target 0.3 m/s",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "figures" / "S0b_T5b_water.png", dpi=90)
    print("drawn", rids)


def mission_fig(rid):
    import s0b_t5_draw as M
    M.rows = rows
    M.Q = Q
    M.Q.Z_FLOAT = 0.16
    M.P2 = P2
    D = M.mission_fig
    # D5.mission_fig builds a T5 machine for the drawn instants and saves S0b_T5_mission.png; T5b's version is that
    # drawing with the T5b builder and its own file name (kept in one place: it is the same code path).
    import types
    src = Path(M.__file__).read_text()
    src = src.replace('Q._make("water", 0.0, "x4", 1e-4, z=Q.Z_FLOAT)', 'Q._make("water", 0.0, "x4", 1e-4, z=0.16, mod="s0b_t5b_machine")')
    src = src.replace('"S0b_T5_mission.png"', '"S0b_T5b_mission.png"').replace("S0b B11: T5 mission row", "S0b B17: T5b mission row")
    src = src.replace("Q.law_cfg()", "Q.law_cfg_b()")
    ns = dict(vars(M))
    exec(compile(src, "s0b_t5_draw_t5b", "exec"), ns)
    ns["rows"] = rows
    ns["Q"] = Q
    ns["mission_fig"](rid)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "built":
        built_fig()
    elif what == "water":
        water_fig(*sys.argv[2:])
    else:
        mission_fig(sys.argv[2])
