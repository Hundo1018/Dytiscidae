"""Figures of T5e (DESIGN_S0b_bold.md, Addenda 3 and 4), drawn from the built model and results/S0b_probes7.jsonl.

    MUJOCO_GL=disable PYTHONPATH=. python experiments/flapper/s0b_t5e_draw.py built
    ... s0b_t5e_draw.py surface <row id> [...]      (figures/S0b_T5e_surface.png)
    ... s0b_t5e_draw.py water <row id> [...]        (figures/S0b_T5e_water.png: track, speed, yaw error, depth)
    ... s0b_t5e_draw.py mission <row id>            (figures/S0b_T5e_mission.png)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0b_t5_draw as D5  # noqa: E402
import s0b_t5e_machine as TE  # noqa: E402
import s0b_t5e_static as S  # noqa: E402
import s0b_t5e_read as RD  # noqa: E402

D5.COL.update({"t5_pont_L": "tab:red", "t5_pont_R": "tab:brown", **{f"t5e_ctank_{i}": None for i in range(8)}})


def _extras(ax, mc, i, j):
    """Columns, booms and fin drawn explicitly (D5.outline has no cylinder or thin-box shape)."""
    geo, xp = mc.t5["geom"], mc.t5["xp"]
    th = np.linspace(0, 2 * math.pi, 40)
    for (a, b) in TE._booms(geo, xp):
        ax.plot([a[i], b[i]], [a[j], b[j]], color="0.35", lw=1.6, zorder=4)
    for (x, y, r, sg) in TE._cols(geo):
        if j == 2:
            c = x if i == 0 else y
            ax.fill([c - r, c + r, c + r, c - r], [geo["cz0"], geo["cz0"], geo["cz1"], geo["cz1"]], color="tab:cyan", alpha=0.6, ec="teal", lw=1.2, zorder=5)
        else:
            ax.fill(x + r * np.cos(th), y + r * np.sin(th), color="tab:cyan", alpha=0.9, ec="teal", lw=1.2, zorder=8)
    if (i, j) == (0, 2):
        ax.fill([geo["fx0"], geo["fx1"], geo["fx1"], geo["fx0"]], [geo["fz0"], geo["fz0"], geo["fz1"], geo["fz1"]], color="tab:orange", alpha=0.7, ec="k", lw=1, zorder=6)
    elif (i, j) == (0, 1):
        ax.plot([geo["fx0"], geo["fx1"]], [0, 0], color="tab:orange", lw=4, zorder=9)
    else:
        ax.plot([0, 0], [geo["fz0"], geo["fz1"]], color="tab:orange", lw=4, zorder=9)


def built_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, mc = S.table(None, pose_names=("raised",), fills=[0.0, 0.5, 0.8], write=False)
    wl = {r["fill"]: r["hull_z"] for r in rows}
    cl = {}
    for ln in open(HERE / "results" / "t5e" / "clear_A.log"):
        p = ln.split()
        if len(p) > 4 and p[1] == "-":
            cl[(p[0], p[2])] = (float(p[3]), " ".join(p[6:]))
    worst = min(cl.items(), key=lambda kv: kv[1][0])
    m, d, mj = mc.m, mc.d, mc.mj
    P = S.poses(mc)
    d.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    d.qpos[mc.qad] = P["raised"]
    mj.mj_forward(m, d)
    o, Rh = d.xpos[mc.hull].copy(), d.xmat[mc.hull].reshape(3, 3).copy()
    fig = plt.figure(figsize=(21, 13))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
    views = ((0, 2, "side: hull x (nose right), hull z, m"), (0, 1, "top: hull x, hull y, m"), (1, 2, "front: hull y, hull z, m"))
    for c, (i, j, lab) in enumerate(views):
        ax = fig.add_subplot(gs[0, c])
        D5.outline(mc, ax, i, j, frame=(o, Rh))
        _extras(ax, mc, i, j)
        if j == 2:
            for f, ls in ((0.0, "-"), (0.5, "--"), (0.8, ":")):
                ax.axhline(-wl[f], color="tab:blue", lw=1.4, ls=ls, label=f"waterline, fill {f} (hull origin {wl[f]:+.3f} m)")
            if c == 0:
                ax.legend(fontsize=7, loc="upper left")
            ax.set_ylim(-0.33, 0.62)
        ax.set_aspect("equal")
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3)
    z = np.load(HERE / "results" / "S0b_T5e_envelope_A.npz")
    geo = mc.t5["geom"]
    dist, occ = S.free_distance(z["zmin"], z["zmax"], geo["cz0"], geo["cz1"])
    ax = fig.add_subplot(gs[1, 0:2])
    ext = (S.X0, S.X1, S.Y0, S.Y1)
    im = ax.imshow(np.where(occ, np.nan, dist).T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=0.3, aspect="equal")
    ax.imshow(np.where(occ, 1.0, np.nan).T, origin="lower", extent=ext, cmap="Reds", vmin=0, vmax=1.4, aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="horizontal distance to the wings' and tail's swept volume\n(heights -0.23..+0.20 m, 0.03 m margin), m")
    th = np.linspace(0, 2 * math.pi, 40)
    for (x, y, r, sg) in TE._cols(geo):
        if sg > 0:
            ax.plot(x + r * np.cos(th), y + r * np.sin(th), "c", lw=2)
            ax.plot(x + (r + 0.03) * np.cos(th), y + (r + 0.03) * np.sin(th), "c--", lw=0.8)
    for (a, b) in TE._booms(geo, mc.t5["xp"]):
        if a[1] > 0:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="w", lw=1.5)
    hl = geo["L"] / 2
    ax.plot([mc.t5["xp"] - hl, mc.t5["xp"] + hl], [geo["ys"], geo["ys"]], color="k", lw=3, alpha=0.6)
    ax.set_title("left half, top view: red = cells a wing or tail surface sweeps at the columns' heights over the laws' full command ranges; cyan = columns (dashed: + 0.03 m); white = booms (they run under the wings at z -0.23); black = pontoon", fontsize=8)
    ax.set_xlabel("hull x, m")
    ax.set_ylabel("|hull y|, m")
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    fsb = TE.free_surface_bound(mc)
    lines = [f"T5e, layout A: T5b + keel fin + 4 columns r {geo['cols'][0][2]} m, hull z {geo['cz0']}..{geo['cz1']},",
             f"at (x {geo['cols'][0][0]}, y +-{geo['cols'][0][1]}) and (x {geo['cols'][1][0]}, y +-{geo['cols'][1][1]}), on booms from the pontoon ends.",
             f"dry mass {mc.t5['m_dry']:.3f} kg: column shells {mc.t5['m_col_shells'] * 1e3:.0f} g, booms {mc.t5['m_booms'] * 1e3:.0f} g, fin {mc.t5['m_fin'] * 1e3:.0f} g;",
             f"floodable {mc.t5['v_flood'] * 1e3:.2f} L (pontoons {2 * mc.t5['v_pont_each'] * 1e3:.2f}, columns {mc.t5['v_cols'] * 1e3:.2f}).",
             f"waterlines (hull origin over water): fill 0 {wl[0.0]:+.3f}, 0.5 {wl[0.5]:+.3f}, 0.8 {wl[0.8]:+.3f} m;",
             "fill 0.95 neutral, fill 1 sinks.", "",
             f"SMALLEST CLEARANCE over the laws' command ranges: {worst[1][0]:+.3f} m, {worst[0][0]} to {worst[0][1]}", f"  at: {worst[1][1]}"]
    for k in (("wing", "dorsal"), ("wing", "pontoon"), ("wing", "boom"), ("wing", "column"), ("wing", "fin"), ("tail", "fin"), ("tail", "boom"), ("tail", "column")):
        if k in cl:
            lines.append(f"  {k[0]} to {k[1]}: {cl[k][0]:+.3f} m")
    lines += ["", f"tank model: slug from each column's foot, valved; neglected: free surface inside {fsb['inside_all']:.3f} N m/rad;",
              f"if open to each other: roll -{fsb['open_roll']:.1f}, pitch -{fsb['open_pitch']:.1f} N m/rad."]
    ax.text(-0.1, 1.0, "\n".join(lines), va="top", fontsize=9, family="monospace")
    fig.suptitle("S0b T5e (layout A) as built", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "figures" / "S0b_T5e_built.png", dpi=80)
    print("drawn built", worst, wl)


def _mode(r):
    return "wetting" if r.get("entrainment") == "wetting" else "default"


def surface_fig(*rids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = {r["id"]: r for r in RD.rows() if r.get("kind") == "mission"}
    n = len(rids)
    fig, axs = plt.subplots(6, n, figsize=(4.6 * n, 15), squeeze=False)
    names = ["CoM z, m", "climb rate, m/s", "pitch (nose up +), deg", "roll, deg", "fill", "wet wing strips (of 32)"]
    for c, rid in enumerate(rids):
        r = R[rid]
        a = np.array([x[:14] for x in r["trace"]], dtype=float)
        p = [q for q in r["phases_out"] if q["kind"] == "surface"][0]
        s = (a[:, 0] >= p["t0"] - 1.0) & (a[:, 0] <= p["t1"] + 8.0)
        t = a[s, 0] - p["t0"]
        mtr = RD.surf_metrics(r)[0]
        for k, col in enumerate((3, 4, 6, 5, 8, 10)):
            ax = axs[k][c]
            ax.plot(t, a[s, col], "k", lw=0.9)
            ax.axvline(p["t1"] - p["t0"], color="tab:green", ls="--", lw=1)
            if mtr.get("t_first_dry") is not None:
                ax.axvline(mtr["t_first_dry"], color="tab:red", ls=":", lw=1)
            if col in (5, 6):
                ax.axhspan(-10, 10, color="tab:green", alpha=0.12)
            if col == 4:
                ax.axhline(0.08, color="tab:orange", ls="--", lw=0.8)
            if c == 0:
                ax.set_ylabel(names[k], fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
        fl = [q for q in r["phases_out"] if q["kind"] == "fly"]
        held = bool(fl and fl[-1].get("hold_z_rel") and not r["abort"])
        axs[0][c].set_title(f"{rid} ({_mode(r)})\npeak roll {mtr.get('peak_roll', float('nan')):.1f}, pitch {mtr.get('peak_pitch', float('nan')):.1f} deg from the first dry strip to 2 s after afloat\n"
                            f"lift-off held: {'yes' if held else 'NO'}; limiter {r['clamped_steps']}, QACC {r['bad_qacc']}", fontsize=7, loc="left")
        axs[5][c].set_xlabel("t in the surfacing phase, s (red: first dry wing strip, green: afloat)", fontsize=7)
    fig.suptitle("S0b T5e B25: surfacing by the heave-rate ballast rise; green band = the 10 deg target, orange = the 0.08 m/s cap", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "figures" / "S0b_T5e_surface.png", dpi=85)
    print("drawn surface", rids)


def water_fig(*rids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = {r["id"]: r for r in RD.rows() if r.get("kind") == "mission"}
    n = len(rids)
    fig, axs = plt.subplots(5, n, figsize=(5.2 * n, 16), squeeze=False)
    for c, rid in enumerate(rids):
        r = R[rid]
        a = np.array([x[:14] for x in r["trace"]], dtype=float)
        sw = [q for q in r["phases_out"] if q["kind"] == "swim"]
        cols = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
        ax = axs[0][c]
        for k, p in enumerate(sw):
            s = (a[:, 0] >= p["t0"]) & (a[:, 0] <= p["t1"])
            ax.plot(a[s, 1], a[s, 2], color=cols[k % 4], lw=1.2, label=f"swim phase {k + 1}")
        ax.plot(a[0, 1], a[0, 2], "ko")
        ax.set_aspect("equal")
        ax.set_xlabel("x, m")
        ax.set_ylabel("track: y, m")
        ax.legend(fontsize=7)
        txt = []
        for k, p in enumerate(sw):
            tr = np.array(p["swim_trace"])
            T = tr[-1, 0]
            for row, col, lab in ((1, 2, "displacement along the command, m"), (2, 4, "yaw error, deg"), (3, 3, "CoM z, m")):
                axs[row][c].plot(p["t0"] + tr[:, 0], tr[:, col], color=cols[k % 4])
                axs[row][c].set_ylabel(lab, fontsize=8)
            txt.append(f"phase {k + 1}: {p['disp_along_cmd'] / T:+.3f} m/s over {T:.0f} s, end yaw error {p['yaw_err_end_deg']:+.0f} deg, z {p['z_range'][0]:.2f}..{p['z_range'][1]:.2f}")
        s = (a[:, 0] >= sw[0]["t0"]) & (a[:, 0] <= sw[-1]["t1"])
        axs[4][c].plot(a[s, 0], a[s, 5], label="roll")
        axs[4][c].plot(a[s, 0], a[s, 6], label="pitch")
        axs[4][c].legend(fontsize=7)
        axs[4][c].set_ylabel("attitude, deg", fontsize=8)
        axs[4][c].set_xlabel("t, s")
        axs[0][c].set_title(f"{rid} ({_mode(r)}); limiter {r['clamped_steps']}, QACC {r['bad_qacc']}\n" + "\n".join(txt), fontsize=7, loc="left")
        for ax in axs[:, c]:
            ax.grid(alpha=0.3)
    fig.suptitle("S0b T5e B26-B27: swim on the heading, forward U-turn on differential thrust, swim on the antipode", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "figures" / "S0b_T5e_water.png", dpi=85)
    print("drawn water", rids)


def mission_fig(rid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = {r["id"]: r for r in RD.rows() if r.get("kind") == "mission"}
    r = R[rid]
    a = np.array([x[:14] for x in r["trace"]], dtype=float)
    t = a[:, 0]
    ph = r["phases_out"]
    fig, axs = plt.subplots(6, 1, figsize=(17, 16), sharex=True)
    axs[0].plot(t, a[:, 3], "k")
    axs[0].axhline(0.0, color="tab:blue", lw=1)
    axs[0].set_ylabel("CoM z, m")
    axs[1].plot(t, a[:, 5], label="roll")
    axs[1].plot(t, a[:, 6], label="pitch (nose up +)")
    axs[1].legend(fontsize=7, ncol=2)
    axs[1].set_ylabel("attitude, deg")
    axs[2].plot(t, a[:, 7], "tab:green")
    axs[2].set_ylabel("yaw, deg")
    axs[3].plot(t, a[:, 8], "tab:blue")
    axs[3].set_ylabel("fill")
    axs[4].plot(t, a[:, 10], "tab:red")
    axs[4].set_ylabel("wet wing strips")
    axs[5].plot(t, a[:, 12], "tab:purple")
    axs[5].set_ylabel("limiter steps, cumulative")
    axs[5].set_xlabel("t, s")
    for ax in axs:
        ax.grid(alpha=0.3)
        for p in ph:
            ax.axvline(p["t0"], color="0.5", lw=0.6, ls=":")
    for p in ph:
        axs[0].text(0.5 * (p["t0"] + p["t1"]), axs[0].get_ylim()[1], p["kind"], ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"S0b T5e B29: {rid} ({_mode(r)}): {' > '.join(p['kind'] for p in ph)}\nlimiter steps {r['clamped_steps']}, QACC {r['bad_qacc']}, abort {r['abort']}, "
                 f"peak torque {({k: round(v) for k, v in r['tq_peak'].items()})}, X2 share {r.get('x2_share')}", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "figures" / "S0b_T5e_mission.png", dpi=85)
    print("drawn mission", rid)


if __name__ == "__main__":
    w = sys.argv[1]
    if w == "built":
        built_fig()
    elif w == "surface":
        surface_fig(*sys.argv[2:])
    elif w == "water":
        water_fig(*sys.argv[2:])
    else:
        mission_fig(sys.argv[2])
