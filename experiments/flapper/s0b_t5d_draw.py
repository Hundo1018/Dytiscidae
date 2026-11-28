"""Figures of T5d (DESIGN_S0b_bold.md, Addendum 3), drawn from the built model and results/S0b_probes6.jsonl.

    MUJOCO_GL=disable PYTHONPATH=. python experiments/flapper/s0b_t5d_draw.py built ['<geom json>']

built: figures/S0b_T5d_built.png: T5d as built (frozen layout: four columns at the pontoon ends up to the dorsal float's axis, the
       keel fin), side / top / front in the hull frame with the wings in the float pose (raised) and flat, the waterlines at
       fill 0, 0.5 and 0.9 (fill 0.95 is neutral and fill 1 sinks: no waterline) from the static table's equilibria, and the map of
       the wings' lower envelope (`s0b_t5d_envelope.py`) with the columns' footprints.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0b_t5_draw as D5  # noqa: E402
import s0b_t5d_envelope as EV  # noqa: E402
import s0b_t5d_machine as TD  # noqa: E402
import s0b_t5d_static as S  # noqa: E402

D5.COL.update({"t5_pont_L": "tab:red", "t5_pont_R": "tab:brown", **{n: None for n in TD.CTANK}})


def built_fig(g=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, mc = S.table(g, pose_names=("raised",), fills=[0.0, 0.5, 0.9], write=False)
    wl = {r["fill"]: r["hull_z"] for r in rows}
    cl = S.clearance(g, nph=48)
    worst = min(v["wing_col"] for v in cl.values())
    m, d, mj = mc.m, mc.d, mc.mj
    P = S.poses(mc)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
    views = ((0, 2, "side: hull x (nose right), hull z, m"), (0, 1, "top: hull x, hull y, m"), (1, 2, "front: hull y, hull z, m"))
    for row, pn in enumerate(("raised", "flat")):
        if row == 1:
            break
        d.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
        d.qpos[mc.qad] = P[pn]
        mj.mj_forward(m, d)
        o, Rh = d.xpos[mc.hull].copy(), d.xmat[mc.hull].reshape(3, 3).copy()
        for c, (i, j, lab) in enumerate(views):
            ax = fig.add_subplot(gs[0, c])
            D5.outline(mc, ax, i, j, frame=(o, Rh))
            # columns and fin drawn explicitly (cylinders and a thin box are not in D5.outline's shapes)
            geo = mc.t5["geom"]
            for (x, y, sg) in TD._cols(geo, mc.t5["xp"]):
                r = geo["cr"]
                if j == 2:
                    a = x if i == 0 else y
                    ax.fill([a - r, a + r, a + r, a - r], [geo["cz0"], geo["cz0"], geo["cz1"], geo["cz1"]], color="tab:cyan", alpha=0.55, ec="teal", lw=1.2, zorder=5)
                else:
                    th = np.linspace(0, 2 * math.pi, 40)
                    ax.fill(x + r * np.cos(th), y + r * np.sin(th), color="tab:cyan", alpha=0.8, ec="teal", lw=1.2, zorder=8)
            if geo.get("fx1", 0) > geo.get("fx0", 0):
                if (i, j) == (0, 2):
                    ax.fill([geo["fx0"], geo["fx1"], geo["fx1"], geo["fx0"]], [geo["fz0"], geo["fz0"], geo["fz1"], geo["fz1"]], color="tab:orange", alpha=0.7, ec="k", lw=1, zorder=6)
                elif (i, j) == (0, 1):
                    ax.plot([geo["fx0"], geo["fx1"]], [0, 0], color="tab:orange", lw=4, zorder=9)
                else:
                    ax.plot([0, 0], [geo["fz0"], geo["fz1"]], color="tab:orange", lw=4, zorder=9)
            if j == 2:
                for f, ls in ((0.0, "-"), (0.5, "--"), (0.9, ":")):
                    ax.axhline(-wl[f], color="tab:blue", lw=1.4, ls=ls, label=f"waterline, fill {f} (hull origin {wl[f]:+.3f} m)")
                if c == 0:
                    ax.legend(fontsize=7, loc="upper left")
            ax.set_aspect("equal")
            ax.set_xlabel(lab)
            ax.grid(alpha=0.3)
            if c != 1:
                ax.set_ylim(-0.32, 0.62)
    # envelope maps
    for c, (nm, f) in enumerate((("all strokes and control inputs", "S0b_T5d_envelope.npz"), ("nominal strokes only", "S0b_T5d_envelope_nominal.npz"))):
        ax = fig.add_subplot(gs[1, c])
        z = np.load(HERE / "results" / f)["zmin_all"]
        zz = np.where(z > 8, np.nan, z)
        im = ax.imshow(zz.T, origin="lower", extent=(EV.X0, EV.X1, EV.Y0, EV.Y1), cmap="viridis", vmin=-0.25, vmax=0.25, aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.8, label="lowest wing point over the cell, hull z, m")
        geo = mc.t5["geom"]
        th = np.linspace(0, 2 * math.pi, 40)
        for (x, y, sg) in TD._cols(geo, mc.t5["xp"]):
            if sg > 0:
                ax.plot(x + geo["cr"] * np.cos(th), y + geo["cr"] * np.sin(th), "r", lw=1.5)
        hc = geo["L"] / 2
        ax.plot([mc.t5["xp"] - hc, mc.t5["xp"] + hc], [geo["ys"], geo["ys"]], "r--", lw=1)
        ax.set_title(f"wings' lower envelope, {nm}\n(white = never over the cell; red = columns, pontoon axis)", fontsize=8)
        ax.set_xlabel("hull x, m")
        ax.set_ylabel("|hull y|, m")
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    fsb = TD.free_surface_bound(mc)
    txt = (f"T5d, frozen layout (Addendum 3): T5b + 4 columns r {geo['cr']} m, hull z {geo['cz0']}..{geo['cz1']}, \nat x = xp {geo['cdx'][0]:+.2f} / {geo['cdx'][1]:+.2f}, y = +-{geo['cy']};\n"
           f"keel fin x {geo['fx0']}..{geo['fx1']}, z {geo['fz0']}..{geo['fz1']} (y = 0).\n"
           f"dry mass {mc.t5['m_dry']:.3f} kg (T5b 2.115); column shell {mc.t5['m_col_shell_each'] * 1e3:.1f} g each, fin {mc.t5['m_fin'] * 1e3:.1f} g;\n"
           f"floodable volume {mc.t5['v_flood'] * 1e3:.2f} L (pontoons 2 x {mc.t5['v_pont_each'] * 1e3:.2f}, columns 4 x {mc.t5['v_col_each'] * 1e3:.2f}).\n"
           f"waterlines (hull origin over water): fill 0 {wl[0.0]:+.3f}, 0.5 {wl[0.5]:+.3f}, 0.9 {wl[0.9]:+.3f} m;\nfill 0.95 neutral, fill 1 sinks.\n\n"
           f"SMALLEST WING-COLUMN DISTANCE over the hover and swim strokes and poses: {worst:+.3f} m\n(negative = the wing passes through the column).\n\n"
           f"tank model: slug from the column's foot; neglected free surface\ninside columns {fsb['inside_all']:.3f} N m/rad;\n"
           f"if the columns were open to each other: roll -{fsb['open_roll']:.2f}, pitch -{fsb['open_pitch']:.2f} N m/rad.")
    ax.text(-0.25, 1.0, txt, va="top", fontsize=8.5, family="monospace")
    fig.suptitle("S0b T5d as built from the frozen description (not flown: probe S stops on wing clearance)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "figures" / "S0b_T5d_built.png", dpi=80)
    print("drawn built; worst wing_col", worst, "waterlines", wl)


if __name__ == "__main__":
    if sys.argv[1] == "built":
        built_fig(json.loads(sys.argv[2]) if len(sys.argv) > 2 else None)
