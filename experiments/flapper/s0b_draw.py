"""S0b B0 figure: T4 as built (`s0b_machine.make`), level, every joint at 0.
Side (x-z) and top (x-y) views in the hull frame.  Everything drawn is read
from the compiled model: geoms (hull meshes, wing and tail boxes, boom),
the keel point mass, each hinge's anchor and axis (stroke red, feather green,
sweep blue, tail grey), the dry centre of mass (+) and the solver's centre of
buoyancy (o).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/s0b_draw.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0b_machine as T  # noqa: E402

COL = {"seg0_hull": "0.6", "seg1_wing": "tab:blue", "seg2_wing": "tab:orange", "seg3_wing": "tab:purple",
       "seg4_wing": "tab:pink", "seg5_strut": "0.3", "seg6_wing": "tab:green", "seg7_wing": "tab:olive"}
JCOL = dict(stroke="red", feather="green", sweep="blue", tail="0.4")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mc = T.make("water", b=1.0, pose=dict(x=-30.0, y=0.0, z=-5.0, vx=0.0, vz=0.0))
    m, d, mj = mc.m, mc.d, mc.mj
    rep = T.report(mc)
    B, rb, _, _ = T.static_buoyancy(mc)
    mj.mj_forward(m, d)
    R = d.xmat[mc.hull].reshape(3, 3)
    o = d.xpos[mc.hull].copy()
    H = lambda p: (np.asarray(p) - o) @ R             # world -> hull frame
    fig, axs = plt.subplots(2, 1, figsize=(13, 15), gridspec_kw=dict(height_ratios=[1, 4]))
    notes = []
    for ax, (i, j, lab) in zip(axs, ((0, 2, "side: hull x (nose +) against hull z (dorsal +), m"),
                                     (0, 1, "top: hull x (nose +) against hull y (left +), m"))):
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b == 0:
                continue
            bn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b)
            gn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
            c, Rg, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
            ty = int(m.geom_type[g])
            if gn == "s0b_keel":
                p = H(c)
                ax.plot(p[i], p[j], "s", color="navy", ms=10, zorder=6)
                ax.annotate(f"ballast {mc.t4['keel_kg']*1000:.0f} g", (p[i], p[j]), xytext=(8, -14),
                            textcoords="offset points", color="navy")
                continue
            if ty == 6:                                # box
                cs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * s
            elif ty == 4:                              # ellipsoid: its outline in each principal plane
                th = np.linspace(0, 2 * np.pi, 60)
                cs = np.concatenate([np.stack([np.cos(th) * s[0], np.sin(th) * s[1], 0 * th], 1),
                                     np.stack([np.cos(th) * s[0], 0 * th, np.sin(th) * s[2]], 1),
                                     np.stack([0 * th, np.cos(th) * s[1], np.sin(th) * s[2]], 1)])
            elif ty == 7:                              # mesh: its vertices' outline
                mid = int(m.geom_dataid[g])
                a0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                cs = np.array(m.mesh_vert[a0:a0 + nv])
            elif ty in (3, 5):                         # capsule / cylinder along its z
                cs = np.array([[0, 0, -s[1]], [0, 0, s[1]]])
            else:
                cs = np.zeros((1, 3))
            P = np.array([H(c + Rg @ q) for q in cs])
            col = COL.get(bn, "k")
            if ty in (6, 7):
                from scipy.spatial import ConvexHull
                try:
                    hv = ConvexHull(P[:, [i, j]]).vertices
                    ax.fill(P[hv, i], P[hv, j], color=col, alpha=0.25, ec=col, lw=1.2)
                except Exception:
                    ax.plot(P[:, i], P[:, j], color=col, lw=2)
            elif ty == 4:
                ax.plot(P[:, i], P[:, j], color=col, lw=0.8)
            else:
                ax.plot(P[:, i], P[:, j], color=col, lw=2.5)
        for n, jn in enumerate(mc.jid):
            a = H(d.xanchor[jn])
            u = d.xaxis[jn] @ R
            L = 0.07
            ax.plot([a[i] - L * u[i], a[i] + L * u[i]], [a[j] - L * u[j], a[j] + L * u[j]], color=JCOL[mc.role[n]],
                    lw=2.2, zorder=5)
            ax.plot(a[i], a[j], ".", color=JCOL[mc.role[n]], ms=7, zorder=5)
        c = H(mc.com())
        ax.plot(c[i], c[j], "k+", ms=16, mew=2.5, zorder=7)
        cb = H(np.array([rb[0], rb[1], o[2]]))
        if j != 2:
            ax.plot(cb[i], cb[j], "o", mfc="none", mec="k", ms=12, zorder=7)
        else:
            ax.axvline(cb[0], color="k", ls=":", lw=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3)
    # check lines: numbers of the model printed on the figure
    wing_x = {}
    for g in range(m.ngeom):
        bn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        if bn in T.WINGS:
            c, Rg, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
            cs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * s
            P = np.array([H(c + Rg @ q) for q in cs])
            wing_x[T.TAG[bn]] = dict(x=[float(P[:, 0].min()), float(P[:, 0].max())],
                                     y=[float(P[:, 1].min()), float(P[:, 1].max())])
    anchors = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, int(jn)): [round(float(v), 4) for v in H(d.xanchor[jn])]
               + [round(float(v), 3) for v in d.xaxis[jn] @ R] for jn in mc.jid}
    txt = (f"T4 as built: mass {rep['mass']:.4f} kg (ballast {mc.t4['keel_kg']:.4f} kg at x {mc.t4['keel_x']:.3f}, "
           f"z {mc.t4['keel_z']:.3f}), buoyancy {B:.3f} N, b = {B/(rep['mass']*T.G):.4f}\n"
           f"CoM (+) hull frame {np.round(rep['com_hull'],4).tolist()}, centre of buoyancy x {float(cb[0]):.4f} (o / dotted)\n"
           f"inertia about CoM, hull axes (kg m2): {np.round(rep['inertia_hull_diag'],4).tolist()}; "
           f"{rep['n_hinges']} hinges, {rep['n_shoulder_hinges']} at the shoulders\n"
           f"hinge axes: stroke red, feather green, sweep blue, tail grey; wings FL blue, FR orange, HL purple, HR pink")
    fig.suptitle(txt, fontsize=10, x=0.02, ha="left")
    out = HERE / "figures" / "S0b_T4_built.png"
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=110)
    side = dict(report=rep, B=B, wing_extent_hull=wing_x, hinge_anchor_axis_hull=anchors,
                cob_x_hull=float(cb[0]), bad_qacc=int(mc.bad_qacc))
    (HERE / "figures" / "S0b_T4_built.json").write_text(json.dumps(side, indent=1))
    print(json.dumps(dict(wing_extent_hull=wing_x, anchors=anchors)))


if __name__ == "__main__":
    main()
