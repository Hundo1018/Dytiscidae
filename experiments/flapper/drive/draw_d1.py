"""Pass D1's figures: each stroke mechanism as built, at four crank angles,
read from the compiled model (geom places after `mj_forward`), seen from
ahead, with the loop's closure measured on the model.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/drive/draw_d1.py
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import d1_run  # noqa: E402
from drive import Drive  # noqa: E402

ANGLES = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:red")


def pose(D, m, d, th_rel):
    """Set every joint of the mechanism for crank angle th0 + th_rel from the
    closed form, the wing to its stroke, and return the stroke (deg)."""
    q = lambda name: int(m.joint(name).qposadr[0])
    if D.kind == "DD":
        b = D.bm * math.sin(th_rel)
        d.qpos[q("seg1_wing_j")] = d.qpos[q("seg2_wing_j")] = b
        return math.degrees(b)
    th = D.th0 + th_rel
    b = float(D.beta(th)[0])
    d.qpos[q("seg1_wing_j")] = d.qpos[q("seg2_wing_j")] = b
    P = D.a * np.array([math.cos(b), math.sin(b)])
    if D.kind in ("FB", "CT"):
        Q0 = D.O2 + D.r * np.array([math.cos(D.th0), math.sin(D.th0)])
        Q = D.O2 + D.r * np.array([math.cos(th), math.sin(th)])
        e0, e = np.array([D.a, 0.0]) - Q0, P - Q
        rel = math.atan2(e[1], e[0]) - math.atan2(e0[1], e0[0]) - th_rel
        for s in ("+1", "-1"):
            d.qpos[q(f"crank{s}_j")] = th_rel
            if D.kind == "FB":
                d.qpos[q(f"coupler{s}_j")] = rel
    else:
        d.qpos[q("crank_j")] = th_rel
        d.qpos[q("yoke_j")] = D.r * math.sin(th)
        d.qpos[q("follower_j")] = D.r * (math.cos(th) - 1.0)
        Y = np.array([D.a, -2.0 * D.a + D.r * math.sin(th)])
        e = P - Y
        rel = math.atan2(e[1], e[0]) - math.pi / 2.0
        for s in ("+1", "-1"):
            d.qpos[q(f"link{s}_j")] = rel
    return math.degrees(b)


def closure(m, d):
    """Largest gap, m, over the model's own connect constraints at this pose."""
    gap = 0.0
    for i in range(m.neq):
        if int(m.eq_type[i]) != mujoco.mjtEq.mjEQ_CONNECT:
            continue
        b1, b2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
        p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[i, 0:3]
        p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ m.eq_data[i, 3:6]
        gap = max(gap, float(np.linalg.norm(p1 - p2)))
    for i in range(m.ntendon):
        gap = max(gap, max(0.0, float(d.ten_length[i] - m.tendon_range[i, 1])))
    return gap


def main():
    import l1a
    from body import built
    from dytiscidae.core.phenotype import build  # noqa: F401
    out = []
    for point, (V, th, f, Phi, Psi, psi0, lev, kw) in d1_run.POINTS.items():
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
        for ax, kind in zip(axes, ("DD", "FB", "SY", "CT")):
            D = Drive(kind, Phi, f, 32, a=lev, k_spring=4.0, feather_axis=0.1)
            p = built(dict(l1a.BODY_S, wing_shape=dict(camber=0.04)))
            m, d, _, _ = D.compile(p)
            hull = m.body("seg0_hull").id
            worst = 0.0
            mech = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) not in
                    (hull, m.body("seg1_wing").id, m.body("seg2_wing").id) and m.geom_bodyid[g] != 0]
            for ang, col in zip(ANGLES, COLOURS):
                d.qpos[:] = m.qpos0
                b = pose(D, m, d, ang)
                mujoco.mj_forward(m, d)
                worst = max(worst, closure(m, d))
                o = d.xpos[hull]
                for g in mech:
                    c = d.geom_xpos[g] - o
                    if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_CAPSULE:
                        h = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g, 1]
                        ax.plot([c[1] - h[1], c[1] + h[1]], [c[2] - h[2], c[2] + h[2]], color=col, lw=2)
                    else:
                        ax.plot(c[1], c[2], "o", color=col, ms=4)
                for wb in ("seg1_wing", "seg2_wing"):
                    j = m.joint(wb + "_j").id
                    a0 = d.xanchor[j] - o
                    sp = d.xmat[m.body(wb).id].reshape(3, 3)[:, 0]
                    ax.plot([a0[1], a0[1] + 0.12 * sp[1]], [a0[2], a0[2] + 0.12 * sp[2]], color=col, lw=1, ls="--")
                for s in range(m.nsite):
                    c = d.site_xpos[s] - o
                    ax.plot(c[1], c[2], "x", color=col, ms=5)
                for t_ in range(m.ntendon):           # a cable: straight from site to site
                    ids = [int(m.wrap_objid[m.tendon_adr[t_] + w_]) for w_ in range(m.tendon_num[t_])]
                    c = np.array([d.site_xpos[i] - o for i in ids])
                    ax.plot(c[:, 1], c[:, 2], color=col, lw=1)
                ax.text(0.02, 0.97 - 0.06 * ANGLES.index(ang), f"crank {math.degrees(ang):.0f}: stroke {b:+.1f} deg",
                        transform=ax.transAxes, color=col, fontsize=8, va="top")
            ax.set_aspect("equal")
            ax.set_xlim(-0.16, 0.16)
            ax.set_ylim(-0.13, 0.11)
            ax.set_title(f"{kind}, {point}: lever {lev * 1e3:.0f} mm, crank r {D.r * 1e3:.1f} mm, "
                         f"link {D.L * 1e3:.0f} mm\nloop closure on the model: worst {worst * 1e3:.3f} mm", fontsize=8)
            ax.set_xlabel("across, m (hull frame)")
            out.append((point, kind, worst, D.r, D.L))
        axes[0].set_ylabel("up, m")
        fig.suptitle(f"Pass D1 stroke mechanisms as built, seen from ahead ({point}; wing span lines dashed, first 0.12 m)")
        fig.tight_layout()
        path = HERE.parent / "figures" / f"D1_{point}.png"
        fig.savefig(path, dpi=105)
        print(path)
    for row in out:
        print("%-8s %s closure %.4f mm r %.4f L %.4f" % (row[0], row[1], row[2] * 1e3, row[3], row[4]))


if __name__ == "__main__":
    main()
