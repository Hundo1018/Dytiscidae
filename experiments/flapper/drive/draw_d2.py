"""Pass D2's figure (G5): FX2, the feather crank geared 1:1 to the stroke's
four-bar, as built and as the solver closes it.  The model is stepped in a
vacuum with the hull held (no fluid, no gravity) and snapped at four crank
angles in its second beat; geoms are read after `mj_forward`.  Two views of
the right wing's mechanism: from the side (chord plane) and from ahead.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/drive/draw_d2.py
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
from drive import Drive  # noqa: E402

# horn, shaft depth (m), arm scale, output name: D2's geometry unless given (D2b)
C_H, DROP, SCALE = (float(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (0.03, 0.08, 1.0)
NAME = sys.argv[4] if len(sys.argv) > 4 else "D2_fx2"
COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:red")


def main():
    import l1a
    from body import built
    D = Drive("FB", 85.0, 4.5, 24, a=0.06, feather_axis=0.1, feather=dict(mode="crank2", c_h=C_H, drop=DROP, arm_scale=SCALE))
    m, d, _, _ = D.compile(built(dict(l1a.BODY_S, wing_shape=dict(camber=0.04))))
    for nm, side in (("seg1_wing_jf", 1), ("seg2_wing_jf", -1)):     # freerig's feather_axis shift
        m.jnt_pos[m.joint(nm).id] += D.lever_local[side] - np.array([D.a, 0.0, 0.0])
    m.opt.timestep, m.opt.gravity[:] = 1e-4, 0.0
    d.qpos[:] = m.qpos0
    D.init_state(m, d)
    D.setup(m, d)
    hull, wing = m.body("seg0_hull").id, m.body("seg1_wing").id
    qc, qs, qf = (int(m.joint(n).qposadr[0]) for n in ("crank+1_j", "seg1_wing_j", "seg1_wing_jf"))
    i_eq = m.eq("featherrod+1").id
    mine = {m.body(n).id for n in ("fcrank+1", "featherrod+1")}
    targets = [2 * math.pi + k * 0.5 * math.pi for k in range(4)]
    snaps, k = [], 0
    worst = (0.0, 0.0)          # largest featherrod gap over the second beat, and its crank angle
    while (targets or d.qpos[qc] < 4 * math.pi) and k < 40000:
        D.control(m, d, k * 1e-4)
        mujoco.mj_step(m, d)
        d.qpos[:7], d.qvel[:6] = m.qpos0[:7], 0.0
        k += 1
        if d.qpos[qc] >= 2 * math.pi:
            b1_ = m.eq_obj1id[i_eq]
            g_ = float(np.linalg.norm(d.xpos[b1_] + d.xmat[b1_].reshape(3, 3) @ m.eq_data[i_eq, 0:3]
                                      - d.xpos[wing] - d.xmat[wing].reshape(3, 3) @ m.eq_data[i_eq, 3:6]))
            if g_ > worst[0]:
                worst = (g_, math.degrees(d.qpos[qc] - 2 * math.pi), math.degrees(d.qpos[qf]))
        if targets and d.qpos[qc] >= targets[0]:
            mujoco.mj_forward(m, d)
            o, R = d.xpos[hull], d.xmat[hull].reshape(3, 3)
            segs = []
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) in mine and int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    h = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g, 1]
                    segs.append((R.T @ (d.geom_xpos[g] - h - o), R.T @ (d.geom_xpos[g] + h - o)))
            Hw = d.xpos[wing] + d.xmat[wing].reshape(3, 3) @ m.eq_data[i_eq, 3:6]
            jf = m.joint("seg1_wing_jf").id
            A = d.xanchor[jf]
            b1 = m.eq_obj1id[i_eq]
            Pw = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[i_eq, 0:3]
            snaps.append(dict(crank=math.degrees(targets[0] - 2 * math.pi), stroke=math.degrees(d.qpos[qs]),
                              feather=math.degrees(d.qpos[qf]), segs=segs, horn=R.T @ (Hw - o),
                              hinge=R.T @ (A - o), gap_mm=1e3 * float(np.linalg.norm(Pw - Hw)),
                              shaft=R.T @ (d.xpos[m.body("fcrank+1").id] - o)))
            targets.pop(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for s, col in zip(snaps, COLOURS):
        for ax, (i, j) in zip(axes, ((0, 2), (1, 2))):
            for p0, p1 in s["segs"]:
                ax.plot([1e3 * p0[i], 1e3 * p1[i]], [1e3 * p0[j], 1e3 * p1[j]], color=col, lw=2)
            ax.plot(1e3 * s["horn"][i], 1e3 * s["horn"][j], "o", color=col, ms=6)
            ax.plot(1e3 * s["hinge"][i], 1e3 * s["hinge"][j], "s", color=col, ms=5)
            ax.plot([1e3 * s["hinge"][i], 1e3 * s["horn"][i]], [1e3 * s["hinge"][j], 1e3 * s["horn"][j]],
                    color=col, lw=1, ls="--")
            ax.plot(1e3 * s["shaft"][i], 1e3 * s["shaft"][j], "+", color="k", ms=10)
        axes[0].text(0.02, 0.97 - 0.06 * snaps.index(s),
                     f"crank {s['crank']:.0f}: stroke {s['stroke']:+.1f}, feather {s['feather']:+.1f} deg, "
                     f"gap {s['gap_mm']:.3f} mm", transform=axes[0].transAxes, color=col, fontsize=8, va="top")
    a = D.arm
    axes[0].set_title(f"FX2 from the side (x ahead, z up), mm: arm {1e3 * a['r2']:.1f}, rod {1e3 * a['ell']:.1f}, "
                      f"horn {1e3 * a['c_h']:.0f} aft of the hinge", fontsize=9)
    axes[1].set_title("from ahead (y span, z up), mm; square: feather hinge, circle: horn, +: feather shaft",
                      fontsize=9)
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = HERE.parent / "figures" / f"{NAME}.png"
    fig.savefig(out, dpi=130)
    for s in snaps:
        print("crank %5.0f stroke %+6.1f feather %+6.1f gap %.3f mm  horn-hinge %.1f mm  shaft-horn(rest) %s" % (
            s["crank"], s["stroke"], s["feather"], s["gap_mm"],
            1e3 * np.linalg.norm(s["horn"] - s["hinge"]), np.round(1e3 * (s["horn"] - s["shaft"]), 1)))
    print("largest gap over the beat %.3f mm at crank %.0f deg, feather %.1f deg" % (1e3 * worst[0], worst[1], worst[2]))
    print("arm", {k_: round(v, 4) if isinstance(v, float) else v for k_, v in a.items()})
    print(out)


if __name__ == "__main__":
    main()
