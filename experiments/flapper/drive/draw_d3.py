"""Pass D3's figures (H10): each twist candidate from the BUILT model, as the
solver closes it, at four crank angles of its second beat.  The model is
stepped in a vacuum with the hull held and no gravity.  Outside freerig, so
the hinges are held here by the model's own joint springs: the root feather
by a stiff spring at -2.94 deg, the chain hinges by springs of the stated k
(T0: stiff springs whose rest follows the servo's reference).

Three views of the right wing: from ahead (span, up), the root's mechanism
from ahead (zoomed), and from the side (chord lines of the root and tip
panels).  Dimensions are measured on the model and printed.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/drive/draw_d3.py
"""
import json
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
from twist import TwistDrive  # noqa: E402

COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:red")
PSI0, A_TIP = math.radians(-2.94), math.radians(15.0)


def draw(name, tw, k, title):
    import dytiscidae.core.phenotype as PH
    PH.AVIONICS_MASS = 0.005
    import l1a
    from body import built
    from dytiscidae.core.phenotype import build_panels  # noqa: F401
    D = TwistDrive(40.0, 2.5, 48, a=0.03, dt=1e-4, feather_axis=0.1, twist=tw)
    m, d, _, pan = D.compile(built(dict(l1a.BODY_S, wing_shape=dict(camber=0.04), wing_panels=6)))
    mode = tw["mode"]
    for side in (1, -1):                       # freerig's hinge shift
        ch = D.chain[side]
        m.jnt_pos[m.joint(ch[0] + "_jf").id] += D.shift_of[ch[0]][0]
        for nm in ch[1:]:
            m.jnt_pos[m.joint(nm + "_j").id] += D.shift_of[nm][0]
    m.actuator_gainprm[:-1, :] = 0.0           # every actuator but the drive motor (the last)
    m.actuator_biasprm[:-1, :] = 0.0
    m.actuator_forcelimited[:-1] = 0
    m.actuator_ctrllimited[:-1] = 0
    rest = math.radians(tw.get("rest_deg", 0.0))
    tw_j = {s: [m.joint(nm + "_j").id for nm in D.chain[s][1:]] for s in (1, -1)}
    for side in (1, -1):
        jf = m.joint(D.chain[side][0] + "_jf")
        m.jnt_stiffness[jf.id], m.qpos_spring[jf.qposadr[0]], m.dof_damping[jf.dofadr[0]] = 200.0, PSI0, 0.5
        for j in tw_j[side]:
            m.jnt_stiffness[j] = 200.0 if mode == "servo" else k
            m.qpos_spring[m.jnt_qposadr[j]] = rest / D.n_h
            m.dof_damping[m.jnt_dofadr[j]] = 0.05 if mode == "servo" else 0.002
    m.opt.timestep, m.opt.gravity[:] = 1e-4, 0.0
    d.qpos[:] = m.qpos0
    for side in (1, -1):
        d.qpos[m.joint(D.chain[side][0] + "_jf").qposadr[0]] = PSI0
    D.init_state(m, d)
    D.setup(m, d)
    hull = m.body("seg0_hull").id
    qc, qs = int(m.joint("crank+1_j").qposadr[0]), int(m.joint("seg1_wing_j").qposadr[0])
    right = set(D.chain[1]) | {"crank+1", "coupler+1", "twistrod+1"}
    mine = {m.body(n).id for n in right if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
    eqs = {m.eq(i).name: i for i in range(m.neq) if int(m.eq_type[i]) == mujoco.mjtEq.mjEQ_CONNECT}
    worst = {n: (0.0, 0.0) for n in eqs}
    targets = [2 * math.pi + i * 0.5 * math.pi for i in range(4)]
    snaps, n_step = [], 0

    def gap(i):
        b1, b2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
        p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[i, 0:3]
        p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ m.eq_data[i, 3:6]
        return float(np.linalg.norm(p1 - p2)), p1, p2

    while (targets or d.qpos[qc] < 4 * math.pi) and n_step < 60000:
        if mode == "servo":                    # the servo's reference, as freerig gives it
            ph = D.phase(m, d, 0.0)
            for side in (1, -1):
                for j in tw_j[side]:
                    m.qpos_spring[m.jnt_qposadr[j]] = -A_TIP * math.cos(ph) / D.n_h
        D.control(m, d, n_step * 1e-4)
        mujoco.mj_step(m, d)
        d.qpos[:7], d.qvel[:6] = m.qpos0[:7], 0.0
        n_step += 1
        if d.qpos[qc] >= 2 * math.pi:
            for n, i in eqs.items():
                g = gap(i)[0]
                if g > worst[n][0]:
                    worst[n] = (g, math.degrees(d.qpos[qc] - 2 * math.pi))
        if targets and d.qpos[qc] >= targets[0]:
            mujoco.mj_forward(m, d)
            o, R = d.xpos[hull], d.xmat[hull].reshape(3, 3)
            hf = lambda w: R.T @ (w - o)
            segs = []
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) in mine and int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    h = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g, 1]
                    segs.append((hf(d.geom_xpos[g] - h), hf(d.geom_xpos[g] + h),
                                 m.body(int(m.geom_bodyid[g])).name))
            panels = []
            for nm in D.chain[1]:
                b = m.body(nm).id
                Rb = d.xmat[b].reshape(3, 3)
                _, cdir, chord = D.shift_of[nm]
                seg = 0.11
                corners = [np.array([0.0, 0.0, 0.0]), np.array([seg, 0.0, 0.0]),
                           np.array([seg, 0.0, 0.0]) + chord * cdir, chord * cdir]
                panels.append([hf(d.xpos[b] + Rb @ c) for c in corners])
            hinges = [hf(d.xanchor[j]) for j in [m.joint("seg1_wing_jf").id] + tw_j[1]]
            s = dict(crank=math.degrees(targets[0] - 2 * math.pi), stroke=math.degrees(d.qpos[qs]),
                     tip=math.degrees(sum(d.qpos[m.jnt_qposadr[j]] for j in tw_j[1])),
                     feather=math.degrees(d.qpos[m.joint("seg1_wing_jf").qposadr[0]]),
                     segs=segs, panels=panels, hinges=hinges,
                     gaps={n: 1e3 * gap(i)[0] for n, i in eqs.items()},
                     shoulder=hf(d.xanchor[m.joint("seg1_wing_j").id]),
                     shaft=hf(d.xpos[m.body("crank+1").id]))
            # measured on the model
            i = eqs["coupler+1"]
            _, p1, p2 = gap(i)
            cb = m.body("coupler+1").id
            s["coupler_mm"] = 1e3 * float(np.linalg.norm(p1 - d.xpos[cb]))
            s["crank_mm"] = 1e3 * float(np.linalg.norm(d.xpos[cb] - d.xpos[m.body("crank+1").id]))
            s["lever_mm"] = 1e3 * float(np.linalg.norm(p2 - d.xanchor[m.joint("seg1_wing_jf").id]))
            tipb = m.body(D.chain[1][-1]).id
            Rt = d.xmat[tipb].reshape(3, 3)
            if mode == "crank":
                i = eqs["twistrod+1"]
                _, p1, p2 = gap(i)
                rb = m.body("twistrod+1").id
                t0, t1 = (d.xpos[tipb] + Rt @ v for v in D.tube_local[1])
                s.update(rod_mm=1e3 * float(np.linalg.norm(p1 - d.xpos[rb])), horn=hf(p2), pin=hf(d.xpos[rb]),
                         tube_mm=1e3 * float(np.linalg.norm(t1 - t0)), horn_mm=1e3 * float(np.linalg.norm(p2 - t1)),
                         arm_mm=1e3 * float(np.linalg.norm((hf(d.xpos[rb]) - s["shaft"])[1:])),
                         arm_aft_mm=1e3 * float((hf(d.xpos[rb]) - s["shaft"])[0]),
                         tube_off_line_mm=1e3 * float(np.linalg.norm(np.cross(t1 - t0, d.xanchor[tw_j[1][0]] - t0))
                                                      / np.linalg.norm(t1 - t0)),
                         design_tip=float(np.degrees(np.interp(math.radians(s["crank"]),
                                                               np.linspace(0, 2 * math.pi, 49),
                                                               np.append(D._design_twist, D._design_twist[0])))))
            if mode == "cable":
                a0 = d.site_xpos[m.site("tc_hull+1").id]
                a1 = d.site_xpos[m.site("tc_tip+1").id]
                te = d.xpos[tipb] + Rt @ D.te_local[1]
                tid = m.tendon("twistcable+1").id
                s.update(cable=(hf(a0), hf(a1)), cable_mm=1e3 * float(d.ten_length[tid]),
                         limit_mm=1e3 * float(m.tendon_range[tid, 1]),
                         post_mm=1e3 * float(np.linalg.norm(a1 - te)),
                         drop_mm=1e3 * float(s["shoulder"][2] - hf(a0)[2]),
                         te_from_shoulder_mm=1e3 * float(hf(te)[1] - s["shoulder"][1]))
            snaps.append(s)
            targets.pop(0)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.0))
    for s, col in zip(snaps, COLOURS):
        for ax, (i, j) in zip(axes, ((1, 2), (1, 2), (0, 2))):
            side_view = ax is axes[2]
            for n, pn in enumerate(s["panels"]):
                if side_view and n not in (0, len(s["panels"]) - 1):
                    continue
                xs = [1e3 * c[i] for c in pn] + [1e3 * pn[0][i]]
                ys = [1e3 * c[j] for c in pn] + [1e3 * pn[0][j]]
                ax.plot(xs, ys, color=col, lw=0.8, ls="-" if n == 0 or not side_view else "--", alpha=0.7)
            for p0, p1, bn in s["segs"]:
                ax.plot([1e3 * p0[i], 1e3 * p1[i]], [1e3 * p0[j], 1e3 * p1[j]], color=col,
                        lw=2.2 if "wing" not in bn else 1.6)
            for hgp in s["hinges"]:
                ax.plot(1e3 * hgp[i], 1e3 * hgp[j], "s", color=col, ms=3)
            ax.plot(1e3 * s["shaft"][i], 1e3 * s["shaft"][j], "+", color="k", ms=9)
            if "cable" in s:
                c0, c1 = s["cable"]
                ax.plot([1e3 * c0[i], 1e3 * c1[i]], [1e3 * c0[j], 1e3 * c1[j]], color=col, lw=1.0,
                        ls="-" if s["cable_mm"] >= s["limit_mm"] - 0.01 else ":")
                ax.plot(1e3 * c0[i], 1e3 * c0[j], "^", color="k", ms=5)
            if "horn" in s:
                ax.plot(1e3 * s["horn"][i], 1e3 * s["horn"][j], "o", color=col, ms=5)
        txt = f"crank {s['crank']:.0f}: stroke {s['stroke']:+.1f}, tip twist {s['tip']:+.1f} deg"
        if "design_tip" in s:
            txt += f" (design {s['design_tip']:+.1f}), rod gap {s['gaps']['twistrod+1']:.3f} mm"
        if "cable" in s:
            txt += f", cable {s['cable_mm']:.1f} of {s['limit_mm']:.1f} mm"
        axes[0].text(0.02, 0.98 - 0.05 * snaps.index(s), txt, transform=axes[0].transAxes, color=col, fontsize=8,
                     va="top")
    sh = snaps[0]["shoulder"]
    axes[1].set_xlim(1e3 * sh[1] - 25, 1e3 * sh[1] + 75)
    axes[1].set_ylim(1e3 * sh[2] - 85, 1e3 * sh[2] + 35)
    axes[0].set_title(title + "\nfrom ahead (y span, z up), mm; squares: hinges, +: crank shaft", fontsize=9)
    axes[1].set_title("the root from ahead, mm", fontsize=9)
    axes[2].set_title("from the side (x ahead, z up), mm: root panel (solid) and tip panel (dashed) outlines",
                      fontsize=9)
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = HERE.parent / "figures" / f"{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("==", name, title)
    le = snaps[0]["panels"][0][0]
    print("hinge line behind the leading edge %.1f mm; panel %.1f mm; chord %.1f mm" % (
        1e3 * float(np.linalg.norm(snaps[0]["hinges"][0] - le)),
        1e3 * float(np.linalg.norm(snaps[0]["panels"][0][1] - snaps[0]["panels"][0][0])),
        1e3 * float(np.linalg.norm(snaps[0]["panels"][0][3] - snaps[0]["panels"][0][0]))))
    keys = ("crank", "stroke", "feather", "tip", "design_tip", "crank_mm", "coupler_mm", "lever_mm", "tube_mm",
            "horn_mm", "arm_mm", "arm_aft_mm", "rod_mm", "tube_off_line_mm", "cable_mm", "limit_mm", "post_mm",
            "drop_mm", "te_from_shoulder_mm")
    for s in snaps:
        print("  " + ", ".join("%s %.2f" % (k_, s[k_]) for k_ in keys if k_ in s)
              + ", gaps mm " + json.dumps({n: round(v, 4) for n, v in s["gaps"].items()}))
    print("  largest gap over the second beat, mm (at crank deg):",
          {n: (round(1e3 * g, 4), round(a_)) for n, (g, a_) in worst.items()})
    print("  design", {k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in D.design.items()})
    print("  ", out)


def main():
    draw("D3_T0_TP", dict(mode="servo"), 0.0, "T0 (servo reference on stiff springs); TP is the same bodies")
    draw("D3_TX", dict(mode="crank", c_h=0.03), 10.0, "TX: tube, horn, rod from a second arm on the stroke crank; springs 10")
    draw("D3_TC", dict(mode="cable", post=0.08), 3.0, "TC: cable from the hull to a post on the tip's trailing edge; springs 3 "
         "(solid cable: taut)")


if __name__ == "__main__":
    main()
