"""Pass D4's figures and its kinematic checks on the built models (V0, V9):
each candidate at both adjuster settings and four crank angles, read from
the compiled model, seen from ahead; the dimensions measured on the model;
the take-off model moved by its adjuster alone against the cruise model.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/drive/draw_d4.py [OUT.jsonl]
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
import d1_run  # noqa: E402
from va import VA  # noqa: E402

ANGLES = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:red")
DEP = {"VL": ("coupler{s}_j",), "VR": ("coupler{s}_j",), "VP": ("rodA{s}_j", "beam{s}_j", "rodB{s}_j")}


def build(cand, point, setting=None):
    import l1a
    from body import built
    V, th, f, Phi, *_ = d1_run.POINTS[point]
    D = VA(cand, point, f, 24, feather_axis=0.1, setting=setting)
    p = built(dict(l1a.BODY_S, wing_shape=dict(camber=0.04)))
    m, d, _, _ = D.compile(p)
    # the feather hinge where freerig puts it (the lever is on that axis)
    for side, nm in ((1, "seg1_wing"), (-1, "seg2_wing")):
        j = m.joint(nm + "_jf").id
        m.jnt_pos[j] = m.jnt_pos[j] + (D.lever_local[side] - np.array([D.a, 0.0, 0.0]))
    return D, m, d


def gaps(m, d):
    out = []
    for i in range(m.neq):
        if int(m.eq_type[i]) != mujoco.mjtEq.mjEQ_CONNECT:
            continue
        b1, b2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
        p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[i, 0:3]
        p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ m.eq_data[i, 3:6]
        out.append(p1 - p2)
    return np.concatenate(out)


class Poser:
    """Sets the cranks (and the adjuster) and solves the other joints so the
    model's own connects close: `mj_kinematics`, no dynamics."""

    def __init__(self, D, m, d):
        from scipy.optimize import least_squares
        self.ls = least_squares
        self.D, self.m, self.d = D, m, d
        q = lambda n: int(m.joint(n).qposadr[0])
        self.q = q
        self.dep = [q(n.format(s=s)) for s in ("+1", "-1") for n in DEP[D.cand]] + [q("seg1_wing_j"), q("seg2_wing_j")]
        self.x = np.array([m.qpos0[i] for i in self.dep])

    def set(self, th_rel, adjust=None):
        """Walks the crank there from where it was in steps of 5 deg at most, so
        the solve stays on the built branch."""
        last = getattr(self, "th_last", 0.0)
        n = max(int(math.ceil(abs(th_rel - last) / math.radians(5.0))), 1)
        for k in range(1, n + 1):
            out = self._set(last + (th_rel - last) * k / n, adjust)
        self.th_last = th_rel
        return out

    def _set(self, th_rel, adjust=None):
        m, d, D, q = self.m, self.d, self.D, self.q
        d.qpos[:] = m.qpos0
        for s in ("+1", "-1"):
            d.qpos[q(f"crank{s}_j")] = th_rel
            if D.cand == "VP":
                d.qpos[q(f"crankB{s}_j")] = th_rel
            if adjust is not None:
                d.qpos[q(f"adjust{s}_j")] = adjust

        def F(x):
            d.qpos[self.dep] = x
            mujoco.mj_kinematics(m, d)
            return gaps(m, d)
        sol = self.ls(F, self.x, xtol=1e-15, ftol=1e-15, gtol=1e-15)
        self.x = sol.x
        d.qpos[self.dep] = sol.x
        mujoco.mj_forward(m, d)
        return float(np.abs(gaps(m, d)).max()), float(np.degrees(d.qpos[q("seg1_wing_j")]))


def hull_frame(m, d, x):
    hb = m.body("seg0_hull").id
    S = d.xanchor[m.joint("seg1_wing_j").id]
    return d.xmat[hb].reshape(3, 3).T @ (np.asarray(x) - S)


def mech_points(m, d):
    """End points of every mechanism capsule's centre line and every block's
    centre, from the right shoulder, hull axes; right side only."""
    skip = (0, m.body("seg0_hull").id, m.body("seg1_wing").id, m.body("seg2_wing").id)
    segs = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        name = m.body(b).name
        if b in skip or not name.endswith("+1"):
            continue
        c = d.geom_xpos[g]
        if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_CAPSULE:
            h = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g, 1]
            segs.append((name, hull_frame(m, d, c - h), hull_frame(m, d, c + h)))
        else:
            segs.append((name, hull_frame(m, d, c), hull_frame(m, d, c)))
    return segs


def measure(D, m, d):
    """The built dimensions, read from the model at its built pose (mm, deg)."""
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    A = lambda n: hull_frame(m, d, d.xanchor[m.joint(n).id])[1:]
    out = {}
    eqp = {}
    for i in range(m.neq):
        if int(m.eq_type[i]) == mujoco.mjtEq.mjEQ_CONNECT:
            b1 = int(m.eq_obj1id[i])
            eqp[m.eq(i).name] = hull_frame(m, d, d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[i, 0:3])[1:]
    if D.cand in ("VL", "VR"):
        O, Q, P = A("crank+1_j"), A("coupler+1_j"), eqp["loop+1"]
        out.update(crank_centre=O, crank_radius=np.linalg.norm(Q - O), coupler=np.linalg.norm(P - Q),
                   lever=np.linalg.norm(P))
    else:
        OA, OB, QA, QB, EA = A("crank+1_j"), A("crankB+1_j"), A("rodA+1_j"), A("rodB+1_j"), A("beam+1_j")
        P, EB = eqp["mid+1"], eqp["endB+1"]
        out.update(centre_A=OA, centre_B=OB, radius_A=np.linalg.norm(QA - OA), radius_B=np.linalg.norm(QB - OB),
                   rod_A=np.linalg.norm(EA - QA), rod_B=np.linalg.norm(EB - QB), beam_half_A=np.linalg.norm(P - EA),
                   beam_half_B=np.linalg.norm(EB - P), lever=np.linalg.norm(P),
                   phase_deg=math.degrees(math.atan2(*(QB - OB)[::-1]) - math.atan2(*(QA - OA)[::-1])) % 360.0)
    return {k: (np.round(1e3 * v, 3).tolist() if k != "phase_deg" else round(float(v), 3)) for k, v in out.items()}


def main():
    rows = []
    for cand in ("VL", "VR", "VP"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
        models = {}
        for ax, point in zip(axes, ("takeoff", "cruise")):
            D, m, d = build(cand, point)
            models[point] = (D, m, d)
            dims = measure(D, m, d)
            po = Poser(D, m, d)
            worst, low, outb, strokes = 0.0, 0.0, 0.0, []
            lean = {}                 # least angle between each link to the wing and the lever (added after data)
            sweep = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
            for ang in sweep:                                 # the envelope and the stroke over a revolution
                g, b = po.set(ang)
                worst = max(worst, g)
                strokes.append(b)
                lev = d.xmat[m.body("seg1_wing").id].reshape(3, 3)[:, 0]
                lev = (d.xmat[m.body("seg0_hull").id].reshape(3, 3).T @ lev)
                for name, p0, p1 in mech_points(m, d):
                    low, outb = min(low, p0[2], p1[2]), max(outb, p0[1], p1[1])
                    if name.startswith(("coupler", "rod")):
                        e = (p1 - p0) / np.linalg.norm(p1 - p0)
                        ang_ = math.degrees(math.acos(min(abs(float(e @ lev)), 1.0)))
                        if ang_ < lean.get(name, (91.0, 0.0))[0]:
                            lean[name] = (ang_, math.degrees(ang))
            kin = float(np.abs(np.radians(strokes) - D.beta(D.th0 + sweep)).max())
            for ang, col in zip(ANGLES, COLOURS):
                g, b = po.set(ang)
                for name, p0, p1 in mech_points(m, d):
                    if np.allclose(p0, p1):
                        ax.plot(p0[1], p0[2], "s", color=col, ms=6)
                    else:
                        ax.plot([p0[1], p1[1]], [p0[2], p1[2]], color=col, lw=2.2 if "crank" in name else 1.4)
                sp = hull_frame(m, d, d.xanchor[m.joint("seg1_wing_j").id]
                                + 0.16 * d.xmat[m.body("seg1_wing").id].reshape(3, 3)[:, 0])
                ax.plot([0, sp[1]], [0, sp[2]], color=col, lw=1, ls="--")
                ax.text(0.02, 0.98 - 0.045 * ANGLES.index(ang),
                        f"crank {math.degrees(ang):.0f}: stroke {b:+.1f} deg, loops {g * 1e3:.4f} mm",
                        transform=ax.transAxes, color=col, fontsize=8, va="top")
            ax.plot(0, 0, "k+", ms=10)
            ax.add_patch(plt.Circle((-0.04, 0.0), 0.04, fill=False, ls=":", color="gray"))
            ax.set_aspect("equal")
            ax.set_xlim(-0.09, 0.21)
            ax.set_ylim(-0.23, 0.10)
            unit = "deg" if cand == "VP" else "mm"
            sv = math.degrees(D.setting) if cand == "VP" else 1e3 * D.setting
            ax.set_title(f"{cand}, {point}: adjuster at {sv:.1f} {unit}; stroke {D.amp_deg:.2f} deg, mid {D.mean_deg:+.2f}\n"
                         + "; ".join(f"{k} {v}" for k, v in dims.items()).replace("; rod_A", ";\nrod_A"), fontsize=6.5)
            ax.set_xlabel("across from the shoulder, m (hull frame; dotted: a 40 mm circle inboard, for scale)")
            rows.append(dict(cand=cand, point=point, setting=D.setting, dims=dims, loop_worst_mm=1e3 * worst,
                             least_angle_to_lever_deg={k: [round(v[0], 1), round(v[1], 0)] for k, v in lean.items()},
                             lowest_mm=-1e3 * low, outermost_mm=1e3 * outb, kin_vs_closed_form_deg=math.degrees(kin),
                             design=dict(amp=D.amp_deg, mean=D.mean_deg, down=D.down_share, h2=D.h2, h3=D.h3,
                                         th0=D.th0, phi=D.phi)))
        axes[0].set_ylabel("up, m")
        fig.suptitle(f"Pass D4, {cand} as built (right wing, seen from ahead; span line dashed; squares: slider, pin block or carrier)")
        fig.tight_layout()
        path = HERE.parent / "figures" / f"D4_{cand}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(path)
        # V0: the take-off model, only its adjuster moved to the cruise setting
        DT, mT, dT = models["takeoff"]
        DC, mC, dC = models["cruise"]
        travel = DC.setting - DT.setting
        poT, poC = Poser(DT, mT, dT), Poser(DC, mC, dC)
        steps = 20
        for k in range(1, steps + 1):                       # walk the adjuster there with the crank still
            poT.set(0.0, adjust=mT.qpos0[poT.q("adjust+1_j")] + travel * k / steps)
        adj = mT.qpos0[poT.q("adjust+1_j")] + travel
        worst, diff = 0.0, 0.0
        for k in range(1, steps + 1):                       # and the cruise model's crank to the same angle
            poC.set((DT.th0 - DC.th0) * k / steps)
        for ang in np.linspace(0.0, 2.0 * math.pi, 36, endpoint=False):
            # the same absolute angle of the first crank in both models
            gT, bT = poT.set(ang, adjust=adj)
            gC, bC = poC.set(ang + DT.th0 - DC.th0)
            worst, diff = max(worst, gT, gC), max(diff, abs(bT - bC))
        rows.append(dict(cand=cand, check="V0", travel=travel, loop_worst_mm=1e3 * worst, stroke_diff_deg=diff))
    for r in rows:
        print(json.dumps(r))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
