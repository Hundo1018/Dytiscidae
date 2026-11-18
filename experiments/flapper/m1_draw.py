"""M1 figures: the built model at several instants of a scored take-off, re-run
through `m1_takeoff.run` (deterministic) with the machine's geoms captured.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m1_draw.py ARM X YAW [J_r gear]

Writes figures/M1_<ARM>_<x>_<yaw>.png: side and front views of every geom of
the machine (wing boxes, hull mesh, boom, tail) and the beach line, at rest,
at the strike, on leaving the ground, at 0.5 s, at 2 s and at 8 s after the
command, with the run's traces (centre-of-mass rise, nose elevation, contact
load, electrical power) and the gates' reads.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m1_takeoff as T  # noqa: E402

SNAP = {"at rest (t_c)": 0.0, "strike (+0.06 s)": 0.06, "+0.15 s": 0.15, "+0.5 s": 0.5, "+2 s": 2.0, "+8 s": 8.0}


def geoms(mc):
    m, d = mc.m, mc.d
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        p = d.geom_xpos[g]
        t = int(m.geom_type[g])
        if t == 6:                                       # box
            s = m.geom_size[g]
            c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * s
            out.append(("box", p + c @ R.T, int(m.geom_bodyid[g])))
        elif t == 7:                                     # mesh
            mid = int(m.geom_dataid[g])
            a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            v = m.mesh_vert[a:a + n]
            out.append(("mesh", p + v @ R.T, int(m.geom_bodyid[g])))
        elif t == 3:                                     # capsule along its z
            h = m.geom_size[g][1]
            out.append(("seg", np.array([p - h * R[:, 2], p + h * R[:, 2]]), int(m.geom_bodyid[g])))
    return out, mc.com().copy()


def hull2d(P):
    from scipy.spatial import ConvexHull
    try:
        h = ConvexHull(P)
        return P[h.vertices]
    except Exception:
        return P


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arm, x, yaw = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    J_r = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-5
    gear = float(sys.argv[5]) if len(sys.argv) > 5 else 24.0
    snaps = {}
    make0 = T.make

    def make(entry, J, g, contact=None):
        mc = make0(entry, J, g, contact)
        step0 = mc.step
        want = {k: int(round((T.T_C + v) / mc.dt)) for k, v in SNAP.items()}
        state = dict(n=0)

        def step(q, qd=None):
            for k, n in want.items():
                if state["n"] == n:
                    snaps[k] = geoms(mc) + (mc.contact_load(),)
            state["n"] += 1
            step0(q, qd)
        mc.step = step
        mc._m1_ground = lambda xs: mc.env.ground_heights(np.asarray(xs))
        snaps["_mc"] = mc
        return mc
    T.make = make
    r = T.run(arm, (x, yaw), J_r=J_r, gear=gear, trace=True)
    mc = snaps.pop("_mc")
    wings = {int(mc.m.jnt_bodyid[j]) for j, ro in zip(mc.jid, mc.role) if ro in ("stroke",)}
    fig = plt.figure(figsize=(18, 11))
    keys = list(SNAP)
    for i, k in enumerate(keys):
        if k not in snaps:
            continue
        gs, c, load = snaps[k]
        for row, (a, b, lab) in enumerate(((0, 2, "side: world x / z"), (1, 2, "front: world y / z"))):
            ax = fig.add_subplot(3, 6, 1 + i + 6 * row)
            for kind, P, body in gs:
                col = "tab:blue" if body in wings else ("tab:green" if body > 3 else "0.3")
                if kind == "seg":
                    ax.plot(P[:, a], P[:, b], color=col, lw=2)
                else:
                    Q = hull2d(P[:, [a, b]])
                    ax.fill(Q[:, 0], Q[:, 1], color=col, alpha=0.45, lw=0.5)
            ax.plot([c[a]], [c[b]], "k+", ms=8)
            if a == 0:
                xs = np.linspace(c[0] - 1.2, c[0] + 1.2, 50)
                ax.plot(xs, mc.env.ground_heights(xs), color="tan", lw=2)
            else:
                ax.axhline(float(mc.env.ground_heights(np.array([c[0]]))[0]), color="tan", lw=2)
            ax.set_aspect("equal")
            ax.set_xlim(c[a] - 1.0, c[a] + 1.0)
            ax.set_ylim(c[2] - 1.0, c[2] + 1.0)
            ax.set_title("%s\n%s, load %.2f W" % (k, lab, load) if row == 0 else lab, fontsize=8)
            ax.tick_params(labelsize=6)
    tr = r["trace"]
    t = np.arange(len(tr["H"])) * 0.05 - T.T_C
    for j, (key, lab) in enumerate((("H", "centre-of-mass rise, m"), ("nose", "nose elevation, deg"),
                                    ("load", "contact load, weights"), ("P", "electrical power, W"))):
        ax = fig.add_subplot(3, 4, 9 + j)
        ax.plot(t, tr[key], lw=0.8)
        if key == "H":
            ax.axhline(2.0, color="r", ls=":", lw=0.8)
        if key == "P":
            ax.axhline(T.THREE_CELLS_W, color="r", ls=":", lw=0.8)
        ax.set_xlabel("s after the command", fontsize=7)
        ax.set_title(lab, fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle("M1 %s at x %.0f, heading %.0f, J_r %.0e gear %d: pass %s (G1-4 %d%d%d%d), t_off %.2f s, H end %.2f m, "
                 "peak wing contact %.1f W, peak power %.0f W (sampled trace, 0.05 s)" % (
                     arm, x, yaw, J_r, gear, T.passed(r), r["G1"], r["G2"], r["G3"], r["G4"], r["t_off"], r["H_end"],
                     r["peak_wing_contact_W"], r["P_peak_W"]), fontsize=10)
    out = HERE / "figures" / ("M1_%s_%d_%d.png" % (arm.replace("+", "plus"), x, yaw))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=90)
    print(out, json.dumps({k: r[k] for k in ("G1", "G2", "G3", "G4", "t_off", "H_end", "peak_wing_contact_W")}))


if __name__ == "__main__":
    main()
