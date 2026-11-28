"""Pass M4 figures from the built model: a recorded half is re-flown from its
spec (deterministic) to the instants asked, the model's geoms, its fluid strips
with their present twist, and the contacts with their forces are drawn in side
(x-z), front (y-z) and top (x-y) views, with the strips' pitch along the span.
The re-flown trace is checked against the recorded one (printed, and in the
figure's title).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m4_draw.py ROWS.jsonl ID OUT.png T1,T2,..
        T: seconds after t_c, or c+X (seconds after first contact)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull
    import m4_lib
    from m3_draw import surface
    rows = {}
    for x in open(sys.argv[1]):
        r = json.loads(x)
        rows[r["id"]] = r
    rid, out = sys.argv[2], sys.argv[3]
    times = [t if t.startswith("c+") else float(t) for t in sys.argv[4].split(",")]
    r = rows[rid]
    spec = dict(id=rid + "_draw", kind=r["kind"], arm=r["arm"], point=r["point"], start=tuple(r["start"]),
                cmd=r["cmd"], E=r.get("E"), ent=r.get("ent", "wetting"), dt=r.get("dt", 1e-4), snap=times, stop_at=True)
    o, T = m4_lib.run_half(spec)
    rec = np.load(HERE / "results" / "M4_traces" / f"{rid}.npz")
    n = min(len(rec["qpos"]), len(T["qpos"]))
    rep = float(np.abs(rec["qpos"][:n] - np.asarray(T["qpos"])[:n]).max())
    print("re-flown against recorded: %d samples, max |qpos diff| %.3g" % (n, rep))
    from mission import Machine
    mc = Machine("land", pose=dict(x=20, z=5))
    m, mj = mc.m, mc.mj
    names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    col = {"seg0_hull": "0.55", "seg1_wing": "tab:blue", "seg2_wing": "tab:orange", "seg3_strut": "0.3",
           "seg4_wing": "tab:green", "seg5_wing": "tab:green"}
    env = mc.env
    snaps = o.get("snaps", [])
    fig, axes = plt.subplots(len(snaps), 4, figsize=(18, 3.8 * len(snaps)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.2, 1, 1, 0.9]))
    for row, s in enumerate(snaps):
        gx, gm = np.array(s["geom_xpos"]), np.array(s["geom_xmat"])
        com = np.array(s["com"])
        sp, sc, sb = np.array(s["strip_pos"]), np.array(s["strip_chord"]), np.array(s["strip_body"])
        for ci, (a, b, lab) in enumerate(((0, 2, "side x-z"), (1, 2, "front y-z"), (0, 1, "top x-y"))):
            ax = axes[row, ci]
            for g in range(m.ngeom):
                bid = int(m.geom_bodyid[g])
                if bid == 0:
                    continue
                P = gx[g] + surface(m, g) @ gm[g].reshape(3, 3).T
                pts = P[:, [a, b]]
                try:
                    h = ConvexHull(pts)
                    ax.add_patch(Polygon(pts[h.vertices], closed=True, fc=col.get(names[bid], "k"), ec="k", lw=0.4,
                                         alpha=0.45))
                except Exception:
                    ax.plot(pts[:, 0], pts[:, 1], color=col.get(names[bid], "k"), lw=1)
            for i in range(len(sp)):
                c = col.get(names[sb[i]], "k")
                p0, p1 = sp[i] - 0.25 * sc[i], sp[i] + 0.75 * sc[i]
                ax.plot([p0[a], p1[a]], [p0[b], p1[b]], color=c, lw=0.9)
            if b == 2:
                if a == 0:
                    xs = np.linspace(com[0] - 1.2, com[0] + 1.2, 60)
                    gz = [max(env.ground_height(x), 0.0) for x in xs]
                    ax.plot(xs, gz, color="saddlebrown" if max(gz) > 0 else "tab:cyan", lw=1.5)
                else:
                    ys = np.linspace(com[1] - 1.2, com[1] + 1.2, 10)
                    ax.plot(ys, [max(env.ground_height(com[0]), 0.0)] * len(ys), color="saddlebrown", lw=1.5)
            for c in s["contacts"]:
                ax.plot(c["pos"][a], c["pos"][b], "kx", ms=6)
            ax.plot(com[a], com[b], "r+", ms=12, mew=2)
            ax.set_aspect("equal")
            ax.set_xlim(com[a] - 0.95, com[a] + 0.95)
            gnd = max(env.ground_height(com[0]), 0.0)
            if b == 2 and com[2] - gnd < 1.2:
                ax.set_ylim(min(com[2] - 0.6, gnd - 0.1), com[2] + 0.6)
            else:
                ax.set_ylim(com[b] - 0.95, com[b] + 0.95)
            fsum = sum(c["fn"] for c in s["contacts"]) / (0.6 * 9.80665)
            ax.set_title("%s t=%.2f s (%s) %s\nroll %.1f pitch %.1f yaw %.1f track %.1f; feather diff %.2f, tip amp diff %.2f deg;"
                         "\nwet %.2f, contacts %d carrying %.2f weights" % (
                             rid, s["t"], s["phase"], lab, s["roll"], s["pitch"], s["yaw"], s["track"],
                             s["lat"]["feather"], s["lat"]["twamp"], s["wet"], len(s["contacts"]), fsum), fontsize=7)
            ax.tick_params(labelsize=6)
        # strip pitch along the span, hull frame
        ax = axes[row, 3]
        q = np.array(s["qpos"])
        w, x, y, z = q[3:7]
        Rh = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        for wing, c in (("seg1_wing", "tab:blue"), ("seg2_wing", "tab:orange")):
            bid = names.index(wing)
            sel = sb == bid
            pl = (sp[sel] - q[:3]) @ Rh
            cl = sc[sel] @ Rh
            ang = np.degrees(np.arctan2(cl[:, 2], -cl[:, 0]))      # chord points aft: + is trailing edge up
            o_ = np.argsort(np.abs(pl[:, 1]))
            ax.plot(np.abs(pl[o_, 1]), ang[o_], "o-", color=c, ms=3, label=wing + (" (+y)" if pl[0, 1] > 0 else " (-y)"))
        ax.set_xlabel("|y| from hull, m", fontsize=7)
        ax.set_ylabel("strip pitch in hull frame, deg (+ trailing edge up = LE down)", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_title("strips' pitch along the span", fontsize=7)
    fig.suptitle("%s   re-flown vs recorded: max |qpos diff| %.2g over %d samples" % (rid, rep, n), fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    main()
