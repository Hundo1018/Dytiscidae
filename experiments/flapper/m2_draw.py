"""Pass M2: figures from the built model (figures/M2_*.png).

    m2_draw.py ARM SEED HALF T1,T2,... [out.png] [rows file]

Re-runs the scored row (the same law, seed and start, deterministic) and draws
every geom of the compiled model at the instants asked: side view (x, z) and
top view (x, y), with the water surface, the strips' wet state in colour and,
in the title, the measured CoM height, pitch and wet strip count.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def geoms(mc):
    """Every machine geom as a polygon of world points (box corners, capsule ends)."""
    m, d = mc.m, mc.d
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b == 0:
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g]
        sz = m.geom_size[g]
        if m.geom_type[g] == 6:                      # box
            pts = [c + R @ (np.array([sx, sy, sz_]) * sz) for sx in (-1, 1) for sy in (-1, 1) for sz_ in (-1, 1)]
            out.append(("box", b, np.array(pts)))
        else:                                        # capsule / cylinder: the axis and its radius
            a = R[:, 2] * sz[1]
            out.append(("cap", b, np.array([c - a, c + a]), float(sz[0])))
    return out


def snap(mc):
    from m2_common import panel_sub
    sub, pos = panel_sub(mc)
    return dict(g=geoms(mc), com=mc.com().copy(), pos=pos, sub=sub,
                pitch=math.degrees(math.asin(max(-1, min(1, mc.d.xmat[mc.hull].reshape(3, 3)[2, 0])))),
                t=None, q=np.degrees(mc.d.qpos[mc.qad]).round(1).tolist())


def capture(arm, seed, half, times, extra=None):
    import m2_run as R
    from m2_common import build, draw, run_law, settle
    from m2_laws import make
    law_spec, start, gains, vb, T = R.spec_for(arm)
    dr = draw(seed)
    mc = build(start, dr, gains=gains, vb=vb, entrainment="wetting")
    if start == "float":
        settle(mc, 3.0)
    law = make(law_spec)
    if hasattr(law, "phi"):
        law.phi = dr["phase"]
    if hasattr(law, "law"):
        law.law.phi = dr["phase"]
    cmd = R.command_for(arm, half, dr, law_spec)
    shots = []
    want = sorted(times)

    def on(rec):
        t = rec.trace[-1]["t"]
        if want and t >= want[0] - 1e-6:
            s = snap(rec.mc)
            s["t"] = t
            s["row"] = rec.trace[-1]
            shots.append(s)
            want.pop(0)
        return not want
    run_law(mc, law, min(T, max(times) + 0.11), cmd, on_sample=on)
    return shots, mc, cmd


def figure(shots, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(shots)
    fig, ax = plt.subplots(2, n, figsize=(3.6 * n, 7.2), squeeze=False)
    for k, s in enumerate(shots):
        c = s["com"]
        for row, (i, j, lab) in enumerate(((0, 2, "z"), (0, 1, "y"))):
            a = ax[row][k]
            for item in s["g"]:
                col = {1: "0.2", 2: "tab:blue", 3: "tab:orange", 4: "tab:green", 5: "tab:green", 6: "tab:green"}.get(item[1], "k")
                if item[0] == "box":
                    P = item[2]
                    from scipy.spatial import ConvexHull
                    try:
                        h = ConvexHull(P[:, [i, j]])
                        V = P[h.vertices][:, [i, j]]
                        a.fill(V[:, 0] - c[i], V[:, 1] - (c[j] if row == 1 else 0.0), color=col, alpha=0.45, lw=0.8)
                    except Exception:
                        a.plot(P[:, i] - c[i], P[:, j] - (c[j] if row == 1 else 0.0), color=col, lw=0.8)
                else:
                    P = item[2]
                    a.plot(P[:, i] - c[i], P[:, j] - (c[j] if row == 1 else 0.0), color=col,
                           lw=max(1.0, item[3] * 140), solid_capstyle="round", alpha=0.8)
            wet = s["sub"] > 0
            a.scatter(s["pos"][wet, i] - c[i], s["pos"][wet, j] - (c[j] if row == 1 else 0.0), s=6, c="tab:cyan", zorder=5)
            a.scatter(s["pos"][~wet, i] - c[i], s["pos"][~wet, j] - (c[j] if row == 1 else 0.0), s=6, c="tab:red", zorder=5)
            if row == 0:
                a.axhline(0.0, color="tab:cyan", lw=1.0)
                a.set_ylim(c[2] - 1.0, c[2] + 1.0)
                a.set_title("t %.1f s  CoM z %+.2f m\npitch %.0f deg  wet %d/47" % (s["t"], c[2], s["pitch"], int(wet.sum())), fontsize=9)
            else:
                a.set_ylim(-1.0, 1.0)
            a.set_xlim(-1.0, 1.0)
            a.set_aspect("equal")
            a.set_xlabel("x - x_CoM, m")
            a.set_ylabel(("z (world), m" if row == 0 else "y - y_CoM, m"))
            a.grid(alpha=0.3)
    fig.suptitle(title + "\nblue/orange: wings (seg1 +y, seg2 -y); green: tail; dots: strip centres, cyan wet, red dry; line: the surface z = 0", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def from_trace(arm, seed, half, times, rows_path=None):
    """Pose the built model from a stored row's 10 Hz trace (hull attitude from its
    nose and dorsal axes, joint angles, CoM) without re-running it: one process,
    no stepping.  The wet state is re-read from the posed strips."""
    import m2_run as R
    from m2_common import build, draw
    rows_path = rows_path or (HERE / "results" / "M2_rows.jsonl")
    row = None
    for line in open(rows_path):
        r = json.loads(line)
        if r.get("kind") == "row" and not r.get("extra") and r["arm"] == arm and r["seed"] == seed and r["half"] == half:
            row = r
    law_spec, start, gains, vb, T = R.spec_for(arm)
    mc = build(start, draw(seed), gains=gains, vb=vb, entrainment="wetting")
    m, d, mj = mc.m, mc.d, mc.mj
    shots = []
    tt = np.array([x["t"] for x in row["trace"]])
    for t in times:
        x = row["trace"][int(np.argmin(np.abs(tt - t)))]
        nose, dors = np.array(x["nose"]), np.array(x["dors"])
        side = np.cross(dors, nose)
        Rm = np.stack([nose, side, dors], 1)
        q = np.zeros(4)
        mj.mju_mat2Quat(q, Rm.flatten())
        d.qpos[:3] = 0.0
        d.qpos[3:7] = q
        d.qpos[mc.qad] = x["q"]
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)
        d.qpos[:3] = np.array(x["com"]) - mc.com()
        mj.mj_forward(m, d)
        s_ = snap(mc)
        s_["t"] = x["t"]
        s_["row"] = x
        shots.append(s_)
    return shots, row


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "trace":
    arm, seed, half = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    times = [float(x) for x in sys.argv[5].split(",")]
    out = sys.argv[6]
    shots, row = from_trace(arm, seed, half, times)
    figure(shots, out, "M2 %s seed %d half %s (posed from the stored 10 Hz trace), command %s" % (
        arm, seed, half, {k: round(v, 2) for k, v in row["cmd"].items()}))
    for s in shots:
        print("t %.1f com %s pitch %.1f wet %d (trace wet %d) q %s" % (s["t"], s["com"].round(3).tolist(), s["pitch"],
              int((s["sub"] > 0).sum()), s["row"]["wet"], s["q"]))
    print(out)
    sys.exit(0)


if __name__ == "__main__":
    arm, seed, half = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    times = [float(x) for x in sys.argv[4].split(",")]
    out = sys.argv[5] if len(sys.argv) > 5 else str(HERE / "figures" / ("M2_%s_s%d%s.png" % (arm, seed, half)))
    shots, mc, cmd = capture(arm, seed, half, times)
    figure(shots, out, "M2 %s seed %d half %s, command %s" % (arm, seed, half, {k: round(v, 2) for k, v in cmd.items()}))
    for s in shots:
        print("t %.1f com %s pitch %.1f wet %d q %s" % (s["t"], s["com"].round(3).tolist(), s["pitch"], int((s["sub"] > 0).sum()), s["q"]))
    print(out)
