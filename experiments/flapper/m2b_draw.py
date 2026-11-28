"""Pass M2b: figures (figures/M2b_*.png).

    m2b_draw.py pose ARM SEED HALF MODE T1,T2,... out.png
        poses the built model (build2, the scored spec) from a stored M2b row's
        10 Hz trace at the instants asked; drawing is m2_draw.figure.
    m2b_draw.py series ARM out.png
        CoM height and heading against time for every scored seed and half of an
        arm, under both entrainment modes, from the stored traces.

MODE is "wetting" or "default" (the row's mode field: True is the default).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROWS = HERE / "results" / "M2b_rows.jsonl"


def _mode_ok(r, mode):
    return (r.get("mode") == "wetting") if mode == "wetting" else (r.get("mode") in (True, "default", "True"))


def rows(arm):
    out = []
    for line in open(ROWS):
        r = json.loads(line)
        if r.get("kind") == "row" and not r.get("extra") and r["arm"] == arm:
            out.append(r)
    return out


def pose(arm, seed, half, mode, times):
    import m2b_run as R
    from m2_common import draw
    from m2_draw import snap
    from m2b_laws import build2
    row = [r for r in rows(arm) if r["seed"] == seed and r["half"] == half and _mode_ok(r, mode)][-1]
    law_spec, start, gains, vb, T = R.spec_for(arm)
    mc = build2(start, draw(seed), gains=gains, vb=vb, entrainment=("wetting" if mode == "wetting" else True))
    m, d, mj = mc.m, mc.d, mc.mj
    tt = np.array([x["t"] for x in row["trace"]])
    shots = []
    for t in times:
        x = row["trace"][int(np.argmin(np.abs(tt - t)))]
        nose, dors = np.array(x["nose"]), np.array(x["dors"])
        Rm = np.stack([nose, np.cross(dors, nose), dors], 1)
        q = np.zeros(4)
        mj.mju_mat2Quat(q, Rm.flatten())
        d.qpos[:3] = 0.0
        d.qpos[3:7] = q
        d.qpos[mc.qad] = x["q"]
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)
        d.qpos[:3] = np.array(x["com"]) - mc.com()
        mj.mj_forward(m, d)
        s = snap(mc)
        s["t"] = x["t"]
        s["row"] = x
        shots.append(s)
    return shots, row


def series(arm, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = rows(arm)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, mode in enumerate(("wetting", "default")):
        for r in rs:
            if not _mode_ok(r, mode):
                continue
            t = np.array([x["t"] for x in r["trace"]])
            com = np.array([x["com"] for x in r["trace"]])
            nose = np.array([x["nose"] for x in r["trace"]])
            hd = np.degrees(np.unwrap(np.arctan2(nose[:, 1], nose[:, 0])))
            lab = "s%d%s" % (r["seed"], r["half"])
            ls = "-" if r["half"] in ("A", "X") else "--"
            l, = ax[0][col].plot(t, com[:, 2], ls, lw=1, label=lab)
            ax[1][col].plot(t, hd - hd[0], ls, lw=1, color=l.get_color())
            if "h" in r["cmd"]:
                want = math.degrees(math.atan2(math.sin(r["cmd"]["h"] - math.atan2(nose[0, 1], nose[0, 0])),
                                               math.cos(r["cmd"]["h"] - math.atan2(nose[0, 1], nose[0, 0]))))
                ax[1][col].axhline(want, color=l.get_color(), lw=0.5, ls=":")
        ax[0][col].set_title("%s, entrainment %s" % (arm, mode))
        ax[0][col].axhline(0.0, color="tab:cyan", lw=0.8)
        ax[0][col].set_ylabel("CoM z (world), m")
        ax[1][col].set_ylabel("nose heading minus start, deg (dotted: command)")
        ax[1][col].set_xlabel("t, s")
        for a in ax[:, col]:
            a.grid(alpha=0.3)
        ax[0][col].legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print(out)


if __name__ == "__main__":
    if sys.argv[1] == "pose":
        arm, seed, half, mode = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
        times = [float(x) for x in sys.argv[6].split(",")]
        out = sys.argv[7]
        from m2_draw import figure
        shots, row = pose(arm, seed, half, mode, times)
        figure(shots, out, "M2b %s seed %d half %s, entrainment %s (posed from the stored 10 Hz trace), command %s" % (
            arm, seed, half, mode, {k: round(v, 2) for k, v in row["cmd"].items()}))
        for s in shots:
            print("t %.1f com %s pitch %.1f wet %d (trace wet %d)" % (s["t"], s["com"].round(3).tolist(), s["pitch"],
                  int((s["sub"] > 0).sum()), s["row"]["wet"]))
        print(out)
    elif sys.argv[1] == "series":
        series(sys.argv[2], sys.argv[3])
