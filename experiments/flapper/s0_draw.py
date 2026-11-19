"""Pass S0 figures: the built model of a row's rung, posed at states of that row
re-run through `s0_run.run_row` (the poses are captured by `s0_machine.Inst`
at the asked simulated times).  Side (x-z), front (y-z) and top (x-y) views in
the world frame about the CoM; the sea line (z = 0) and the beach; the CoM (+);
MuJoCo's contacts (x); the parts a rung adds in their own colour (sweep motors
red, keel ballast blue, a hind wing pair purple).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/s0_draw.py SPECS.json ID OUT.png t1,t2,... "title"

The re-run's row is written beside the figure (OUT.json) so the figure can be
checked against the scored row (bit-exact re-run: same spec, same code).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m3b_draw import surface  # noqa: E402

COL = {"seg0_hull": "0.55", "seg1_wing": "tab:blue", "seg2_wing": "tab:orange", "seg3_strut": "0.3",
       "seg4_wing": "tab:green", "seg5_wing": "tab:green"}


def capture(spec, times):
    import s0_machine as S
    import s0_run
    S.CAPTURE[:] = sorted(times)
    S.SNAPS.clear()
    row = s0_run.run_row(spec)
    return row, list(S.SNAPS)


def figure(spec, snaps, out, title="", notes=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull
    import s0_machine as S
    S.configure(spec.get("cfg", {}))
    mc = S.make("land", pose=dict(x=20, z=5))
    m, d, mj = mc.m, mc.d, mc.mj
    names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    gnames = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    hull = mc.hull
    fig, axes = plt.subplots(len(snaps), 3, figsize=(15, 3.4 * len(snaps)), squeeze=False)
    for row, sn in enumerate(snaps):
        d.qpos[:] = sn["qpos"]
        d.qvel[:] = 0
        mj.mj_forward(m, d)
        com = np.array(sn["com"])
        R = d.xmat[hull].reshape(3, 3)
        psi = math.atan2(R[1, 0], R[0, 0])
        e1 = np.array([math.cos(psi), math.sin(psi), 0.0])
        e2 = np.array([-math.sin(psi), math.cos(psi), 0.0])
        for ci, (e, lab) in enumerate(((e1, "side (right = nose heading)"), (e2, "front (right = body left)"),
                                       (None, "top (x = heading, y = body left)"))):
            ax = axes[row, ci]
            top = e is None
            proj = (lambda P: np.stack([P @ e1, P @ e2], -1)) if top else (lambda P, e=e: np.stack([P @ e, P[..., 2]], -1))
            for g in range(m.ngeom):
                bid = int(m.geom_bodyid[g])
                if bid == 0:
                    continue
                nm = gnames[g]
                c = ("red" if "sweep_motor" in nm else "blue" if nm == "s0_keel" else
                     "tab:purple" if (names[bid] not in COL and "wing" in (names[bid] or "")) else COL.get(names[bid], "k"))
                P = d.geom_xpos[g] + surface(m, g) @ d.geom_xmat[g].reshape(3, 3).T
                pts = proj(P)
                a = 0.9 if c in ("red", "blue") else 0.5
                try:
                    hh = ConvexHull(pts)
                    ax.add_patch(Polygon(pts[hh.vertices], closed=True, fc=c, ec="k", lw=0.5, alpha=a, zorder=3 if a > 0.8 else 2))
                except Exception:
                    ax.plot(pts[:, 0], pts[:, 1], color=c, lw=1)
            c2 = proj(com)
            if not top:
                ss = np.linspace(-1.5, 1.5, 60)
                line = com[None, :] + ss[:, None] * e[None, :]
                gz = np.array([mc.env.ground_height(p[0]) for p in line])
                ax.plot(line @ e, gz, color="saddlebrown", lw=1.5)
                ax.axhline(0.0, color="tab:cyan", lw=1, ls="--")
            ax.plot(c2[0], c2[1], "r+", ms=12, mew=2, zorder=5)
            for pos, fn in sn["contacts"]:
                p2 = proj(np.array(pos))
                ax.plot(p2[0], p2[1], "kx", ms=6, zorder=6)
            ax.set_aspect("equal")
            ax.set_xlim(c2[0] - 0.9, c2[0] + 0.9)
            ax.set_ylim(c2[1] - 0.8, c2[1] + 0.8)
            nose = R[:, 0]
            ax.set_title("t=%.2f s  %s\nCoM z %.2f m, nose elev %.0f deg, contact %.1f W, max |tau| %.0f N m, m %.3f kg" % (
                sn["t"], lab, com[2], math.degrees(math.asin(max(-1, min(1, nose[2])))), sn["contact_W"],
                max(sn["tq"]), sn["mass"]), fontsize=7.5)
            ax.tick_params(labelsize=7)
    fig.suptitle((title or spec["id"]) + ("\n" + " | ".join(notes) if notes else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=105)
    print("wrote", out)


def main(specs, rid, out, times, title=""):
    sp = next(s for s in json.load(open(specs)) if s["id"] == rid)
    ts = [float(x) for x in times.split(",")]
    row, snaps = capture(sp, ts)
    Path(out).with_suffix(".json").write_text(json.dumps(dict(id=rid, row={k: v for k, v in row.items() if k != "result"},
                                                               snaps=[{k: v for k, v in s.items() if k != "contacts"}
                                                                      for s in snaps]), default=float))
    figure(sp, snaps, out, title)


if __name__ == "__main__":
    main(*sys.argv[1:])
