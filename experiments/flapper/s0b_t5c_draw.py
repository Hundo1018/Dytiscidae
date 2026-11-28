"""Figures of B19-B23 (results/S0b_probes.md), drawn from rows of results/S0b_probes5.jsonl and the built T5b model.

    MUJOCO_GL=disable PYTHONPATH=. python experiments/flapper/s0b_t5c_draw.py surface
    ... s0b_t5c_draw.py water <fwd id default> <anti id default> <fwd id wetting> <anti id wetting>
    ... s0b_t5c_draw.py mission <row id> [<row id>]

surface: figures/S0b_T5c_surface.png: the reported surfacing rows (b19_final_*): CoM height, pitch, roll, fill, the sweep command,
         wet wing strips against time, and T5b drawn (side view, at the recorded height, roll and pitch) at six instants of one row.
water:   figures/S0b_T5c_water.png: the swim on the heading, the turn and the swim on the antipode, both modes.
mission: figures/S0b_T5c_mission.png: the whole chain.
No geometry changed (T5b is used as built), so there is no figures/S0b_T5c_built.png.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0b_laws as LW  # noqa: E402
import s0b_probe2 as P2  # noqa: E402
import s0b_t5_draw as D5  # noqa: E402
import s0b_t5b_probe as Q  # noqa: E402

D5.COL.update({"t5_pont_L": "tab:red", "t5_pont_R": "tab:brown"})
COLS = "t x y z vz roll pitch yaw fill hull_z pont_z dorsal_top_z w_pont w_dorsal w_hull w_wing w_tail n_wet_wing Fz Mx My cob_dx cob_dy sw_cmd phase".split()


def rows():
    return [json.loads(x) for x in open(HERE / "results" / "S0b_probes5.jsonl")]


def _bodies(ax, insts, span, q_pose, ymin=-1.0, ymax=1.0):
    """T5b side view at instants [(t, z, roll, pitch, sweep, label)]: wings level at ``sweep`` (or raised)."""
    mc = Q._make("water", 0.0, "x4", 1e-4, z=0.2, mod="s0b_t5b_machine")
    m, d, mj = mc.m, mc.d, mc.mj
    for n_, (z, roll, pitch, pose, label) in enumerate(insts):
        d.qpos[:3] = (0.0, 0.0, 0.0)
        d.qpos[3:7] = P2._quat(roll, pitch)
        d.qpos[mc.qad] = pose(mc)
        mj.mj_forward(m, d)
        d.qpos[:3] += np.array([0.0, 0.0, z]) - mc.com()
        mj.mj_forward(m, d)
        off = np.array([n_ * span + 0.35 - 0.235, 0.0, 0.0])
        D5.outline(mc, ax, 0, 2, off=off, alpha=0.35)
        ax.text(n_ * span + 0.35, ymin + 0.02, label, ha="center", fontsize=7)
    ax.axhspan(ymin, 0.0, color="tab:cyan", alpha=0.15, zorder=0)
    ax.axhline(0.0, color="tab:blue", lw=1.2)
    ax.set_xlim(-0.3, len(insts) * span)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])


def surface_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = {r["id"]: r for r in rows()}
    ids = ["b19_final_S4_default", "b19_final_S4_wetting", "b19_final_S5_default", "b19_final_S6_wetting"]
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(7, 4, height_ratios=[2.2, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0])
    # six instants of the first row: rise, exit start, wings out, afloat, settled, lift-off
    r0 = R[ids[0]]
    a0 = np.array([x[:24] for x in r0["trace"]], dtype=float)
    ph0 = [x[24] for x in r0["trace"]]
    ev = r0["events"]
    t_w = a0[np.argmax(a0[:, 15] < 0.9), 0]
    pk = a0[(a0[:, 0] >= t_w) & (a0[:, 0] <= ev["afloat_t"] + 2.0)]
    t_pk = pk[np.argmax(np.abs(pk[:, 6])), 0]
    t_pts = [(1.0, "rise, 1 s"), (t_w - 3.0, "3 s before the wings clear"), (t_w, "first wing out"), (t_pk, "peak pitch"), (ev["afloat_t"], "afloat"),
             (ev["afloat_t"] + 6.0, "settled, wings raised")]
    insts = []
    for tt, lab in t_pts:
        i = int(np.argmin(np.abs(a0[:, 0] - tt)))
        sw = a0[i, 23]
        raised = ph0[i] in ("settle", "fly")

        def pose(mc, sw=sw, raised=raised):
            q = np.zeros(len(mc.aid))
            if raised:
                Pl = dict(LW.LEVEL, **Q.law_cfg_b()["P"])
                return LW.targets_level(mc, 0.0, 1.0, Pl, osc=0.0)[0]
            for tag, L in (("FL", 1.0), ("FR", -1.0), ("HL", 1.0), ("HR", -1.0)):
                q[mc.ix[(tag, "sweep")]] = -L * sw
            return q
        insts.append((a0[i, 3], a0[i, 5], a0[i, 6], pose, f"{tt:.1f} s\n{lab}\nroll {a0[i, 5]:.0f} pitch {a0[i, 6]:.0f}"))
    axb = fig.add_subplot(gs[0, :])
    _bodies(axb, insts, 1.3, None, ymin=-0.9, ymax=0.95)
    axb.set_title(f"{ids[0]}: T5b at the recorded CoM height, roll and pitch (wings level at the commanded sweep until afloat, then raised); water above the blue line is air", fontsize=8, loc="left")
    names = ["CoM z, m", "pitch (nose up +), deg", "roll, deg", "tank fill", "sweep command, rad", "wet wing strips (of 32)"]
    for c, rid in enumerate(ids):
        r = R[rid]
        a = np.array([x[:24] for x in r["trace"]], dtype=float)
        e = r["events"]
        ta = e.get("afloat_t")
        sel = a[:, 0] <= (ta + 12.0 if ta else 1e9)
        for k, (col, nm) in enumerate(zip([3, 6, 5, 8, 23, 17], names)):
            ax = fig.add_subplot(gs[1 + k, c])
            ax.plot(a[sel, 0], a[sel, col], "k", lw=0.9)
            if ta:
                ax.axvline(ta, color="tab:green", lw=1, ls="--")
            if k == 0:
                ax.set_title(f"{rid}\n{('wetting' if r.get('entrainment')=='wetting' else 'default')}", fontsize=8)
                ax.axhline(0.0, color="tab:blue", lw=0.8)
            if c == 0:
                ax.set_ylabel(nm, fontsize=7)
            if col in (5, 6):
                ax.axhspan(-10, 10, color="tab:green", alpha=0.12)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=6)
            if k == 5:
                ax.set_xlabel("t, s (dashed: afloat)", fontsize=7)
    ok = {rid: (not R[rid]["abort"]) for rid in ids}
    fig.suptitle("S0b B19: T5b surfacing, recipe R (tanks emptied at 0.3/s, wings level, sweep pitch loop + one-sided dihedral roll loop), reported starts S4-S6; "
                 "green band = the 10 deg target; lift-off to 2 m held: " + ", ".join(f"{k[10:]} {'yes' if v else 'NO'}" for k, v in ok.items()), fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "figures" / "S0b_T5c_surface.png", dpi=85)
    print("drawn surface")


def water_fig(*rids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = {r["id"]: r for r in rows()}
    n = len(rids)
    fig, axs = plt.subplots(5, n, figsize=(5.0 * n, 14), squeeze=False)
    for c, rid in enumerate(rids):
        r = R[rid]
        tr = np.array([x[:3] + [x[3]] + x[4:8] + x[8:10] for x in [[v for v in row[:10]] for row in r["trace"]]], dtype=float)
        t = tr[:, 0]
        yaw0 = r.get("yaw0_deg", 0.0)
        anti = "anti" in rid
        h = np.radians(yaw0 + (180.0 if anti else 0.0))
        hc = np.array([math.cos(h), math.sin(h)])
        # the run starts at the end of the first ballast phase (the swim, flip or turn starts there)
        t0 = r["phases_out"][0]["t1"]
        sel = t >= t0
        p = tr[sel][:, 1:3] - tr[sel][0, 1:3]
        d_cmd = p @ hc
        axs[0][c].plot(t[sel] - t0, d_cmd, "k")
        axs[0][c].set_ylabel("displacement along the commanded\ndirection, m")
        tt = t[sel] - t0
        v = np.gradient(d_cmd, tt) if len(tt) > 2 else np.zeros_like(tt)
        k = max(int(10 / max(np.median(np.diff(tt)), 1e-3) / 1.0), 3)
        axs[1][c].plot(tt, np.convolve(v, np.ones(k) / k, mode="same"), "k", lw=0.8)
        axs[1][c].axhline(0.3, color="tab:green", ls="--", lw=1, label="0.3 m/s")
        axs[1][c].set_ylabel("speed along the command, 10 s mean")
        axs[2][c].plot(tt, tr[sel][:, 3], "tab:blue")
        axs[2][c].set_ylabel("CoM z, m")
        axs[3][c].plot(tt, (tr[sel][:, 7] - (yaw0 + (180.0 if anti else 0.0)) + 180) % 360 - 180, "tab:red")
        axs[3][c].set_ylabel("yaw error to the command, deg")
        axs[4][c].plot(tt, tr[sel][:, 5], label="roll")
        axs[4][c].plot(tt, tr[sel][:, 6], label="pitch")
        axs[4][c].legend(fontsize=7)
        axs[4][c].set_ylabel("attitude, deg")
        axs[4][c].set_xlabel("t from the end of the first ballast phase, s")
        T = tt[-1] if len(tt) else 1.0
        axs[0][c].set_title(f"{rid} ({('wetting' if r.get('entrainment')=='wetting' else 'default')})\nwhole-run speed {d_cmd[-1] / T:+.3f} m/s over {T:.0f} s; z range {tr[sel][:, 3].min():.2f}..{tr[sel][:, 3].max():.2f}; "
                            f"limiter {r['clamped_steps']}, QACC {r['bad_qacc']}", fontsize=7, loc="left")
        for ax in axs[:, c]:
            ax.grid(alpha=0.3)
    fig.suptitle("S0b B20-B21: T5b swim on a heading and on its antipode (flip-turn then swim), both modes", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "figures" / "S0b_T5c_water.png", dpi=85)
    print("drawn water", rids)


def mission_fig(rid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = {r["id"]: r for r in rows()}
    r = R[rid]
    a = np.array([x[:14] for x in r["trace"]], dtype=float)
    t = a[:, 0]
    ph = r["phases_out"]
    fig, axs = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
    axs[0].plot(t, a[:, 3], "k")
    axs[0].axhline(0.0, color="tab:blue", lw=1)
    axs[0].set_ylabel("CoM z, m")
    axs[1].plot(t, a[:, 5], label="roll")
    axs[1].plot(t, a[:, 6], label="pitch (nose up +)")
    axs[1].plot(t, a[:, 7], label="yaw", lw=0.8)
    axs[1].legend(fontsize=7, ncol=3)
    axs[1].set_ylabel("attitude, deg")
    axs[2].plot(t, a[:, 8], "tab:blue", label="fill")
    axs[2].set_ylabel("fill")
    axs[3].plot(t, a[:, 10], "tab:red", label="wet main-wing strips")
    axs[3].set_ylabel("wet wing strips")
    axs[4].plot(t, a[:, 12], "tab:purple")
    axs[4].set_ylabel("limiter steps, cumulative")
    axs[4].set_xlabel("t, s")
    for ax in axs:
        ax.grid(alpha=0.3)
        for p in ph:
            ax.axvline(p["t0"], color="0.5", lw=0.6, ls=":")
    for p in ph:
        axs[0].text(0.5 * (p["t0"] + p["t1"]), axs[0].get_ylim()[1], p["kind"], ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"S0b B23: {rid} ({('wetting' if r.get('entrainment')=='wetting' else 'default')}): {' > '.join(p['kind'] for p in ph)}; limiter steps {r['clamped_steps']}, QACC {r['bad_qacc']}, abort {r['abort']}, "
                 f"peak torque {({k: round(v) for k, v in r['tq_peak'].items()})}", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "figures" / "S0b_T5c_mission.png", dpi=85)
    print("drawn mission", rid)


if __name__ == "__main__":
    w = sys.argv[1]
    if w == "surface":
        surface_fig()
    elif w == "water":
        water_fig(*sys.argv[2:])
    else:
        mission_fig(sys.argv[2])
