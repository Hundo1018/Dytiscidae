"""A drawing of the mechanism a pass tested, from the built model itself:
top, side and front views with the lifting strips, joints, centre of mass and
the main dimensions.  Every pass writes one beside its result.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/draw.py S1|L1m
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def draw(body: dict, out, title: str, notes: list, *, stroke_deg: float = 0.0,
         feather_deg: float = 0.0, mass: float = 0.60, spec: dict | None = None,
         tol: float = 0.02) -> list:
    """spec: the design's figures (m, m2, kg) for any of span, wing_area,
    chord, mass, tail_span, tail_area, tail_chord.  Each is compared with what
    the built model measures; the table is printed on the drawing and
    returned as (name, design, built, ok).  A drawing is also looked at: the
    table cannot see a part that is in the wrong place."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import dytiscidae.core.phenotype as PH
    PH.AVIONICS_MASS = 0.005
    from body import flapper
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    from dytiscidae.physics.fluid import WING

    env = TriphibianEnv(build(flapper(**body)), seed=0, timestep=1e-3)
    env.reset(Domain.AIR, randomise=False)
    d, m, mj = env.data, env.model, env._mj
    hull = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "seg0_hull")
    dry = env.solver._dry_mass.copy()
    dry[hull] += mass - dry.sum()                      # the rig's ballast, at the hull's centre
    p = env.solver.panels
    jn = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]

    def pose(stroke, feather):
        d.qpos[:] = m.qpos0
        d.qpos[:3] = 0.0
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for j, name in enumerate(jn):
            b = int(m.jnt_bodyid[j])
            if m.jnt_type[j] != 3 or int(m.body_parentid[b]) != hull:
                continue
            d.qpos[m.jnt_qposadr[j]] = math.radians(feather if name.endswith("f") else stroke)
        mj.mj_forward(m, d)
        quads = []
        for bid in np.unique(p.body_id):
            idx = [i for i in range(p.n) if p.body_id[i] == bid and p.kind[i] == WING]
            if not idx:
                continue
            R = d.xmat[bid].reshape(3, 3)
            s = R @ p.span_local[idx[0]]
            x = d.xpos[bid] + p.pos_local[idx] @ R.T
            u = x @ s
            order = np.argsort(u)
            # The strips are quadrature nodes (half-width ones at the two ends,
            # centred on them): drawn as the cells between them, so the outline
            # is the surface the model integrates over.
            edges = np.concatenate([[u[order[0]]], 0.5 * (u[order[1:]] + u[order[:-1]]), [u[order[-1]]]])
            for n_, i_ in enumerate(order):
                i = idx[i_]
                c = R @ p.chord_local[i]
                base = x[i_] - u[i_] * s
                le, te = base - 0.25 * p.chord[i] * c, base + 0.75 * p.chord[i] * c
                a0, a1 = edges[n_] * s, edges[n_ + 1] * s
                quads.append((np.array([le + a0, le + a1, te + a1, te + a0, le + a0]), int(bid)))
        return quads

    rest = pose(0.0, 0.0)
    com = (dry[:, None] * d.xipos).sum(0) / dry.sum()
    anchors = [(jn[j], d.xanchor[j].copy(), d.xaxis[j].copy(), int(m.jnt_bodyid[j]))
               for j in range(m.njnt) if m.jnt_type[j] == 3]
    # The hull is a mesh: drawn as its length and radius along its own axis.
    Rh = d.xmat[hull].reshape(3, 3)
    ax_h = Rh[:, int(np.argmax(np.abs(m.body_ipos[hull])))]
    geoms = [(float(body["hull_radius"]), d.xpos[hull].copy(), d.xpos[hull] + float(body["hull_length"]) * ax_h)]
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != 0 and int(m.geom_type[g]) == 3:          # capsules: booms, struts
            R = d.geom_xmat[g].reshape(3, 3)
            geoms.append((float(m.geom_size[g][0]), d.geom_xpos[g] - m.geom_size[g][1] * R[:, 2],
                          d.geom_xpos[g] + m.geom_size[g][1] * R[:, 2]))
    tail_b = {b for b in range(m.nbody) if int(m.body_parentid[b]) not in (0, hull) and b != hull}
    up, dn = pose(stroke_deg / 2, feather_deg), pose(-stroke_deg / 2, -feather_deg)

    fig, axs = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.25]))
    views = ((0, 1, "top view (x forward, y to the left), m"), (0, 2, "side view, m"),
             (1, 2, "front view, m"))
    for ax, (a, b, name) in zip(axs, views):
        for rad, e0, e1 in geoms:
            ax.plot([e0[a], e1[a]], [e0[b], e1[b]], color="0.45", lw=max(rad * 260, 1.5),
                    solid_capstyle="round", zorder=1)
        for q, bid in rest:
            col = "tab:orange" if bid in tail_b else "tab:blue"
            ax.fill(q[:, a], q[:, b], facecolor=col, alpha=0.35, edgecolor=col, lw=0.6, zorder=2)
        if stroke_deg and (a, b) != (0, 1):
            for qs in (up, dn):
                for q, bid in qs:
                    if bid not in tail_b:
                        ax.plot(q[:, a], q[:, b], color="tab:blue", lw=0.5, alpha=0.5, zorder=2)
        seen = set()
        for name_j, x, axis, bid in anchors:
            key = (round(x[a], 3), round(x[b], 3), name_j.endswith("f"), bid in tail_b)
            if key in seen:
                continue
            seen.add(key)
            col = "tab:red" if bid in tail_b else ("tab:green" if name_j.endswith("f") else "k")
            e = 0.09 * axis
            ax.plot([x[a] - e[a], x[a] + e[a]], [x[b] - e[b], x[b] + e[b]], color=col, lw=2.2, zorder=4)
            ax.plot(x[a], x[b], "o", color=col, ms=4, zorder=4)
        ax.plot(com[a], com[b], marker="P", color="crimson", ms=11, zorder=5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(name, fontsize=10)
    wing = [q for q, bid in rest if bid not in tail_b]
    tail = [q for q, bid in rest if bid in tail_b]
    area = lambda qs: sum(abs(np.cross(q[1] - q[0], q[3] - q[0])[2]) for q in qs)
    chord = lambda on_tail: float(p.chord[[i for i in range(p.n) if p.kind[i] == WING
                                           and (int(p.body_id[i]) in tail_b) == on_tail]].mean())
    width = lambda qs: max(q[:, 1].max() for q in qs) - min(q[:, 1].min() for q in qs)
    built = dict(span=width(wing), wing_area=area(wing), chord=chord(False), mass=float(dry.sum()))
    lines = ["wing: span %.2f m, chord %.3f m, area %.3f m2" % (built["span"], built["chord"], built["wing_area"]),
             "all-up mass %.2f kg; centre of mass (red cross) %.3f m ahead of the hull's tail end" % (mass, com[0])]
    if tail:
        tx = np.mean([q[:4, 0].mean() for q in tail])
        built.update(tail_span=width(tail), tail_area=area(tail), tail_chord=chord(True))
        lines.append("tail: span %.2f m, chord %.3f m, area %.3f m2, %.2f m behind the centre of mass" % (
            built["tail_span"], built["tail_chord"], built["tail_area"], com[0] - tx))
    table = [(k, v, built[k], abs(built[k] - v) <= tol * abs(v)) for k, v in (spec or {}).items()]
    cells = ["%s %.3g design, %.3g built: %s" % (k, v, bv, "OK" if ok else "DIFFERS") for k, v, bv, ok in table]
    for n_ in range(0, len(cells), 4):
        lines.append(("against the design (within %.0f%%): " % (100 * tol) if n_ == 0 else "    ")
                     + "; ".join(cells[n_:n_ + 4]))
    lines.append("black: stroke axis; green: feather axis (span); red: tail hinge; thin outlines: stroke ends")
    fig.suptitle(title, fontsize=13, y=0.99)
    fig.text(0.01, 0.01, "\n".join(lines + list(notes)), fontsize=9, va="bottom", family="monospace")
    fig.tight_layout(rect=(0, 0.05 + 0.028 * (len(lines) + len(notes)), 1, 0.95))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    for k, v, bv, ok in table:
        print("%-22s %-10s design %.4g built %.4g %s" % (Path(out).name, k, v, bv, "OK" if ok else "DIFFERS"))
    return table


def main() -> int:
    from l1a import AREA, BODY, LENGTH, MASS, SPAN
    which = sys.argv[1]
    base = dict(span=SPAN, wing_area=AREA, chord=AREA / (2 * LENGTH), mass=MASS)
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    if which == "S1":
        for ks in (0.4, 0.6):
            draw(dict(BODY, wing_length=ks * LENGTH), figs / f"S1_wing{ks}.png",
                 f"Pass S1: the water wing, folded to {ks} of its length, on the two rigs",
                 ["body held at 0.3 m/s in water; stroke 40 or 85 deg at 0.25-1 Hz; feather 20-60 deg, cam-driven",
                  "held rig: motion prescribed; torque-driven rig: stroke and feather tracked by servos",
                  "the fold is modelled as a shorter wing of the same chord; its mechanism is not modelled"],
                 stroke_deg=85.0, feather_deg=30.0,
                 spec=dict(span=2 * (ks * LENGTH + BODY["hull_radius"]), wing_area=ks * AREA,
                           chord=base["chord"], mass=MASS))
    elif which == "L1m":
        for name, tail in (("0.06", dict(stab_length=0.25, stab_chord=0.12, stab_boom=0.3)),
                           ("0.03", dict(stab_length=0.18, stab_chord=0.085, stab_boom=0.3))):
            draw(dict(BODY, **tail), figs / f"L1m_tail{name}.png",
                 f"Pass 1m: the cam-driven body with an all-moving tail of {name} m2 on a 0.3 m boom",
                 ["released in x, z and pitch at 6-8 m/s; stroke 60-85 deg at 3-4 Hz; feather 20-30 deg, cam-driven",
                  "tail: both halves turn together about their span; fixed at zero incidence, or pitch feedback",
                  "not modelled: the tail in the wings' wake; roll and yaw; the boom's and the cam's own structure"],
                 stroke_deg=60.0, feather_deg=20.0,
                 spec=dict(base, tail_span=2 * tail["stab_length"], tail_chord=tail["stab_chord"],
                           tail_area=2 * tail["stab_length"] * tail["stab_chord"]))
    else:
        raise SystemExit("unknown figure set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
