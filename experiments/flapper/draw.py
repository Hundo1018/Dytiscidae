"""A drawing of the mechanism a pass tested, from the built model itself:
top, side and front views with the lifting strips, joints, centre of mass and
the main dimensions, and a table of the design's figures against the built
ones.  Every pass writes one beside its result, and it is looked at.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/draw.py S1|L1m|L1n|past
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def draw(body: dict, out, title: str, notes: list, *, stroke_deg: float = 0.0,
         feather_deg: float = 0.0, mass: float | None = 0.60, spec: dict | None = None,
         tol: float = 0.02, feather_axis: float = 0.0, flap: dict | None = None) -> list:
    """spec: the design's figures (m, m2, kg) for any of span, wing_area,
    chord, mass, other_span, other_area, other_chord ("other": every lifting
    surface that is not a main wing).  Each is compared with what the built
    model measures; the table is printed on the drawing and returned as
    (name, design, built, ok).  A drawing is also looked at: the table cannot
    see a part that is in the wrong place.
    mass: the rig's ballasted all-up mass; None draws the model as built.
    feather_axis, flap: as `rig.run`'s feather_axis and one_way_flap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import dytiscidae.core.phenotype as PH
    PH.AVIONICS_MASS = 0.005
    from body import built
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    from dytiscidae.physics.fluid import WING

    ph = built(body)
    if flap is not None:
        import dytiscidae.envs.triphibian as TR
        from flapwing import compile_with_flap
        orig = TR.compile_phenotype
        TR.compile_phenotype = lambda p_, **kw: compile_with_flap(p_, **flap, **kw)
        try:
            env = TriphibianEnv(ph, seed=0, timestep=1e-3)
        finally:
            TR.compile_phenotype = orig
    else:
        env = TriphibianEnv(ph, seed=0, timestep=1e-3)
    env.reset(Domain.AIR, randomise=False)
    d, m, mj = env.data, env.model, env._mj
    hull = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "seg0_hull")
    dry = env.solver._dry_mass.copy()
    if mass is not None:
        dry[hull] += mass - dry.sum()                  # the rig's ballast, at the hull's centre
    p = env.solver.panels
    jn = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == 3]
    # A main wing: a body on the hull with a stroke hinge and a feather hinge.
    main = {int(m.jnt_bodyid[j]) for j in hinge if jn[j].endswith("_jf")
            and int(m.body_parentid[int(m.jnt_bodyid[j])]) == hull}

    def is_main(b):
        while b not in (0, hull):
            if b in main:
                return True
            b = int(m.body_parentid[b])
        return False

    if feather_axis:
        for j in hinge:
            b = int(m.jnt_bodyid[j])
            if not jn[j].endswith("_jf") or b not in main:
                continue
            sel = (p.body_id == b) & (p.kind == WING)
            cdir = p.chord_local[sel][0] / np.linalg.norm(p.chord_local[sel][0])
            if float(p.pos_local[sel].mean(0) @ cdir) < 0.0:
                cdir = -cdir
            m.jnt_pos[j] = m.jnt_pos[j] + feather_axis * float(p.chord[sel].mean()) * cdir

    def pose(stroke, feather):
        d.qpos[:] = m.qpos0
        d.qpos[:3] = 0.0
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for j in hinge:
            if int(m.jnt_bodyid[j]) in main:
                d.qpos[m.jnt_qposadr[j]] = math.radians(feather if jn[j].endswith("_jf") else stroke)
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
            if len(idx) > 1:
                edges = np.concatenate([[u[order[0]]], 0.5 * (u[order[1:]] + u[order[:-1]]), [u[order[-1]]]])
            else:
                edges = np.array([u[0] - 0.5 * p.dr[idx[0]], u[0] + 0.5 * p.dr[idx[0]]])
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
    anchors = [(jn[j], d.xanchor[j].copy(), d.xaxis[j].copy(), int(m.jnt_bodyid[j])) for j in hinge]
    # The hull is a mesh: drawn as its length and radius along its own axis.
    from body import DEFAULTS
    full = dict(DEFAULTS, **body)
    Rh = d.xmat[hull].reshape(3, 3)
    ax_h = Rh[:, int(np.argmax(np.abs(m.body_ipos[hull])))]
    geoms = [(float(full["hull_radius"]), d.xpos[hull].copy(), d.xpos[hull] + float(full["hull_length"]) * ax_h)]
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != 0 and int(m.geom_type[g]) == 3:          # capsules: booms, struts
            R = d.geom_xmat[g].reshape(3, 3)
            geoms.append((float(m.geom_size[g][0]), d.geom_xpos[g] - m.geom_size[g][1] * R[:, 2],
                          d.geom_xpos[g] + m.geom_size[g][1] * R[:, 2]))
    up, dn = pose(stroke_deg / 2, feather_deg), pose(-stroke_deg / 2, -feather_deg)

    fig, axs = plt.subplots(1, 3, figsize=(17, 6.4), gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.25]))
    views = ((0, 1, "top view (x forward, y to the left), m"), (0, 2, "side view, m"),
             (1, 2, "front view, m"))
    for ax, (a, b, name) in zip(axs, views):
        for rad, e0, e1 in geoms:
            ax.plot([e0[a], e1[a]], [e0[b], e1[b]], color="0.45", lw=max(rad * 260, 1.5),
                    solid_capstyle="round", zorder=1)
        for q, bid in rest:
            col = "tab:blue" if is_main(bid) and bid in main else ("tab:purple" if is_main(bid) else "tab:orange")
            ax.fill(q[:, a], q[:, b], facecolor=col, alpha=0.35, edgecolor=col, lw=0.6, zorder=2)
        if stroke_deg and (a, b) != (0, 1):
            for qs in (up, dn):
                for q, bid in qs:
                    if is_main(bid):
                        ax.plot(q[:, a], q[:, b], color="tab:blue", lw=0.5, alpha=0.5, zorder=2)
        seen = set()
        for name_j, x, axis, bid in anchors:
            kind = ("feather" if name_j.endswith("_jf") else "stroke") if bid in main else "other"
            key = (round(x[a], 3), round(x[b], 3), kind)
            if key in seen:
                continue
            seen.add(key)
            col = {"stroke": "k", "feather": "tab:green", "other": "tab:red"}[kind]
            e = 0.16 * max(float(full["wing_length"]), 0.1) * axis
            ax.plot([x[a] - e[a], x[a] + e[a]], [x[b] - e[b], x[b] + e[b]], color=col, lw=2.2, zorder=4)
            ax.plot(x[a], x[b], "o", color=col, ms=4, zorder=4)
        ax.plot(com[a], com[b], marker="P", color="crimson", ms=11, zorder=5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(name, fontsize=10)
    if full.get("wing_section"):
        import sections
        xy = sections.coordinates(full["wing_section"])
        ins = axs[1].inset_axes([0.02, 0.80, 0.55, 0.18])
        ins.plot(xy[:, 0], xy[:, 1], color="tab:blue", lw=1.2)
        ins.set_aspect("equal")
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_title("section %s: thickness %.1f%%, height %.1f%% of chord" % (
            full["wing_section"], 100 * _thickness(xy), 100 * (xy[:, 1].max() - min(xy[0, 1], xy[-1, 1]))), fontsize=7)
    wing = [q for q, bid in rest if is_main(bid)]
    other = [q for q, bid in rest if not is_main(bid)]

    def area(qs):
        return sum(float(np.linalg.norm(np.cross(q[1] - q[0], q[3] - q[0]))) for q in qs)

    def chord(on_main):
        return float(p.chord[[i for i in range(p.n) if p.kind[i] == WING
                              and is_main(int(p.body_id[i])) == on_main]].mean())

    width = lambda qs: max(q[:, 1].max() for q in qs) - min(q[:, 1].min() for q in qs)
    built_ = dict(span=width(wing), wing_area=area(wing), chord=chord(True), mass=float(dry.sum()))
    lines = ["main wings: span %.3f m, chord %.3f m, area %.4f m2" % (built_["span"], built_["chord"], built_["wing_area"]),
             "mass %.3f kg%s; centre of mass (red cross) %.3f m ahead of the hull's tail end, %.3f m above its axis" % (
                 dry.sum(), " (ballasted)" if mass is not None else " (as built)", com[0], com[2])]
    if other:
        ox = np.mean([q[:4, 0].mean() for q in other])
        built_.update(other_span=width(other), other_area=area(other), other_chord=chord(False))
        lines.append("other surfaces (orange): span %.3f m, chord %.3f m, area %.4f m2, centred %.3f m behind the centre of mass" % (
            built_["other_span"], built_["other_chord"], built_["other_area"], com[0] - ox))
    table = [(k, v, built_[k], abs(built_[k] - v) <= tol * abs(v)) for k, v in (spec or {}).items()]
    cells = ["%s %.3g design, %.3g built: %s" % (k, v, bv, "OK" if ok else "DIFFERS") for k, v, bv, ok in table]
    for n_ in range(0, len(cells), 4):
        lines.append(("against the design (within %.0f%%): " % (100 * tol) if n_ == 0 else "    ")
                     + "; ".join(cells[n_:n_ + 4]))
    lines.append("black: stroke axis; green: feather axis; red: any other hinge; purple: a panel carried by a wing; thin outlines: stroke ends")
    fig.suptitle(title, fontsize=13, y=0.99)
    fig.text(0.01, 0.01, "\n".join(lines + list(notes)), fontsize=9, va="bottom", family="monospace")
    fig.tight_layout(rect=(0, 0.05 + 0.028 * (len(lines) + len(notes)), 1, 0.95))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    for k, v, bv, ok in table:
        print("%-24s %-11s design %.4g built %.4g %s" % (Path(out).name, k, v, bv, "OK" if ok else "DIFFERS"))
    return table


def _thickness(xy) -> float:
    """Largest gap between the two surfaces of a profile given as one loop
    from the trailing edge round the leading edge and back."""
    i = int(np.argmin(xy[:, 0]))
    up, lo = xy[: i + 1][::-1], xy[i:]
    x = np.linspace(0.02, 0.98, 97)
    return float(np.max(np.abs(np.interp(x, up[:, 0], up[:, 1]) - np.interp(x, lo[:, 0], lo[:, 1]))))


def asked(body: dict, **more) -> dict:
    """The design figures a body dict asks for: two wings of its length and
    chord, their roots on the hull's sides."""
    from body import DEFAULTS
    e = dict(DEFAULTS, **body)
    return dict(span=2 * (e["wing_length"] + e["hull_radius"]), wing_area=2 * e["wing_length"] * e["wing_chord"],
                chord=e["wing_chord"], **more)


def main() -> int:
    from l1a import AREA, BODY, BODY_S, LENGTH, MASS, SPAN
    which = sys.argv[1]
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    base = dict(span=SPAN, wing_area=AREA, chord=AREA / (2 * LENGTH), mass=MASS)
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
                 spec=dict(base, other_span=2 * tail["stab_length"], other_chord=tail["stab_chord"],
                           other_area=2 * tail["stab_length"] * tail["stab_chord"]))
    elif which == "L1n":
        tail = dict(stab_length=0.25, stab_chord=0.12, stab_boom=0.3)
        draw(dict(BODY_S, **tail), figs / "L1n_fixture.png",
             "Pass 1n: the fixture with its shoulders on the hull's sides, and the 0.06 m2 tail",
             ["air: take-off at rest and cruise at 6-11 m/s, cam-driven feather; released runs carry the tail",
              "water: the wing folded to 0.4 and 0.6 of its length (drawn separately), no tail",
              "not modelled: the tail in the wings' wake; roll and yaw; the fold's and the cam's own structure"],
             stroke_deg=85.0, feather_deg=30.0,
             spec=dict(base, other_span=0.5, other_chord=0.12, other_area=0.06))
        for ks in (0.4, 0.6):
            draw(dict(BODY_S, wing_length=ks * LENGTH), figs / f"L1n_water{ks}.png",
                 f"Pass 1n: the water wing folded to {ks} of its length, shoulders on the hull's sides",
                 ["body held at 0.3 m/s in water; stroke 40 or 85 deg at 0.25-1 Hz; feather 20-60 deg, cam-driven",
                  "the fold is modelled as a shorter wing of the same chord; its mechanism is not modelled"],
                 stroke_deg=85.0, feather_deg=30.0,
                 spec=dict(span=2 * (ks * LENGTH + BODY["hull_radius"]), wing_area=ks * AREA,
                           chord=base["chord"], mass=MASS))
    elif which == "R0s":
        import reproduce as RP
        for name, k in RP.MACHINES.items():
            body = dict(RP.body(name), wing_seat=0.04)
            draw(body, figs / f"R0s_{name}.png", f"R0 again: {name} at its reported span",
                 ["body held, lift trimmed to the weight; rectangular flat wings, no tail (as R0)"],
                 stroke_deg=60.0, mass=k["mass"], spec=dict(span=k["span"], wing_area=k["area"], mass=k["mass"]))
    elif which == "W4":
        for lam in (1.0, 0.6, 0.4):
            body = dict(BODY_S, wing_shape=dict(camber=0.04, taper=lam, twist_tip_deg=-8.0), wing_section="goe417a")
            draw(body, figs / f"W4_taper{lam}.png",
                 f"Pass W4: tip chord {lam} of the root's, 8 deg of washout, section Goe 417a",
                 ["held, 6 and 8 m/s, stroke 30-60 deg at 2-3.5 Hz, feather 0-20 deg; take-off at rest; twist 0 and -8 deg both run",
                  "the section is a measured polar applied strip by strip; the wing is rigid; the leading edge is straight"],
                 stroke_deg=40.0, feather_deg=0.0, spec=dict(span=SPAN, wing_area=AREA, mass=MASS))
    elif which == "W6":
        # draw.py W6 <cruise twist, deg> <tip feather of a share-1 beat, deg>
        twc, psi = float(sys.argv[2]), float(sys.argv[3])
        for tag, tw, what in (
                ("takeoff", 0.0, "the take-off setting: no twist"),
                ("cruise", twc, f"the cruise setting: the tip held {twc:+.0f} deg against the root"),
                ("beat_down", twc - psi, f"a tip-feathered beat at its nose-down end: tip {twc - psi:+.0f} deg, root unmoved"),
                ("beat_up", twc + psi, f"a tip-feathered beat at its nose-up end: tip {twc + psi:+.0f} deg, root unmoved")):
            body = dict(BODY_S, wing_shape=dict(camber=0.04, twist_tip_deg=tw), wing_section="goe417a+lift")
            draw(body, figs / f"W6_{tag}.png", f"Pass W6: Goe 417a rectangle, {what}",
                 ["the twist is linear along the span and turns the fluid strips only: no torsion structure, no twist actuator is modelled",
                  "held rig; cruise 6 and 8 m/s, stroke 30-60 deg at 2-3.5 Hz; take-off at rest, stroke 85 deg at 3-5 Hz"],
                 stroke_deg=40.0, feather_deg=0.0, spec=dict(span=SPAN, wing_area=AREA, mass=MASS))
    elif which == "past":
        import landwalk as LW
        import reproduce as RP
        small = dict(wing_length=0.25, wing_chord=0.25 / 3, stroke_range=1.45)
        jobs = [
            ("F0_F3_W_small", small, "Passes F0, F0b/c, F3, W1, W2: the small flapper (wing 0.25 m, aspect ratio 3)",
             dict(stroke_deg=120.0, feather_deg=45.0, mass=None, spec=asked(small))),
            ("F1a_hand", dict(small, hand_fraction=0.5), "Pass F1a: the two-panel wing (the outer half turns on a wrist)",
             dict(stroke_deg=120.0, feather_deg=30.0, mass=None, spec=asked(small))),
            ("F2_layoutA", LW.LAYOUTS["A"], "Passes F2, F3b: layout A (thumbs under the wings, a tail paddle)",
             dict(stroke_deg=60.0, mass=None, spec=asked(LW.LAYOUTS["A"]))),
            ("F2_layoutB", dict(LW.LAYOUTS["B"], hull_radius=0.0209),
             "Passes F2-F4, T1: layout B (two wing pairs, a thumb under each), hull radius 0.0209",
             dict(stroke_deg=60.0, mass=None,
                  spec=dict(span=2 * (0.21 + 0.0209), wing_area=4 * 0.21 * 0.07, chord=0.07))),
        ]
        for name in RP.MACHINES:
            k = RP.MACHINES[name]
            jobs.append((f"R0_{name}", RP.body(name), f"Pass R0: {name} as reproduced",
                         dict(stroke_deg=60.0, mass=k["mass"], spec=dict(span=k["span"], wing_area=k["area"], mass=k["mass"]))))
        jobs += [
            ("L1a_reference", BODY, "Passes 1a, 1b, 1e-1i, 1k: the reference machine as those passes built it",
             dict(stroke_deg=85.0, feather_deg=30.0, spec=base)),
            ("L1g_axis0.3", BODY, "Passes 1g-1i: the feather axis moved aft to 0.3 chord (0-0.4 swept)",
             dict(stroke_deg=85.0, feather_deg=30.0, spec=base, feather_axis=0.3)),
            ("L1d_fold0.6x0.6", dict(BODY, wing_length=0.6 * LENGTH, wing_chord=0.6 * BODY["wing_chord"]),
             "Pass 1d: the wing folded to 0.6 of its length and 0.6 of its chord",
             dict(stroke_deg=85.0, feather_deg=30.0,
                  spec=dict(span=2 * (0.6 * LENGTH + 0.04), wing_area=0.36 * AREA, chord=0.6 * base["chord"], mass=MASS))),
            ("L1j_flap", dict(BODY, wing_length=0.4 * LENGTH), "Pass 1j: the one-way flap (rear 60% of the 0.4 wing on a hinge)",
             dict(stroke_deg=85.0, feather_deg=0.0, flap=dict(x_h=0.4, open_deg=80.0, sign=1, k_f=0.005, damp_f=0.002),
                  spec=dict(span=2 * (0.4 * LENGTH + 0.04), wing_area=0.4 * AREA, chord=base["chord"], mass=MASS))),
        ]
        only = sys.argv[2:] or None
        for name, body, title, kw in jobs:
            if only and name not in only:
                continue
            draw(body, figs / f"past_{name}.png", title, ["drawn after the fact from the pass's own body definition"], **kw)
    else:
        raise SystemExit("unknown figure set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
