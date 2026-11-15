"""A wing with a one-way flap, built as a body of its own (pass 1j).

Each wing body is cut along the span at x_h of its chord.  The front part
stays on the wing body; the rear part is a child body on a hinge along the
cut, free to swing one way up to ``open_deg`` and stopped flat against the
front part the other way (a joint limit).  Its fluid strips ride on that
body, so the flap's area, incidence and added mass follow its real motion;
nothing is switched in time, unlike pass 1e's `chord_law`.

Mass is split by chord, each part a uniform plate.  ``sign`` +1 lets the flap
swing up (the level machine's +Z) on both wings, -1 down, 0 welds it shut (the
control).
Strips assume an untwisted wing whose chord axis is the body's -Y.
"""
import xml.etree.ElementTree as ET

import numpy as np


def compile_with_flap(p, *, x_h: float, open_deg: float, sign: int, k_f: float, damp_f: float,
                      **kw):
    import mujoco

    from dytiscidae.core.mjcf import build_model_xml
    from dytiscidae.core.phenotype import build_panels
    from dytiscidae.physics.fluid import WING, PanelSet

    xml, names = build_model_xml(p, **kw)
    root = ET.fromstring(xml)
    wings = [b for b in root.iter("body")
             if b.find(f"joint[@name='{b.get('name')}_jf']") is not None]
    # The mirrored wing is the first turned half a turn about the hull's long
    # axis, so its local normal points the other way in the world.  ``sign`` is
    # a world direction (+1: the flap swings up, towards +Z of the level
    # machine), so it is flipped per wing; without this the two wings' flaps
    # opened on opposite half-strokes and +1 and -1 were mirror images.
    plain = mujoco.MjModel.from_xml_string(xml)
    pd = mujoco.MjData(plain)
    mujoco.mj_forward(plain, pd)
    hull_R = pd.xmat[mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_BODY, "seg0_hull")].reshape(3, 3)
    up = {}
    for b in wings:
        R = pd.xmat[mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_BODY, b.get("name"))].reshape(3, 3)
        nz = float((hull_R.T @ (R @ np.array([0.0, 0.0, -1.0])))[2])   # n = span x chord = -Z local
        assert abs(nz) > 0.9, "a wing's normal is expected along the hull's vertical at rest"
        up[b.get("name")] = 1 if nz > 0 else -1
    for b in wings:
        g = b.find("geom")
        sx, sy, sz = (float(v) for v in g.get("size").split())
        mass = float(g.get("mass"))
        c = 2.0 * sy
        g.set("size", f"{sx} {x_h * c / 2} {sz}")
        g.set("pos", f"{sx} {-x_h * c / 2} 0")
        g.set("mass", f"{mass * x_h}")
        fl = ET.SubElement(b, "body", name=f"{b.get('name')}_flap", pos=f"0 {-x_h * c} 0")
        if sign:
            lim = f"0 {open_deg}" if sign * up[b.get("name")] > 0 else f"{-open_deg} 0"
            ET.SubElement(fl, "joint", name=f"{b.get('name')}_jflap", type="hinge", axis="1 0 0",
                          range=lim, limited="true", stiffness=f"{k_f}", damping=f"{damp_f}",
                          armature="1e-5", solreflimit="0.004 1")
        ET.SubElement(fl, "geom", name=f"{b.get('name')}_flapg", type="box",
                      size=f"{sx} {(1 - x_h) * c / 2} {sz}", pos=f"{sx} {-(1 - x_h) * c / 2} 0",
                      mass=f"{mass * (1 - x_h)}", contype="0", conaffinity="0",
                      rgba=g.get("rgba", "0.8 0.5 0.2 0.9"))
    comp = root.find("compiler")
    if comp is not None and comp.get("angle") == "radian":     # the project's models are
        for j in root.iter("joint"):
            if j.get("name", "").endswith("_jflap"):
                lo, hi = (float(v) for v in j.get("range").split())
                j.set("range", f"{np.radians(lo)} {np.radians(hi)}")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    pan = build_panels(p, model)
    wing_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b.get("name")): b.get("name")
                for b in wings}
    sel = np.isin(pan.body_id, list(wing_ids)) & (pan.kind == WING)
    ch = pan.chord_local[sel]
    assert np.allclose(ch, [0.0, -1.0, 0.0], atol=1e-6), "flap strips assume an untwisted wing"
    base = pan.pos_local[sel] - 0.25 * pan.chord[sel, None] * ch       # the leading edge

    def part(idx, frac, offset_body):
        kw2 = {}
        for name in ("span_local", "chord_local", "half_height", "kind", "aspect_ratio",
                     "cd_bluff", "pitch_axis", "camber"):
            kw2[name] = getattr(pan, name)[idx]
        return dict(kw2, chord=pan.chord[idx] * frac, dr=pan.dr[idx],
                    volume=pan.volume[idx] * frac, volume_buoyant=pan.volume_buoyant[idx] * frac,
                    body_id=offset_body)

    rest = PanelSet(**{k: getattr(pan, k)[~sel] for k in (
        "body_id", "pos_local", "span_local", "chord_local", "chord", "dr", "volume",
        "half_height", "kind", "aspect_ratio", "cd_bluff", "pitch_axis", "volume_buoyant",
        "camber")}, ext_local=pan.ext_local[~sel])
    idx = np.flatnonzero(sel)
    front = part(idx, x_h, pan.body_id[idx])
    front["pos_local"] = base + 0.25 * x_h * pan.chord[idx, None] * ch
    flap_body = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                            wing_ids[int(b)] + "_flap") for b in pan.body_id[idx]])
    rear = part(idx, 1 - x_h, flap_body)
    rear["pos_local"] = base + 0.25 * (1 - x_h) * pan.chord[idx, None] * ch
    panels = PanelSet.concat([rest, PanelSet(**front), PanelSet(**rear)])
    return model, data, names, panels
