"""S0b, machine T5b (results/S0b_probes.md, D1-D2 and B14-B17): T5 with a dorsal float.

Why (measured on T5, B11 and D2 of this part): flooded, the pontoons' water and their own volume cancel at
every fill near 1, so the righting stiffness of the neutral machine is that of the dry structure alone,
K = (dry structure's weight) x (height of its buoyancy centre over its mass centre) = 0.52 N m/rad on T5.
Water ballast cannot raise it wherever it is put.  What can: a fixed air volume high on the machine, which the
neutral condition (weight = buoyancy) pays for with dense dry mass; the pontoon shells, which are derived
from that condition, carry it, and they are the lowest thing on the machine.

T5b = T5 (`s0b_t5_machine`, unchanged and imported) plus
- a dorsal float: a capsule (radius ``r``, overall length ``L``) on the centre line of the hull at hull height
  ``zu``, its station at the pontoons' station (so the float and the flooded machine stay trimmed in pitch);
  the wing roots are 0.11 m + the hull radius off the centre line, so it sits in the gap between the raised
  wings (clearance read by `s0b_t5b_clear.py`).  Mass 1 g; fluid elements built as the pontoons' slices on the
  hull body (bluff, cd 0.20).  Nothing is typed: its buoyancy and drag come from the solver.
- the pontoon mass is whatever neutrality at F_NEUTRAL then requires (T5's rule), so the dry mass grows by
  the dorsal float's buoyancy; part of it spread on the capsule, the rest at one point low in the pontoon
  (``low`` x r under the axis), that point's station solved so the dry mass centre is over the pontoons' centre.
"""
from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0b_machine as T  # noqa: E402
import s0b_t5_machine as T5  # noqa: E402
from s0b_t5_machine import (F_NEUTRAL, G, GEOM, PONT, TANK, ROOT_CUT, SWEEP, FEATHER, N_SLICE, STRUT_R, STRUT_KG_M,  # noqa: E402,F401
                            TANK_EMPTY, _capsule_slices, _strut_ends, capsule_volume, set_fill, facts, _raise, _hull_frame)

CALIB = HERE / "results" / "S0b_T5b_calib.json"
#: default T5b: T5's pontoons lowered to z -0.23 (the swim stroke's flipping trailing edge sweeps 0.15 m under the wing plane: B15), plus the dorsal float
#: (radius 0.025, length 0.40, height 0.20 m): the sizing in results/S0b_probes.md
GEOM_B = dict(GEOM, zp=-0.23, dr=0.025, dL=0.40, dz=0.20, low=0.6, cap=0.5)


def _hooks(g, xp, m_cap, m_pt, x_pt):
    xh0, ph0 = T5._hooks({k: g[k] for k in ("r", "L", "ys", "zp")}, xp, m_cap, m_pt, x_pt)
    dr, dL, dz = g.get("dr", 0.0), g.get("dL", 0.0), g.get("dz", 0.0)
    low = g.get("low", 0.6)
    hc = g["L"] / 2 - g["r"]

    def xml_hook(root, hull):
        xh0(root, hull)
        if dr > 0.0:
            h = dL / 2 - dr
            ET.SubElement(hull, "geom", name="t5b_dorsal", type="capsule",
                          fromto=f"{xp - h:.6f} 0 {dz:.6f} {xp + h:.6f} 0 {dz:.6f}", size=f"{dr}", mass="0.001",
                          rgba="0.9 0.8 0.2 1")
        if low != 0.6:                               # the pontoon's low point at ``low`` x r under its axis
            for b in hull.findall("body"):
                if b.get("name") in PONT:
                    for gg in b.findall("geom"):
                        if gg.get("name", "").endswith("_k"):
                            gg.set("pos", f"{x_pt:.6f} 0 {-low * g['r']:.6f}")

    def panel_hook(model, pan):
        import mujoco
        from dytiscidae.physics.fluid import BLUFF, PanelSet
        pan = ph0(model, pan)
        if dr <= 0.0:
            return pan
        hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seg0_hull")
        sl = _capsule_slices(dr, dL)
        n = len(sl)
        vol = np.array([s[2] for s in sl])
        re = np.array([s[3] for s in sl])
        dx = np.array([s[1] for s in sl])
        ps = PanelSet(
            body_id=np.full(n, hid), pos_local=np.array([[xp + s[0], 0.0, dz] for s in sl]),
            span_local=np.tile([1.0, 0.0, 0.0], (n, 1)), chord_local=np.tile([0.0, -1.0, 0.0], (n, 1)),
            chord=2 * re, dr=dx, volume=vol, volume_buoyant=vol.copy(), half_height=re.copy(),
            kind=np.full(n, BLUFF), aspect_ratio=np.ones(n), cd_bluff=np.full(n, 0.20),
            pitch_axis=np.full(n, 0.5), ext_local=np.stack([dx, 2 * re, 2 * re], axis=1))
        return PanelSet.concat([pan, ps])
    return xml_hook, panel_hook


def _build(domain, g, xp, m_cap, m_pt, x_pt, motor="x4", **kw):
    xh, ph = _hooks(g, xp, m_cap, m_pt, x_pt)
    fr = tuple(g["feather_range"]) if g.get("feather_range") else FEATHER      # B15g: a feather range wide enough to flip the wing
    mc = T.make(domain, b=None, motor=motor, root_cut=ROOT_CUT, sweep_range_deg=SWEEP, feather_range_deg=fr,
                xml_hook=xh, panel_hook=ph, **kw)
    mj, m = mc.mj, mc.m
    mc.pont = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in PONT]
    mc.tank = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in TANK]
    mc.rho_w = float(mc.env.medium.water.rho)
    mc.v_pont = capsule_volume(g["r"], g["L"])
    mc.tank_I0 = mc.env.solver._dry_inertia[mc.tank[0]].copy() / TANK_EMPTY
    mc.fill = 0.0
    return mc


def calibrate(g=None, force=False) -> dict:
    """T5's solve (xp, pontoon mass, low point's station; four passes), with the dorsal float in the
    non-pontoon structure."""
    g = dict(GEOM_B, **(g or {}))
    key = json.dumps(g, sort_keys=True)
    store = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    if key in store and not force:
        return store[key]
    xp, m_cap, m_pt, x_pt = 0.235, 0.06, 0.06, 0.0
    out = {}
    for it in range(5):
        mc = _build("water", g, xp, m_cap, m_pt, x_pt, pose=dict(x=-30.0, y=0.0, z=-5.0, vx=0.0, vz=0.0))
        dry = mc.env.solver._dry_mass
        rest = np.ones(mc.m.nbody, bool)
        rest[mc.pont + mc.tank] = False
        rest[0] = False
        B, rb, _, sub = T.static_buoyancy(mc)
        Bp = 2 * mc.rho_w * G * mc.v_pont
        cob_x = float(_hull_frame(mc, [rb[0], rb[1], mc.d.xpos[mc.hull][2]])[0])
        B_np = B - Bp
        cob_np_x = (B * cob_x - Bp * xp) / B_np
        _raise(mc)
        m_rest = float(dry[rest].sum())
        x_rest = float(_hull_frame(mc, (dry[rest, None] * mc.d.xipos[rest]).sum(0) / m_rest)[0])
        m_p = 0.5 * ((B_np + (1.0 - F_NEUTRAL) * Bp) / G - m_rest)
        M = m_rest + 2 * m_p
        out = dict(geom=g, xp=xp, m_cap=m_cap, m_pt=m_pt, x_pt=x_pt, m_rest=m_rest, x_rest_raised=x_rest,
                   B_all=B, B_np=B_np, B_pontoons=Bp, cob_np_x=cob_np_x, m_dry=float(dry.sum()),
                   v_pont_each=mc.v_pont, max_submerged=sub, passes=it + 1, resid_xp=cob_np_x - xp,
                   f_neutral=F_NEUTRAL, resid_mass=float(dry.sum()) - (B_np + (1.0 - F_NEUTRAL) * Bp) / G)
        if m_p <= 2e-3:
            out["no_spare"] = True
            break
        xp = cob_np_x
        m_cap = g.get("cap", 0.5) * m_p
        m_pt = m_p - m_cap
        x_pm = (M * xp - m_rest * x_rest) / (2 * m_p)
        x_pt = (x_pm - xp) * m_p / m_pt
        x_pt = min(max(x_pt, -(g["L"] / 2 - g["r"])), g["L"] / 2 - g["r"])
    store[key] = out
    CALIB.write_text(json.dumps(store, indent=1))
    return out


def make(domain="air", motor="x4", g=None, fill=0.0, **kw):
    c = calibrate(g)
    mc = _build(domain, c["geom"], c["xp"], c["m_cap"], c["m_pt"], c["x_pt"], motor=motor, **kw)
    mc.t5 = dict(c, motor=motor)
    if fill:
        set_fill(mc, fill)
    return mc


def dry_com(mc):
    dry = mc.env.solver._dry_mass
    return (dry[:, None] * mc.d.xipos).sum(0) / dry.sum()


def wrench_about_dry_com(mc):
    """(force, moment) of what the solver applied in the last step, about the DRY centre of mass (the weight acts
    there; `T.fluid_wrench` takes the moment about `mc.com()`, which carries the entrained water)."""
    m, d = mc.m, mc.d
    F = d.xfrc_applied[:, :3].copy()
    F[:, 2] -= (m.body_mass - mc.env.solver._dry_mass) * G
    c = dry_com(mc)
    M = d.xfrc_applied[:, 3:].sum(0) + np.cross(d.xipos - c, F).sum(0)
    return F.sum(0), M


def righting(mc, depth=5.0, eps=0.02):
    """Roll and pitch stiffness (N m/rad) of the machine as posed (wings, fill), fully submerged, at rest: the
    buoyancy's moment about the dry centre of mass at a small tilt about it (the weight acts at that point).
    Positive = restoring."""
    out = {}
    for axis, ax in (("roll", 0), ("pitch", 1)):
        q0 = mc.d.qpos[:7].copy()
        ms = []
        for sg in (+1.0, -1.0):
            h = sg * eps / 2
            dq = np.zeros(4)
            dq[0] = math.cos(h)
            dq[1 + ax] = math.sin(h)
            w1, x1, y1, z1 = q0[3:7]
            w2, x2, y2, z2 = dq
            mc.d.qpos[:7] = q0
            mc.d.qpos[3:7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                              w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
            mc.mj.mj_forward(mc.m, mc.d)
            mc.d.qpos[2] += -depth - float(dry_com(mc)[2])
            mc.mj.mj_forward(mc.m, mc.d)
            mc.d.qvel[:] = 0.0
            mc.d.xfrc_applied[:] = 0.0
            mc.env.solver.reset()
            mc.env.solver.apply(mc.d, 0.0)
            F, M = wrench_about_dry_com(mc)
            ms.append(float(M[ax]))
            mc.d.xfrc_applied[:] = 0.0
        mc.d.qpos[:7] = q0
        mc.mj.mj_forward(mc.m, mc.d)
        out[axis] = -(ms[0] - ms[1]) / (2 * eps)
    return out


if __name__ == "__main__":
    c = calibrate(force="--force" in sys.argv)
    print(json.dumps({k: v for k, v in c.items() if k != "geom"}))
