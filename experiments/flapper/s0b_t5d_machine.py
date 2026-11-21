"""S0b, machine T5d (DESIGN_S0b_bold.md, Addendum 3): T5b plus floodable columns through the waterline and a keel fin.

T5d = T5b (`s0b_t5b_machine`, imported, not edited: hull, wings, hinges, pontoons, struts, dorsal float) plus

- **columns**: vertical cylinders (radius ``cr``) from hull height ``cz0`` to ``cz1`` at hull stations ``xp + cdx`` (each ``cdx``
  on both sides, y = +-``cy``).  Geoms on the hull body (contact as every machine geom; the machine's own geoms do not collide with
  each other, so wing clearance is read kinematically, `s0b_t5d_static.py`).  Fluid: ``CN`` bluff slices each (cd 0.9, the solver's
  strut value for a cylinder across the flow), displaced volume = the slice's, so buoyancy and drag are the solver's, nothing typed.
- **ballast**: one fill scalar in [0, 1] over the whole floodable volume (pontoons + columns).  Order (a choice, stated): the
  pontoons flood first, then the four columns together from the bottom.  **Tank model**: the pontoons as T5 (uniform fill of the
  capsule, mass and inertia at its centre); each column's water is a solid slug standing on the column's foot, height
  ``fill_c x h``, its centre of mass at half that height (the tank body's `body_ipos`, mass and inertia are rewritten at every fill
  change, then `mj_setConst`).  The slug tilts with the column: no free surface inside a column and no flow between tanks during a
  manoeuvre (valved).  Size of what that neglects: `free_surface_bound`.
- **keel fin**: a flat plate in the hull's xz plane (y = 0), x from ``fx0`` to ``fx1`` (hull frame), z from ``fz0`` to ``fz1``:
  a box geom on the hull body and ``FN`` wing strips (span along hull z, chord along -x, normal +-y), as the tail's own strips.
- masses: column shells and the fin at the built main wing's mass per area (``AREAL``, read from the built T5b: 0.0342 kg on
  0.55 x 0.2576 m^2; a number to carry the bookkeeping, not a material); the pontoon mass is what neutrality at ``F_NEUTRAL`` of the
  whole floodable volume then requires (T5's rule), the low point's station solved so the dry CoM is over the pontoons' centre.
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
import s0b_t5b_machine as T5B  # noqa: E402
from s0b_t5_machine import (F_NEUTRAL, G, PONT, TANK, ROOT_CUT, SWEEP, FEATHER, TANK_EMPTY, capsule_volume, facts, _raise,  # noqa: E402,F401
                            _hull_frame)
from s0b_t5b_machine import dry_com, wrench_about_dry_com, righting  # noqa: E402,F401

CALIB = HERE / "results" / "S0b_T5d_calib.json"
AREAL = 0.0342 / (0.55 * 0.2576)          # kg/m^2: the built main wing (seg1_wing: 0.0342 kg, 0.55 m x 0.2576 m)
CN = 16                                   # fluid slices per column
FN = 6                                    # strips on the fin
CTANK = ("t5d_ctank_0", "t5d_ctank_1", "t5d_ctank_2", "t5d_ctank_3")
#: default T5d geometry: T5b's (pontoons, dorsal float) + columns + fin.  ``cdx``: column stations relative to xp.
GEOM_D = dict(T5B.GEOM_B, cr=0.025, cz0=-0.20, cz1=0.20, cdx=[-0.19, 0.19], cy=0.20,
              fx0=-0.12, fx1=0.08, fz0=-0.20, fz1=-0.06)


def _cols(g, xp):
    """[(x, y, side)] of the columns in the hull frame; side +1 left."""
    return [(xp + dx, sg * g["cy"], sg) for dx in g.get("cdx", []) for sg in (1.0, -1.0)]


def _hooks(g, xp, m_cap, m_pt, x_pt):
    xh0, ph0 = T5B._hooks(g, xp, m_cap, m_pt, x_pt)
    cr, z0, z1 = g.get("cr", 0.0), g.get("cz0", 0.0), g.get("cz1", 0.0)
    h = z1 - z0
    cols = _cols(g, xp) if cr > 0 else []
    fin = g.get("fx1", 0.0) > g.get("fx0", 0.0)

    def xml_hook(root, hull):
        xh0(root, hull)
        for i, (x, y, sg) in enumerate(cols):
            ET.SubElement(hull, "geom", name=f"t5d_col_{i}", type="cylinder",
                          fromto=f"{x:.6f} {y:.6f} {z0:.6f} {x:.6f} {y:.6f} {z1:.6f}", size=f"{cr}",
                          mass=f"{AREAL * (2 * math.pi * cr * h + 2 * math.pi * cr * cr):.9g}", rgba="0.2 0.7 0.7 0.6")
            tb = ET.SubElement(hull, "body", name=CTANK[i], pos=f"{x:.6f} {y:.6f} {z0:.6f}")
            ET.SubElement(tb, "geom", name=CTANK[i] + "_g", type="sphere", size="0.004", mass=f"{TANK_EMPTY}",
                          contype="0", conaffinity="0", rgba="0.2 0.4 0.9 0.3")
        if fin:
            fx0, fx1, fz0, fz1 = g["fx0"], g["fx1"], g["fz0"], g["fz1"]
            ET.SubElement(hull, "geom", name="t5d_fin", type="box",
                          pos=f"{0.5 * (fx0 + fx1):.6f} 0 {0.5 * (fz0 + fz1):.6f}",
                          size=f"{0.5 * (fx1 - fx0):.6f} 0.002 {0.5 * (fz1 - fz0):.6f}",
                          mass=f"{AREAL * (fx1 - fx0) * (fz1 - fz0):.9g}", rgba="0.9 0.5 0.1 0.8")

    def panel_hook(model, pan):
        import mujoco
        from dytiscidae.physics.fluid import BLUFF, PanelSet
        pan = ph0(model, pan)
        hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seg0_hull")
        sets = [pan]
        if cols:
            dz = h / CN
            zc = z0 + dz * (np.arange(CN) + 0.5)
            v = math.pi * cr * cr * dz
            for (x, y, sg) in cols:
                sets.append(PanelSet(
                    body_id=np.full(CN, hid), pos_local=np.stack([np.full(CN, x), np.full(CN, y), zc], axis=1),
                    span_local=np.tile([0.0, 0.0, 1.0], (CN, 1)), chord_local=np.tile([-1.0, 0.0, 0.0], (CN, 1)),
                    chord=np.full(CN, 2 * cr), dr=np.full(CN, dz), volume=np.full(CN, v), volume_buoyant=np.full(CN, v),
                    half_height=np.full(CN, 0.5 * dz), kind=np.full(CN, BLUFF), aspect_ratio=np.ones(CN),
                    cd_bluff=np.full(CN, 0.9), pitch_axis=np.full(CN, 0.5),
                    ext_local=np.tile([dz, 2 * cr, 2 * cr], (CN, 1))))
        if fin:
            from dytiscidae.physics.fluid import WING
            fx0, fx1, fz0, fz1 = g["fx0"], g["fx1"], g["fz0"], g["fz1"]
            c, b = fx1 - fx0, fz1 - fz0
            dz = b / FN
            zc = fz0 + dz * (np.arange(FN) + 0.5)
            # centroid at mid-chord, as the solver's own wing strips (seg6_wing: pos_local at the strip's centre)
            sets.append(PanelSet(
                body_id=np.full(FN, hid), pos_local=np.stack([np.full(FN, 0.5 * (fx0 + fx1)), np.zeros(FN), zc], axis=1),
                span_local=np.tile([0.0, 0.0, 1.0], (FN, 1)), chord_local=np.tile([-1.0, 0.0, 0.0], (FN, 1)),
                chord=np.full(FN, c), dr=np.full(FN, dz), volume=np.zeros(FN), volume_buoyant=np.zeros(FN),
                half_height=np.full(FN, 0.5 * dz), kind=np.full(FN, WING), aspect_ratio=np.full(FN, b / c),
                cd_bluff=np.zeros(FN), pitch_axis=np.full(FN, 0.25), camber=np.zeros(FN)))
        return PanelSet.concat(sets) if len(sets) > 1 else pan
    return xml_hook, panel_hook


def _build(domain, g, xp, m_cap, m_pt, x_pt, motor="x4", **kw):
    xh, ph = _hooks(g, xp, m_cap, m_pt, x_pt)
    fr = tuple(g["feather_range"]) if g.get("feather_range") else FEATHER
    mc = T.make(domain, b=None, motor=motor, root_cut=ROOT_CUT, sweep_range_deg=SWEEP, feather_range_deg=fr,
                xml_hook=xh, panel_hook=ph, **kw)
    mj, m = mc.mj, mc.m
    mc.pont = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in PONT]
    mc.tank = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in TANK]
    mc.rho_w = float(mc.env.medium.water.rho)
    mc.v_pont = capsule_volume(g["r"], g["L"])
    mc.tank_I0 = mc.env.solver._dry_inertia[mc.tank[0]].copy() / TANK_EMPTY
    cols = _cols(g, xp) if g.get("cr", 0.0) > 0 else []
    mc.ctank = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, CTANK[i]) for i in range(len(cols))]
    mc.col_side = [c[2] for c in cols]
    mc.col_r, mc.col_h = g.get("cr", 0.0), g.get("cz1", 0.0) - g.get("cz0", 0.0)
    mc.v_col = math.pi * mc.col_r ** 2 * mc.col_h
    mc.v_flood = 2 * mc.v_pont + len(cols) * mc.v_col
    mc.fill = 0.0
    mc.fill_p, mc.fill_c = 0.0, 0.0
    return mc


def split_fill(mc, fill):
    """(pontoon fill, column fill) for the whole-volume fill: pontoons first, then the columns."""
    w = min(max(float(fill), 0.0), 1.0) * mc.v_flood
    fp = min(w / (2 * mc.v_pont), 1.0)
    fc = (w - 2 * mc.v_pont * fp) / (len(mc.ctank) * mc.v_col) if mc.ctank and mc.v_col > 0 else 0.0
    return fp, min(max(fc, 0.0), 1.0)


def set_fill(mc, fill: float, eps: float = 0.0) -> None:
    """Flood to ``fill`` of the whole floodable volume (pontoons first, then columns).  ``eps``: a left-right mismatch, every
    left tank at its fill x (1 + eps), every right one x (1 - eps)."""
    s, m = mc.env.solver, mc.m
    fill = min(max(float(fill), 0.0), 1.0)
    fp, fc = split_fill(mc, fill)
    for b, sg in zip(mc.tank, (1.0, -1.0)):
        mt = TANK_EMPTY + min(max(fp * (1.0 + sg * eps), 0.0), 1.0) * mc.rho_w * mc.v_pont
        m.body_mass[b] += mt - s._dry_mass[b]
        s._dry_mass[b] = mt
        s._dry_inertia[b] = mc.tank_I0 * mt
        m.body_inertia[b] = s._dry_inertia[b]
    r, h = mc.col_r, mc.col_h
    for b, sg in zip(mc.ctank, mc.col_side):
        f = min(max(fc * (1.0 + sg * eps), 0.0), 1.0)
        mw = f * mc.rho_w * mc.v_col
        mt = TANK_EMPTY + mw
        hw = f * h
        m.body_mass[b] += mt - s._dry_mass[b]
        s._dry_mass[b] = mt
        m.body_ipos[b] = (0.0, 0.0, 0.5 * hw * mw / mt)
        I = np.array([mw * (3 * r * r + hw * hw) / 12.0, mw * (3 * r * r + hw * hw) / 12.0, 0.5 * mw * r * r]) + 1e-9
        s._dry_inertia[b] = I
        m.body_inertia[b] = I
    if getattr(mc, "_const_scratch", None) is None:
        mc._const_scratch = mc.mj.MjData(m)
    mc.mj.mj_setConst(m, mc._const_scratch)
    mc.mass = float(s._dry_mass.sum())
    mc.fill, mc.fill_p, mc.fill_c = fill, fp, fc


def free_surface_bound(mc):
    """What the slug model leaves out, N m/rad of righting stiffness lost (an upper bound, both to be subtracted from K):
    (a) a free surface inside each column, rho g pi r^4 / 4 each; (b) if the four columns were NOT valved but open to each other,
    rho g sum(A y^2) in roll and rho g sum(A dx^2) in pitch (the whole waterplane term of the inside water, i.e. the columns would
    then add nothing while they hold water at a common level)."""
    rg = mc.rho_w * G
    A = math.pi * mc.col_r ** 2
    n = len(mc.ctank)
    g = mc.t5["geom"]
    dx = [d_ for d_ in g.get("cdx", []) for _ in (0, 1)]
    return dict(inside_each=rg * math.pi * mc.col_r ** 4 / 4, inside_all=n * rg * math.pi * mc.col_r ** 4 / 4,
                open_roll=rg * n * A * g.get("cy", 0.0) ** 2, open_pitch=rg * A * float(np.sum(np.square(dx))))


def calibrate(g=None, force=False) -> dict:
    """T5b's solve with the columns and the fin in the structure and the columns in the floodable volume."""
    g = dict(GEOM_D, **(g or {}))
    key = json.dumps(g, sort_keys=True)
    store = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    if key in store and not force:
        return store[key]
    xp, m_cap, m_pt, x_pt = 0.235, 0.06, 0.06, 0.0
    out = {}
    for it in range(6):
        mc = _build("water", g, xp, m_cap, m_pt, x_pt, pose=dict(x=-30.0, y=0.0, z=-5.0, vx=0.0, vz=0.0))
        dry = mc.env.solver._dry_mass
        rest = np.ones(mc.m.nbody, bool)
        rest[mc.pont + mc.tank + mc.ctank] = False
        rest[0] = False
        B, rb, _, sub = T.static_buoyancy(mc)
        Bp = 2 * mc.rho_w * G * mc.v_pont
        Bc = len(mc.ctank) * mc.rho_w * G * mc.v_col
        cob_x = float(_hull_frame(mc, [rb[0], rb[1], mc.d.xpos[mc.hull][2]])[0])
        B_np = B - Bp
        cob_np_x = (B * cob_x - Bp * xp) / B_np
        _raise(mc)
        m_rest = float(dry[rest].sum())
        x_rest = float(_hull_frame(mc, (dry[rest, None] * mc.d.xipos[rest]).sum(0) / m_rest)[0])
        m_tk = float(dry[mc.ctank].sum())
        m_p = 0.5 * ((B - F_NEUTRAL * (Bp + Bc)) / G - m_rest - m_tk)
        M = m_rest + m_tk + 2 * m_p
        out = dict(geom=g, xp=xp, m_cap=m_cap, m_pt=m_pt, x_pt=x_pt, m_rest=m_rest, x_rest_raised=x_rest, B_all=B, B_np=B_np,
                   B_pontoons=Bp, B_columns=Bc, cob_np_x=cob_np_x, m_dry=float(dry.sum()), v_pont_each=mc.v_pont,
                   v_col_each=mc.v_col, v_flood=mc.v_flood, max_submerged=sub, passes=it + 1, resid_xp=cob_np_x - xp,
                   f_neutral=F_NEUTRAL, resid_mass=float(dry.sum()) - (B - F_NEUTRAL * (Bp + Bc)) / G,
                   m_col_shell_each=AREAL * (2 * math.pi * g.get("cr", 0) * (g.get("cz1", 0) - g.get("cz0", 0)) + 2 * math.pi * g.get("cr", 0) ** 2),
                   m_fin=AREAL * max(g.get("fx1", 0) - g.get("fx0", 0), 0) * max(g.get("fz1", 0) - g.get("fz0", 0), 0))
        if m_p <= 2e-3:
            out["no_spare"] = True
            break
        xp = cob_np_x
        m_cap = g.get("cap", 0.5) * m_p
        m_pt = m_p - m_cap
        x_pm = (M * xp - (m_rest + m_tk) * x_rest) / (2 * m_p)
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


if __name__ == "__main__":
    c = calibrate(force="--force" in sys.argv)
    print(json.dumps({k: v for k, v in c.items() if k != "geom"}))
