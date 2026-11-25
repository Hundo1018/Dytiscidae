"""S0b, machine T5e (DESIGN_S0b_bold.md, Addendum 4): T5b + keel fin + floodable columns where the wings are not.

Two layouts through one builder (geometry dict ``g``):

- **A** (`GEOM_A`): four columns fore and aft of the wing discs.  ``cols`` = [[x, y, r], ...] in the hull frame, each mirrored to
  -y; every column a vertical cylinder from ``cz0`` to ``cz1``.  Each is carried by a slender boom (a capsule of the struts' radius
  and linear mass, `s0b_t5_machine.STRUT_R`, `STRUT_KG_M`) from the nearer end of the pontoon on its side, at the pontoon axis'
  height, to the column's foot.
- **B** (`GEOM_B2`): the four wing shoulders raised by ``sh`` (the wing bodies moved up in the hull frame, each on a post of the
  struts' radius and linear mass from the hull's axis to the shoulder), columns at the pontoon ends as T5d.

Common, as T5d (`s0b_t5d_machine`, whose constants and tank rule are used): the keel fin (Addendum 3: x -0.12..+0.08, z -0.20..-0.06,
y = 0, 6 wing strips); one fill scalar over the whole floodable volume, the pontoons flooding first, then the columns together from
the bottom; a column's water is a slug standing on its foot (centre of mass at half its height, `body_ipos` rewritten), valved;
column shells and fin at the built main wing's mass per area; pontoon mass by neutrality at fill 0.95 of the floodable volume.
Fluid: every column 16 bluff slices (cd 0.9), every boom / post one bluff element (cd 0.9, the solver's strut value), buoyancy and
drag from the solver.  Nothing is typed.
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
from s0b_t5_machine import (F_NEUTRAL, G, PONT, TANK, ROOT_CUT, SWEEP, FEATHER, TANK_EMPTY, STRUT_R, STRUT_KG_M, capsule_volume,  # noqa: E402,F401
                            facts, _raise, _hull_frame)
from s0b_t5b_machine import dry_com, wrench_about_dry_com, righting  # noqa: E402,F401
from s0b_t5d_machine import AREAL, CN, FN  # noqa: E402

CALIB = HERE / "results" / "S0b_T5e_calib.json"
FIN = dict(fx0=-0.12, fx1=0.08, fz0=-0.20, fz1=-0.06)
#: layout A (sized by probe S, results/S0b_probes.md part "T5e"): columns [x, y, r]
GEOM_A = dict(T5B.GEOM_B, **FIN, cols=[[0.84, 0.60, 0.022], [-0.41, 0.60, 0.022]], cz0=-0.23, cz1=0.20, sh=0.0)
#: layout B: shoulders raised, columns at the pontoon ends (T5d's stations, x = xp -+ 0.19)
GEOM_B2 = dict(T5B.GEOM_B, **FIN, cols=[[0.425, 0.20, 0.030], [0.045, 0.20, 0.030]], cz0=-0.20, cz1=0.20, sh=0.40)


def _cols(g):
    """[(x, y, r, side)], side +1 = left."""
    return [(c[0], sg * c[1], c[2], sg) for c in g.get("cols", []) for sg in (1.0, -1.0)]


def _booms(g, xp):
    """[(a, b)] hull-frame end points of the booms: pontoon end (nearer in x) -> column foot, at the pontoon axis' height."""
    out = []
    hl = g["L"] / 2
    for (x, y, r, sg) in _cols(g):
        a = np.array([xp + (hl if x > xp else -hl), sg * g["ys"], g["zp"]])
        b = np.array([x, y, g["zp"]])
        if np.linalg.norm(b - a) > 1e-6 and (abs(x - xp) > hl or abs(abs(y) - g["ys"]) > 1e-6):
            out.append((a, b))
    return out


def _hooks(g, xp, m_cap, m_pt, x_pt):
    xh0, ph0 = T5B._hooks(g, xp, m_cap, m_pt, x_pt)
    z0, z1 = g.get("cz0", 0.0), g.get("cz1", 0.0)
    h = z1 - z0
    cols = _cols(g)
    booms = _booms(g, xp)
    sh = g.get("sh", 0.0)
    fin = g.get("fx1", 0.0) > g.get("fx0", 0.0)
    posts = []

    def xml_hook(root, hull):
        xh0(root, hull)
        if sh > 0.0:
            for wb in hull.findall("body"):
                if wb.get("name") in T.WINGS:
                    p = [float(v) for v in wb.get("pos").split()]
                    posts.append((np.array([p[0], p[1], 0.0]), np.array([p[0], p[1], sh])))
                    p[2] += sh
                    wb.set("pos", " ".join(f"{v:.9g}" for v in p))
            for i, (a, b) in enumerate(posts):
                ET.SubElement(hull, "geom", name=f"t5e_post_{i}", type="capsule", fromto=" ".join(f"{v:.6f}" for v in (*a, *b)),
                              size=f"{STRUT_R}", mass=f"{STRUT_KG_M * sh:.9g}", rgba="0.4 0.4 0.45 1")
        for i, (a, b) in enumerate(booms):
            ET.SubElement(hull, "geom", name=f"t5e_boom_{i}", type="capsule", fromto=" ".join(f"{v:.6f}" for v in (*a, *b)),
                          size=f"{STRUT_R}", mass=f"{STRUT_KG_M * float(np.linalg.norm(b - a)):.9g}", rgba="0.4 0.4 0.45 1")
        for i, (x, y, r, sg) in enumerate(cols):
            ET.SubElement(hull, "geom", name=f"t5e_col_{i}", type="cylinder",
                          fromto=f"{x:.6f} {y:.6f} {z0:.6f} {x:.6f} {y:.6f} {z1:.6f}", size=f"{r}",
                          mass=f"{AREAL * (2 * math.pi * r * h + 2 * math.pi * r * r):.9g}", rgba="0.2 0.7 0.7 0.6")
            tb = ET.SubElement(hull, "body", name=f"t5e_ctank_{i}", pos=f"{x:.6f} {y:.6f} {z0:.6f}")
            ET.SubElement(tb, "geom", name=f"t5e_ctank_{i}_g", type="sphere", size="0.004", mass=f"{TANK_EMPTY}",
                          contype="0", conaffinity="0", rgba="0.2 0.4 0.9 0.3")
        if fin:
            fx0, fx1, fz0, fz1 = g["fx0"], g["fx1"], g["fz0"], g["fz1"]
            ET.SubElement(hull, "geom", name="t5e_fin", type="box", pos=f"{0.5 * (fx0 + fx1):.6f} 0 {0.5 * (fz0 + fz1):.6f}",
                          size=f"{0.5 * (fx1 - fx0):.6f} 0.002 {0.5 * (fz1 - fz0):.6f}",
                          mass=f"{AREAL * (fx1 - fx0) * (fz1 - fz0):.9g}", rgba="0.9 0.5 0.1 0.8")

    def panel_hook(model, pan):
        import mujoco
        from dytiscidae.physics.fluid import BLUFF, WING, PanelSet
        pan = ph0(model, pan)
        hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seg0_hull")
        sets = [pan]
        for (x, y, r, sg) in cols:
            dz = h / CN
            zc = z0 + dz * (np.arange(CN) + 0.5)
            v = math.pi * r * r * dz
            sets.append(PanelSet(
                body_id=np.full(CN, hid), pos_local=np.stack([np.full(CN, x), np.full(CN, y), zc], axis=1),
                span_local=np.tile([0.0, 0.0, 1.0], (CN, 1)), chord_local=np.tile([-1.0, 0.0, 0.0], (CN, 1)),
                chord=np.full(CN, 2 * r), dr=np.full(CN, dz), volume=np.full(CN, v), volume_buoyant=np.full(CN, v),
                half_height=np.full(CN, 0.5 * dz), kind=np.full(CN, BLUFF), aspect_ratio=np.ones(CN),
                cd_bluff=np.full(CN, 0.9), pitch_axis=np.full(CN, 0.5), ext_local=np.tile([dz, 2 * r, 2 * r], (CN, 1))))
        for (a, b) in booms + posts:
            ln = float(np.linalg.norm(b - a))
            ax = (b - a) / ln
            ch = np.cross(ax, [0.0, 0.0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            ch = ch / np.linalg.norm(ch)
            v = math.pi * STRUT_R ** 2 * ln
            sets.append(PanelSet(
                body_id=np.array([hid]), pos_local=np.array([0.5 * (a + b)]), span_local=np.array([ax]), chord_local=np.array([ch]),
                chord=np.array([2 * STRUT_R]), dr=np.array([ln]), volume=np.array([v]), volume_buoyant=np.array([v]),
                half_height=np.array([max(0.5 * abs(b[2] - a[2]), 5e-3)]), kind=np.array([BLUFF]), aspect_ratio=np.array([1.0]),
                cd_bluff=np.array([0.9]), pitch_axis=np.array([0.5]), ext_local=np.array([[ln, 2 * STRUT_R, 2 * STRUT_R]])))
        if fin:
            fx0, fx1, fz0, fz1 = g["fx0"], g["fx1"], g["fz0"], g["fz1"]
            c, b = fx1 - fx0, fz1 - fz0
            dz = b / FN
            zc = fz0 + dz * (np.arange(FN) + 0.5)
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
    cols = _cols(g)
    mc.ctank = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"t5e_ctank_{i}") for i in range(len(cols))]
    mc.col_side = [c[3] for c in cols]
    mc.col_r = [c[2] for c in cols]
    mc.col_h = g.get("cz1", 0.0) - g.get("cz0", 0.0)
    mc.v_col = [math.pi * r * r * mc.col_h for r in mc.col_r]
    mc.v_cols = float(sum(mc.v_col))
    mc.v_flood = 2 * mc.v_pont + mc.v_cols
    mc.fill, mc.fill_p, mc.fill_c = 0.0, 0.0, 0.0
    return mc


def split_fill(mc, fill):
    w = min(max(float(fill), 0.0), 1.0) * mc.v_flood
    fp = min(w / (2 * mc.v_pont), 1.0)
    fc = (w - 2 * mc.v_pont * fp) / mc.v_cols if mc.v_cols > 0 else 0.0
    return fp, min(max(fc, 0.0), 1.0)


def set_fill(mc, fill: float, eps: float = 0.0) -> None:
    """Flood to ``fill`` of the whole floodable volume (pontoons first, then the columns together).  ``eps``: left tanks at
    their fill x (1 + eps), right x (1 - eps)."""
    s, m = mc.env.solver, mc.m
    fill = min(max(float(fill), 0.0), 1.0)
    fp, fc = split_fill(mc, fill)
    for b, sg in zip(mc.tank, (1.0, -1.0)):
        mt = TANK_EMPTY + min(max(fp * (1.0 + sg * eps), 0.0), 1.0) * mc.rho_w * mc.v_pont
        m.body_mass[b] += mt - s._dry_mass[b]
        s._dry_mass[b] = mt
        s._dry_inertia[b] = mc.tank_I0 * mt
        m.body_inertia[b] = s._dry_inertia[b]
    h = mc.col_h
    for b, sg, r, vc in zip(mc.ctank, mc.col_side, mc.col_r, mc.v_col):
        f = min(max(fc * (1.0 + sg * eps), 0.0), 1.0)
        mw = f * mc.rho_w * vc
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
    """N m/rad to subtract from K for what the slug model leaves out: (inside) a free surface in each column; (open_*) the whole
    waterplane term of the inside water if the columns were open to each other instead of valved."""
    rg = mc.rho_w * G
    g = mc.t5["geom"]
    xp = mc.t5["xp"]
    cs = _cols(g)
    return dict(inside_all=float(sum(rg * math.pi * c[2] ** 4 / 4 for c in cs)),
                open_roll=float(sum(rg * math.pi * c[2] ** 2 * c[1] ** 2 for c in cs)),
                open_pitch=float(sum(rg * math.pi * c[2] ** 2 * (c[0] - xp) ** 2 for c in cs)))


def calibrate(g=None, force=False) -> dict:
    g = dict(GEOM_A, **(g or {}))
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
        Bc = mc.rho_w * G * mc.v_cols
        cob_x = float(_hull_frame(mc, [rb[0], rb[1], mc.d.xpos[mc.hull][2]])[0])
        B_np = B - Bp
        cob_np_x = (B * cob_x - Bp * xp) / B_np
        _raise(mc)
        m_rest = float(dry[rest].sum())
        x_rest = float(_hull_frame(mc, (dry[rest, None] * mc.d.xipos[rest]).sum(0) / m_rest)[0])
        m_tk = float(dry[mc.ctank].sum())
        m_p = 0.5 * ((B - F_NEUTRAL * (Bp + Bc)) / G - m_rest - m_tk)
        M = m_rest + m_tk + 2 * m_p
        cs = _cols(g)
        out = dict(geom=g, xp=xp, m_cap=m_cap, m_pt=m_pt, x_pt=x_pt, m_rest=m_rest, x_rest_raised=x_rest, B_all=B, B_np=B_np,
                   B_pontoons=Bp, B_columns=Bc, cob_np_x=cob_np_x, m_dry=float(dry.sum()), v_pont_each=mc.v_pont, v_cols=mc.v_cols,
                   v_flood=mc.v_flood, max_submerged=sub, passes=it + 1, resid_xp=cob_np_x - xp, f_neutral=F_NEUTRAL,
                   resid_mass=float(dry.sum()) - (B - F_NEUTRAL * (Bp + Bc)) / G,
                   m_col_shells=float(sum(AREAL * (2 * math.pi * c[2] * mc.col_h + 2 * math.pi * c[2] ** 2) for c in cs)),
                   m_booms=float(sum(STRUT_KG_M * np.linalg.norm(b - a) for a, b in _booms(g, xp))),
                   m_posts=4 * STRUT_KG_M * g.get("sh", 0.0),
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
    c = calibrate(json.loads(sys.argv[1]) if len(sys.argv) > 1 else None)
    print(json.dumps({k: v for k, v in c.items() if k != "geom"}))
