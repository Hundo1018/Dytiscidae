"""S0b, machine T5 (DESIGN_S0b_bold.md, addendum 2): T4c with the heights separated.

T5 = T4 (`s0b_machine`) with the inner ROOT_CUT of each main wing absent (T4c), the sweep and feather ranges
of the level hover, no keel, and below the hull:

- two pontoons, capsules parallel to the hull's axis at hull (xp, +-ys, zp), each its own body welded to the
  hull (no joint).  Contact as every machine geom.  Fluid: each is N_SLICE bluff slices along its axis, built
  as the hull's own slices are (`phenotype.build_panels`: centre, extents, displaced volume, half height,
  cd 0.20 = the hull's), so buoyancy and drag come from the solver, nothing is typed;
- four struts, capsules of the tail boom's radius and linear mass: at two stations a post down from the
  hull's underside and a crossbar between the pontoons; one bluff element each (cd 0.9, the solver's strut value) on the hull body;
- ballast = the pontoons' own volume, flooded: a "tank" body at each pontoon's centre (1 g when empty) whose
  mass and inertia are raised to ``fill x rho_water x V_pontoon`` (a uniform fill of the capsule; body_mass,
  the solver's dry mass, `mj_setConst`).  Nothing else changes with the fill: the displaced volume stays.

Masses are closed by the model, not by a material: the pontoons' mass is whatever makes the fully flooded
machine b = 1.00 at fill F_NEUTRAL (weight = the solver's buoyancy of everything that is not pontoon, plus
the unflooded 5% of the pontoons), half of it spread over
the capsule and half at one point low in the pontoon, and that point's station is the one that puts the dry
centre of mass over the pontoons' centre.  The pontoon station xp is the centre of buoyancy of everything
that is not pontoon, so the float (pontoons only) and the flooded machine (everything) are both trimmed.
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

G = 9.80665
ROOT_CUT = 0.11
SWEEP = 75.0
FEATHER = (-150.0, 60.0)
N_SLICE = 8
STRUT_R = 0.004
STRUT_KG_M = 0.00527305 / 0.30            # the tail boom's built mass per length
CAP_SHARE = 0.5                           # of a pontoon's mass spread over its capsule; the rest at the low point
TANK_EMPTY = 1e-3                         # kg, the tank body when empty (taken from the capsule's share)
PONT = ("t5_pont_L", "t5_pont_R")
TANK = ("t5_tank_L", "t5_tank_R")
#: the fill at which the machine is b = 1.00.  Not 1: a machine that is exactly neutral with every tank full
#: cannot leave the surface without thrust; the last 5% of the pontoons' volume is the depth control.
F_NEUTRAL = 0.95
CALIB = HERE / "results" / "S0b_T5_calib.json"
#: the geometry sized in B9 (results/S0b_probes.md); r, L (overall), ys, zp in the hull frame, m
GEOM = dict(r=0.030, L=0.50, ys=0.20, zp=-0.160)
POSE_DIH = math.radians(35.0)


def _capsule_slices(r, L, n=N_SLICE):
    """n slices along a capsule of overall length L: (x centre, dx, volume, mean radius)."""
    xe = np.linspace(-L / 2, L / 2, n + 1)
    hc = L / 2 - r                                     # half length of the cylinder part

    def vol_to(x):                                     # volume from -L/2 to x
        xs = np.linspace(-L / 2, x, 400)
        rr = np.where(np.abs(xs) <= hc, r, np.sqrt(np.maximum(r * r - (np.abs(xs) - hc) ** 2, 0.0)))
        return float(np.trapezoid(np.pi * rr * rr, xs))
    out = []
    for a, b in zip(xe[:-1], xe[1:]):
        v = vol_to(b) - vol_to(a)
        out.append((0.5 * (a + b), b - a, v, math.sqrt(v / (math.pi * (b - a)))))
    return out


def capsule_volume(r, L):
    return math.pi * r * r * (L - 2 * r) + 4.0 / 3.0 * math.pi * r ** 3


def _strut_ends(g, xp):
    """Hull-frame end points of the four struts.  At each of two stations (xp -+ L/4): one post straight down
    from the hull's underside on the centre line to the pontoons' axis height, and one crossbar at that height
    from pontoon to pontoon.  (Struts running diagonally from the hull to each pontoon were built first and
    crossed the wing roots' stroke: -0.007 m at sweep -14 deg, A 44.)"""
    out = []
    for dx in (-0.25 * g["L"], 0.25 * g["L"]):
        out.append((np.array([xp + dx, 0.0, -0.03]), np.array([xp + dx, 0.0, g["zp"]])))
        out.append((np.array([xp + dx, -g["ys"], g["zp"]]), np.array([xp + dx, g["ys"], g["zp"]])))
    return out


def _hooks(g, xp, m_cap, m_pt, x_pt):
    """m_cap: mass on each capsule; m_pt: the point mass in each pontoon at pontoon-frame x ``x_pt``."""
    r, L, ys, zp = g["r"], g["L"], g["ys"], g["zp"]
    hc = L / 2 - r

    def xml_hook(root, hull):
        for a, b in _strut_ends(g, xp):
            ln = float(np.linalg.norm(b - a))
            ET.SubElement(hull, "geom", name=f"t5_strut_{len(hull.findall('geom'))}", type="capsule",
                          fromto=" ".join(f"{v:.6f}" for v in (*a, *b)), size=f"{STRUT_R}",
                          mass=f"{STRUT_KG_M * ln:.9g}", rgba="0.4 0.4 0.45 1")
        for nm, tk, side in ((PONT[0], TANK[0], 1.0), (PONT[1], TANK[1], -1.0)):
            pb = ET.SubElement(hull, "body", name=nm, pos=f"{xp:.6f} {side * ys:.6f} {zp:.6f}")
            ET.SubElement(pb, "geom", name=nm + "_g", type="capsule", fromto=f"{-hc:.6f} 0 0 {hc:.6f} 0 0",
                          size=f"{r}", mass=f"{max(m_cap - TANK_EMPTY, 1e-4):.9g}", rgba="0.2 0.6 0.3 1")
            ET.SubElement(pb, "geom", name=nm + "_k", type="sphere", size="0.006",
                          pos=f"{x_pt:.6f} 0 {-0.6 * r:.6f}", mass=f"{max(m_pt, 1e-4):.9g}",
                          contype="0", conaffinity="0", rgba="0.1 0.1 0.9 1")
            tb = ET.SubElement(hull, "body", name=tk, pos=f"{xp:.6f} {side * ys:.6f} {zp:.6f}")
            ET.SubElement(tb, "geom", name=tk + "_g", type="capsule", fromto=f"{-hc:.6f} 0 0 {hc:.6f} 0 0",
                          size=f"{r * 0.98:.6f}", mass=f"{TANK_EMPTY}", contype="0", conaffinity="0",
                          rgba="0.2 0.4 0.9 0.3")

    def panel_hook(model, pan):
        import mujoco
        from dytiscidae.physics.fluid import BLUFF, PanelSet
        sets = [pan]
        sl = _capsule_slices(r, L)
        for nm in PONT:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            n = len(sl)
            vol = np.array([s[2] for s in sl])
            re = np.array([s[3] for s in sl])
            dx = np.array([s[1] for s in sl])
            sets.append(PanelSet(
                body_id=np.full(n, bid), pos_local=np.array([[s[0], 0.0, 0.0] for s in sl]),
                span_local=np.tile([1.0, 0.0, 0.0], (n, 1)), chord_local=np.tile([0.0, -1.0, 0.0], (n, 1)),
                chord=2 * re, dr=dx, volume=vol, volume_buoyant=vol.copy(), half_height=re.copy(),
                kind=np.full(n, BLUFF), aspect_ratio=np.ones(n), cd_bluff=np.full(n, 0.20),
                pitch_axis=np.full(n, 0.5), ext_local=np.stack([dx, 2 * re, 2 * re], axis=1)))
        hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seg0_hull")
        for a, b in _strut_ends(g, xp):
            ln = float(np.linalg.norm(b - a))
            ax = (b - a) / ln
            ch = np.array([1.0, 0.0, 0.0])                     # the struts lie in a plane of constant x
            v = math.pi * STRUT_R ** 2 * ln
            sets.append(PanelSet(
                body_id=np.array([hid]), pos_local=np.array([0.5 * (a + b)]), span_local=np.array([ax]),
                chord_local=np.array([ch]), chord=np.array([2 * STRUT_R]), dr=np.array([ln]),
                volume=np.array([v]), volume_buoyant=np.array([v]),
                half_height=np.array([max(0.5 * abs(b[2] - a[2]), 5e-3)]), kind=np.array([BLUFF]),
                aspect_ratio=np.array([1.0]), cd_bluff=np.array([0.9]), pitch_axis=np.array([0.5]),
                ext_local=np.array([[ln, 2 * STRUT_R, 2 * STRUT_R]])))
        return PanelSet.concat(sets)
    return xml_hook, panel_hook


def _build(domain, g, xp, m_cap, m_pt, x_pt, motor="x4", **kw):
    xh, ph = _hooks(g, xp, m_cap, m_pt, x_pt)
    mc = T.make(domain, b=None, motor=motor, root_cut=ROOT_CUT, sweep_range_deg=SWEEP, feather_range_deg=FEATHER,
                xml_hook=xh, panel_hook=ph, **kw)
    mj, m = mc.mj, mc.m
    mc.pont = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in PONT]
    mc.tank = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in TANK]
    mc.rho_w = float(mc.env.medium.water.rho)
    mc.v_pont = capsule_volume(g["r"], g["L"])
    mc.tank_I0 = mc.env.solver._dry_inertia[mc.tank[0]].copy() / TANK_EMPTY      # per kg of a uniform fill
    mc.fill = 0.0
    return mc


def _raise(mc, dih=POSE_DIH):
    q = np.zeros(len(mc.aid))
    for tag in ("FL", "FR", "HL", "HR"):
        q[mc.ix[(tag, "stroke")]] = dih
    mc.d.qpos[mc.qad] = q
    mc.mj.mj_forward(mc.m, mc.d)


def _hull_frame(mc, p):
    R = mc.d.xmat[mc.hull].reshape(3, 3)
    return R.T @ (np.asarray(p) - mc.d.xpos[mc.hull])


def calibrate(g=None, force=False) -> dict:
    """Solve xp, the pontoon mass and the low point's station from the built model (three passes)."""
    g = dict(GEOM, **(g or {}))
    key = json.dumps(g, sort_keys=True)
    store = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    if key in store and not force:
        return store[key]
    xp, m_cap, m_pt, x_pt = 0.233, 0.06, 0.06, 0.0
    out = {}
    for it in range(4):
        mc = _build("water", g, xp, m_cap, m_pt, x_pt, pose=dict(x=-30.0, y=0.0, z=-5.0, vx=0.0, vz=0.0))
        dry = mc.env.solver._dry_mass
        rest = np.ones(mc.m.nbody, bool)
        rest[mc.pont + mc.tank] = False
        rest[0] = False
        B, rb, _, sub = T.static_buoyancy(mc)                    # wings flat, deep: everything displaced
        Bp = 2 * mc.rho_w * G * mc.v_pont
        cob_x = float(_hull_frame(mc, [rb[0], rb[1], mc.d.xpos[mc.hull][2]])[0])
        B_np = B - Bp
        cob_np_x = (B * cob_x - Bp * xp) / B_np
        _raise(mc)                                               # the float pose: wings raised
        m_rest = float(dry[rest].sum())
        x_rest = float(_hull_frame(mc, (dry[rest, None] * mc.d.xipos[rest]).sum(0) / m_rest)[0])
        m_p = 0.5 * ((B_np + (1.0 - F_NEUTRAL) * Bp) / G - m_rest)   # each pontoon, tank shell included
        M = m_rest + 2 * m_p
        out = dict(geom=g, xp=xp, m_cap=m_cap, m_pt=m_pt, x_pt=x_pt, m_rest=m_rest, x_rest_raised=x_rest,
                   B_all=B, B_np=B_np, B_pontoons=Bp, cob_np_x=cob_np_x, m_dry=float(dry.sum()),
                   v_pont_each=mc.v_pont, max_submerged=sub, passes=it + 1,
                   resid_xp=cob_np_x - xp, f_neutral=F_NEUTRAL,
                   resid_mass=float(dry.sum()) - (B_np + (1.0 - F_NEUTRAL) * Bp) / G)
        if m_p <= 2e-3:
            out["no_spare"] = True                               # heavier than its non-pontoon buoyancy
            break
        xp = cob_np_x
        m_cap = CAP_SHARE * m_p
        m_pt = m_p - m_cap
        x_pm = (M * xp - m_rest * x_rest) / (2 * m_p)            # where the pontoons' mass centre must be (hull x)
        x_pt = (x_pm - xp) * m_p / m_pt
        x_pt = min(max(x_pt, -(g["L"] / 2 - g["r"])), g["L"] / 2 - g["r"])
    store[key] = out
    CALIB.write_text(json.dumps(store, indent=1))
    return out


def make(domain="air", motor="x4", g=None, fill=0.0, **kw):
    """T5.  ``fill``: the flooded share of the pontoons' volume (0 = float, F_NEUTRAL = b 1.00)."""
    c = calibrate(g)
    mc = _build(domain, c["geom"], c["xp"], c["m_cap"], c["m_pt"], c["x_pt"], motor=motor, **kw)
    mc.t5 = dict(c, motor=motor)
    if fill:
        set_fill(mc, fill)
    return mc


def set_fill(mc, fill: float, eps: float = 0.0) -> None:
    """Flood the pontoons to ``fill`` of their volume: the two tank bodies' mass and inertia (uniform fill).
    ``eps``: a left-right mismatch, the left tank at fill (1 + eps), the right at fill (1 - eps) (a seed for
    modes a mirror-symmetric rollout cannot excite)."""
    s, m = mc.env.solver, mc.m
    fill = min(max(float(fill), 0.0), 1.0)
    for b, sg in zip(mc.tank, (1.0, -1.0)):
        mt = TANK_EMPTY + min(max(fill * (1.0 + sg * eps), 0.0), 1.0) * mc.rho_w * mc.v_pont
        m.body_mass[b] += mt - s._dry_mass[b]
        s._dry_mass[b] = mt
        s._dry_inertia[b] = mc.tank_I0 * mt
        m.body_inertia[b] = s._dry_inertia[b]
    if getattr(mc, "_const_scratch", None) is None:
        mc._const_scratch = mc.mj.MjData(m)
    mc.mj.mj_setConst(m, mc._const_scratch)
    mc.mass = float(s._dry_mass.sum())
    mc.fill = fill


def facts(mc) -> dict:
    """Numbers of the built model in the hull frame (wings as posed)."""
    mj, m, d = mc.mj, mc.m, mc.d
    mj.mj_forward(m, d)
    dry = mc.env.solver._dry_mass
    rep = T.report(mc)
    nm = lambda b: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b)
    return dict(mass=float(dry.sum()), com_hull=_hull_frame(mc, mc.com()).tolist(),
                inertia_hull_diag=rep["inertia_hull_diag"], body_mass={nm(b): float(dry[b]) for b in range(1, m.nbody)},
                fill=mc.fill, v_pont_each=mc.v_pont, panels=int(mc.env.solver.panels.n), nbody=int(m.nbody),
                n_hinges=rep["n_hinges"])


if __name__ == "__main__":
    c = calibrate(force="--force" in sys.argv)
    print(json.dumps(c))
    for fill in (0.0, F_NEUTRAL, 1.0):
        mc = make("water", fill=fill, pose=dict(x=-30.0, y=0.0, z=-5.0, vx=0.0, vz=0.0))
        B, rb, Bd, sub = T.static_buoyancy(mc)
        f = facts(mc)
        mc.env.solver.reset()
        mc.mj.mj_forward(mc.m, mc.d)
        Mm = np.zeros((mc.m.nv, mc.m.nv))
        for i in (0, 2):
            e, col = np.zeros(mc.m.nv), np.zeros(mc.m.nv)
            e[i] = 1.0
            mc.mj.mj_mulM(mc.m, mc.d, col, e)
            Mm[i, i] = col[i]
        print(json.dumps(dict(fill=fill, mass=f["mass"], b=B / (f["mass"] * G), B=B, com=f["com_hull"],
                              cob_x=float(_hull_frame(mc, [rb[0], rb[1], mc.d.xpos[mc.hull][2]])[0]),
                              I=f["inertia_hull_diag"], M00=float(Mm[0, 0]), M22=float(Mm[2, 2]),
                              bodies=f["body_mass"], qacc=mc.bad_qacc)))
