"""Pass S0: the row lists (results/S0_plan.md), one function per scenario.

    python experiments/flapper/s0_specs.py RUNG SCENARIO[,SCENARIO] OUT.json [scored|tuning]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CRUISE = dict(scored=dict(Q1=(3.0, 30.0, 8.2), Q2=(-4.0, -150.0, 7.8)),
              tuning=dict(QT1=(2.0, 20.0, 8.0), QT2=(-2.0, -110.0, 8.1)))
TAKEOFF = dict(scored=[(19.5, 30.0), (26.5, 210.0)], tuning=[(23.5, 330.0), (20.5, 120.0)])
WALK = dict(scored=[("SW1", 5, 0.25, 20, 0), ("SW2", 82, 0.75, 26, 90)], tuning=[("TW1", -8, 0.1, 23, 0)])
TURN = dict(scored=[("SR1", 40), ("SR2", 130)], tuning=[("TR1", 60)])
STAND = dict(scored=[("SS1", 20, 19), ("SS2", 200, 27)], tuning=[])
ENTRY = dict(scored=dict(ES1=(3.0, 2.0, 8.1, 0.1), ES2=(-3.0, -4.0, 7.9, -0.2)),
             tuning=dict(ET1=(5.0, -2.0, 8.0, 0.25), ET2=(-5.0, 4.0, 8.2, -0.1)))
XE = dict(water=(-45.0, -33.0), land=(16.0, 28.0))
SEEDS = dict(scored=[20, 21], tuning=[200, 201])
MODES = ("wetting", True)


def frozen_E(medium):
    return json.loads((HERE / "results" / "M4_frozen.json").read_text())[medium]["E"]


def k_gait():
    return json.loads((HERE / "results" / "M3b_frozen.json").read_text())["K"]


def m2b(name):
    import m2b_run
    return dict(m2b_run.FROZEN[name])


def mode_tag(m):
    return "wet" if m == "wetting" else "def"


# ------------------------------------------------------------------ air
def air(rung, cfg, which="scored", point="P35", tag="air"):
    out = []
    for name, st in CRUISE[which].items():
        p0 = st[1]
        for arm, half, cmd in (("L", "H", p0), ("L", "Tp", p0 + 90.0), ("L", "Tm", p0 - 90.0), ("N", "all", p0)):
            rid = f"{rung}_{tag}_{name}_{arm}_{half}"
            out.append(dict(id=rid, scenario=tag, leg="cruise", rung=rung, cfg=cfg, kind="cruise", which=which,
                            half=dict(kind="cruise", arm=arm, point=point, start=st, start_name=name, cmd=cmd, id=rid,
                                      half=half)))
    return out


# ------------------------------------------------------------------ land
def land(rung, cfg, which="scored", gait=None, tag="land"):
    g = gait or k_gait()
    c = dict(cfg, sweep=True, sweep_free=True)
    out = []

    def row(rid, leg, m3):
        out.append(dict(id=rid, scenario=tag, leg=leg, rung=rung, cfg=c, kind="land", which=which,
                        m3b=dict(m3, id=rid, machine="sweep", family="K", gait=g if m3["arm"] != "N0" else {})))
    for pk, yaw, ph, x, h in WALK[which]:
        for arm in ("K", "U"):
            for j in (0, 1):
                row(f"{rung}_{tag}_{pk}_{arm}_h{h + 180 * j}", "walk",
                    dict(arm=arm, kind="walk", pair=pk, yaw0_deg=yaw, phase=ph, x=x, heading_deg=h + 180 * j, window_s=14.0))
        row(f"{rung}_{tag}_{pk}_N0_h{h}", "walk",
            dict(arm="N0", kind="walk", pair=pk, yaw0_deg=yaw, phase=ph, x=x, heading_deg=h, window_s=14.0))
    for pk, yaw in TURN[which]:
        for td in (90, -90):
            row(f"{rung}_{tag}_{pk}_K_t{td:+d}", "turn",
                dict(arm="K", kind="turn", pair=pk, yaw0_deg=yaw, phase=0.0, x=20, turn_deg=td, window_s=14.0))
        row(f"{rung}_{tag}_{pk}_N0_t+90", "turn",
            dict(arm="N0", kind="turn", pair=pk, yaw0_deg=yaw, phase=0.0, x=20, turn_deg=90, window_s=14.0))
    for i, (pk, yaw, x) in enumerate(STAND[which]):
        row(f"{rung}_{tag}_{pk}_KS", "stand", dict(arm="KS", kind="stand", pair=pk, yaw0_deg=yaw, phase=0.0, x=x, window_s=10))
        if i == 0:
            row(f"{rung}_{tag}_{pk}_N0s", "stand", dict(arm="N0", kind="stand", pair=pk, yaw0_deg=yaw, phase=0.0, x=x,
                                                       window_s=10))
    return out


# ------------------------------------------------------------------ water
def water(rung, cfg, which="scored", laws=None, tag="water", legs=("D", "GW", "U", "GS")):
    L = laws or dict(WT=m2b("WT"), ST=m2b("ST"))
    still = dict(law="still")
    out = []
    for s in SEEDS[which]:
        for m in MODES:
            mt = mode_tag(m)

            def row(rid, leg, start, gains, law, half, kd, arm):
                out.append(dict(id=rid, scenario=tag, leg=leg, rung=rung, cfg=cfg, kind="water", which=which, arm=arm,
                                seed=s, mode=m, half=half,
                                water=dict(seed=s, mode=m, half=half, start=start, gains=gains, law=law,
                                           cmd_from_draw=kd, T=40.0)))
            if "D" in legs:
                row(f"{rung}_{tag}_D_s{s}_{mt}", "dive", "float", "here", L["WT"], "A", "D", "WT")
                row(f"{rung}_{tag}_D0_s{s}_{mt}", "dive", "float", "here", still, "A", "D", "still")
            if "GW" in legs:
                for h in ("A", "B"):
                    row(f"{rung}_{tag}_GW_s{s}_{h}_{mt}", "swim", "depth", "here", L["WT"], h, "GW", "WT")
                row(f"{rung}_{tag}_W0_s{s}_{mt}", "swim", "depth", "here", still, "A", "GW", "still")
            if "U" in legs:
                row(f"{rung}_{tag}_U_s{s}_{mt}", "surface", "depth", "here", L["WT"], "A", "U", "WT")
                row(f"{rung}_{tag}_U0_s{s}_{mt}", "surface", "depth", "here", still, "A", "U", "still")
            if "GS" in legs:
                for h in ("A", "B"):
                    row(f"{rung}_{tag}_GS_s{s}_{h}_{mt}", "surface_turn", "float", "here", L["ST"], h, "GS", "ST")
                row(f"{rung}_{tag}_S0_s{s}_{mt}", "surface_turn", "float", "here", still, "A", "GS", "still")
    return out


# ------------------------------------------------------------------ land<->air
def land_air(rung, cfg, which="scored", landing="E", legs=("takeoff", "cruise", "landing"), tag="landair"):
    c = dict(cfg, sweep=True)
    out = []
    if "takeoff" in legs:
        for x, yaw in TAKEOFF[which]:
            for arm in ("V+", "N0"):
                rid = f"{rung}_{tag}_TO_{arm}_{x:g}_{yaw:g}"
                out.append(dict(id=rid, scenario=tag, leg="takeoff", rung=rung, cfg=c, kind="takeoff", which=which,
                                arm=arm, entry=[x, yaw]))
    if "cruise" in legs:
        for r in air(rung, c, which, tag=tag):
            if r["half"]["half"] in ("H", "all"):
                out.append(r)
    if "landing" in legs:
        out += entries(rung, c, "land", which, landing, again="takeoff", tag=tag, modes=("wetting",))
    return out


def entries(rung, cfg, medium, which, law, again, tag, modes):
    E = frozen_E(medium)
    near, far = XE[medium]
    null = "NW" if medium == "water" else "NL"
    out = []
    for m in modes:
        mt = mode_tag(m)
        for name, st in ENTRY[which].items():
            for arm, half, cmd in (("E", "near", near), ("E", "far", far), (null, "all", near)):
                rid = f"{rung}_{tag}_{medium}_{name}_{arm if arm != 'E' else law}_{half}_{mt}"
                h = dict(kind=medium, arm=arm if arm != "E" else ("A4" if law == "A4" else "E"), point="P35",
                         start=st, start_name=name, cmd=cmd, E=E, ent=m, id=rid, half=half)
                if law == "A4" and arm == "E":
                    h["law"] = "A4"
                out.append(dict(id=rid, scenario=tag, leg="landing" if medium == "land" else "entry", rung=rung,
                                cfg=cfg, kind="entry", which=which, half=h, again=again, mode=m))
    return out


# ------------------------------------------------------------------ water<->air
def water_air(rung, cfg, which="scored", landing="E", legs=("exit", "entry"), tag="waterair", xs=None):
    out = []
    if "exit" in legs:
        for s in SEEDS[which]:
            for m in MODES:
                mt = mode_tag(m)
                for arm, law in (("XS", xs or m2b("XS")), ("X0f", dict(law="still"))):
                    rid = f"{rung}_{tag}_{arm}_s{s}_{mt}"
                    out.append(dict(id=rid, scenario=tag, leg="exit", rung=rung, cfg=cfg, kind="water", which=which,
                                    arm=arm, seed=s, mode=m, half="X",
                                    water=dict(seed=s, mode=m, half="X", start="float", gains="air", law=law,
                                               cmd_from_draw=None, T=40.0)))
    if "entry" in legs:
        out += entries(rung, cfg, "water", which, landing, again="exit", tag=tag, modes=MODES)
    return out


SCEN = dict(air=air, land=land, water=water, landair=land_air, waterair=water_air)


def main(rung, scen, out, which="scored"):
    cfg = {}
    specs = []
    for s in scen.split(","):
        specs += SCEN[s](rung, cfg, which)
    json.dump(specs, open(out, "w"), indent=0)
    print(len(specs), "specs")


if __name__ == "__main__":
    main(*sys.argv[1:])
