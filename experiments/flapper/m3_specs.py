"""Pass M3: the scored run list of PREREG_M3_land.md section 3, from the frozen parameters.

    python experiments/flapper/m3_specs.py results/M3_frozen.json OUT.json [main|J]
"""
import json
import sys

WALK = [("W1", 0, 0.0, 18, 0), ("W2", 15, 1 / 3, 24, 0), ("W3", -15, 2 / 3, 30, 0),
        ("W4", 90, 0.0, 18, 90), ("W5", 105, 1 / 3, 24, 90), ("W6", 75, 2 / 3, 30, 90)]
TURN = [("R1", 0), ("R2", 90), ("R3", 180)]
STAND = [("S1", 0, 18), ("S2", 45, 22), ("S3", 90, 26), ("S4", 180, 30)]


def walk(arm, fam, gait, tag, jr=None, halves=(0, 1)):
    out = []
    for pk, yaw, ph, x, h in WALK:
        for j in halves:
            hd = h + 180 * j
            s = dict(id=f"{tag}_{pk}_h{hd}", arm=arm, family=fam, kind="walk", pair=pk, yaw0_deg=yaw, phase=ph,
                     x=x, heading_deg=hd, window_s=12, gait=gait)
            if jr:
                s["J_r"] = jr
            out.append(s)
    return out


def turn(arm, fam, gait, tag, jr=None, halves=(90, -90)):
    out = []
    for pk, yaw in TURN:
        for td in halves:
            s = dict(id=f"{tag}_{pk}_t{td:+d}", arm=arm, family=fam, kind="turn", pair=pk, yaw0_deg=yaw, phase=0.0,
                     x=20, turn_deg=td, window_s=14, gait=gait)
            if jr:
                s["J_r"] = jr
            out.append(s)
    return out


def stand(arm, st, tag, jr=None):
    out = []
    for pk, yaw, x in STAND:
        s = dict(id=f"{tag}_{pk}", arm=arm, kind="stand", pair=pk, yaw0_deg=yaw, phase=0.0, x=x, window_s=10, stand=st)
        if jr:
            s["J_r"] = jr
        out.append(s)
    return out


def main(frozen, out, which="main"):
    F = json.load(open(frozen))
    specs = []
    if which == "main":
        specs += stand("S", F["S"], "S") + stand("N0", {}, "N0s")
        specs += walk("N0", "", {}, "N0w", halves=(0,)) + turn("N0", "", {}, "N0r", halves=(90,))
        for fam in ("G1", "G2"):
            g = F[fam]
            specs += walk(fam, fam, g, fam) + turn(fam, fam, g, f"{fam}r")
            specs += walk("U", fam, g, f"U{fam}") + turn("U", fam, g, f"U{fam}r")
            specs += walk("N1", fam, g, f"N1{fam}", halves=(0,)) + turn("N1", fam, g, f"N1{fam}r", halves=(90,))
        if "H" in F:                              # added after tuning (disclosed): the forward hop
            specs += walk("H", "G1", F["H"], "H") + turn("H", "G1", F["H"], "Hr")
        best = F.get("best", "G1")
        for j in (0, 1):
            pk, yaw, ph, x, h = WALK[0]
            specs.append(dict(id=f"F_{pk}_h{h + 180 * j}", arm="F", family=best, kind="walk", pair=pk, yaw0_deg=yaw,
                              phase=ph, x=x, heading_deg=h + 180 * j, frozen_heading_deg=h, window_s=12, gait=F[best]))
    else:
        best = F.get("best", "G1")
        for jr in (5e-6, 1e-6):
            tag = f"J{jr:.0e}"
            specs += walk(best, best, F[best], f"{tag}{best}", jr) + turn(best, best, F[best], f"{tag}{best}r", jr)
            specs += stand("S", F["S"], f"{tag}S", jr)
    json.dump(specs, open(out, "w"))
    print(len(specs), "specs")


if __name__ == "__main__":
    main(*sys.argv[1:])
