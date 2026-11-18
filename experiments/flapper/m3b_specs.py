"""Pass M3b: the scored run list (PREREG_M3b section 3) from the frozen sets.

    python experiments/flapper/m3b_specs.py results/M3b_frozen.json OUT.json [T|K|both]
"""
import json
import sys

from m3_specs import STAND, TURN, WALK

MACH = {"T": "ref", "K": "sweep"}


def runs(arm, gait, fam, tag, halves_w=(0, 1), halves_t=(90, -90), window=24.0):
    mach = MACH[fam]
    out = []
    for pk, yaw, ph, x, h in WALK:
        for j in halves_w:
            out.append(dict(id=f"{tag}_{pk}_h{h + 180 * j}", machine=mach, arm=arm, family=fam, kind="walk", pair=pk,
                            yaw0_deg=yaw, phase=ph, x=x, heading_deg=h + 180 * j, window_s=window, gait=gait))
    for pk, yaw in TURN:
        for td in halves_t:
            out.append(dict(id=f"{tag}_{pk}_t{td:+d}", machine=mach, arm=arm, family=fam, kind="turn", pair=pk,
                            yaw0_deg=yaw, phase=0.0, x=20, turn_deg=td, window_s=window, gait=gait))
    return out


def main(frozen, out, which="both"):
    F = json.load(open(frozen))
    specs = []
    for fam in (["T", "K"] if which == "both" else [which]):
        g = F[fam]
        specs += runs(fam, g, fam, fam) + runs("U", g, fam, f"U{fam}")
        specs += runs("N1", g, fam, f"N1{fam}", halves_w=(0,), halves_t=(90,))
        specs += runs("N0", {}, fam, f"N0{fam}", halves_w=(0,), halves_t=(90,))
        if fam == "K":
            for pk, yaw, x in STAND:
                specs.append(dict(id=f"KS_{pk}", machine="sweep", arm="KS", family="K", kind="stand", pair=pk,
                                  yaw0_deg=yaw, phase=0.0, x=x, window_s=10, gait=g))
    json.dump(specs, open(out, "w"))
    print(len(specs), "specs")


if __name__ == "__main__":
    main(*sys.argv[1:])
