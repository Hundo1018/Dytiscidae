"""Read pass W4 against its frozen text (PREREG_W4_wing.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/w4_read.py results/W4_wing.jsonl
"""
import json
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
W = rows[0]["real_weight"]
print("rows", len(rows), "not finite", sum(not r["finite"] for r in rows))
to_ok = lambda r: r["Fz"] / W >= 1.6 and r["crank_eq_peak"] <= 12 and r["aero_power_W"] <= 150
combos = sorted({(str(r["section"]), r["taper"], r["twist"]) for r in rows}, key=lambda c: (c[0] != "None", c))
print("\n%-9s %5s %5s | %-44s | %-44s | take-off gate" % ("section", "taper", "twist", "6 m/s: sustain, cheapest (f, stroke, feather)", "8 m/s"))
for sec, lam, tw in combos:
    s = [r for r in rows if str(r["section"]) == sec and r["taper"] == lam and r["twist"] == tw]
    cells = []
    for V in (6.0, 8.0):
        c = [r for r in s if r["kind"] == "cruise" and r["V"] == V]
        sus = [r for r in c if r["sustains"] and r["crank_eq_peak"] <= 12]
        if sus:
            b = min(sus, key=lambda r: r["aero_power_W"])
            cells.append("%3d of %d, %5.1f W crank %4.1f (%.1f Hz, %2.0f, %2.0f)" % (
                len(sus), len(c), b["aero_power_W"], b["crank_eq_peak"], b["f"], b["Phi_deg"], b["Psi_deg"]))
        else:
            t = [r for r in c if r["trimmed"]]
            b = max(t, key=lambda r: r["Fx"]) if t else None
            cells.append("  0 of %d, nearest %+.2f N at %.1f W" % (len(c), b["Fx"], b["aero_power_W"]) if b else "  0, none trims")
    t = [r for r in s if r["kind"] == "rest"]
    ok = [r for r in t if to_ok(r)]
    best = max(t, key=lambda r: r["Fz"] if r["crank_eq_peak"] <= 12 and r["aero_power_W"] <= 150 else -1) if t else None
    tk = ("%2d of %d; most lift under the caps %.2f weights" % (len(ok), len(t), best["Fz"] / W)) if t else "not run"
    print("%-9s %5.1f %5.0f | %-44s | %-44s | %s" % (sec, lam, tw, cells[0], cells[1], tk))
