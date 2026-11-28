"""Read pass W6 against its frozen text (PREREG_W6_twist.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/w6_read.py results/W6_twist.jsonl
"""
import json
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
W = rows[0]["real_weight"]
print("rows", len(rows), "not finite", sum(not r["finite"] for r in rows))
to_ok = lambda r: r["Fz"] / W >= 1.6 and r["crank_eq_peak"] <= 12 and r["aero_power_W"] <= 150
pt = lambda b: "%.1f Hz, %2.0f, %2.0f, inc %+.1f" % (b["f"], b["Phi_deg"], b["Psi_deg"], b["psi0_deg"])

print("\ncruise: sustaining with the crank under 12 N m; cheapest by fluid power (f, stroke, feather, incidence)")
for sh in (0.0, 1.0):
    for tw in (0.0, -4.0, -8.0, -12.0, -16.0):
        s = [r for r in rows if r["kind"] == "cruise" and r["share"] == sh and r["twist"] == tw]
        line = "share %.0f twist %+3.0f | trimmed %3d of %d" % (sh, tw, sum(r["trimmed"] for r in s), len(s))
        for V in (6.0, 8.0):
            c = [r for r in s if r["V"] == V]
            sus = [r for r in c if r["sustains"] and r["crank_eq_peak"] <= 12]
            if sus:
                b = min(sus, key=lambda r: r["aero_power_W"])
                line += " | %.0f m/s: %2d, %5.1f W crank %4.1f (%s)" % (V, len(sus), b["aero_power_W"], b["crank_eq_peak"], pt(b))
                m = [r for r in sus if r["Fx"] >= 0.1]
                line += "; +0.1 N: %s" % ("%5.1f W" % min(r["aero_power_W"] for r in m) if m else "none")
            else:
                t = [r for r in c if r["trimmed"]]
                line += " | %.0f m/s:  0%s" % (V, ", nearest %+.2f N at %.1f W" % max((r["Fx"], r["aero_power_W"]) for r in t) if t else "")
        print(line)

print("\ntake-off: gate points; most lift under the caps (pitch, f, feather)")
for sh in (0.0, 0.5, 1.0):
    for tw in (8.0, 4.0, 0.0, -4.0, -8.0):
        t = [r for r in rows if r["kind"] == "rest" and r["share"] == sh and r["twist"] == tw]
        ok = [r for r in t if to_ok(r)]
        cap = [r for r in t if r["crank_eq_peak"] <= 12 and r["aero_power_W"] <= 150]
        b = max(cap, key=lambda r: r["Fz"]) if cap else None
        line = "share %.1f twist %+3.0f | gate %2d of %d" % (sh, tw, len(ok), len(t))
        if b:
            line += " | %.2f weights at %.0f W, crank %.1f (pitch %.0f, %.1f Hz, feather %.0f)" % (
                b["Fz"] / W, b["aero_power_W"], b["crank_eq_peak"], b["theta_deg"], b["f"], b["Psi_deg"])
        if ok:
            c = min(ok, key=lambda r: r["aero_power_W"])
            line += " | cheapest gate point %.0f W" % c["aero_power_W"]
        print(line)
