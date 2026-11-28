"""Read pass W8's cruise re-measurement.

    .venv/bin/python experiments/flapper/w8_read.py results/W8_cruise.jsonl
"""
import json
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
pt = lambda r: "%.1f Hz, stroke %.0f, tip %.0f, offset %+.1f" % (r["f"], r["Phi_deg"], r["Psi_deg"], r["psi0_deg"])
for V in (8.0, 6.0):
    print("%g m/s" % V)
    for t in ("normal", "extend", "hold"):
        q = [r for r in rows if r["V"] == V and r["treat"] == t]
        sus = [r for r in q if r["sustains"] and r["crank_eq_peak"] <= 12]
        tr = [r for r in q if r["trimmed"]]
        line = "  %-7s trimmed %2d, sustaining %2d of %d" % (t, len(tr), len(sus), len(q))
        if sus:
            b = min(sus, key=lambda r: r["aero_power_W"])
            m = [r for r in sus if r["Fx"] >= 0.1]
            line += "; cheapest %.2f W (%s), thrust %+.3f N; with +0.1 N: %s" % (
                b["aero_power_W"], pt(b), b["Fx"], ("%.2f W" % min(r["aero_power_W"] for r in m)) if m else "none")
        elif tr:
            b = max(tr, key=lambda r: r["Fx"])
            line += "; nearest: thrust %+.3f N at %.1f W (%s)" % (b["Fx"], b["aero_power_W"], pt(b))
        print(line)
    # the same kinematics across treatments
    key = lambda r: (r["f"], r["Phi_deg"], r["Psi_deg"])
    by = {t: {key(r): r for r in rows if r["V"] == V and r["treat"] == t and r["trimmed"]} for t in ("normal", "extend", "hold")}
    common = sorted(set(by["normal"]) & set(by["extend"]) & set(by["hold"]))
    d = [by["extend"][k]["Fx"] - by["normal"][k]["Fx"] for k in common]
    h = [by["hold"][k]["Fx"] - by["extend"][k]["Fx"] for k in common]
    if d:
        print("  thrust gained, normal -> extend, over %d common trims: %.3f..%.3f N (median %.3f); extend -> hold: %.3f..%.3f (median %.3f)" % (
            len(d), min(d), max(d), sorted(d)[len(d) // 2], min(h), max(h), sorted(h)[len(h) // 2]))
