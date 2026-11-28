"""Read pass W3's sweep against its frozen text (PREREG_W3_section.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/w3_read.py results/W3_section.jsonl results/L1n_air.jsonl
"""
import json
import statistics as st
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
base = [json.loads(x) for x in open(sys.argv[2])]
W = base[0]["real_weight"]
kr = lambda r: (r["kind"], r["theta_deg"], r["V"], r["f"], r["Phi_deg"], r["Psi_deg"], r["axis"], r["law"])
to_ok = lambda r: r["kind"] == "rest" and r["Fz"] / W >= 1.6 and r["crank_eq_peak"] <= 12 and r["aero_power_W"] <= 150
cr_ok = lambda r: r["kind"] == "cruise" and r["sustains"] and r["aero_power_W"] <= 41 and r["crank_eq_peak"] <= 12
B = {kr(r): r for r in base}
print("rows", len(rows), "not finite", sum(not r["finite"] for r in rows))


def cruise(name, src):
    c = [r for r in src if r["kind"] == "cruise"]
    sus = [r for r in c if r["sustains"]]
    line = "%-12s cruise: trimmed %d, sustain %d, gate %d of %d" % (
        name, sum(r["trimmed"] for r in c), len(sus), sum(cr_ok(r) for r in c), len(c))
    if sus:
        b = min(sus, key=lambda r: r["aero_power_W"])
        line += "; cheapest sustaining %.1f W, crank %.1f N m (V %.0f, f %.1f, Phi %.0f, Psi %.0f, h %.1f, %s, offset %+.1f)" % (
            b["aero_power_W"], b["crank_eq_peak"], b["V"], b["f"], b["Phi_deg"], b["Psi_deg"], b["axis"], b["law"], b["psi0_deg"])
    print(line)
    for V in (6.0, 8.0):
        s = [r for r in sus if r["V"] == V]
        if s:
            print("      at %.0f m/s: %d sustain, cheapest %.1f W" % (V, len(s), min(r["aero_power_W"] for r in s)))


cruise("default law", base)
for sec in ("goe417a", "sd7003", "e387a"):
    s = [r for r in rows if r["section"] == sec]
    cruise(sec, s)
    both = [(r, B[kr(r)]) for r in s if r["kind"] == "cruise" and r["sustains"] and B[kr(r)]["sustains"]]
    if both:
        print("      power where it sustains under both laws, section/default: median %.2f (%.2f..%.2f, n %d)" % (
            st.median(r["aero_power_W"] / b["aero_power_W"] for r, b in both),
            min(r["aero_power_W"] / b["aero_power_W"] for r, b in both),
            max(r["aero_power_W"] / b["aero_power_W"] for r, b in both), len(both)))

t = [r for r in rows if r["kind"] == "rest"]
old = {k for k, r in B.items() if to_ok(r)}
new = {kr(r) for r in t if to_ok(r)}
N = {kr(r): r for r in t}
print("\ntake-off, goe417a: gate points default %d, section %d, in both %d" % (len(old), len(new), len(old & new)))
for q, lab in (("Fz", "lift"), ("crank_eq_peak", "crank"), ("aero_power_W", "power")):
    x = [N[k][q] / B[k][q] for k in old]
    print("   at the default's gate points, section/default %s: %.3f..%.3f (median %.3f)" % (lab, min(x), max(x), st.median(x)))
