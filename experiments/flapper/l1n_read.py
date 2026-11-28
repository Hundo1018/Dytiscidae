"""Read pass 1n against its frozen text (PREREG_L1n_refit.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/l1n_read.py results
"""
import json
import statistics as st
import sys
from pathlib import Path

D = Path(sys.argv[1])
load = lambda n: [json.loads(x) for x in open(D / n)]
air, old = load("L1n_air.jsonl"), [r for r in load("L1i_cam.jsonl") if r["kind"] != "passive"]
W = old[0]["real_weight"]
kr = lambda r: (r["kind"], r["theta_deg"], r["V"], r["f"], r["Phi_deg"], r["Psi_deg"], r["axis"], r["law"])
to_ok = lambda r: r["kind"] == "rest" and r["Fz"] / W >= 1.6 and r["crank_eq_peak"] <= 12 and r["aero_power_W"] <= 150
cr_ok = lambda r: r["kind"] == "cruise" and r["sustains"] and r["aero_power_W"] <= 41 and r["crank_eq_peak"] <= 12
N, O = {kr(r): r for r in air}, {kr(r): r for r in old}
print("air rows", len(air), "not finite", sum(not r["finite"] for r in air), "matched", len(set(N) & set(O)))
for name, ok in (("take-off", to_ok), ("cruise", cr_ok)):
    a, b = {k for k, r in O.items() if ok(r)}, {k for k, r in N.items() if ok(r)}
    print(f"\n{name}: gate points centre-line {len(a)}, seated {len(b)}, in both {len(a & b)}")
    for law in ("plate", "insect"):
        print("   %s: %d -> %d" % (law, sum(k[7] == law for k in a), sum(k[7] == law for k in b)))
    if name == "take-off":
        for q, lab in (("Fz", "lift"), ("crank_eq_peak", "crank"), ("aero_power_W", "power")):
            x = [N[k][q] / O[k][q] for k in a]
            print("   at the centre-line gate points, seated/centre-line %s: %.3f..%.3f (median %.3f)" % (
                lab, min(x), max(x), st.median(x)))
        best = [r for r in N.values() if to_ok(r)]
        if best:
            print("   seated gate points: lift %.2f..%.2f weights, crank %.1f..%.1f N m, %.0f..%.0f W" % (
                min(r["Fz"] / W for r in best), max(r["Fz"] / W for r in best),
                min(r["crank_eq_peak"] for r in best), max(r["crank_eq_peak"] for r in best),
                min(r["aero_power_W"] for r in best), max(r["aero_power_W"] for r in best)))
    else:
        for src, lab in ((O, "centre-line"), (N, "seated")):
            sus = [r for r in src.values() if r["kind"] == "cruise" and r["sustains"]]
            c = min(sus, key=lambda r: r["aero_power_W"])
            print("   %s: %d sustain; cheapest %.1f W, crank %.1f N m (V %.0f, f %.1f, Phi %.0f, Psi %.0f, h %.1f, %s)" % (
                lab, len(sus), c["aero_power_W"], c["crank_eq_peak"], c["V"], c["f"], c["Phi_deg"], c["Psi_deg"],
                c["axis"], c["law"]))
        both = [k for k in a & b]
        x = [N[k]["aero_power_W"] / O[k]["aero_power_W"] for k in both]
        if x:
            print("   power at points that meet the gate both ways, seated/centre-line: %.3f..%.3f (median %.3f)" % (
                min(x), max(x), st.median(x)))

fr = load("L1n_free.jsonl")
print("\ntorque-driven rig, seated: rows", len(fr), "not finite", sum(not r["finite"] for r in fr))
t = [r for r in fr if r["kind"] == "takeoff"]
print("  take-off held: %d of %d meet the gate (lift >= 1.6, <= 150 W, crank <= 12)" % (
    sum(r["lift_over_weight"] >= 1.6 and r["power_W"] <= 150 and r["crank_eq_peak"] <= 12 for r in t), len(t)))
for free in ("", "xz"):
    c = [r for r in fr if r["kind"] == "cruise" and r["free"] == free]
    ok = [r for r in c if abs(r["lift_over_weight"] - 1) <= 0.03 and r["Fx"] >= 0 and r["power_W"] <= 41
          and r["crank_eq_peak"] <= 12]
    print("  cruise %-4s: %d of %d meet the gate" % (free or "held", len(ok), len(c))
          + ("; cheapest %.1f W at crank %.1f" % (min(r["power_W"] for r in ok),
                                                 min(ok, key=lambda r: r["power_W"])["crank_eq_peak"]) if ok else ""))

wn, wo = load("L1n_water.jsonl"), [r for r in load("S1_entrain.jsonl") if r["arm"] in ("held", "heldref", "free")]
kw = lambda r: (r["fold"], r["Phi_deg"], r["f"], r["Psi_deg"], r["law"], r["axis"])


def cells(rows):
    arm = {}
    for r in rows:
        arm.setdefault(r["arm"], {})[kw(r)] = r
    h, hr = arm["held"], arm["heldref"]
    out = {}
    for ks in (0.4, 0.6):
        drag = [r["Fx"] for k, r in h.items() if k[0] == ks and k[2] == 0][0]
        for law in (None, "plate", "insect"):
            for Psi in (20.0, 30.0, 50.0, 60.0):
                s = [k for k in h if k[2] > 0 and k[0] == ks and k[4] == law and k[3] == Psi]
                cap = [k for k in s if all(t_[k]["stroke_torque_peak"] <= 15.8 and t_[k]["crank_eq_peak"] <= 12
                                           for t_ in (h, hr))]
                ok = [k for k in cap if all(t_[k]["Fx"] >= -drag for t_ in (h, hr))]
                b = max(ok, key=lambda k: min(h[k]["Fx"], hr[k]["Fx"])) if ok else None
                out[(ks, law, Psi)] = (len(ok), min(h[b]["Fx"], hr[b]["Fx"]) if b else None,
                                       max(h[b]["crank_eq_peak"], hr[b]["crank_eq_peak"]) if b else None)
    return out, arm


co, _ = cells(wo)
cn, an = cells(wn)
print("\nwater, gate met without and with the reference: cell (wing, law, feather): centre-line -> seated")
same = 0
for k in co:
    same += (co[k][0] > 0) == (cn[k][0] > 0)
    f = lambda c: "%d" % c[0] + (" (%.1f N, crank %.1f)" % (c[1], c[2]) if c[0] else "")
    print("  ", k, f(co[k]), "->", f(cn[k]))
print("  cells with the same verdict: %d of %d" % (same, len(co)))
x = [cn[k][1] / co[k][1] for k in co if co[k][0] and cn[k][0]]
print("  best force where both pass, seated/centre-line: %.2f..%.2f (median %.2f)" % (min(x), max(x), st.median(x)))
mov = [k for k in an["free"]]
print("  torque-driven, wetting: finite and tracking < 0.5 deg at %d of %d" % (
    sum(an["free"][k]["finite"] and an["free"][k]["track_err_deg"] < 0.5 for k in mov), len(mov)))
