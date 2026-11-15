"""Read pass S1 against its frozen text (PREREG_S1_entrainment.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/s1_read.py results/S1_entrain.jsonl results/L1l_camwater.jsonl results/L1j_flap_r.jsonl
"""
import json
import statistics as st
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
old = [json.loads(x) for x in open(sys.argv[2])]
oldflap = [json.loads(x) for x in open(sys.argv[3])]
key = lambda r: (r["fold"], r["Phi_deg"], r["f"], r["Psi_deg"], r["law"], r["axis"])
arm = {}
for r in rows:
    arm.setdefault(r["arm"], {})[key(r)] = r
old = {key(r): r for r in old}
med = lambda xs: st.median(xs) if xs else float("nan")
moving = [k for k in arm["held"] if k[2] > 0]
print("rows", len(rows), {a: len(v) for a, v in arm.items()})

print("\ncontrol: held as run against L1l_camwater")
for k, r in arm["ctrl"].items():
    o = old[k]
    print("  ", k, "dFx %.1e dP %.1e dcrank %.1e" % (r["Fx"] - o["Fx"], r["aero_power_W"] - o["aero_power_W"],
                                                    r["crank_eq_peak"] - o["crank_eq_peak"]))

print("\nP1 torque-driven, wetting")
fr = arm["free"]
bad = [k for k in moving if not fr[k]["finite"] or fr[k]["track_err_deg"] >= 0.5]
print("  finite and tracking < 0.5 deg: %d of %d; worst tracking %.2f deg" % (
    len(moving) - len(bad), len(moving), max(fr[k]["track_err_deg"] for k in moving)))
for k in bad:
    print("   bad", k, fr[k]["finite"], "%.2f" % fr[k]["track_err_deg"])
for k, r in arm["freedt"].items():
    a = fr[k]
    rel = lambda x, y: abs(x - y) / max(abs(y), 1e-9)
    print("  dt/2", k, "Fx %+.2f/%+.2f (%.1f%%) P %.2f/%.2f (%.1f%%) crank %.2f/%.2f (%.1f%%)" % (
        r["Fx"], a["Fx"], 100 * rel(r["Fx"], a["Fx"]), r["power_W"], a["power_W"],
        100 * rel(r["power_W"], a["power_W"]), r["crank_eq_peak"], a["crank_eq_peak"],
        100 * rel(r["crank_eq_peak"], a["crank_eq_peak"])))

print("\nP2 torque-driven against held (reaction off)")
h = arm["held"]
okF = [k for k in moving if abs(fr[k]["Fx"] - h[k]["Fx"]) <= max(0.05 * abs(h[k]["Fx"]), 0.3)]
okP = [k for k in moving if abs(fr[k]["power_W"] - h[k]["aero_power_W"]) <= 0.05 * abs(h[k]["aero_power_W"])]
both = set(okF) & set(okP)
print("  Fx within max(5%%, 0.3 N): %d; power within 5%%: %d; both: %d of %d" % (len(okF), len(okP), len(both), len(moving)))
print("  median |dFx| %.2f N, median power ratio free/held %.3f" % (
    med([abs(fr[k]["Fx"] - h[k]["Fx"]) for k in moving]),
    med([fr[k]["power_W"] / h[k]["aero_power_W"] for k in moving])))
for Psi in (20.0, 30.0, 50.0, 60.0):
    s = [k for k in moving if k[3] == Psi]
    print("   Psi %.0f: both %d of %d, median power ratio %.3f" % (
        Psi, len(set(s) & both), len(s), med([fr[k]["power_W"] / h[k]["aero_power_W"] for k in s])))

print("\nP3 corrected held crank metric against pass 1l as run")
ratio = [h[k]["crank_eq_peak"] / old[k]["crank_eq_peak"] for k in moving]
print("  lower at %d of %d; median ratio %.2f (min %.2f, max %.2f)" % (
    sum(x < 1 for x in ratio), len(ratio), med(ratio), min(ratio), max(ratio)))
ratio = [h[k]["stroke_torque_peak"] / old[k]["stroke_torque_peak"] for k in moving]
print("  fluid stroke torque: median ratio %.2f" % med(ratio))


def gate(src, ks, law, Psi, drag):
    s = [k for k in moving if k[0] == ks and k[4] == law and k[3] == Psi]
    cap = [k for k in s if all(t[k]["stroke_torque_peak"] <= 15.8 and t[k]["crank_eq_peak"] <= 12 for t in src)]
    ok = [k for k in cap if all(t[k]["Fx"] >= -drag for t in src)]
    return s, cap, ok


for ks in (0.4, 0.6):
    drag = [r["Fx"] for k, r in h.items() if k[0] == ks and k[2] == 0][0]
    print(f"\nwing {ks}: drag at rest {drag:+.2f} N; gate Fx >= {-drag:.2f} N, stroke torque <= 15.8, crank <= 12")
    for law in (None, "plate", "insect"):
        for Psi in (20.0, 30.0, 50.0, 60.0):
            out = []
            for name, src in (("1l as run", [old]), ("held", [h]), ("held+ref", [arm["heldref"]]),
                              ("both", [h, arm["heldref"]]), ("free", [fr])):
                s, cap, ok = gate(src, ks, law, Psi, drag)
                b = max(cap, key=lambda k: min(t[k]["Fx"] for t in src)) if cap else None
                out.append("%s %d" % (name, len(ok)) + (" (best %+.2f N, crank %.1f, f %.2f, Phi %.0f)" % (
                    min(t[b]["Fx"] for t in src), max(t[b]["crank_eq_peak"] for t in src), b[2], b[1]) if b else " (none under caps)"))
            print(f"  {str(law):6} Psi {Psi:.0f}: " + "; ".join(out))

print("\nP4 held: the solver's default reaction against off")
e = arm["heldent"]
loss = [(h[k]["Fx"] - e[k]["Fx"]) / abs(h[k]["Fx"]) for k in moving if abs(h[k]["Fx"]) > 0.3]
print("  Fx lower with it on at %d of %d; median loss %.1f%% of |Fx| (n %d); median power ratio on/off %.3f" % (
    sum(e[k]["Fx"] < h[k]["Fx"] for k in moving), len(moving), 100 * med(loss), len(loss),
    med([e[k]["aero_power_W"] / h[k]["aero_power_W"] for k in moving])))

print("\nP5 crank metric: torque-driven against held with the reference")
hr = arm["heldref"]
near = [k for k in moving if abs(fr[k]["crank_eq_peak"] - hr[k]["crank_eq_peak"]) <= 0.2 * hr[k]["crank_eq_peak"]]
closer = [k for k in moving if abs(fr[k]["crank_eq_peak"] - hr[k]["crank_eq_peak"]) < abs(fr[k]["crank_eq_peak"] - h[k]["crank_eq_peak"])]
print("  within 20%%: %d of %d; nearer the reference than held without it: %d" % (len(near), len(moving), len(closer)))
print("  median crank: free %.2f, held %.2f, held+ref %.2f; median ratio free/held+ref %.2f, held+ref/held %.2f" % (
    med([fr[k]["crank_eq_peak"] for k in moving]), med([h[k]["crank_eq_peak"] for k in moving]),
    med([hr[k]["crank_eq_peak"] for k in moving]),
    med([fr[k]["crank_eq_peak"] / hr[k]["crank_eq_peak"] for k in moving]),
    med([hr[k]["crank_eq_peak"] / h[k]["crank_eq_peak"] for k in moving])))
for f in (0.25, 0.5, 0.75, 1.0):
    s = [k for k in moving if k[2] == f]
    print("   f %.2f: free/held+ref %.2f, held+ref/held %.2f, free/held %.2f" % (
        f, med([fr[k]["crank_eq_peak"] / hr[k]["crank_eq_peak"] for k in s]),
        med([hr[k]["crank_eq_peak"] / h[k]["crank_eq_peak"] for k in s]),
        med([fr[k]["crank_eq_peak"] / h[k]["crank_eq_peak"] for k in s])))

print("\nP6 the reference's mean thrust and power")
up = [k for k in moving if hr[k]["Fx"] > h[k]["Fx"]]
gain = [(hr[k]["Fx"] - h[k]["Fx"]) / abs(h[k]["Fx"]) for k in moving if abs(h[k]["Fx"]) > 0.3]
print("  raises Fx at %d of %d; median gain %.0f%% of |Fx|; median power ratio %.2f" % (
    len(up), len(moving), 100 * med(gain), med([hr[k]["aero_power_W"] / h[k]["aero_power_W"] for k in moving])))
print("  negative mean power: held %d, held+ref %d, free %d" % (
    sum(h[k]["aero_power_W"] < 0 for k in moving), sum(hr[k]["aero_power_W"] < 0 for k in moving),
    sum(fr[k]["power_W"] < 0 for k in moving)))

print("\nP7 the flap's negative-power points")
fkey = lambda r: (r["fold"], r["Phi_deg"], r["f"], r["Psi_deg"])
oldneg = {fkey(r): r for r in oldflap if r["f"] > 0 and r["aero_power_W"] < 0}
fa = {a: {fkey(r): r for r in rows if r["arm"] == a} for a in ("flapctrl", "flapheld", "flapfree")}
for k, o in oldneg.items():
    c, hh, ff = fa["flapctrl"][k], fa["flapheld"][k], fa["flapfree"][k]
    print("  ", k, "as run %+.2f W (control d %.1e); corrected held %+.2f W, Fx %+.2f, torque %.1f; "
          "torque-driven %+.2f W, Fx %+.2f, torque %.1f, finite %s, track %.2f deg" % (
              o["aero_power_W"], c["aero_power_W"] - o["aero_power_W"], hh["aero_power_W"], hh["Fx"],
              hh["stroke_torque_peak"], ff["power_W"], ff["Fx"], ff["stroke_torque_peak"], ff["finite"],
              ff["track_err_deg"]))
