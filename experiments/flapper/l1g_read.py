"""Read pass 1g against its frozen gates (PREREG_L1g_axis.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/l1g_read.py results/L1g_axis.jsonl
"""
import json
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
rest = [r for r in rows if r["kind"] == "rest"]
cru = [r for r in rows if r["kind"] == "cruise"]
print("rows", len(rows), "rest", len(rest), "cruise", len(cru),
      "not finite", sum(not r["finite"] for r in rows))
W = rows[0]["real_weight"]


def tq(r):
    return r["drive_torque_peak"]


print("\n(a) take-off: gate >= 1.6 weights, <= 12 N m, <= 150 W")
ok = [r for r in rest if r["finite"] and r["Fz"] / W >= 1.6 and tq(r) <= 12 and r["aero_power_W"] <= 150]
print("meets gate:", len(ok))
for h in (0.0, 0.1, 0.2, 0.3):
    for z in (0.3, 1.0):
        s = [r for r in rest if r["finite"] and r["axis"] == h and r["zeta"] == z]
        lift = [r for r in s if r["Fz"] / W >= 1.6]
        best = max(s, key=lambda r: r["Fz"])
        cheap = min(lift, key=tq) if lift else None
        print(f" h {h} zeta {z}: max lift {best['Fz']/W:.2f} W (f {best['f']}, th {best['theta_deg']}, k {best['k_spring']},"
              f" {tq(best):.1f} N m, {best['aero_power_W']:.0f} W);"
              + (f" cheapest >=1.6: {tq(cheap):.1f} N m, {cheap['aero_power_W']:.0f} W (f {cheap['f']}, k {cheap['k_spring']})"
                 if cheap else " none >= 1.6"))

print("\n(b) cruise: gate sustains, <= 41 W, <= 12 N m")
okc = [r for r in cru if r["sustains"] and r["aero_power_W"] <= 41 and tq(r) <= 12]
print("meets gate:", len(okc), " sustains at all:", sum(r["sustains"] for r in cru),
      " trimmed:", sum(r["trimmed"] for r in cru))
bestF = {}
for h in (0.0, 0.1, 0.2, 0.3):
    for z in (0.3, 1.0):
        s = [r for r in cru if r["finite"] and r["trimmed"] and r["axis"] == h and r["zeta"] == z]
        if not s:
            print(f" h {h} zeta {z}: none trimmed")
            continue
        b = max(s, key=lambda r: r["Fx"])
        bestF[h] = max(bestF.get(h, -1e9), b["Fx"])
        sus = [r for r in s if r["Fx"] >= 0]
        low = min(sus, key=lambda r: r["aero_power_W"]) if sus else None
        print(f" h {h} zeta {z}: trimmed {len(s)}, sustain {len(sus)}; best Fx {b['Fx']:+.2f} N"
              f" (V {b['V']}, f {b['f']}, Phi {b['Phi_deg']}, k {b['k_spring']}, {b['aero_power_W']:.0f} W, {tq(b):.1f} N m)"
              + (f"; cheapest sustaining {low['aero_power_W']:.0f} W, {tq(low):.1f} N m" if low else ""))
print("\nP3: best trimmed Fx at h > 0 minus at h = 0:",
      round(max(v for h, v in bestF.items() if h > 0) - bestF.get(0.0, float("nan")), 3))
print("P4: sustaining cruise points at h = 0:", sum(r["sustains"] for r in cru if r["axis"] == 0.0))
