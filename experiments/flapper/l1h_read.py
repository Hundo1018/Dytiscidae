"""Read pass 1h against its frozen gates (PREREG_L1h_cop.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/l1h_read.py results/L1h_cop.jsonl
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
for h in (0.0, 0.1, 0.2, 0.3, 0.4):
    for z in ("plate", "insect"):
        s = [r for r in rest if r["finite"] and r["axis"] == h and r["law"] == z]
        lift = [r for r in s if r["Fz"] / W >= 1.6]
        best = max(s, key=lambda r: r["Fz"])
        cheap = min(lift, key=tq) if lift else None
        print(f" h {h} law {z}: max lift {best['Fz']/W:.2f} W (f {best['f']}, th {best['theta_deg']}, k {best['k_spring']},"
              f" {tq(best):.1f} N m, {best['aero_power_W']:.0f} W);"
              + (f" cheapest >=1.6: {tq(cheap):.1f} N m, {cheap['aero_power_W']:.0f} W (f {cheap['f']}, k {cheap['k_spring']})"
                 if cheap else " none >= 1.6"))

print("\n(b) cruise: gate sustains, <= 41 W, <= 12 N m")
okc = [r for r in cru if r["sustains"] and r["aero_power_W"] <= 41 and tq(r) <= 12]
print("meets gate:", len(okc), " sustains at all:", sum(r["sustains"] for r in cru),
      " trimmed:", sum(r["trimmed"] for r in cru))
bestF = {}
for h in (0.0, 0.1, 0.2, 0.3, 0.4):
    for z in ("plate", "insect"):
        s = [r for r in cru if r["finite"] and r["trimmed"] and r["axis"] == h and r["law"] == z]
        if not s:
            print(f" h {h} law {z}: none trimmed")
            continue
        b = max(s, key=lambda r: r["Fx"])
        bestF[h] = max(bestF.get(h, -1e9), b["Fx"])
        sus = [r for r in s if r["Fx"] >= 0]
        low = min(sus, key=lambda r: r["aero_power_W"]) if sus else None
        print(f" h {h} law {z}: trimmed {len(s)}, sustain {len(sus)}; best Fx {b['Fx']:+.2f} N"
              f" (V {b['V']}, f {b['f']}, Phi {b['Phi_deg']}, k {b['k_spring']}, {b['aero_power_W']:.0f} W, {tq(b):.1f} N m)"
              + (f"; cheapest sustaining {low['aero_power_W']:.0f} W, {tq(low):.1f} N m" if low else ""))
for z in ("plate", "insect"):
    def most(h):
        s_ = [r["Fz"] / W for r in rest if r["finite"] and r["law"] == z and r["axis"] == h and tq(r) <= 12]
        return max(s_) if s_ else float("nan")
    aft = max(most(h) for h in (0.2, 0.3, 0.4))
    print(f"P2 {z}: most lift at <= 12 N m, h = 0: {most(0.0):.3f} W; best of h 0.2-0.4: {aft:.3f} W"
          f" ({(aft / most(0.0) - 1) * 100:+.0f}%)")
    print(f"P4 {z}: sustaining cruise points at h >= 0.2:",
          sum(r["sustains"] for r in cru if r["law"] == z and r["axis"] >= 0.2))
