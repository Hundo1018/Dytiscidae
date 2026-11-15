"""Read pass 1i against its frozen gates (PREREG_L1i_cam.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/l1i_read.py results/L1i_cam.jsonl results/L1h_cop.jsonl
"""
import json
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
W = rows[0]["real_weight"]
rest = [r for r in rows if r["kind"] == "rest" and r["finite"]]
cru = [r for r in rows if r["kind"] == "cruise" and r["finite"]]
pas = [r for r in rows if r["kind"] == "passive"]
print("rows", len(rows), "rest", len(rest), "cruise", len(cru), "passive", len(pas),
      "not finite", sum(not r["finite"] for r in rows))

print("\n(a) take-off with cam: >= 1.6 weights, crank <= 12 N m, <= 150 W")
for law in ("plate", "insect"):
    for h in (0.0, 0.1, 0.2):
        s = [r for r in rest if r["law"] == law and r["axis"] == h]
        ok = [r for r in s if r["Fz"] / W >= 1.6 and r["crank_eq_peak"] <= 12 and r["aero_power_W"] <= 150]
        lift = [r for r in s if r["Fz"] / W >= 1.6]
        c = min(lift, key=lambda r: r["crank_eq_peak"]) if lift else None
        print(f" {law} h {h}: meet {len(ok)}"
              + (f"; cheapest >=1.6: crank {c['crank_eq_peak']:.1f} (stroke only {c['crank_eq_stroke_only_peak']:.1f},"
                 f" shoulder {c['drive_torque_peak']:.1f}) N m, {c['aero_power_W']:.0f} W, f {c['f']}, Psi {c['Psi_deg']},"
                 f" body {c['theta_deg']}, feather torque {c['feather_torque_peak']:.2f}" if c else ""))
    ok = [r for r in rest if r["law"] == law and r["Fz"] / W >= 1.6 and r["crank_eq_peak"] <= 12
          and r["aero_power_W"] <= 150]
    if ok:
        c = min(ok, key=lambda r: r["crank_eq_peak"])
        print(f" P3 {law}: cheapest gate point, cam over stroke only: "
              f"{(c['crank_eq_peak'] / c['crank_eq_stroke_only_peak'] - 1) * 100:+.0f}%")

print("\n(b) cruise with cam: sustains, <= 41 W, crank <= 12 N m")
for law in ("plate", "insect"):
    for h in (0.0, 0.1, 0.2):
        s = [r for r in cru if r["law"] == law and r["axis"] == h]
        sus = [r for r in s if r["sustains"]]
        ok = [r for r in sus if r["aero_power_W"] <= 41 and r["crank_eq_peak"] <= 12]
        c = min(sus, key=lambda r: r["aero_power_W"]) if sus else None
        print(f" {law} h {h}: trimmed {sum(r['trimmed'] for r in s)}, sustain {len(sus)}, meet {len(ok)}"
              + (f"; cheapest sustaining {c['aero_power_W']:.1f} W, crank {c['crank_eq_peak']:.1f} N m"
                 f" (V {c['V']}, f {c['f']}, Phi {c['Phi_deg']}, Psi {c['Psi_deg']})" if c else ""))

print("\nP4 and control: pass 1h's passive points under the crank metric")
old = {}
if len(sys.argv) > 2:
    for r in map(json.loads, open(sys.argv[2])):
        if r["kind"] == "rest":
            old[(r["law"], r["axis"], r["theta_deg"], r["f"], r["k_spring"])] = r
for r in pas:
    o = old.get((r["law"], r["axis"], r["theta_deg"], r["f"], r["k_spring"])) if r["V"] == 0 else None
    tag = (f" | 1h: {o['Fz'] / W:.2f} W, {o['drive_torque_peak']:.1f} N m" if o else "")
    print(f" {r['law']} h {r['axis']} V {r['V']} f {r['f']} Phi {r['Phi_deg']}: {r['Fz'] / W:.2f} W, Fx {r['Fx']:+.2f},"
          f" {r['aero_power_W']:.0f} W, shoulder {r['drive_torque_peak']:.1f}, crank {r['crank_eq_peak']:.1f} N m{tag}")
