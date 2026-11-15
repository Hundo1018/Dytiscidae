"""Read pass 1l against its frozen text (PREREG_L1l_camwater.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/l1l_read.py results/L1l_camwater.jsonl
"""
import json
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
print("rows", len(rows), "not finite", sum(not r["finite"] for r in rows))
for ks in (0.4, 0.6):
    drag = [r["Fx"] for r in rows if r["fold"] == ks and r["f"] == 0][0]
    print(f"\nwing {ks}: drag at rest {drag:+.2f} N. Gate: Fx >= {-drag:.2f} N, fluid stroke torque <= 15.8 N m, crank metric (cam) <= 12 N m")
    for law in (None, "plate", "insect"):
        for Psi in (20.0, 30.0, 50.0, 60.0):
            s = [r for r in rows if r["fold"] == ks and r["law"] == law and r["Psi_deg"] == Psi and r["f"] > 0 and r["finite"]]
            ok = [r for r in s if r["Fx"] >= -drag and r["stroke_torque_peak"] <= 15.8 and r["crank_eq_peak"] <= 12]
            cap = [r for r in s if r["stroke_torque_peak"] <= 15.8 and r["crank_eq_peak"] <= 12]
            b = max(cap, key=lambda r: r["Fx"]) if cap else None
            print(f"  {str(law):6} Psi {Psi:.0f}: gate {len(ok)} of {len(s)}"
                  + (f"; best under caps {b['Fx']:+.2f} N (f {b['f']}, Phi {b['Phi_deg']:.0f}, {b['aero_power_W']:.1f} W,"
                     f" crank {b['crank_eq_peak']:.1f}, stroke only {b['crank_eq_stroke_only_peak']:.1f},"
                     f" feather torque {b['feather_torque_peak']:.2f} N m)" if b else "; nothing under the caps"))
neg = [r for r in rows if r["f"] > 0 and r["aero_power_W"] < 0]
print("\npoints with negative mean power:", len(neg), "of", sum(r["f"] > 0 for r in rows))
