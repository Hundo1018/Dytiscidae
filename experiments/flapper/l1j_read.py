"""Read pass 1j against its frozen text (PREREG_L1j_flap.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/l1j_read.py results/L1j_flap.jsonl
"""
import json
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
print("rows", len(rows), "not finite", sum(not r["finite"] for r in rows))
ARM = {0: "welded", 1: "flap up", -1: "flap down"}
cap = lambda r: r["stroke_torque_peak"] <= 15.8 and r["crank_torque"] <= 12.0   # noqa: E731

print("\nBlock A: up-down stroke, 0.3 m/s. Best Fx under both caps (pass 1e's table)")
for ks in (0.4, 0.6):
    drag = [r["Fx"] for r in rows if r["block"] == "A" and r["fold"] == ks and r["f"] == 0 and r["sign"] == 0][0]
    line = f" wing {ks}: drag at rest {drag:+.2f} N |"
    best = {}
    for sg in (0, 1, -1):
        s = [r for r in rows if r["block"] == "A" and r["fold"] == ks and r["sign"] == sg and r["f"] > 0
             and r["finite"] and cap(r)]
        b = max(s, key=lambda r: r["Fx"])
        best[sg] = b["Fx"]
        gate = sum(r["Fx"] >= -drag for r in s)
        line += (f" {ARM[sg]} {b['Fx']:+.2f} N (f {b['f']}, Phi {b['Phi_deg']:.0f}, Psi {b['Psi_deg']:.0f},"
                 f" {b['stroke_torque_peak']:.1f} N m, {b['aero_power_W']:.1f} W; gate {gate}) |")
    print(line)
    print(f"   P1: better flap arm / welded = {max(best[1], best[-1]) / best[0]:.2f}")
fa = [r for r in rows if r["block"] == "A" and r["sign"] != 0 and r["f"] > 0]
print(f" P3: flap points with negative mean power: {sum(r['aero_power_W'] < 0 for r in fa)} of {len(fa)}")

print("\nBlock B: rowing stroke (hull vertical). Fx by arm; in brackets the change from that arm at rest")
for ks in (0.4, 0.6):
    for V in (0.0, 0.3):
        rest = {sg: ([r["Fx"] for r in rows if r["block"] == "B" and r["fold"] == ks and r["sign"] == sg
                      and r["f"] == 0] or [0.0])[0] if V > 0 else 0.0 for sg in (0, 1, -1)}
        if V > 0:
            print(f" wing {ks} V {V}: at rest " + ", ".join(f"{ARM[sg]} {rest[sg]:+.2f}" for sg in (0, 1, -1)))
        for Phi in (40.0, 85.0):
            for f in (0.25, 0.5, 0.75, 1.0):
                line = f" wing {ks} V {V} Phi {Phi:.0f} f {f}:"
                for sg in (0, 1, -1):
                    r = [r for r in rows if r["block"] == "B" and r["fold"] == ks and r["sign"] == sg
                         and r["V"] == V and r["f"] == f and r["Phi_deg"] == Phi][0]
                    line += (f" {ARM[sg]} {r['Fx']:+.2f} [{r['Fx'] - rest[sg]:+.2f}] N,"
                             f" {r['stroke_torque_peak']:.1f} N m{'' if cap(r) else '*'},"
                             f" {r['aero_power_W']:.1f} W |")
                print(line)
print(" (* over a torque cap)")
fb = [r for r in rows if r["block"] == "B" and r["sign"] != 0 and r["f"] > 0]
print(f" flap points with negative mean power: {sum(r['aero_power_W'] < 0 for r in fb)} of {len(fb)}")

print("\nBlock C: the flap itself (rowing, at rest in still water, wing 0.4, stroke 85, 0.5 Hz, flap up)")
for r in rows:
    if r["block"] == "C":
        print(f" x_h {r['x_h']} k_f {r['k_f']}: Fx {r['Fx']:+.2f} N, {r['stroke_torque_peak']:.1f} N m,"
              f" {r['aero_power_W']:.2f} W, flap max {r['flap_open_max_deg'][0]:.0f} mean {r['flap_open_mean_deg'][0]:.0f} deg")
