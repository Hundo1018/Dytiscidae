"""Read pass 1k against its frozen text (PREREG_L1k_free.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/l1k_read.py results/L1k_free.jsonl
"""
import json
import statistics as st
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
print("rows", len(rows), "not finite", sum(not r["finite"] for r in rows),
      "worst tracking error %.2f deg" % max(r["track_err_deg"] for r in rows))
old = {}
for r in map(json.loads, open(sys.argv[1].replace("L1k_free", "L1i_cam"))):
    old[(r.get("V"), r["theta_deg"], r["f"], r["Phi_deg"], r["Psi_deg"], round(r["psi0_deg"], 6),
         r.get("axis"), r.get("law"))] = r
W = 0.60 * 9.80665


def sel(kind, free, damp=0.0):
    return [r for r in rows if r["kind"] == kind and r["free"] == free and r["joint_damping"] == damp]


def o(r):
    return old[(r["V"], r["theta_deg"], r["f"], r["Phi_deg"], r["Psi_deg"], round(r["psi0_1i"], 6),
                r["axis"], r["law"])]


for kind in ("takeoff", "cruise"):
    h = sel(kind, "")
    dl = [r["Fz"] / o(r)["Fz"] - 1 for r in h]
    dp = [r["power_W"] / o(r)["aero_power_W"] - 1 for r in h]
    dc = [r["crank_eq_peak"] / o(r)["crank_eq_peak"] - 1 for r in h]
    print(f"\n{kind}: {len(h)} points. Held, new rig against pass 1i: lift {min(dl):+.1%}..{max(dl):+.1%},"
          f" power {min(dp):+.1%}..{max(dp):+.1%}, crank metric {min(dc):+.1%}..{max(dc):+.1%}")
    b = sel(kind, "", None)
    print(f" with the built joint damping: power x{st.median(r['power_W'] / q['power_W'] for r, q in zip(b, h)):.2f}"
          f" (median), crank metric x{st.median(r['crank_eq_peak'] / q['crank_eq_peak'] for r, q in zip(b, h)):.2f}")
    pm = [r["pitch_moment_peak"] for r in h]
    print(f" pitching moment the holder takes: peak {min(pm):.1f}..{max(pm):.1f} N m,"
          f" mean {min(r['pitch_moment_mean'] for r in h):+.2f}..{max(r['pitch_moment_mean'] for r in h):+.2f}")

h, x, xp = sel("takeoff", ""), sel("takeoff", "xz"), sel("takeoff", "xzp")
gate = lambda r, lift: lift >= 1.6 and r["crank_eq_peak"] <= 12 and r["power_W"] <= 150   # noqa: E731
print("\nTake-off gate (>= 1.6 weights, crank <= 12 N m, <= 150 W)")
print(" held, new rig:", sum(gate(r, r["lift_over_weight"]) for r in h), "of", len(h))
print(" x and z free, lift over the first beat:", sum(gate(r, r["lift_first_beat"]) for r in x), "of", len(x))
print("  first-beat lift / held lift: %.2f..%.2f" % (min(a["lift_first_beat"] / b["lift_over_weight"] for a, b in zip(x, h)),
                                                    max(a["lift_first_beat"] / b["lift_over_weight"] for a, b in zip(x, h))))
print("  climb in 4 beats: %.2f..%.2f m; crank %.1f..%.1f N m; power %.0f..%.0f W; bob %.0f..%.0f mm" % (
    min(r["dz"] for r in x), max(r["dz"] for r in x), min(r["crank_eq_peak"] for r in x),
    max(r["crank_eq_peak"] for r in x), min(r["power_W"] for r in x), max(r["power_W"] for r in x),
    min(r["bob_mm"] for r in x), max(r["bob_mm"] for r in x)))
print(" pitch free too: climb %.2f..%.2f m; pitch excursion %.0f..%.0f deg; within +-45 deg and climbing >= 1 m: %d of %d;"
      " crank %.1f..%.1f N m (<= 12: %d)" % (
          min(r["dz"] for r in xp), max(r["dz"] for r in xp),
          min(max(abs(r["pitch_min_deg"]), abs(r["pitch_max_deg"])) for r in xp),
          max(max(abs(r["pitch_min_deg"]), abs(r["pitch_max_deg"])) for r in xp),
          sum(max(abs(r["pitch_min_deg"]), abs(r["pitch_max_deg"])) <= 45 and r["dz"] >= 1 for r in xp), len(xp),
          min(r["crank_eq_peak"] for r in xp), max(r["crank_eq_peak"] for r in xp),
          sum(r["crank_eq_peak"] <= 12 for r in xp)))

h, x, xp = sel("cruise", ""), sel("cruise", "xz"), sel("cruise", "xzp")
cg = lambda r: abs(r["lift_over_weight"] - 1) <= 0.03 and r["Fx"] >= 0 and r["power_W"] <= 41 and r["crank_eq_peak"] <= 12  # noqa: E731
print("\nCruise gate (lift = weight, Fx >= 0, <= 41 W, crank <= 12 N m)")
print(" held, new rig:", sum(cg(r) for r in h), "of", len(h))
print(" x and z free, offset re-trimmed:", sum(cg(r) for r in x), "of", len(x),
      "; trimmed", sum(abs(r["lift_over_weight"] - 1) <= 0.03 for r in x))
print("  Fx free minus held: %.2f..%.2f N; cheapest gate point %s" % (
    min(a["Fx"] - b["Fx"] for a, b in zip(x, h)), max(a["Fx"] - b["Fx"] for a, b in zip(x, h)),
    min(("%.1f W, crank %.1f N m (V %s, f %s, Phi %.0f)" % (r["power_W"], r["crank_eq_peak"], r["V"], r["f"], r["Phi_deg"])
         for r in x if cg(r)), default="none")))
for r in sorted(x, key=lambda r: r["power_W"])[:6]:
    print("   V %s f %s Phi %.0f Psi %.0f h %s %s: lift %.3f Fx %+.2f P %.1f crank %.1f psi0 %.1f (1i %.1f) bob %.0f mm" % (
        r["V"], r["f"], r["Phi_deg"], r["Psi_deg"], r["axis"], r["law"], r["lift_over_weight"], r["Fx"],
        r["power_W"], r["crank_eq_peak"], r["psi0_deg"], r["psi0_1i"], r["bob_mm"]))
print(" pitch free too, no tail: pitch excursion in the first beat %.0f..%.0f deg; over 4 beats %.0f..%.0f deg;"
      " beyond 45 deg: %d of %d" % (
          min(r["pitch_first_beat_deg"] for r in xp), max(r["pitch_first_beat_deg"] for r in xp),
          min(max(abs(r["pitch_min_deg"]), abs(r["pitch_max_deg"])) for r in xp),
          max(max(abs(r["pitch_min_deg"]), abs(r["pitch_max_deg"])) for r in xp),
          sum(max(abs(r["pitch_min_deg"]), abs(r["pitch_max_deg"])) > 45 for r in xp), len(xp)))
