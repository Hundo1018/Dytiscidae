"""Read pass W7 (PREREG_W7_twistmech.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/w7_read.py results/W7_twistmech.jsonl
"""
import json
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
W = 0.60 * 9.80665
print("cruise, 8 m/s (trimmed cam offset; power = stroke + feather + twist drive)")
pts = []
for r in rows:
    if r["kind"] == "cruise" and r["point"] not in pts:
        pts.append(r["point"])
for pt in pts:
    print(" point f %.1f stroke %.0f tip %.0f" % tuple(pt))
    for r in (r for r in rows if r["kind"] == "cruise" and r["point"] == pt):
        tw = ""
        if "twist_mode" in r:
            tw = " | twist P %+.2f (pos %.2f) W, peak %.2f N m | tip mean %+.1f swing %.1f lag %+.0f deg | hinges %s" % (
                r["twist_power_W"], r["twist_power_pos_W"], r["twist_torque_peak"], r["tip_twist_mean_deg"],
                r["tip_twist_amp_deg"], r["tip_twist_lag_deg"], " ".join("%.1f" % x for x in r["hinge_swing_deg"]))
        print("   %-20s psi0 %+6.2f L/W %.3f Fx %+.3f %s P %5.2f W crank %4.2f err %.2f%s" % (
            r["arm"], r["psi0_deg"], r["lift_over_weight"], r["Fx"], "sustains" if r["sustains"] else
            ("trimmed " if r["trimmed"] else "NO TRIM "), r["power_W"], r["crank_eq_peak"], r["track_err_deg"], tw))
print("\ntake-off (body pitch 90, stroke 85, root feather 50)")
for f in sorted({r["point"][0] for r in rows if r["kind"] == "takeoff"}, reverse=True):
    print(" %.1f Hz" % f)
    k1 = next(r for r in rows if r["kind"] == "takeoff" and r["point"][0] == f and r["arm"].startswith("K1"))
    for r in (r for r in rows if r["kind"] == "takeoff" and r["point"][0] == f):
        tw = ""
        if "twist_mode" in r:
            tw = " | twist peak %.2f N m, P pos %.2f W | tip mean %+.1f swing %.1f (%+.1f..%+.1f) | hinges %s" % (
                r["twist_torque_peak"], r["twist_power_pos_W"], r["tip_twist_mean_deg"], r["tip_twist_amp_deg"],
                r["tip_twist_min_deg"], r["tip_twist_max_deg"], " ".join("%.1f" % x for x in r["hinge_swing_deg"]))
        gate = r["lift_over_weight"] >= 1.6 and r["crank_eq_peak"] <= 12 and r["power_W"] <= 150
        print("   %-20s L/W %.3f (%+5.1f%%) P %6.1f W crank %5.2f %s%s" % (
            r["arm"], r["lift_over_weight"], 100 * (r["lift_over_weight"] / k1["lift_over_weight"] - 1),
            r["power_W"], r["crank_eq_peak"], "gate" if gate else "    ", tw))
