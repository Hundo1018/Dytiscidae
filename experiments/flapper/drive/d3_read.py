"""Pass D3: read the grid and the trimmed rows against PREREG_D3_tip_twist.md.

    .venv/bin/python experiments/flapper/drive/d3_read.py results/D3_grid.jsonl [results/D3_trim.jsonl]
"""
import json
import sys

ORDER = ("base", "B", "T0", "TX", "TP", "TC")


def line(r):
    dv = r.get("drive") or {}
    tw = dv.get("twist") or {}
    f = lambda v, n=2: "-" if v is None else ("%.*f" % (n, v))
    force = "cp %s" % f(tw.get("coupler_force_peak_N", dv.get("loop_force_peak_N")), 0)
    if "rod_force_peak_N" in tw:
        force += " rod %s" % f(tw["rod_force_peak_N"], 1)
    if "cable_force_peak_N" in tw:
        force += " cable %s taut %s" % (f(tw["cable_force_peak_N"], 1), f(tw["cable_taut_share"]))
    gap = "cp %s" % f(tw.get("coupler_gap_max_mm", dv.get("loop_gap_max_mm")), 3)
    if "rod_gap_max_mm" in tw:
        gap += " rod %s" % f(tw["rod_gap_max_mm"], 3)
    tq = r.get("twist_torque_peak") if r.get("twist_mode") == "servo" else tw.get("hinge_torque_peak")
    return ("%-28s psi0 %6.2f | tip %5s mean %6s lead %7s | L/W %6.3f Fx %+7.3f | mech %6.2f elec %6s W | %s N | gap %s mm "
            "| hinge %s N m, feather hold %s | beat %s lim %s kin %s | twP+ %s" % (
                r["label"], r["psi0_deg"], f(tw.get("tip_fund_deg"), 1), f(tw.get("tip_mean_deg"), 1),
                f(tw.get("tip_lead_deg"), 0), r["lift_over_weight"], r["Fx"], r["mech_W"], f(r.get("elec_W")),
                force, gap, f(tq), f(r.get("feather_torque_peak")), f(dv.get("freq_ratio"), 4), f(dv.get("at_limit")),
                f(dv.get("kin_err_deg")), f(r.get("twist_power_pos_W"))))


def main():
    for path in sys.argv[1:]:
        rows = [json.loads(x) for x in open(path)]
        rows.sort(key=lambda r: (r.get("vacuum", False), ORDER.index(r["cand"]), r["label"]))
        print("==", path, len(rows), "rows")
        for r in rows:
            if "error" in r:
                print("%-28s ERROR %s" % (r["label"], r["error"]))
                continue
            print(line(r))
            if r.get("vacuum"):
                dv = r["drive"]
                print("   H1: total input %.4f W, motor peak shaft %.3f W: %.2f%%" % (
                    r["mech_W"], dv["shaft_peak_W"], 100.0 * r["mech_W"] / dv["shaft_peak_W"]))
            tw = (r.get("drive") or {}).get("twist") or {}
            if tw.get("hinge_swing_deg"):
                print("   hinges swing %s mean %s; tip min %.1f max %.1f; finite %s clamped %s%s" % (
                    [round(x, 1) for x in tw["hinge_swing_deg"]], [round(x, 1) for x in tw["hinge_mean_deg"]],
                    tw["tip_min_deg"], tw["tip_max_deg"], r["finite"], r["clamped"],
                    ("; trimmed %s in %d runs" % (r["trimmed"], r["evals"])) if "evals" in r else ""))


if __name__ == "__main__":
    main()
