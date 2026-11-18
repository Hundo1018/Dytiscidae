"""Pass M3: per-arm summary of loads, energy and motion (reported, not gated).

    python experiments/flapper/m3_table.py results/M3_land.jsonl
"""
import json
import sys
from collections import defaultdict

G, M = 9.80665, 0.6


def rng(v, f="%.2f"):
    v = [x for x in v if x == x]
    return (f % min(v) + "-" + f % max(v)) if v else "-"


def main(path):
    rows = [json.loads(x) for x in open(path)]
    arms = defaultdict(list)
    for r in rows:
        arms[(r["arm"], r.get("family", ""), r["kind"], r.get("J_r", 2e-5))].append(r)
    print("| arm | kind | J_r | runs | qacc | |along|/window m/s | P W | peak P window W | peak P run W | CoT | peak wing N (weights) | peak body N | stroke tau N m | feather tau N m | on curve stroke | at range feather | airborne | body contact | wing+tail share | pitch p2p deg |")
    print("|" + "---|" * 20)
    for k in sorted(arms, key=str):
        rs = arms[k]
        arm, fam, kind, jr = k
        sp = [abs(r["along"]) / r["window_s"] for r in rs if "along" in r]
        cot = [r["P_window_W"] * r["window_s"] / (M * G * max(abs(r["along"]), 1e-9)) for r in rs if "along" in r and abs(r["along"]) > 0.06]
        wing = [max(r["peak_norm_N"]["wingL"], r["peak_norm_N"]["wingR"]) for r in rs]
        print("| %s %s | %s | %.0e | %d | %d | %s | %s | %s | %s | %s | %s (%s) | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            arm, fam, kind, jr, len(rs), sum(r["bad_qacc"] > 0 for r in rs), rng(sp, "%.3f"),
            rng([r["P_window_W"] for r in rs], "%.1f"), rng([r["peak_P_window_W"] for r in rs], "%.0f"),
            rng([r["peak_P_run_W"] for r in rs], "%.0f"), rng(cot, "%.0f"), rng(wing, "%.0f"),
            rng([w / (M * G) for w in wing], "%.1f"), rng([r["peak_norm_N"]["body"] for r in rs], "%.0f"),
            rng([r["peak_tau_Nm"]["stroke"] for r in rs], "%.1f"), rng([r["peak_tau_Nm"]["feather"] for r in rs], "%.1f"),
            rng([r["at_curve_frac"]["stroke"] for r in rs], "%.3f"), rng([r["at_range_frac"]["feather"] for r in rs], "%.3f"),
            rng([r.get("airborne_frac", float("nan")) for r in rs], "%.2f"), rng([r["body_contact_frac"] for r in rs], "%.2f"),
            rng([r["share"]["wingL"] + r["share"]["wingR"] + r["share"]["tail"] for r in rs], "%.2f"),
            rng([r["pitch_p2p_deg"] for r in rs], "%.0f")))


if __name__ == "__main__":
    main(sys.argv[1])
