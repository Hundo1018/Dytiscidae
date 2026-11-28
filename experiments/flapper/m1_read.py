"""Read results/M1_takeoff.jsonl into the tables of RESULT_M1_takeoff.md.

    .venv/bin/python experiments/flapper/m1_read.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def rng(v, f="%.2f"):
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    return (f % min(v)) if abs(min(v) - max(v)) < 1e-12 else (f % min(v) + "-" + f % max(v))


def main():
    rows = [json.loads(l) for l in open(HERE / "results" / "M1_takeoff.jsonl")]
    fr = [r for r in rows if r.get("kind") == "freeze"]
    print("freeze rows:", [(f["sha256"][:16], f["file"]) for f in fr])
    rows = [r for r in rows if r.get("kind") == "score"]
    groups = defaultdict(list)
    for r in rows:
        key = (r["set"], r["arm"], r["J_r"], r["gear"], json.dumps(r["contact"]) if r["contact"] else "")
        groups[key].append(r)
    print("| set | arm | J_r (gear) | n | pass | G1 | G2 | G3 | G4 | QACC | t_off s | t H>=2 s | H first 0.5 s | H end | nose min | rate last 3 s | drift m | wing contact W | P peak W | P 0.1 s W | P last 3 s W | stroke on curve, strike | contact/fluid moment |")
    print("|" + "---|" * 23)
    order = ["V", "V+", "Vs", "P", "P+", "VO", "R0", "N0", "N1", "N2"]
    for key in sorted(groups, key=lambda k: (k[0] != "scoring", order.index(k[1]), -k[2], k[3], k[4])):
        g = groups[key]
        n = len(g)
        ok = [r for r in g if r["passed"]]
        q = [r for r in g if r["bad_qacc"]]
        print("| %s%s | %s | %.0e (%d) | %d | **%d** | %d | %d | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            key[0], (" " + key[4]) if key[4] else "", key[1], key[2], key[3], n, len(ok),
            sum(r["G1"] for r in g), sum(r["G2"] for r in g), sum(r["G3"] for r in g), sum(r["G4"] for r in g), len(q),
            rng([r["t_off"] for r in g]), rng([r["t_H2"] for r in g]), rng([r["H_max_first05"] for r in g]),
            rng([r["H_end"] for r in g]), rng([r["nose_min_after"] for r in g], "%.0f"),
            rng([r["rate_max_last3"] for r in g]), rng([r["drift_m"] for r in g], "%.1f"),
            rng([r["peak_wing_contact_W"] for r in g], "%.1f"), rng([r["P_peak_W"] for r in g], "%.0f"),
            rng([r["P_peak_01s_W"] for r in g], "%.0f"), rng([r["P_last3s_W"] for r in g], "%.0f"),
            rng([r["stroke_on_limit_strike"] for r in g]), rng([r["moment_ratio"] for r in g], "%.1f")))
    print()
    print("QACC rows:", [(r["arm"], r["entry"], r["J_r"], r["gear"], r["bad_qacc"]) for r in rows if r["bad_qacc"]])
    print("hinge peak N m by arm:", {a: rng([r["peak_stroke_hinge_Nm"] for r in rows if r["arm"] == a and r["set"] == "scoring"], "%.1f") for a in order})
    print("energy J (t_c to end), scoring, 2e-5:", {a: rng([r["E_elec_J"] for r in rows if r["arm"] == a and r["J_r"] == 2e-5 and r["set"] == "scoring"], "%.0f") for a in order})
    print("first 2 s mean W, scoring:", {(a, r_J): rng([r["P_first2s_W"] for r in rows if r["arm"] == a and r["J_r"] == r_J and r["set"] == "scoring" and r["gear"] == (24.0 if r_J == 2e-5 else 32.0)], "%.0f") for a in ("V", "V+", "P+") for r_J in (2e-5, 5e-6, 1e-6)})
    print("R0 drift / end x:", [(r["entry"], round(r["drift_m"], 2)) for r in rows if r["arm"] == "R0"])
    print("failing rows of V+/P+ (scoring):", [(r["arm"], r["entry"], r["J_r"], r["gear"], "G%d%d%d%d" % (r["G1"], r["G2"], r["G3"], r["G4"])) for r in rows if r["arm"] in ("V+", "P+") and not r["passed"]])


if __name__ == "__main__":
    main()
