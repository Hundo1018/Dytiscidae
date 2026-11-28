"""Read results/M1b_takeoff.jsonl (PREREG_M1b_takeoff.md).

    .venv/bin/python experiments/flapper/m1b_read.py
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rng(v, f="%.2f"):
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    a, b = min(v), max(v)
    return f % a if abs(a - b) < 1e-12 else f % a + "-" + f % b


def main():
    rows = [json.loads(l) for l in open(HERE / "results" / "M1b_takeoff.jsonl")]
    print("freeze:", [{k: (v[:16] if isinstance(v, str) else v) for k, v in r.items() if k != "kind"} for r in rows if r["kind"] == "freeze"])
    rows = [r for r in rows if r["kind"] == "score"]
    print("rows", len(rows), "QACC rows", [(r["arm"], r["entry"]) for r in rows if r["bad_qacc"]])
    g = defaultdict(list)
    for r in rows:
        g[(r["arm"], r["J_r"], r["gear"], r["jitter_seed"] is not None, r["cls"])].append(r)
    print("| arm | J_r (gear) | rows | class | pass | G1 G2 G3 G4 | t_off s | t H>=2 s | H end | span max deg | rate last 3 s | drift m | wing contact W | P peak W | P 0.1 s W | P last 3 s W | H max |")
    print("|" + "---|" * 17)
    order = ["V+", "P+", "Vs+", "N0", "N2", "VO"]
    cls_o = ["updown", "across", "oblique"]
    for k in sorted(g, key=lambda k: (order.index(k[0]), -k[1], k[3], cls_o.index(k[4]))):
        G = g[k]
        print("| %s | %.0e (%d) | %s | %s | **%d/%d** | %d %d %d %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            k[0], k[1], k[2], "jitter" if k[3] else "fresh", k[4], sum(r["passed"] for r in G), len(G),
            sum(r["G1"] for r in G), sum(r["G2"] for r in G), sum(r["G3"] for r in G), sum(r["G4"] for r in G),
            rng([r["t_off"] for r in G]), rng([r["t_H2"] for r in G]), rng([r["H_end"] for r in G]),
            rng([r["span_absmax_after"] for r in G], "%.1f"), rng([r["rate_max_last3"] for r in G]),
            rng([r["drift_m"] for r in G], "%.1f"), rng([r["peak_wing_contact_W"] for r in G], "%.1f"),
            rng([r["P_peak_W"] for r in G], "%.0f"), rng([r["P_peak_01s_W"] for r in G], "%.0f"),
            rng([r["P_last3s_W"] for r in G], "%.0f"), rng([r["H_max"] for r in G])))
    print()
    for arm in ("V+", "P+", "Vs+"):
        fresh = [r for r in rows if r["arm"] == arm and r["J_r"] == 2e-5 and r["jitter_seed"] is None]
        jit = [r for r in rows if r["arm"] == arm and r["jitter_seed"] is not None]
        ok_f = sum(r["passed"] for r in fresh)
        ok_j = sum(r["passed"] for r in jit)
        print("%s: fresh %d/%d, jitter %d/%d -> stands: %s" % (arm, ok_f, len(fresh), ok_j, len(jit),
                                                               ok_f >= 12 and ok_j >= 10))
        for r in fresh + jit:
            if not r["passed"]:
                print("   fails:", r["entry"], "jitter", r["jitter_seed"], "G%d%d%d%d" % (r["G1"], r["G2"], r["G3"], r["G4"]),
                      "t_off %.2f H_end %.2f span %s rate %.2f nose_min %s" % (r["t_off"], r["H_end"], r["span_absmax_after"],
                                                                             r["rate_max_last3"], r["nose_min_after"]))
    vp = {tuple(r["entry"]): r for r in rows if r["arm"] == "V+" and r["J_r"] == 2e-5 and r["jitter_seed"] is None}
    vs = {tuple(r["entry"]): r for r in rows if r["arm"] == "Vs+" and r["jitter_seed"] is None}
    print("Vs+ wing contact / V+'s, per fresh entry:", {str(e): round(vs[e]["peak_wing_contact_W"] / vp[e]["peak_wing_contact_W"], 2) for e in vs if e in vp})
    print("N2 H_max:", rng([r["H_max"] for r in rows if r["arm"] == "N2"]))


if __name__ == "__main__":
    main()
