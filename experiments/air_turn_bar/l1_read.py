"""E2 air L1 read for one run, from its gate and screen outputs.

Reproduces the arch56 table in runs/arch56_notes.md (F1 134/832, F2 25/104, S1
26/111, positive share 21.1%, R1 0, R2 certified 7/111 vs 0 vs 0, R3 3 of 8).
Thresholds are the frozen ones (R1 >= 5 sign-test passes, R2 comp:air>=0.012
certified against still and base, R3 >= 5 above-floor elites retaining >= 0.5 of
their recorded air on the fresh draws); nothing here sets a threshold.

Usage: python experiments/air_turn_bar/l1_read.py arch56 [arch58 ...]
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FLOOR = 0.012
R1_NEED, R3_NEED, R3_RETAIN = 5, 5, 0.5
MEASURABLE = 4


def read(run):
    gate = json.load(open(HERE / f"gate_{run}.json"))
    screen = json.load(open(HERE / f"screen_{run}.json"))
    rows = [json.loads(l) for l in open(HERE / f"screen_{run}.rows.jsonl")]
    arch = json.load(open(ROOT / "runs" / run / "archive_air.json"))["elites"]

    el = [r for r in rows if r["arm"] == "elite"]
    gan = [r for r in el if r["plan"] == "gannet"]
    f1 = (sum("battery" in (r["failure"] or "").lower() for r in gan), len(gan))
    ag = [e for e in arch if e["meta"].get("body_plan") == "gannet"]
    f2 = (sum((e["meta"].get("n_rotors") or 0) > 0 for e in ag), len(ag))

    per = screen["per_elite"]
    elites = [v for v in per.values() if v["arm"] == "elite"]
    meas = [v for v in elites if v["measurable"] >= MEASURABLE]
    s1 = (len(meas), len(elites))
    pos_share = (sum(v["positive"] for v in meas) / sum(v["measurable"] for v in meas)) if meas else float("nan")
    r1 = [v for v in elites if v["verdict"] == "pass"]
    with_pos = sum(v["positive"] > 0 for v in elites)

    bar = next(t for t in gate["table"] if t["bar"] == f"comp:air>={FLOOR}")
    r2 = (bar["elite"], bar["still"], bar["base"], bar["verdict_still"], bar["verdict_base"])
    r2_ok = bar["verdict_still"] == "certified" and bar["verdict_base"] == "certified" and bar["elite"][0] >= 5

    by = collections.defaultdict(list)
    for r in el:
        by[r["index"]].append(r)
    ret = []
    for i, rs in by.items():
        rec = rs[0]["recorded_air"] or 0.0
        if rec >= FLOOR:
            fresh = sum(r["air"] or 0.0 for r in rs) / len(rs)
            ret.append((i, rec, fresh, fresh / rec))
    ret.sort(key=lambda t: -t[3])
    r3 = sum(t[3] >= R3_RETAIN for t in ret)

    still = [v for v in per.values() if v["arm"] == "still" and v["measurable"] >= 1]
    still_pos = sum(v["positive"] for v in still)
    still_max = max((v["mean"] for v in still), default=float("nan"))

    return {
        "run": run, "F1": f1, "F2": f2, "S1": s1, "positive_share": pos_share,
        "R1_passes": len(r1), "elites_with_positive_draw": with_pos,
        "expected_passes_under_null": screen["summary"]["expected_passes_under_null"],
        "R2": r2, "R2_met": r2_ok, "R3": r3, "R3_of": len(ret), "retention": ret,
        "still_measurable_bodies": len(still), "still_positive_draws": still_pos, "still_max_mean": still_max,
        "L1_met": len(r1) >= R1_NEED and r2_ok and r3 >= R3_NEED,
    }


def fmt(d):
    f1, f2, s1 = d["F1"], d["F2"], d["S1"]
    pct = lambda a, b: f"{a}/{b} = {100 * a / b:.1f}%" if b else "n/a"
    r2 = d["R2"]
    lines = [
        f"## {d['run']}",
        "| read | value | bar | verdict |", "|---|---|---|---|",
        f"| F1 gannet fresh draws battery exhausted | {pct(*f1)} | (reported) | |",
        f"| F2 air-archive gannets with rotors | {pct(*f2)} | (reported) | |",
        f"| S1 elites measurable on >= {MEASURABLE}/8 draws | {pct(*s1)} | arch56 23.4% | |",
        f"| positive share of measurable draws | {100 * d['positive_share']:.1f}% | arch56 21.1% | |",
        f"| R1 sign-test passes | {d['R1_passes']} ({d['elites_with_positive_draw']} elites with a positive draw, "
        f"expected under null {d['expected_passes_under_null']}) | >= {R1_NEED} | {'met' if d['R1_passes'] >= R1_NEED else 'missed'} |",
        f"| R2 gate comp:air>={FLOOR} | elite {r2[0][0]}/{r2[0][1]}, still {r2[1][0]}/{r2[1][1]} ({r2[3]}), "
        f"base {r2[2][0]}/{r2[2][1]} ({r2[4]}) | certified both, >= 5 | {'met' if d['R2_met'] else 'missed'} |",
        f"| R3 retention >= {R3_RETAIN} | {d['R3']} of {d['R3_of']} above-floor elites | >= {R3_NEED} | {'met' if d['R3'] >= R3_NEED else 'missed'} |",
        f"| still arm | {d['still_measurable_bodies']} measurable bodies, {d['still_positive_draws']} positive draws, "
        f"max mean {d['still_max_mean']:.3f} | <= 0 | {'null holds' if d['still_max_mean'] <= 0 else 'NULL BROKEN'} |",
        f"| **L1 (R1 and R2 and R3)** | | | **{'met' if d['L1_met'] else 'not met'}** |",
        "",
        "Retention per above-floor elite (index, recorded air, mean fresh air, ratio): "
        + ", ".join(f"{i} {rec:.3f}->{fr:.3f} ({ra:.2f})" for i, rec, fr, ra in d["retention"]),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    for run in sys.argv[1:] or ["arch56"]:
        print(fmt(read(run)))
