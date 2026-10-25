"""Rule-8 check of the existence bar: does `comp:air >= 0.012` hold on fresh seeds?

The no-model gate certifies the bar at each elite's recorded (gen, seed); the
8-draw screens (`screen_<run>.rows.jsonl`) hold the per-draw fresh `air` of the
same bodies, elite and still arms.  A body passes here when its mean fresh air
over its K draws is >= 0.012 (the bar is on competence; the mean is what N11's
median draw approximates).  Verdicts use the gate's own rule
(`domain.evidence.no_model_verdict`, elite lower bound against still upper bound,
one-sided 95%).  The screens carry no base arm; the base-gait comparison stays
with the recorded-seed gate.

Usage: python experiments/air_turn_bar/fresh_existence.py arch53 arch54 arch55 arch56 arch57
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from dytiscidae.domain.evidence import clopper_pearson, no_model_verdict  # noqa: E402

FLOOR = 0.012


def counts(run):
    rows = [json.loads(l) for l in open(HERE / f"screen_{run}.rows.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["index"], r["arm"])].append(r)
    out = {}
    for arm in ("elite", "still"):
        bodies = [rs for (i, a), rs in by.items() if a == arm]
        mean_pass = sum((sum(r["air"] or 0.0 for r in rs) / len(rs)) >= FLOOR for rs in bodies)
        any_pass = sum(any((r["air"] or 0.0) >= FLOOR for r in rs) for rs in bodies)
        draws = [r for rs in bodies for r in rs]
        out[arm] = {"bodies": len(bodies), "mean_pass": mean_pass, "any_pass": any_pass,
                    "draws": len(draws), "draw_pass": sum((r["air"] or 0.0) >= FLOOR for r in draws)}
    return out


def verdict_line(run, c):
    e, s = c["elite"], c["still"]
    v = no_model_verdict(s["mean_pass"], s["bodies"], e["mean_pass"], e["bodies"])
    elo, _ = clopper_pearson(e["mean_pass"], e["bodies"])
    _, shi = clopper_pearson(s["mean_pass"], s["bodies"])
    return (f"| {run} | {e['mean_pass']}/{e['bodies']} (lower {100 * elo:.1f}%) | "
            f"{s['mean_pass']}/{s['bodies']} (upper {100 * shi:.1f}%) | {v} | "
            f"{e['any_pass']}/{e['bodies']} vs {s['any_pass']}/{s['bodies']} | "
            f"{e['draw_pass']}/{e['draws']} vs {s['draw_pass']}/{s['draws']} |")


if __name__ == "__main__":
    runs = sys.argv[1:] or ["arch53", "arch54", "arch55", "arch56", "arch57"]
    print(f"Fresh-seed existence bar `comp:air >= {FLOOR}` (mean over K fresh draws), elite vs still; no base arm in the screens.")
    print("| population | elite bodies >= bar | still bodies >= bar | verdict (gate rule) | any draw >= bar (elite vs still) | draws >= bar (elite vs still) |")
    print("|---|---|---|---|---|---|")
    pooled = {a: collections.Counter() for a in ("elite", "still")}
    for run in runs:
        c = counts(run)
        print(verdict_line(run, c))
        for a in pooled:
            pooled[a].update(c[a])
    print(verdict_line(f"pooled {len(runs)} populations", {a: dict(pooled[a]) for a in pooled}))
