"""Pass X2, stage B: B1-B3 against the frozen wording, from results.jsonl
(the last CFD row per station, and the strip solver's rows).

    python3 experiments/flapper/x2/compare_b.py [CFD case-name substring]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLMAX = 1.355


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else ""
    cfd, strip, still = {}, {}, None
    for line in (HERE / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("stage") == "B" and r.get("station") and sub in r["case"] and r.get("CN") is not None:
            cfd[r["station"]] = r
        elif r.get("stage") == "B-strip":
            strip[(r["case"], r["mode"])] = r
        elif r.get("stage") == "B" and "still" in r.get("case", ""):
            still = r
    if still:
        print(f"still-fluid plunge: added mass {still['added_mass']:.4f} vs pi c^2/4 {still['added_mass_ref']:.4f}: "
              f"ratio {still['ratio']:.3f} -> {'within' if abs(still['ratio'] - 1) <= 0.10 else 'NOT within'} 10%")
    print("| station | quantity | strip (hinge) | strip (turn) | outside solver |")
    print("|---|---|---|---|---|")
    for st in ("B8", "B6"):
        s, t, c = strip.get((st, "hinge")), strip.get((st, "turn")), cfd.get(st)
        for q in ("CN", "CX", "CP"):
            f = lambda r: "-" if r is None else f"{r[q]:.4f}"
            print(f"| {st} | {q} | {f(s)} | {f(t)} | {f(c)} |")
        if c:
            print(f"| {st} | CN max, last beat (downstroke) | {s['CN_max']:.3f} | {t['CN_max']:.3f} | "
                  f"{c['CN_max_last_beat']:.3f} ({c['CN_max_downstroke']}) |")
    s8, c8 = strip.get(("B8", "hinge")), cfd.get("B8")
    if s8 and c8:
        dn = (c8["CN"] - s8["CN"]) / s8["CN"]
        dx = c8["CX"] - s8["CX"]
        dp = (c8["CP"] - s8["CP"]) / s8["CP"]
        ok = abs(dn) <= 0.15 and abs(dx) <= 0.03 and abs(dp) <= 0.20
        print(f"B1 (B8): normal {100 * dn:+.1f}% (<= 15%), streamwise {dx:+.4f} (<= 0.03), power {100 * dp:+.1f}% (<= 20%): "
              f"{'holds' if ok else 'fails'}")
    s6, c6 = strip.get(("B6", "hinge")), cfd.get("B6")
    if s6 and c6:
        dn = (c6["CN"] - s6["CN"]) / s6["CN"]
        dp = (c6["CP"] - s6["CP"]) / s6["CP"]
        ok = dn > 0.15 and dp > 0
        print(f"B2 (B6): normal {100 * dn:+.1f}% (must be > +15%), power {100 * dp:+.1f}% (must be > 0): "
              f"{'holds' if ok else 'fails'}")
        pk = c6["CN_max_downstroke"]
        print(f"B3 (B6): downstroke peak CN {pk:.3f} vs static max {CLMAX}: {'holds' if pk is not None and pk > CLMAX else 'fails'}")


if __name__ == "__main__":
    main()
