"""Where does a run's capability come from, generation by generation?

Reads runs/<run>/snapshots/genNNNN_<island>.json (every 50 gens, all islands),
pools the elites of every island at each snapshot, and reports per medium the
share of elites standing at or above each rung, plus wall seconds per generation
from generations.jsonl. Within-run only: ladders differ between runs.

    python experiments/budget_sweetspot/retro.py runs/arch45 runs/arch46
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

COMP = ("air", "water", "land")


def snapshots(run):
    by = defaultdict(list)
    for p in sorted((run / "snapshots").glob("gen*_*.json")):
        g = int(re.match(r"gen(\d+)_", p.name).group(1))
        by[g].extend(json.loads(p.read_text())["elites"])
    return dict(sorted(by.items()))


def wall(run):
    t = {}
    for line in open(run / "generations.jsonl"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "generation" in r:
            t[r["generation"]] = r.get("elapsed", r.get("t"))
    return t


def main(paths):
    for run in map(Path, paths):
        snaps, t = snapshots(run), wall(run)
        print(f"\n== {run.name}  ({len(snaps)} snapshots, last gen {max(t)})")
        # Elites carry per-medium competence in meta; rungs live only on best_meta.
        print("gen   hours    n  " + "  ".join(f"{m:>5}: med top10 >=.3" for m in COMP))
        out = []
        for g, es in snaps.items():
            row = {"gen": g, "hours": round(t.get(g, 0) / 3600, 1), "n": len(es)}
            cells = []
            for m in COMP:
                v = sorted((e["meta"].get(m) or 0.0) for e in es)
                med, top10 = v[len(v) // 2], sum(v[-10:]) / min(10, len(v))
                hi = sum(x >= 0.3 for x in v)
                row[m] = {"median": med, "top10": top10, "n_ge_0.3": hi}
                cells.append(f"{med:5.3f} {top10:5.3f} {hi:4d}")
            out.append(row)
            print(f"{g:4d} {row['hours']:6.1f} {row['n']:4d}  " + "  ".join(cells))
        gens = sorted(t)
        for a, b in ((0, 6), (6, 100), (100, 300), (300, 500), (500, 700), (700, 899)):
            if a in t and b in t:
                print(f"s/gen {a:3d}-{b:3d}: {(t[b]-t[a])/(b-a):6.1f}")
        (run / "budget_retro.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:])
