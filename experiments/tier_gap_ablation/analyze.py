#!/usr/bin/env python3
"""Tables from raw.json: every rung over the elites ALL listed rungs ran on.

    PYTHONPATH=. .venv/bin/python experiments/tier_gap_ablation/analyze.py R0 R0s R1 R2 ...

Per medium and rung: the median over elites of the per-elite median over seeds, the
share >= 0.15 among Tier-1 passes (``pos``: recorded >= 0.15) and among the matched
controls (``neg``), the paired median of (rung - R0s) on the same elites, and the
Spearman against R0 over all elites.  Restricting to the common elites keeps rows
comparable: a rung run on a subset is never set against a baseline run on all of them.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import BAR, per_item, spearman  # noqa: E402


def headings():
    """The heading sweep (H0..H7 = -pi .. 3pi/4 in steps of pi/4, everything else the eval draw's)."""
    d = Path(__file__).resolve().parent
    raw = json.loads((d / "raw.json").read_text())
    sel = json.loads((d / "results.json").read_text())["sel"]
    for medium in ("water", "land"):
        ids = sorted(set(sel["sets"][medium]) & {r["idx"] for r in raw if r["rung"] == "H0" and r["medium"] == medium})
        mat = {}
        for k in range(8):
            v = per_item(raw, f"H{k}", medium)
            for i in ids:
                mat.setdefault(i, []).append(v.get(i, float("nan")))
        pos = [i for i in ids if sel["recorded"][str(i)][medium] >= BAR]
        neg = [i for i in ids if sel["recorded"][str(i)][medium] < BAR]
        print(f"\n### {medium}: competence at 8 headings (eval seed kept); {len(pos)} passes + {len(neg)} controls")
        for name, grp in (("pos", pos), ("neg", neg)):
            a = np.array([mat[i] for i in grp], float)
            print(f"{name}: cells >= {BAR}: {int((a >= BAR).sum())}/{a.size} ({(a >= BAR).mean():.2f}); "
                  f"headings per elite >= {BAR}: {sorted(int((r >= BAR).sum()) for r in a)}; "
                  f"mean {a.mean():.3f}, median {np.median(a):.3f}")
        a = np.array([mat[i] for i in pos], float)
        print("passes, per elite values by heading (-pi..3pi/4):")
        for i, row in zip(pos, a):
            print(f"  elite {i:3d} rec {sel['recorded'][str(i)][medium]:.2f}: " + " ".join(f"{x:.2f}" for x in row))


def main():
    if sys.argv[1:] == ["headings"]:
        return headings()
    rungs = sys.argv[1:] or ["R0", "R0s", "R1"]
    d = Path(__file__).resolve().parent
    raw = json.loads((d / "raw.json").read_text())
    sel = json.loads((d / "results.json").read_text())["sel"]
    walls = json.loads((d / "raw.walls.json").read_text())
    for medium in ("water", "land"):
        vals = {r: per_item(raw, r, medium) for r in rungs}
        ids = set(sel["sets"][medium])
        for r in rungs:
            ids &= set(vals[r])
        ids = sorted(ids)
        rec = {i: sel["recorded"][str(i)][medium] for i in ids}
        pos = [i for i in ids if rec[i] >= BAR]
        neg = [i for i in ids if rec[i] < BAR]
        print(f"\n### {medium}: {len(pos)} Tier-1 passes (recorded >= {BAR}) + {len(neg)} controls, "
              f"{len(ids)} elites common to all rungs")
        print("| rung | wall s | median pos | mean pos | share pos | share neg | median all | "
              "d(rung-R0s) pos | rho vs R0 | rho vs rec |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        base = vals["R0"]
        ref = vals.get("R0s")
        for r in rungs:
            v = vals[r]
            a = lambda s: np.array([v[i] for i in s], float)  # noqa: E731
            dd = (np.median(a(pos) - np.array([ref[i] for i in pos]))
                  if ref and pos else float("nan"))
            rho = spearman(a(ids), [base[i] for i in ids])
            rho2 = spearman(a(ids), [rec[i] for i in ids])
            f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
            print(f"| {r} | {walls.get(r, '')} | {np.median(a(pos)):.3f} | {a(pos).mean():.3f} | "
                  f"{int((a(pos) >= BAR).sum())}/{len(pos)} | {int((a(neg) >= BAR).sum())}/{len(neg)} | "
                  f"{np.median(a(ids)):.3f} | {dd:+.3f} | {f(rho)} | {f(rho2)} |")


if __name__ == "__main__":
    main()
