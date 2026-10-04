"""What predicts a design's evaluation wall: DOF, rotors, or both?

`--pool-balance` assigns machines to shards by a predicted cost.  Until
2026-10-03 that cost was `35 + 0.9 * n_actuated` (DOF only), and the budget
sweep found the cost is carried by rotors.  This fits ordinary least squares on
every `evaluate` event of the named runs and prints, per model, R^2 and the
coefficients, so the constants in `envs/actors.shard_cost` come from a read.

    python experiments/budget_sweetspot/cost_model.py runs/arch45 runs/arch46
"""
import json
import sys
from pathlib import Path

import numpy as np


def rows(run):
    out = []
    for line in open(Path(run) / "events.jsonl"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "evaluate" and e.get("wall") and "dof" in e:
            out.append((float(e["wall"]), float(e["dof"]), float(e.get("n_rotors", 0))))
    return np.array(out)


def fit(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
    return beta, r2


def main(paths):
    for run in paths:
        d = rows(run)
        y, dof, rot = d[:, 0], d[:, 1], d[:, 2]
        print(f"== {Path(run).name}: {len(y)} evaluations, wall median {np.median(y):.1f} s")
        for name, cols in (("dof", [dof]), ("rotors", [rot]), ("dof+rotors", [dof, rot])):
            beta, r2 = fit(y, cols)
            print(f"  {name:11s} R2 {r2:.3f}  coef " + " ".join(f"{b:+.3f}" for b in beta))
        # Ranking is what balance uses: Spearman between prediction and wall.
        for name, cols in (("dof", [dof]), ("dof+rotors", [dof, rot])):
            beta, _ = fit(y, cols)
            pred = np.column_stack([np.ones(len(y))] + cols) @ beta
            rp, ry = np.argsort(np.argsort(pred)), np.argsort(np.argsort(y))
            print(f"  spearman {name:11s} {np.corrcoef(rp, ry)[0, 1]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
