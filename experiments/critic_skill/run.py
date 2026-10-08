"""Why does arch48's critic have no skill?  (arch48 work list item 3.)

Reads the critic's own labels from a finished run's ``search_state.pkl`` and
asks, per medium: how many Tier-2 labels are non-zero, how well the cheap
(Tier-1) score ranks Tier-2, how well the critic's out-of-fold prediction does,
and which cheap feature correlates with Tier-2 at all.  No simulator.

    PYTHONPATH=. .venv/bin/python experiments/critic_skill/run.py --run runs/arch48
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from dytiscidae.evolution.critic import CRITIC_FEATURES, CRITIC_TARGETS, _TARGET_IDX, _corr, _ridge


def oof(Z, y, ridge, k=5):
    folds = np.arange(len(y)) % k
    p = np.empty(len(y))
    for f in range(k):
        tr, te = folds != f, folds == f
        b, w = _ridge(Z[tr], y[tr], ridge)
        p[te] = Z[te] @ w + b
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/arch48")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    state = pickle.load(open(Path(args.run) / "search_state.pkl", "rb"))
    c = state["critic"]
    X, Y = np.asarray(c._x, float), np.asarray(c._y, float)
    sd = X.std(0)
    Z = (X - X.mean(0)) / np.where(sd > 1e-9, sd, 1.0)
    out = {"run": args.run, "labels": int(len(X)), "media": {}}
    for j, k in enumerate(CRITIC_TARGETS):
        rows = np.isfinite(Y[:, j])
        cheap = np.clip(X[rows, _TARGET_IDX[j]], 0.0, 1.0)
        obs = Y[rows, j] + cheap
        implied = np.clip(oof(Z[rows], Y[rows, j], c.ridge), -1, 1) + cheap
        direct = oof(Z[rows], obs, c.ridge)
        feats = sorted(((round(_corr(X[rows, i], obs), 3), CRITIC_FEATURES[i])
                        for i in range(X.shape[1])), key=lambda t: -abs(t[0]))[:4]
        out["media"][k] = {
            "n": int(rows.sum()), "tier2_nonzero": int((obs > 0).sum()),
            "tier2_mean": float(obs.mean()), "tier1_mean": float(cheap.mean()),
            "corr_tier1_tier2": round(_corr(cheap, obs), 3),
            "corr_critic_oof_tier2": round(_corr(implied, obs), 3),
            "corr_direct_oof_tier2": round(_corr(direct, obs), 3),
            "top_features": feats,
        }
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
