"""Q5 (ROADMAP 2026-10-10, later, section 4): the K curve on real draws.

Reads only stored rows (CPU, no run directory touched):
    experiments/draw_variance/results_arch49.json.rows.jsonl
For each medium: Spearman between the mean of k fresh draws and the mean of the
other 6-k fresh draws, averaged over random k-subsets, against the formula.

Two independent means of k and m draws of the same design have
  corr = var_d / sqrt((var_d + s2/k)(var_d + s2/m)) = sqrt(rel_k * rel_m),
  rel_n = n / (n + R),  R = s2 / var_d  (Spearman-Brown form).
Run:  PYTHONPATH=. .venv/bin/python experiments/draw_variance/k_curve.py
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from run import spearman, components  # same estimators as C2

HERE = Path(__file__).parent
SRC = HERE / "results_arch49.json.rows.jsonl"
OUT = HERE / "k_curve_arch49.json"
N_SUB, SEED = 200, 20261010

rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
filed = json.loads((HERE / "results_arch49.json").read_text())["table"]
rng = np.random.default_rng(SEED)
rel = lambda n, R: n / (n + R)
out = {}
for med in ("air", "water", "land"):
    F = np.array([[d[med] for d in r["score"]["fresh"]] for r in rows])  # (229, 6)
    keep = (F > 0).any(axis=1)
    X = F[keep]
    n_ex = int((~keep).sum())
    _, _, R_all = components(F)
    _, _, R_inc = components(X)
    res = {"n": int(len(rows)), "excluded_all_zero": n_ex, "n_used": int(keep.sum()),
           "R_file": filed[med]["R"], "R_recomputed_all": R_all, "R_recomputed_included": R_inc,
           "k": {}}
    for k in range(1, 6):
        combos = list(itertools.combinations(range(6), k))
        pick = [combos[i] for i in rng.choice(len(combos), size=min(N_SUB, len(combos)), replace=False)] \
            if len(combos) > N_SUB else combos
        sp, pe = [], []
        for c in pick:
            rest = [j for j in range(6) if j not in c]
            a, b = X[:, list(c)].mean(1), X[:, rest].mean(1)
            sp.append(spearman(a, b)); pe.append(float(np.corrcoef(a, b)[0, 1]))
        # note: only C(6,k) <= 20 distinct subsets exist, so all are used (not 200)
        f = {tag: float(np.sqrt(rel(k, R) * rel(6 - k, R)))
             for tag, R in (("inc", R_inc), ("all", R_all), ("file", filed[med]["R"]))}
        res["k"][k] = {"n_subsets": len(pick), "spearman_mean": float(np.mean(sp)),
                       "spearman_sd": float(np.std(sp, ddof=1)), "pearson_mean": float(np.mean(pe)),
                       "formula_R_included": f["inc"], "formula_R_all": f["all"], "formula_R_file": f["file"],
                       "diff_vs_included": float(np.mean(sp)) - f["inc"],
                       "diff_vs_file": float(np.mean(sp)) - f["file"]}
    res["K_needed"] = {str(t): {"R_included": t / (1 - t) * R_inc, "R_file": t / (1 - t) * filed[med]["R"]}
                       for t in (0.8, 0.9, 0.95)}
    out[med] = res
OUT.write_text(json.dumps(out, indent=1))
for med, r in out.items():
    print(f"{med}: n_used={r['n_used']} excluded={r['excluded_all_zero']} R_file={r['R_file']:.2f} R_all={r['R_recomputed_all']:.2f} R_inc={r['R_recomputed_included']:.2f}")
    for k, v in r["k"].items():
        print(f"  k={k} subsets={v['n_subsets']:3d} spearman={v['spearman_mean']:.3f}+-{v['spearman_sd']:.3f} pearson={v['pearson_mean']:.3f} "
              f"formula(inc)={v['formula_R_included']:.3f} formula(file)={v['formula_R_file']:.3f} d_inc={v['diff_vs_included']:+.3f} d_file={v['diff_vs_file']:+.3f}")
    print("  K for 0.8/0.9/0.95 (R_inc):", [round(r['K_needed'][t]['R_included'], 1) for t in ('0.8', '0.9', '0.95')],
          "(R_file):", [round(r['K_needed'][t]['R_file'], 1) for t in ('0.8', '0.9', '0.95')])
