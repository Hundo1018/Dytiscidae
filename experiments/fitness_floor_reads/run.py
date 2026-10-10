"""Four stored-data reads from ROADMAP section "2026-10-10 -- why the search does not progress".

    N2   the quantile floor in the curriculum windows        (search_state.pkl)
    N3   draw-variance retention against `improvements`       (draw_variance rows x archives)
    N4   parent-selection weight of Tier-2 elites             (archive_<island>.pkl fronts)
    N10  PPO return decomposition, if logged                  (events.jsonl `ppo`)

Read-only on runs/arch*.  Usage (repo root):

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/fitness_floor_reads/run.py n2 n3 n4 n10

Each subcommand writes results_<item>.json next to this file and prints its table.
"""
import collections
import glob
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# NOTE: the pickles read here are this project's own run outputs (runs/arch*), not untrusted input.
HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
ISLANDS = ["air", "water", "land", "amphibian", "aerial_diver", "land_air",
           "triphibian", "generalist"]


def dump(name, obj):
    (HERE / f"results_{name}.json").write_text(json.dumps(obj, indent=1, default=float))


def ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size)
    r[order] = np.arange(x.size, dtype=float)
    _, inv = np.unique(x, return_inverse=True)
    return (np.bincount(inv, weights=r) / np.bincount(inv))[inv]


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_ci(fn, n, rng, B=2000):
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if not math.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


# ---------------------------------------------------------------- N2
def n2():
    out = {}
    for run in ("arch49", "arch50"):
        st = pickle.load(open(RUNS / run / "search_state.pkl", "rb"))
        res = {"saved_generation": st["generation"], "evaluated": st["evaluated"], "islands": {}}
        for isl, cur in st["curricula"].items():
            for stage, w in cur._recent.items():
                a = np.array(w, float)
                if len(a) < cur.min_rank_samples:
                    res["islands"].setdefault(isl, {})[f"stage{stage}"] = {"n": len(a), "skipped": "< 12 samples"}
                    continue
                isl_s, cur_s, mis = a[:, 0], a[:, 1], a[:, 2]
                # blend a design that is zero on all three receives (stage-0 handover w = 0.25)
                wgt = 1.0 if stage >= 4 else max(stage / 4, 0.25)
                zq = (float(np.mean(isl_s <= 0)), float(np.mean(cur_s <= 0)), float(np.mean(mis <= 0)))
                blend0 = 0.7 * (wgt * zq[0] + (1 - wgt) * zq[1]) + 0.3 * zq[2]
                res["islands"].setdefault(isl, {})[f"stage{stage}"] = {
                    "window": int(len(a)),
                    "island_score_eq0": float(np.mean(isl_s == 0)),
                    "island_score_le1e-6": float(np.mean(isl_s <= 1e-6)),
                    "island_q_of_zero_scorer": zq[0],
                    "island_nonzero_n": int(np.sum(isl_s > 0)),
                    "island_max": float(isl_s.max()),
                    "curr_score_eq0": float(np.mean(cur_s == 0)),
                    "curr_q_of_zero_scorer": zq[1],
                    "mission_eq0": float(np.mean(mis == 0)),
                    "mission_le1e-6": float(np.mean(mis <= 1e-6)),
                    "mission_q_of_zero_scorer": zq[2],
                    "blend_of_all_zero_scorer": float(blend0),
                    "n_negative": int(np.sum(a < 0)),
                }
        out[run] = res
    dump("n2", out)
    for run, res in out.items():
        print(f"[{run}] saved generation {res['saved_generation']}, evaluated {res['evaluated']}")
        print("island        stage n   isl==0  isl<=1e-6  q(isl|0)  nz  q(cur|0)  mis==0  mis<=1e-6  blend(all-zero)")
        for isl in ISLANDS:
            for sk, r in res["islands"].get(isl, {}).items():
                if "skipped" in r:
                    print(f"{isl:13s} {sk} n={r['n']} skipped"); continue
                print(f"{isl:13s} {sk} {r['window']:3d}  {r['island_score_eq0']:.3f}   {r['island_score_le1e-6']:.3f}     "
                      f"{r['island_q_of_zero_scorer']:.3f}   {r['island_nonzero_n']:3d}  {r['curr_q_of_zero_scorer']:.3f}   "
                      f"{r['mission_eq0']:.3f}   {r['mission_le1e-6']:.3f}      {r['blend_of_all_zero_scorer']:.3f}")


# ---------------------------------------------------------------- N3
def n3():
    rows = [json.loads(l) for l in open(ROOT / "experiments/draw_variance/results_arch49.json.rows.jsonl")]
    # Exact join: the draw-variance run indexed the elites of ops.run.load_run_archive (the cross-island
    # merge, 229 occupants).  row["index"] is the position in list(merged.cells.values()); every join is
    # then verified against (island, gen, eval_seed) and the three recorded medium scores.
    from dytiscidae.ops.run import load_run_archive
    merged, _ = load_run_archive(str(RUNS / "arch49"))
    els = list(merged.cells.values())
    joined, miss, mism = [], [], []
    for r in rows:
        i = int(r["index"])
        if i >= len(els):
            miss.append(i); continue
        e = els[i]; m = e.meta or {}
        ok = (m.get("island") == r["island"] and int(m.get("gen") or 0) == int(r["gen"])
              and int(m.get("eval_seed") or 0) == int(r["seed"])
              and all(abs(float(m.get(k) or 0) - float(r["recorded"][k])) < 5e-4 for k in ("air", "water", "land")))
        if not ok:
            mism.append(i); continue
        joined.append((r, {"improvements": e.improvements, "curiosity": e.curiosity, "tier": e.tier}))
    # cross-check with the per-island JSON archives, disambiguating the shared (gen, eval_seed) by the scores
    arch = {}
    for p in sorted(glob.glob(str(RUNS / "arch49/archive_*.json"))):
        d = json.load(open(p))
        for e in d["elites"]:
            m = e["meta"]
            arch.setdefault((m.get("island"), int(m.get("gen") or 0), int(m.get("eval_seed") or 0)), []).append(e)
    json_agree = 0
    for r, e in joined:
        c = [x for x in arch.get((r["island"], int(r["gen"]), int(r["seed"])), [])
             if all(abs(float(x["meta"].get(k) or 0) - float(r["recorded"][k])) < 5e-4 for k in ("air", "water", "land"))]
        json_agree += bool(c) and all(x["improvements"] == e["improvements"] for x in c)
    multi = 0
    rng = np.random.default_rng(20261010)
    media = ("air", "water", "land")
    res = {"join_key": "row.index == position in list(load_run_archive(runs/arch49).cells.values()); verified on (island, gen, eval_seed) and the three recorded scores (<5e-4)",
           "rows": len(rows), "joined_verified": len(joined), "index_out_of_range": miss, "failed_verification": mism,
           "merged_archive_elites": len(els),
           "json_crosscheck_agree(improvements equal in archive_<island>.json, matched on island/gen/seed/scores)": json_agree,
           "(island,gen,eval_seed) alone is ambiguous: distinct keys": len(arch), "archive_json_elites": sum(len(v) for v in arch.values()),
           "media": {}}
    imp = np.array([e["improvements"] for _, e in joined], float)
    cur = np.array([e["curiosity"] for _, e in joined], float)
    gen = np.array([r["gen"] for r, _ in joined], float)
    res["improvements_distribution"] = {str(k): int(v) for k, v in sorted(collections.Counter(imp.astype(int)).items())}
    for m in media:
        rec = np.array([r["recorded"][m] for r, _ in joined], float)
        fr = np.array([np.mean([f[m] for f in r["score"]["fresh"]]) for r, _ in joined], float)
        d = rec - fr
        sp_imp = spearman(d, imp)
        ci = boot_ci(lambda idx: spearman(d[idx], imp[idx]), len(d), rng)
        # robustness: partial Spearman controlling for recorded level (rank regression-to-mean)
        rr, rd, ri = ranks(rec), ranks(d), ranks(imp)

        def resid(y, x):
            b = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
            return y - b * x
        pr = float(np.corrcoef(resid(rd, rr), resid(ri, rr))[0, 1]) if rr.std() > 0 else float("nan")
        # retention among elites with recorded > 0
        pos = rec > 0
        ret = np.where(pos, fr / np.where(pos, rec, 1), np.nan)
        g0, g3 = pos & (imp == 0), pos & (imp >= 3)
        m0 = float(np.median(ret[g0])) if g0.sum() else float("nan")
        m3 = float(np.median(ret[g3])) if g3.sum() else float("nan")
        # pooled retention (sum fresh / sum recorded) as a second view, robust to tiny denominators
        p0 = float(fr[g0].sum() / rec[g0].sum()) if g0.sum() else float("nan")
        p3 = float(fr[g3].sum() / rec[g3].sum()) if g3.sum() else float("nan")

        def ratio_med(idx):
            ii = imp[idx]; rr_ = ret[idx]; pp = pos[idx]
            a = rr_[pp & (ii == 0)]; b = rr_[pp & (ii >= 3)]
            return float(np.median(b) / np.median(a)) if len(a) and len(b) and np.median(a) > 0 else float("nan")
        rci = boot_ci(ratio_med, len(d), rng)
        res["media"][m] = {
            "n_joined": len(d), "n_recorded_gt0": int(pos.sum()), "n_fresh_nonzero_any": int(np.sum(fr > 0)),
            "mean_recorded": float(rec.mean()), "mean_fresh": float(fr.mean()),
            "spearman(rec-fresh, improvements)": sp_imp, "spearman_ci95": ci,
            "partial_spearman_given_recorded_rank": pr,
            "spearman(rec-fresh, curiosity)": spearman(d, cur),
            "spearman(rec-fresh, meta.gen)": spearman(d, gen),
            "spearman(retention, improvements) [rec>0]": spearman(ret[pos], imp[pos]) if pos.sum() > 3 else float("nan"),
            "imp0": {"n_all": int((imp == 0).sum()), "n_rec>0": int(g0.sum()), "median_retention": m0,
                     "q25_q75": [float(np.nanpercentile(ret[g0], 25)), float(np.nanpercentile(ret[g0], 75))] if g0.sum() else None,
                     "pooled_retention": p0},
            "imp>=3": {"n_all": int((imp >= 3).sum()), "n_rec>0": int(g3.sum()), "median_retention": m3,
                       "q25_q75": [float(np.nanpercentile(ret[g3], 25)), float(np.nanpercentile(ret[g3], 75))] if g3.sum() else None,
                       "pooled_retention": p3},
            "ratio_median_ret(imp>=3 / imp0)": (m3 / m0) if m0 and m0 > 0 else float("nan"),
            "ratio_ci95": rci,
            "ratio_pooled": (p3 / p0) if p0 and p0 > 0 else float("nan"),
        }
    dump("n3", res)
    print(json.dumps({k: v for k, v in res.items() if k != "media"}, indent=1))
    for m, v in res["media"].items():
        print(f"== {m}")
        for k, x in v.items():
            print(f"  {k}: {x}")


# ---------------------------------------------------------------- N4
def _load_arch_pkl(run, isl):
    return pickle.load(open(RUNS / run / f"archive_{isl}.pkl", "rb"))


def base_weights(a):
    """select_parent's base weight over every non-tainted front member (curator.py:292-358)."""
    cells = [e for c, front in a["fronts"].items() if c not in a["tainted"] for e in front]
    if not cells:
        cells = [e for c, e in a["cells"].items() if c not in a["tainted"]]
    fit = np.array([e.fitness for e in cells], float)
    cur = np.array([e.curiosity for e in cells], float)
    keys = list(a["cells"])

    def density(cell, radius=1):
        return sum(all(abs(x - y) <= radius for x, y in zip(cell, o)) for o in keys)
    den = np.array([density(e.cell) for e in cells], float)
    age = np.array([a["generation"] - e.born_at for e in cells], float)

    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-12 else np.zeros_like(x)
    w = 0.35 * norm(fit) + 0.30 * cur + 0.25 * (1 - norm(den)) + 0.10 * np.exp(-age / 25.0)
    return cells, np.maximum(w, 1e-6)


def n4():
    res = {}
    for run in ("arch48", "arch49", "arch50"):
        r = {"islands": {}}
        for isl in ISLANDS:
            a = _load_arch_pkl(run, isl)
            cells, w = base_weights(a)
            tier = np.array([e.tier for e in cells])
            fit = np.array([e.fitness for e in cells])
            n_json = len(json.load(open(RUNS / run / f"archive_{isl}.json"))["elites"])
            ent = {"saved_generation": a["generation"], "pool": len(cells), "cells": len(a["cells"]),
                   "json_elites": n_json, "tainted": len(a["tainted"]),
                   "tier2": int((tier >= 2).sum()), "island_median_w": float(np.median(w)),
                   "p10_w": float(np.percentile(w, 10))}
            if (tier >= 2).any():
                t2 = tier >= 2
                # decile of each tier-2 member: 1 = bottom tenth; fraction of pool with strictly lower weight
                lower = np.array([np.mean(w < x) + 0.5 * np.mean(w == x) for x in w[t2]])
                dec = np.minimum((lower * 10).astype(int) + 1, 10)
                ent.update({
                    "t2_deciles": dec.tolist(),
                    "t2_share_bottom_decile": float(np.mean(w[t2] <= np.percentile(w, 10))),
                    "t2_median_w": float(np.median(w[t2])),
                    "t2_median_over_island_median": float(np.median(w[t2]) / np.median(w)),
                    "t1_median_w": float(np.median(w[~t2])) if (~t2).any() else None,
                    "t2_median_fitness": float(np.median(fit[t2])),
                    "t1_median_fitness": float(np.median(fit[~t2])) if (~t2).any() else None,
                    "t2_median_decile": float(np.median(dec)),
                    # emigrant rule: top n by fitness among cell representatives
                    "t2_min_fitness_rank_among_representatives": int(min(
                        1 + sum(x.fitness > e.fitness for x in a["cells"].values())
                        for e in cells if e.tier >= 2)),
                })
            r["islands"][isl] = ent
        has = [i for i, v in r["islands"].items() if v["tier2"] > 0]
        r["islands_with_tier2"] = has
        r["islands_median_t2_in_bottom_decile"] = [i for i in has if r["islands"][i]["t2_median_w"] <= r["islands"][i]["p10_w"]]
        r["islands_t2_median_above_island_median"] = [i for i in has if r["islands"][i]["t2_median_over_island_median"] > 1]
        # migrants: replay of evaluate events to bound the top-2 emigrants at every migration
        ev = [json.loads(l) for l in open(RUNS / run / "events.jsonl")]
        prom = [e for e in ev if e["kind"] == "promote"]
        mig = [e for e in ev if e["kind"] == "migrate"]
        evals = [e for e in ev if e["kind"] == "evaluate"]
        gens = [json.loads(l) for l in open(RUNS / run / "generations.jsonl")]
        gens = [x for x in gens if "generation" in x]
        mrep = []
        for m in mig:
            G = m["gen"]
            T = max([p["tier2_fitness"] for p in prom if p["gen"] < G] or [0.0])
            per = {}
            for isl in ISLANDS:
                # latest generation record of this island before the migration: N cells, S = qd_score (sum of fitness)
                last = [x for x in gens if x["island"] == isl and x["generation"] < G][-1]
                N, S, b = last["filled"], last["qd_score"], min(1.0, m["islands"][isl]["best"])
                # S <= k*b + (N-k)*T  =>  at least k elites have fitness > T
                kmin = max(0, math.ceil((S - N * T) / (b - T) - 1e-9)) if b > T else 0
                per[isl] = {"last_visit_gen": last["generation"], "cells": N, "qd_score": S,
                            "best": b, "k_min_above_T": int(kmin)}
            mrep.append({"gen": G, "max_tier2_fitness_so_far": T,
                         "promotions_so_far": sum(p["gen"] < G for p in prom), "per_island": per,
                         "migrations": m["migrations"], "hybrids": m["hybrids"],
                         "all_islands_have_ge2_above_T": all(v["k_min_above_T"] >= 2 for v in per.values())})
        r["migration_replay"] = mrep
        r["promote_events"] = len(prom)
        r["promote_events_with_tier2_fraction_gt0"] = sum(1 for p in prom if p.get("tier2_fraction", 0) > 0)
        r["max_tier2_fitness_all"] = max(p["tier2_fitness"] for p in prom)
        r["migrant_evaluate_events"] = sum('migrant' in e.get("operators", []) for e in evals)
        r["migrant_evaluate_tiers"] = dict(collections.Counter(e["tier"] for e in evals if 'migrant' in e.get("operators", [])))
        res[run] = r
    dump("n4", res)
    for run, r in res.items():
        print(f"[{run}] islands with Tier-2 elites: {r['islands_with_tier2']}")
        print("  island        pool t2  p10_w  med_w  t2_med_w  ratio  share<=p10  t2_deciles  t2_fit  t1_fit  min_fit_rank")
        for isl, v in r["islands"].items():
            if v["tier2"]:
                print(f"  {isl:13s} {v['pool']:4d} {v['tier2']:2d}  {v['p10_w']:.3f}  {v['island_median_w']:.3f}  "
                      f"{v['t2_median_w']:.3f}    {v['t2_median_over_island_median']:.2f}   {v['t2_share_bottom_decile']:.2f}   "
                      f"{v['t2_deciles']}  {v['t2_median_fitness']:.3f} {v['t1_median_fitness']:.3f}  {v['t2_min_fitness_rank_among_representatives']}")
            else:
                print(f"  {isl:13s} {v['pool']:4d}  0")
        print("  median Tier-2 weight in bottom decile:", r["islands_median_t2_in_bottom_decile"])
        print("  median Tier-2 weight above island median:", r["islands_t2_median_above_island_median"])
        print(f"  promote events {r['promote_events']}, with a completed leg {r['promote_events_with_tier2_fraction_gt0']}, "
              f"max tier2_fitness {r['max_tier2_fitness_all']}, migrant evaluate events {r['migrant_evaluate_events']} tiers {r['migrant_evaluate_tiers']}")
        for m in r["migration_replay"]:
            print(f"  migration gen {m['gen']}: T={m['max_tier2_fitness_so_far']:.3f} promotions_before={m['promotions_so_far']} "
                  f"all islands >=2 above T: {m['all_islands_have_ge2_above_T']}  "
                  f"min k_min_above_T={min(v['k_min_above_T'] for v in m['per_island'].values())}")


# ---------------------------------------------------------------- N10
def n10():
    res = {}
    for run in ("arch48", "arch49", "arch50"):
        ev = [json.loads(l) for l in open(RUNS / run / "events.jsonl")]
        ppo = [e for e in ev if e["kind"] == "ppo"]
        keys = sorted({k for e in ppo for k in e})
        bytag = collections.defaultdict(list)
        for e in ppo:
            for t, (mu, sd, n) in e["reward_by_tag"].items():
                bytag[t].append((mu, sd, n, e["island"], e["gen"]))
        tags = {}
        for t, v in sorted(bytag.items()):
            mu = np.array([x[0] for x in v]); sd = np.array([x[1] for x in v]); n = np.array([x[2] for x in v])
            tags[t] = {"events": len(v), "trajectories": int(n.sum()),
                       "weighted_mean_terminal": float((mu * n).sum() / n.sum()),
                       "events_mean0_and_std0": float(np.mean((mu == 0) & (sd == 0))),
                       "events_mean_gt0": float(np.mean(mu > 0)),
                       "max_event_mean": float(mu.max()), "max_event_std": float(sd.max())}
        res[run] = {"ppo_events": len(ppo), "ppo_event_keys": keys,
                    "shaping_values": sorted({e["shaping"] for e in ppo}), "tags": tags}
    # grep inventory
    grep = {}
    for run in ("arch49", "arch50"):
        for f in ("generations.jsonl", "events.jsonl"):
            txt = open(RUNS / run / f).read()
            grep[f"{run}/{f}"] = {s: txt.count(s) for s in ("reward_by_tag", "shaping", "terminal", "shaping_return",
                                                              "return_by_tag", "terminal_return", "\"ppo\"")}
    res["grep_counts"] = grep
    dump("n10", res)
    for run in ("arch48", "arch49", "arch50"):
        r = res[run]
        print(f"[{run}] ppo events {r['ppo_events']}, shaping values {r['shaping_values']}")
        print("  keys:", r["ppo_event_keys"])
        for t, v in r["tags"].items():
            print(f"  {t:26s} events {v['events']:3d} traj {v['trajectories']:5d} mean_terminal {v['weighted_mean_terminal']:.4f} "
                  f"all-zero events {v['events_mean0_and_std0']:.3f} events>0 {v['events_mean_gt0']:.3f} max_mean {v['max_event_mean']:.4f}")
    print(json.dumps(grep, indent=1))


if __name__ == "__main__":
    t0 = time.time()
    for name in sys.argv[1:]:
        {"n2": n2, "n3": n3, "n4": n4, "n10": n10}[name]()
    print(f"wall {time.time() - t0:.1f}s")
