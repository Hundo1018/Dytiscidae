"""N1 -- still-twin fitness probe (ARCH51_SPEC §B N1; ROADMAP 2026-10-10 item N1).

For every elite of a finished run, the *same body* is scored twice through the
batched Tier-1 evaluator the search wrote its archive with
(``batchroll.evaluate_tier1_batch``), grouped by (gen, eval_seed), under the
network that scored it:

  elite  its own policy + stored bases + the scoring network
  still  ``TriphibianEnv.held_still_params()``, no policy, no shared network

and each result goes through the search's own scalar, ``loop._score_candidate``
(``commit=False``: nothing is written), against the final archive / curriculum /
judge restored from ``search_state.pkl``.  Recorded per arm: fit, base, isl_q,
cur_q, mis_q, w, stage, objectives, competences; and for the twin its dominance
relation to the elite's own cell front and ``Archive.would_add`` at its own cell.

Must run against commit 5cc9db9 (before the competence floor changes):

  cd <main> && PYTHONPATH=<n1-tree> MUJOCO_GL=disable \
     DYTISCIDAE_KERNEL_DIR=<main>/mojo/build .venv/bin/python \
     experiments/still_twin_fitness/run.py --run runs/arch49 \
     --out experiments/still_twin_fitness/results_arch49.json

  --limit N        first N elites only (smoke)
  --from-rows F    re-aggregate a rows.jsonl without simulating
"""
import argparse
import collections
import gc
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MEDIA = ("air", "water", "land")
SPECIALISTS = ("air", "water", "land")
TOL = 0.005
# Frozen thresholds (ROADMAP N1 / ARCH51_SPEC §B N1); not to be edited.
PRED_MEDIAN_FIT = 0.80
PRED_NONDOM_SHARE = 0.60
FALSIFY_SPECIALIST_MEDIAN = 0.50
FALSIFY_NONDOM_SHARE = 0.30
REVISED_DOMINATED_SHARE = 0.70


def _q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def aggregate(rows):
    """Per-island table, pooled row, and the frozen-threshold verdict inputs."""
    by = collections.defaultdict(lambda: {"elite": {}, "still": {}})
    for r in rows:
        by[r["index"]][r["arm"]] = r
    pairs = [(v["elite"], v["still"]) for v in by.values() if "elite" in v and "still" in v]
    islands = sorted({e["island"] for e, _ in pairs})

    def stats(sub):
        n = len(sub)
        tf = np.array([s["fit"] for _, s in sub], float)
        ef = np.array([e["fit"] for e, _ in sub], float)
        nd = np.array([s["nondom"] for _, s in sub], bool)
        dm = np.array([s["dominates"] for _, s in sub], bool)
        wa = np.array([s["would_add"] != "rejected" for _, s in sub], bool)
        rep = {md: [abs(e["comp"].get(md, 0.0) - float(e["recorded"][md])) <= TOL
                    for e, _ in sub if e.get("recorded", {}).get(md) is not None] for md in MEDIA}
        return {
            "n": n,
            "twin_fit_median": _q(tf, 50), "twin_fit_q25": _q(tf, 25), "twin_fit_q75": _q(tf, 75),
            "twin_fit_mean": float(tf.mean()) if n else float("nan"),
            "twin_fit_sd": float(tf.std(ddof=1)) if n > 1 else float("nan"),
            "twin_isl_q_median": _q([s["isl_q"] for _, s in sub], 50),
            "twin_mis_q_median": _q([s["mis_q"] for _, s in sub], 50),
            "twin_cur_q_median": _q([s["cur_q"] for _, s in sub], 50),
            "share_nondom_or_dominating": float(nd.mean()) if n else float("nan"),
            "share_dominating": float(dm.mean()) if n else float("nan"),
            "share_dominated": float((~nd).mean()) if n else float("nan"),
            "share_would_add": float(wa.mean()) if n else float("nan"),
            "elite_fit_median": _q(ef, 50), "elite_fit_q25": _q(ef, 25), "elite_fit_q75": _q(ef, 75),
            "elite_recorded_fit_median": _q([e["recorded_fit"] for e, _ in sub
                                              if e.get("recorded_fit") is not None], 50),
            "twin_over_elite_fit_median": _q(tf / np.maximum(ef, 1e-9), 50),
            "reproduction": {md: [int(sum(v)), len(v)] for md, v in rep.items()},
        }

    per = {isl: stats([p for p in pairs if p[0]["island"] == isl]) for isl in islands}
    pooled = stats(pairs)
    spec = [per[i] for i in SPECIALISTS if i in per]
    verdict = {
        "median_fit_ge_0.80_every_island": bool(all(per[i]["twin_fit_median"] >= PRED_MEDIAN_FIT for i in islands)),
        "nondom_share_pooled": pooled["share_nondom_or_dominating"],
        "nondom_share_ge_0.60": bool(pooled["share_nondom_or_dominating"] >= PRED_NONDOM_SHARE),
        "specialist_median_fit": {i: per[i]["twin_fit_median"] for i in SPECIALISTS if i in per},
        "falsified_by_specialist_median_lt_0.5": bool(any(s["twin_fit_median"] < FALSIFY_SPECIALIST_MEDIAN for s in spec)),
        "falsified_by_nondom_share_lt_0.30": bool(pooled["share_nondom_or_dominating"] < FALSIFY_NONDOM_SHARE),
        "dominated_share_pooled": pooled["share_dominated"],
        "revised_T1_back_if_dominated_ge_0.70": bool(pooled["share_dominated"] >= REVISED_DOMINATED_SHARE),
    }
    pred_holds = verdict["median_fit_ge_0.80_every_island"] and verdict["nondom_share_ge_0.60"]
    fals = verdict["falsified_by_specialist_median_lt_0.5"] or verdict["falsified_by_nondom_share_lt_0.30"]
    verdict["prediction_holds"] = bool(pred_holds)
    verdict["falsified"] = bool(fals)
    verdict["outcome"] = ("REFUTED" if fals else "CONFIRMED" if pred_holds
                          else "NEITHER (not confirmed, not falsified)")
    return per, pooled, verdict


def fmt_table(per, pooled):
    f = lambda x: f"{x:.3f}"
    hdr = ("| island | n | twin fit median [IQR] | twin isl_q | twin mis_q | non-dom or dominating | dominated "
           "| would_add != rejected | elite fit median (final windows) | repro air/water/land (<=0.005) |")
    lines = [hdr, "|" + "---|" * 10]
    for name, s in list(per.items()) + [("ALL", pooled)]:
        rp = s["reproduction"]
        lines.append(
            f"| {name} | {s['n']} | {f(s['twin_fit_median'])} [{f(s['twin_fit_q25'])}, {f(s['twin_fit_q75'])}] "
            f"| {f(s['twin_isl_q_median'])} | {f(s['twin_mis_q_median'])} "
            f"| {f(s['share_nondom_or_dominating'])} | {f(s['share_dominated'])} | {f(s['share_would_add'])} "
            f"| {f(s['elite_fit_median'])} | " + " / ".join(f"{rp[m][0]}/{rp[m][1]}" for m in MEDIA) + " |")
    return "\n".join(lines)


def measure(args, rows_path):
    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites, plain_controller
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.evolution import loop
    from dytiscidae.evolution.archive import Archive
    from dytiscidae.evolution.loop import SearchConfig
    from dytiscidae.ops.run import load_run_archive
    from dytiscidae.viz.film import control_laws, run_provenance

    run = Path(args.run)
    ok, why = batchroll.usable()
    assert ok, f"batched evaluator unusable: {why}"
    prov = run_provenance(run)
    rcfg = prov.get("config") or {}
    seg = float(rcfg.get("segment_seconds") or 8.0)
    n_modes = int(rcfg.get("n_modes") or 6)
    cfg = SearchConfig(**{k: v for k, v in rcfg.items() if k in SearchConfig.__dataclass_fields__})
    D = pickle.load(open(run / "search_state.pkl", "rb"))
    print(f"search_state: generation {D['generation']}, evaluated {D['evaluated']}; "
          f"mission_weight {cfg.mission_weight}; segment_seconds {seg}; n_modes {n_modes}", flush=True)

    elites = load_elites(run)
    groups = collections.defaultdict(list)
    for i, e in enumerate(elites):
        m = e.meta or {}
        groups[(int(m.get("gen") or 0), int(m.get("eval_seed") or 0))].append(i)
    keys = sorted(groups)
    order = [(k, i) for k in keys for i in groups[k]]
    if args.limit:
        order = order[: args.limit]
        groups = collections.defaultdict(list)
        for k, i in order:
            groups[k].append(i)
        keys = sorted(groups)
    print(f"{len(elites)} elites in the merged archive; measuring {len(order)} in {len(keys)} "
          f"(gen, seed) groups", flush=True)

    archives = {}

    def arch_of(island):
        if island not in archives:
            archives[island] = load_run_archive(run, island)[0]
        return archives[island]

    spec = MissionSpec()
    t0, skipped = time.time(), []
    fh = open(rows_path, "w")
    for gi, key in enumerate(keys):
        phenos, ctrls, keep, like = [], [], [], None
        for i in groups[key]:
            p = build(elites[i].genome)
            c, sh = plain_controller(run, elites[i], p, key[1])
            if c is None:
                skipped.append(int(i))
                continue
            like = like or sh
            phenos.append(p)
            ctrls.append(c)
            keep.append(i)
        if not keep:
            continue
        _, net = next(iter(control_laws(elites[keep[0]], run, like)))
        envs = [TriphibianEnv(p, seed=key[1]) for p in phenos]
        stills = [Controller(params=e.held_still_params(), policy=None, bases=c.bases)
                  for e, c in zip(envs, ctrls)]
        del envs
        out = {}
        for arm, cs, sh in (("elite", ctrls, net), ("still", stills, None)):
            out[arm] = batchroll.evaluate_tier1_batch(
                phenos, spec=spec, controllers=cs, segment_seconds=seg,
                identify_axes=False, seed=key[1], shared=sh, n_modes=n_modes)
        for slot, i in enumerate(keep):
            el = elites[i]
            m = el.meta or {}
            island = m.get("island") or "generalist"
            arch = arch_of(island)
            ns = SimpleNamespace(config=cfg, descriptors=D["descriptors"], judge=D["judge"],
                                 critic=D["critic"], island=island, archive=arch,
                                 curriculum=D["curricula"][island])
            ecell = arch.cell_of(el.descriptor)
            parent = SimpleNamespace(cell=ecell)
            F = arch.fronts.get(ecell) or []
            for arm in ("elite", "still"):
                res = out[arm][slot]
                sc = loop._score_candidate(ns, phenos[slot], res, parent, commit=False)
                obj = np.asarray(sc["obj"], float)
                rec = {"index": int(i), "arm": arm, "gen": key[0], "eval_seed": key[1],
                       "island": island, "plan": m.get("body_plan") or "?",
                       "n_rotors": int(m.get("n_rotors") or 0),
                       "fit": float(sc["fit"]), "base": float(sc["base"]),
                       "isl_q": float(sc["isl_q"]), "cur_q": float(sc["cur_q"]),
                       "mis_q": float(sc["mis_q"]), "w": float(sc["w"]),
                       "discount": float(sc["discount"]),
                       "stage": int(sc["sr"].stage), "obj": obj.tolist(),
                       "mission_fraction": float(res.mission_fraction),
                       "cell": list(sc["cell"]), "elite_cell": list(ecell),
                       "comp": {d: float(res.segments[d].competence) for d in MEDIA if d in res.segments},
                       "bad_qacc": {d: int(res.segments[d].bad_qacc) for d in MEDIA if d in res.segments}}
                if arm == "elite":
                    rec["recorded"] = {d: m.get(d) for d in MEDIA}
                    rec["recorded_fit"] = float(el.fitness)
                    rec["front_size"] = len(F)
                else:
                    rec["nondom"] = bool(not any(Archive._dominates(e.objectives, obj) for e in F))
                    rec["dominates"] = bool(any(Archive._dominates(obj, e.objectives) for e in F))
                    rec["would_add"] = arch.would_add(sc["fit"], sc["bd"], obj)
                    rec["same_cell_as_elite"] = bool(tuple(sc["cell"]) == tuple(ecell))
                fh.write(json.dumps(rec) + "\n")
        fh.flush()
        del out, phenos, ctrls, stills
        gc.collect()
        print(f"group {gi + 1}/{len(keys)} gen {key[0]} n={len(keep)} wall {time.time() - t0:.0f}s",
              flush=True)
    fh.close()
    return time.time() - t0, skipped, len(elites), rcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--from-rows")
    args = ap.parse_args()
    out = Path(args.out)
    rows_path = Path(args.from_rows) if args.from_rows else Path(str(out) + ".rows.jsonl")
    if args.from_rows:
        wall, skipped, n_archive, rcfg = float("nan"), [], 0, {}
    else:
        wall, skipped, n_archive, rcfg = measure(args, rows_path)
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    per, pooled, verdict = aggregate(rows)
    md = fmt_table(per, pooled)
    tree = Path(sys.modules["dytiscidae"].__file__).resolve().parents[1]
    try:
        sha = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    meta = {"run": args.run, "package_tree": str(tree), "package_commit": sha,
            "n_elites_in_archive": n_archive, "n_measured": pooled["n"], "skipped_no_controller": skipped,
            "wall_s": wall, "limit": args.limit, "verdict": verdict, "pooled": pooled, "per_island": per}
    out.write_text(json.dumps(meta, indent=1))
    Path(str(out).replace(".json", "") + "_table.md").write_text(md + "\n")
    print(json.dumps({"verdict": verdict, "package_tree": str(tree), "package_commit": sha,
                      "skipped": skipped, "wall_s": wall}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
