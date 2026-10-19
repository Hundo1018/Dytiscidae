"""Air L1 against the null: does an elite turn toward the commanded side over K fresh draws?

README.md in this directory is the frozen design.  Usage:

  run.py screen  --run runs/arch51 --out experiments/air_turn_bar/screen.json [--k 8] [--seed-base 2026101900]
  run.py repl    --run runs/arch51 --out experiments/air_turn_bar/repl.json --passers experiments/air_turn_bar/screen.json --seed-base 2026101910
  run.py report  --screen screen.json [--repl repl.json]
"""
from __future__ import annotations

import argparse
import collections
import copy
import gc
import json
import math
import sys
import time
from pathlib import Path

MEASURABLE = 4  # draws with pair_turn_factor > 0 needed for a verdict


def binom_sf(k, n, p=0.5):
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def measure(args, rows_path, indices=None):
    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites, plain_controller
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.viz.film import control_laws, run_provenance

    run = Path(args.run)
    ok, why = batchroll.usable()
    assert ok, f"batched evaluator unusable: {why}"
    cfg = run_provenance(run).get("config") or {}
    seg = float(cfg.get("segment_seconds") or 8.0)
    n_modes = int(cfg.get("n_modes") or 6)
    elites = load_elites(run)
    pick = list(range(len(elites))) if indices is None else [i for i in indices if i < len(elites)]
    groups = collections.defaultdict(list)
    for i in pick:
        m = elites[i].meta or {}
        groups[(int(m.get("gen") or 0), int(m.get("eval_seed") or 0))].append(i)
    keys = sorted(groups)[: args.max_groups or None]
    seeds = [args.seed_base + j for j in range(args.k)]
    print(f"{len(elites)} elites, measuring {sum(len(groups[k]) for k in keys)} in {len(keys)} groups; "
          f"K={args.k} seeds {seeds[0]}..{seeds[-1]}; segment_seconds {seg}", flush=True)
    spec = MissionSpec()
    t0 = time.time()
    fh = open(rows_path, "w")
    for gi, key in enumerate(keys):
        phenos, ctrls, keep, like = [], [], [], None
        for i in groups[key]:
            p = build(elites[i].genome)
            c, sh = plain_controller(run, elites[i], p, key[1])
            if c is None:
                continue
            like = like or sh
            phenos.append(p); ctrls.append(c); keep.append(i)
        if not keep:
            continue
        _, net = next(iter(control_laws(elites[keep[0]], run, like)))
        envs = [TriphibianEnv(p, seed=key[1]) for p in phenos]
        stills = [Controller(params=e.held_still_params(), policy=None, bases=c.bases)
                  for e, c in zip(envs, ctrls)]
        del envs
        # K copies of each machine, one fresh seed per copy: the batched path takes one seed per machine.
        P = [p for p in phenos for _ in seeds]
        S = [s for _ in phenos for s in seeds]
        for arm, cs, sh in (("elite", ctrls, net), ("still", stills, None)):
            C = [copy.deepcopy(c) for c in cs for _ in seeds]
            out = batchroll.evaluate_tier1_batch(P, spec=spec, controllers=C, segment_seconds=seg,
                                                 identify_axes=False, seed=S, shared=sh, n_modes=n_modes)
            for slot, i in enumerate(keep):
                m = elites[i].meta or {}
                for j, s in enumerate(seeds):
                    res = out[slot * len(seeds) + j]
                    a = res.segments.get("air")
                    mm = (a.measurements or {}) if a is not None else {}
                    fh.write(json.dumps({
                        "index": int(i), "arm": arm, "draw": j, "seed": s, "gen": key[0],
                        "island": m.get("island"), "plan": m.get("body_plan"),
                        "recorded_air": m.get("air"), "rungs": (m.get("rungs") or {}).get("air"),
                        "air": float(a.competence) if a is not None else None,
                        "bad_qacc": int(a.bad_qacc) if a is not None else None,
                        "failure": a.failure if a is not None else "no air segment",
                        "pair_turn": mm.get("pair_turn"), "pair_turn_mean": mm.get("pair_turn_mean"),
                        "pair_turn_factor": mm.get("pair_turn_factor"), "turn_response": mm.get("turn_response"),
                    }) + "\n")
            fh.flush()
            del out, C
        del P, S, phenos, ctrls, stills
        gc.collect()
        print(f"group {gi + 1}/{len(keys)} gen {key[0]} n={len(keep)} wall {time.time() - t0:.0f}s", flush=True)
    fh.close()
    return time.time() - t0


def aggregate(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["index"], r["arm"])].append(r)
    out = {}
    for (i, arm), rs in sorted(by.items()):
        meas = [r for r in rs if (r.get("pair_turn_factor") or 0.0) > 0.0 and r.get("pair_turn_mean") is not None]
        xs = [float(r["pair_turn_mean"]) for r in meas]
        n = len(xs); k = sum(x > 0 for x in xs)
        if n:
            mu = sum(xs) / n
            sd = (sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else float("nan")
        else:
            mu = sd = float("nan")
        p = binom_sf(k, n) if n else None
        verdict = ("unmeasured" if n < MEASURABLE else "pass" if p < 0.05 else "fail")
        out[f"{i}:{arm}"] = {"index": i, "arm": arm, "n_draws": len(rs), "measurable": n, "positive": k,
                             "p_sign": p, "mean": mu, "sd": sd, "verdict": verdict,
                             "max_abs_still": max((abs(x) for x in xs), default=0.0) if arm == "still" else None,
                             "island": rs[0]["island"], "plan": rs[0]["plan"], "recorded_air": rs[0]["recorded_air"],
                             "rung": rs[0]["rungs"], "bad_qacc": sum(r.get("bad_qacc") or 0 for r in rs)}
    return out


def summary(agg, label):
    el = [v for v in agg.values() if v["arm"] == "elite"]
    st = [v for v in agg.values() if v["arm"] == "still"]
    c = collections.Counter(v["verdict"] for v in el)
    meas = [v for v in el if v["verdict"] != "unmeasured"]
    exp = sum(binom_sf(math.ceil(v["measurable"] * 0.5) if False else min_k(v["measurable"]), v["measurable"])
              for v in meas)
    s = {"label": label, "elites": len(el), "verdicts": dict(c),
         "expected_passes_under_null": round(exp, 2),
         "still_max_abs_mean": max((v["max_abs_still"] or 0.0) for v in st) if st else None,
         "still_measurable": sum(v["measurable"] for v in st),
         "passes": [{k: v[k] for k in ("index", "island", "plan", "measurable", "positive", "p_sign", "mean", "sd", "recorded_air", "rung")}
                    for v in el if v["verdict"] == "pass"],
         "mean_of_means": (sum(v["mean"] for v in meas) / len(meas)) if meas else None,
         "share_positive_draws": (sum(v["positive"] for v in meas) / sum(v["measurable"] for v in meas)) if meas else None}
    return s


def min_k(n):
    """Smallest k with P(X >= k | n, 0.5) < 0.05."""
    for k in range(n + 1):
        if binom_sf(k, n) < 0.05:
            return k
    return n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["screen", "repl", "report"])
    ap.add_argument("--run", default="runs/arch51")
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=2026101900)
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--passers", help="screen.json whose passes are re-drawn")
    ap.add_argument("--screen"); ap.add_argument("--repl")
    args = ap.parse_args()
    if args.mode in ("screen", "repl"):
        out = Path(args.out); rows_path = out.with_suffix(".rows.jsonl")
        idx = None
        if args.mode == "repl":
            idx = [p["index"] for p in json.load(open(args.passers))["summary"]["passes"]]
            print(f"replicating {len(idx)} screen passers", flush=True)
            if not idx:
                out.write_text(json.dumps({"summary": {"label": "repl", "elites": 0, "passes": []}}, indent=1)); return
        wall = measure(args, rows_path, idx)
        rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
        agg = aggregate(rows)
        res = {"run": args.run, "k": args.k, "seed_base": args.seed_base, "wall_s": wall,
               "summary": summary(agg, args.mode), "per_elite": agg}
        out.write_text(json.dumps(res, indent=1))
        print(json.dumps(res["summary"], indent=1))
    else:
        s = json.load(open(args.screen))["summary"]; print("screen", json.dumps(s, indent=1))
        if args.repl:
            r = json.load(open(args.repl))["summary"]; print("repl", json.dumps(r, indent=1))
            both = {p["index"] for p in s["passes"]} & {p["index"] for p in r["passes"]}
            print("replicated passes:", sorted(both))


if __name__ == "__main__":
    main()
