#!/usr/bin/env python3
"""N8 -- random sampling against the archive (ARCH51_SPEC §B N8 and §D' item 6; ROADMAP 2026-10-10 item N8; T5).

The question: has selection contributed anything yet?  300 bodies drawn from the
run's own seeding distribution, scored through the search's own scorer at their
own draws, are binned on arch49's final descriptor axes and compared, island by
island, with what arch49's archives held at the same number of evaluations.

Frozen prediction (verbatim from ROADMAP §"2026-10-10" item N8 and ARCH51_SPEC
§B N8; not to be edited):

    *Prediction:* coverage and summed competence within 15% of arch49's gen-37
    island archives (the same evaluation count per island). *Falsified* if
    arch49 leads by > 30%.

ROADMAP wording after the §D' correction: "within 15% of arch49's island
archives at the snapshot nearest 300 evaluations per island (about gen 150;
'gen-37' in the first draft was an arithmetic slip), verdict per island".

The comparison base (ARCH51_SPEC §D' item 6):
  * The literal gen-37 comparison is NOT COMPARABLE: arch49's islands reach about
    300 evaluate events near gen 150, not gen 37 (water 587, land 604, air 611
    evaluate events over gens 0-299; 90-94 per island by gen 37), snapshots exist
    only at gens 0, 50, 100, 150, 200, 250, and evaluate events carry no
    ``features``, so a gen-37 archive cannot be rebuilt.
  * The intent is run against the snapshot nearest 300 evaluations per island:
    the generation is computed from events.jsonl (``kind == "evaluate"``, per
    ``island``, cumulative to each snapshot generation), which should be gen 150.
    Gen 50 is given as context only.
  * Verdict per island, with a pooled row (sum over the eight islands of filled
    cells and of summed competence, each side) as a secondary read.

The prior: the run's own seeding distribution (loop.py:2516-2517):
``seed_population(rng, n_reference_seeds)`` plus ``random_genome(rng)`` x
``n_random_seeds``, ``rng = default_rng(20261010)``, taken from arch49's
provenance (12:8; the spec text writes 20:8, which is ``SearchConfig``'s default,
not what arch49 ran).  Drawn in rounds of that shape until 300 bodies pass
Tier-0; round r takes ``seed_population(rng, n_ref * (r + 1))[n_ref * r:]`` so the
clean archetypes appear once and later rounds are the perturbed ones.  Tier-0
rejects are counted (arch49's evaluate events exclude them too).

Scoring: ``loop.evaluate_candidates(genomes, cfg, identify=True,
spec=MissionSpec(), seeds=<rng.integers(1<<30) each>, shared=<arch49's scoring
network at the comparison generation>, pool=None)`` in batches of 16, ``cfg``
from arch49's provenance.  Fitness is not used, so N8 does not depend on N5.

Binning: ``bd = D["descriptors"].project(episode_features(result, pheno))`` with
``D = search_state.pkl``; per island the axes of ``archive_<island>.pkl``;
``Archive(axes).cell_of(bd)``.  The arch49 side: the elites of
``snapshots/gen<G>_<island>.json`` re-projected from ``meta["features"]`` with the
same ``D["descriptors"]`` into the same axes.  Per cell the island's own
competence, the max kept:
  air, water, land: that medium;  amphibian, aerial_diver, land_air: min of the
  two;  triphibian: triphibian_score([air, water, land]);  generalist:
  mission_fraction.
Coverage = filled cells / 625; summed competence = the sum of the kept values.

Decision rule (the frozen text, with the denominators written out): for an
island, "within 15%" is |random / arch49 - 1| <= 0.15 on BOTH coverage and summed
competence (both zero counts as within); "arch49 leads by > 30%" is
arch49 / random - 1 > 0.30 on EITHER (arch49 > 0 and random = 0 counts as a
lead).  CONFIRMED if within on both, REFUTED if the lead holds on either,
otherwise NEITHER.  A 95% bootstrap interval over the 300 random bodies
(random side only; the arch49 snapshot is a fixed set) is reported for every
random-side number.

    PYTHONPATH=. MUJOCO_GL=disable DYTISCIDAE_KERNEL_DIR=<main>/mojo/build \\
        .venv/bin/python experiments/random_sampling_control/run.py --run runs/arch49

  --limit N      smoke: draw until N bodies pass Tier-0 and score only those
                 (writes *_smoke files, never N8_result.md)
  --cpu          the numpy path: sets ``batchroll.AVAILABLE = False`` so
                 ``evaluate_candidates`` falls back to ``evaluate_candidate`` per
                 body.  That path does NOT apply ``shared`` and does not run the
                 controller refinement; it is not the search's scorer.  For
                 smoke tests while the GPU is busy; never a verdict.
  --from-rows F  re-aggregate a rows.jsonl (still reads the run's snapshots)
  --selftest     the pure functions, no simulator
"""
import argparse
import collections
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MEDIA = ("air", "water", "land")
ISLANDS = ("air", "water", "land", "amphibian", "aerial_diver", "land_air", "triphibian", "generalist")
SNAPSHOT_GENS = (0, 50, 100, 150, 200, 250)
TARGET_EVALS = 300
CAPACITY = 625

# Frozen thresholds (ROADMAP N8 / ARCH51_SPEC §B N8); not to be edited.
WITHIN = 0.15
LEAD = 0.30

PREDICTION = (
    "*Prediction:* coverage and summed competence within 15% of arch49's gen-37 island archives "
    "(the same evaluation count per island). *Falsified* if arch49 leads by > 30%.")


# --------------------------------------------------------------------------
# Pure functions (no simulator)
# --------------------------------------------------------------------------


def island_value(island, comps, mission_fraction):
    """The island's own competence for one design (ARCH51_SPEC §B N8).

    ``comps``: ``{"air", "water", "land"}`` competences (missing = 0).
    """
    from dytiscidae.evolution.islands import ISLANDS as SPEC, triphibian_score
    c = {m: float(comps.get(m) or 0.0) for m in MEDIA}
    if island == "generalist":
        return float(mission_fraction or 0.0)
    doms = SPEC[island]["domains"]
    if island == "triphibian":
        return float(triphibian_score([c["air"], c["water"], c["land"]]))
    if len(doms) == 1:
        return c[doms[0]]
    return float(min(c[d] for d in doms))


def comparison_gens(events, target=TARGET_EVALS, snapshot_gens=SNAPSHOT_GENS):
    """``{island: {"gen": nearest snapshot gen, "evals": cumulative events there, "by_gen": {...}}}``.

    Cumulative evaluate events with ``gen <= snapshot gen``, per island.
    """
    per = collections.defaultdict(list)
    for e in events:
        if e.get("kind") == "evaluate" and e.get("island") is not None:
            per[e["island"]].append(int(e["gen"]))
    out = {}
    for isl, gens in per.items():
        g = np.asarray(gens)
        cum = {int(s): int((g <= s).sum()) for s in snapshot_gens}
        best = min(snapshot_gens, key=lambda s: (abs(cum[s] - target), s))
        out[isl] = {"gen": int(best), "evals": cum[best], "by_gen": cum, "total": int(len(g))}
    return out


def bin_max(cells, values):
    """``{cell: max value}``."""
    out = {}
    for c, v in zip(cells, values):
        c = tuple(int(x) for x in c)
        if c not in out or v > out[c]:
            out[c] = float(v)
    return out


def summarize(binned):
    return {"filled": len(binned), "coverage": len(binned) / CAPACITY, "summed": float(sum(binned.values()))}


def compare(rand, arch):
    """One island's (or the pool's) verdict inputs: ratios and the two frozen tests."""
    def ratio(r, a):
        if a == 0:
            return float("nan") if r == 0 else float("inf")
        return r / a
    out = {}
    within_all, lead_any = True, False
    for k in ("coverage", "summed"):
        r, a = rand[k], arch[k]
        rr = ratio(r, a)
        within = (r == 0 and a == 0) or (a > 0 and abs(r / a - 1.0) <= WITHIN)
        lead = (a > 0 and (r == 0 or a / r - 1.0 > LEAD))
        out[k] = {"random": r, "arch49": a, "random_over_arch49": rr, "arch49_over_random_minus_1":
                  (float("inf") if (r == 0 and a > 0) else (a / r - 1.0 if r > 0 else 0.0)),
                  "within_15": bool(within), "arch49_leads_gt_30": bool(lead)}
        within_all &= within
        lead_any |= lead
    out["outcome"] = "REFUTED" if lead_any else "CONFIRMED" if within_all else "NEITHER (not confirmed, not falsified)"
    return out


def selftest():
    assert island_value("air", {"air": 0.4, "water": 0.1}, 0.9) == 0.4
    assert island_value("amphibian", {"water": 0.3, "land": 0.2}, 0.0) == 0.2
    assert island_value("land_air", {"air": 0.7, "land": 0.5}, 0.0) == 0.5
    assert island_value("generalist", {"air": 1.0}, 0.25) == 0.25
    from dytiscidae.evolution.islands import triphibian_score
    assert island_value("triphibian", {"air": 0.1, "water": 0.2, "land": 0.3}, 0.0) == triphibian_score([0.1, 0.2, 0.3])
    ev = ([{"kind": "evaluate", "island": "air", "gen": g} for g in (0, 10, 40, 60, 90, 120, 149, 149, 160)]
          + [{"kind": "tier0_reject", "island": "air", "gen": 3}])
    cg = comparison_gens(ev, target=7)
    assert cg["air"]["gen"] == 150 and cg["air"]["evals"] == 8, cg      # 0,10,40,60,90,120,149,149 <= 150
    b = bin_max([(0, 1), (0, 1), (2, 2)], [0.2, 0.5, 0.1])
    assert b == {(0, 1): 0.5, (2, 2): 0.1} and abs(summarize(b)["summed"] - 0.6) < 1e-12
    mk = lambda cov, s: {"filled": 0, "coverage": cov, "summed": s}
    assert compare(mk(0.10, 10.0), mk(0.11, 10.5))["outcome"] == "CONFIRMED"
    assert compare(mk(0.07, 10.0), mk(0.11, 10.5))["outcome"] == "REFUTED"           # 0.11/0.07 - 1 = 0.57
    assert compare(mk(0.10, 10.0), mk(0.125, 10.5))["outcome"].startswith("NEITHER")  # coverage 1.25x: not within, not > 30%
    assert compare(mk(0.0, 0.0), mk(0.1, 5.0))["outcome"] == "REFUTED"
    assert compare(mk(0.0, 0.0), mk(0.0, 0.0))["outcome"] == "CONFIRMED"
    print("random_sampling_control selftest passed")


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


def draw_bodies(rng, n_pass, cfg, n_ref, n_rand, spec):
    """Genomes that pass Tier-0, in draw order, and the reject accounting."""
    from dytiscidae.core.bodyplans import seed_population
    from dytiscidae.core.genome import random_genome
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import evaluate_tier0

    kept, drawn, rejects = [], 0, collections.Counter()
    rnd = 0
    while len(kept) < n_pass:
        refs = seed_population(rng, n_ref * (rnd + 1))[n_ref * rnd:]
        batch = list(refs) + [random_genome(rng) for _ in range(n_rand)]
        for k, g in enumerate(batch):
            if len(kept) >= n_pass:
                break
            drawn += 1
            kind = "reference" if k < len(refs) else "random"
            try:
                pheno = build(g)
                t0 = evaluate_tier0(pheno, spec)
            except Exception as exc:  # noqa: BLE001
                rejects[f"{kind}:build_error"] += 1
                continue
            if pheno.report.gate_margin < cfg.tier0_gate or t0.mission_fraction <= 0.0:
                rejects[f"{kind}:tier0"] += 1
                continue
            g.genome_id = f"rs{len(kept)}"
            kept.append((g, kind))
        rnd += 1
    return kept, drawn, dict(rejects), rnd


def measure(args, rows_path):
    from dytiscidae.control.cpg import TWIST_DIM
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.evolution import loop
    from dytiscidae.evolution.descriptors import episode_features
    from dytiscidae.evolution.loop import SearchConfig
    from dytiscidae.learning import ppo
    from dytiscidae.viz.film import _load_network, run_provenance

    run = Path(args.run)
    rcfg = run_provenance(run).get("config") or {}
    cfg = SearchConfig(**{k: v for k, v in rcfg.items() if k in SearchConfig.__dataclass_fields__})
    n_ref = args.n_ref if args.n_ref is not None else int(cfg.n_reference_seeds)
    n_rand = args.n_rand if args.n_rand is not None else int(cfg.n_random_seeds)
    events = [json.loads(ln) for ln in (run / "events.jsonl").read_text().splitlines() if ln.strip()]
    cg = comparison_gens(events)
    gens = sorted({v["gen"] for v in cg.values()})
    net_gen = max(set(v["gen"] for v in cg.values()), key=[v["gen"] for v in cg.values()].count)
    net_path = run / "scoring_networks" / f"gen{net_gen:05d}.npz"
    template = ppo.SharedPolicy(TriphibianEnv.OBS_DIM, TWIST_DIM + (1 if cfg.gait_gain else 0),
                                hidden=cfg.shared_hidden)
    net = _load_network(net_path, template)
    print(f"comparison gens per island: { {i: cg[i]['gen'] for i in ISLANDS} }; scoring network gen {net_gen} "
          f"({'loaded' if net is not None else 'MISSING'}); seeding ratio {n_ref}:{n_rand}", flush=True)

    if args.cpu:
        batchroll.AVAILABLE = False
        print("--cpu: batchroll.AVAILABLE = False; evaluate_candidates takes the per-body numpy fallback "
              "(no shared network, no controller refinement)", flush=True)
    else:
        ok, why = batchroll.usable()
        assert ok, f"batched evaluator unusable: {why}"

    spec = MissionSpec()
    n_pass = args.limit or args.n
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    bodies, drawn, rejects, rounds = draw_bodies(rng, n_pass, cfg, n_ref, n_rand, spec)
    seeds = [int(rng.integers(1 << 30)) for _ in bodies]
    t_draw = time.time() - t0
    print(f"drew {drawn} genomes in {rounds} rounds, {len(bodies)} passed Tier-0, rejects {rejects}; "
          f"draw+Tier-0 {t_draw:.0f}s", flush=True)

    fh = open(rows_path, "w")
    fh.write(json.dumps({"kind": "meta", "drawn": drawn, "rounds": rounds, "rejects": rejects, "n_ref": n_ref,
                         "n_rand": n_rand, "seed": args.seed, "comparison": cg, "net_gen": net_gen,
                         "net_loaded": net is not None, "cpu": bool(args.cpu), "cfg_seed": cfg.seed,
                         "wall_draw_s": t_draw}) + "\n")
    t_eval, per_batch = time.time(), []
    for lo in range(0, len(bodies), args.batch):
        chunk = bodies[lo: lo + args.batch]
        tb = time.time()
        out = loop.evaluate_candidates([g for g, _ in chunk], cfg, identify=True, spec=spec,
                                       seeds=seeds[lo: lo + args.batch], shared=net, pool=None)
        for (g, kind), seed, got in zip(chunk, seeds[lo: lo + args.batch], out):
            if got is None or got[2] is None:
                fh.write(json.dumps({"kind": "body", "id": g.genome_id, "origin": kind, "seed": seed,
                                     "dropped": "no tier-1 result (rejected at the evaluator's Tier-0)"}) + "\n")
                continue
            pheno, res, _ctrl = got
            segs = getattr(res, "segments", {}) or {}
            comps = {m: float(segs[m].competence) for m in MEDIA if m in segs}
            fh.write(json.dumps({
                "kind": "body", "id": g.genome_id, "origin": kind, "seed": seed,
                "plan": getattr(pheno.genome, "body_plan", None) or "?",
                "comps": comps, "mission_fraction": float(res.mission_fraction),
                "features": [float(x) for x in episode_features(res, pheno)],
                "bad_qacc": {m: int(segs[m].bad_qacc) for m in MEDIA if m in segs}}) + "\n")
        fh.flush()
        per_batch.append(time.time() - tb)
        print(f"batch {lo // args.batch + 1}/{-(-len(bodies) // args.batch)} n={len(chunk)} {per_batch[-1]:.1f}s "
              f"(total {time.time() - t_eval:.0f}s)", flush=True)
    fh.close()
    return time.time() - t0, t_draw, per_batch


def arch_side(run, island, gen, D):
    """The snapshot's elites re-projected into the final axes -> binned dict + counts."""
    from dytiscidae.evolution.archive import Archive

    snap = json.loads((run / "snapshots" / f"gen{gen:04d}_{island}.json").read_text())
    axes = Archive.load(run / f"archive_{island}.pkl").axes
    arch = Archive(axes)
    cells, vals, n = [], [], 0
    for e in snap["elites"]:
        m = e.get("meta") or {}
        f = m.get("features")
        if not isinstance(f, (list, tuple)) or len(f) == 0:
            continue
        n += 1
        cells.append(arch.cell_of(D["descriptors"].project(np.asarray(f, float))))
        vals.append(island_value(island, {k: m.get(k) for k in MEDIA}, m.get("mission_fraction")))
    return bin_max(cells, vals), n, len(snap["elites"])


def random_side(run, island, bodies, D):
    from dytiscidae.evolution.archive import Archive
    axes = Archive.load(run / f"archive_{island}.pkl").axes
    arch = Archive(axes)
    cells = [arch.cell_of(D["descriptors"].project(np.asarray(b["features"], float))) for b in bodies]
    vals = [island_value(island, b["comps"], b["mission_fraction"]) for b in bodies]
    return cells, vals


def bootstrap(cells, vals, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(vals)
    cov, tot = [], []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        s = summarize(bin_max([cells[i] for i in idx], [vals[i] for i in idx]))
        cov.append(s["coverage"])
        tot.append(s["summed"])
    return {"coverage": [float(np.percentile(cov, 2.5)), float(np.percentile(cov, 97.5))],
            "summed": [float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))]}


def aggregate(run, rows):
    run = Path(run)
    meta = next(r for r in rows if r["kind"] == "meta")
    bodies = [r for r in rows if r["kind"] == "body" and "comps" in r]
    dropped = [r for r in rows if r["kind"] == "body" and "comps" not in r]
    # arch49's own checkpoint file, written by this project's search: trusted input.
    D = pickle.load(open(run / "search_state.pkl", "rb"))
    cg = meta["comparison"]
    per, pool_r, pool_a = {}, {"filled": 0, "summed": 0.0}, {"filled": 0, "summed": 0.0}
    for isl in ISLANDS:
        g = cg[isl]["gen"]
        a_bin, a_n, a_total = arch_side(run, isl, g, D)
        cells, vals = random_side(run, isl, bodies, D)
        r_bin = bin_max(cells, vals)
        a_sum, r_sum = summarize(a_bin), summarize(r_bin)
        ctx_gen = 50
        c_bin, _, _ = arch_side(run, isl, ctx_gen, D)
        per[isl] = {"compare_gen": g, "arch49_evals": cg[isl]["evals"], "arch49_snapshot_elites": a_total,
                    "arch49_elites_with_features": a_n, "random_n": len(bodies),
                    "arch49": a_sum, "random": r_sum, "random_ci95": bootstrap(cells, vals),
                    "context_gen50": {"evals": cg[isl]["by_gen"]["50"],
                                      **summarize(c_bin)},
                    "random_nonzero_cells": int(sum(1 for v in r_bin.values() if v > 0)),
                    "arch49_nonzero_cells": int(sum(1 for v in a_bin.values() if v > 0)),
                    "verdict": compare(r_sum, a_sum)}
        for pool, s in ((pool_r, r_sum), (pool_a, a_sum)):
            pool["filled"] += s["filled"]
            pool["summed"] += s["summed"]
    pr = {"filled": pool_r["filled"], "coverage": pool_r["filled"] / (CAPACITY * len(ISLANDS)), "summed": pool_r["summed"]}
    pa = {"filled": pool_a["filled"], "coverage": pool_a["filled"] / (CAPACITY * len(ISLANDS)), "summed": pool_a["summed"]}
    pooled = {"arch49": pa, "random": pr, "verdict": compare(pr, pa)}
    counts = collections.Counter(v["verdict"]["outcome"] for v in per.values())
    plans = collections.Counter(b.get("plan") for b in bodies)
    return {"meta": meta, "n_scored": len(bodies), "dropped": [{"id": d["id"], "why": d["dropped"]} for d in dropped],
            "per_island": per, "pooled": pooled, "verdict_counts": dict(counts),
            "origins": dict(collections.Counter(b["origin"] for b in bodies)), "plans": dict(plans),
            "bad_qacc_bodies": int(sum(1 for b in bodies if any(v > 0 for v in (b.get("bad_qacc") or {}).values())))}


# --------------------------------------------------------------------------
# Report, in the shape of runs/analysis_1010_failure_theory/REPORT_FORMAT.md
# --------------------------------------------------------------------------


def _f(x, k=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{k}f}"


def write_report(path, *, agg, meta_run):
    m = agg["meta"]
    vc = agg["verdict_counts"]
    pooled = agg["pooled"]
    unanimous = len(vc) == 1
    outcome = (next(iter(vc)) if unanimous else
               "MIXED per island (" + ", ".join(f"{k.split(' ')[0]} {v}" for k, v in sorted(vc.items())) + " of 8)")
    L = []
    L.append(f"# N8 — random sampling against the archive, {meta_run['date']}, commit {meta_run['commit']}")
    L.append("")
    L.append("## 6 What ran")
    L.append(f"command:        {meta_run['command']}")
    L.append(f"commit:         {meta_run['commit']}{' (working tree dirty: ' + meta_run['dirty'] + ')' if meta_run['dirty'] else ''}")
    L.append(f"seed(s):        default_rng({m['seed']}) for the genome draw, then one rng.integers(1<<30) per body for its evaluation draw; "
             f"bootstrap default_rng(0)")
    L.append(f"inputs:         {meta_run['run']}/search_state.pkl (descriptors), archive_<island>.pkl (axes), snapshots/gen<G>_<island>.json, "
             f"events.jsonl (comparison generation), scoring_networks/gen{m['net_gen']:05d}.npz, checkpoint.json (config)")
    L.append(f"wall:           {meta_run['wall_s']:.0f} s (draw + Tier-0 {m['wall_draw_s']:.0f} s)")
    eff = [
        "literal 'gen-37' comparison: NOT COMPARABLE (arch49 islands reach about 300 evaluate events near gen 150; snapshots only at "
        "gens 0, 50, 100, 150, 200, 250; evaluate events carry no features, so a gen-37 archive cannot be rebuilt). The comparison run is "
        "the §D' item 6 reading: the snapshot nearest 300 evaluations per island, verdict per island, pooled row secondary.",
        "comparison generation per island (cumulative evaluate events -> nearest snapshot): "
        + "; ".join(f"{i} gen {m['comparison'][i]['gen']} ({m['comparison'][i]['evals']} evals)" for i in ISLANDS),
        f"seeding ratio {m['n_ref']} reference : {m['n_rand']} random per round, taken from arch49's provenance "
        f"(ARCH51_SPEC text says 20:8, which is SearchConfig's default; arch49 ran 12:8); {m['rounds']} rounds, {m['drawn']} genomes drawn.",
        f"scoring network: arch49 scoring_networks/gen{m['net_gen']:05d}.npz ({'loaded' if m['net_loaded'] else 'MISSING: shared=None'}).",
    ]
    if m.get("cpu"):
        eff.append("--cpu: THE NUMPY FALLBACK PATH. evaluate_candidates ran evaluate_candidate per body: no shared network, no controller "
                   "refinement. This is not the search's scorer; do not read the verdict.")
    if meta_run.get("limit"):
        eff.append(f"--limit {meta_run['limit']}: SMOKE RUN, not the N8 measurement.")
    L.append("effective config: " + eff[0])
    for e in eff[1:]:
        L.append("                " + e)
    L.append("")
    L.append("## 7 Data quality")
    L.append(f"n:              {agg['n_scored']} random bodies scored (origins {agg['origins']}); arch49 side: "
             + ", ".join(f"{i} {agg['per_island'][i]['arch49_elites_with_features']}/{agg['per_island'][i]['arch49_snapshot_elites']}"
                         for i in ISLANDS) + " snapshot elites with features / in the snapshot")
    L.append(f"dropped:        Tier-0 rejects while drawing: {m['rejects'] or 'none'} (of {m['drawn']} drawn); "
             f"scored bodies dropped by the evaluator: {len(agg['dropped'])}")
    L.append(f"artifacts:      bodies with a QACC reset in any segment: {agg['bad_qacc_bodies']} (scored as the scorer scores them, zeroed on "
             f"divergence); arch49 snapshot elites re-projected with the FINAL descriptors (not the descriptors live at gen G), so the "
             f"arch49 side is the snapshot as the final axes see it; cells where two re-projected elites collide keep the max; "
             f"plans among the random bodies: {agg['plans']}")
    L.append("")
    L.append("## 8 Numbers (the table the ROADMAP item asked for, nothing else)")
    L.append("| island | arch49 snapshot (gen / evals) | coverage arch49 / random [95% CI] / ratio | summed competence arch49 / random [95% CI] / ratio "
             "| within 15% (cov, sum) | arch49 leads > 30% (cov, sum) | verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for isl in ISLANDS:
        p = agg["per_island"][isl]
        v = p["verdict"]
        L.append(f"| {isl} | {p['compare_gen']} / {p['arch49_evals']} | {_f(p['arch49']['coverage'], 4)} ({p['arch49']['filled']}) / "
                 f"{_f(p['random']['coverage'], 4)} ({p['random']['filled']}) [{_f(p['random_ci95']['coverage'][0], 4)}, {_f(p['random_ci95']['coverage'][1], 4)}] / "
                 f"{_f(v['coverage']['random_over_arch49'], 2)} | {_f(p['arch49']['summed'], 3)} / {_f(p['random']['summed'], 3)} "
                 f"[{_f(p['random_ci95']['summed'][0], 3)}, {_f(p['random_ci95']['summed'][1], 3)}] / {_f(v['summed']['random_over_arch49'], 2)} "
                 f"| {v['coverage']['within_15']}, {v['summed']['within_15']} | {v['coverage']['arch49_leads_gt_30']}, {v['summed']['arch49_leads_gt_30']} "
                 f"| {v['outcome'].split(' ')[0]} |")
    pv = pooled["verdict"]
    L.append(f"| POOLED (sum over 8 islands; secondary) | - | {_f(pooled['arch49']['coverage'], 4)} ({pooled['arch49']['filled']}) / "
             f"{_f(pooled['random']['coverage'], 4)} ({pooled['random']['filled']}) / {_f(pv['coverage']['random_over_arch49'], 2)} "
             f"| {_f(pooled['arch49']['summed'], 3)} / {_f(pooled['random']['summed'], 3)} / {_f(pv['summed']['random_over_arch49'], 2)} "
             f"| {pv['coverage']['within_15']}, {pv['summed']['within_15']} | {pv['coverage']['arch49_leads_gt_30']}, {pv['summed']['arch49_leads_gt_30']} "
             f"| {pv['outcome'].split(' ')[0]} |")
    L.append("")
    L.append("Context only (gen 50 snapshot, fewer evaluations than the 300 random bodies): "
             + "; ".join(f"{i} {agg['per_island'][i]['context_gen50']['filled']} cells, sum {_f(agg['per_island'][i]['context_gen50']['summed'], 3)} "
                         f"({agg['per_island'][i]['context_gen50']['evals']} evals)" for i in ISLANDS))
    L.append("Cells with a competence above zero (arch49 / random): "
             + "; ".join(f"{i} {agg['per_island'][i]['arch49_nonzero_cells']} / {agg['per_island'][i]['random_nonzero_cells']}" for i in ISLANDS))
    L.append("")
    L.append("spread:         95% bootstrap interval over the 300 random bodies in every random-side cell of the table (1000 resamples); "
             "the arch49 side is one fixed snapshot with no resampling.")
    L.append("")
    L.append("## 9 Against the frozen prediction (quote it verbatim, then one word)")
    L.append(f"prediction:     \"{PREDICTION}\"")
    L.append("                (ROADMAP after the §D' correction: 'within 15% of arch49's island archives at the snapshot nearest 300 evaluations "
             "per island ... verdict per island'; literal gen-37: NOT COMPARABLE, see section 6)")
    if meta_run.get("limit") or m.get("cpu"):
        outcome = (f"NOT A MEASUREMENT ({'smoke, --limit ' + str(meta_run['limit']) if meta_run.get('limit') else '--cpu path'}"
                   f"{'; --cpu path' if (meta_run.get('limit') and m.get('cpu')) else ''}; the rule would read: {outcome})")
    L.append(f"outcome:        {outcome}")
    L.append("by:             per island: " + "; ".join(f"{i} {agg['per_island'][i]['verdict']['outcome'].split(' ')[0]} "
                                                        f"(cov x{_f(agg['per_island'][i]['verdict']['coverage']['random_over_arch49'], 2)}, "
                                                        f"sum x{_f(agg['per_island'][i]['verdict']['summed']['random_over_arch49'], 2)})" for i in ISLANDS)
             + f"; pooled {pv['outcome'].split(' ')[0]} (cov x{_f(pv['coverage']['random_over_arch49'], 2)}, sum x{_f(pv['summed']['random_over_arch49'], 2)})")
    L.append("decision rule:  within 15% = |random / arch49 - 1| <= 0.15 on both coverage and summed competence; arch49 leads by > 30% = "
             "arch49 / random - 1 > 0.30 on either (arch49 > 0 with random = 0 counts); CONFIRMED / REFUTED / NEITHER as in the script docstring.")
    L.append("")
    L.append("## 12 Reproducer")
    L.append("```bash")
    L.append("cd /home/hundo/Projects/Dytiscidae/dytiscidae")
    L.append(meta_run["command"])
    L.append("```")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n")


def git_state():
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--short", "--", "dytiscidae", "tests"],
                               capture_output=True, text=True, timeout=10).stdout.strip().replace("\n", "; ")
    except Exception:  # noqa: BLE001
        sha, dirty = "unknown", ""
    return sha, dirty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/arch49")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--n", type=int, default=TARGET_EVALS, help="bodies that must pass Tier-0")
    ap.add_argument("--limit", type=int, default=0, help="smoke: draw and score only this many bodies")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--n-ref", type=int, default=None, help="reference seeds per round (default: arch49's provenance)")
    ap.add_argument("--n-rand", type=int, default=None, help="random seeds per round (default: arch49's provenance)")
    ap.add_argument("--cpu", action="store_true", help="numpy fallback path; no shared network; smoke only")
    ap.add_argument("--from-rows")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    tag = Path(args.run).name
    smoke = bool(args.limit)
    here = Path(__file__).parent
    out = Path(args.out) if args.out else here / f"results_{tag}{'_smoke' if smoke else ''}.json"
    report = Path(args.report) if args.report else (
        here / "smoke_N8_result.md" if smoke else ROOT / "runs/analysis_1010_failure_theory/N8_result.md")
    for p in (out, report):
        if "runs/arch" in str(p.resolve()):
            raise SystemExit("refusing to write into runs/arch*")
    rows_path = Path(args.from_rows) if args.from_rows else Path(str(out) + ".rows.jsonl")
    if args.from_rows:
        wall = float("nan")
    else:
        wall, _t_draw, per_batch = measure(args, rows_path)
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines() if ln.strip()]
    agg = aggregate(args.run, rows)
    sha, dirty = git_state()
    cmd = ("PYTHONPATH=. MUJOCO_GL=disable DYTISCIDAE_KERNEL_DIR=/home/hundo/Projects/Dytiscidae/dytiscidae/mojo/build "
           ".venv/bin/python experiments/random_sampling_control/run.py " + " ".join(sys.argv[1:]))
    meta_run = {"date": time.strftime("%Y-%m-%d"), "commit": sha, "dirty": dirty, "command": cmd.strip(),
                "run": args.run, "wall_s": wall, "limit": args.limit}
    out.write_text(json.dumps({"run": meta_run, "aggregate": agg}, indent=1, default=str))
    write_report(report, agg=agg, meta_run=meta_run)
    brief = {i: {"gen": p["compare_gen"], "arch49": p["arch49"], "random": p["random"], "verdict": p["verdict"]["outcome"]}
             for i, p in agg["per_island"].items()}
    print(json.dumps({"n_scored": agg["n_scored"], "rejects": agg["meta"]["rejects"], "drawn": agg["meta"]["drawn"],
                      "per_island": brief, "pooled": {"arch49": agg["pooled"]["arch49"], "random": agg["pooled"]["random"],
                                                      "verdict": agg["pooled"]["verdict"]["outcome"]},
                      "wall_s": wall}, indent=1, default=str))
    print(f"report: {report}")


if __name__ == "__main__":
    main()
