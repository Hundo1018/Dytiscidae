"""No-model admission gate for every Tier-1 bar the search rewards.

NeutronGym (``docs/papers/2610.03631.md`` §3.5) admits a benchmark task only if
probes that use no model fail it.  CLAUDE.md "Designing a measurement" lists
nine bars that paid for passive motion; each was found by hand, and the still
check was run only on transitions (``experiments/transition_distance``).  This
runs it on *every* bar a Tier-1 evaluation can clear, so the tenth is found
before a run spends 12 h on it.

For every elite of a finished run, through the batched evaluator the search
scores with (``batchroll.evaluate_tier1_batch``: the three medium segments, the
run's scatter and task draws, then the transitions), under the network that
scored it (``scoring_networks/gen<gen>.npz``), three arms on the same body:

  elite  its own gait + own policy + the scoring network (what the archive holds)
  still  ``TriphibianEnv.held_still_params()``: every actuator still, rotors off
  base   the body's base CPG gait at its base amplitudes, no policy, no shared
         network -- NeutronGym's "fixed answer" analogue (the gait never searched)

Every bar is a boolean per (elite, arm), the bars being enumerated from the
code, not typed here (``enumerate_bars``).  ``gate_table`` turns counts into
the verdicts of ``domain.evidence.no_model_verdict``.

    systemd-run --user --unit nmg-arch48 -p MemoryMax=3500M env MUJOCO_GL=disable \
        PYTHONPATH=. DYTISCIDAE_KERNEL_DIR=<main>/mojo/build \
        .venv/bin/python experiments/no_model_gate/run.py \
        --run runs/arch48 --out experiments/no_model_gate/results_arch48.json

    python experiments/no_model_gate/run.py --selftest      # no simulator
    python experiments/no_model_gate/run.py --from-rows experiments/no_model_gate/results_arch48.json.rows.jsonl \
        --run runs/arch48 --out ...                          # re-aggregate only
"""
import argparse
import collections
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

MEDIA = ("air", "water", "land")
ARMS = ("elite", "still", "base")
KINDS = ("air_to_water", "water_to_air", "water_to_land", "land_to_air")


# --------------------------------------------------------------------------
# Counts -> verdicts (pure Python, no simulator)
# --------------------------------------------------------------------------


def gate_table(rows):
    """Counts to verdict rows.

    ``rows``: dicts with ``bar`` and ``elite``, ``still``, ``base`` each a
    ``(k, n)`` pair, and optionally ``elite_plans`` / ``still_plans`` /
    ``base_plans`` (the body plan of each pass).  Returns one dict per row with
    the Clopper-Pearson bounds (elite lower, still and base upper, one-sided
    95%), ``verdict_still`` and ``verdict_base`` from ``no_model_verdict``
    against the elites, and ``distinct_share`` of each arm's passes by plan.
    """
    from dytiscidae.domain.evidence import clopper_pearson, distinct_share, no_model_verdict
    out = []
    for r in rows:
        ek, en = r["elite"]
        sk, sn = r["still"]
        bk, bn = r["base"]
        elo, ehi = clopper_pearson(ek, en)
        slo, shi = clopper_pearson(sk, sn)
        blo, bhi = clopper_pearson(bk, bn)
        row = dict(r)
        row.update({
            "elite_ci": [elo, ehi], "still_ci": [slo, shi], "base_ci": [blo, bhi],
            "elite_share": ek / en if en else float("nan"),
            "still_share": sk / sn if sn else float("nan"),
            "base_share": bk / bn if bn else float("nan"),
            "verdict_still": no_model_verdict(sk, sn, ek, en),
            "verdict_base": no_model_verdict(bk, bn, ek, en),
        })
        for arm in ARMS:
            if f"{arm}_plans" in r:
                row[f"{arm}_distinct"] = list(distinct_share(r[f"{arm}_plans"]))
        out.append(row)
    return out


def selftest():
    t = gate_table([
        {"bar": "clear", "elite": (100, 200), "still": (0, 200), "base": (0, 200)},
        {"bar": "leak", "elite": (10, 200), "still": (12, 200), "base": (3, 200)},
        {"bar": "thin", "elite": (3, 200), "still": (0, 200), "base": (0, 200)},
        {"bar": "none", "elite": (0, 200), "still": (0, 200), "base": (0, 200)},
        {"bar": "fam", "elite": (5, 200), "still": (0, 200), "base": (0, 200),
         "elite_plans": ["gannet", "gannet", "medusa"], "still_plans": []},
    ])
    got = {r["bar"]: (r["verdict_still"], r["verdict_base"]) for r in t}
    want = {"clear": ("certified", "certified"), "leak": ("leak", "underpowered"),
            "thin": ("underpowered", "underpowered"), "none": ("unmeasured", "unmeasured"),
            "fam": ("underpowered", "underpowered")}
    # "leak" base: 3/200 vs elite 10/200 -- upper(3/200) 4.4% > lower(10/200) 2.9% -> leak
    want["leak"] = ("leak", "leak")
    assert got == want, (got, want)
    assert t[4]["elite_distinct"] == [2, 3], t[4]
    print("gate_table selftest passed")


# --------------------------------------------------------------------------
# The bars, enumerated from the code
# --------------------------------------------------------------------------


def competence_thresholds():
    """Every per-medium competence threshold the search uses, with its source."""
    from dytiscidae.envs.evaluate import LEG_COMPETENCE_BAR
    from dytiscidae.evolution.curriculum import STAGES, WEAKEST_BARS
    src = {}
    for v, s in [(WEAKEST_BARS[1], "curriculum.WEAKEST_BARS[1] / three_media / TRIPHIBIAN_FLOOR*2"),
                 (WEAKEST_BARS[0], "curriculum.WEAKEST_BARS[0]"),
                 (LEG_COMPETENCE_BAR, "evaluate.LEG_COMPETENCE_BAR"),
                 (STAGES[0][2], "curriculum.STAGES single"),
                 (STAGES[3][2], "curriculum.STAGES chain / WEAKEST_BARS[3]"),
                 (STAGES[1][2], "curriculum.STAGES directed, crossing / WEAKEST_BARS[2]")]:
        src.setdefault(round(float(v), 6), []).append(s)
    return {k: "; ".join(v) for k, v in sorted(src.items())}


def enumerate_bars():
    """``[(name, family, threshold_text, source)]`` for every bar evaluated.

    The names are the keys of ``bars_of``'s result; ``bars_of`` builds them from
    the same constants, so a bar added to ``judge.LADDER`` appears here and there.
    """
    from dytiscidae.evolution.curriculum import STAGES, WEAKEST_BARS
    from dytiscidae.evolution.judge import LADDER, LOWER_IS_BETTER
    bars = []
    for dom, rungs in LADDER.items():
        for r, (nm, metric, thr) in enumerate(rungs, 1):
            op = "<=" if metric in LOWER_IS_BETTER else ">="
            bars.append((f"rung:{dom}>={r}:{nm}", "ladder", f"{metric} {op} {thr}",
                         "judge.LADDER" + (" (defined; not judged by _score_candidate)"
                                           if dom == "takeoff" else "")))
    for dom in MEDIA:
        for thr, src in competence_thresholds().items():
            bars.append((f"comp:{dom}>={thr:g}", "competence", f"competence >= {thr:g}", src))
    bars.append((f"weakest:c2>={WEAKEST_BARS[0]:g}", "weakest", f"2nd-best medium >= {WEAKEST_BARS[0]:g}",
                 "curriculum.WEAKEST_BARS[0]"))
    bars.append((f"weakest:c3>={WEAKEST_BARS[1]:g}", "weakest",
                 f"weakest medium >= {WEAKEST_BARS[1]:g} (three_media)",
                 "curriculum.WEAKEST_BARS[1]; loop.py three_media"))
    for s in range(4):
        bars.append((f"stage{s}:pass", "curriculum", "stage_score >= island's own bar "
                     f"(std {STAGES[s][2]:g} / weakest {WEAKEST_BARS[s]:g})",
                     "curriculum.Curriculum.bar via islands.curriculum_for"))
    bars.append(("stage4:mission>0", "curriculum", "mission_fraction > 0 (stage-4 bar is 0.0)",
                 "curriculum.STAGES mission"))
    for k in KINDS:
        bars.append((f"crossed:{k}", "transition", "crossed", "transitions.CrossingTracker"))
    return bars


def bars_of(res, island):
    """Every bar for one MissionResult: ``{bar_name: bool}``."""
    from dytiscidae.evolution.curriculum import WEAKEST_BARS, stage_score
    from dytiscidae.evolution.islands import curriculum_for
    from dytiscidae.evolution.judge import LADDER, rung_reached
    from dytiscidae.envs.transitions import TransitionSet

    out = {}
    segs = res.segments
    tset = getattr(res, "transitions", None) or TransitionSet()
    tmeas = tset.component_means()
    tmeas["crossed_fraction"] = tset.crossed_fraction
    meas = {d: (getattr(s, "measurements", {}) or {}) for d, s in segs.items()}
    for dom, rungs in LADDER.items():
        if dom == "transition":
            m = tmeas
        elif dom == "takeoff":
            m = meas.get("land", {})        # takeoff_height is a land-segment measurement
        else:
            m = meas.get(dom, {})
        k = rung_reached(dom, m) if (dom == "transition" or dom in segs or dom == "takeoff") else 0
        for r in range(1, len(rungs) + 1):
            out[f"rung:{dom}>={r}:{rungs[r - 1][0]}"] = bool(k >= r)
    comp = {d: float(getattr(segs[d], "competence", 0.0)) if d in segs else 0.0 for d in MEDIA}
    for dom in MEDIA:
        for thr in competence_thresholds():
            out[f"comp:{dom}>={thr:g}"] = bool(comp[dom] >= thr)
    s = sorted(comp.values(), reverse=True)
    out[f"weakest:c2>={WEAKEST_BARS[0]:g}"] = bool(s[1] >= WEAKEST_BARS[0])
    out[f"weakest:c3>={WEAKEST_BARS[1]:g}"] = bool(s[2] >= WEAKEST_BARS[1])
    cur = curriculum_for(island)
    own = dict(domains=cur.domains, transition_names=cur.transition_names, weakest=cur.weakest)
    for st in range(4):
        out[f"stage{st}:pass"] = bool(stage_score(st, res, tset, **own) >= cur.bar(st))
    out["stage4:mission>0"] = bool(float(res.mission_fraction) > 0.0)
    for k in KINDS:
        t = (tset.results or {}).get(k)
        out[f"crossed:{k}"] = bool(t is not None and t.crossed)
    return out


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate(rows, bars):
    """Counts per bar and arm from per-(elite, arm) rows -> ``gate_table`` rows."""
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["index"]][r["arm"]] = r
    idx = sorted(i for i, a in by.items() if all(x in a for x in ARMS))
    table_in, per_island = [], {}
    for name, family, thr, src in bars:
        row = {"bar": name, "family": family, "threshold": thr, "source": src}
        for arm in ARMS:
            passed = [i for i in idx if by[i][arm]["bars"].get(name)]
            row[arm] = (len(passed), len(idx))
            row[f"{arm}_plans"] = [by[i][arm]["plan"] for i in passed]
            row[f"{arm}_passed_indices"] = passed
        table_in.append(row)
    for isl in sorted({by[i]["elite"]["island"] for i in idx}):
        sub = [i for i in idx if by[i]["elite"]["island"] == isl]
        per_island[isl] = {name: {arm: [sum(bool(by[i][arm]["bars"].get(name)) for i in sub), len(sub)]
                                  for arm in ARMS} for name, *_ in bars}
    return gate_table(table_in), per_island, idx


def fmt_table(t, n):
    def kn(k, n_):
        return f"{k}/{n_}"
    lines = ["| bar | elite k/n (lo) | still k/n (hi) | base k/n (hi) | still | base "
             "| plans elite | plans still |",
             "|---|---|---|---|---|---|---|---|"]
    hidden = 0
    for r in t:
        if not (r["elite"][0] or r["still"][0] or r["base"][0]):
            hidden += 1
            continue
        d = lambda a: ("%d/%d" % tuple(r.get(f"{a}_distinct", (0, 0)))) if r[a][0] else "-"
        lines.append(
            f"| {r['bar']} | {kn(*r['elite'])} ({r['elite_ci'][0]:.3f}) "
            f"| {kn(*r['still'])} ({r['still_ci'][1]:.3f}) | {kn(*r['base'])} ({r['base_ci'][1]:.3f}) "
            f"| {r['verdict_still']} | {r['verdict_base']} | {d('elite')} | {d('still')} |")
    lines.append("")
    lines.append(f"{hidden} further bars were cleared by no elite, no still machine and no base "
                 f"machine out of n = {n} (all `unmeasured`); they are in the JSON.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


def measure(args, rows_path):
    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites, plain_controller, stratified
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
    if args.island:
        # One island's own archive, not the merged one (the merge keeps one occupant
        # per cell across islands, so an island's elite can be gone from it).
        from dytiscidae.ops.run import load_run_archive
        elites = list(load_run_archive(run, args.island)[0].cells.values())
    else:
        elites = load_elites(run)
    pick = stratified(elites, args.n) if args.n else list(range(len(elites)))
    groups = collections.defaultdict(list)
    for i in pick:
        m = elites[i].meta or {}
        groups[(int(m.get("gen") or 0), int(m.get("eval_seed") or 0))].append(i)
    keys = sorted(groups)[: args.max_groups or None]
    print(f"{len(elites)} elites, measuring {sum(len(groups[k]) for k in keys)} in {len(keys)} "
          f"(gen, seed) groups; segment_seconds {seg}; arms {ARMS}", flush=True)

    spec = MissionSpec()
    t0, skipped, repro = time.time(), [], []
    done = set()
    if rows_path.exists() and args.resume:
        for ln in rows_path.read_text().splitlines():
            done.add(json.loads(ln)["index"])
    fh = open(rows_path, "a" if args.resume else "w")
    for gi, key in enumerate(keys):
        idx = [i for i in groups[key] if i not in done]
        phenos, ctrls, keep, like = [], [], [], None
        for i in idx:
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
        for arm, cs, sh in (("elite", ctrls, net), ("still", stills, None), ("base", None, None)):
            out[arm] = batchroll.evaluate_tier1_batch(
                phenos, spec=spec, controllers=cs, segment_seconds=seg,
                identify_axes=False, seed=key[1], shared=sh, n_modes=n_modes)
        for slot, i in enumerate(keep):
            m = elites[i].meta or {}
            island = args.island or m.get("island") or "generalist"
            for arm in ARMS:
                res = out[arm][slot]
                rec = {"index": int(i), "arm": arm, "gen": key[0], "island": island,
                       "plan": m.get("body_plan") or "?",
                       "bars": bars_of(res, island),
                       "comp": {d: float(res.segments[d].competence) for d in MEDIA if d in res.segments},
                       "mission_fraction": float(res.mission_fraction),
                       "bad_qacc": {d: int(res.segments[d].bad_qacc) for d in MEDIA if d in res.segments},
                       "failures": {d: res.segments[d].failure for d in MEDIA
                                    if d in res.segments and res.segments[d].failure}}
                if arm == "elite":
                    rec["recorded"] = {d: m.get(d) for d in MEDIA}
                    rec["recorded_rungs"] = m.get("rungs")
                fh.write(json.dumps(rec) + "\n")
        fh.flush()
        del out, phenos, ctrls, stills
        gc.collect()
        print(f"group {gi + 1}/{len(keys)} gen {key[0]} n={len(keep)} wall {time.time() - t0:.0f}s",
              flush=True)
    fh.close()
    return time.time() - t0, skipped, len(elites)


def reproduction(rows):
    """Does the elite arm reproduce what the archive recorded?"""
    el = [r for r in rows if r["arm"] == "elite"]
    cmp_ = {d: [abs(r["comp"].get(d, 0.0) - float(r["recorded"][d])) <= 0.005
                for r in el if r.get("recorded", {}).get(d) is not None] for d in MEDIA}
    return {d: {"within_0.005": int(sum(v)), "n": len(v)} for d, v in cmp_.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=0, help="0 = every elite, else a stratified n")
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--island", default="", help="that island's own archive instead of the merged one")
    ap.add_argument("--resume", action="store_true", help="append to <out>.rows.jsonl, skipping done elites")
    ap.add_argument("--from-rows", help="re-aggregate a rows.jsonl instead of simulating")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    out = Path(args.out)
    rows_path = Path(args.from_rows) if args.from_rows else Path(str(out) + ".rows.jsonl")
    if args.from_rows:
        wall, skipped, n_archive = float("nan"), [], 0
    else:
        wall, skipped, n_archive = measure(args, rows_path)
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    bars = enumerate_bars()
    table, per_island, idx = aggregate(rows, bars)
    md = fmt_table(table, len(idx))
    meta = {"run": args.run, "n_elites": len(idx), "of_archive": n_archive, "arms": ARMS,
            "wall_s": wall, "skipped_no_controller": skipped, "reproduction": reproduction(rows),
            "competence_thresholds": {str(k): v for k, v in competence_thresholds().items()},
            "plan_counts": dict(collections.Counter(
                r["plan"] for r in rows if r["arm"] == "elite")),
            "mean_competence": {arm: {d: float(np.mean([r["comp"].get(d, 0.0) for r in rows if r["arm"] == arm]))
                                      for d in MEDIA} for arm in ARMS},
            "verdict_counts": {a: dict(collections.Counter(r[f"verdict_{a}"] for r in table))
                               for a in ("still", "base")}}
    out.write_text(json.dumps({**meta, "table": table, "per_island": per_island}, indent=1))
    Path(str(out).replace(".json", "") + "_table.md").write_text(md + "\n")
    print(json.dumps(meta, indent=1))
    print(md)


if __name__ == "__main__":
    main()
