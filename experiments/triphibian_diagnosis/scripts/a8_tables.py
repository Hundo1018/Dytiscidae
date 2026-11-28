"""A8 pass D: assemble tables T1-T8 from results/{segments,vocab,archives,events_<run>}.json
and the film manifests, into results/a8_tables.md (pasted into the report) and
results/a8_tables.json (the numbers).

No run data is read here except runs/<run>/checkpoint.json (commit) and
runs/<run>/media/film_manifest.json (one filmed design per run).
Usage: nice -n 10 .venv/bin/python experiments/triphibian_diagnosis/scripts/a8_tables.py
"""
import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
RES = Path(__file__).resolve().parents[1] / "results"
RUNS_ALL = [f"arch{n}" for n in range(43, 61)]
FLOOR = 0.012
KINDS = ("air_to_water", "water_to_air", "water_to_land", "land_to_air")
TAGS = ("air", "water", "land", "transition:air_to_water", "transition:water_to_air",
        "transition:water_to_land", "transition:land_to_air")
ISLAND_MEDIA = {
    "air": ("air",), "water": ("water",), "land": ("land",),
    "amphibian": ("water", "land"), "aerial_diver": ("air", "water"),
    "land_air": ("land", "air"), "triphibian": ("air", "water", "land"),
    "generalist": ("air", "water", "land"),
}
ABBR = {"air": "air", "water": "wat", "land": "lnd", "amphibian": "amph", "aerial_diver": "a.div",
        "land_air": "l.air", "triphibian": "tri", "generalist": "gen"}


def num(run):
    return int(run[4:])


def regime(run):
    n = num(run)
    return {
        "cross": "X0" if n <= 46 else "X1",
        "air": "A0" if n <= 50 else ("A1d" if n == 60 else "A1"),
        "wl": "W0" if n <= 48 else ("W2" if n == 60 else "W1"),
        "fit": "F0" if n <= 50 else "F1",
    }


def pct(a, b, d=1):
    if not b:
        return "n/a"
    return f"{100.0 * a / b:.{d}f}%"


def frac(a, b, d=1):
    return f"{a}/{b} ({pct(a, b, d)})" if b else f"{a}/0"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def f3(x, d=3):
    if x is None:
        return "-"
    if isinstance(x, float) and (abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0)):
        return f"{x:.2e}"
    return f"{x:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def load():
    ev = {r: json.load(open(RES / f"events_{r}.json")) for r in RUNS_ALL}
    ar = json.load(open(RES / "archives.json"))
    seg = json.load(open(RES / "segments.json"))
    voc = json.load(open(RES / "vocab.json"))
    ck, film = {}, {}
    for r in RUNS_ALL:
        try:
            ck[r] = json.load(open(RUNS / r / "checkpoint.json"))["provenance"]
        except Exception:
            ck[r] = {}
        fp = RUNS / r / "media" / "film_manifest.json"
        film[r] = json.load(open(fp)) if fp.exists() else None
    return ev, ar, seg, voc, ck, film


def t1(ev, ar, seg, voc, ck):
    rows, data = [], {}
    for r in RUNS_ALL:
        e = ev[r]
        g, te, p, cfg = e["T1_gen"], e["T1_events"], e["phys"], e["config"]
        n_isl = len(cfg.get("islands") or [])
        kk = te["kinds_kept"]
        eval_events = kk.get("evaluate", 0)
        reev = kk.get("reevaluate", 0)
        t0_non = p.get("tier0_nonseed", 0)
        seeds_phys = te["seeds_physical"] + (p.get("tier0_seed_events", 0) / max(n_isl, 1))
        nonseed_eval = eval_events - te["seed_events"]
        ident = (g["evaluated_last"] or 0) - (nonseed_eval + t0_non + reev + seeds_phys)
        ppo = e["ppo"]
        upd = [x for x in ppo if not x["skipped"]]
        trans = [x["transitions"] or 0 for x in ppo]
        trajs = [x["trajectories"] or 0 for x in ppo]
        grpo = sum((x.get("grpo_transitions") or 0) for x in ppo)
        isl_v = g["islands_visited"]
        isl_s = ("air only" if list(isl_v) == ["air"] else
                 f"{len(isl_v)} ({min(isl_v.values())}-{max(isl_v.values())} gens each)")
        ms = te["memory_stops"]
        mem = f"ceiling {cfg.get('memory_ceiling_mb')}"
        if ms:
            mem += f"; {len(ms)} memory_stop ({', '.join(str(x['resident_mb']) for x in ms)} MB)"
        if g.get("resident_mb_present"):
            mem += f"; resident_mb on {g['resident_mb_present']} gen lines, max {g['resident_mb_max']}"
        rg = regime(r)
        rows.append([
            r, (ck[r].get("git") or "?")[:7], f"{rg['cross']} {rg['air']} {rg['wl']} {rg['fit']}",
            f"{g['gens_kept']}/{cfg.get('generations')}",
            f"{g['processes']} ({g['gen_lines_discarded']} lines, {te['kinds_discarded'].get('evaluate', 0)} evals re-run)",
            isl_s, cfg.get("evaluator") or "batched",
            p.get("evals"), eval_events, reev, kk.get("tier0_reject", 0),
            f"{g['evaluated_last']} ({ident:+.0f}; exploit lines {e['exploits']['lines_all_processes']})",
            g["rollouts"], frac(g["diverged_rollouts"], g["rollouts"], 2),
            f"{g['wall_h_kept_gens']} / {g['wall_h_all_processes']}", g["median_gen_s"],
            f"{len(upd)}/{len(ppo)}", f"{sum(trans):.3g}", f"{mean(trans):.0f}" if trans else "-",
            f"{mean(trajs):.0f}" if trajs else "-", mem,
        ])
        data[r] = {"commit": ck[r].get("git"), "gens_kept": g["gens_kept"], "gens_planned": cfg.get("generations"),
                   "processes": g["processes"], "islands": isl_v, "evaluator": cfg.get("evaluator") or "batched",
                   "phys_evals": p.get("evals"), "evaluate_events": eval_events, "reevaluate": reev,
                   "tier0": kk.get("tier0_reject", 0), "evaluated_last": g["evaluated_last"],
                   "identity_residual": ident, "rollouts": g["rollouts"], "diverged": g["diverged_rollouts"],
                   "wall_h_kept": g["wall_h_kept_gens"], "wall_h_all": g["wall_h_all_processes"],
                   "wall_h_cost": g["wall_h_cost_seconds"], "median_gen_s": g["median_gen_s"],
                   "ppo_updates": len(upd), "ppo_events": len(ppo), "learner_transitions": sum(trans),
                   "learner_transitions_per_update": mean(trans), "trajectories_per_update": mean(trajs),
                   "grpo_transitions": grpo, "memory": {"ceiling": cfg.get("memory_ceiling_mb"), "stops": ms,
                                                        "resident_mb_lines": g.get("resident_mb_present"),
                                                        "resident_mb_max": g.get("resident_mb_max")},
                   "exploit_lines": e["exploits"]["lines_all_processes"],
                   "steps_total": g["steps_total"]}
    hdr = ["run", "commit (last checkpoint)", "regimes cross/air/water-land/fitness", "gens kept/planned",
           "processes (discarded)", "islands", "evaluator",
           "Tier-1 evals (seeds once)", "evaluate events", "reevaluate", "tier0 rejects",
           "`evaluated` last (identity residual; exploits.jsonl lines)", "rollouts", "diverged",
           "wall h kept / all procs",
           "median s/gen", "PPO updates/events", "learner transitions", "per update", "trajectories/update",
           "memory MB"]
    return table(hdr, rows), data


def t2(ev, film):
    rows, krows, xrows, data = [], [], [], {}
    x1tot = collections.Counter()
    pooled = {k: collections.Counter() for k in KINDS}
    pooled_x0 = {k: collections.Counter() for k in KINDS}
    for r in RUNS_ALL:
        e = ev[r]
        p, g = e["phys"], e["T1_gen"]
        nh = {int(k): v for k, v in e["ncross_hist"].items()}
        tot_cross = sum(k * v for k, v in nh.items())
        ge1 = sum(v for k, v in nh.items() if k >= 1)
        ge2 = sum(v for k, v in nh.items() if k >= 2)
        gens = e["gen_series"]["gen"]
        nbg = {int(k): v for k, v in e["ncross_by_gen"].items()}
        n = len(gens)
        q = max(1, n // 5)
        first, last = gens[:q], gens[-q:]
        c_first = sum(nbg.get(x, 0) for x in first)
        c_last = sum(nbg.get(x, 0) for x in last)
        pr = e["promote"]
        t2pos = [x for x in pr if (x["t2f"] or 0) > 0]
        t2max = max([x["t2f"] or 0 for x in pr], default=None)
        t2_two = sum(1 for x in pr if (x["t2f"] or 0) >= 2 / 9 - 1e-9)
        jt = g.get("judge_transition_last") or {}
        fm = film[r]
        if not fm:
            fstr = "no manifest"
        elif fm.get("film_error"):
            fstr = f"not filmed (film_error: {fm.get('film_error')}; {fm.get('island')}, {(fm.get('design') or {}).get('body_plan')})"
        else:
            fstr = (f"{(fm.get('mission') or {}).get('transitions')}, on-task "
                    f"{f3((fm.get('mission') or {}).get('on_task'), 2)} ({fm.get('island')}, "
                    f"{(fm.get('design') or {}).get('body_plan')})")
        mfg = {int(k): v for k, v in e["mf_pos_by_gen"].items()}
        rows.append([
            r, regime(r)["cross"], p.get("evals"),
            frac(p.get("mf_pos", 0), p.get("evals")), f3(p.get("mf_max_x1e4", 0) / 1e4, 4),
            f"{sum(1 for x in gens if mfg.get(x))}/{n}",
            frac(p.get("all3_pos", 0), p.get("evals")),
            f"{len(t2pos)}/{len(pr)}", f3(t2max, 3) if t2max is not None else "-", t2_two,
            (f3(g.get("mission_best_max"), 4) if g.get("mission_best_present") else "absent (no generalist island)"),
            (str(g.get("three_media_max")) if g.get("weakest_best_present") else "absent (no triphibian island)"),
            fstr,
        ])
        xrows.append([r, regime(r)["cross"], p.get("with_transition_record", 0), tot_cross,
                      frac(ge1, p.get("evals"), 2), ge2, f"{sum(1 for x in gens if nbg.get(x))}/{n}",
                      f"{c_first} -> {c_last}",
                      f"{jt.get('metric')} bar {jt.get('bar')}, tightenings {jt.get('tightenings')}"])
        if num(r) >= 47:
            for kk_, vv_ in (("evals", p.get("evals", 0)), ("crossings", tot_cross), ("ge1", ge1), ("ge2", ge2),
                             ("mf_pos", p.get("mf_pos", 0)), ("t2pos", len(t2pos)), ("promotions", len(pr))):
                x1tot[kk_] += vv_
        ko = e["kind_outcome"]
        for k in KINDS:
            c = ko[k]
            tgt = pooled if num(r) >= 47 else pooled_x0
            for kk, v in c.items():
                tgt[k][kk] += v
        data[r] = {"evals": p.get("evals"), "mf_pos": p.get("mf_pos", 0), "mf_max": p.get("mf_max_x1e4", 0) / 1e4,
                   "all3_pos": p.get("all3_pos", 0), "tier2_pos": len(t2pos), "promotions": len(pr), "tier2_max": t2max,
                   "tier2_ge_2_legs": t2_two, "crossings": tot_cross, "evals_ge1_cross": ge1, "evals_ge2_cross": ge2,
                   "gens_with_cross": sum(1 for x in gens if nbg.get(x)), "gens": n,
                   "cross_first20": c_first, "cross_last20": c_last, "kind_outcome": ko,
                   "approach_pos_no_cross": p.get("approach_pos_no_cross", 0),
                   "judge_transition_last": jt, "film": (fm or {}).get("mission")}
        # per-kind compact row
        cells = []
        for k in KINDS:
            c = ko[k]
            nrec = sum(v for kk, v in c.items() if kk != "no_record")
            ab = c.get("battery", 0) + c.get("unstable", 0) + c.get("diverged", 0)
            cut = c.get("not_crossed_note_cut", 0) + c.get("ambiguous_note_cut", 0)
            cells.append(f"{c.get('crossed', 0)} / {c.get('crossed_without_hold', 0)} / "
                         f"{c.get('never_crossed', 0)} / {ab} / {cut}")
        nrec_all = p.get("with_transition_record", 0)
        krows.append([r, regime(r)["cross"], nrec_all] + cells +
                     [frac(p.get("approach_pos_no_cross", 0), nh.get(0, 0))])
    hdr = ["run", "crossing regime", "Tier-1 evals", "Tier-1 mission_fraction>0", "max",
           "gens with any mission_fraction>0", "all three competences >0 (3 dp)",
           "Tier-2 tier2_fraction>0 / promotions", "max tier2_fraction", "promotions with >=2 legs",
           "generalist `mission_best` max", "triphibian `three_media` max",
           "film: continuous mission (one design)"]
    xhdr = ["run", "regime", "evals with a crossing record", "crossings (sum of 4 x crossed_fraction)",
            "evals with >=1 crossing", "evals with >=2", "gens with any crossing",
            "crossings in first 20% -> last 20% of gens", "judge transition bar (last line)"]
    data["X1_totals"] = dict(x1tot)
    khdr = ["run", "regime", "evals with a crossing record"] + \
           [f"{k}: crossed / crossed w/o hold / never crossed / aborted / note cut" for k in KINDS] + \
           ["0-crossing evals with graded approach > 0"]
    # pooled
    prow = []
    for lab, src in (("X1 pooled arch47-60", pooled), ("X0 pooled arch43-46", pooled_x0)):
        for k in KINDS:
            c = src[k]
            nrec = sum(v for kk, v in c.items() if kk != "no_record")
            ab = c.get("battery", 0) + c.get("unstable", 0) + c.get("diverged", 0)
            cut = c.get("not_crossed_note_cut", 0) + c.get("ambiguous_note_cut", 0)
            cr, cwh, nv = c.get("crossed", 0), c.get("crossed_without_hold", 0), c.get("never_crossed", 0)
            # hold >= HOLD_PASS is known true for `crossed`, known false for `crossed w/o hold`,
            # unknown for never-crossed, aborted (hold set to 0 when aborted: transitions.py:466) and cut.
            prow.append([lab, k, nrec, frac(cr, nrec, 3), frac(cwh, nrec), frac(nv, nrec), frac(ab, nrec), frac(cut, nrec),
                         f"{pct(cr, nrec, 2)} .. {pct(cr + nv + cut, nrec)}"])
    phdr = ["pool", "kind", "n", "crossed", "crossed without hold (reached target, hold < 0.5)",
            "never crossed (hold unknown)", "aborted (battery/unstable/diverged; hold := 0)", "note cut (not crossed)",
            "hold-precondition share, bounds"]
    data["pooled_X1"] = {k: dict(v) for k, v in pooled.items()}
    data["pooled_X0"] = {k: dict(v) for k, v in pooled_x0.items()}
    xrows.append(["**X1 total arch47-60**", "X1", "", x1tot["crossings"], frac(x1tot["ge1"], x1tot["evals"], 3),
                  x1tot["ge2"], "", "", ""])
    return table(hdr, rows), table(xhdr, xrows), table(khdr, krows), table(phdr, prow), data


def t3(ev, ar):
    erows, arows, data = [], [], {"evals": {}, "archives": {}}
    pool_e = collections.defaultdict(collections.Counter)
    for r in RUNS_ALL:
        p = ev[r]["phys"]
        n = p.get("evals", 0)
        erows.append([r, f"{regime(r)['air']}/{regime(r)['wl']}", n,
                      pct(p.get("floor_air", 0), n, 2), pct(p.get("floor_water", 0), n, 2), pct(p.get("floor_land", 0), n, 2),
                      pct(p.get("floor_air_water", 0), n, 2), pct(p.get("floor_air_land", 0), n, 2),
                      pct(p.get("floor_water_land", 0), n, 2),
                      frac(p.get("floor_media_2", 0) + p.get("floor_media_3", 0), n, 2), frac(p.get("floor_media_3", 0), n, 3)])
        key = f"{regime(r)['air']}/{regime(r)['wl']}"
        for k in ("evals", "floor_air", "floor_water", "floor_land", "floor_media_2", "floor_media_3",
                  "floor_air_water", "floor_air_land", "floor_water_land"):
            pool_e[key][k] += p.get(k, 0)
        data["evals"][r] = {k: p.get(k, 0) for k in ("evals", "floor_air", "floor_water", "floor_land",
                                                      "floor_media_2", "floor_media_3", "floor_air_water",
                                                      "floor_air_land", "floor_water_land")}
        for isl, s in ar[r]["islands"].items():
            c = s["counts"]
            nn = c.get("n", 0)
            arows.append([r, ABBR.get(isl, isl), s["generation"], nn, pct(c.get("air", 0), nn), pct(c.get("water", 0), nn),
                          pct(c.get("land", 0), nn), frac(c.get("m2", 0) + c.get("m3", 0), nn),
                          frac(c.get("m3", 0), nn), frac(c.get("floor_member", 0), nn),
                          f3(s["own_max"], 3)])
            data["archives"].setdefault(r, {})[isl] = c
    for key, c in sorted(pool_e.items()):
        n = c["evals"]
        erows.append([f"**pooled**", key, n, pct(c["floor_air"], n, 2), pct(c["floor_water"], n, 2), pct(c["floor_land"], n, 2),
                      pct(c["floor_air_water"], n, 2), pct(c["floor_air_land"], n, 2), pct(c["floor_water_land"], n, 2),
                      frac(c["floor_media_2"] + c["floor_media_3"], n, 2), frac(c["floor_media_3"], n, 3)])
    ehdr = ["run", "regime air/water-land", "Tier-1 evals", "air>=0.012", "water>=0.012", "land>=0.012",
            "air+water", "air+land", "water+land", ">=2 media", "all 3 media"]
    ahdr = ["run", "island", "archive gen", "elites", "air>=0.012", "water>=0.012", "land>=0.012",
            ">=2 media (same elite)", "all 3 (same elite)", "N5 floor members (>=1 island medium)",
            "max over elites of min(island media)"]
    # archive pooled by regime, multi-island runs only (single-island archives select on air alone)
    pool_a = collections.defaultdict(collections.Counter)
    for r in RUNS_ALL:
        for isl, s in ar[r]["islands"].items():
            key = f"{regime(r)['air']}/{regime(r)['wl']} {'air-only runs' if num(r) >= 52 else 'multi-island runs'}"
            for k in ("n", "air", "water", "land", "m2", "m3"):
                pool_a[key][k] += s["counts"].get(k, 0)
    prow = []
    for key, c in sorted(pool_a.items()):
        n = c["n"]
        prow.append([key, n, pct(c["air"], n), pct(c["water"], n), pct(c["land"], n),
                     frac(c["m2"] + c["m3"], n), frac(c["m3"], n, 2)])
    c = collections.Counter()
    for r in [f"arch{n}" for n in range(51, 60)]:
        for isl, s in ar[r]["islands"].items():
            for k in ("n", "air", "water", "land", "m2", "m3"):
                c[k] += s["counts"].get(k, 0)
    n = c["n"]
    prow.append(["**arch51-59 all islands (PREREG prediction 3)**", n, pct(c["air"], n), pct(c["water"], n),
                 pct(c["land"], n), frac(c["m2"] + c["m3"], n), frac(c["m3"], n, 2)])
    data["archives_arch51_59"] = dict(c)
    # archive.json check for single-island runs killed after their last checkpoint
    for r in RUNS_ALL:
        s = ar[r].get("archive_json_summary")
        if s:
            cc = s["counts"]
            nn = cc.get("n", 0)
            prow.append([f"{r} archive.json (gen {s['generation']}, check)", nn, pct(cc.get("air", 0), nn),
                         pct(cc.get("water", 0), nn), pct(cc.get("land", 0), nn),
                         frac(cc.get("m2", 0) + cc.get("m3", 0), nn), frac(cc.get("m3", 0), nn, 2)])
    phdr = ["pool (elites summed over islands; a migrant can sit on two islands)", "elites", "air", "water", "land",
            ">=2 media", "all 3"]
    # by plan / rotor, pooled within A1/W1 (arch51-59) over evaluations
    plan_rows, rot_rows = [], []
    for lab, runs in (("A1/W1 arch51-59", [f"arch{n}" for n in range(51, 60)]),
                      ("A0/W1 arch49-50", ["arch49", "arch50"]),
                      ("A0/W0 arch43-48", [f"arch{n}" for n in range(43, 49)])):
        bp = collections.defaultdict(collections.Counter)
        br = collections.defaultdict(collections.Counter)
        for r in runs:
            for k, v in ev[r]["media_by_plan"].items():
                bp[k].update(v)
            for k, v in ev[r]["media_by_rotor"].items():
                br[k].update(v)
        for k, c in sorted(bp.items(), key=lambda kv: -kv[1]["n"]):
            n = c["n"]
            plan_rows.append([lab, k, n, pct(c["air"], n, 2), pct(c["water"], n, 2), pct(c["land"], n, 2),
                              frac(c["m2"] + c["m3"], n, 2), frac(c["m3"], n, 3)])
        order = {"0": 0, "1-2": 1, "3-5": 2, "6+": 3, "unknown": 4}
        for k, c in sorted(br.items(), key=lambda kv: order.get(kv[0], 9)):
            n = c["n"]
            rot_rows.append([lab, k, n, pct(c["air"], n, 2), pct(c["water"], n, 2), pct(c["land"], n, 2),
                             frac(c["m2"] + c["m3"], n, 2), frac(c["m3"], n, 3)])
    bhdr = ["pool (Tier-1 evals)", "body plan", "evals", "air", "water", "land", ">=2 media", "all 3"]
    rhdr = ["pool (Tier-1 evals)", "rotor count", "evals", "air", "water", "land", ">=2 media", "all 3"]
    # top 10 elites
    tops = []
    seen = set()
    for r in RUNS_ALL:
        for isl, s in ar[r]["islands"].items():
            for t in s["top"]:
                key = (r, t["plan"], t["mass"], t["gen"], t["air"], t["water"], t["land"])
                if key in seen:
                    continue
                seen.add(key)
                tops.append({**t, "run": r})
    tops.sort(key=lambda t: -t["min"])
    trows = []
    for t in tops[:10]:
        rg = regime(t["run"])
        trows.append([t["run"], f"{rg['air']}/{rg['wl']}", ABBR.get(t["island"], t["island"]), t["plan"], t["mass"],
                      t["n_rotors"] if t["n_rotors"] is not None else "n/a", f3(t["air"]), f3(t["water"]), f3(t["land"]),
                      f3(t["min"]), t["gen"], t["tier"], t["n_draws"]])
    thdr = ["run", "regime", "island", "plan", "mass kg", "rotors", "air", "water", "land", "min", "scored gen",
            "tier", "draws"]
    # top 10 evaluations (events) for comparison
    etops = []
    for r in RUNS_ALL:
        for t in ev[r]["top10_evals"]:
            etops.append({**t, "run": r})
    etops.sort(key=lambda t: -t["min"])
    tev = [[t["run"], ABBR.get(t["island"], t["island"]), t["plan"], t["mass"],
            t["n_rotors"] if t["n_rotors"] is not None else "n/a", f3(t["air"]), f3(t["water"]), f3(t["land"]),
            f3(t["min"]), t["gen"]] for t in etops[:10]]
    tehdr = ["run", "island filed", "plan", "mass kg", "rotors", "air", "water", "land", "min", "gen"]
    data["top10_elites"] = tops[:10]
    data["top10_evals"] = etops[:10]
    return (table(ehdr, erows), table(ahdr, arows), table(phdr, prow), table(bhdr, plan_rows),
            table(rhdr, rot_rows), table(thdr, trows), table(tehdr, tev), data)


def window_stats(rows, getter):
    n = len(rows)
    if n < 5:
        return None
    q = max(1, n // 5)
    a = [getter(x) for x in rows[:q]]
    b = [getter(x) for x in rows[-q:]]
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    allv = [getter(x) for x in rows]
    allv = [x for x in allv if x is not None]
    if not a or not b:
        return None
    s = sd(allv) or 0.0
    return {"first": mean(a), "first_sd": sd(a), "last": mean(b), "last_sd": sd(b), "sd_all": s,
            "delta_over_sd": ((mean(b) - mean(a)) / s) if s > 0 else None, "n": n}


def t4(ev):
    rows, srows, ret_rows, data = [], [], [], {}
    for r in RUNS_ALL:
        ppo = [x for x in ev[r]["ppo"] if not x["skipped"]]
        d = {}
        cells = []
        for tag in TAGS:
            def g(x, tag=tag):
                v = (x["rbt"] or {}).get(tag)
                return v[0] if v else None
            st = window_stats(ppo, g)
            d[tag] = st
            if st is None:
                cells.append("-")
            else:
                cells.append(f"{f3(st['first'], 4)}±{f3(st['first_sd'], 4)} -> {f3(st['last'], 4)}±{f3(st['last_sd'], 4)}"
                             f" ({f3(st['delta_over_sd'], 2)} sd)")
        # return_by_tag (N10, arch51 on): discounted return incl. shaping, per tag
        rcells = []
        for tag in TAGS:
            st = window_stats(ppo, lambda x, tag=tag: ((x.get("ret") or {}).get(tag) or [None])[0])
            d[f"return:{tag}"] = st
            rcells.append("-" if st is None else
                          f"{f3(st['first'], 3)} -> {f3(st['last'], 3)} ({f3(st['delta_over_sd'], 2)} sd)")
        # share of the return's magnitude that is terminal (the competence the search
        # selects on) rather than potential-based shaping: sum|terminal| / sum(|shaping|+|terminal|)
        shares = []
        for tag in ("air", "water", "land", "transition"):
            sh = te = 0.0
            for x in ppo:
                for t, v in (x.get("ret") or {}).items():
                    if (t == tag or (tag == "transition" and t.startswith("transition:"))) and len(v) >= 6 \
                            and v[4] is not None and v[5] is not None:
                        sh += abs(v[4]) * (v[3] or 0)
                        te += abs(v[5]) * (v[3] or 0)
            shares.append(pct(te, sh + te, 1) if sh + te > 0 else "-")
        d["terminal_share"] = shares
        if any(c != "-" for c in rcells):
            ret_rows.append([r, len(ppo)] + rcells + shares)
        for key in ("entropy", "log_std", "kl", "v_loss", "clipfrac"):
            d[key] = window_stats(ppo, lambda x, key=key: x.get(key))
        ret_air = window_stats(ppo, lambda x: ((x.get("ret") or {}).get("air") or [None])[0])
        d["return_air"] = ret_air
        stopped = sum(1 for x in ppo if x.get("stopped_early"))
        skipped = sum(1 for x in ev[r]["ppo"] if x["skipped"])
        n_air_traj = mean([((x["rbt"] or {}).get("air") or [None, None, None])[2] for x in ppo])
        # generations in which any crossing trajectory carried a nonzero terminal reward
        nz = 0
        ntraj = 0
        for x in ppo:
            hit = False
            for tag in TAGS[3:]:
                v = (x["rbt"] or {}).get(tag)
                if v:
                    ntraj += v[2]
                    if (v[0] or 0) != 0 or (v[1] or 0) != 0:
                        hit = True
            nz += int(hit)
        d["crossing_reward_gens"] = nz
        d["crossing_trajectories"] = ntraj
        rows.append([r, len(ppo)] + cells + [f"{nz}/{len(ppo)} ({ntraj} crossing trajectories)"])
        def ss(k, dd=d):
            s = dd.get(k)
            return "-" if not s else f"{f3(s['first'], 3)} -> {f3(s['last'], 3)}"
        srows.append([r, len(ppo), skipped, ss("entropy"), ss("log_std"), f3((d['kl'] or {}).get('first'), 4) + " -> " + f3((d['kl'] or {}).get('last'), 4) if d['kl'] else "-",
                      ss("v_loss"), ss("clipfrac"), frac(stopped, len(ppo)),
                      "-" if not ret_air else f"{f3(ret_air['first'], 3)} -> {f3(ret_air['last'], 3)}",
                      f3(n_air_traj, 1) if n_air_traj is not None else "-", "UNAVAILABLE"])
        data[r] = d
    hdr = ["run", "PPO updates"] + [f"{t}: first 20% -> last 20% (delta/sd)" for t in TAGS] + \
          ["updates with any nonzero crossing reward"]
    shdr = ["run", "updates", "skipped", "entropy", "log_std", "kl", "v_loss", "clipfrac",
            "stopped early (target KL)", "return_by_tag.air.return", "air trajectories/update", "explained variance"]
    rhdr = ["run", "updates"] + [f"return_by_tag.{t}.return first -> last 20% (delta/sd)" for t in TAGS] + \
           [f"terminal share of abs(return), {t}" for t in ("air", "water", "land", "crossings")]
    return table(hdr, rows), table(shdr, srows), table(rhdr, ret_rows), data


def first_reach(series, frac_=0.9):
    """series: list of (gen, value) with value a running max; gen at which value >= frac*final."""
    if not series:
        return None, None
    final = series[-1][1]
    if final <= 0:
        return 0.0, None
    for g, v in series:
        if v >= frac_ * final:
            return final, g
    return final, None


def t5(ev, ar):
    rows, data = [], {}
    for r in RUNS_ALL:
        isl_ser = ev[r]["island_series"]
        last_gen = ev[r]["T1_gen"]["gen_max"]
        for isl, ser in isl_ser.items():
            pts = sorted((int(g), v) for g, v in ser.items())
            run_max, rm = [], 0.0
            for g, v in pts:
                rm = max(rm, v[0])
                run_max.append((g, rm))
            final, g90 = first_reach(run_max)
            _, g50 = first_reach(run_max, 0.5)
            # floor-member counts: snapshots + the final archive
            snaps = (ar[r].get("snapshots") or {}).get(isl, [])
            fin = (ar[r]["islands"].get(isl) or {})
            cnt = [(s["gen"], s["floor_member"]) for s in snaps]
            if fin:
                cnt.append((fin.get("generation"), (fin.get("counts") or {}).get("floor_member", 0)))
            cnt = [c for c in cnt if c[0] is not None]
            if len(cnt) < 2:
                fv = cnt[-1][1] if cnt else None
                rows.append([r, ABBR.get(isl, isl), "+".join(ISLAND_MEDIA.get(isl, ())), len(pts), f3(final, 3),
                             g50 if g50 is not None else "-", g90 if g90 is not None else "-", last_gen,
                             f"{fv} (final archive only)", "UNAVAILABLE (no snapshots)"])
                data.setdefault(r, {})[isl] = {"final_own_max": final, "gen50": g50, "gen90": g90,
                                               "last_gen": last_gen, "floor_count_final": fv,
                                               "floor_count_series": cnt}
                continue
            if cnt:
                cfinal = cnt[-1][1]
                cmax = max(c[1] for c in cnt)
                cg90 = next((g for g, v in cnt if cfinal > 0 and v >= 0.9 * cfinal), None)
            else:
                cfinal = cmax = cg90 = None
            media = "+".join(ISLAND_MEDIA.get(isl, ()))
            rows.append([r, ABBR.get(isl, isl), media, len(pts), f3(final, 3),
                         g50 if g50 is not None else "-", g90 if g90 is not None else "-", last_gen,
                         (f"{cfinal} (max {cmax}, {len(cnt)} points)" if cnt else "UNAVAILABLE (no snapshots)"),
                         cg90 if cg90 is not None else "-"])
            data.setdefault(r, {})[isl] = {"final_own_max": final, "gen50": g50, "gen90": g90, "last_gen": last_gen,
                                           "floor_count_final": cfinal, "floor_count_max": cmax, "floor_count_gen90": cg90,
                                           "floor_count_series": cnt}
    hdr = ["run", "island", "own media (min over)", "gens visited", "final running max own competence",
           "gen reaching 50%", "gen reaching 90%", "last gen", "N5 floor members final", "gen reaching 90% of final count"]
    return table(hdr, rows), data


def t6(ev):
    rows, crows, lrows, data = [], [], [], {}
    for r in RUNS_ALL:
        so = ev[r]["seg_outcome"]
        a, w, l_ = so["air"], so["water"], so["land"]
        n = a.get("n", 0)
        pair = a.get("pair_present", 0)
        rows.append([r, n,
                     pct(a.get("measured", 0), n), pct(a.get("airborne_too_short", 0), n), pct(a.get("static_only", 0), n),
                     pct(a.get("no_record", 0), n), pct(a.get("air_gate_failed", 0), n),
                     (frac(a.get("pair_measurable", 0), n) if pair else "n/a (pre-pair)"),
                     (frac(a.get("pair_mean_positive", 0), n) if pair else "n/a"),
                     pct(a.get("competence_ge_floor", 0), n, 2),
                     pct(w.get("both_phases", 0), n), pct(w.get("one_phase", 0) + w.get("no_phase", 0), n),
                     pct(w.get("no_record", 0), n), pct(w.get("competence_ge_floor", 0), n, 2),
                     pct(l_.get("both_phases", 0), n), pct(l_.get("one_phase", 0) + l_.get("no_phase", 0), n),
                     pct(l_.get("no_record", 0), n), pct(l_.get("competence_ge_floor", 0), n, 2)])
        ko = ev[r]["kind_outcome"]
        cells = []
        cross_div = 0
        for k in KINDS:
            c = ko[k]
            nrec = sum(v for kk, v in c.items() if kk != "no_record")
            cells.append(f"{pct(c.get('battery', 0), nrec)} / {pct(c.get('unstable', 0), nrec)} / {pct(c.get('diverged', 0), nrec)}")
            cross_div += c.get("unstable", 0) + c.get("diverged", 0)
        g = ev[r]["T1_gen"]
        crows.append([r] + cells + [frac(g["diverged_rollouts"], g["rollouts"], 2),
                                    f"{cross_div} of {g['diverged_rollouts']}"])
        pr = ev[r]["promote"]
        lf = collections.Counter()
        for x in pr:
            s = x["leg_fail"] or (("battery: " + x["battery"]) if x["battery"] else "no leg failed")
            if s.startswith("leg "):
                leg = s.split(" ")[1]
                med = s.split("(")[1].split(")")[0] if "(" in s else "?"
                why = s.split(": ", 1)[1] if ": " in s else "?"
                lf[f"leg {leg} {med}: {why}"] += 1
            else:
                lf[s[:50]] += 1
        t15 = collections.Counter((x["t15"][3] or "ok") for x in pr)
        lrows.append([r, len(pr), "; ".join(f"{k} {v}" for k, v in lf.most_common(6)),
                      "; ".join(f"{k} {v}" for k, v in t15.most_common(4))])
        data[r] = {"seg_outcome": so, "leg_failures": dict(lf), "tier1_5_failures": dict(t15)}
    hdr = ["run", "Tier-1 evals", "air: measured (>= 2.8 s airborne)", "air: airborne < 2.8 s", "air: static only",
           "air: no record", "air gate failed (Tier-0 airworthiness)", "air pair measurable (factor > 0.5)",
           "air pair mean > 0", "air >= floor", "water: both phases measured", "water: one/no phase",
           "water: no record", "water >= floor", "land: both phases", "land: one/no phase", "land: no record",
           "land >= floor"]
    chdr = ["run"] + [f"{k}: battery / unstable / diverged" for k in KINDS] + \
           ["diverged+unstable rollouts (gen lines, all 7 per eval)", "of which crossing probes (notes)"]
    lhdr = ["run", "Tier-2 promotions", "continuous-mission (Tier-2) first failure", "tier1_5 failure"]
    return table(hdr, rows), table(chdr, crows), table(lhdr, lrows), data


def t7(ev, ar):
    rows, rrows, prow, data = [], [], [], {}
    for r in RUNS_ALL:
        e = ev[r]
        n_el = sum(s["n"] for s in ar[r]["islands"].values())
        phys = e["phys"].get("evals", 0)
        reev = e["T1_events"]["kinds_kept"].get("reevaluate", 0)
        re_st = collections.Counter()
        for isl, c in e["reeval"].items():
            re_st.update(c)
        cells = sum(e["reeval_cells"].values())
        dl = collections.Counter()
        for s in ar[r]["islands"].values():
            for k, v in s["draws_len"].items():
                dl[int(k)] += v
        nd = sum(dl.values())
        mdraw = (sum(k * v for k, v in dl.items()) / nd) if nd else None
        ge2 = sum(v for k, v in dl.items() if k >= 2)
        rows.append([r, n_el, phys, reev, f3(phys / n_el, 1) if n_el else "-",
                     f3((phys + reev) / n_el, 1) if n_el else "-",
                     ", ".join(f"{k} {v}" for k, v in re_st.most_common()) or "-",
                     cells or "-", (f3(mdraw, 2) if nd and mdraw else "no draws buffer"),
                     (frac(ge2, nd) if nd and mdraw else "-"), (max(dl) if dl and mdraw else "-")])
        # retest and retention, island media only
        rt = []
        for isl, s in ar[r]["islands"].items():
            for k in ISLAND_MEDIA.get(isl, ()):
                x = s["retest"][k]
                ret = s["retention"]
                if x["pairs"]:
                    rt.append((isl, k, x["pairs"], x["r"], x["first_ge_floor"], x["later_ge_floor_given_first"],
                               ret.get(f"{k}_recorded_ge_floor", 0), ret.get(f"{k}_draws_ge_floor", 0), ret.get(f"{k}_draws", 0)))
        for isl, k, npair, rr, fge, lge, rec, dge, dn in rt:
            rrows.append([r, ABBR.get(isl, isl), k, npair, f3(rr, 3) if rr is not None else "-",
                          frac(lge, fge) if fge else "0 first draws >= floor",
                          f"{rec} elites; {frac(dge, dn)} of their draws >= floor" if rec else "-"])
        # Tier-1 vs Tier-2 per medium, from promote events
        pr = e["promote"]
        cors = []
        for m in ("air", "water", "land"):
            xs = [((x["t1m"] or {}).get(m)) for x in pr if x["t1m"] and x["t2m"]]
            ys = [((x["t2m"] or {}).get(m)) for x in pr if x["t1m"] and x["t2m"]]
            c = pearson(xs, ys)
            cors.append(f"{f3(c, 2) if c is not None else '-'} (n={len(xs)})")
        prow.append([r, len(pr)] + cors)
        data[r] = {"elites": n_el, "phys_evals": phys, "reevals": reev, "reeval_status": dict(re_st),
                   "cells_reevaluated": cells, "draws_len": dict(dl), "retest": rt}
    hdr = ["run", "final elites (all islands)", "Tier-1 evals", "re-evaluations", "evals per elite",
           "(evals + re-evals) per elite", "re-evaluation status", "cells re-evaluated (distinct per island)",
           "mean draws per elite (archive buffer)", "elites with >= 2 draws", "max draws"]
    rhdr = ["run", "island", "medium", "draw pairs (first vs each later)", "Pearson r",
            "later draw >= floor given first >= floor", "elites recorded >= floor: their draws >= floor"]
    phdr = ["run", "promotions", "r(Tier-1, Tier-2) air", "water", "land"]
    return table(hdr, rows), table(rhdr, rrows), table(phdr, prow), data


def t8(ar):
    rows, data = [], {}
    for r in RUNS_ALL:
        plans = collections.Counter()
        rot = collections.Counter()
        rh = collections.Counter()
        n = 0
        for isl, s in ar[r]["islands"].items():
            for k, v in s["by_plan"].items():
                plans[k] += v["n"]
            for k, v in s["by_rotor"].items():
                rot[k] += v["n"]
            for k, v in s["rotors_hist"].items():
                rh[k] += v
            n += s["n"]
        rows.append([r, n, ", ".join(f"{k} {v}" for k, v in plans.most_common()),
                     (frac(rot.get("rotor", 0), n) if "unknown" not in rot else "UNAVAILABLE (n_rotors not recorded)"),
                     ", ".join(f"{k}:{v}" for k, v in sorted(rh.items(), key=lambda kv: (kv[0] == 'unknown', kv[0])))])
        data[r] = {"n": n, "plans": dict(plans), "rotor": dict(rot), "rotors_hist": dict(rh)}
    hdr = ["run", "elites (summed over islands)", "body plans", "elites with >= 1 rotor", "rotor count histogram (9 = 9+)"]
    return table(hdr, rows), data


def headline(ev, ar, j):
    """The numbers quoted in the report's summary, each with its denominator."""
    x1 = [f"arch{n}" for n in range(47, 61)]
    x0 = [f"arch{n}" for n in range(43, 47)]
    T2 = j["T2"]
    pk = T2["pooled_X1"]
    tot = T2["X1_totals"]
    lines = []
    cr = {k: pk[k].get("crossed", 0) for k in KINDS}
    nrec = sum(v for kk, v in pk["air_to_water"].items() if kk != "no_record")
    lines.append(f"- X1 (arch47-60): {tot['crossings']} crossings in {tot['evals']} Tier-1 evaluations "
                 f"({pct(tot['ge1'], tot['evals'], 3)} with >= 1), 0 evaluations with >= 2; by kind "
                 + ", ".join(f"{k} {cr[k]}" for k in KINDS) + ".")
    a2w = pk["air_to_water"]
    lines.append(f"- X1 air_to_water: reached the water before the command (hold < 0.5) in "
                 f"{frac(a2w.get('crossed_without_hold', 0), nrec)}; never crossed {frac(a2w.get('never_crossed', 0), nrec)}; "
                 f"aborted {frac(a2w.get('battery', 0) + a2w.get('unstable', 0) + a2w.get('diverged', 0), nrec)}.")
    x0c = sum(T2[r]["crossings"] for r in x0)
    x0e = sum(T2[r]["evals"] for r in x0)
    x0ge2 = sum(T2[r]["evals_ge2_cross"] for r in x0)
    lines.append(f"- X0 (arch43-46, before the 10-03 commanded crossing): {x0c} crossings in {x0e} evaluations; "
                 f"{frac(x0ge2, x0e)} had >= 2.")
    mf_x1 = tot["mf_pos"]
    mfmax = max(T2[r]["mf_max"] for r in x1)
    lines.append(f"- Tier-1 mission_fraction > 0: X1 {frac(mf_x1, tot['evals'], 3)} (max {mfmax:.4f}); "
                 f"X0 {frac(sum(T2[r]['mf_pos'] for r in x0), x0e)} (max {max(T2[r]['mf_max'] for r in x0):.4f}).")
    t2p = sum(T2[r]["tier2_pos"] for r in RUNS_ALL)
    t2n = sum(T2[r]["promotions"] for r in RUNS_ALL)
    t2max = max((T2[r]["tier2_max"] or 0) for r in RUNS_ALL)
    t2two = sum(T2[r]["tier2_ge_2_legs"] for r in RUNS_ALL)
    lines.append(f"- Tier-2 mission (9 legs, 6 in arch45/46; each 25 s simulated = 300 s / compression 12, re-placed "
                 f"in its medium, passed iff competence >= 0.15; no crossings): tier2_fraction > 0 in "
                 f"{frac(t2p, t2n)} promotions over arch43-60 (X1: {frac(tot['t2pos'], tot['promotions'])}); "
                 f"max {t2max:.3f} (= 1 leg of 6); promotions with >= 2 legs passed: {t2two}.")
    lc = collections.Counter()
    for r in RUNS_ALL:
        for p in ev[r]["promote"]:
            if p["leg_fail"]:
                lc["failed leg " + p["leg_fail"].split(" ")[1]] += 1
            elif p["battery"]:
                lc["battery after leg 0"] += 1
            else:
                lc["no failure note"] += 1
    lines.append("- Tier-2 stopping points over all promotions of arch43-60: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(lc.items())) + f" (total {sum(lc.values())}).")
    fl = []
    for r in RUNS_ALL:
        fm = T2[r]["film"]
        fl.append(f"{r[4:]} {'-' if fm is None else fm.get('transitions')}")
    lines.append("- Film manifests, continuous mission transitions (one design per run; '0/0' = film_error "
                 "'no GL context', '-' = no manifest): " + ", ".join(fl) + ".")
    ap = []
    for r in x1:
        e = ev[r]
        nh = {int(k): v for k, v in e["ncross_hist"].items()}
        zero = nh.get(0, 0)
        ap.append(e["phys"].get("approach_pos_no_cross", 0) / zero if zero else None)
    ap = [x for x in ap if x is not None]
    lines.append(f"- X1 evaluations with no crossing but graded approach > 0 (some nonzero hold): "
                 f"{100 * min(ap):.1f}%-{100 * max(ap):.1f}% per run.")
    T4 = j["T4"]
    mv_r, mv_t = [], []
    for r in RUNS_ALL:
        for tag in TAGS:
            s = T4[r].get(tag)
            if s and s.get("delta_over_sd") is not None:
                mv_r.append((abs(s["delta_over_sd"]), r, tag, s["delta_over_sd"]))
            s = T4[r].get(f"return:{tag}")
            if s and s.get("delta_over_sd") is not None:
                mv_t.append((abs(s["delta_over_sd"]), r, tag, s["delta_over_sd"]))
    for lab, mv in (("reward_by_tag", mv_r), ("return_by_tag", mv_t)):
        mv.sort(reverse=True)
        lines.append(f"- {lab} first-to-last-fifth move, all (run, tag) pairs with a defined spread: "
                     f"{sum(1 for x in mv if x[0] < 1)}/{len(mv)} under 1 sd, {sum(1 for x in mv if x[0] >= 1.5)} at >= 1.5 sd; "
                     "largest " + ", ".join(f"{r} {t} {d:+.2f}" for _, r, t, d in mv[:4]) + ".")
    T3 = j["T3"]
    pe = collections.Counter()
    for r in [f"arch{n}" for n in range(51, 60)]:
        pe.update(T3["evals"][r])
    lines.append(f"- Co-occurrence, A1/W1 Tier-1 evaluations arch51-59: >= 2 media {frac(pe['floor_media_2'] + pe['floor_media_3'], pe['evals'], 2)}, "
                 f"all 3 {frac(pe['floor_media_3'], pe['evals'], 3)}; final archives arch51-59: "
                 f">= 2 {frac(T3['archives_arch51_59']['m2'] + T3['archives_arch51_59']['m3'], T3['archives_arch51_59']['n'])}, "
                 f"all 3 {frac(T3['archives_arch51_59']['m3'], T3['archives_arch51_59']['n'], 2)}.")
    pe0 = collections.Counter()
    for r in [f"arch{n}" for n in range(43, 49)]:
        pe0.update(T3["evals"][r])
    lines.append(f"- Co-occurrence, A0/W0 Tier-1 evaluations arch43-48 (not comparable with the line above): "
                 f">= 2 media {frac(pe0['floor_media_2'] + pe0['floor_media_3'], pe0['evals'], 2)}, all 3 "
                 f"{frac(pe0['floor_media_3'], pe0['evals'], 3)}.")
    ts = [T4[r].get("terminal_share") for r in [f"arch{n}" for n in range(51, 61)] if T4[r].get("terminal_share")]
    lines.append("- Learner (arch51-60): terminal (competence) share of the return magnitude, air "
                 + ", ".join(t[0] for t in ts) + "; crossings " + ", ".join(t[3] for t in ts) + ".")
    nz = [(r, T4[r]["crossing_reward_gens"], len([x for x in ev[r]["ppo"] if not x["skipped"]])) for r in x1]
    lines.append("- Updates with any nonzero crossing reward, X1: "
                 + ", ".join(f"{r[4:]} {a}/{b}" for r, a, b in nz) + "; X0: every update.")
    T1 = j["T1"]
    lines.append(f"- Scale: {sum(T1[r]['phys_evals'] for r in RUNS_ALL)} Tier-1 evaluations (seeds once) and "
                 f"{sum(T1[r]['reevaluate'] for r in RUNS_ALL)} re-evaluations on the surviving timelines of the 18 runs; "
                 f"{sum(T1[r]['wall_h_all'] for r in RUNS_ALL):.0f} process-hours to the last generation line "
                 f"({sum(T1[r]['wall_h_kept'] for r in RUNS_ALL):.0f} h on kept generations).")
    lt = [(r, T1[r]["learner_transitions"]) for r in RUNS_ALL]
    lines.append("- Learner transitions per run (PPO buffer, all updates): "
                 + ", ".join(f"{r[4:]} {v:.2g}" for r, v in lt) + ".")
    return "\n".join(lines)


def fill(report: Path, blocks: dict) -> int:
    """Replace the text between <!-- BEGIN name --> and <!-- END name --> markers."""
    if not report.exists():
        return 0
    txt = report.read_text()
    n = 0
    for name, body in blocks.items():
        b, e = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
        i, k = txt.find(b), txt.find(e)
        if i < 0 or k < 0:
            continue
        txt = txt[:i + len(b)] + "\n" + body + "\n" + txt[k:]
        n += 1
    report.write_text(txt)
    return n


def main():
    ev, ar, seg, voc, ck, film = load()
    j, B = {}, {}
    B["T1"], j["T1"] = t1(ev, ar, seg, voc, ck)
    B["T2a"], B["T2b"], B["T2c"], B["T2d"], j["T2"] = t2(ev, film)
    (B["T3a"], B["T3b"], B["T3c"], B["T3d"], B["T3e"], B["T3f"], B["T3g"], j["T3"]) = t3(ev, ar)
    B["T4a"], B["T4b"], B["T4c"], j["T4"] = t4(ev)
    B["T5"], j["T5"] = t5(ev, ar)
    B["T6a"], B["T6b"], B["T6c"], j["T6"] = t6(ev)
    B["T7a"], B["T7b"], B["T7c"], j["T7"] = t7(ev, ar)
    B["T8"], j["T8"] = t8(ar)
    B["HEADLINE"] = headline(ev, ar, j)
    j["headline"] = B["HEADLINE"]
    (RES / "a8_tables.json").write_text(json.dumps(j, indent=1, default=str))
    report = RES.parent / "reports" / "A8_run_forensics.md"
    n = fill(report, B)
    print("blocks", len(B), "filled in report:", n)
    print(B["HEADLINE"])


if __name__ == "__main__":
    main()
