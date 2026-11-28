"""A8 pass B: one streaming pass over each run's generations.jsonl, events.jsonl and
exploits.jsonl, on the surviving timeline (a8_segments.py), aggregated as it goes.

Nothing is held per evaluation except fixed-size counters, a top-10 heap and
per-generation series (<= 900 entries).  Output: results/events_<run>.json.

Field meanings relied on (source in the repo):
  generation line  rollouts / diverged_rollouts: per generation, segments + crossings
                   whose failure is diverged|unstable (envs/evaluate.py:242-248,
                   evolution/loop.py:2178-2193); evaluated / tier0_rejected: cumulative
                   state counters (loop.py:2401-2402); mission_best: generalist island
                   only; weakest_best / three_media: triphibian island only
                   (loop.py:2405-2431); cost.seconds: per phase, keys with '.' are
                   sub-phases.
  evaluate event   one _place() call (loop.py:1431-1566); air/water/land = segment
                   competence rounded to 3 dp, mission_fraction = Tier-1 product rounded
                   to 4 dp (loop.py:1223-1255, envs/evaluate.py:168-223); notes =
                   result.notes[:8]; crossing notes are "<kind>: <failure>"
                   (envs/batchroll.py:1253, envs/transitions.py:496-683);
                   ladder_measurements.transition.crossed_fraction = crossed / 4
                   (transitions.py:805), .crossed = mean graded approach.
                   Seeds (operators == ["seed"]) are one evaluation filed on every
                   island (loop.py:2996-3004), so they are de-duplicated on
                   (eval_seed, body_plan, mass) for every physical count.
  ppo event        reward_by_tag[tag] = [mean, std, n] of the terminal reward of the
                   trajectories the update saw (loop.py:2257-2271); return_by_tag (N10)
                   = discounted return split shaping/terminal (learning/ppo.py:879-884).
  promote event    Tier-2: tier2_fraction = legs completed / legs (9 legs, each re-placed
                   in its medium, no crossings) (envs/evaluate.py:575-635).

Usage: nice -n 10 .venv/bin/python experiments/triphibian_diagnosis/scripts/a8_events.py [run ...]
"""
import collections
import heapq
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
RES = Path(__file__).resolve().parents[1] / "results"
DEFAULT = [f"arch{n}" for n in range(43, 61)]
FLOOR = 0.012  # evolution/islands.py:144 COMPETENCE_FLOOR = curriculum.WEAKEST_BARS[1]
KINDS = ("air_to_water", "water_to_air", "water_to_land", "land_to_air")
ISLAND_MEDIA = {  # evolution/islands.py ISLANDS[...]["domains"]
    "air": ("air",), "water": ("water",), "land": ("land",),
    "amphibian": ("water", "land"), "aerial_diver": ("air", "water"),
    "land_air": ("land", "air"), "triphibian": ("air", "water", "land"),
    "generalist": ("air", "water", "land"),
}
MEDIA = ("air", "water", "land")


def note_class(failure):
    f = failure.strip()
    if f.startswith("never crossed"):
        return "never_crossed"
    if f.startswith("did not hold"):
        return "crossed_without_hold"
    if f.startswith("battery exhausted"):
        return "battery"
    if f == "unstable":
        return "unstable"
    if f == "diverged":
        return "diverged"
    if f.startswith("did not start"):
        return "did_not_start"
    return "other:" + f[:40]


class Keep:
    """Surviving-timeline filter for one stream, driven by t resets / run_start lines."""

    def __init__(self, keep_below):
        self.kb = keep_below
        self.seg = -1
        self.t_prev = None

    def event(self, t, gen):
        if self.seg < 0 or (self.t_prev is not None and t < self.t_prev - 1.0):
            self.seg += 1
        self.t_prev = t
        kb = self.kb[self.seg] if self.seg < len(self.kb) else 10 ** 9
        return gen is None or gen < kb


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def scan(run, segs):
    d = RUNS / run
    out = {"run": run}
    # The evaluate event kept result.notes[:3] until eb9b79d (2026-10-08, "the
    # evaluate event keeps eight notes, not three (land_to_air was cut)"), first in
    # arch50's commit 2b7bb53; arch49's 7a894b7 predates it.
    note_cap = 3 if int(run[4:]) <= 49 else 8
    out["note_cap"] = note_cap

    # ---------------- generations.jsonl ----------------
    gkb = segs["gen_keep_below"]
    seg = -1
    gen_rows = {}
    config = None
    seg_t = []
    discarded_gen_lines = 0
    with open(d / "generations.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "run_start":
                seg += 1
                seg_t.append({"t_last": 0.0, "lines": 0})
                if config is None:
                    config = r.get("config") or {}
                continue
            if seg < 0:
                seg = 0
                seg_t.append({"t_last": 0.0, "lines": 0})
            g = r.get("generation")
            t = float(r.get("t") or 0.0)
            prev_t = seg_t[seg]["t_last"]
            seg_t[seg]["t_last"] = t
            seg_t[seg]["lines"] += 1
            if not (isinstance(g, int) and g < gkb[seg]):
                discarded_gen_lines += 1
                continue
            cs = (r.get("cost") or {}).get("seconds") or {}
            st = (r.get("cost") or {}).get("steps") or {}
            jb = (r.get("judge") or {}).get("bars") or {}
            gen_rows[g] = {
                "island": r.get("island"),
                "evaluated": r.get("evaluated"), "tier0_rejected": r.get("tier0_rejected"),
                "rollouts": r.get("rollouts") or 0, "diverged": r.get("diverged_rollouts") or 0,
                "dt": round(t - prev_t, 2),
                "cost_s": round(sum(v for k, v in cs.items() if "." not in k and isinstance(v, (int, float))), 2),
                "cost_ppo": cs.get("ppo"), "cost_eval": cs.get("evaluate"),
                "steps": {k: v for k, v in st.items()},
                "mission_best": r.get("mission_best"), "weakest_best": r.get("weakest_best"),
                "three_media": r.get("three_media"),
                "filled": r.get("filled"), "coverage": r.get("coverage"),
                "judge_transition": jb.get("transition"),
                "judge_bars": {k: (v or {}).get("bar") for k, v in jb.items()},
                "resident_mb": r.get("resident_mb"),
                "invalidated": (r.get("auditor") or {}).get("invalidated"),
            }
    gens = sorted(gen_rows)
    out["config"] = {k: (config or {}).get(k) for k in (
        "generations", "batch", "seed", "workers", "segment_seconds", "islands", "evaluator",
        "use_shared_policy", "shared_learner", "shared_hidden", "shared_lr", "shared_epochs",
        "shared_minibatch", "shared_target_kl", "shared_ent_coef", "shared_log_std",
        "reeval_per_generation", "reeval_depth", "placement_draws", "draw_per_candidate",
        "mission_weight", "reward_shaping", "air_turn_reward", "controller_refine_sigma",
        "floor_tiebreak", "k_mean", "k_sampled", "memory_ceiling_mb", "tier2_every",
        "controller_refine_steps", "promotion_refine_steps", "grpo_group", "grpo_bodies")}
    steps_tot = collections.Counter()
    for g in gens:
        for k, v in (gen_rows[g]["steps"] or {}).items():
            if isinstance(v, (int, float)):
                steps_tot[k] += v
    last = gen_rows[gens[-1]] if gens else {}
    out["T1_gen"] = {
        "gens_kept": len(gens), "gen_min": gens[0] if gens else None, "gen_max": gens[-1] if gens else None,
        "gen_lines_discarded": discarded_gen_lines, "processes": len(seg_t),
        "islands_visited": dict(collections.Counter(gen_rows[g]["island"] for g in gens)),
        "evaluated_last": last.get("evaluated"), "tier0_rejected_last": last.get("tier0_rejected"),
        "rollouts": sum(gen_rows[g]["rollouts"] for g in gens),
        "diverged_rollouts": sum(gen_rows[g]["diverged"] for g in gens),
        "wall_h_all_processes": round(sum(s["t_last"] for s in seg_t) / 3600, 2),
        "wall_h_kept_gens": round(sum(gen_rows[g]["dt"] for g in gens) / 3600, 2),
        "wall_h_cost_seconds": round(sum(gen_rows[g]["cost_s"] for g in gens) / 3600, 2),
        "ppo_seconds": round(sum((gen_rows[g]["cost_ppo"] or 0) for g in gens), 1),
        "median_gen_s": (sorted(gen_rows[g]["dt"] for g in gens)[len(gens) // 2] if gens else None),
        "steps_total": dict(steps_tot),
        "mission_best_max": max((gen_rows[g]["mission_best"] or 0.0) for g in gens) if gens else None,
        "mission_best_present": sum(gen_rows[g]["mission_best"] is not None for g in gens),
        "weakest_best_max": max((gen_rows[g]["weakest_best"] or 0.0) for g in gens) if gens else None,
        "weakest_best_present": sum(gen_rows[g]["weakest_best"] is not None for g in gens),
        "three_media_max": max((gen_rows[g]["three_media"] or 0) for g in gens) if gens else None,
        "judge_transition_last": last.get("judge_transition"),
        "judge_bars_last": last.get("judge_bars"),
        "resident_mb_present": sum(gen_rows[g]["resident_mb"] is not None for g in gens),
        "resident_mb_max": max((gen_rows[g]["resident_mb"] or 0) for g in gens) if gens else None,
        "resident_mb_last": next((gen_rows[g]["resident_mb"] for g in reversed(gens)
                                  if gen_rows[g]["resident_mb"] is not None), None),
        "auditor_invalidated_last": last.get("invalidated"),
    }
    out["gen_series"] = {"gen": gens,
                         "island": [gen_rows[g]["island"] for g in gens],
                         "rollouts": [gen_rows[g]["rollouts"] for g in gens],
                         "diverged": [gen_rows[g]["diverged"] for g in gens],
                         "dt": [gen_rows[g]["dt"] for g in gens]}

    # ---------------- events.jsonl ----------------
    keep = Keep(segs["event_keep_below"])
    kinds_kept, kinds_disc = collections.Counter(), collections.Counter()
    status_kept = collections.Counter()
    first_op = collections.Counter()
    seeds_seen = set()
    seed_events = 0
    # physical (seed-deduplicated) evaluation aggregates
    P = collections.Counter()
    kind_outcome = {k: collections.Counter() for k in KINDS}
    ncross_hist = collections.Counter()
    ncross_by_gen = collections.Counter()
    mismatch = collections.Counter()
    notes_len = collections.Counter()
    media_by_plan = collections.defaultdict(collections.Counter)
    media_by_rotor = collections.defaultdict(collections.Counter)
    seg_outcome = {m: collections.Counter() for m in MEDIA}
    crossers = []  # small: evaluations with >= 1 crossing
    top10 = []  # heap of (min competence, ...)
    mf_pos_by_gen = collections.Counter()
    # per island (not deduplicated: an island's own filing)
    isl = collections.defaultdict(dict)  # island -> gen -> [max_own, max_a, max_w, max_l, n, n_own_floor]
    reeval = collections.defaultdict(collections.Counter)
    reeval_cells = collections.defaultdict(set)
    reeval_draws_hist = collections.Counter()
    promote = []
    ppo_rows = []
    memory_stops = []
    other_kinds = collections.Counter()
    with open(d / "events.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = r.get("kind", "?")
            g = r.get("gen")
            if not keep.event(float(r.get("t") or 0.0), g if isinstance(g, int) else None):
                kinds_disc[k] += 1
                continue
            kinds_kept[k] += 1
            if k == "evaluate":
                status_kept[str(r.get("status"))] += 1
                ops = r.get("operators") or []
                first_op[(ops[0] if ops else "?").split(":")[0]] += 1
                island = r.get("island")
                a = fnum(r.get("air")) or 0.0
                w = fnum(r.get("water")) or 0.0
                l_ = fnum(r.get("land")) or 0.0
                comp = {"air": a, "water": w, "land": l_}
                # --- per island series (every filing) ---
                media = ISLAND_MEDIA.get(island, MEDIA)
                own = min(comp[m] for m in media)
                row = isl[island].get(g)
                if row is None:
                    row = isl[island][g] = [0.0, 0.0, 0.0, 0.0, 0, 0, 0]
                row[0] = max(row[0], own)
                row[1] = max(row[1], a)
                row[2] = max(row[2], w)
                row[3] = max(row[3], l_)
                row[4] += 1
                row[5] += int(any(comp[m] >= FLOOR for m in media))   # N5 floor member
                row[6] += int(all(comp[m] >= FLOOR for m in media))   # all own media
                # --- physical evaluation (seeds once) ---
                is_seed = ops == ["seed"]
                if is_seed:
                    seed_events += 1
                    key = (r.get("eval_seed"), r.get("body_plan"), r.get("mass"))
                    if key in seeds_seen:
                        continue
                    seeds_seen.add(key)
                P["evals"] += 1
                plan = r.get("body_plan") or "?"
                nr = r.get("n_rotors")
                rot = ("unknown" if nr is None else
                       "0" if nr == 0 else "1-2" if nr <= 2 else "3-5" if nr <= 5 else "6+")
                mf = fnum(r.get("mission_fraction")) or 0.0
                if mf > 0:
                    P["mf_pos"] += 1
                    mf_pos_by_gen[g] += 1
                P["mf_max_x1e4"] = max(P["mf_max_x1e4"], int(round(mf * 1e4)))
                if min(a, w, l_) > 0:
                    P["all3_pos"] += 1
                nfl = sum(c >= FLOOR for c in (a, w, l_))
                P[f"floor_media_{nfl}"] += 1
                for c, m in ((a, "air"), (w, "water"), (l_, "land")):
                    if c >= FLOOR:
                        P[f"floor_{m}"] += 1
                pair_flags = [("air_water", a, w), ("air_land", a, l_), ("water_land", w, l_)]
                for nm, x, y in pair_flags:
                    if x >= FLOOR and y >= FLOOR:
                        P[f"floor_{nm}"] += 1
                for bucket, tab in ((plan, media_by_plan), (rot, media_by_rotor)):
                    tab[bucket]["n"] += 1
                    tab[bucket][f"m{nfl}"] += 1
                    for c, m in ((a, "air"), (w, "water"), (l_, "land")):
                        tab[bucket][m] += int(c >= FLOOR)
                mn = min(a, w, l_)
                item = (mn, a, w, l_, g, island, plan, r.get("mass"), nr, r.get("eval_seed"), mf)
                if len(top10) < 10:
                    heapq.heappush(top10, item)
                elif item > top10[0]:
                    heapq.heapreplace(top10, item)
                # --- crossings ---
                lm = r.get("ladder_measurements") or {}
                tr = lm.get("transition") or {}
                cf = fnum(tr.get("crossed_fraction")) or 0.0
                appr = fnum(tr.get("crossed")) or 0.0
                notes = r.get("notes") or []
                notes_len[min(len(notes), 8)] += 1
                trunc = len(notes) >= note_cap
                per = {}
                for n in notes:
                    s = str(n)
                    for kk in KINDS:
                        if s.startswith(kk + ":"):
                            per.setdefault(kk, note_class(s[len(kk) + 1:]))
                n_cross = int(round(4 * cf))
                has_rec = bool(per) or cf > 0
                noteless = [kk for kk in KINDS if kk not in per]
                if not has_rec:
                    P["no_transition_record"] += 1
                    for kk in KINDS:
                        kind_outcome[kk]["no_record"] += 1
                else:
                    P["with_transition_record"] += 1
                    if len(noteless) == n_cross:
                        resolved = {kk: "crossed" for kk in noteless}
                        mismatch["resolved"] += 1
                    elif not trunc:
                        resolved = {kk: "noteless_not_crossed_UNEXPECTED" for kk in noteless}
                        mismatch["unexpected_untruncated"] += 1
                    elif n_cross == 0:
                        resolved = {kk: "not_crossed_note_cut" for kk in noteless}
                        mismatch["cut_none_crossed"] += 1
                    else:
                        resolved = {kk: "ambiguous_note_cut" for kk in noteless}
                        mismatch["cut_ambiguous"] += 1
                    for kk in KINDS:
                        kind_outcome[kk][per.get(kk) or resolved[kk]] += 1
                ncross_hist[n_cross] += 1
                if n_cross:
                    ncross_by_gen[g] += n_cross
                    if len(crossers) < 400:
                        crossers.append({"gen": g, "island": island, "plan": plan, "n_rotors": nr,
                                         "n_crossed": n_cross, "air": a, "water": w, "land": l_,
                                         "kinds_crossed": noteless if len(noteless) == n_cross else None,
                                         "notes": [str(x)[:70] for x in notes],
                                         "operators": ops[:2]})
                if n_cross == 0 and appr > 0:
                    P["approach_pos_no_cross"] += 1
                # --- segment outcome proxies (T6) ---
                la = lm.get("air") or {}
                if "task_score" in la:
                    seg_outcome["air"]["measured"] += 1
                    ptf = fnum(la.get("pair_turn_factor"))
                    if ptf is not None:
                        seg_outcome["air"]["pair_present"] += 1
                        if ptf > 0.5 and la.get("pair_turn_mean") is not None:
                            seg_outcome["air"]["pair_measurable"] += 1
                            if (fnum(la.get("pair_turn_mean")) or 0.0) > 0:
                                seg_outcome["air"]["pair_mean_positive"] += 1
                elif "airborne_seconds" in la:
                    seg_outcome["air"]["airborne_too_short"] += 1
                elif "spin_rate" in la or "lift_margin" in la:
                    seg_outcome["air"]["static_only"] += 1
                else:
                    seg_outcome["air"]["no_record"] += 1
                if r.get("air_gates"):
                    seg_outcome["air"]["air_gate_failed"] += 1
                lw = lm.get("water") or {}
                if "max_depth" not in lw:
                    seg_outcome["water"]["no_record"] += 1
                else:
                    c1, c2 = "cruise_tracking" in lw, "hold_error" in lw
                    seg_outcome["water"]["both_phases" if c1 and c2 else ("one_phase" if c1 or c2 else "no_phase")] += 1
                ll = lm.get("land") or {}
                if "upright" not in ll:
                    seg_outcome["land"]["no_record"] += 1
                else:
                    c1, c2 = "walk_tracking" in ll, "stop_score" in ll
                    seg_outcome["land"]["both_phases" if c1 and c2 else ("one_phase" if c1 or c2 else "no_phase")] += 1
                for m, c in comp.items():
                    seg_outcome[m]["competence_ge_floor"] += int(c >= FLOOR)
                    seg_outcome[m]["competence_pos"] += int(c > 0)
                    seg_outcome[m]["n"] += 1
            elif k == "reevaluate":
                st = str(r.get("status"))
                island = r.get("island")
                reeval[island][st] += 1
                reeval_cells[island].add(tuple(r.get("cell") or ()))
                dr = r.get("draws")
                if isinstance(dr, int):
                    reeval_draws_hist[dr] += 1
                elif isinstance(dr, dict):
                    reeval_draws_hist["dict"] += 1
            elif k == "promote":
                nts = r.get("notes") or []
                legs_failed = None
                for n in nts:
                    s = str(n)
                    if s.startswith("leg ") and legs_failed is None:
                        legs_failed = s[:90]
                promote.append({
                    "gen": g, "island": r.get("island"),
                    "t2f": r.get("tier2_fraction"), "t1f": r.get("tier1_fraction"),
                    "t2m": r.get("tier2_media"), "t1m": r.get("tier1_media"),
                    "t15": [r.get("tier1_5_domain"), r.get("tier1_5_competence"), r.get("tier1_5_short"),
                            r.get("tier1_5_failure")],
                    "leg_fail": legs_failed,
                    "battery": next((str(n) for n in nts if str(n).startswith("battery")), None),
                    "exploit": r.get("exploit") or ""})
            elif k == "ppo":
                rbt = r.get("reward_by_tag") or {}
                ret = r.get("return_by_tag") or {}
                ppo_rows.append({
                    "gen": g, "island": r.get("island"),
                    "transitions": r.get("transitions"), "trajectories": r.get("trajectories"),
                    "skipped": bool(r.get("skipped")), "pi_loss": r.get("pi_loss"), "v_loss": r.get("v_loss"),
                    "kl": r.get("kl"), "entropy": r.get("entropy"), "clipfrac": r.get("clipfrac"),
                    "lr": r.get("lr"), "log_std": r.get("log_std"), "grad_steps": r.get("grad_steps"),
                    "stopped_early": r.get("stopped_early"),
                    "grpo_transitions": r.get("grpo_transitions"),
                    "rbt": {t: v for t, v in rbt.items()},
                    "ret": {t: [v.get("return"), v.get("shaping_return"), v.get("terminal_return"), v.get("n"),
                                v.get("abs_shaping"), v.get("abs_terminal")]
                            for t, v in ret.items() if isinstance(v, dict)},
                })
            elif k == "tier0_reject":
                ops = r.get("operators") or []
                P["tier0_seed_events" if ops == ["seed"] else "tier0_nonseed"] += 1
            elif k == "memory_stop":
                memory_stops.append({"gen": g, "resident_mb": r.get("resident_mb"), "ceiling_mb": r.get("ceiling_mb")})
            elif k not in ("stages", "stage_wall", "audit", "descriptor_refit",
                           "scout_fit", "judge_tighten", "migrate", "descriptor_refit_skipped",
                           "critic_fit", "judge_veto"):
                other_kinds[k] += 1

    out["T1_events"] = {
        "kinds_kept": dict(kinds_kept), "kinds_discarded": dict(kinds_disc),
        "evaluate_status_kept": dict(status_kept), "first_operator": dict(first_op),
        "seed_events": seed_events, "seeds_physical": len(seeds_seen),
        "memory_stops": memory_stops, "other_kinds": dict(other_kinds),
    }
    out["phys"] = dict(P)
    out["kind_outcome"] = {k: dict(v) for k, v in kind_outcome.items()}
    out["ncross_hist"] = {str(k): v for k, v in sorted(ncross_hist.items())}
    out["ncross_by_gen"] = {str(k): v for k, v in sorted(ncross_by_gen.items())}
    out["mf_pos_by_gen"] = {str(k): v for k, v in sorted(mf_pos_by_gen.items())}
    out["cross_note_consistency"] = dict(mismatch)
    out["notes_len"] = {str(k): v for k, v in sorted(notes_len.items())}
    out["crossers"] = crossers
    out["media_by_plan"] = {k: dict(v) for k, v in media_by_plan.items()}
    out["media_by_rotor"] = {k: dict(v) for k, v in media_by_rotor.items()}
    out["seg_outcome"] = {k: dict(v) for k, v in seg_outcome.items()}
    out["top10_evals"] = [dict(zip(("min", "air", "water", "land", "gen", "island", "plan", "mass",
                                     "n_rotors", "eval_seed", "mission_fraction"), x))
                          for x in sorted(top10, reverse=True)]
    out["island_series"] = {i: {str(gg): v for gg, v in sorted(rows.items())} for i, rows in isl.items()}
    out["reeval"] = {i: dict(c) for i, c in reeval.items()}
    out["reeval_cells"] = {i: len(c) for i, c in reeval_cells.items()}
    out["reeval_draws_hist"] = {str(k): v for k, v in reeval_draws_hist.items()}
    out["promote"] = promote
    out["ppo"] = ppo_rows

    # ---------------- exploits.jsonl ----------------
    # Its processes cannot be told apart (a process that caught no exploit writes no
    # line, so a t reset does not identify the process): total lines only.
    ex = d / "exploits.jsonl"
    n_ex = 0
    if ex.exists():
        with open(ex) as f:
            for line in f:
                n_ex += 1
    out["exploits"] = {"lines_all_processes": n_ex}
    return out


def main():
    runs = sys.argv[1:] or DEFAULT
    segs = json.load(open(RES / "segments.json"))
    for run in runs:
        t0 = time.time()
        rec = scan(run, segs[run])
        (RES / f"events_{run}.json").write_text(json.dumps(rec, separators=(",", ":")))
        p = rec["phys"]
        print(run, f"{time.time() - t0:.1f}s", "evals", p.get("evals"), "mf>0", p.get("mf_pos", 0),
              "ncross", rec["ncross_hist"], "consistency", rec["cross_note_consistency"], flush=True)


if __name__ == "__main__":
    main()
