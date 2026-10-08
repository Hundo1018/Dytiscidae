"""What does one controller-refinement step buy, under which criterion, on which draw?

See README.md for the question, the pre-registered predictions and the decision
rule.  For every elite of a finished run: the base controller and ``T``
single-step trials (``w + N(0, sigma)`` on the elite's own policy weights, the
search's ``_refine_controllers`` step), each scored through
``batchroll.evaluate_tier1_batch`` with the scoring network and no
identification, at the elite's own draw ``s`` and at a fresh draw ``s'``.

    systemd-run --user --unit refine-arch48 -p MemoryMax=3500M --same-dir \
        env MUJOCO_GL=disable PYTHONPATH=. \
        .venv/bin/python experiments/refine_criterion/run.py \
        --run runs/arch48 --out experiments/refine_criterion/results_arch48.json

    python experiments/refine_criterion/run.py --selftest
    python experiments/refine_criterion/run.py --from-rows <out>.rows.jsonl --out <out>
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
CRITERIA = ("mission", "island", "stage")
DRAWS = ("s", "fresh")


def fresh_seed(s: int) -> int:
    """A draw used only by this experiment, derived from the recorded one."""
    return int(np.random.default_rng([int(s), 0xF5E5]).integers(1 << 30))


# --------------------------------------------------------------------------
# Rows -> the table (pure Python + numpy, no simulator)
# --------------------------------------------------------------------------


def _ci(x, seed=0, n_boot=4000):
    x = np.asarray(x, float)
    if x.size == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, x.size)).mean(axis=1)
    return [float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def analyse(rows):
    """Per criterion: acceptance, same-draw gain, fresh-draw gain, the control.

    ``rows``: one dict per elite with ``scores[draw]["base"][criterion]`` and
    ``scores[draw]["trials"][t][criterion]``.
    """
    out = {}
    for c in CRITERIA:
        acc_s, acc_f, ctl_f, n_trials = [], [], [], 0
        per_elite_acc = 0
        for r in rows:
            b_s = r["scores"]["s"]["base"][c]
            b_f = r["scores"]["fresh"]["base"][c]
            any_acc = False
            for t_s, t_f in zip(r["scores"]["s"]["trials"], r["scores"]["fresh"]["trials"]):
                n_trials += 1
                g_s, g_f = t_s[c] - b_s, t_f[c] - b_f
                ctl_f.append(g_f)
                if g_s > 0.0:
                    acc_s.append(g_s)
                    acc_f.append(g_f)
                    any_acc = True
            per_elite_acc += any_acc
        m_s, m_f = (float(np.mean(acc_s)) if acc_s else float("nan"),
                    float(np.mean(acc_f)) if acc_f else float("nan"))
        diff = np.asarray(acc_f, float)
        out[c] = {
            "trials": n_trials,
            "accepted": len(acc_s),
            "accept_share": len(acc_s) / n_trials if n_trials else float("nan"),
            "elites_with_an_acceptance": per_elite_acc,
            "gain_s": _ci(acc_s),
            "gain_fresh": _ci(acc_f),
            "control_fresh": _ci(ctl_f),
            "retained": (m_f / m_s) if acc_s and m_s > 0 else float("nan"),
            "fresh_gain_positive_share": float((diff > 0).mean()) if diff.size else float("nan"),
        }
    return out


def verdicts(table):
    """The README's predictions and decision rule, read off ``analyse``."""
    v = {"P1": table["mission"]["accept_share"] < 0.02}
    v["P2"] = all(table[c]["accept_share"] >= 0.20 for c in ("island", "stage"))
    v["P3"] = all(table[c]["retained"] < 0.5 for c in ("island", "stage")
                  if table[c]["accepted"])
    above = {}
    for c in CRITERIA:
        t = table[c]
        # above the control: accepted fresh-gain CI lower bound > control's mean
        above[c] = bool(t["accepted"] and t["gain_fresh"][1] > t["control_fresh"][0])
    v["P4"] = not any(above[c] for c in ("island", "stage"))
    v["above_control"] = above
    decide = {}
    for c in ("island", "stage"):
        t = table[c]
        if not above[c]:
            decide[c] = "selects noise: refine-steps 0"
        elif t["retained"] >= 0.5:
            decide[c] = "switch criterion, same draw"
        else:
            decide[c] = "switch criterion, report a draw other than the one that chose"
    v["decision"] = decide
    return v


def seed_spread(rows):
    """base(s) vs base(s') per medium: the two-draw spread of one elite."""
    out = {}
    for d in MEDIA:
        a = np.array([r["comp"]["s"].get(d, 0.0) for r in rows])
        b = np.array([r["comp"]["fresh"].get(d, 0.0) for r in rows])
        if a.size < 2:
            continue
        within = float(np.mean((a - b) ** 2) / 2.0)          # per-elite draw variance
        between = float(np.var((a + b) / 2.0, ddof=1))        # variance of the 2-draw mean
        out[d] = {"n": int(a.size), "mean_s": float(a.mean()), "mean_fresh": float(b.mean()),
                  "draw_var": within, "design_var": max(between - within / 2.0, 0.0),
                  "nonzero_s": int((a > 0).sum()), "nonzero_fresh": int((b > 0).sum())}
        dv = out[d]["design_var"]
        out[d]["ratio_draw_to_design"] = within / dv if dv > 0 else float("inf")
    return out


def selftest():
    def row(bs, ts, bf, tf):
        f = lambda x: {c: x for c in CRITERIA}
        return {"scores": {"s": {"base": f(bs), "trials": [f(x) for x in ts]},
                           "fresh": {"base": f(bf), "trials": [f(x) for x in tf]}}}
    # Two elites: elite 0's accepted trial keeps its gain at s', elite 1's loses it.
    rows = [row(0.1, [0.3, 0.0], 0.1, [0.3, 0.1]),
            row(0.2, [0.1, 0.5], 0.2, [0.2, 0.2])]
    t = analyse(rows)["island"]
    assert t["trials"] == 4 and t["accepted"] == 2, t
    assert abs(t["gain_s"][0] - 0.25) < 1e-12, t       # (0.2 + 0.3) / 2
    assert abs(t["gain_fresh"][0] - 0.1) < 1e-12, t    # (0.2 + 0.0) / 2
    assert abs(t["retained"] - 0.4) < 1e-12, t
    assert abs(t["control_fresh"][0] - 0.05) < 1e-12, t  # (0.2, 0.0, 0.0, 0.0)
    assert t["elites_with_an_acceptance"] == 2
    # A tie is not an acceptance (strict >, as the search).
    t0 = analyse([row(0.0, [0.0, 0.0], 0.0, [0.0, 0.0])])["mission"]
    assert t0["accepted"] == 0 and t0["accept_share"] == 0.0, t0
    # Seed spread: identical draws -> zero draw variance.
    sp = seed_spread([{"comp": {"s": {"air": x}, "fresh": {"air": x}}} for x in (0.1, 0.2, 0.4)])
    assert sp["air"]["draw_var"] == 0.0 and sp["air"]["design_var"] > 0, sp
    print("refine_criterion selftest passed")


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


def criteria_of(res, island, stage):
    from dytiscidae.envs.transitions import TransitionSet
    from dytiscidae.evolution.curriculum import stage_score
    from dytiscidae.evolution.islands import curriculum_for, island_score
    tset = getattr(res, "transitions", None) or TransitionSet()
    cur = curriculum_for(island)
    st = int(min(max(stage, 0), 4))
    return {
        "mission": float(res.mission_fraction),
        "island": float(island_score(island, res, tset)),
        "stage": float(stage_score(st, res, tset, domains=cur.domains,
                                   transition_names=cur.transition_names,
                                   weakest=cur.weakest)),
    }


def measure(args, rows_path):
    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites, plain_controller, stratified
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import MissionSpec
    from dytiscidae.evolution.loop import _copy_policy
    from dytiscidae.viz.film import control_laws, run_provenance

    run = Path(args.run)
    ok, why = batchroll.usable()
    assert ok, f"batched evaluator unusable: {why}"
    cfg = run_provenance(run).get("config") or {}
    seg = float(cfg.get("segment_seconds") or 8.0)
    n_modes = int(cfg.get("n_modes") or 6)
    sigma = float(cfg.get("controller_refine_sigma") or 0.1)
    elites = load_elites(run)
    pick = stratified(elites, args.n) if args.n else list(range(len(elites)))
    groups = collections.defaultdict(list)
    for i in pick:
        m = elites[i].meta or {}
        groups[(int(m.get("gen") or 0), int(m.get("eval_seed") or 0))].append(i)
    keys = sorted(groups)[: args.max_groups or None]
    print(f"{len(elites)} elites, measuring {sum(len(groups[k]) for k in keys)} in {len(keys)} "
          f"(gen, seed) groups; T={args.trials}, sigma={sigma}, segment_seconds {seg}", flush=True)

    spec = MissionSpec()
    t0, skipped = time.time(), {"no_controller": [], "no_own_weights": []}
    done = set()
    if rows_path.exists() and args.resume:
        done = {json.loads(ln)["index"] for ln in rows_path.read_text().splitlines()}
    fh = open(rows_path, "a" if args.resume else "w")
    for gi, key in enumerate(keys):
        phenos, ctrls, keep, like = [], [], [], None
        for i in groups[key]:
            if i in done:
                continue
            p = build(elites[i].genome)
            c, sh = plain_controller(run, elites[i], p, key[1])
            if c is None:
                skipped["no_controller"].append(int(i))
                continue
            w = c.policy.weights if c.policy is not None else None
            if w is None or np.asarray(w).size == 0:
                skipped["no_own_weights"].append(int(i))
                continue
            like = like or sh
            phenos.append(p)
            ctrls.append(c)
            keep.append(i)
        if not keep:
            continue
        _, net = next(iter(control_laws(elites[keep[0]], run, like)))
        # Slot layout: [base_0..base_k, trial0_0..trial0_k, ..., trial{T-1}_k]
        all_p, all_c = list(phenos), list(ctrls)
        for t in range(args.trials):
            for slot, (i, c) in enumerate(zip(keep, ctrls)):
                rng = np.random.default_rng([key[1], int(i), t])
                w = np.asarray(c.policy.weights, float)
                all_p.append(phenos[slot])
                all_c.append(Controller(params=c.params,
                                        policy=_copy_policy(c.policy, w + rng.normal(0.0, sigma, w.shape)),
                                        bases=c.bases))
        res = {}
        for draw, seed in (("s", key[1]), ("fresh", fresh_seed(key[1]))):
            res[draw] = batchroll.evaluate_tier1_batch(
                all_p, spec=spec, controllers=all_c, segment_seconds=seg,
                identify_axes=False, seed=seed, shared=net, n_modes=n_modes)
        k = len(keep)
        for slot, i in enumerate(keep):
            m = elites[i].meta or {}
            island = m.get("island") or "generalist"
            stage = int(m.get("stage") or 0)
            rec = {"index": int(i), "gen": key[0], "seed": key[1], "fresh_seed": fresh_seed(key[1]),
                   "island": island, "stage": stage, "plan": m.get("body_plan") or "?",
                   "recorded": {d: m.get(d) for d in MEDIA},
                   "recorded_mission": m.get("mission_fraction"),
                   "scores": {}, "comp": {}, "trial_comp": {}, "bad_qacc": {}}
            for draw in DRAWS:
                rs = res[draw]
                base = rs[slot]
                trials = [rs[(t + 1) * k + slot] for t in range(args.trials)]
                rec["scores"][draw] = {"base": criteria_of(base, island, stage),
                                       "trials": [criteria_of(r, island, stage) for r in trials]}
                rec["comp"][draw] = {d: float(base.segments[d].competence)
                                     for d in MEDIA if d in base.segments}
                rec["trial_comp"][draw] = [{d: float(r.segments[d].competence)
                                            for d in MEDIA if d in r.segments} for r in trials]
                rec["bad_qacc"][draw] = int(sum(int(getattr(r.segments[d], "bad_qacc", 0))
                                                for r in [base] + trials for d in MEDIA
                                                if d in r.segments))
            fh.write(json.dumps(rec) + "\n")
        fh.flush()
        del res, all_p, all_c, phenos, ctrls
        gc.collect()
        print(f"group {gi + 1}/{len(keys)} gen {key[0]} n={k} wall {time.time() - t0:.0f}s", flush=True)
    fh.close()
    return time.time() - t0, skipped, len(elites)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=0, help="0 = every elite, else a stratified n")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--from-rows")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    out = Path(args.out)
    rows_path = Path(args.from_rows) if args.from_rows else Path(str(out) + ".rows.jsonl")
    if args.from_rows:
        wall, skipped, n_archive = float("nan"), {}, 0
    else:
        wall, skipped, n_archive = measure(args, rows_path)
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    table = analyse(rows)
    result = {"run": args.run, "n_elites": len(rows), "of_archive": n_archive,
              "trials_per_elite": args.trials, "wall_s": wall, "skipped": skipped,
              "table": table, "verdicts": verdicts(table), "seed_spread": seed_spread(rows),
              "islands": dict(collections.Counter(r["island"] for r in rows))}
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
