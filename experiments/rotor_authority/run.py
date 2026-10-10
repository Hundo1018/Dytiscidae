#!/usr/bin/env python3
"""N6 -- rotor authority through the basis (ARCH51_SPEC §B N6; ROADMAP 2026-10-10 item N6; T3a).

The question: can the controller the search keeps -- a shared policy that
commands a six-axis body twist, delivered to the gait only through the body's
identified mobility basis (``MobilityBasis.coeffs_for_twist`` then
``command_params``) -- move a rotor at all?  The rotor channel is a speed in
rad/s inside a basis identified with probes of sigma 0.35 (``cpg.py:553``), so
"one unit of coefficient" may be a few rad/s of a rotor that spins at hundreds.

Frozen prediction (verbatim from ROADMAP §"2026-10-10" item N6 and ARCH51_SPEC
§B N6; not to be edited):

    *Prediction:* (i) scores air < 0.3 while (ii) scores >= 0.8 on the same body
    and seed, and the basis moves rotor speed by < 5% of top speed per unit
    coefficient. *Falsified* if (i) >= 0.6.

where, from the item: "a hand PD hover law applied (i) through
``coeffs_for_twist`` and (ii) directly to ``ctrl``".

Bodies: arch49's eight island archives (``load_run_archive(run, island)``),
deduplicated by ``genome_id``, with ``meta["n_rotors"] > 0`` (191 of 645 on
2026-10-10).  The probe reads stored meta and steps the env, so any tree works.

Part A (static, every rotor elite, seconds): per elite, from the stored air (and
water) basis,
  m_frac = max_k max_{i in R} |modes[k, 2n+i]| / hi[i]
           fraction of top speed per unit coefficient;
  f_frac = max_{j<6} max_{i in R} |(modes.T @ coeffs_for_twist(e_j))[2n+i]| / hi[i]
           at full intent, through INTENT_AUTHORITY 0.5;
  rotor_norm_share_k = sum_{i in R} modes[k, 2n+i]^2  (reported with the row's
           own squared norm so a share can be read).
R = the "_r" actuator channels.  Medians and p90 over elites.

Part B (dynamic, 24 bodies, stratified by n_rotors and island with
``default_rng(0)``): bodies whose rotors cannot lift
(sum_i bemt(spec_i, hi[i], 0, 0, AIR.rho, AIR.mu)[0] < 1.2 m g) are excluded
first and counted.  Each body's air segment is set up as
experiments/rotor/fly.py:99-105 does it (reset, scatter, task from
``_scatter_seed(eval_seed, AIR)``, ``env.rollout(8.0, domain=AIR)``) and scored
by ``seg.competence``.  That is the single-machine path, the only one that
accepts a hand law in ``env.cpg.command``; air agrees across the two paths
(``test_the_two_evaluation_paths_score_the_same_machine_the_same``).

  (ii) direct on ctrl: the rotor entries R of ``orig(params, t)`` are replaced
       by fly.py's cascade and mixer (rotor subset only).
  (i)  through the basis: ``orig(B.command_params(params,
       B.coeffs_for_twist(intent), n), t)`` with the body-frame intent
         surge, sway   0.5 (v_des - v)
         heave         0.5 (z0 - z) - 0.5 v_z
         roll, pitch   -1.0 angle - 0.3 rate
         yaw           1.0 heading_err - 0.3 w_z
       each clipped to [-1, 1], all gains scaled by G in {0.5, 1, 2}; the best
       G per body is reported, so bad gains are not the explanation.
  Seed: each body's own ``eval_seed``.  A fresh env per rollout, so every arm
  of a body starts from the identical draw.

Decision rule (typed by the probe writer; the frozen text gives the per-body
statement only): CONFIRMED if the median (i)-best < 0.3 and the median (ii) >=
0.8 over the bodies that ran and the median Part-A m_frac < 0.05; REFUTED if
the median (i)-best >= 0.6; otherwise NEITHER.  The per-body counts that
decide each clause are in the table, so a reader can apply another reading.

    PYTHONPATH=. MUJOCO_GL=disable DYTISCIDAE_KERNEL_DIR=<main>/mojo/build \\
        .venv/bin/python experiments/rotor_authority/run.py --run runs/arch49

  --limit N      smoke: scan rotor elites in archive order until N liftable bodies are
                 found (Part A rows for those scanned), run Part B on those N
                 (writes *_smoke files, never the N6_result.md)
  --cpu          accepted for symmetry with N8; this probe is numpy-only always
  --from-rows F  re-aggregate and re-write the report without simulating
  --selftest     the pure aggregation, no simulator
"""
import argparse
import collections
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ISLANDS = ("air", "water", "land", "amphibian", "aerial_diver", "land_air", "triphibian", "generalist")
GAINS = (0.5, 1.0, 2.0)
SEG_SECONDS = 8.0

# Frozen thresholds (ROADMAP N6 / ARCH51_SPEC §B N6); not to be edited.
PRED_I_BELOW = 0.3
PRED_II_AT_LEAST = 0.8
PRED_M_FRAC_BELOW = 0.05
FALSIFY_I_AT_LEAST = 0.6
LIFT_MARGIN = 1.2

PREDICTION = (
    "*Prediction:* (i) scores air < 0.3 while (ii) scores >= 0.8 on the same body and seed, "
    "and the basis moves rotor speed by < 5% of top speed per unit coefficient. "
    "*Falsified* if (i) >= 0.6.")


# --------------------------------------------------------------------------
# Pure aggregation (no simulator)
# --------------------------------------------------------------------------


def _q(v, p):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.percentile(v, p)) if len(v) else float("nan")


def stats(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return {"n": len(v), "median": _q(v, 50), "q25": _q(v, 25), "q75": _q(v, 75), "p90": _q(v, 90),
            "min": float(min(v)) if v else float("nan"), "max": float(max(v)) if v else float("nan")}


def aggregate(rows):
    """Part A and Part B tables and the verdict inputs from the rows."""
    A = [r for r in rows if r["part"] == "A"]
    B = [r for r in rows if r["part"] == "B"]
    out = {"n_rotor_elites": len(A)}
    for med in ("air", "water"):
        sub = [r["basis"][med] for r in A if med in r.get("basis", {})]
        out[med] = {
            "n": len(sub),
            "m_frac": stats([s["m_frac"] for s in sub]),
            "f_frac": stats([s["f_frac"] for s in sub]),
            "rotor_norm_share_max_k": stats([s["rotor_norm_share_max_k"] for s in sub]),
            "row_norm2_of_that_mode": stats([s["row_norm2_at_max_k"] for s in sub]),
            "share_m_frac_below_0.05": (float(np.mean([s["m_frac"] < PRED_M_FRAC_BELOW for s in sub]))
                                        if sub else float("nan")),
        }
    out["lift_excluded"] = int(sum(1 for r in A if not r["lift_ok"]))
    out["lift_excluded_spec"] = int(sum(1 for r in A if not r.get("lift_ok_spec", r["lift_ok"])))
    out["lift_excluded_up"] = int(sum(1 for r in A if not r.get("lift_ok_up", r["lift_ok"])))
    out["thrust_over_weight"] = stats([r["thrust_over_weight"] for r in A])
    out["thrust_over_weight_up"] = stats([r.get("thrust_over_weight_up", r["thrust_over_weight"]) for r in A])
    ran = [r for r in B if r.get("i_best") is not None and r.get("ii") is not None]
    out["n_bodies_ran"] = len(ran)
    out["n_bodies_planned"] = len(B)
    i_best = [r["i_best"] for r in ran]
    ii = [r["ii"] for r in ran]
    out["i_best"] = stats(i_best)
    out["ii"] = stats(ii)
    out["share_i_lt_0.3_and_ii_ge_0.8"] = (float(np.mean([a < PRED_I_BELOW and b >= PRED_II_AT_LEAST
                                                          for a, b in zip(i_best, ii)])) if ran else float("nan"))
    out["share_i_lt_0.3"] = float(np.mean([a < PRED_I_BELOW for a in i_best])) if ran else float("nan")
    out["share_ii_ge_0.8"] = float(np.mean([b >= PRED_II_AT_LEAST for b in ii])) if ran else float("nan")
    out["count_i_ge_0.6"] = int(sum(a >= FALSIFY_I_AT_LEAST for a in i_best))
    out["count_i_lt_0.3_and_ii_ge_0.8"] = int(sum(a < PRED_I_BELOW and b >= PRED_II_AT_LEAST
                                                  for a, b in zip(i_best, ii)))
    flies = [r for r in ran if r["ii"] >= PRED_II_AT_LEAST]
    out["restricted"] = {"n": len(flies), "i_best": stats([r["i_best"] for r in flies]),
                         "count_i_lt_0.3": int(sum(r["i_best"] < PRED_I_BELOW for r in flies)),
                         "count_i_ge_0.6": int(sum(r["i_best"] >= FALSIFY_I_AT_LEAST for r in flies))}
    out["best_gain_counts"] = dict(collections.Counter(str(r["g_best"]) for r in ran))
    qc = [r for r in rows if r["part"] == "C"]
    out["quad_check"] = qc[0] if qc else None
    out["dropped"] = [{"index": r["index"], "why": r.get("why") or "no result"} for r in B
                      if r not in ran]
    return out


def verdict(agg):
    n = agg["n_bodies_ran"]
    m_med = agg["air"]["m_frac"]["median"] if agg["air"]["n"] else float("nan")
    if n == 0 or not np.isfinite(m_med):
        return {"outcome": "NOT COMPARABLE",
                "by": "no body completed arms (i) and (ii), or no stored air basis to read m_frac from"}
    i_med, ii_med = agg["i_best"]["median"], agg["ii"]["median"]
    c_i = i_med < PRED_I_BELOW
    c_ii = ii_med >= PRED_II_AT_LEAST
    c_m = m_med < PRED_M_FRAC_BELOW
    fals = i_med >= FALSIFY_I_AT_LEAST
    out = "REFUTED" if fals else "CONFIRMED" if (c_i and c_ii and c_m) else "NEITHER (not confirmed, not falsified)"
    by = (f"over {n} bodies: median (i)-best {i_med:.3f} (< 0.3: {c_i}; falsifier >= 0.6: {fals}), "
          f"median (ii) {ii_med:.3f} (>= 0.8: {c_ii}), median m_frac {m_med:.4f} (< 0.05: {c_m}); "
          f"per body: (i) < 0.3 and (ii) >= 0.8 in {agg['count_i_lt_0.3_and_ii_ge_0.8']}/{n}, "
          f"(i) >= 0.6 in {agg['count_i_ge_0.6']}/{n}; the hand law flies (ii >= 0.8) {agg['restricted']['n']}/{n} bodies")
    return {"outcome": out, "by": by, "clauses": {"i_lt_0.3": bool(c_i), "ii_ge_0.8": bool(c_ii),
                                                  "m_frac_lt_0.05": bool(c_m), "falsified": bool(fals)}}


def selftest():
    rows = []
    for k in range(4):
        rows.append({"part": "A", "index": k, "lift_ok": k != 3, "thrust_over_weight": 2.0 + k,
                     "basis": {"air": {"m_frac": 0.01 * (k + 1), "f_frac": 0.02, "rotor_norm_share_max_k": 0.1,
                                       "row_norm2_at_max_k": 1.0}}})
    for k, (a, b) in enumerate([(0.1, 0.9), (0.2, 0.85), (0.7, 0.9), (0.0, 0.5)]):
        rows.append({"part": "B", "index": k, "i_best": a, "ii": b, "g_best": 1.0})
    rows.append({"part": "B", "index": 9, "i_best": None, "ii": None, "why": "diverged"})
    agg = aggregate(rows)
    assert agg["n_bodies_ran"] == 4 and agg["n_bodies_planned"] == 5, agg
    assert agg["count_i_ge_0.6"] == 1 and agg["count_i_lt_0.3_and_ii_ge_0.8"] == 2, agg
    assert agg["lift_excluded"] == 1 and agg["dropped"] == [{"index": 9, "why": "diverged"}], agg
    v = verdict(agg)
    assert v["outcome"] == "CONFIRMED", v        # medians: (i) 0.15, (ii) 0.875, m_frac 0.025
    hi_m = [dict(r, basis={"air": dict(r["basis"]["air"], m_frac=0.5)}) if r["part"] == "A" else r for r in rows]
    assert verdict(aggregate(hi_m))["outcome"].startswith("NEITHER")      # the basis does move the rotors
    for r in rows:
        if r["part"] == "B" and r["i_best"] is not None:
            r["i_best"] = 0.7
    assert verdict(aggregate(rows))["outcome"] == "REFUTED"
    assert verdict(aggregate([]))["outcome"] == "NOT COMPARABLE"
    print("rotor_authority aggregation selftest passed")


# --------------------------------------------------------------------------
# Report writer, in the shape of runs/analysis_1010_failure_theory/REPORT_FORMAT.md
# --------------------------------------------------------------------------


def _f(x, k=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{k}f}"


def write_report(path, *, agg, ver, meta, rows):
    B = [r for r in rows if r["part"] == "B" and r.get("i_best") is not None and r.get("ii") is not None]
    lines = []
    lines.append(f"# N6 — rotor authority through the basis, {meta['date']}, commit {meta['commit']}")
    lines.append("")
    lines.append("## 6 What ran")
    lines.append(f"command:        {meta['command']}")
    lines.append(f"commit:         {meta['commit']}{' (working tree dirty: ' + meta['dirty'] + ')' if meta['dirty'] else ''}")
    lines.append("seed(s):        np.random.default_rng(0) for the Part B stratified draw; each body's own "
                 "meta['eval_seed'] for its rollouts (scatter and task from _scatter_seed(eval_seed, AIR))")
    lines.append(f"inputs:         {meta['run']}/archive_<island>.pkl for the eight islands (load_run_archive per island, "
                 f"deduplicated by genome_id; {agg['n_rotor_elites']} rotor elites)")
    lines.append(f"wall:           {meta['wall_s']:.0f} s (Part A {meta['wall_a_s']:.0f} s, Part B {meta['wall_b_s']:.0f} s)")
    eff = [
        "Part B is the single-machine numpy path (the only one that accepts a hand law in env.cpg.command); "
        "air agrees across the two paths (test_the_two_evaluation_paths_score_the_same_machine_the_same).",
        f"control law evaluated {'at every physics step' if not meta['hz'] else 'at ' + str(meta['hz']) + ' Hz'} "
        "for both arms, so the control rate is not the difference between (i) and (ii).",
        "the cascade is fly.py's, generalised to the rotor subset R of a mixed body: its mixer uses each rotor's own "
        "axis, handedness, arm and spin-reaction coefficient (RotorSet.apply's convention) and a bounded least squares "
        "(0 <= thrust <= thrust at top speed) with a weak pull toward an equal share, where fly.py's four-rotor "
        "lstsq-and-clip is exact only for four rotors.",
        "lift test: experiments/rotor/fly.py lines 43-48 hold k_t and kappa, not a lift test, so ARCH51_SPEC's "
        "'sum_i bemt(spec_i, hi[i], 0, 0, AIR.rho, AIR.mu)[0] < 1.2 m g' is implemented from the spec text. "
        f"Applied: --lift-test {meta.get('lift_test', 'up')}; 'spec' = the sum as typed (direction-blind), "
        "'up' = the same sum keeping only the thrust component along body +z (RotorSet.apply pushes along "
        "handed * ax; a speed channel with lo = 0 cannot reverse, and on chained 'eel' bodies half the rotors point down). "
        "Both counts are in section 7; the other rule is one flag away.",
        f"Part B bodies: planned {agg['n_bodies_planned']}, completed arms (i) and (ii): {agg['n_bodies_ran']} "
        f"(time budget {meta['budget_s']:.0f} s; gains {list(GAINS)}).",
    ]
    if meta.get("limit"):
        eff.append(f"--limit {meta['limit']}: SMOKE RUN, not the N6 measurement.")
    lines.append("effective config: " + eff[0])
    for e in eff[1:]:
        lines.append("                " + e)
    lines.append("")
    lines.append("## 7 Data quality")
    lines.append(f"n:              Part A {agg['n_rotor_elites']} rotor elites; Part B {agg['n_bodies_ran']} bodies x 4 rollouts "
                 f"(arm (ii) once, arm (i) at G = {', '.join(str(g) for g in GAINS)})")
    drop = agg["dropped"]
    lines.append(f"dropped:        Part B: {len(drop)} bodies without both arms ({'; '.join(str(d['index']) + ': ' + d['why'] for d in drop) or 'none'}); "
                 f"{agg['lift_excluded']} of {agg['n_rotor_elites']} rotor elites excluded from Part B by the lift test "
                 f"(--lift-test {meta.get('lift_test', 'up')}; as typed in the spec, direction-blind: {agg['lift_excluded_spec']} excluded; "
                 f"upward component only: {agg['lift_excluded_up']} excluded; thrust at top speed over m g, direction-blind: "
                 f"median {_f(agg['thrust_over_weight']['median'], 2)}, min {_f(agg['thrust_over_weight']['min'], 2)}, "
                 f"max {_f(agg['thrust_over_weight']['max'], 2)}; upward only: median {_f(agg['thrust_over_weight_up']['median'], 2)})")
    qc = agg.get("quad_check")
    lines.append("control check:  " + (
        f"the arm-(ii) law on REFERENCE_PLANS['quad'] (seed 0, the fly.py body) scores {_f(qc['competence'])} "
        f"(experiments/rotor/fly.py: 0.86-0.99 across seeds), airborne fraction {_f(qc['airborne_fraction'])}"
        if qc else "not run"))
    lines.append(f"artifacts:      rollouts with a MuJoCo QACC reset are zeroed by the scorer (res.survived False); "
                 f"bad_qacc per rollout is in the JSON. Elites with no stored water basis are absent from the water rows "
                 f"(n = {agg['water']['n']} of {agg['n_rotor_elites']}).")
    lines.append("")
    lines.append("## 8 Numbers (the table the ROADMAP item asked for, nothing else)")
    lines.append("Part A, per unit coefficient (m_frac) and at full intent (f_frac), as a fraction of rotor top speed:")
    lines.append("| basis | n | m_frac median [IQR] | m_frac p90 | f_frac median [IQR] | f_frac p90 | share m_frac < 0.05 | "
                 "rotor_norm_share_k (max over k) median / p90 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for med in ("air", "water"):
        a = agg[med]
        lines.append(f"| {med} | {a['n']} | {_f(a['m_frac']['median'], 4)} [{_f(a['m_frac']['q25'], 4)}, {_f(a['m_frac']['q75'], 4)}] "
                     f"| {_f(a['m_frac']['p90'], 4)} | {_f(a['f_frac']['median'], 4)} [{_f(a['f_frac']['q25'], 4)}, {_f(a['f_frac']['q75'], 4)}] "
                     f"| {_f(a['f_frac']['p90'], 4)} | {_f(a['share_m_frac_below_0.05'], 3)} "
                     f"| {_f(a['rotor_norm_share_max_k']['median'], 4)} / {_f(a['rotor_norm_share_max_k']['p90'], 4)} "
                     f"(that mode's whole-row squared norm: median {_f(a['row_norm2_of_that_mode']['median'], 3)}) |")
    lines.append("")
    lines.append("Part B, air competence on the same body and seed:")
    lines.append("| body | island | plan | n_rotors | (i)-best [G] | (i) at G 0.5 / 1 / 2 | (ii) | m_frac | f_frac |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in B:
        gs = " / ".join(_f(r["i"].get(str(g)), 3) for g in GAINS)
        lines.append(f"| {r['index']} | {r['island']} | {r['plan']} | {r['n_rotors']} | {_f(r['i_best'], 3)} [{r['g_best']}] "
                     f"| {gs} | {_f(r['ii'], 3)} | {_f(r['m_frac'], 4)} | {_f(r['f_frac'], 4)} |")
    lines.append("")
    lines.append(f"shares:         (i)-best < 0.3: {_f(agg['share_i_lt_0.3'])}; (ii) >= 0.8: {_f(agg['share_ii_ge_0.8'])}; "
                 f"both: {_f(agg['share_i_lt_0.3_and_ii_ge_0.8'])} ({agg['count_i_lt_0.3_and_ii_ge_0.8']}/{agg['n_bodies_ran']}); "
                 f"falsifier, bodies with (i)-best >= 0.6: {agg['count_i_ge_0.6']}/{agg['n_bodies_ran']}; "
                 f"best gain counts {agg['best_gain_counts']}")
    lines.append(f"(i)-best:       median {_f(agg['i_best']['median'])} [IQR {_f(agg['i_best']['q25'])}, {_f(agg['i_best']['q75'])}], "
                 f"max {_f(agg['i_best']['max'])};  (ii): median {_f(agg['ii']['median'])} "
                 f"[IQR {_f(agg['ii']['q25'])}, {_f(agg['ii']['q75'])}], min {_f(agg['ii']['min'])}")
    rs = agg["restricted"]
    lines.append(f"restricted:     on the {rs['n']} bodies the hand law flies directly ((ii) >= 0.8): (i)-best median "
                 f"{_f(rs['i_best']['median'])}, max {_f(rs['i_best']['max'])}; (i)-best < 0.3 in {rs['count_i_lt_0.3']}/{rs['n']}, "
                 f">= 0.6 in {rs['count_i_ge_0.6']}/{rs['n']}  (the prediction compares (i) with (ii) on the same body, so this is "
                 f"the reading the unrestricted medians dilute when the law cannot fly a body)")
    lines.append("")
    lines.append("spread:         IQR in the table above for every median; per-body values listed in full in the Part B table "
                 "(one rollout per cell, no resampling: a body's score depends on its single eval_seed draw).")
    lines.append("")
    lines.append("## 9 Against the frozen prediction (quote it verbatim, then one word)")
    lines.append(f"prediction:     \"{PREDICTION}\"")
    outcome = (ver["outcome"] if not meta.get("limit") else
               f"NOT A MEASUREMENT (smoke, --limit {meta['limit']}; the rule would read: {ver['outcome']})")
    lines.append(f"outcome:        {outcome}")
    lines.append(f"by:             {ver['by']}")
    lines.append("decision rule:  typed by the probe writer (the frozen text states the claim per body): CONFIRMED if median "
                 "(i)-best < 0.3, median (ii) >= 0.8 and median Part-A m_frac < 0.05; REFUTED if median (i)-best >= 0.6; "
                 "else NEITHER. The per-body counts above let a reader apply another reading.")
    lines.append("")
    lines.append("## 12 Reproducer")
    lines.append("```bash")
    lines.append("cd /home/hundo/Projects/Dytiscidae/dytiscidae")
    lines.append(meta["command"])
    lines.append("```")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Simulator side
# --------------------------------------------------------------------------


def rotor_channels(env):
    """Indices of the "_r" actuator channels, and the matching rotor-set slots."""
    import mujoco
    R = [i for i, a in enumerate(env.act_names) if a.endswith("_r")]
    slots = []
    for i in R:
        b = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.act_names[i][:-2] + "_rot")
        slots.append(env.rotors.body.index(b))
    return R, slots


def part_a_basis(B, env, R):
    """m_frac, f_frac and the rotor share of each mode for one basis."""
    n = env.cpg.n
    hi = np.asarray(env.cpg.hi, float)
    modes = np.asarray(B.modes, float)
    assert modes.shape[1] == 3 * n + 1, (modes.shape, n)
    cols = [2 * n + i for i in R]
    top = np.array([hi[i] for i in R], float)
    m_frac = float(np.max(np.abs(modes[:, cols]) / top[None, :])) if len(cols) and modes.size else 0.0
    f_frac = 0.0
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1.0
        d = modes.T @ B.coeffs_for_twist(e)
        f_frac = max(f_frac, float(np.max(np.abs(d[cols]) / top))) if len(cols) else f_frac
    share = (modes[:, cols] ** 2).sum(axis=1) if len(cols) else np.zeros(len(modes))
    k = int(np.argmax(share)) if len(share) else 0
    return {"m_frac": m_frac, "f_frac": f_frac, "n_modes": int(modes.shape[0]),
            "rotor_norm_share_max_k": float(share[k]) if len(share) else 0.0,
            "row_norm2_at_max_k": float((modes[k] ** 2).sum()) if len(share) else 0.0,
            "rotor_norm_share_by_k": [float(x) for x in share]}


def make_env(elite, seed):
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import TriphibianEnv
    return TriphibianEnv(build(elite.genome), seed=seed)


def thrust_over_weight(env, R, slots):
    """``(spec, up, mass)``.

    ``spec``: the lift test as ARCH51_SPEC §B N6 writes it, sum_i bemt(spec_i,
    hi[i], 0, 0, AIR.rho, AIR.mu)[0] over m g, which counts every rotor's thrust
    whichever way it points.  ``up``: the same sum keeping only the component
    along the body's +z (``RotorSet.apply``'s thrust direction is
    ``handed * ax``): a rotor whose axis points down with a speed channel that
    cannot reverse (lo = 0) contributes nothing to a hover, and on arch49's
    chained "eel" bodies half the rotors do.
    """
    import mujoco
    from dytiscidae.physics.medium import AIR, GRAVITY
    from dytiscidae.physics.rotor import bemt
    mujoco.mj_forward(env.model, env.data)
    Rm = env.data.xmat[env.root_body].reshape(3, 3)
    hi = np.asarray(env.cpg.hi, float)
    tot = up = 0.0
    for i, k in zip(R, slots):
        T = float(bemt(env.rotors.spec[k], float(hi[i]), 0.0, 0.0, AIR.rho, AIR.mu)[0])
        ax = Rm.T @ (env.data.xmat[env.rotors.body[k]].reshape(3, 3) @ np.asarray(env.rotors.axis_local[k], float))
        dz = float(getattr(env.rotors.spec[k], "handed", 1.0)) * float(ax[2])
        tot += T
        up += T * max(dz, 0.0)
    mass = float(env.model.body_subtreemass[env.root_body])
    w = max(mass * GRAVITY, 1e-9)
    return tot / w, up / w, mass


class Hover:
    """fly.py's cascade and mixer for the rotor subset R; and the basis intent law.

    One instance per env.  ``direct`` returns the rotor speeds for arm (ii),
    ``intent`` the six-axis body-twist intent for arm (i).  Both read the same
    state, and both are evaluated at every physics step unless ``hz`` is set.
    """

    def __init__(self, env, R, slots):
        import mujoco
        from dytiscidae.physics.medium import AIR
        from dytiscidae.physics.rotor import bemt
        self.mj, self.env, self.R, self.slots = mujoco, env, R, slots
        m = env.model
        self.hi = np.asarray(env.cpg.hi, float)[R]
        self.lo = np.asarray(env.cpg.lo, float)[R]
        self.bodies = [env.rotors.body[k] for k in slots]
        self.axis_local = [np.asarray(env.rotors.axis_local[k], float) for k in slots]
        self.handed = [float(getattr(env.rotors.spec[k], "handed", 1.0)) for k in slots]
        self.k_t, self.kappa = [], []
        for k in slots:
            T0, Q0 = bemt(env.rotors.spec[k], 600.0, 0.0, 0.0, AIR.rho, AIR.mu)
            self.k_t.append(T0 / 600.0 ** 2)
            self.kappa.append(Q0 / T0)
        self.M = np.zeros((m.nv, m.nv))
        self.z0 = None
        self.model = m

    # ---- state ----------------------------------------------------------
    def _state(self):
        env, d = self.env, self.env.data
        self.mj.mj_fullM(self.model, d, self.M)
        mass = float(self.M[0, 0])
        inertia = np.diag(self.M[3:6, 3:6]).copy()      # per axis: an eel's roll inertia is ~1/50 of its mean
        Rm = d.xmat[env.root_body].reshape(3, 3)
        pos = d.xipos[env.root_body]
        v = d.qvel[:3].copy()
        w_body = d.qvel[3:6].copy()
        if self.z0 is None:
            self.z0 = float(pos[2])
        ph = env._phase_now()
        head = float(ph.heading) if ph is not None else 0.0
        speed = float(ph.speed) if (ph is not None and ph.speed) else 0.0
        return mass, inertia, Rm, pos, v, w_body, head, speed

    # ---- arm (ii): the cascade and mixer (fly.py:42-96) ---------------------
    def direct(self):
        from dytiscidae.physics.medium import GRAVITY
        d = self.env.data
        mass, inertia, Rm, pos, v, w_body, head, speed = self._state()
        v_des = np.array([speed * math.cos(head), speed * math.sin(head), 0.0])
        a = 1.2 * (v_des - v)
        a[2] = 3.0 * (self.z0 - pos[2]) - 2.5 * v[2]
        a = a + np.array([0.0, 0.0, GRAVITY])
        h = np.linalg.norm(a[:2])
        lim = a[2] * math.tan(math.radians(35))
        if h > lim > 0:
            a[:2] *= lim / h
        f = mass * a
        z_des = f / np.linalg.norm(f)
        x_c = np.array([math.cos(head), math.sin(head), 0.0])
        y_d = np.cross(z_des, x_c)
        y_d /= np.linalg.norm(y_d)
        x_d = np.cross(y_d, z_des)
        Rd = np.stack([x_d, y_d, z_des], 1)
        E = 0.5 * (Rd.T @ Rm - Rm.T @ Rd)
        e_r = np.array([E[2, 1], E[0, 2], E[1, 0]])
        wn = 7.0
        tau = inertia * (-(wn ** 2) * e_r - 2 * 0.8 * wn * w_body)
        thrust = float(f @ Rm[:, 2])
        A = np.zeros((4, len(self.bodies)))
        for c, b in enumerate(self.bodies):
            ax_b = Rm.T @ (d.xmat[b].reshape(3, 3) @ self.axis_local[c])
            d_b = self.handed[c] * ax_b
            r = Rm.T @ (d.xipos[b] - pos)
            A[:, c] = [d_b[2], *(np.cross(r, d_b) - self.kappa[c] * ax_b)]
        b = np.array([thrust, tau[0], tau[1], tau[2]])
        # fly.py solves the four-row mixer with lstsq and clips: exact for four
        # rotors, but with n >> 4 the min-norm solution goes negative on some
        # rotors and the clip then breaks the torque balance.  A bounded least
        # squares (0 <= t_i <= thrust at top speed) with the rows scaled to
        # comparable units (hover weight; the torque for 0.2 rad at the cascade's
        # natural frequency) is the same mixer, honest about the limits.
        w = np.array([1.0 / (mass * GRAVITY), *(1.0 / (inertia * 7.0 ** 2 * 1.0))])
        ub = np.array(self.k_t) * self.hi ** 2
        nrot = A.shape[1]
        up = np.maximum(A[0], 0.0)                           # rotors that push up at all
        t_nom = up * mass * GRAVITY / max(float(up @ up), 1e-12)   # the minimum-norm hover share
        # A small pull toward an equal share: with many more rotors than rows the
        # allocation is redundant, and without it the solver lands on a vertex
        # (rotors alternating between 0 and top speed from one step to the next).
        ridge = 0.05 * np.eye(nrot) / max(float(t_nom.max()), 1e-9)
        try:
            from scipy.optimize import lsq_linear
            t_i = lsq_linear(np.vstack([A * w[:, None], ridge]),
                             np.concatenate([b * w, ridge @ t_nom]),
                             bounds=(np.zeros_like(ub), ub), method="bvls").x
        except ImportError:                                  # no scipy: fly.py's own solve
            t_i = np.linalg.lstsq(A, b, rcond=None)[0]
        om = np.sqrt(np.maximum(t_i, 0.0) / np.array(self.k_t))
        return np.clip(om, self.lo, self.hi)

    # ---- arm (i): intent in body-twist units --------------------------------
    def intent(self, G):
        mass, inertia, Rm, pos, v, w_body, head, speed = self._state()
        v_des = np.array([speed * math.cos(head), speed * math.sin(head), 0.0])
        u_w = np.array([0.5 * (v_des[0] - v[0]), 0.5 * (v_des[1] - v[1]),
                        0.5 * (self.z0 - pos[2]) - 0.5 * v[2]])
        lin = Rm.T @ u_w
        roll = math.atan2(Rm[2, 1], Rm[2, 2])
        pitch = -math.asin(float(np.clip(Rm[2, 0], -1.0, 1.0)))
        yaw = math.atan2(Rm[1, 0], Rm[0, 0])
        err = (head - yaw + math.pi) % (2 * math.pi) - math.pi
        out = np.array([lin[0], lin[1], lin[2],
                        -1.0 * roll - 0.3 * w_body[0],
                        -1.0 * pitch - 0.3 * w_body[1],
                        1.0 * err - 0.3 * w_body[2]])
        return np.clip(G * out, -1.0, 1.0)


def fly_arm(elite, seed, R, slots, arm, G=1.0, hz=0.0, basis=None):
    """One air segment under a hand law; returns (competence, details)."""
    from dytiscidae.envs.evaluate import _scatter_seed
    from dytiscidae.envs.tasks import schedule_for, task_seed
    from dytiscidae.envs.triphibian import Domain

    env = make_env(elite, seed)
    env.reset(Domain.AIR)
    env.scatter(np.random.default_rng(_scatter_seed(seed, Domain.AIR)))
    env.task = schedule_for(Domain.AIR, np.random.default_rng(task_seed(_scatter_seed(seed, Domain.AIR))))
    orig = env.cpg.command
    hv = Hover(env, R, slots)
    n = env.cpg.n
    every = max(1, int(round(1.0 / (hz * env.timestep)))) if hz else 1
    st = {"i": 0, "rot": None, "params": None, "src": None, "z": hv.z0}

    def command(params, t):
        if st["i"] % every == 0 or st["src"] is not params:
            st["src"] = params
            if arm == "direct":
                st["rot"] = hv.direct()
            else:
                coeffs = basis.coeffs_for_twist(hv.intent(G))
                st["params"] = basis.command_params(params, coeffs, n)
        st["i"] += 1
        if arm == "direct":
            ang = np.array(orig(params, t), float)
            ang[R] = st["rot"]
            return ang
        return orig(st["params"], t)

    env.cpg.command = command
    t0 = time.time()
    zs = []
    seg = env.rollout(SEG_SECONDS, domain=Domain.AIR,
                      on_step=lambda e, i: zs.append(float(e.data.xpos[e.root_body][2])) if i % 25 == 0 else None)
    mm = seg.measurements or {}
    return float(seg.competence), {
        "survived": bool(seg.survived), "failure": seg.failure, "bad_qacc": int(seg.bad_qacc),
        "z_start": hv.z0, "z_end": zs[-1] if zs else None, "z_min": min(zs) if zs else None,
        "airborne_fraction": mm.get("airborne_fraction"), "wall": time.time() - t0}


def load_rotor_elites(run):
    """``[(island, elite)]``: eight island archives, deduplicated by genome_id, n_rotors > 0."""
    from dytiscidae.ops.run import load_run_archive
    seen, out = set(), []
    for isl in ISLANDS:
        a, _ = load_run_archive(run, isl)
        if a is None:
            continue
        for e in a.cells.values():
            gid = getattr(e.genome, "genome_id", "") or f"{isl}:{e.cell}"
            if gid in seen:
                continue
            seen.add(gid)
            if int((e.meta or {}).get("n_rotors") or 0) > 0:
                out.append((isl, e))
    return out


def stratified_bodies(pool, n, seed=0):
    """Round-robin over (island, n_rotors) strata, shuffled within each, ``default_rng(seed)``."""
    rng = np.random.default_rng(seed)
    strata = collections.defaultdict(list)
    for p in pool:
        strata[(p["island"], p["n_rotors"])].append(p)
    keys = sorted(strata)
    for k in keys:
        strata[k] = [strata[k][i] for i in rng.permutation(len(strata[k]))]
    order = [keys[i] for i in rng.permutation(len(keys))]
    out = []
    while len(out) < n and any(strata[k] for k in order):
        for k in order:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop(0))
    return out


def measure(args, rows_path):
    from dytiscidae.control.cpg import MobilityBasis

    run = Path(args.run)
    t_all = time.time()
    pool = load_rotor_elites(run)
    print(f"{len(pool)} rotor elites in the eight archives (deduplicated by genome_id)", flush=True)
    fh = open(rows_path, "w")
    # ---- Part A ----------------------------------------------------------
    t_a = time.time()
    candidates = []
    for gi, (isl, e) in enumerate(pool):
        meta = e.meta or {}
        seed = int(meta.get("eval_seed") or 0)
        env = make_env(e, seed)
        R, slots = rotor_channels(env)
        tow, tow_up, mass = thrust_over_weight(env, R, slots)
        ok_spec, ok_up = bool(tow >= LIFT_MARGIN), bool(tow_up >= LIFT_MARGIN)
        bases = MobilityBasis.bases_from_record(meta.get("mobility_basis"))
        rec = {"part": "A", "index": gi, "island": isl, "plan": meta.get("body_plan") or "?",
               "n_rotors": int(meta.get("n_rotors") or 0), "n_rotor_channels": len(R), "eval_seed": seed,
               "genome_id": getattr(e.genome, "genome_id", ""), "mass": mass,
               "thrust_over_weight": float(tow), "thrust_over_weight_up": float(tow_up),
               "lift_ok_spec": ok_spec, "lift_ok_up": ok_up,
               "lift_ok": ok_spec if args.lift_test == "spec" else ok_up, "basis": {}}
        for med in ("air", "water"):
            if med in bases:
                rec["basis"][med] = part_a_basis(bases[med], env, R)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if rec["lift_ok"] and "air" in bases:
            candidates.append({"index": gi, "island": isl, "n_rotors": rec["n_rotors"], "plan": rec["plan"],
                               "elite": e, "seed": seed, "R": R, "slots": slots,
                               "m_frac": rec["basis"]["air"]["m_frac"], "f_frac": rec["basis"]["air"]["f_frac"],
                               "basis": bases["air"]})
        del env
        if args.limit and len(candidates) >= args.limit:
            print(f"--limit {args.limit}: {len(candidates)} liftable bodies found after scanning {gi + 1} elites", flush=True)
            break
        if (gi + 1) % 25 == 0:
            print(f"part A {gi + 1}/{len(pool)} wall {time.time() - t_a:.0f}s", flush=True)
    wall_a = time.time() - t_a
    print(f"part A done: {gi + 1} of {len(pool)} elites scanned, {len(candidates)} liftable with a stored air basis, wall {wall_a:.0f}s", flush=True)

    # ---- control check: the arm-(ii) law on fly.py's own quad ---------------
    from dytiscidae.core.bodyplans import REFERENCE_PLANS
    from types import SimpleNamespace
    qel = SimpleNamespace(genome=REFERENCE_PLANS["quad"]())
    qenv = make_env(qel, 0)
    qR, qslots = rotor_channels(qenv)
    qc, qd = fly_arm(qel, 0, qR, qslots, "direct", hz=args.hz)
    fh.write(json.dumps({"part": "C", "plan": "quad", "seed": 0, "competence": qc,
                         "airborne_fraction": qd.get("airborne_fraction")}) + "\n")
    print(f"control check: quad, arm (ii): competence {qc:.3f}", flush=True)

    # ---- Part B ----------------------------------------------------------
    t_b = time.time()
    nb = min(args.bodies, args.limit) if args.limit else args.bodies
    chosen = stratified_bodies(candidates, nb, seed=0)
    print(f"part B: {len(chosen)} bodies chosen from {len(candidates)} (budget {args.budget_s:.0f}s)", flush=True)
    per_body = []
    for bi, c in enumerate(chosen):
        if per_body and (time.time() - t_b) + float(np.mean(per_body)) > args.budget_s:
            print(f"part B: time budget reached, stopping before body {bi}", flush=True)
            fh.write(json.dumps({"part": "B", "index": c["index"], "island": c["island"], "plan": c["plan"],
                                 "n_rotors": c["n_rotors"], "i_best": None, "ii": None,
                                 "why": "not run: time budget"}) + "\n")
            for rest in chosen[bi + 1:]:
                fh.write(json.dumps({"part": "B", "index": rest["index"], "island": rest["island"],
                                     "plan": rest["plan"], "n_rotors": rest["n_rotors"],
                                     "i_best": None, "ii": None, "why": "not run: time budget"}) + "\n")
            break
        tb = time.time()
        rec = {"part": "B", "index": c["index"], "island": c["island"], "plan": c["plan"],
               "n_rotors": c["n_rotors"], "eval_seed": c["seed"], "m_frac": c["m_frac"], "f_frac": c["f_frac"],
               "i": {}, "i_detail": {}}
        try:
            ii, ii_d = fly_arm(c["elite"], c["seed"], c["R"], c["slots"], "direct", hz=args.hz)
            rec["ii"], rec["ii_detail"] = ii, ii_d
            for G in GAINS:
                v, dd = fly_arm(c["elite"], c["seed"], c["R"], c["slots"], "basis", G=G, hz=args.hz,
                                basis=c["basis"])
                rec["i"][str(G)], rec["i_detail"][str(G)] = v, dd
            g_best = max(GAINS, key=lambda g: rec["i"][str(g)])
            rec["g_best"], rec["i_best"] = g_best, rec["i"][str(g_best)]
        except Exception as exc:  # noqa: BLE001
            rec.update({"i_best": None, "ii": None, "why": f"{type(exc).__name__}: {exc}"})
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        per_body.append(time.time() - tb)
        print(f"part B body {bi + 1}/{len(chosen)} (elite {c['index']}, {c['island']}, {c['n_rotors']} rotors): "
              f"(ii) {rec.get('ii')} (i)-best {rec.get('i_best')} [G {rec.get('g_best')}] "
              f"{per_body[-1]:.1f}s/body", flush=True)
    fh.close()
    return time.time() - t_a, wall_a, time.time() - t_b, per_body, len(candidates)


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
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N rotor elites; <= N bodies in Part B")
    ap.add_argument("--bodies", type=int, default=24)
    ap.add_argument("--budget-s", type=float, default=1800.0, help="Part B wall budget; stops starting bodies past it")
    ap.add_argument("--lift-test", choices=("up", "spec"), default="up",
                    help="up (default): upward thrust at top speed >= 1.2 m g; spec: the direction-blind sum as typed")
    ap.add_argument("--hz", type=float, default=0.0, help="control-law rate; 0 = every physics step")
    ap.add_argument("--cpu", action="store_true", help="accepted for symmetry with N8; always numpy here")
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
        here / f"smoke_N6_result.md" if smoke else ROOT / "runs/analysis_1010_failure_theory/N6_result.md")
    if "runs/arch" in str(report.resolve()) or "runs/arch" in str(out.resolve()):
        raise SystemExit("refusing to write into runs/arch*")
    rows_path = Path(args.from_rows) if args.from_rows else Path(str(out) + ".rows.jsonl")
    if args.from_rows:
        wall, wall_a, wall_b, per_body, n_cand = float("nan"), float("nan"), float("nan"), [], 0
    else:
        wall, wall_a, wall_b, per_body, n_cand = measure(args, rows_path)
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines() if ln.strip()]
    agg = aggregate(rows)
    ver = verdict(agg)
    sha, dirty = git_state()
    cmd = ("PYTHONPATH=. MUJOCO_GL=disable DYTISCIDAE_KERNEL_DIR=/home/hundo/Projects/Dytiscidae/dytiscidae/mojo/build "
           ".venv/bin/python experiments/rotor_authority/run.py " + " ".join(sys.argv[1:]))
    meta = {"date": time.strftime("%Y-%m-%d"), "commit": sha, "dirty": dirty, "command": cmd.strip(), "run": args.run,
            "wall_s": wall, "wall_a_s": wall_a, "wall_b_s": wall_b, "hz": args.hz, "budget_s": args.budget_s,
            "limit": args.limit, "lift_test": args.lift_test, "per_body_wall_s": per_body, "n_liftable_with_air_basis": n_cand}
    out.write_text(json.dumps({"meta": meta, "aggregate": agg, "verdict": ver}, indent=1))
    write_report(report, agg=agg, ver=ver, meta=meta, rows=rows)
    print(json.dumps({"verdict": ver, "n_rotor_elites": agg["n_rotor_elites"], "lift_excluded": agg["lift_excluded"],
                      "n_bodies_ran": agg["n_bodies_ran"], "m_frac_air": agg["air"]["m_frac"],
                      "f_frac_air": agg["air"]["f_frac"], "i_best": agg["i_best"], "ii": agg["ii"],
                      "per_body_wall_s": per_body, "wall_a_s": wall_a, "wall_b_s": wall_b}, indent=1, default=str))
    print(f"report: {report}")


if __name__ == "__main__":
    main()
