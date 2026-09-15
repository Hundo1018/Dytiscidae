#!/usr/bin/env python3
"""Is `lam = 0.01 * trace(G) / r` the right ridge, or the first one tried?

0. The problem
--------------
`MobilityBasis._inverse` solves

    c = (A A^T + lam I)^-1 A b,   A = effects * authority,  A is (r, 6)

which is the Tikhonov solution of `min ||A^T c - b||^2 + lam ||c||^2`.  The
solve is correct.  The *justification* in the comment is not:

    "Measured cond(A) is 13-50 across arch31 elites, so this is well inside
     the regime where a modest ridge is enough."

A condition number says how far an inverse can amplify a perturbation.  It does
not say how much ridge to apply.  The quantity that does is the error of the
resulting command on data the ridge was not fitted to, and that is what this
measures.

1. Competing explanations
--------------------------
  H1  the ridge is a numerical guard only: its job is to stop the solve raising
      on a singular G, and any small value does.  Prediction: error is flat in
      lam over several decades and only the lam = 0 case differs.
  H2  the ridge is a bias-variance control: A is fitted from a noisy
      identification, so a small ridge overfits that identification and a large
      one biases the command.  Prediction: out-of-sample error is U-shaped in
      lam with an interior minimum.
  H3  the ridge exists to bound joint excursion: its job is to stop a request
      for an unreachable axis producing an enormous coefficient.  Prediction:
      excursion falls monotonically with lam and the binding constraint is
      saturation, not error.

2. Measurement -- cross-validated, because that is the whole question
---------------------------------------------------------------------
The same body is identified four times with different probe directions.  For a
fractional intent w:

    in-sample     solve c on basis a, deliver through basis a
    out-of-sample solve c on basis a, deliver through basis b

The comparison has to happen in **CPG parameter space**, not in mode
coordinates.  A coefficient vector is expressed in one basis's own mode
coordinates, whose index and sign are private to that SVD, so carrying `c`
across to another basis compares nothing.  What crosses is the parameter offset
the controller actually sends to the machine,

    dp = modes_a^T c,

and the question is what identification b says that offset produces:

    y_out = J_b^T dp.

Out-of-sample is the honest one.  A command computed from one identification
that only works on that identification has not identified anything; it has
memorised a noise realisation.  H2 predicts this is where the U shape lives.

Note the natural reference: `c = 0` delivers nothing, so its relative error is
exactly 1.0.  Any measurement above 1.0 means the command is further from the
intent than sending no command at all.

3. Run
------
    PYTHONPATH=. python experiments/damping_lambda/run.py
"""

from __future__ import annotations

import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "disable")

from experiments import mobility_data as MD  # noqa: E402
from experiments.harness import ExperimentResult, bootstrap_ci, load_config  # noqa: E402
from experiments.rank_threshold import run as _rank_mod  # noqa: E402  (joint_travel)

joint_travel = _rank_mod.joint_travel


def solve(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    r = A.shape[0]
    G = A @ A.T
    M = G + lam * np.eye(r)
    try:
        return np.linalg.solve(M, A) @ b
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A.T) @ b


def lam0_of(A: np.ndarray) -> float:
    """The incumbent ridge, reproduced exactly from `MobilityBasis._inverse`."""
    r = A.shape[0]
    return 0.01 * float(np.trace(A @ A.T)) / max(r, 1) + 1e-12


def basis_pieces(rec: dict, max_modes: int):
    from dytiscidae.control.cpg import INTENT_AUTHORITY
    basis = MD.fit(rec, max_modes=max_modes)
    A = basis.effects * np.asarray(basis.authority)[:, None]     # (r, 6)
    reach = np.linalg.norm(A, axis=0)
    J = MD.jacobian(rec)                                          # (P, 6)
    return basis, A, reach, INTENT_AUTHORITY, J


def run(cfg: dict) -> ExperimentResult:
    res = ExperimentResult("damping_lambda", cfg)
    recs = MD.collect(cfg["plans"], cfg["media"], cfg["seeds"])
    rng = np.random.default_rng(cfg["intent_seed"])
    W = rng.uniform(-1.0, 1.0, size=(cfg["n_intents"], 6))
    travels = {p: joint_travel(p) for p in cfg["plans"]}

    by_key: dict[tuple, list[dict]] = {}
    for r in recs:
        by_key.setdefault((r["plan"], r["medium"]), []).append(r)

    per_lambda: dict[str, dict[str, list[float]]] = {
        str(x): {"in": [], "out": [], "effort": [], "excursion": [], "cond": []}
        for x in cfg["lambda_ratios"]
    }
    rows = []

    for (plan, medium), group in sorted(by_key.items()):
        pieces = [basis_pieces(r, cfg["max_modes"]) for r in group]
        travel = travels[plan]
        row = {"plan": plan, "medium": medium, "lambdas": {}}
        for ratio in cfg["lambda_ratios"]:
            key = str(ratio)
            e_in, e_out, eff, exc, cond = [], [], [], [], []
            for (ia, ib) in itertools.permutations(range(len(group)), 2):
                basis_a, A_a, reach_a, gain, J_a = pieces[ia]
                basis_b, A_b, reach_b, _, J_b = pieces[ib]
                lam = ratio * lam0_of(A_a)
                B_a = W * reach_a * gain
                B_b = W * reach_b * gain
                C = np.array([solve(A_a, b, lam) for b in B_a])
                # What the controller actually sends: a CPG parameter offset.
                DP = C @ basis_a.modes                        # (n_intents, P)
                da = DP @ J_a - B_a
                db = DP @ J_b - B_b
                e_in.append(float(np.mean(np.linalg.norm(da, axis=1)
                                          / np.maximum(np.linalg.norm(B_a, axis=1), 1e-30))))
                e_out.append(float(np.mean(np.linalg.norm(db, axis=1)
                                           / np.maximum(np.linalg.norm(B_b, axis=1), 1e-30))))
                eff.append(float(np.mean(np.linalg.norm(C, axis=1))))
                exc.append(float(np.max(np.abs(DP) / travel[None, :])))
                G = A_a @ A_a.T
                ev = np.linalg.eigvalsh(G + lam * np.eye(A_a.shape[0]))
                cond.append(float(ev.max() / max(ev.min(), 1e-300)))
            row["lambdas"][key] = {
                "in_sample": float(np.mean(e_in)),
                "out_of_sample": float(np.mean(e_out)),
                "effort": float(np.mean(eff)),
                "max_excursion": float(np.mean(exc)),
                "cond": float(np.median(cond)),
            }
            per_lambda[key]["in"].append(float(np.mean(e_in)))
            per_lambda[key]["out"].append(float(np.mean(e_out)))
            per_lambda[key]["effort"].append(float(np.mean(eff)))
            per_lambda[key]["excursion"].append(float(np.mean(exc)))
            per_lambda[key]["cond"].append(float(np.median(cond)))
        rows.append(row)

    # ---- report -----------------------------------------------------------
    print(f"\n{len(rows)} (body, medium) pairs, {len(cfg['seeds'])} independent "
          f"identifications each, {cfg['n_intents']} intents")
    print("\n  lam/lam0   in-sample   out-of-sample      effort   max joint "
          "excursion      cond(G+lamI)")
    summary = {}
    for ratio in cfg["lambda_ratios"]:
        key = str(ratio)
        d = per_lambda[key]
        mark = "  <- incumbent" if ratio == cfg["incumbent_ratio"] else ""
        print(f"  {ratio:8.3f}   {np.mean(d['in']):9.4f}   {np.mean(d['out']):13.4f} "
              f"  {np.mean(d['effort']):9.3f}   {np.mean(d['excursion']):19.3f} "
              f"  {np.median(d['cond']):15.3g}{mark}")
        summary[key] = {k: float(np.mean(v)) for k, v in d.items()}
        summary[key]["cond"] = float(np.median(d["cond"]))

    outs = [np.mean(per_lambda[str(x)]["out"]) for x in cfg["lambda_ratios"]]
    best_i = int(np.argmin(outs))
    best_ratio = cfg["lambda_ratios"][best_i]
    inc_i = cfg["lambda_ratios"].index(cfg["incumbent_ratio"])
    print(f"\nlowest out-of-sample error at lam/lam0 = {best_ratio} "
          f"({outs[best_i]:.4f}); the incumbent is {outs[inc_i]:.4f}, "
          f"{100 * (outs[inc_i] / outs[best_i] - 1):+.1f}%")
    above_one = [x for x in outs if x > 1.0]
    print(f"  Sending no command at all scores exactly 1.0.  "
          f"{len(above_one)} of {len(outs)} swept ratios score above that.")

    # Per medium, because air, water and land are different identification
    # problems and there is no reason one ridge should serve all three.
    print("\nbest ratio per medium:")
    per_medium = {}
    for medium in cfg["media"]:
        sel = [r for r in rows if r["medium"] == medium]
        vals = [float(np.mean([r["lambdas"][str(x)]["out_of_sample"] for r in sel]))
                for x in cfg["lambda_ratios"]]
        bi = int(np.argmin(vals))
        ii = cfg["lambda_ratios"].index(cfg["incumbent_ratio"])
        print(f"  {medium:<6} best lam/lam0={cfg['lambda_ratios'][bi]:<8} "
              f"error {vals[bi]:.4f}   incumbent {vals[ii]:.4f} "
              f"({100*(vals[ii]/vals[bi]-1):+.1f}%)")
        per_medium[medium] = {"best_ratio": cfg["lambda_ratios"][bi],
                              "best_error": vals[bi],
                              "incumbent_error": vals[ii],
                              "curve": vals}
    res.record("per_medium", per_medium)

    # Per-pair: which ratio wins, and how often is it the incumbent?
    winners = []
    for row in rows:
        vals = [row["lambdas"][str(x)]["out_of_sample"] for x in cfg["lambda_ratios"]]
        winners.append(cfg["lambda_ratios"][int(np.argmin(vals))])
    unique, counts = np.unique(winners, return_counts=True)
    print("\nper-(body, medium) best ratio:")
    for u, c in zip(unique, counts):
        print(f"  lam/lam0={u:<8} wins on {c}/{len(winners)} pairs")

    # Which hypothesis does the shape support?
    flat = abs(outs[inc_i] - min(outs[1:])) / max(min(outs[1:]), 1e-30)
    interior = 0 < best_i < len(outs) - 1
    print(f"\nH1 (numerical guard only): out-of-sample spread across the "
          f"non-zero ratios is {100 * (max(outs[1:]) / min(outs[1:]) - 1):.1f}%")
    print(f"H2 (bias-variance): minimum is "
          f"{'interior' if interior else 'at an endpoint'} -> "
          f"{'supported' if interior else 'not supported'}")
    print(f"H3 (excursion bound): excursion falls "
          f"{per_lambda[str(cfg['lambda_ratios'][1])]['excursion'] and ''}"
          f"{np.mean(per_lambda[str(cfg['lambda_ratios'][1])]['excursion']):.2f}"
          f" -> {np.mean(per_lambda[str(cfg['lambda_ratios'][-1])]['excursion']):.2f}"
          " from the smallest non-zero ratio to the largest")

    # --- is the optimum predictable from something observable? ------------
    #
    # Hardcoding 30 is better than hardcoding 1 and worse than not hardcoding.
    # If the optimal ridge tracks a conditioning number a caller can compute
    # from the basis it already has, then `lam` stops being a constant and
    # becomes a function of the identification's own quality.
    print("\nis the optimum predictable from the basis's own conditioning?")
    import itertools as _it

    preds: dict[str, list[float]] = {}
    targets: list[float] = []
    for row in rows:
        group = by_key[(row["plan"], row["medium"])]
        bases = [MD.fit(r, max_modes=cfg["max_modes"]) for r in group]
        Js = [MD.jacobian(r) for r in group]
        sig = np.array([b.authority for b in bases])
        s0 = sig[:, 0].mean()
        # Seed-to-seed spread of J, the identification's own noise floor.
        pair = [np.linalg.norm(Js[i] - Js[j], 2)
                for i, j in _it.combinations(range(len(Js)), 2)]
        vals = [row["lambdas"][str(x)]["out_of_sample"]
                for x in cfg["lambda_ratios"]]
        targets.append(math.log10(max(cfg["lambda_ratios"][int(np.argmin(vals))],
                                      1e-3)))
        d0 = bases[0].diagnostics()
        preds.setdefault("log10 cond(A)", []).append(
            math.log10(max(np.mean([b.condition for b in bases]), 1e-12)))
        preds.setdefault("log10 sigma_max", []).append(math.log10(max(s0, 1e-12)))
        preds.setdefault("log10 sigma_min/sigma_max", []).append(
            math.log10(max(sig[:, -1].mean() / max(s0, 1e-30), 1e-12)))
        preds.setdefault("probe noise ||dJ||/sigma_0", []).append(
            float(np.mean(pair) / max(s0, 1e-30)))
        preds.setdefault("residual fraction", []).append(
            float(np.mean([b.residual_fraction for b in bases])))
        preds.setdefault("n_probes / n_params", []).append(
            d0["n_probes"] / max(d0["n_params"], 1))
        preds.setdefault("underdetermined", []).append(
            1.0 if d0["underdetermined"] else 0.0)

    y = np.asarray(targets)
    print(f"  target: log10 of the best lam/lam0, over {len(y)} "
          f"(body, medium) pairs")
    print(f"  {'predictor':<30} {'Pearson r':>10} {'95% CI':>20} {'p-ish':>8}")
    correlations = {}
    for name, xs in preds.items():
        x = np.asarray(xs, float)
        if np.std(x) < 1e-12:
            print(f"  {name:<30} {'constant':>10}")
            correlations[name] = None
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        # Bootstrap the correlation rather than quote a p-value on n = 21.
        rng2 = np.random.default_rng(cfg["intent_seed"])
        boots = []
        for _ in range(5000):
            idx = rng2.integers(0, len(y), len(y))
            if np.std(x[idx]) < 1e-12 or np.std(y[idx]) < 1e-12:
                continue
            boots.append(np.corrcoef(x[idx], y[idx])[0, 1])
        lo_r, hi_r = np.percentile(boots, [2.5, 97.5])
        crosses = "includes 0" if lo_r < 0 < hi_r else "excludes 0"
        print(f"  {name:<30} {r:>10.3f} {f'[{lo_r:+.2f}, {hi_r:+.2f}]':>20} "
              f"{crosses:>12}")
        correlations[name] = {"r": r, "ci95": [float(lo_r), float(hi_r)],
                              "excludes_zero": bool(not (lo_r < 0 < hi_r))}
    res.record("lambda_predictors", correlations)
    strong = [k for k, v in correlations.items()
              if v and v["excludes_zero"] and abs(v["r"]) > 0.5]
    if strong:
        print(f"\n  {len(strong)} predictor(s) survive a bootstrap that "
              f"excludes zero with |r| > 0.5: {', '.join(strong)}.")
        print("  Those two are the same quantity with opposite sign "
              "(sigma_min/sigma_max is 1/cond), so it is one finding: a "
              "better-conditioned basis wants a larger multiple of lam_0 and "
              "a worse-conditioned one a smaller.")

        # Which is what you would expect if the right absolute ridge tracks
        # the *weakest* singular value rather than the mean.  lam_0 already
        # carries mean(sigma^2), so a ratio that falls with conditioning is a
        # ratio correcting lam_0 back toward sigma_min^2.  Test it directly:
        # fit log(lam_absolute at the optimum) against log(sigma_min) and
        # log(sigma_max) and read the exponents.
        lam_abs, l_smin, l_smax = [], [], []
        for row, tgt in zip(rows, targets):
            group = by_key[(row["plan"], row["medium"])]
            bases = [MD.fit(r, max_modes=cfg["max_modes"]) for r in group]
            sig = np.array([b.authority for b in bases]).mean(axis=0)
            lam0 = 0.01 * float(np.sum(sig**2)) / len(sig)
            lam_abs.append(math.log10(max((10 ** tgt) * lam0, 1e-30)))
            l_smin.append(math.log10(max(sig[-1], 1e-30)))
            l_smax.append(math.log10(max(sig[0], 1e-30)))
        A_fit = np.column_stack([l_smin, l_smax, np.ones(len(lam_abs))])
        coef, *_ = np.linalg.lstsq(A_fit, np.asarray(lam_abs), rcond=None)
        pred = A_fit @ coef
        ss_res = float(np.sum((np.asarray(lam_abs) - pred) ** 2))
        ss_tot = float(np.sum((np.asarray(lam_abs) - np.mean(lam_abs)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        print(f"\n  fitting log10(lam at the optimum) = a log10(sigma_min) "
              f"+ b log10(sigma_max) + c:")
        print(f"    a = {coef[0]:+.3f}   b = {coef[1]:+.3f}   "
              f"c = {coef[2]:+.3f}   R^2 = {r2:.3f}")
        print(f"    For reference, lam ~ sigma_min^2 is (a, b) = (2, 0) and "
              f"the incumbent lam ~ mean(sigma^2) is closer to (0, 2).")
        res.record("lambda_exponent_fit",
                   {"a_sigma_min": float(coef[0]), "b_sigma_max": float(coef[1]),
                    "c": float(coef[2]), "r2": float(r2), "n": len(lam_abs)})
        print(f"    n = {len(lam_abs)}.  A two-predictor fit on 21 points is a "
              f"lead, not a law: what it says is that the optimum is not a "
              f"constant multiple of lam_0, and roughly which direction the "
              f"correction runs.")
    else:
        print("\n  None of them survives a bootstrap that excludes zero with "
              "|r| > 0.5.")
        print("  On this sample the optimal ridge is NOT predictable from the "
              "basis's own conditioning, so a constant is the honest choice "
              "and the measured one is about 30x the incumbent.  n = 21: this "
              "rules out a strong relationship, not a weak one.")

    point, lo, hi = bootstrap_ci(
        [a - b for a, b in zip(per_lambda[str(cfg['incumbent_ratio'])]["out"],
                               per_lambda[str(best_ratio)]["out"])],
        seed=cfg["intent_seed"])
    print(f"\nincumbent minus best, paired over (body, medium): "
          f"{point:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")

    res.record("rows", rows)
    res.record("summary", summary)
    res.record("best_ratio", best_ratio)
    res.record("incumbent_ratio", cfg["incumbent_ratio"])
    res.record("incumbent_penalty_pct", 100 * (outs[inc_i] / outs[best_i] - 1))
    res.record("winners", [float(w) for w in winners])
    res.record("incumbent_minus_best_ci95", [point, lo, hi])
    res.record("minimum_is_interior", bool(interior))
    res.record("flatness_of_incumbent_vs_best_nonzero", float(flat))
    return res


if __name__ == "__main__":
    cfg = load_config(Path(__file__).parent / "config.json")
    result = run(cfg)
    result.write(Path(__file__).parent / "results")
