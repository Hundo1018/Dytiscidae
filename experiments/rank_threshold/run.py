#!/usr/bin/env python3
"""Is `0.08 * sigma_0` a numerical rank, or an engineering guess?

0. The problem
--------------
`MobilityBasis.rank` counted singular values above 8% of the largest and called
the result "how many genuinely independent things can this machine do".  That
sentence is a claim about numerical rank.  (This experiment is what produced the
rename: the attribute is now `control_rank`, with `numerical_rank` beside it and
the 8% exposed as `authority_threshold`.  The text below is kept in the tense it
was measured in.)  Numerical rank is not a fraction someone picked; it is the
count of singular values above the level at which the matrix's own uncertainty
could have produced them.  The two coincide only by accident.

1. Competing explanations for what the threshold should be
----------------------------------------------------------
  H1  machine precision.  rank = #{sigma > max(P,6) * eps * sigma_0}.
      That level is about 1.3e-15; 0.08 is thirteen orders above it, so this
      is not what the threshold is doing.
  H2  measurement noise.  Identification is an experiment: a different set of
      probe directions gives a different J.  By Weyl's inequality no singular
      value can be resolved below ||dJ||_2, so the honest floor is the
      seed-to-seed spread of J itself.
  H3  model error.  The map from CPG parameters to mean twist is not linear,
      and the least-squares residual bounds how much of the response a linear
      J explains at all.  A mode below the residual is fitting curvature.
  H4  usefulness.  A mode is worth keeping if commanding it delivers twist
      without saturating the joints.
  H5  underdetermination.  A machine with n joints presents P = 3n+1
      parameters to n_probes = 24 probes.  For P > 24 the least squares is
      underdetermined, so `lstsq` returns the minimum-norm solution inside the
      24-dimensional row space of whichever directions were drawn.  A second
      seed draws a different subspace, so J moves for a reason that has
      nothing to do with noise and everything to do with the probe budget --
      and the residual reads exactly zero either way, because an
      underdetermined system always fits.

2. Prediction
-------------
If 0.08 is a numerical rank, it sits at or above the H2 floor on most bodies,
and the singular-value spectrum shows a gap there.  If it is a guess, the
spectrum is a continuum through 0.08 and the floor is somewhere else entirely.

3. Measurement
--------------
Per (plan, medium, seed): the spectrum, the fit residual, rank at each
threshold, truncated reconstruction error, and the control error and joint
excursion of a damped solve restricted to that rank.
Per (plan, medium, seed-pair): ||J_a - J_b||_2 / sigma_0, the H2 floor, and the
principal angles between the two r-dimensional parameter subspaces -- because
mode index and sign are private coordinates of an SVD and the subspace is not.

Run:  PYTHONPATH=. python experiments/rank_threshold/run.py
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "disable")

from experiments import mobility_data as MD  # noqa: E402
from experiments.harness import (  # noqa: E402
    ExperimentResult, bootstrap_ci, load_config, subspace_angles,
)

TWIST_SCALE = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])


def damped_solve(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """The project's own inverse: c = (A A^T + lam I)^-1 A b, A is (r, 6)."""
    r = A.shape[0]
    if r == 0:
        return np.zeros(0)
    G = A @ A.T
    return np.linalg.solve(G + lam * np.eye(r), A) @ b


def joint_travel(plan: str) -> np.ndarray:
    """Per-parameter travel of the CPG vector, for a saturation measure.

    `CPGParams.clipped` bounds amplitude and offset by the joint range and the
    frequency to [0.1, 20] Hz; phase is unbounded.  The travel vector below is
    what a parameter excursion should be compared against.
    """
    env = MD.build_env(plan)
    span = env.cpg.hi - env.cpg.lo
    n = env.cpg.n
    return np.concatenate([span, np.full(n, 2 * np.pi), span, [19.9]])


def analyse(rec: dict, cfg: dict, travel: np.ndarray, rng) -> dict:
    basis = MD.fit(rec)
    J = MD.jacobian(rec)
    sigma = np.asarray(basis.authority, float)
    s0 = float(sigma[0]) if sigma.size else 0.0

    Y = rec["responses"] * TWIST_SCALE
    resid = Y - rec["deltas"] @ J
    residual_fraction = float(np.linalg.norm(resid) / max(np.linalg.norm(Y), 1e-30))

    # Random intents in [-1, 1]^6, scaled the way `coeffs_for_twist` scales them.
    from dytiscidae.control.cpg import INTENT_AUTHORITY
    A_full = basis.effects * sigma[:, None]
    reach = np.linalg.norm(A_full, axis=0)
    W = rng.uniform(-1.0, 1.0, size=(cfg["n_intents"], 6))
    B = W * reach * INTENT_AUTHORITY

    per_threshold = {}
    for tau in cfg["thresholds"]:
        r = int(np.sum(sigma > tau * s0)) if s0 > 0 else 0
        A = A_full[:r]
        if r == 0:
            per_threshold[str(tau)] = {
                "rank": 0, "recon_error": 1.0, "control_error": 1.0,
                "cond": float("inf"), "max_excursion": 0.0, "effort": 0.0}
            continue
        lam = 0.01 * float(np.trace(A @ A.T)) / max(r, 1) + 1e-12
        C = np.array([damped_solve(A, b, lam) for b in B])
        delivered = C @ A
        err = np.linalg.norm(delivered - B, axis=1) / np.maximum(
            np.linalg.norm(B, axis=1), 1e-30)
        # Truncated reconstruction of the Jacobian itself.
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        Jr = (U[:, :r] * S[:r]) @ Vt[:r]
        recon = float(np.linalg.norm(J - Jr) / max(np.linalg.norm(J), 1e-30))
        delta = C @ basis.modes[:r]            # (n_intents, P) parameter offsets
        excursion = float(np.max(np.abs(delta) / travel[None, :]))
        per_threshold[str(tau)] = {
            "rank": r,
            "recon_error": recon,
            "control_error": float(np.mean(err)),
            "control_error_p95": float(np.percentile(err, 95)),
            "cond": float(sigma[0] / max(sigma[r - 1], 1e-30)),
            "max_excursion": excursion,
            "effort": float(np.mean(np.linalg.norm(C, axis=1))),
        }

    return {
        "plan": rec["plan"], "medium": rec["medium"], "seed": rec["seed"],
        "n_params": int(rec["n_params"]),
        "n_probes": int(rec["deltas"].shape[0]),
        "underdetermined": bool(rec["n_params"] > rec["deltas"].shape[0]),
        "sigma": sigma.tolist(),
        "sigma_normalised": (sigma / max(s0, 1e-30)).tolist(),
        "residual_fraction": residual_fraction,
        "eps_rank_threshold": float(max(J.shape) * np.finfo(float).eps),
        "thresholds": per_threshold,
    }


def noise_floor(recs_by_seed: list[dict]) -> dict:
    """H2: how much does J move when only the probe directions change?"""
    Js = [MD.jacobian(r) for r in recs_by_seed]
    svds = [np.linalg.svd(J, full_matrices=False) for J in Js]
    s0 = [float(s[1][0]) for s in svds]
    pairs = list(itertools.combinations(range(len(Js)), 2))
    rel = []
    mode_ang: dict[int, list[float]] = {}
    effect_ang: dict[int, list[float]] = {}
    for i, j in pairs:
        d = np.linalg.norm(Js[i] - Js[j], 2)
        rel.append(d / max(0.5 * (s0[i] + s0[j]), 1e-30))
        Ui, _, Vti = svds[i]
        Uj, _, Vtj = svds[j]
        for r in range(1, min(6, Ui.shape[1]) + 1):
            mode_ang.setdefault(r, []).append(float(np.degrees(
                subspace_angles(Ui[:, :r], Uj[:, :r]).max())))
            effect_ang.setdefault(r, []).append(float(np.degrees(
                subspace_angles(Vti[:r].T, Vtj[:r].T).max())))
    # Reproducibility of the singular values themselves.
    sig = np.array([s[1] for s in svds])
    cv = np.std(sig, axis=0) / np.maximum(np.mean(sig, axis=0), 1e-30)
    return {
        "n_pairs": len(pairs),
        "relative_dJ_mean": float(np.mean(rel)),
        "relative_dJ_max": float(np.max(rel)),
        "mode_subspace_angle_deg": {str(r): float(np.mean(v))
                                    for r, v in sorted(mode_ang.items())},
        "effect_subspace_angle_deg": {str(r): float(np.mean(v))
                                      for r, v in sorted(effect_ang.items())},
        "sigma_cv": cv.tolist(),
    }


def run(cfg: dict) -> ExperimentResult:
    res = ExperimentResult("rank_threshold", cfg)
    recs = MD.collect(cfg["plans"], cfg["media"], cfg["seeds"])
    rng = np.random.default_rng(cfg["intent_seed"])
    travels = {p: joint_travel(p) for p in cfg["plans"]}

    rows = [analyse(r, cfg, travels[r["plan"]], rng) for r in recs]
    floors = {}
    for plan in cfg["plans"]:
        for medium in cfg["media"]:
            group = [r for r in recs if r["plan"] == plan and r["medium"] == medium]
            if len(group) > 1:
                floors[f"{plan}/{medium}"] = noise_floor(group)

    tau_inc = cfg["incumbent_threshold"]

    # ---- report -----------------------------------------------------------
    print("\nH5 -- is the fit even determined?")
    print(f"  {'plan':<8} {'params':>7} {'probes':>7}  verdict")
    for plan in cfg["plans"]:
        row = next(r for r in rows if r["plan"] == plan)
        verdict = ("UNDERDETERMINED -- lstsq returns the minimum-norm solution "
                   "inside the probes' own row space"
                   if row["underdetermined"] else "determined")
        print(f"  {plan:<8} {row['n_params']:>7} {row['n_probes']:>7}  {verdict}")
    n_under = len({r["plan"] for r in rows if r["underdetermined"]})
    print(f"  {n_under} of {len(cfg['plans'])} seed plans are underdetermined.")

    print("\nsingular-value spectra, normalised to sigma_0 "
          f"(incumbent threshold {tau_inc})")
    for row in rows:
        if row["seed"] != cfg["seeds"][0]:
            continue
        spec = " ".join(f"{v:6.3f}" for v in row["sigma_normalised"])
        print(f"  {row['plan']:<7} {row['medium']:<5} {spec}   "
              f"resid={row['residual_fraction']:.3f}"
              f"{'  (exact: underdetermined)' if row['underdetermined'] else ''}")

    print(f"\nH2 -- identification noise floor, ||dJ||_2 / sigma_0 across "
          f"{len(cfg['seeds'])} probe seeds")
    floor_vals = []
    for key, f in sorted(floors.items()):
        floor_vals.append(f["relative_dJ_mean"])
        verdict = "ABOVE floor" if tau_inc > f["relative_dJ_mean"] else "BELOW floor"
        print(f"  {key:<16} mean={f['relative_dJ_mean']:.3f} "
              f"max={f['relative_dJ_max']:.3f}   0.08 is {verdict}")
    point, lo, hi = bootstrap_ci(floor_vals, seed=cfg["intent_seed"])
    print(f"  across all bodies and media: mean {point:.3f} "
          f"(95% CI {lo:.3f}-{hi:.3f}), n={len(floor_vals)}")
    print(f"  For scale: two matrices drawn independently with the same "
          f"spectrum would sit near sqrt(2) = 1.414; two identical ones at 0.")

    print("\nwhich side of the decomposition survives a reseed?")
    print(f"  {'rank r':>7}  {'modes (parameter side)':>24}  "
          f"{'effects (twist side)':>22}")
    ang_summary = {}
    for r in range(1, 7):
        ma = [f["mode_subspace_angle_deg"].get(str(r)) for f in floors.values()]
        ea = [f["effect_subspace_angle_deg"].get(str(r)) for f in floors.values()]
        ma = [x for x in ma if x is not None]
        ea = [x for x in ea if x is not None]
        if not ma:
            continue
        print(f"  {r:>7}  {np.mean(ma):>21.1f} deg  {np.mean(ea):>19.1f} deg")
        ang_summary[str(r)] = {"modes_deg": float(np.mean(ma)),
                               "effects_deg": float(np.mean(ea))}
    print("  90 deg is 'the two identifications share nothing in that "
          "subspace'; 0 deg is 'they agree exactly'.")
    cvs = np.array([f["sigma_cv"] for f in floors.values()])
    print("\n  coefficient of variation of each singular value across the "
          "four seeds:")
    print("   " + "  ".join(f"s{i}={v:.2f}" for i, v in
                            enumerate(np.mean(cvs, axis=0))))

    print("\nrank as a function of threshold (mean over bodies/media/seeds)")
    print(f"  {'tau':>7}  {'rank':>5}  {'recon err':>9}  {'ctrl err':>9}  "
          f"{'cond':>8}  {'max joint excursion':>20}")
    summary = {}
    for tau in cfg["thresholds"]:
        key = str(tau)
        ranks = [r["thresholds"][key]["rank"] for r in rows]
        recon = [r["thresholds"][key]["recon_error"] for r in rows]
        ctrl = [r["thresholds"][key]["control_error"] for r in rows]
        cond = [r["thresholds"][key]["cond"] for r in rows]
        exc = [r["thresholds"][key]["max_excursion"] for r in rows]
        mark = "  <- incumbent" if tau == tau_inc else ""
        print(f"  {tau:7.3f}  {np.mean(ranks):5.2f}  {np.mean(recon):9.4f}  "
              f"{np.mean(ctrl):9.4f}  {np.median(cond):8.1f}  "
              f"{np.mean(exc):20.3f}{mark}")
        summary[key] = {
            "mean_rank": float(np.mean(ranks)),
            "rank_spread": [int(np.min(ranks)), int(np.max(ranks))],
            "mean_recon_error": float(np.mean(recon)),
            "mean_control_error": float(np.mean(ctrl)),
            "median_cond": float(np.median(cond)),
            "mean_max_excursion": float(np.mean(exc)),
        }

    # Is there a gap in the spectrum at the incumbent threshold?
    allnorm = np.concatenate([np.asarray(r["sigma_normalised"])[1:] for r in rows])
    below = float(np.mean(allnorm < tau_inc))
    near = float(np.mean((allnorm > tau_inc / 2) & (allnorm < tau_inc * 2)))
    print(f"\nspectral gap test: of {allnorm.size} non-leading singular values, "
          f"{100 * below:.1f}% fall below {tau_inc}")
    print(f"  and {100 * near:.1f}% land within a factor of two of it -- "
          "a gap would put this near zero")

    # Rank stability: does the rank a threshold reports survive a reseed?
    print("\nrank stability across probe seeds (how often all "
          f"{len(cfg['seeds'])} seeds agree on the rank)")
    stability = {}
    for tau in cfg["thresholds"]:
        agree = []
        for plan in cfg["plans"]:
            for medium in cfg["media"]:
                rs = [r["thresholds"][str(tau)]["rank"] for r in rows
                      if r["plan"] == plan and r["medium"] == medium]
                if rs:
                    agree.append(len(set(rs)) == 1)
        stability[str(tau)] = float(np.mean(agree))
        print(f"  tau={tau:<6} unanimous on {100 * np.mean(agree):5.1f}% "
              f"of the {len(agree)} (body, medium) pairs")

    res.record("rows", rows)
    res.record("subspace_angles", ang_summary)
    res.record("sigma_cv_mean", np.mean(cvs, axis=0).tolist())
    res.record("n_underdetermined_plans", int(n_under))
    res.record("noise_floors", floors)
    res.record("noise_floor_mean", point)
    res.record("noise_floor_ci95", [lo, hi])
    res.record("threshold_summary", summary)
    res.record("rank_stability", stability)
    res.record("fraction_below_incumbent", below)
    res.record("fraction_within_factor_two", near)
    return res


if __name__ == "__main__":
    cfg = load_config(Path(__file__).parent / "config.json")
    result = run(cfg)
    result.write(Path(__file__).parent / "results")
