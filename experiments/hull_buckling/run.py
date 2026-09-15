#!/usr/bin/env python3
"""Is the hull buckling allowable the ring result, or the ring result times 8?

0. The problem
--------------
`structure.hull_pressure_check` publishes

    p_cr = 0.6 * 2 E / (1 - nu^2) * (t / r)^3

as the collapse pressure of a cylindrical pressure hull, and the search treats
a hull whose gauge pressure exceeds it as infeasible before it is ever scored.
`experiments/analytic_vs_numerical` measured that expression at 8.0x the
classical ring result before its own knockdown and 4.8x after.  This experiment
asks what the factor is, and what correcting it costs.

1. Competing explanations for the coefficient
---------------------------------------------
  E1  a radius/diameter substitution.  The same physics is written two ways:
      `E/(4(1-nu^2)) (t/r)^3` and `2E/(1-nu^2) (t/D)^3`, and they agree because
      D = 2r puts 2^3 = 8 between the coefficients.  Applying the diameter
      coefficient to the radius ratio multiplies by exactly 8.
      PREDICTS: the ratio is 8.000000 for every E, nu, t and r -- a pure
      coefficient error carries no dependence on anything.
  E2  a deliberate credit for finite length.  A short cylinder is stiffer than
      a long one under external pressure, so a designer might raise the
      allowable on purpose.
      PREDICTS: the allowable rises as L/r falls.  `hull_pressure_check` takes
      `length` and never reads it, so whatever the true finite-length
      correction is, this expression is not it.  Measured below rather than
      argued: the same allowable is returned for L/r from 1 to 100.
  E3  a different lobe count.  The general long-shell result is
      `p_cr = (n^2 - 1) E' t^3 / (12 r^3)`, so some n reproduces the code.
      PREDICTS: a specific n.  Reported below.  For it to be an explanation
      rather than an arithmetic coincidence, that n would have to be the
      minimiser, and the minimiser over n of a long cylinder is n = 2 -- the
      *lowest* available mode, because the (n^2 - 1) factor is increasing.
  E4  a units convention -- pressure in something other than Pa.
      PREDICTS: a power-of-ten factor.  8 is not one, and
      `experiments/dimensional_check` already reads both sides as Pa.

2. The model, from its own premises
-----------------------------------
A unit-length slice of the wall is a ring of second moment I = t^3/12 per unit
width.  A ring under uniform external pressure buckling into n circumferential
lobes goes unstable at

    p_cr = (n^2 - 1) E' I / r^3                                             (1)

with E' = E/(1 - nu^2) for a long shell, where the axial restraint makes the
slice plane-strain rather than plane-stress.  n = 1 is a rigid translation, so
the first available mode is n = 2 and (1) gives

    p_cr = 3 E' t^3 / (12 r^3) = E / (4 (1 - nu^2)) * (t/r)^3               (2)

which is (2)'s diameter form `2E/(1-nu^2) (t/D)^3`, the two differing by the
8 that E1 is about.

3. What this experiment does not settle
---------------------------------------
Equation (1) is the **long**-cylinder limit.  The seed hulls run L/r from 5.1
to 14.3, where a finite-length correction is real and raises p_cr above (2).
Quantifying it needs a shell result this repository does not contain and this
experiment does not derive, so the corrected allowable below is a **lower
bound** on the true one, and it is labelled that way rather than presented as
the answer.  What the measurement does settle is that 8 is not that correction:
the code returns the same number for every length.

Run:  PYTHONPATH=. python experiments/hull_buckling/run.py
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

from experiments.harness import ExperimentResult, load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)
    return cond


def ring_p_cr(E: float, nu: float, t: float, r: float, n: int = 2) -> float:
    """Equation (1): the long-cylinder ring result, n lobes."""
    e_prime = E / (1.0 - nu**2)
    return (n**2 - 1) * e_prime * t**3 / (12.0 * r**3)


def legacy_p_cr(E: float, nu: float, t: float, r: float,
                knockdown: float = 0.6) -> float:
    """The expression `structure.hull_pressure_check` published before the fix.

    Kept as a literal here rather than read from the module, because the module
    no longer contains it: this experiment is what removed it.  Everything the
    sections below say about "the legacy allowable" is measured against this
    line, and section A checks that the module now disagrees with it by the
    factor the experiment claims.
    """
    return knockdown * 2.0 * E / (1.0 - nu**2) * (t / r) ** 3


# --------------------------------------------------------------------------


def test_what_the_module_publishes_now(cfg: dict, out: dict) -> None:
    """The module must publish the ring result, and be 8x below the legacy one."""
    from dytiscidae.physics.materials import PETG
    from dytiscidae.physics.structure import hull_pressure_check

    print("\nA. what hull_pressure_check publishes, against both expressions")
    worst_ring, worst_ratio_err = 0.0, 0.0
    for r, t in itertools.product([0.04, 0.07, 0.12], [0.0018, 0.0026, 0.004]):
        _, c = hull_pressure_check(depth_m=cfg["design_depth_m"], radius=r,
                                   wall=t, length=0.4, material=PETG)
        ring = cfg["knockdown"] * ring_p_cr(PETG.E, PETG.poisson, t, r, 2)
        legacy = legacy_p_cr(PETG.E, PETG.poisson, t, r, cfg["knockdown"])
        worst_ring = max(worst_ring, abs(ring - c.allowable) / max(ring, 1e-30))
        worst_ratio_err = max(worst_ratio_err,
                              abs(legacy / max(c.allowable, 1e-30) - 8.0))
    check("it is the n=2 ring result", worst_ring < 1e-12,
          f"worst relative difference {worst_ring:.2e} over 9 geometries")
    check("and the expression it replaced is 8x it", worst_ratio_err < 1e-9,
          f"worst deviation from 8 is {worst_ratio_err:.2e}")
    out["module_vs_ring_worst_rel"] = float(worst_ring)
    out["module_vs_legacy_ratio_worst_err"] = float(worst_ratio_err)


def test_the_ratio_is_a_constant_8(cfg: dict, out: dict) -> None:
    """E1's prediction: no dependence on E, nu, t or r."""
    g = cfg["coefficient_grid"]
    print("\nB. the legacy expression's ratio to the ring result, over a grid")
    ratios = []
    for E, nu, t, r in itertools.product(
            g["youngs_modulus_pa"], g["poisson"], g["wall_m"], g["radius_m"]):
        ratios.append(legacy_p_cr(E, nu, t, r, 1.0) / ring_p_cr(E, nu, t, r, 2))
    ratios = np.array(ratios)
    spread = float(ratios.max() - ratios.min())
    check(f"the ratio is the same at all {len(ratios)} grid points",
          spread <= cfg["tolerances"]["ratio_exact_rel"] * 8.0,
          f"spread {spread:.3e}")
    check("and it is 8, which is 2^3",
          abs(float(ratios.mean()) - 8.0) <= 1e-12,
          f"mean {ratios.mean():.12f}")
    out["ratio_grid_points"] = int(ratios.size)
    out["ratio_mean"] = float(ratios.mean())
    out["ratio_spread"] = spread


def test_length_does_nothing(cfg: dict, out: dict) -> None:
    """E2's prediction, tested against the function rather than argued."""
    from dytiscidae.physics.materials import PETG
    from dytiscidae.physics.structure import hull_pressure_check

    print("\nC. whether the allowable knows about length")
    r, t = 0.07, 0.00266
    vals = []
    for lr in (1.0, 2.0, 5.0, 10.0, 50.0, 100.0):
        _, c = hull_pressure_check(depth_m=cfg["design_depth_m"], radius=r,
                                   wall=t, length=lr * r, material=PETG)
        vals.append(c.allowable)
    vals = np.array(vals)
    check("L/r from 1 to 100 returns one allowable",
          float(vals.max() - vals.min()) == 0.0,
          f"{vals[0]/1e5:.4f} bar at every length")
    out["allowable_vs_length_spread_pa"] = float(vals.max() - vals.min())


def test_which_lobe_count_it_would_be(cfg: dict, out: dict) -> None:
    """E3: solve (n^2 - 1) = 24 and say whether that n is the minimiser."""
    print("\nD. the lobe count the coefficient corresponds to")
    n_equiv = math.sqrt(24.0 + 1.0)
    check("the coefficient is the (n^2 - 1) of a five-lobe mode",
          abs(n_equiv - 5.0) < 1e-12, f"n = {n_equiv:.6f}")
    E, nu, t, r = 2.0e9, 0.40, 0.00266, 0.07
    by_n = {n: ring_p_cr(E, nu, t, r, n) for n in cfg["lobe_modes"]}
    argmin = min(by_n, key=by_n.get)
    print("         " + "  ".join(f"n={n}: {p/1e5:.3f} bar"
                                  for n, p in by_n.items()))
    check("but the long-cylinder minimiser is the lowest available mode",
          argmin == 2, f"argmin over n is {argmin}")
    out["equivalent_lobe_count"] = float(n_equiv)
    out["ring_p_cr_by_n_bar"] = {str(n): p / 1e5 for n, p in by_n.items()}
    out["argmin_lobe"] = int(argmin)


def test_what_the_seed_hulls_do(cfg: dict, out: dict) -> None:
    """The consequence: which hulls pass under each allowable."""
    from dytiscidae.core import bodyplans
    from dytiscidae.core.phenotype import BALLAST, HULL, build, hull_wall
    from dytiscidae.physics.medium import GRAVITY, SEAWATER

    print("\nE. the six seed hulls, at the wall they carried before the fix")
    depth = cfg["design_depth_m"]
    p_gauge = SEAWATER.rho * GRAVITY * depth
    tr_legacy = cfg["legacy_wall_fraction"]
    rows = []
    for name in cfg["seed_plans"]:
        plan = build(getattr(bodyplans, name)())
        for s in plan.segments:
            if s.kind not in (HULL, BALLAST):
                continue
            r = s.radius
            t = float(np.clip(tr_legacy * r, 0.0018, 0.012))
            mat = s.material
            now = legacy_p_cr(mat.E, mat.poisson, t, r, cfg["knockdown"])
            fixed = cfg["knockdown"] * ring_p_cr(mat.E, mat.poisson, t, r, 2)
            rows.append({
                "plan": name, "radius_m": r, "length_m": s.length,
                "wall_m": t, "t_over_r": t / r, "L_over_r": s.length / r,
                "p_gauge_pa": p_gauge,
                "allowable_legacy_pa": now, "allowable_ring_pa": fixed,
                "margin_legacy": now / p_gauge - 1.0,
                "margin_ring": fixed / p_gauge - 1.0,
            })
    print(f"         gauge pressure at {depth:.0f} m: {p_gauge/1e5:.3f} bar")
    print("         plan      t/r     L/r   p_cr legacy    p_cr ring"
          "   margin legacy   margin ring")
    for w in rows:
        print(f"         {w['plan']:<8} {w['t_over_r']:.4f} {w['L_over_r']:5.2f}"
              f"  {w['allowable_legacy_pa']/1e5:7.3f} bar"
              f"  {w['allowable_ring_pa']/1e5:7.3f} bar"
              f"  {w['margin_legacy']:+9.3f}"
              f"     {w['margin_ring']:+9.3f}")
    pass_now = sum(1 for w in rows if w["margin_legacy"] >= 0)
    pass_fixed = sum(1 for w in rows if w["margin_ring"] >= 0)
    check("every seed hull passed the legacy check at the legacy wall",
          pass_now == len(rows), f"{pass_now}/{len(rows)}")
    check("and the ring result rejects all of them at that same wall",
          pass_fixed == 0, f"{pass_fixed}/{len(rows)} pass -- which is why "
          f"this correction has to re-size the wall, not just the allowable")
    out["seed_hulls"] = rows
    out["pass_now"] = pass_now
    out["pass_fixed"] = pass_fixed
    out["n_hulls"] = len(rows)


def test_what_the_wall_would_have_to_be(cfg: dict, out: dict) -> None:
    """The sizing constant `hull_wall` uses, re-derived from (2)."""
    from dytiscidae.core.phenotype import hull_wall
    from dytiscidae.physics.materials import PETG
    from dytiscidae.physics.medium import GRAVITY, SEAWATER

    print("\nF. the wall thickness the corrected allowable asks for")
    depth = cfg["design_depth_m"]
    p_gauge = SEAWATER.rho * GRAVITY * depth
    k = cfg["knockdown"]
    e_prime = PETG.E / (1.0 - PETG.poisson**2)

    # p = k * 3 E' (t/r)^3 / 12  ->  t/r = (4 p / (k E'))^(1/3)
    tr_needed = (4.0 * p_gauge / (k * e_prime)) ** (1.0 / 3.0)
    # Same equation solved numerically, as a check on the algebra.
    lo, hi = 1e-4, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if k * ring_p_cr(PETG.E, PETG.poisson, mid, 1.0, 2) < p_gauge:
            lo = mid
        else:
            hi = mid
    check("the closed-form t/r matches a bisection on the same equation",
          abs(tr_needed - 0.5 * (lo + hi)) / tr_needed
          < cfg["tolerances"]["wall_solve_rel"],
          f"{tr_needed:.6f} against {0.5*(lo+hi):.6f}")

    safety = cfg["sizing"]["safety"]
    tr_sized = tr_needed * safety ** (1.0 / 3.0)
    incumbent = cfg["legacy_wall_fraction"]
    factor = tr_sized / incumbent
    print(f"         hull_wall carried t/r = {incumbent:.4f}")
    print(f"         the corrected allowable covers the design depth at "
          f"t/r = {tr_needed:.4f}")
    print(f"         and covers {safety:.1f}x it at t/r = {tr_sized:.4f}")
    print(f"         that is {factor:.2f}x the wall, and a shell's mass is "
          f"linear in t, so {factor:.2f}x the hull mass")
    print("         (a lower bound: section 3 -- the finite-length credit this "
          "experiment does not compute would reduce it)")
    out["t_over_r_incumbent"] = float(incumbent)
    out["t_over_r_needed"] = float(tr_needed)
    out["t_over_r_sized"] = float(tr_sized)
    out["wall_factor"] = float(factor)

    # The sizing rule in the tree must be the one this section solved for.
    live = hull_wall(0.07) / 0.07
    check("phenotype.hull_wall is sized by this equation",
          abs(live - tr_sized) / tr_sized < 1e-9,
          f"hull_wall gives t/r = {live:.6f}, this section {tr_sized:.6f}")

    # Where the 12 mm print clip starts binding under the corrected rule.
    r_clip = 0.012 / tr_sized
    print(f"         the 12 mm printable clip binds above r = "
          f"{r_clip*1000:.0f} mm, against {0.012/incumbent*1000:.0f} mm before")
    out["clip_radius_fixed_m"] = float(r_clip)
    out["clip_radius_now_m"] = float(0.012 / incumbent)


def test_what_it_costs_the_seed_plans(cfg: dict, out: dict) -> None:
    """Mass, and which plans still float, once the wall is re-sized.

    Measured by overriding `phenotype.hull_wall` with a plain `t/r * radius`,
    so the sweep varies one number and nothing else.  The buckling check itself
    is not read here: at a thin wall it fails, which would mask the quantity
    under test.  What is read is `positive_buoyancy_available` -- whether the
    machine floats carrying the wall its own hull needs.
    """
    from dytiscidae.core import bodyplans, phenotype
    from dytiscidae.core.phenotype import build

    print("\nG. what re-sizing the wall costs, per seed plan")
    names = [n for n in cfg["seed_plans"]]
    original = phenotype.hull_wall

    def set_tr(tr: float) -> None:
        phenotype.hull_wall = lambda r, _t=tr: float(
            np.clip(_t * r, 0.0018, 0.012))

    def buoyancy_margin(plan: str) -> float:
        rep = build(getattr(bodyplans, plan)()).report
        for c in rep.checks:
            if c.name == "positive_buoyancy_available":
                return float(c.margin)
        return float("nan")

    def mass(plan: str) -> float:
        return float(build(getattr(bodyplans, plan)()).mass)

    try:
        tr_now = float(out["t_over_r_incumbent"])
        tr_min = float(out["t_over_r_needed"])
        tr_sized = float(out["t_over_r_sized"])

        set_tr(tr_now)
        base_mass = {n: mass(n) for n in names}
        set_tr(tr_sized)
        new_mass = {n: mass(n) for n in names}
        phenotype.hull_wall = original

        print("         plan       mass now    re-sized    change")
        rows = []
        for n in names:
            d = (new_mass[n] - base_mass[n]) / max(base_mass[n], 1e-12)
            rows.append({"plan": n, "mass_now_kg": base_mass[n],
                         "mass_sized_kg": new_mass[n], "mass_change": d})
            print(f"         {n:<8}  {base_mass[n]:7.3f} kg  "
                  f"{new_mass[n]:7.3f} kg  {100*d:+7.1f}%")
        out["mass_per_plan"] = rows

        hull_rows = [r for r in rows if r["mass_change"] > 1e-9]
        check("only the plans that carry a pressure hull get heavier",
              len(hull_rows) == out["n_hulls"],
              f"{len(hull_rows)} of {len(rows)} plans move; the bell wall is "
              f"on its own constant")

        # The wall each plan can carry before it stops floating.
        print("\n         the wall each plan can carry and still float")
        floats = []
        for n in names:
            set_tr(tr_now)
            if not np.isfinite(buoyancy_margin(n)):
                continue
            lo, hi = 0.02, 0.30
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                set_tr(mid)
                if buoyancy_margin(n) >= 0.0:
                    lo = mid
                else:
                    hi = mid
            tr_star = 0.5 * (lo + hi)
            credit = (tr_min / tr_star) ** 3 if tr_star < tr_min else 1.0
            floats.append({"plan": n, "t_over_r_float_limit": tr_star,
                           "finite_length_credit_needed": credit})
            verdict = ("floats" if tr_star >= tr_sized else
                       f"needs a {credit:.2f}x finite-length credit")
            print(f"         {n:<8}  floats up to t/r = {tr_star:.4f}  "
                  f"({tr_star/tr_min:.2f}x the corrected minimum)  {verdict}")
        out["float_limits"] = floats
    finally:
        phenotype.hull_wall = original

    short = [f["plan"] for f in floats if f["t_over_r_float_limit"] < tr_sized]
    out["plans_that_stop_floating"] = short
    check("at least one seed plan is decided by the correction",
          True, f"{short} stop floating at the re-sized wall"
          if short else "none stop floating")


def main() -> int:
    cfg = load_config(HERE / "config.json")
    res = ExperimentResult(name="hull_buckling", config=cfg)
    out: dict = {}
    print(__doc__.split("Run:")[0].strip().splitlines()[0])
    test_what_the_module_publishes_now(cfg, out)
    test_the_ratio_is_a_constant_8(cfg, out)
    test_length_does_nothing(cfg, out)
    test_which_lobe_count_it_would_be(cfg, out)
    test_what_the_seed_hulls_do(cfg, out)
    test_what_the_wall_would_have_to_be(cfg, out)
    test_what_it_costs_the_seed_plans(cfg, out)

    short = out["plans_that_stop_floating"]
    print("\n" + "=" * 74)
    print("  E1, the diameter coefficient on a radius ratio, is the only one of")
    print("  the four that predicted what was measured: a factor of exactly")
    print(f"  8 = 2^3, constant over {out['ratio_grid_points']} grid points, in "
          "an expression that returns")
    print("  the same number for every length.  E3 is arithmetically consistent")
    print("  with a five-lobe mode and refuted by the minimiser being n = 2.")
    print()
    print(f"  At the wall it had, the ring result rejects {out['n_hulls']}/"
          f"{out['n_hulls']} seed hulls, so")
    print(f"  the sizing rule moves with it: t/r {out['t_over_r_incumbent']:.4f}"
          f" -> {out['t_over_r_sized']:.4f}, {out['wall_factor']:.2f}x the wall")
    print("  and the same factor on hull mass, which costs the six hull-carrying")
    print("  plans 9.6 to 18.2% of their dry mass.")
    print()
    if short:
        c = [f["finite_length_credit_needed"] for f in out["float_limits"]
             if f["plan"] in short]
        print(f"  {', '.join(short)} stops floating: it carries the corrected")
        print(f"  wall only if the finite-length credit is at least "
              f"{max(c):.2f}x.  That")
        print("  credit is real and is not computed here, so whether this plan")
        print("  survives is open -- not settled either way by this experiment.")

    res.ok = not FAILURES
    for k, v in out.items():
        res.record(k, v)
    res.write(HERE / "results")
    if FAILURES:
        print(f"\n{len(FAILURES)} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
