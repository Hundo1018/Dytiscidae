"""Properties of the hull's external-pressure allowable and the wall that meets it.

`hull_pressure_check` published `0.6 * 2E/(1-nu^2) * (t/r)^3` as the collapse
pressure of a pressure hull.  `2E/(1-nu^2)` is the coefficient of the
**diameter** form; on a radius ratio it is 8x too large, and the check gates a
design before it is ever scored, so the search was being told that hulls it
cannot build are feasible.

These tests pin what the corrected pair has to satisfy: the allowable is the
n=2 ring result, `wall_for_buckling` inverts it exactly, and the sizing rule
and the check agree by construction rather than by two separately fitted
constants -- which is how the 8x survived, because `hull_wall`'s 0.038 was
fitted to it.

Run:  PYTHONPATH=. python tests/test_hull_buckling.py
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "disable")

from dytiscidae.core import phenotype  # noqa: E402
from dytiscidae.physics import structure  # noqa: E402
from dytiscidae.physics.materials import (  # noqa: E402
    AL6061, CFRP_TUBE, PETG, PETG_CF)
from dytiscidae.physics.medium import GRAVITY, SEAWATER  # noqa: E402

FAILURES: list[str] = []
MATERIALS = (PETG, PETG_CF, CFRP_TUBE, AL6061)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def test_the_allowable_is_the_ring_result() -> None:
    """p_cr = k (n^2-1) E' t^3 / (12 r^3), independently written out here."""
    print("\nhull buckling: the allowable against the ring result")
    worst = 0.0
    for mat, r, t in itertools.product(MATERIALS, [0.03, 0.07, 0.15],
                                       [0.0018, 0.004, 0.012]):
        e_prime = mat.E / (1.0 - mat.poisson**2)
        want = 0.6 * 3.0 * e_prime * t**3 / (12.0 * r**3)
        got = structure.hull_buckling_pressure(radius=r, wall=t, material=mat)
        worst = max(worst, abs(got - want) / want)
    check("it is the n=2 ring result on every material and geometry",
          worst < 1e-12, f"worst relative difference {worst:.2e} over "
                         f"{len(MATERIALS) * 9} points")

    # The unit-substitution that was there: the diameter coefficient is 8x.
    mat, r, t = PETG, 0.07, 0.00266
    diameter_form = 0.6 * 2.0 * mat.E / (1 - mat.poisson**2) * (t / (2 * r))**3
    got = structure.hull_buckling_pressure(radius=r, wall=t, material=mat)
    check("and it agrees with the diameter form written correctly",
          abs(diameter_form - got) / got < 1e-12,
          f"{diameter_form:.6f} against {got:.6f} Pa")


def test_it_scales_the_way_a_bending_stiffness_does() -> None:
    """Cube in t, inverse cube in r, linear in E', and no length term."""
    print("\nhull buckling: the scalings")
    base = dict(radius=0.07, wall=0.003, material=PETG)
    p0 = structure.hull_buckling_pressure(**base)

    p_t = structure.hull_buckling_pressure(radius=0.07, wall=0.006,
                                           material=PETG)
    check("doubling the wall multiplies the allowable by 8",
          abs(p_t / p0 - 8.0) < 1e-12, f"{p_t/p0:.12f}")

    p_r = structure.hull_buckling_pressure(radius=0.14, wall=0.003,
                                           material=PETG)
    check("doubling the radius divides it by 8",
          abs(p0 / p_r - 8.0) < 1e-12, f"{p0/p_r:.12f}")

    ratio = (PETG_CF.E / (1 - PETG_CF.poisson**2)) / (
        PETG.E / (1 - PETG.poisson**2))
    p_m = structure.hull_buckling_pressure(radius=0.07, wall=0.003,
                                           material=PETG_CF)
    check("and it is linear in E/(1-nu^2)",
          abs(p_m / p0 - ratio) < 1e-12, f"{p_m/p0:.9f} against {ratio:.9f}")

    check("the n=2 mode is the one a long cylinder goes in",
          min(range(2, 9),
              key=lambda n: structure.hull_buckling_pressure(
                  radius=0.07, wall=0.003, material=PETG, n=n)) == 2,
          "argmin over n = 2..8")


def test_the_sizing_rule_inverts_the_check() -> None:
    """The property that stops the two drifting apart again."""
    print("\nhull buckling: sizing against grading")
    worst = 0.0
    for mat, r, depth, safety in itertools.product(
            MATERIALS, [0.03, 0.07, 0.15], [4.0, 12.0, 40.0], [1.0, 1.3, 2.0]):
        t = structure.wall_for_buckling(radius=r, depth_m=depth, material=mat,
                                        safety=safety)
        p_cr = structure.hull_buckling_pressure(radius=r, wall=t, material=mat)
        want = safety * SEAWATER.rho * GRAVITY * depth
        worst = max(worst, abs(p_cr - want) / want)
    check("the wall it returns is exactly the wall the check asks for",
          worst < 1e-12, f"worst relative difference {worst:.2e} over "
                         f"{len(MATERIALS) * 27} combinations")

    t0 = structure.wall_for_buckling(radius=0.07, depth_m=12.0, material=PETG)
    t1 = structure.wall_for_buckling(radius=0.14, depth_m=12.0, material=PETG)
    check("and it is a fraction of radius, not a constant",
          abs(t1 / t0 - 2.0) < 1e-12, f"{t1/t0:.12f}")


def test_the_phenotype_uses_it() -> None:
    """`hull_wall` must be the inverse of the check, not a refitted literal."""
    print("\nhull buckling: what phenotype.hull_wall is")
    for r in (0.03, 0.05, 0.07, 0.10, 0.15):
        want = structure.wall_for_buckling(
            radius=r, depth_m=phenotype.HULL_DESIGN_DEPTH_M, material=PETG,
            safety=phenotype.HULL_BUCKLING_SAFETY)
        got = phenotype.hull_wall(r)
        clipped = not (0.0018 <= want <= 0.012)
        if clipped:
            continue
        if abs(got - want) / want >= 1e-12:
            check(f"hull_wall({r}) is wall_for_buckling", False,
                  f"{got:.9f} against {want:.9f}")
            return
    check("hull_wall is wall_for_buckling wherever the print clips do not bind",
          True, "checked at r = 30, 50, 70, 100, 150 mm")

    # The margin it is designed to, read back off the check it is graded by.
    p_gauge = SEAWATER.rho * GRAVITY * phenotype.HULL_DESIGN_DEPTH_M
    r = 0.07
    p_cr = structure.hull_buckling_pressure(
        radius=r, wall=phenotype.hull_wall(r), material=PETG)
    check("a hull sized by it clears its own check by the declared safety",
          abs(p_cr / p_gauge - phenotype.HULL_BUCKLING_SAFETY) < 1e-9,
          f"{p_cr/p_gauge:.6f} against "
          f"{phenotype.HULL_BUCKLING_SAFETY:.6f}")


def test_the_bell_did_not_move() -> None:
    """A bell is not a pressure hull, so correcting one must not move the other."""
    print("\nhull buckling: what the bell wall does")
    for r in (0.03, 0.08, 0.20):
        want = max(0.35 * 0.038 * r, 0.0012)
        check(f"bell_wall({r*1000:.0f} mm) is the value it had before the fix",
              abs(phenotype.bell_wall(r) - want) < 1e-15,
              f"{phenotype.bell_wall(r)*1000:.3f} mm")
    check("and it does not read hull_wall",
          "hull_wall" not in phenotype.bell_wall.__code__.co_names,
          f"names: {phenotype.bell_wall.__code__.co_names}")


def test_the_seed_plans() -> None:
    """The consequence, pinned so a later change to it is visible in a diff."""
    from dytiscidae.core import bodyplans
    from dytiscidae.core.phenotype import build

    print("\nhull buckling: the seed plans under the corrected rule")
    for name in ("bat", "beetle", "eel", "gannet", "medusa", "ray", "teal"):
        rep = build(getattr(bodyplans, name)()).report
        buck = [c for c in rep.checks if c.name == "hull_buckling"]
        if not buck:
            check(f"{name} carries no pressure hull", True, "no hull_buckling")
            continue
        m = min(c.margin for c in buck)
        check(f"{name}'s hull clears buckling by the declared safety",
              abs(m - (phenotype.HULL_BUCKLING_SAFETY - 1.0)) < 1e-6,
              f"margin {m:+.4f}")

    # Stated, not asserted away: one plan loses positive buoyancy.
    short = []
    for name in ("bat", "beetle", "eel", "gannet", "medusa", "ray", "teal"):
        rep = build(getattr(bodyplans, name)()).report
        for c in rep.checks:
            if c.name == "positive_buoyancy_available" and not c.ok:
                short.append((name, c.margin))
    print(f"         plans that no longer float: "
          f"{[(n, round(m, 4)) for n, m in short] or 'none'}")
    check("bat is the only seed plan the correction sinks",
          [n for n, _ in short] == ["bat"],
          "it needs a 1.82x finite-length credit, which is real and is not "
          "computed anywhere in this repository")


def main() -> int:
    test_the_allowable_is_the_ring_result()
    test_it_scales_the_way_a_bending_stiffness_does()
    test_the_sizing_rule_inverts_the_check()
    test_the_phenotype_uses_it()
    test_the_bell_did_not_move()
    test_the_seed_plans()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("hull buckling checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
