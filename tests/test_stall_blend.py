"""The handover between the attached and separated branches of `lift_coefficient`.

The model writes CL as a partition of unity:

    CL = (1 - w) * CL_alpha * alpha  +  w * CL_max * sin(2 alpha)

A partition of unity is a statement about *where each branch is valid*.  The
attached branch is valid below stall and the separated one above it, so `w` has
to be exactly 0 below some angle and exactly 1 above another.

It was a logistic, which is never either.  At the static stall angle its tail
reached zero incidence with weight 0.138, so a gliding wing lost up to 9.6% of
the attached lift slope the docstring names; and 10 degrees past stall an
unbounded linear extrapolation still carried 16% of the weight.  `MATH_AUDIT`
F-05, measured in `experiments/stall_blend`.

These tests pin the properties the replacement has to have, and the two it
must not quietly acquire: a kink for the optimiser to exploit, and a different
model above stall than the one that was there.

Run:  PYTHONPATH=. python tests/test_stall_blend.py
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "disable")

from dytiscidae.physics.fluid import (  # noqa: E402
    SEPARATION_COMPLETE, lift_coefficient)

FAILURES: list[str] = []

AR = (2.0, 4.0, 8.0, 16.0)
RE = (1.0e4, 1.0e5, 1.0e6)
KAPPA = (0.0, 0.05, 0.15, 0.30)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def cl(alpha_rad, ar: float, re: float, kappa: float) -> np.ndarray:
    a = np.atleast_1d(np.asarray(alpha_rad, float))
    return lift_coefficient(a, np.full(a.shape, re), np.full(a.shape, ar),
                            np.full(a.shape, kappa))


def attached_slope(ar: float) -> float:
    return 2.0 * np.pi / (1.0 + 2.0 / max(ar, 0.5))


def stall_angle(re: float, kappa: float) -> float:
    lev = min(kappa / 0.30, 1.0)
    a = np.radians(11.0 + 26.0 * lev)
    return a * float(np.clip(0.55 + 0.45 * np.log10(max(re, 10.0)) / 5.0,
                             0.5, 1.0))


def test_the_attached_slope_is_delivered_at_zero() -> None:
    """The defect F-05 names, over the whole regime."""
    print("\nstall blend: the lift slope at zero incidence")
    worst, where = 0.0, None
    for ar, re, k in itertools.product(AR, RE, KAPPA):
        h = 1e-5
        c = cl([-h, h], ar, re, k)
        got = float((c[1] - c[0]) / (2 * h))
        d = got / attached_slope(ar) - 1.0
        if abs(d) > abs(worst):
            worst, where = d, (ar, re, k)
    check("the realised slope is the attached slope everywhere",
          abs(worst) < 1e-9,
          f"worst deviation {100*worst:+.3e}% over "
          f"{len(AR)*len(RE)*len(KAPPA)} points"
          + (f" at AR={where[0]} Re={where[1]:.0e} kappa={where[2]}"
             if abs(worst) >= 1e-9 else ""))


def test_the_weight_is_compactly_supported() -> None:
    """Exactly zero at zero incidence, exactly one past the separation width."""
    print("\nstall blend: where each branch is weighted")
    # At alpha = 0 both branches give 0, so the weight is read off the slope:
    # the separated branch contributes 2*CL_max of slope, and the attached one
    # CL_alpha.  Recovering exactly CL_alpha is w(0) = 0 and w'(0) = 0 at once.
    ar, re, k = 16.0, 1.0e4, 0.0  # the worst case in the audit's measurement
    h = 1e-5
    c = cl([-h, h], ar, re, k)
    got = float((c[1] - c[0]) / (2 * h))
    check("w(0) = 0 and w'(0) = 0, so neither lift nor slope leaks to zero "
          "incidence", abs(got / attached_slope(ar) - 1.0) < 1e-9,
          f"slope {got:.6f} against {attached_slope(ar):.6f}")

    for ar, re, k in itertools.product((4.0,), RE, KAPPA):
        a_s = stall_angle(re, k)
        past = a_s + SEPARATION_COMPLETE + np.radians(0.5)
        lev = min(k / 0.30, 1.0)
        cmax = 1.10 + 0.80 * lev
        want = cmax * np.sin(2.0 * past)
        got = float(cl([past], ar, re, k)[0])
        if abs(got - want) > 1e-9:
            check("past the separation width it is the separated branch alone",
                  False, f"{got:.6f} against {want:.6f} at Re={re:.0e} "
                         f"kappa={k}")
            return
    check("past the separation width it is the separated branch alone", True,
          f"no attached-branch contribution beyond alpha_stall + "
          f"{np.degrees(SEPARATION_COMPLETE):.0f} deg")


def test_it_has_no_kink() -> None:
    """The property the logistic was chosen for, which must survive.

    A corner in CL' is a discontinuity: the largest step in a numerical first
    derivative does not shrink when the grid is refined.  A C1 curve's does,
    roughly in proportion to the step.
    """
    print("\nstall blend: whether the optimiser can find a corner")
    worst = 0.0
    for ar, re, k in itertools.product(AR, (1.0e5,), KAPPA):
        lev = min(k / 0.30, 1.0)
        clip = 1.2 * (1.10 + 0.80 * lev)
        jumps = []
        for n in (4001, 8001):
            a = np.radians(np.linspace(-90.0, 90.0, n))
            c = cl(a, ar, re, k)
            d1 = np.gradient(c, a)
            clipped = np.abs(np.abs(c) - clip) <= 1e-9
            free = ~clipped
            for shift in (1, 2):
                free &= ~np.roll(clipped, shift) & ~np.roll(clipped, -shift)
            steps = np.abs(np.diff(d1))
            keep = free[:-1] & free[1:]
            jumps.append(float(steps[keep].max()) if keep.any() else 0.0)
        worst = max(worst, jumps[1] / max(jumps[0], 1e-30))
    check("halving the grid halves the largest step in CL', so CL' is "
          "continuous", worst < 0.75, f"worst ratio {worst:.3f}, against 0.5 "
                                      f"for a smooth curve and 1 for a corner")


def test_the_clip_is_no_longer_reached() -> None:
    """`np.clip(cl, -1.2 CL_max, 1.2 CL_max)` is a corner wherever it binds."""
    print("\nstall blend: the envelope clip")
    worst = 0.0
    a = np.radians(np.linspace(-90.0, 90.0, 3601))
    for ar, re, k in itertools.product(AR, RE, KAPPA):
        lev = min(k / 0.30, 1.0)
        clip = 1.2 * (1.10 + 0.80 * lev)
        worst = max(worst, float(np.max(np.abs(cl(a, ar, re, k)))) / clip)
    check("CL stays strictly inside the envelope, so the clip never binds",
          worst < 1.0 - 1e-6,
          f"max |CL| reaches {100*worst:.2f}% of 1.2 CL_max; the logistic "
          f"reached 100.00%")


def test_the_separated_branch_is_untouched() -> None:
    """Above the separation width nothing has changed, by construction."""
    print("\nstall blend: what happens well past stall")
    worst = 0.0
    for ar, re, k in itertools.product(AR, RE, KAPPA):
        lev = min(k / 0.30, 1.0)
        cmax = 1.10 + 0.80 * lev
        a_s = stall_angle(re, k)
        a = np.linspace(a_s + SEPARATION_COMPLETE, np.radians(90.0), 400)
        want = cmax * np.sin(2.0 * a)
        worst = max(worst, float(np.max(np.abs(cl(a, ar, re, k) - want))))
    check("it is CL_max sin(2 alpha), the same separated branch as before",
          worst < 1e-9, f"worst absolute difference {worst:.2e} over "
                        f"{len(AR)*len(RE)*len(KAPPA)} regime points")


def test_the_separation_width_is_the_measured_one() -> None:
    print("\nstall blend: the one constant this introduces")
    deg = float(np.degrees(SEPARATION_COMPLETE))
    check("SEPARATION_COMPLETE is 16 degrees, as experiments/stall_blend "
          "measured", abs(deg - 16.0) < 1e-9, f"{deg:.4f} deg")
    check("and it is declared as an angle, not as a bare number",
          0.0 < SEPARATION_COMPLETE < np.pi / 2,
          f"{SEPARATION_COMPLETE:.6f} rad")


def main() -> int:
    test_the_attached_slope_is_delivered_at_zero()
    test_the_weight_is_compactly_supported()
    test_it_has_no_kink()
    test_the_clip_is_no_longer_reached()
    test_the_separated_branch_is_untouched()
    test_the_separation_width_is_the_measured_one()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("stall blend checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
