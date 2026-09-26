"""The handover between the attached and separated branches of `lift_coefficient`.

The model writes CL as a partition of unity:

    CL = (1 - w) * CL_alpha * alpha_e  +  w * CN(lev) * sin(alpha) cos(alpha)

with ``w`` zero up to the static stall angle and one ``SEPARATION_COMPLETE``
past it, raised toward one by the leading-edge-vortex strength ``lev``.

Two defects this file has pinned, in order:

* F-05 (closed 2026-09-18): the weight was a logistic, never 0 and never 1, so
  each branch leaked into the other's domain -- 13.8% of the separated branch at
  zero incidence.  ``w`` and ``w'`` are exactly zero there, on a wing with no
  leading-edge vortex.
* F-08 (closed 2026-09-23): the handover ran from zero incidence to
  ``alpha_stall + 16 deg``, so CL rose monotonically to 45 degrees and the model
  had no stall.  It now starts at the stall angle, and lift falls after it.

And the separated branch is a normal-force plate, ``CN sin a cos a``, with
``CN`` from 1.98 (plate) to 3.4 (robofly, full LEV) -- the same ``CN`` the
drag's pressure term uses.

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
    CN_LEV, CN_PLATE, SEPARATION_COMPLETE, lift_coefficient)

FAILURES: list[str] = []

AR = (2.0, 4.0, 8.0, 16.0)
RE = (1.0e4, 1.0e5, 1.0e6)
LEV = (0.0, 0.25, 0.6, 1.0)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def cl(alpha_rad, ar: float, re: float, lev: float) -> np.ndarray:
    a = np.atleast_1d(np.asarray(alpha_rad, float))
    return lift_coefficient(a, np.full(a.shape, re), np.full(a.shape, ar),
                            np.full(a.shape, lev))


def attached_slope(ar: float) -> float:
    return 2.0 * np.pi / (1.0 + 2.0 / max(ar, 0.5))


def cn(lev: float) -> float:
    return CN_PLATE + (CN_LEV - CN_PLATE) * lev


def stall_angle(re: float) -> float:
    return np.radians(11.0) * float(np.clip(
        0.55 + 0.45 * np.log10(max(re, 10.0)) / 5.0, 0.5, 1.0))


def test_the_slope_at_zero() -> None:
    """F-05, over the whole regime: attached slope alone without an LEV, and
    exactly the declared mixture with one."""
    print("\nstall blend: the lift slope at zero incidence")
    worst, where = 0.0, None
    h = 1e-5
    for ar, re, lev in itertools.product(AR, RE, LEV):
        c = cl([-h, h], ar, re, lev)
        got = float((c[1] - c[0]) / (2 * h))
        want = (1.0 - lev) * attached_slope(ar) + lev * cn(lev)
        d = got / want - 1.0
        if abs(d) > abs(worst):
            worst, where = d, (ar, re, lev)
    check("the realised slope is (1-lev) CL_alpha + lev CN everywhere, and so "
          "the attached slope alone without an LEV", abs(worst) < 1e-6,
          f"worst deviation {100*worst:+.3e}% over "
          f"{len(AR)*len(RE)*len(LEV)} points"
          + (f" at AR={where[0]} Re={where[1]:.0e} lev={where[2]}"
             if abs(worst) >= 1e-6 else ""))


def test_the_weight_is_compactly_supported() -> None:
    """Exactly the attached branch up to stall, exactly the separated one past
    the separation width."""
    print("\nstall blend: where each branch is weighted")
    worst_a, worst_s, n = 0.0, 0.0, 0
    for ar, re in itertools.product(AR, RE):
        a_s = stall_angle(re)
        below = np.linspace(-a_s, a_s, 50)
        worst_a = max(worst_a, float(np.max(np.abs(
            cl(below, ar, re, 0.0) - attached_slope(ar) * below))))
        for lev in LEV:
            past = np.linspace(a_s + SEPARATION_COMPLETE, np.radians(90.0), 60)
            want = cn(lev) * np.sin(past) * np.cos(past)
            worst_s = max(worst_s, float(np.max(np.abs(cl(past, ar, re, lev) - want))))
            n += 1
    check("below stall, without an LEV, it is the attached branch alone",
          worst_a < 1e-12, f"worst residual {worst_a:.2e}")
    check("past the separation width it is CN(lev) sin a cos a alone",
          n > 0 and worst_s < 1e-12,
          f"{n} regime points, worst residual {worst_s:.2e}")


def test_it_stalls() -> None:
    """F-08: without an LEV, lift peaks at stall and is lower after it."""
    print("\nstall blend: a wing without a vortex stalls")
    rows = []
    for ar, re in itertools.product((4.0, 8.0, 16.0), (1.0e5, 1.0e6)):
        # Up to the end of the handover: a flat plate's *global* maximum is its
        # second peak near 45 degrees, CN sin a cos a, which a low-AR wing's
        # attached peak can be below -- as on real plates.  F-08 is the local
        # maximum at stall and the loss after it.
        a = np.linspace(0.0, stall_angle(re) + SEPARATION_COMPLETE, 3001)
        c = cl(a, ar, re, 0.0)
        a_pk = float(np.degrees(a[np.argmax(c)]))
        a_s = float(np.degrees(stall_angle(re)))
        after = float(cl([stall_angle(re) + SEPARATION_COMPLETE], ar, re, 0.0)[0])
        rows.append((ar, re, a_pk, a_s, float(c.max()), after))
    # The peak sits just past the stall angle, not on it: the handover is C1,
    # so w' = 0 at stall and the attached branch keeps rising for a moment.
    # "Just past" is the first third of the handover.
    third = np.degrees(SEPARATION_COMPLETE) / 3.0
    ok = all(s <= pk <= s + third and after < 0.9 * peak
             for _, _, pk, s, peak, after in rows)
    ar, re, pk, s, peak, after = rows[0]
    check("CL peaks within the first third of the handover and has lost >10% by its end",
          ok, f"AR {ar:g} Re {re:.0e}: peak {peak:.3f} at {pk:.2f} deg "
              f"(stall {s:.2f}), {after:.3f} {np.degrees(SEPARATION_COMPLETE):.0f} deg "
              f"later; {len(rows)} regimes")
    lev_rows = [float(cl([np.radians(25.0)], ar, 1e5, 1.0)[0]) for ar in AR]
    check("with a full LEV there is no drop: 25 deg is CN_LEV sin a cos a",
          all(abs(v - CN_LEV * np.sin(np.radians(25)) * np.cos(np.radians(25))) < 1e-12
              for v in lev_rows), f"{lev_rows[0]:.3f}")


def test_it_has_no_kink() -> None:
    """A corner in CL' is a discontinuity: the largest step in a numerical first
    derivative does not shrink when the grid is refined.  A C1 curve's does."""
    print("\nstall blend: whether the optimiser can find a corner")
    worst = 0.0
    for ar, re, lev in itertools.product(AR, (1.0e5,), LEV):
        jumps = []
        for n in (4001, 8001):
            a = np.radians(np.linspace(-90.0, 90.0, n))
            d1 = np.gradient(cl(a, ar, re, lev), a)
            jumps.append(float(np.abs(np.diff(d1)).max()))
        worst = max(worst, jumps[1] / max(jumps[0], 1e-30))
    check("halving the grid halves the largest step in CL', so CL' is "
          "continuous", worst < 0.75, f"worst ratio {worst:.3f}, against 0.5 "
                                      f"for a smooth curve and 1 for a corner")


def test_the_envelope() -> None:
    """No clip any more: the model is bounded by construction."""
    print("\nstall blend: the envelope")
    worst = 0.0
    a = np.radians(np.linspace(-90.0, 90.0, 3601))
    for ar, re, lev in itertools.product(AR, RE, LEV):
        worst = max(worst, float(np.max(np.abs(cl(a, ar, re, lev)))))
    check("|CL| never exceeds CN_LEV / 2", worst <= CN_LEV / 2 + 1e-12,
          f"max |CL| {worst:.4f} against {CN_LEV / 2:.2f}")


def test_the_separation_width() -> None:
    print("\nstall blend: the one angle this introduces")
    deg = float(np.degrees(SEPARATION_COMPLETE))
    check("SEPARATION_COMPLETE is 6 degrees (a modelling choice, MATH_AUDIT F-08)",
          abs(deg - 6.0) < 1e-9, f"{deg:.4f} deg")
    check("and it is declared as an angle, not as a bare number",
          0.0 < SEPARATION_COMPLETE < np.pi / 2,
          f"{SEPARATION_COMPLETE:.6f} rad")


def main() -> int:
    test_the_slope_at_zero()
    test_the_weight_is_compactly_supported()
    test_it_stalls()
    test_it_has_no_kink()
    test_the_envelope()
    test_the_separation_width()
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
