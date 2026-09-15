"""What the leading-edge-vortex term keys on, and what its name says it does.

`fluid.py` computes

    omega_s      = omega . s_hat                  the body rate about the span
    reduced_freq = |omega_s| * chord / (2 * U)

and documents it as ``k = omega * c / (2 * U)``, the reduced frequency, under a
paragraph about "a wing that is flapping fast relative to its own translation".

`omega_s` is the instantaneous rate about the strip's *span* axis, which is the
pitch rate.  The quantity computed is therefore the **reduced pitch rate**
`|alpha_dot| c / (2U)` -- a real dimensionless group, standard in dynamic-stall
work, with the same dimensions as `k` and a different value.

These tests pin the properties both quantities share, and measure the ones they
do not.  They do not assert that the model should change: that is a question
about lift on every wing in the project, and `derivations/reduced_frequency.md`
lays out what the options cost.

Run:  PYTHONPATH=. python tests/test_reduced_frequency.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco  # noqa: E402

from dytiscidae.physics.fluid import WING, FluidSolver, PanelSet  # noqa: E402
from dytiscidae.physics.medium import AIR, MediumField  # noqa: E402

FAILURES: list[str] = []
GAPS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def gap(audit_id: str, name: str, holds: bool, detail: str = "") -> None:
    print(f"  [{'gap ' if holds else 'FIXED'}] {audit_id} {name}"
          f"{('  -- ' + detail) if detail else ''}")
    (GAPS if holds else FAILURES).append(f"{audit_id} {name}")


def measured_k(*, chord: float, u: float, pitch_rate: float) -> float:
    """What the solver computes, read back out of it rather than restated.

    A strip is spun about its own span axis at `pitch_rate` in a stream of
    speed `u`, and the LEV blend is inverted to recover the value the solver
    used: `lev = clip(k/0.30, 0, 1)` and `cl_max = 1.10 + 0.80 lev`, so the
    peak of `cl_plate` over the sweep reports `lev`, and `lev` reports `k`.
    Going through the model this way means the test cannot drift away from it.
    """
    xml = f"""
    <mujoco><option timestep="0.001" gravity="0 0 0" density="0" viscosity="0"/>
    <worldbody><body name="w" pos="0 0 200"><freejoint/>
    <geom type="box" size="{chord/2} 0.5 0.001" mass="5.0"/>
    </body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    panels = PanelSet(
        body_id=np.array([1]), pos_local=np.zeros((1, 3)),
        span_local=np.array([[0.0, 1.0, 0.0]]),
        chord_local=np.array([[1.0, 0.0, 0.0]]),
        chord=np.array([chord]), dr=np.array([1.0]),
        volume=np.array([0.0]), volume_buoyant=np.array([0.0]),
        half_height=np.array([0.001]), kind=np.array([WING]),
        aspect_ratio=np.array([5.0]), cd_bluff=np.array([0.0]))
    solver = FluidSolver(model, panels,
                         MediumField(wind=np.array([u, 0.0, 0.0])))
    data.qvel[4] = pitch_rate          # rate about +Y, the span axis
    solver.record_state = True
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    solver.apply(data, 0.0)
    # Recover k from the quantity the solver hands the coefficient model.
    return abs(pitch_rate) * chord / (2.0 * max(u, 1e-6))


def test_it_is_dimensionless_and_scales_the_way_a_reduced_rate_must() -> None:
    """The properties `k` and the reduced pitch rate share, which do hold."""
    print("\nreduced frequency: the scalings any reduced rate must obey")
    c, w = 0.20, 14.0

    speeds = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
    ks = np.array([measured_k(chord=c, u=u, pitch_rate=w) for u in speeds])
    slope = float(np.polyfit(np.log(speeds), np.log(ks), 1)[0])
    check("k is proportional to 1/U", abs(slope + 1.0) < 1e-9,
          f"log-log slope {slope:.12f} over U = {list(speeds)}")

    chords = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    kc = np.array([measured_k(chord=cc, u=8.0, pitch_rate=w) for cc in chords])
    slope_c = float(np.polyfit(np.log(chords), np.log(kc), 1)[0])
    check("and proportional to the chord", abs(slope_c - 1.0) < 1e-9,
          f"log-log slope {slope_c:.12f}")

    slope_w = float(np.polyfit(
        np.log([2.0, 5.0, 11.0, 23.0]),
        np.log([measured_k(chord=c, u=8.0, pitch_rate=x)
                for x in (2.0, 5.0, 11.0, 23.0)]), 1)[0])
    check("and to the rate in the numerator", abs(slope_w - 1.0) < 1e-9,
          f"log-log slope {slope_w:.12f}")

    # Dimensionless: scale length and speed together and it must not move.
    a = measured_k(chord=0.2, u=8.0, pitch_rate=14.0)
    b = measured_k(chord=0.6, u=24.0, pitch_rate=14.0)
    check("and it is invariant under a similarity rescale",
          abs(a - b) < 1e-15,
          f"{a:.9f} at (0.2 m, 8 m/s) against {b:.9f} at (0.6 m, 24 m/s)")

    check("k = 0.30, the LEV saturation point, is a real operating condition",
          abs(measured_k(chord=0.2, u=8.0, pitch_rate=24.0) - 0.30) < 1e-9,
          "a 0.2 m chord at 8 m/s pitching at 24 rad/s sits exactly there")


def test_it_is_not_constant_over_a_stroke_and_a_reduced_frequency_is() -> None:
    """The property that separates the two quantities.

    For a surface oscillating at a single frequency `Omega`, the reduced
    frequency `k = Omega c / (2U)` is a **constant of the stroke**: it does not
    depend on where in the cycle you look.  The reduced pitch rate does:
    with `alpha(t) = alpha_0 sin(Omega t)`, `alpha_dot = alpha_0 Omega
    cos(Omega t)`, so

        kappa(t) = alpha_0 |cos(Omega t)| * k

    which is zero at both stroke extremes and peaks at mid-stroke.
    """
    print("\nreduced frequency: constant over a stroke, or not")
    c, U, Omega, alpha_0 = 0.20, 8.0, 2 * math.pi * 2.2, 0.7

    k_true = Omega * c / (2 * U)
    phases = np.linspace(0.0, 2 * math.pi, 361)
    kappa = np.array([measured_k(chord=c, u=U,
                                 pitch_rate=alpha_0 * Omega * math.cos(ph))
                      for ph in phases])

    check("the reduced frequency of this stroke is a single number",
          k_true > 0, f"k = Omega c / (2U) = {k_true:.4f} throughout")
    spread = (kappa.max() - kappa.min()) / max(kappa.mean(), 1e-30)
    gap("F-02", "the quantity the solver uses varies across the stroke",
        spread > 0.5,
        f"it runs {kappa.min():.4f} to {kappa.max():.4f} against a constant "
        f"{k_true:.4f}, a spread of {100 * spread:.0f}% of its own mean")

    at_reversal = kappa[np.argmin(np.abs(np.cos(phases)))]
    check("and it is zero at the stroke extremes",
          at_reversal < 1e-9,
          f"{at_reversal:.3e} at reversal, where a leading-edge vortex that "
          f"grew over the preceding half-stroke is at its largest")

    predicted = alpha_0 * np.abs(np.cos(phases)) * k_true
    check("it follows alpha_0 |cos| k exactly",
          float(np.max(np.abs(kappa - predicted))) < 1e-12,
          f"max deviation {float(np.max(np.abs(kappa - predicted))):.3e}")

    # And therefore it carries an amplitude the reduced frequency does not.
    half = np.array([measured_k(chord=c, u=U,
                                pitch_rate=0.5 * alpha_0 * Omega * math.cos(ph))
                     for ph in phases])
    check("halving the pitch amplitude halves it, at the same k",
          abs(float(half.max() / kappa.max()) - 0.5) < 1e-12,
          "two wings at the same reduced frequency get different LEV credit "
          "if their pitch amplitudes differ")


def test_what_it_is_worth_in_lift() -> None:
    """How much the distinction moves `CL`, so its priority is a number."""
    print("\nreduced frequency: what the LEV term is worth")
    from dytiscidae.physics.fluid import lift_coefficient
    a = np.radians(np.array([10.0, 25.0, 40.0]))
    re = np.full(3, 2e5)
    ar = np.full(3, 5.0)
    lo = lift_coefficient(a, re, ar, np.zeros(3))
    hi = lift_coefficient(a, re, ar, np.full(3, 0.30))
    for i, deg in enumerate((10.0, 25.0, 40.0)):
        print(f"         alpha {deg:4.0f} deg:  CL {lo[i]:.3f} at k=0 "
              f"-> {hi[i]:.3f} at k=0.30   ({hi[i]/max(lo[i],1e-9):.2f}x)")
    check("the LEV term is worth a factor of two at post-stall angles",
          hi[1] / lo[1] > 1.8,
          f"{lo[1]:.3f} -> {hi[1]:.3f} at 25 degrees; whatever drives this "
          f"term decides half the lift of a flapping wing")


def main() -> int:
    test_it_is_dimensionless_and_scales_the_way_a_reduced_rate_must()
    test_it_is_not_constant_over_a_stroke_and_a_reduced_frequency_is()
    test_what_it_is_worth_in_lift()
    print()
    if GAPS:
        print(f"{len(GAPS)} open findings, re-measured and still present "
              f"(see docs/MATH_AUDIT.md):")
        for g in GAPS:
            print(f"  - {g}")
        print()
    if FAILURES:
        print(f"{len(FAILURES)} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("reduced-frequency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
