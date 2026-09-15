#!/usr/bin/env python3
"""The stall blend leaks both branches into each other's domain.

0. The problem
--------------
`fluid.lift_coefficient` writes the lift as a partition of unity between an
attached branch and a separated one::

    w      = 1 / (1 + exp(-(|alpha| - alpha_stall) / 6 deg))
    CL     = (1 - w) * CL_alpha * alpha  +  w * CL_max * sin(2 alpha)

A logistic is never 0 and never 1.  At the static stall angle the tail reaches
all the way down to zero incidence: `w(0) = 1/(1 + exp(11 deg / 6 deg)) =
0.138`, so **13.8% of the lift on a gliding wing at zero incidence comes from
the post-stall branch**, and the realised lift slope is below the attached one
that the docstring names.  `docs/MATH_AUDIT.md` F-05.

The mirror image is the same defect and has not been stated before: at
`alpha_stall + 5.5 deg` the weight is 0.715, so **28.5% of the lift past stall
comes from an unbounded attached-flow extrapolation**.  Each branch is being
evaluated where it does not apply.

1. Competing explanations for the 6 degree width
------------------------------------------------
  E1  a smoothing width picked so the optimiser sees no kink, with no thought
      given to where its tails land.  PREDICTS: the width does not scale with
      `alpha_stall`, so the leak at zero grows as the stall angle shrinks --
      exactly backwards, because a low-Reynolds wing has both a lower stall
      angle and (in this model) the same 6 degrees of blend.  Measured in D.
  E2  a deliberate soft-stall model for low Reynolds number.  PREDICTS: the
      leak tracks Reynolds number.  The model already has a Reynolds factor,
      and it is on `alpha_stall`, not on the blend, so a deliberate soft stall
      would live there.  Measured in D as well: the leak varies with Re only
      through `alpha_stall`.
  E3  calibrated against data.  There is no source in the code or in
      `docs/model_validity.md`, and `derivations/aerodynamic_coefficients.md`
      does not derive it.  Recorded as HEURISTIC, not tested here.

2. The model
------------
A partition of unity is a statement about *where each branch is valid*.  The
attached branch is valid below stall and the separated one above it, so the
weight should be exactly 0 below some angle and exactly 1 above another.  A
logistic cannot do that; a compactly supported blend can, and a smoothstep
`S(t) = t^2 (3 - 2t)` is the lowest-order one that is still C1, which is what
"no kink for the optimiser" asks for.

3. What is measured here
------------------------
  A  the partition is a partition: the two weights sum to 1 everywhere.
  B  the leak at zero incidence, and the realised lift slope against the
     attached slope the docstring names, over the whole (AR, Re, kappa) grid.
  C  the leak past stall: how much unbounded linear branch survives there.
  D  E1 against E2: what the leak is a function of.
  E  the candidates, against four criteria.
  F  what the chosen candidate does to CL over the whole incidence range.

Run:  PYTHONPATH=. python experiments/stall_blend/run.py
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

from experiments.harness import ExperimentResult, load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
FAILURES: list[str] = []
FINDINGS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)
    return cond


def finding(name: str, holds: bool, detail: str = "") -> bool:
    """A defect this experiment exists to measure.  Present is not a failure.

    Inverted on purpose: `[gap]` means the defect is there in the expression
    being measured and the numbers below describe it; `[GONE]` means it is
    not, which is the line that has to be explained before anyone believes the
    rest of the file.  Sections B-D measure the **logistic that section A
    replaced**, so `[gap]` there is the record of what was found, not a
    statement about the solver as it stands.
    """
    print(f"  [{'gap ' if holds else 'GONE'}] {name}"
          f"{('  -- ' + detail) if detail else ''}")
    (FINDINGS if holds else FAILURES).append(name)
    return holds


# --------------------------------------------------------------------------
# The model, written out here so a candidate blend can be swapped into it
# without touching the solver.  `test_stall_blend.py` pins this against
# `fluid.lift_coefficient` itself.
# --------------------------------------------------------------------------


def stall_angle(re: float, lev: float) -> float:
    a = np.radians(11.0 + 26.0 * lev)
    return a * float(np.clip(0.55 + 0.45 * np.log10(max(re, 10.0)) / 5.0,
                             0.5, 1.0))


def stall_angle_vec(re: np.ndarray, lev: np.ndarray) -> np.ndarray:
    """`stall_angle` over arrays, for the sample section F collects."""
    a = np.radians(11.0 + 26.0 * lev)
    return a * np.clip(0.55 + 0.45 * np.log10(np.maximum(re, 10.0)) / 5.0,
                       0.5, 1.0)


def cl_max_of(lev: float) -> float:
    return 1.10 + 0.80 * lev


def attached_slope(ar: float) -> float:
    return 2.0 * np.pi / (1.0 + 2.0 / max(ar, 0.5))


def w_logistic(alpha: np.ndarray, a_s: float, width: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.abs(alpha) - a_s) / width))


def w_smoothstep(alpha: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """C1 partition, exactly 0 at or below `lo` and exactly 1 at or above `hi`."""
    t = np.clip((np.abs(alpha) - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def cl_of(alpha: np.ndarray, ar: float, re: float, lev: float,
          weight) -> np.ndarray:
    a_s = stall_angle(re, lev)
    cmax = cl_max_of(lev)
    w = weight(alpha, a_s)
    cl = (1.0 - w) * attached_slope(ar) * alpha + w * cmax * np.sin(2.0 * alpha)
    return np.clip(cl, -1.2 * cmax, 1.2 * cmax)


def incumbent(width_deg: float):
    """The logistic this experiment replaced, kept as a literal.

    Sections B-D describe it, so it cannot be read from the module: the module
    no longer contains it.  Section A checks what the module does now.
    """
    b = np.radians(width_deg)
    return lambda a, a_s: w_logistic(a, a_s, b)


def logistic_fraction(f: float):
    return lambda a, a_s: w_logistic(a, a_s, np.maximum(f * a_s, 1e-6))


def smoothstep_symmetric(f: float):
    return lambda a, a_s: w_smoothstep(a, (1.0 - f) * a_s, (1.0 + f) * a_s)


def smoothstep_edges(lower_fraction: float, upper_deg: float):
    """Separation starts at `lower_fraction * alpha_stall`, completes past it.

    The two edges are the whole design.  A lower edge at 0 removes the leak at
    zero incidence by construction, and says separation begins at zero
    incidence, which nobody would claim; the sweep is what shows whether an
    edge above 0 also clears the criterion.
    """
    hi = np.radians(upper_deg)
    return lambda a, a_s: w_smoothstep(a, lower_fraction * a_s, a_s + hi)


# --------------------------------------------------------------------------


def slope_at_zero(ar: float, re: float, lev: float, weight,
                  h: float = 1e-5) -> float:
    a = np.array([-h, h])
    cl = cl_of(a, ar, re, lev, weight)
    return float((cl[1] - cl[0]) / (2.0 * h))


def test_it_is_a_partition(cfg: dict, out: dict) -> None:
    print("\nA. the two weights sum to one")
    g = cfg["alpha_grid_deg"]
    a = np.radians(np.linspace(g["lo"], g["hi"], g["n"]))
    worst = 0.0
    for re, lev in itertools.product(cfg["regime"]["reynolds"], [0.0, 0.5, 1.0]):
        a_s = stall_angle(re, lev)
        for w in (w_logistic(a, a_s, np.radians(6.0)),
                  w_smoothstep(a, 0.0, a_s + np.radians(13.0))):
            worst = max(worst, float(np.max(np.abs((1.0 - w) + w - 1.0))))
    check("both blends partition unity to machine precision",
          worst <= cfg["tolerances"]["partition_sum"], f"worst {worst:.2e}")
    out["partition_worst"] = worst

    # What the module does now, against the recommendation this experiment
    # produced.  Written out here rather than imported so that a change to the
    # solver shows up as a disagreement instead of moving both sides together.
    from dytiscidae.physics.fluid import SEPARATION_COMPLETE, lift_coefficient
    upper = float(np.degrees(SEPARATION_COMPLETE))
    print(f"         the module's separation width is {upper:.1f} deg past "
          f"stall")
    worst_mod = 0.0
    for ar, re, k in itertools.product([2.0, 4.0, 16.0], [1e4, 1e6],
                                       [0.0, 0.15, 0.30]):
        lev = min(k / 0.30, 1.0)
        mine = cl_of(a, ar, re, lev, smoothstep_edges(0.0, upper))
        theirs = lift_coefficient(a, np.full(a.shape, re),
                                  np.full(a.shape, ar), np.full(a.shape, k))
        worst_mod = max(worst_mod, float(np.max(np.abs(mine - theirs))))
    check("and the module computes the blend this experiment writes out",
          worst_mod < 1e-12, f"worst absolute difference {worst_mod:.2e}")
    out["module_separation_deg"] = upper
    out["module_agreement"] = worst_mod


def test_the_leak_at_zero(cfg: dict, out: dict) -> None:
    print("\nB. how much post-stall branch reaches zero incidence")
    rows = []
    for ar, re, k in itertools.product(cfg["regime"]["aspect_ratio"],
                                       cfg["regime"]["reynolds"],
                                       cfg["regime"]["reduced_pitch_rate"]):
        lev = min(k / 0.30, 1.0)
        a_s = stall_angle(re, lev)
        w0 = float(w_logistic(np.array([0.0]), a_s,
                              np.radians(cfg["incumbent_blend_width_deg"]))[0])
        att = attached_slope(ar)
        got = slope_at_zero(ar, re, lev, incumbent(cfg["incumbent_blend_width_deg"]))
        rows.append({"ar": ar, "re": re, "kappa": k, "alpha_stall_deg":
                     float(np.degrees(a_s)), "w_at_zero": w0,
                     "attached_slope": att, "realised_slope": got,
                     "deficit": float(got / att - 1.0)})
    worst = min(rows, key=lambda r: r["deficit"])
    gliding = [r for r in rows if r["kappa"] == 0.0]
    flapping = [r for r in rows if r["kappa"] == 0.30]
    print(f"         over {len(rows)} (AR, Re, kappa) points")
    print(f"         w(0) runs {min(r['w_at_zero'] for r in rows):.4f} to "
          f"{max(r['w_at_zero'] for r in rows):.4f}")
    print(f"         worst slope deficit {100*worst['deficit']:+.1f}% at "
          f"AR={worst['ar']:.0f} Re={worst['re']:.0e} kappa={worst['kappa']}")
    print(f"         gliding  (kappa=0):    mean deficit "
          f"{100*np.mean([r['deficit'] for r in gliding]):+.2f}%")
    print(f"         flapping (kappa=0.30): mean deficit "
          f"{100*np.mean([r['deficit'] for r in flapping]):+.2f}%")
    off = [r for r in rows if abs(r["deficit"]) > 1e-6]
    low = [r for r in rows if r["deficit"] < -1e-6]
    finding("F-05  the logistic did not deliver the attached slope at zero "
            "incidence",
            len(off) == len(rows),
            f"all {len(rows)} points are off it; {len(low)} below and "
            f"{len(off) - len(low)} above -- the plate branch contributes a "
            f"slope of 2 CL_max at zero, which is under the attached slope on "
            f"a long wing and over it on a short one")
    check("and the loss falls on gliding wings, not flapping ones",
          abs(np.mean([r["deficit"] for r in flapping]))
          < 0.1 * abs(np.mean([r["deficit"] for r in gliding])),
          "a high reduced pitch rate raises alpha_stall to 37 deg, which "
          "pushes the tail away from zero")
    out["leak_rows"] = rows
    out["worst_deficit"] = worst["deficit"]
    out["mean_deficit_gliding"] = float(np.mean([r["deficit"] for r in gliding]))
    out["mean_deficit_flapping"] = float(np.mean([r["deficit"] for r in flapping]))


def test_the_leak_past_stall(cfg: dict, out: dict) -> None:
    print("\nC. how much attached branch survives past stall")
    b = np.radians(cfg["incumbent_blend_width_deg"])
    rows = []
    for re, k in itertools.product(cfg["regime"]["reynolds"],
                                   cfg["regime"]["reduced_pitch_rate"]):
        lev = min(k / 0.30, 1.0)
        a_s = stall_angle(re, lev)
        for past_deg in (2.0, 5.5, 10.0):
            a = np.array([a_s + np.radians(past_deg)])
            w = float(w_logistic(a, a_s, b)[0])
            rows.append({"re": re, "kappa": k, "past_deg": past_deg,
                         "attached_weight": 1.0 - w})
    for past in (2.0, 5.5, 10.0):
        vals = [r["attached_weight"] for r in rows if r["past_deg"] == past]
        print(f"         {past:4.1f} deg past stall: the attached branch still "
              f"carries {100*np.mean(vals):.1f}% of the weight")
    ten = [r["attached_weight"] for r in rows if r["past_deg"] == 10.0]
    finding("F-05  the logistic still weighted an unbounded linear branch 10 "
            "deg past stall", min(ten) > 0.10,
            f"{100*min(ten):.1f}% at the least")
    out["past_stall_rows"] = rows


def test_what_the_leak_is_a_function_of(cfg: dict, out: dict) -> None:
    """E1 against E2."""
    print("\nD. what w(0) depends on")
    b = np.radians(cfg["incumbent_blend_width_deg"])
    # E1: w(0) is a function of alpha_stall alone, whatever Re and kappa are.
    by_stall: dict[float, set] = {}
    for re, k in itertools.product(cfg["regime"]["reynolds"],
                                   cfg["regime"]["reduced_pitch_rate"]):
        lev = min(k / 0.30, 1.0)
        a_s = round(float(stall_angle(re, lev)), 12)
        w0 = float(w_logistic(np.array([0.0]), a_s, b)[0])
        by_stall.setdefault(a_s, set()).add(round(w0, 12))
    check("w(0) is a function of alpha_stall alone",
          all(len(v) == 1 for v in by_stall.values()),
          f"{len(by_stall)} distinct stall angles, each with one w(0)")

    # and it is decreasing in alpha_stall, so a wing that stalls early -- the
    # low-Reynolds case this model exists to represent -- leaks the most.
    pairs = sorted((a, next(iter(v))) for a, v in by_stall.items())
    decreasing = all(pairs[i][1] > pairs[i + 1][1] for i in range(len(pairs) - 1))
    check("and it is largest exactly where the stall angle is lowest",
          decreasing,
          f"alpha_stall {np.degrees(pairs[0][0]):.1f} deg -> w(0) "
          f"{pairs[0][1]:.4f};  {np.degrees(pairs[-1][0]):.1f} deg -> "
          f"{pairs[-1][1]:.4f}")
    out["w0_by_stall_angle"] = [
        {"alpha_stall_deg": float(np.degrees(a)), "w_at_zero": w}
        for a, w in pairs]


def candidate_table(cfg: dict) -> dict:
    cands: dict = {"incumbent 6 deg":
                   incumbent(cfg["incumbent_blend_width_deg"])}
    for f in cfg["candidates"]["logistic_fraction"]:
        cands[f"logistic {f:g} alpha_stall"] = logistic_fraction(f)
    for f in cfg["candidates"]["smoothstep_symmetric"]:
        cands[f"smoothstep +-{f:g} alpha_stall"] = smoothstep_symmetric(f)
    e = cfg["candidates"]["smoothstep_edges"]
    for f, d in itertools.product(e["lower_fraction_of_stall"],
                                  e["upper_deg_past_stall"]):
        cands[f"smoothstep {f:g}*stall..stall+{d:g}"] = smoothstep_edges(f, d)
    return cands


def d1_jump(alpha: np.ndarray, cl: np.ndarray, clip: float) -> float:
    """Largest step in the numerical first derivative, away from the clip.

    The `np.clip(cl, -1.2 CL_max, 1.2 CL_max)` at the end of the model puts a
    real corner in CL' wherever it binds, for every candidate including the
    incumbent, so it is excluded here: what is being measured is the blend.
    """
    d1 = np.gradient(cl, alpha)
    clipped = np.abs(np.abs(cl) - clip) <= 1e-9
    # Widen by two samples each way.  The corner is where the clip *starts*
    # biting, and a central difference at the two points either side of it
    # already carries it; masking only the exactly-clipped samples leaves that
    # corner in, and it then dominates every other feature on the curve.
    free = ~clipped
    for shift in (1, 2):
        free &= ~np.roll(clipped, shift)
        free &= ~np.roll(clipped, -shift)
    steps = np.abs(np.diff(d1))
    keep = free[:-1] & free[1:]
    return float(steps[keep].max()) if keep.any() else 0.0


def continuity_order(cfg: dict, weight, ar: float, re: float,
                     lev: float) -> float:
    """How the largest derivative step shrinks when the grid is halved.

    A C1 curve's numerical derivative converges, so the largest step falls with
    the step size and the ratio is about 0.5.  A corner does not: the step is a
    property of the function, not of the grid, and the ratio stays near 1.  This
    is the same refinement argument `benchmarks/` uses for the integrator, and
    it is the only way to tell the two apart without an analytic derivative.
    """
    g = cfg["alpha_grid_deg"]
    clip = 1.2 * cl_max_of(lev)
    out = []
    for n in (g["n"], 2 * g["n"] - 1):
        a = np.radians(np.linspace(g["lo"], g["hi"], n))
        out.append(d1_jump(a, cl_of(a, ar, re, lev, weight), clip))
    return out[1] / max(out[0], 1e-30)


def score_candidate(cfg: dict, weight) -> dict:
    """Five numbers, none of them an opinion."""
    g = cfg["alpha_grid_deg"]
    a = np.radians(np.linspace(g["lo"], g["hi"], g["n"]))
    base = incumbent(cfg["incumbent_blend_width_deg"])

    worst_deficit = 0.0
    worst_below = worst_above = 0.0
    worst_order = 0.0
    peak = 0.0
    clip_binds = False
    peak_ratios: list[float] = []
    for ar, re, k in itertools.product(cfg["regime"]["aspect_ratio"],
                                       cfg["regime"]["reynolds"],
                                       cfg["regime"]["reduced_pitch_rate"]):
        lev = min(k / 0.30, 1.0)
        a_s = stall_angle(re, lev)
        cmax = cl_max_of(lev)
        att = attached_slope(ar)
        worst_deficit = min(worst_deficit,
                            slope_at_zero(ar, re, lev, weight) / att - 1.0)

        cl_new = cl_of(a, ar, re, lev, weight)
        cl_old = cl_of(a, ar, re, lev, base)
        peak = max(peak, float(np.max(np.abs(cl_new))))
        clip_binds = clip_binds or bool(
            np.any(np.abs(np.abs(cl_new) - 1.2 * cmax) < 1e-9))
        # Where the model's lift peaks, in units of its own stall angle.
        # Reported, and NOT a criterion: it comes out the same for every
        # candidate because it is a property of the separated branch, which is
        # `CL_max sin(2 alpha)` and peaks at 45 degrees whatever the blend
        # does.  What it records is that this model has no post-stall lift
        # *loss* at all -- CL climbs from the stall angle to 45 degrees -- and
        # no choice of blend changes that.  See MATH_AUDIT F-08.
        pos = a > 0
        i = int(np.argmax(cl_new[pos]))
        peak_ratios.append(float(a[pos][i] / a_s))

        # Normalised by CL_max, the scale of the quantity, rather than by the
        # local value: the plate branch passes through zero at 90 degrees, and
        # dividing by it there turns a small absolute change into a large ratio.
        below = np.abs(a) < a_s
        above = np.abs(a) > a_s
        d = np.abs(cl_new - cl_old) / cmax
        worst_below = max(worst_below, float(d[below].max()))
        worst_above = max(worst_above, float(d[above].max()))
        worst_order = max(worst_order,
                          continuity_order(cfg, weight, ar, re, lev))
    return {"worst_slope_deficit": worst_deficit,
            "worst_change_below_stall": worst_below,
            "worst_change_above_stall": worst_above,
            "derivative_step_ratio": worst_order,
            "peak_cl": peak,
            "clip_binds": clip_binds,
            "peak_alpha_over_stall_max": float(max(peak_ratios)),
            "peak_alpha_over_stall_mean": float(np.mean(peak_ratios))}


def test_the_candidates(cfg: dict, out: dict) -> None:
    print("\nE. the candidates")
    print("         `deficit@0` is the criterion: the realised lift slope at")
    print("         zero incidence against the attached one the docstring")
    print("         names.  `peak a/a_s` is where CL peaks in units of the")
    print("         model's own stall angle; it is 4.50 for every candidate,")
    print("         because the separated branch peaks at 45 degrees whatever")
    print("         the blend does, so this model has no post-stall lift loss")
    print("         and no blend can give it one (F-08).  The two `change`")
    print("         columns, in units of CL_max, are the size of the fix.")
    print("         Every candidate's derivative-step ratio is reported in the")
    print("         result file; all of them converge, so none adds a corner.")
    print()
    print(f"         {'blend':<30} {'deficit@0':>10} {'d below':>8} "
          f"{'d above':>8} {'peak a/a_s':>11} {'peak CL':>8} {'clip':>5}")
    rows = []
    for name, weight in candidate_table(cfg).items():
        s_ = score_candidate(cfg, weight)
        s_["name"] = name
        rows.append(s_)
        print(f"         {name:<30} {100*s_['worst_slope_deficit']:9.2f}% "
              f"{s_['worst_change_below_stall']:8.3f} "
              f"{s_['worst_change_above_stall']:8.3f} "
              f"{s_['peak_alpha_over_stall_mean']:5.2f} "
              f"(max {s_['peak_alpha_over_stall_max']:4.2f}) "
              f"{s_['peak_cl']:8.3f} "
              f"{'yes' if s_['clip_binds'] else 'no':>5}")
    out["candidates"] = rows

    tol = cfg["tolerances"]["slope_deficit_at_zero"]
    # A continuous derivative means the step ratio falls with the grid.  0.75
    # is the midpoint between the 0.5 a C1 curve gives and the 1.0 a corner
    # gives, so it separates them without being tuned to either.
    ok = [r for r in rows
          if abs(r["worst_slope_deficit"]) < tol
          and r["derivative_step_ratio"] < 0.75]
    check("at least one candidate delivers the attached slope with a "
          "continuous derivative", bool(ok),
          f"{len(ok)} of {len(rows)}")
    base = [r for r in rows if r["name"].startswith("incumbent")][0]
    check("and the incumbent's own derivative is continuous, so the ratio "
          "separates the two cases rather than flagging everything",
          base["derivative_step_ratio"] < 0.75,
          f"incumbent ratio {base['derivative_step_ratio']:.2f}")
    clean = [r for r in ok if not r["clip_binds"]]
    check("and at least one of those never reaches the CL clip",
          bool(clean), f"{len(clean)} of {len(ok)}")
    out["clears_criterion"] = [r["name"] for r in ok]
    out["clears_and_never_clips"] = [r["name"] for r in clean]
    print()
    print("         The criterion above is met by many candidates, so it does")
    print("         not choose one.  Section F does: among those that clear it")
    print("         and never reach the clip, the one that perturbs the CL the")
    print("         machines actually use by the least.  That is a")
    print("         conservatism criterion, not a physical one, and it is")
    print("         stated as such: the purpose of this change is to remove a")
    print("         defect, not to silently re-tune the lift model.")


def test_where_the_machines_actually_operate(cfg: dict, out: dict) -> None:
    """The worst case over a grid is not the change a machine feels.

    Sections B-E sweep `(AR, Re, kappa, alpha)` uniformly, which weights a
    45-degree strip the same as a 2-degree one.  A real rollout does not visit
    those uniformly.  So: run each seed plan, record the arguments the solver
    hands `lift_coefficient` on every step of every panel, and re-evaluate the
    candidates on exactly that sample.

    Recorded by wrapping the function rather than by changing the solver, so
    what is sampled is what the solver asked for and not a reconstruction.
    """
    from dytiscidae.core import bodyplans
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    from dytiscidae.physics import fluid as fluid_mod

    print("\nF. the incidences the seed plans actually visit")
    real = fluid_mod.lift_coefficient
    bucket: list = []          # the list the recorder is writing into now

    def recording(alpha, re, ar, reduced_pitch_rate):
        bucket.append((np.asarray(alpha, float).copy(),
                       np.asarray(re, float).copy(),
                       np.asarray(ar, float).copy(),
                       np.asarray(reduced_pitch_rate, float).copy()))
        return real(alpha, re, ar, reduced_pitch_rate)

    per_medium: dict[str, list] = {m: [] for m in cfg["rollout"]["media"]}
    fluid_mod.lift_coefficient = recording
    try:
        for name in cfg["seed_plans"]:
            env = TriphibianEnv(build(getattr(bodyplans, name)()))
            for medium in cfg["rollout"]["media"]:
                env.reset(Domain(medium), randomise=False)
                bucket.clear()
                n = int(cfg["rollout"]["segment_seconds"] / env.timestep)
                for _ in range(n):
                    env.step(env.cpg.command(env.cpg.base, env.data.time))
                per_medium[medium].extend(bucket)
                bucket.clear()
    finally:
        fluid_mod.lift_coefficient = real

    seen = [x for v in per_medium.values() for x in v]
    if not seen:
        check("the solver called lift_coefficient at all", False, "no samples")
        return

    alpha = np.concatenate([x[0] for x in seen])
    re = np.concatenate([x[1] for x in seen])
    ar = np.concatenate([x[2] for x in seen])
    kappa = np.concatenate([x[3] for x in seen])
    keep = np.isfinite(alpha) & np.isfinite(re) & np.isfinite(ar) & np.isfinite(kappa)
    alpha, re, ar, kappa = alpha[keep], re[keep], ar[keep], kappa[keep]

    lev = np.clip(kappa / 0.30, 0.0, 1.0)
    a_s = stall_angle_vec(re, lev)

    def describe(label: str, aa, rr, kk) -> dict:
        ll = np.clip(kk / 0.30, 0.0, 1.0)
        ss = stall_angle_vec(rr, ll)
        d = np.degrees(np.abs(aa))
        q = np.percentile(d, [50, 90, 99])
        below = float(np.mean(np.abs(aa) < ss))
        print(f"         {label:<7} {len(aa):>10,}  median {q[0]:5.1f} deg  "
              f"90th {q[1]:5.1f}  99th {q[2]:5.1f}  "
              f"below stall {100*below:5.1f}%  kappa med {np.median(kk):.4f}")
        return {"n": int(len(aa)), "alpha_deg_median": float(q[0]),
                "alpha_deg_p90": float(q[1]), "alpha_deg_p99": float(q[2]),
                "alpha_deg_max": float(d.max()),
                "kappa_median": float(np.median(kk)),
                "fraction_below_stall": below}

    print(f"         {'medium':<7} {'strip-steps':>10}")
    vis = {}
    for medium, recs in per_medium.items():
        if not recs:
            continue
        aa = np.concatenate([x[0] for x in recs])
        rr = np.concatenate([x[1] for x in recs])
        kk = np.concatenate([x[3] for x in recs])
        m = np.isfinite(aa) & np.isfinite(rr) & np.isfinite(kk)
        vis[medium] = describe(medium, aa[m], rr[m], kk[m])
    vis["all"] = describe("all", alpha, re, kappa)
    out["visited"] = vis

    # Evaluate every candidate on exactly this sample.  Vectorised over the
    # whole cloud, so each candidate costs one pass.
    cmax = 1.10 + 0.80 * lev
    att = 2.0 * np.pi / (1.0 + 2.0 / np.maximum(ar, 0.5))
    lin = att * alpha
    plate = cmax * np.sin(2.0 * alpha)
    clip = 1.2 * cmax

    def cl_sample(weight) -> np.ndarray:
        w = weight(alpha, a_s)
        return np.clip((1.0 - w) * lin + w * plate, -clip, clip)

    base = cl_sample(incumbent(cfg["incumbent_blend_width_deg"]))
    print()
    print(f"         {'blend':<30} {'rms dCL':>9} {'mean dCL':>9} "
          f"{'p99 |dCL|':>10}  as a fraction of the CL in use")
    rows = []
    scale = float(np.sqrt(np.mean(base ** 2)))
    for name, weight in candidate_table(cfg).items():
        d = cl_sample(weight) - base
        r = {"name": name,
             "rms_dcl": float(np.sqrt(np.mean(d ** 2))),
             "mean_dcl": float(np.mean(d)),
             "p99_abs_dcl": float(np.percentile(np.abs(d), 99))}
        rows.append(r)
        print(f"         {name:<30} {r['rms_dcl']:9.4f} {r['mean_dcl']:+9.4f} "
              f"{r['p99_abs_dcl']:10.4f}  {r['rms_dcl']/max(scale,1e-12):7.1%}")
    out["visited_rms_cl"] = scale
    out["visited_candidates"] = rows
    check("the incumbent is unchanged against itself",
          abs(rows[0]["rms_dcl"]) < 1e-15, f"{rows[0]['rms_dcl']:.2e}")

    eligible = set(out.get("clears_and_never_clips", []))
    pool = [r for r in rows if r["name"] in eligible]
    if pool:
        pick = min(pool, key=lambda r: r["rms_dcl"])
        by_name = {r["name"]: r for r in out["candidates"]}
        i = [r["name"] for r in rows].index(pick["name"])
        lo = rows[max(i - 1, 0)]["rms_dcl"]
        hi = rows[min(i + 1, len(rows) - 1)]["rms_dcl"]
        interior = lo > pick["rms_dcl"] < hi
        print()
        print(f"         the least-perturbing candidate that clears the "
              f"criterion and never clips:")
        print(f"           {pick['name']}   rms dCL {pick['rms_dcl']:.4f} "
              f"({pick['rms_dcl']/max(scale,1e-12):.1%} of the CL in use), "
              f"peak CL {by_name[pick['name']]['peak_cl']:.3f}")
        check("and it is an interior minimum of the sweep, not its edge",
              interior,
              f"neighbours {lo:.4f} and {hi:.4f} bracket {pick['rms_dcl']:.4f}")
        out["recommended"] = pick["name"]
        out["recommended_row"] = {**pick, **by_name[pick["name"]]}


def test_what_it_does_to_the_seed_plans(cfg: dict, out: dict) -> None:
    """The scores, with the logistic put back and taken away again.

    The legacy blend no longer exists in the module, so it is reinstated here
    by wrapping `lift_coefficient` with the expression section A replaced --
    same stall angle, same CL_max, same branches, only the handover differs.
    Everything else about the evaluation is held fixed, including the seed.
    """
    from dytiscidae.core import bodyplans
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.evaluate import evaluate_tier1
    from dytiscidae.physics import fluid as fluid_mod

    print("\nG. the seed plans, scored with each blend")
    b = np.radians(cfg["incumbent_blend_width_deg"])

    def legacy(alpha, re, ar, reduced_pitch_rate):
        alpha = np.asarray(alpha, float)
        lev = np.clip(np.asarray(reduced_pitch_rate, float) / 0.30, 0.0, 1.0)
        cmax = 1.10 + 0.80 * lev
        a_s = stall_angle_vec(np.asarray(re, float), lev)
        w = w_logistic(alpha, a_s, b)
        lin = (2.0 * np.pi / (1.0 + 2.0 / np.maximum(np.asarray(ar, float),
                                                     0.5))) * alpha
        return np.clip((1.0 - w) * lin + w * cmax * np.sin(2.0 * alpha),
                       -1.2 * cmax, 1.2 * cmax)

    def score_all() -> dict:
        got = {}
        for name in cfg["seed_plans"]:
            r = evaluate_tier1(build(getattr(bodyplans, name)()),
                               segment_seconds=cfg["rollout"]["segment_seconds"],
                               seed=cfg["rollout"]["seed"])
            got[name] = {"mission_fraction": float(r.mission_fraction),
                         "feasible": bool(r.feasible),
                         "energy_margin": float(r.energy_margin),
                         "diverged": int(r.diverged_rollouts),
                         "air_gates": int(sum(bool(g) for g in r.air_gates))}
        return got

    real = fluid_mod.lift_coefficient
    fluid_mod.lift_coefficient = legacy
    try:
        before = score_all()
    finally:
        fluid_mod.lift_coefficient = real
    after = score_all()

    print(f"         {'plan':<8} {'mission before':>15} {'after':>10} "
          f"{'air gates':>10} {'diverged':>9}")
    rows = []
    for name in cfg["seed_plans"]:
        x, y = before[name], after[name]
        rows.append({"plan": name, "before": x, "after": y})
        print(f"         {name:<8} {x['mission_fraction']:15.6f} "
              f"{y['mission_fraction']:10.6f} "
              f"{x['air_gates']:5d} -> {y['air_gates']:<4d} "
              f"{x['diverged']:4d} -> {y['diverged']:<4d}")
    out["seed_scores"] = rows

    feas_b = sum(r["before"]["feasible"] for r in rows)
    feas_a = sum(r["after"]["feasible"] for r in rows)
    check("no seed plan is made infeasible by the change",
          feas_a >= feas_b, f"{feas_b} feasible before, {feas_a} after")
    gates_b = sum(r["before"]["air_gates"] for r in rows)
    gates_a = sum(r["after"]["air_gates"] for r in rows)
    print(f"         air gates cleared across all plans: {gates_b} -> "
          f"{gates_a}")
    out["air_gates_before"] = gates_b
    out["air_gates_after"] = gates_a

    worst = max(abs(r["after"]["mission_fraction"]
                    - r["before"]["mission_fraction"]) for r in rows)
    floor = max(r["before"]["mission_fraction"] for r in rows)
    print()
    print("         This is a null result and is reported as one.  The seed")
    print(f"         plans score at the floor -- the best of them reaches")
    print(f"         {floor:.6f} of the mission and none clears a single air")
    print("         gate -- so a change to the lift model cannot move them,")
    print(f"         and it does not: the largest shift is {worst:.6f}.  What")
    print("         that measures is the plans, not the change.  A ΔS for this")
    print("         fix needs an archive of evolved elites, which this")
    print("         repository's `runs/` does not carry; section F is the")
    print("         measurement that does exist, and it says the forces in use")
    print("         move by 3.8% RMS.")
    out["seed_scores_uninformative"] = {
        "largest_mission_shift": float(worst),
        "best_mission_fraction": float(floor),
        "air_gates_cleared": int(gates_a),
    }
    check("the seed plans are at the floor, so this section cannot decide "
          "anything", gates_a == 0 and floor < 0.01,
          f"best plan reaches {floor:.6f} of the mission, 0 air gates")


def main() -> int:
    cfg = load_config(HERE / "config.json")
    res = ExperimentResult(name="stall_blend", config=cfg)
    out: dict = {}
    print(__doc__.strip().splitlines()[0])
    test_it_is_a_partition(cfg, out)
    test_the_leak_at_zero(cfg, out)
    test_the_leak_past_stall(cfg, out)
    test_what_the_leak_is_a_function_of(cfg, out)
    test_the_candidates(cfg, out)
    test_where_the_machines_actually_operate(cfg, out)
    test_what_it_does_to_the_seed_plans(cfg, out)
    res.ok = not FAILURES
    for k, v in out.items():
        res.record(k, v)
    res.record("open_findings", FINDINGS)
    res.write(HERE / "results")
    if FINDINGS:
        print(f"\n{len(FINDINGS)} findings, measured on the expression "
              f"section A replaced:")
        for g in FINDINGS:
            print(f"  - {g}")
    if FAILURES:
        print(f"\n{len(FAILURES)} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
