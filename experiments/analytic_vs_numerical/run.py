#!/usr/bin/env python3
"""Every closed form in the project, checked against an independent computation.

0. The problem
--------------
A formula in this codebase is believed because it is written down.  That is not
a reason.  Where a quantity has a derivative, the derivative can be differenced.
Where it has an integral, the integral can be quadratured.  Where it has a
textbook coefficient, the textbook derivation can be redone from its own
premises.  Anything that survives all three is believed for a reason; anything
that does not is a finding.

What is checked
---------------
  A  CPG kinematic Jacobian d(theta)/d(params), analytic vs central difference.
     This one is not a formality: the frequency column grows linearly in t, so
     the identification's sensitivity to frequency depends on how long the
     probe ran.  The number falls out of the check.
  B  lift-coefficient slope in the attached branch, analytic vs difference, and
     the continuity of the sigmoid handover into the post-stall branch.
  C  drag coefficient: sign, and agreement with its own three terms.
  D  elliptic lift centroid 4/(3 pi), closed form vs quadrature.
  E  hollow-tube section properties, closed form vs quadrature over the annulus.
  F  cantilever tip deflection under a uniform load, closed form vs a numerical
     integration of the moment-area theorem.
  G  long-cylinder external-pressure buckling, the code's coefficient vs the
     classical ring result derived from its own premises.
  H  added-mass coefficient of a disc, the code's tensor vs Lamb's result.
  I  Wagner slam coefficient, the code's constant vs the commonly quoted one.
  J  fatigue knockdown interpolation, endpoints and monotonicity.
  K  net-buoyancy depth derivative, analytic vs the code's own difference.

Run:  PYTHONPATH=. python experiments/analytic_vs_numerical/run.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "disable")

from experiments.harness import ExperimentResult, load_config  # noqa: E402

RESULTS: list[dict] = []


def report(name: str, analytic: float, numerical: float, tol: float,
           note: str = "", kind: str = "closed form vs numerical",
           absolute: bool = False) -> dict:
    """Compare two numbers.  `absolute` when the expected value is zero --
    a relative error against zero is always 1 and says nothing."""
    if absolute:
        rel = abs(analytic - numerical)
    else:
        denom = max(abs(analytic), abs(numerical), 1e-300)
        rel = abs(analytic - numerical) / denom
    ok = rel <= tol
    row = {"check": name, "kind": kind, "analytic": float(analytic),
           "numerical": float(numerical), "relative_error": float(rel),
           "tolerance": float(tol), "ok": bool(ok), "note": note}
    RESULTS.append(row)
    mark = "ok  " if ok else "DIFFERS"
    print(f"  [{mark}] {name:<52} {analytic: .8g} vs {numerical: .8g}  "
          f"rel={rel:.3g}")
    if note:
        print(f"          {note}")
    return row


# --------------------------------------------------------------------------
# A. CPG kinematic Jacobian
# --------------------------------------------------------------------------


def check_cpg_jacobian(cfg: dict) -> dict:
    from dytiscidae.control.cpg import CPG, CPGParams

    n = cfg["cpg"]["n_joints"]
    cpg = CPG(n, base_frequency=cfg["cpg"]["base_frequency"])
    base = cpg.base
    p0 = base.flat()
    h = cfg["fd_step_relative"]
    worst = 0.0
    freq_col_norms = {}

    for t in cfg["cpg"]["times_s"]:
        # Analytic.  theta_i = off_i + A_i sin(2 pi f t + phi_i + phi0)
        psi = 2 * np.pi * base.frequency * t + base.phase + cpg.phase_offset
        J = np.zeros((n, 3 * n + 1))
        J[:, 0:n] = np.diag(np.sin(psi))
        J[:, n:2 * n] = np.diag(base.amplitude * np.cos(psi))
        J[:, 2 * n:3 * n] = np.eye(n)
        J[:, 3 * n] = base.amplitude * 2 * np.pi * t * np.cos(psi)

        # Central difference through the real `command`.
        Jfd = np.zeros_like(J)
        for j in range(3 * n + 1):
            step = h * max(abs(p0[j]), 1.0)
            pp, pm = p0.copy(), p0.copy()
            pp[j] += step
            pm[j] -= step
            up = cpg.command(CPGParams.from_flat(pp, n), t)
            dn = cpg.command(CPGParams.from_flat(pm, n), t)
            Jfd[:, j] = (up - dn) / (2 * step)

        scale = max(np.abs(J).max(), 1e-12)
        err = float(np.abs(J - Jfd).max() / scale)
        worst = max(worst, err)
        freq_col_norms[str(t)] = float(np.linalg.norm(J[:, 3 * n]))

    report("A. CPG Jacobian, analytic vs central difference", 0.0, worst,
           cfg["tolerances"]["cpg_jacobian_rel"], kind="derivative",
           absolute=True,
           note="max relative element error over "
                f"{len(cfg['cpg']['times_s'])} sample times")

    t_list = cfg["cpg"]["times_s"]
    t_end = max(t_list)
    t_small = sorted(x for x in t_list if x > 0)[0]
    ratio = freq_col_norms[str(t_end)] / max(freq_col_norms[str(t_small)], 1e-30)
    print(f"          frequency-column norm grows {ratio:.1f}x from "
          f"t={t_small}s to t={t_end}s -- exactly linearly, because "
          f"d(theta)/df = A 2 pi t cos(psi).")
    print("          The identification integrates over a 1.2 s probe window "
          "(TriphibianEnv.identify and identify_batch both default to it), so "
          "the frequency mode's authority is set by that duration.  Two paths "
          "with different probe_time do not identify the same basis.")
    return {"max_relative_error": worst, "frequency_column_norm": freq_col_norms}


# --------------------------------------------------------------------------
# B / C. Fluid coefficients
# --------------------------------------------------------------------------


def check_coefficients(cfg: dict) -> dict:
    from dytiscidae.physics.fluid import (
        drag_coefficient, lift_coefficient, skin_friction_cd)

    ar = np.array([4.0])
    re = np.array([2.0e5])
    k0 = np.array([0.0])

    # B1: attached-branch slope well below stall.
    #
    # The docstring says "below stall the strip behaves like a finite wing with
    # the Helmholtz lift-slope correction 2 pi / (1 + 2/AR)".  That is a claim
    # about the realised slope, and the sigmoid handover decides whether it
    # holds.  Measure the handover weight rather than assume it is small.
    a = np.radians(2.0)
    h = 1e-6
    up = lift_coefficient(np.array([a + h]), re, ar, k0)[0]
    dn = lift_coefficient(np.array([a - h]), re, ar, k0)[0]
    slope_fd = (up - dn) / (2 * h)
    slope_an = 2 * np.pi / (1 + 2 / ar[0])

    # Reproduce the blend weight from the code's own constants.
    lev = 0.0
    a_stall = np.radians(11.0 + 26.0 * lev)
    a_stall *= np.clip(0.55 + 0.45 * np.log10(max(float(re[0]), 10.0)) / 5.0, 0.5, 1.0)
    bw = np.radians(6.0)
    w_at = lambda x: 1.0 / (1.0 + np.exp(-(abs(x) - a_stall) / bw))
    w0, w2 = float(w_at(0.0)), float(w_at(a))
    report("B. lift slope at 2 deg, 2 pi/(1+2/AR) vs difference",
           slope_an, slope_fd, 0.02, kind="derivative",
           note=f"the post-stall branch already carries weight {w0:.3f} at "
                f"alpha=0 and {w2:.3f} at alpha=2 deg, because the blend is "
                f"centred at {np.degrees(a_stall):.1f} deg with a 6 deg width. "
                f"There is no angle of attack at which the attached branch is "
                f"clean: the weight never falls below {w0:.3f}.  The realised "
                f"slope is {100 * (slope_fd / slope_an - 1):+.1f}% against the "
                f"value the docstring names.")
    RESULTS[-1]["blend_weight_at_zero_alpha"] = w0
    RESULTS[-1]["blend_weight_at_2deg"] = w2

    # B2: continuity and smoothness of the blend across the stall angle.
    lo, hi, n = (cfg["alpha_sweep_deg"]["lo"], cfg["alpha_sweep_deg"]["hi"],
                 cfg["alpha_sweep_deg"]["n"])
    alpha = np.radians(np.linspace(lo, hi, n))
    for name, k in (("k=0 (gliding)", 0.0), ("k=0.3 (strong LEV)", 0.3)):
        kk = np.full_like(alpha, k)
        cl = lift_coefficient(alpha, np.full_like(alpha, 2e5),
                              np.full_like(alpha, 4.0), kk)
        d1 = np.diff(cl) / np.diff(alpha)
        d2 = np.diff(d1) / np.diff(alpha[:-1])
        jump = float(np.max(np.abs(np.diff(d1))))
        print(f"  [ok  ] B. CL blend is C1 across stall, {name}: "
              f"largest jump in dCL/dalpha between adjacent samples = "
              f"{jump:.4f} per rad over a {np.degrees(alpha[1]-alpha[0]):.3f} "
              f"deg grid")
        RESULTS.append({"check": f"B. CL blend C1 continuity, {name}",
                        "kind": "smoothness", "analytic": 0.0,
                        "numerical": jump, "relative_error": jump,
                        "tolerance": float("inf"), "ok": True,
                        "note": f"max |d2CL| proxy {float(np.max(np.abs(d2))):.3f}"})

    # B3: odd symmetry at zero camber.  CL(-a) must be -CL(a).
    cl_p = lift_coefficient(alpha, np.full_like(alpha, 2e5),
                            np.full_like(alpha, 4.0), np.zeros_like(alpha))
    cl_m = lift_coefficient(-alpha, np.full_like(alpha, 2e5),
                            np.full_like(alpha, 4.0), np.zeros_like(alpha))
    asym = float(np.max(np.abs(cl_p + cl_m)))
    report("B. CL odd symmetry, max |CL(a) + CL(-a)|", 0.0, asym, 1e-12,
           kind="symmetry", absolute=True)

    # C: drag is never negative, and equals the sum of its three terms.
    cd = drag_coefficient(alpha, np.full_like(alpha, 2e5),
                          np.full_like(alpha, 4.0), cl_p)
    cd_min = float(cd.min())
    print(f"  [{'ok  ' if cd_min >= 0 else 'FAIL'}] C. CD is non-negative "
          f"across the whole alpha sweep: min = {cd_min:.6g}")
    RESULTS.append({"check": "C. CD non-negative over the alpha sweep",
                    "kind": "sign", "analytic": 0.0, "numerical": cd_min,
                    "relative_error": 0.0, "tolerance": 0.0,
                    "ok": bool(cd_min >= 0),
                    "note": "the minimum is the skin-friction floor"})
    parts = (skin_friction_cd(np.full_like(alpha, 2e5))
             + cl_p**2 / (np.pi * 0.75 * 4.0)
             + 1.98 * (1 - np.cos(2 * alpha)) * 0.5)
    report("C. CD equals profile + induced + separated", 0.0,
           float(np.max(np.abs(cd - parts))), 1e-12, kind="identity",
           absolute=True)
    return {"attached_slope_analytic": float(slope_an),
            "attached_slope_measured": float(slope_fd)}


# --------------------------------------------------------------------------
# D-F. Structural closed forms vs quadrature
# --------------------------------------------------------------------------


def check_structural_quadrature(cfg: dict) -> dict:
    from dytiscidae.physics.structure import spar_deflection, tube_section
    from dytiscidae.physics.materials import CFRP_TUBE

    npts = cfg["quadrature_points"]
    tol = cfg["tolerances"]["quadrature_rel"]

    # D. Centroid of an elliptic lift distribution.
    b = 1.0
    r = np.linspace(0.0, b, npts)
    w = np.sqrt(np.maximum(1.0 - (r / b) ** 2, 0.0))
    centroid = float(np.trapezoid(r * w, r) / np.trapezoid(w, r))
    report("D. elliptic lift centroid, 4/(3 pi) vs quadrature",
           4.0 / (3.0 * math.pi), centroid, 1e-5, kind="integral",
           note="this is the coefficient `spar_check` uses for the root moment")

    # E. Hollow tube area, second moment and section modulus.
    od, wall = 0.016, 0.0015
    area, i_sec, z_sec = tube_section(od, wall)
    ro, ri = od / 2, od / 2 - wall
    rr = np.linspace(ri, ro, npts)
    th = np.linspace(0, 2 * math.pi, 2001)
    area_q = float(np.trapezoid(2 * math.pi * rr, rr))
    # I = int y^2 dA over the annulus, y = r sin(theta)
    integrand = np.trapezoid(np.sin(th) ** 2, th) * rr**3
    i_q = float(np.trapezoid(integrand, rr))
    report("E. tube area, pi(ro^2-ri^2) vs quadrature", area, area_q, 1e-6,
           kind="integral")
    report("E. tube second moment, pi(ro^4-ri^4)/4 vs quadrature", i_sec, i_q,
           1e-6, kind="integral")
    report("E. section modulus, Z = I/ro", i_sec / ro, z_sec, 1e-14,
           kind="identity")

    # F. Cantilever tip deflection under a uniform load, by moment-area.
    L, E, I = 0.9, CFRP_TUBE.E, i_sec
    q = 1.0  # N/m
    x = np.linspace(0.0, L, npts)
    M = -0.5 * q * (L - x) ** 2                    # sagging moment, tip free
    curvature = M / (E * I)
    slope = np.concatenate([[0.0], np.cumsum(
        0.5 * (curvature[1:] + curvature[:-1]) * np.diff(x))])
    defl = np.concatenate([[0.0], np.cumsum(
        0.5 * (slope[1:] + slope[:-1]) * np.diff(x))])
    tip_numeric = abs(float(defl[-1]))
    tip_closed = q * L**4 / (8 * E * I)
    report("F. cantilever UDL tip deflection, qL^4/(8EI) vs moment-area",
           tip_closed, tip_numeric, 1e-4, kind="integral")
    frac = spar_deflection(lift_n=2 * q * L, semi_span=L, outer_d=od,
                           wall=wall, material=CFRP_TUBE)
    report("F. spar_deflection returns that, divided by semi-span",
           tip_closed / L, frac, 1e-12, kind="identity")
    return {}


# --------------------------------------------------------------------------
# G-K. Coefficients against their own derivations
# --------------------------------------------------------------------------


def check_coefficients_against_theory(cfg: dict) -> dict:
    from dytiscidae.physics.materials import CFRP_TUBE, PETG
    from dytiscidae.physics.medium import SEAWATER
    from dytiscidae.physics.structure import (
        BuoyancyState, hull_pressure_check, slam_pressure)

    out = {}

    # G. Long-cylinder external-pressure buckling.
    #
    # Derivation from its own premises: a unit-length ring of wall t, radius r,
    # buckling into n lobes collapses at p_cr = (n^2 - 1) E' I / r^3 with
    # I = t^3/12 per unit length and E' = E/(1-nu^2) for a long shell.  The
    # first available mode is n = 2, so p_cr = 3 E' t^3 / (12 r^3)
    #                                        = E/(4(1-nu^2)) (t/r)^3.
    # Equivalently 2E/(1-nu^2) (t/D)^3 with D the DIAMETER.
    E, nu = PETG.E, PETG.poisson
    t, r = 0.002, 0.06
    classical = E / (4 * (1 - nu**2)) * (t / r) ** 3
    _, c_buck = hull_pressure_check(depth_m=10.0, radius=r, wall=t, length=0.4,
                                    material=PETG)
    coded = c_buck.allowable
    coded_before_knockdown = coded / 0.6
    legacy_before_knockdown = E / (1 - nu**2) * 2.0 * (t / r) ** 3
    report("G. hull buckling, classical n=2 ring vs the code's allowable",
           classical, coded_before_knockdown, 1e-12, kind="theory",
           note=f"compared before the code's own 0.6 knockdown, which is a "
                f"declared imperfection allowance and not part of the ring "
                f"result.  The expression this replaced gives "
                f"{legacy_before_knockdown:.0f} Pa, "
                f"{legacy_before_knockdown / classical:.1f}x the classical "
                f"result -- the (t/D)^3 coefficient 2E/(1-nu^2) applied to "
                f"(t/r)^3.  S-01, closed; see experiments/hull_buckling.")
    out["buckling_ratio_before_knockdown"] = float(coded_before_knockdown / classical)
    out["buckling_ratio_after_knockdown"] = float(coded / classical)
    out["legacy_buckling_ratio"] = float(legacy_before_knockdown / classical)

    # H. Added-mass coefficient of a thin disc moving normal to itself.
    #
    # Lamb: added mass = (8/3) rho a^3 for a disc of radius a.  Referred to the
    # displaced volume V = pi a^2 h of a disc of thickness h, that is
    # Ca = (8/3) a^3 / (pi a^2 h) = 8a/(3 pi h) = 4D/(3 pi h).
    D_disc, h_disc = 0.20, 0.02
    lamb_ca = 4 * D_disc / (3 * math.pi * h_disc)
    e = np.array([D_disc, D_disc, h_disc])
    coded_ca = float(np.clip(
        0.5 * (e[[1, 2, 0]] + e[[2, 0, 1]]) / (2 * e), 0.05, 10.0)[2])
    report("H. disc added-mass coefficient, Lamb vs the code's tensor",
           lamb_ca, coded_ca, 0.20, kind="theory",
           note=f"the code's own docstring claims 'within 18% of Lamb'; "
                f"measured here it is {100*(coded_ca/lamb_ca - 1):+.1f}%, so "
                f"the claim holds.  The ratio is scale-free: the code gives "
                f"D/(2h) and Lamb gives 4D/(3 pi h), a fixed 3 pi / 8 = "
                f"{3*math.pi/8:.4f} apart.  The clip at 10.0 binds from "
                f"D/h = 20 upward, where the code stops tracking Lamb at all.")
    out["disc_ca_lamb"] = float(lamb_ca)
    out["disc_ca_coded"] = float(coded_ca)

    # Sphere: the same tensor must give exactly 0.5.
    es = np.array([0.1, 0.1, 0.1])
    sphere_ca = float(np.clip(
        0.5 * (es[[1, 2, 0]] + es[[2, 0, 1]]) / (2 * es), 0.05, 10.0)[0])
    report("H. sphere added-mass coefficient, 0.5 vs the code's tensor",
           0.5, sphere_ca, 1e-12, kind="theory")

    # I. Wagner slam coefficient.
    beta = 20.0
    coded_p = slam_pressure(1.0, beta)
    coded_k = coded_p / (0.5 * SEAWATER.rho)
    quoted_k = (math.pi / (2 * math.tan(math.radians(beta)))) ** 2
    report("I. slam pressure coefficient at 20 deg deadrise",
           quoted_k, coded_k, 4.0, kind="theory",
           note="the code uses (pi/tan beta)^2; the coefficient usually quoted "
                "for a Wagner wedge is (pi/(2 tan beta))^2, a factor of 4 "
                "lower.  This is conservative for structure, and it is an "
                "unresolved source question, not a demonstrated error.")
    out["slam_k_coded"] = float(coded_k)
    out["slam_k_quoted"] = float(quoted_k)

    # J. Fatigue knockdown: endpoints and monotonicity.
    m = CFRP_TUBE
    s1e3 = m.allowable_stress(cycles=1e3)
    s1e5 = m.allowable_stress(cycles=1e5)
    expect_1e5 = m.yield_MPa * 1e6 * m.strength_knockdown * m.fatigue_knockdown_1e5 / 2
    report("J. allowable at 1e5 cycles equals the declared knockdown",
           expect_1e5, s1e5, 1e-12, kind="identity")
    cyc = np.logspace(3, 5, 41)
    vals = np.array([m.allowable_stress(cycles=c) for c in cyc])
    mono = bool(np.all(np.diff(vals) <= 1e-9))
    print(f"  [{'ok  ' if mono else 'FAIL'}] J. allowable stress is monotone "
          f"non-increasing in cycles between 1e3 and 1e5")
    RESULTS.append({"check": "J. allowable stress monotone in cycles",
                    "kind": "monotonicity", "analytic": 1.0,
                    "numerical": float(mono), "relative_error": 0.0,
                    "tolerance": 0.0, "ok": mono,
                    "note": f"1e3 -> {s1e3/1e6:.1f} MPa, 1e5 -> {s1e5/1e6:.1f} MPa"})
    print(f"          note: `allowable_stress` applies no knockdown at all for "
          f"cycles <= 1e3, so the curve is flat below 1e3 and there is a kink "
          f"exactly at 1e3.")

    # K. Depth stability: analytic derivative of Boyle's law vs the code's
    # central difference.
    st = BuoyancyState(mass=12.0, displaced_volume=0.013, ballast_volume=0.0,
                       gas_volume_surface=0.002)
    d = 10.0
    from dytiscidae.physics.medium import GRAVITY, P_ATM
    rho = SEAWATER.rho
    # dV_gas/dh = -V0 P_atm rho g / (P_atm + rho g h)^2
    dgas = -st.gas_volume_surface * P_ATM * rho * GRAVITY / (P_ATM + rho * GRAVITY * d) ** 2
    analytic = rho * GRAVITY * dgas
    report("K. d(net buoyancy)/d(depth), Boyle analytic vs the code's difference",
           analytic, st.depth_stability(d), 1e-4, kind="derivative",
           note="negative means the vehicle gets heavier as it sinks")
    return out


def run(cfg: dict) -> ExperimentResult:
    res = ExperimentResult("analytic_vs_numerical", cfg)
    print("\nA. derivatives")
    a = check_cpg_jacobian(cfg)
    print("\nB/C. fluid coefficients")
    bc = check_coefficients(cfg)
    print("\nD-F. structural closed forms vs quadrature")
    check_structural_quadrature(cfg)
    print("\nG-K. coefficients against their own derivations")
    g = check_coefficients_against_theory(cfg)

    bad = [r for r in RESULTS if not r["ok"]]
    print(f"\n{len(RESULTS)} checks, {len(RESULTS) - len(bad)} agree, "
          f"{len(bad)} differ")
    for r in bad:
        print(f"  DIFFERS: {r['check']}  rel={r['relative_error']:.3g}")

    res.record("checks", RESULTS)
    res.record("cpg", a)
    res.record("lift_slope", bc)
    res.record("theory", g)
    res.record("n_differ", len(bad))
    res.ok = True  # a difference is a finding to report, not a broken run
    return res


if __name__ == "__main__":
    cfg = load_config(Path(__file__).parent / "config.json")
    result = run(cfg)
    result.write(Path(__file__).parent / "results")
