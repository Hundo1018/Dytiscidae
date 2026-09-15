#!/usr/bin/env python3
"""Dimensional audit of every physics and control formula in the project.

The question
------------
Does each formula the solver evaluates produce the dimension its name and its
docstring claim?  A formula that is dimensionally wrong is wrong.  A formula
that is dimensionally *right only if some bare literal carries a hidden unit*
is a different and more interesting case: it works, but the literal is a
physical constant nobody declared, and changing anything around it silently
changes the model.  Both are reported.

The design
----------
Each check restates one formula with `experiments.units.Q` and asserts the
result dimension.  A restatement can drift away from the code it claims to
describe, so every check also carries the exact source fragment it is a
restatement of, and the run fails if that fragment is no longer in the file.
That is the whole anti-drift mechanism: the check cannot outlive the line.

Run:  PYTHONPATH=. python experiments/dimensional_check/run.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.harness import ExperimentResult, load_config  # noqa: E402
from experiments.units import (  # noqa: E402
    ACCEL, ANG_ACCEL, ANG_VEL, AREA_MOMENT, DENSITY, Dim, DimensionError,
    ENERGY, FORCE, HZ, INERTIA, KG, M, M2, M3, MOMENT, MOTOR_CONSTANT, ONE,
    POWER, PRESSURE, Q, S, SECTION_MODULUS, VELOCITY, VISCOSITY, VOLUME_FLOW,
)


@dataclass
class Check:
    """One formula, restated dimensionally and tied to its source line."""

    name: str
    source: str           # "path/to/file.py"
    snippet: str          # must appear verbatim in that file
    expect: Dim
    compute: callable
    #: A literal in the formula that must carry a unit for the dimensions to
    #: work out, but is written as a bare number.  Empty when there is none.
    implicit_unit: str = ""
    note: str = ""
    # filled in by run()
    got: str = field(default="", init=False)
    value: float = field(default=0.0, init=False)
    status: str = field(default="", init=False)
    detail: str = field(default="", init=False)


# --------------------------------------------------------------------------
# Representative values.  Dimensions do not depend on them; they are here so a
# failure prints a number a reader can sanity-check by eye.
# --------------------------------------------------------------------------

rho_w = Q.of(1025.0, DENSITY)      # seawater
rho_a = Q.of(1.225, DENSITY)       # air
mu_w = Q.of(1.08e-3, VISCOSITY)
g = Q.of(9.80665, ACCEL)
U = Q.of(8.0, VELOCITY)            # relative flow speed
c = Q.of(0.18, M)                  # chord
dr = Q.of(0.05, M)                 # strip width
b = Q.of(0.9, M)                   # semi-span
S_area = Q.of(0.16, M2)
V = Q.of(0.004, M3)
omega = Q.of(13.8, ANG_VEL)        # rate about the span axis
alpha_rate = omega
E = Q.of(1.35e11, PRESSURE)
t_wall = Q.of(0.0015, M)
r_hull = Q.of(0.09, M)
mass = Q.of(12.0, KG)
torque = Q.of(1.4, MOMENT)
km = Q.of(0.05, MOTOR_CONSTANT)
Q_flow = Q.of(0.0025, VOLUME_FLOW)
A_orifice = Q.of(0.0012, M2)
dt = Q.of(0.002, S)
depth = Q.of(10.0, M)


CHECKS: list[Check] = [
    # ---------------------------------------------------------------- fluid
    Check(
        "dynamic pressure  q = 0.5 rho U^2",
        "dytiscidae/physics/fluid.py",
        "q = 0.5 * rho * U**2",
        PRESSURE,
        lambda: 0.5 * rho_w * U**2,
    ),
    Check(
        "Reynolds number  Re = rho U c / mu",
        "dytiscidae/physics/fluid.py",
        "re = rho * U * p.chord / np.maximum(mu, 1e-12)",
        ONE,
        lambda: rho_w * U * c / mu_w,
    ),
    Check(
        "circulatory lift  L = q S CL",
        "dytiscidae/physics/fluid.py",
        "L = q * p.area * cl * self.lift_scale",
        FORCE,
        lambda: (0.5 * rho_w * U**2) * S_area * Q.of(1.1),
    ),
    Check(
        "reduced pitch rate  kappa = |alpha_dot| c / (2 U)",
        "dytiscidae/physics/fluid.py",
        "reduced_pitch_rate = np.abs(omega_s) * p.chord / (2.0 * U_safe)",
        ONE,
        lambda: abs(alpha_rate) * c / (2.0 * U),
        note="dimensionally a reduced rate, and now named for the quantity it "
             "holds: omega_s is the instantaneous rate about the span axis, "
             "not the flapping angular frequency.  Whether the LEV term should "
             "key on it at all is MATH_AUDIT F-02.",
    ),
    Check(
        "LEV blend  lev = clip(kappa / 0.30, 0, 1)",
        "dytiscidae/physics/fluid.py",
        "lev = np.clip(reduced_pitch_rate / 0.30, 0.0, 1.0)",
        ONE,
        lambda: abs(alpha_rate) * c / (2.0 * U) / Q.of(0.30),
        implicit_unit="none -- 0.30 is a reduced rate, correctly dimensionless",
    ),
    Check(
        "stall angle  alpha_stall = 11 deg + 26 deg * lev",
        "dytiscidae/physics/fluid.py",
        "alpha_stall = np.radians(11.0 + 26.0 * lev)",
        ONE,
        lambda: Q.of(0.192) + Q.of(0.454) * Q.of(0.5),
        implicit_unit="11.0 and 26.0 are degrees, made explicit by np.radians",
    ),
    Check(
        "low-Re stall knockdown  0.55 + 0.45 log10(Re)/5",
        "dytiscidae/physics/fluid.py",
        "alpha_stall *= np.clip(0.55 + 0.45 * np.log10(np.maximum(re, 10.0)) / 5.0",
        ONE,
        lambda: Q.of(0.55) + Q.of(0.45) * Q.of(5.28) / Q.of(5.0),
        implicit_unit="the 5.0 divides log10(Re); it is 5 decades of Reynolds "
                      "number, i.e. Re=1e5 is the anchor where the factor "
                      "reaches 1.0.  Nothing in the code says so.",
    ),
    Check(
        "finite-wing lift slope  2 pi / (1 + 2/AR)",
        "dytiscidae/physics/fluid.py",
        "cl_alpha = 2.0 * np.pi / (1.0 + 2.0 / np.maximum(ar, 0.5))",
        ONE,
        lambda: Q.of(6.2832) / (Q.of(1.0) + Q.of(2.0) / Q.of(4.0)),
        note="per radian; radian is dimensionless so this is [1].",
    ),
    Check(
        "induced drag  cd_i = CL^2 / (pi e AR)",
        "dytiscidae/physics/fluid.py",
        "cd_i = cl**2 / (np.pi * oswald * np.maximum(ar, 0.5))",
        ONE,
        lambda: Q.of(1.1) ** 2 / (Q.of(3.14159) * Q.of(0.75) * Q.of(4.0)),
    ),
    Check(
        "skin friction, laminar branch  1.328 / sqrt(Re)",
        "dytiscidae/physics/fluid.py",
        "lam = 1.328 / np.sqrt(re)",
        ONE,
        lambda: Q.of(1.328) / (Q.of(1.9e6) ** 0.5),
    ),
    Check(
        "skin friction, turbulent branch  0.074 / Re^0.2",
        "dytiscidae/physics/fluid.py",
        "turb = 0.074 / re**0.2",
        ONE,
        lambda: Q.of(0.074) / (Q.of(1.9e6) ** 0.2),
    ),
    Check(
        "skin friction blend centre  log10(Re) - 5.7",
        "dytiscidae/physics/fluid.py",
        "w = 1.0 / (1.0 + np.exp(-(np.log10(re) - 5.7) * 4.0))",
        ONE,
        lambda: Q.of(1.0) / (Q.of(1.0) + Q.of(0.30)),
        implicit_unit="5.7 is log10(Re) at the transition centre, i.e. "
                      "Re = 5.0e5; the 4.0 is 1/decade of blend width.",
    ),
    Check(
        "Kramer rotational force  C_rot rho U omega c^2 dr",
        "dytiscidae/physics/fluid.py",
        "self.c_rot * rho * U * omega_s * p.chord**2 * p.dr",
        FORCE,
        lambda: Q.of(1.57) * rho_w * U * omega * c**2 * dr,
    ),
    Check(
        "strip added mass  rho pi c^2 / 4 per unit span",
        "dytiscidae/physics/fluid.py",
        "rho * np.pi * p.chord**2 * 0.25 * p.dr",
        KG,
        lambda: rho_w * Q.of(3.14159) * c**2 * Q.of(0.25) * dr,
    ),
    Check(
        "bluff added mass  Ca rho V",
        "dytiscidae/physics/fluid.py",
        "ca_eff * rho * p.volume",
        KG,
        lambda: Q.of(0.5) * rho_w * V,
    ),
    Check(
        "added rotational inertia  m_add * lever^2",
        "dytiscidae/physics/fluid.py",
        "m_body * self._lever2",
        INERTIA,
        lambda: Q.of(32.0, KG) * Q.of(0.25, M2),
    ),
    Check(
        "buoyancy  rho g V f_sub",
        "dytiscidae/physics/fluid.py",
        "f_buoy = self.medium.water.rho * GRAVITY * p.volume_buoyant * subf",
        FORCE,
        lambda: rho_w * g * V * Q.of(0.6),
    ),
    Check(
        "slam force  |d(m_add)/dt * v_n|",
        "dytiscidae/physics/fluid.py",
        "slam = float(np.abs((m_add - self._prev_ma) / dt * vn).max())",
        FORCE,
        lambda: (Q.of(32.0, KG) / dt) * Q.of(3.0, VELOCITY),
        note="FluidDiagnostics.slam is documented as a force in N; "
             "transitions.py divides it by an area to get a pressure.",
    ),
    Check(
        "bound circulation  Gamma = 0.5 CL U c",
        "dytiscidae/physics/fluid.py",
        '"gamma": 0.5 * cl * U * p.chord',
        M2 / S,
        lambda: Q.of(0.5) * Q.of(1.1) * U * c,
        note="Kutta-Joukowski: L' = rho U Gamma with L' in N/m.",
    ),
    Check(
        "bluff axial force  0.5 rho |v| v (Cd A_front + Cf A_wet)",
        "dytiscidae/physics/fluid.py",
        "* (p.cd_bluff * ey * ez + skin_friction_cd(re_b) * wetted)",
        FORCE,
        lambda: Q.of(0.5) * rho_w * abs(U) * U
        * (Q.of(0.9) * Q.of(0.04, M2) + Q.of(0.006) * Q.of(0.3, M2)),
    ),
    Check(
        "bluff cross-flow force  0.5 rho u_c^2 Cd_cross A_side",
        "dytiscidae/physics/fluid.py",
        "f_cross = self.cd_scale * 0.5 * rho * u_cross**2 * CD_CROSSFLOW * a_side",
        FORCE,
        lambda: Q.of(0.5) * rho_w * U**2 * Q.of(1.1) * Q.of(0.05, M2),
    ),
    # ---------------------------------------------------------------- medium
    Check(
        "hydrostatic pressure  p_atm + rho g h",
        "dytiscidae/physics/medium.py",
        "return P_ATM + water.rho * GRAVITY * max(depth_m, 0.0)",
        PRESSURE,
        lambda: Q.of(101325.0, PRESSURE) + rho_w * g * depth,
    ),
    Check(
        "Airy deep-water orbital speed  a omega e^{kz}",
        "dytiscidae/physics/medium.py",
        "u_mag = self.amplitude * omega * decay",
        VELOCITY,
        lambda: Q.of(0.05, M) * Q.of(3.14, ANG_VEL) * Q.of(0.7),
    ),
    Check(
        "geometric viscosity blend  mu_a^(1-f) mu_w^f",
        "dytiscidae/physics/medium.py",
        "mu = self.air.mu ** (1.0 - f) * self.water.mu**f",
        VISCOSITY,
        lambda: Q.of(1.81e-5 ** 0.4 * 1.002e-3 ** 0.6, VISCOSITY),
        implicit_unit="raising a dimensional quantity to a non-integer power is "
                      "only meaningful because both factors carry the same "
                      "unit and the exponents sum to 1.  It is a weighted "
                      "geometric mean, and it is dimensionally sound *only* "
                      "for that reason.",
    ),
    # ------------------------------------------------------------------- jet
    Check(
        "jet thrust  rho Q^2 / A",
        "dytiscidae/physics/jet.py",
        "thrust_mag = coeff * rho * q * np.abs(q) / area",
        FORCE,
        lambda: rho_w * Q_flow * abs(Q_flow) / A_orifice,
    ),
    Check(
        "jet pumping power, as the docstring states it",
        "dytiscidae/physics/jet.py",
        "P_pump = p * Q = 0.5 * rho * Q^3 / A^2 = 0.5 * m_dot * v_e^2",
        POWER,
        lambda: Q.of(0.5) * rho_w * Q_flow**3 / A_orifice**2,
        note="anchored to the module docstring rather than to a line of code, "
             "because the implementation folds this into a damping "
             "coefficient and never forms the pressure as a named variable.  "
             "A documented relation is source too, and a docstring that "
             "drifts from its own dimensions is worth catching.",
    ),
    Check(
        "jet pumping load  c = 0.5 rho (dV/dtheta)^3 |omega| / A^2",
        "dytiscidae/physics/jet.py",
        "c_pump = (coeff * 0.5 * rho * dv_dtheta**3 * np.abs(omega) * subf",
        MOMENT * S,
        lambda: Q.of(1.0) * Q.of(0.5) * rho_w
        * Q.of(1.5e-3, M3) ** 3 * omega * Q.of(0.8) / A_orifice**2,
        note="a damping coefficient, N.m.s per radian.  dV/dtheta is a volume "
             "per radian and radian is dimensionless, so it carries m^3; the "
             "cube of it over an area squared leaves m^5, and rho times that "
             "over a second is N.m.s.  It is applied through `dof_damping` "
             "rather than as an explicit torque because the explicit form "
             "diverges -- see the module docstring.",
    ),
    Check(
        "jet volume rate from joint rate  -V0 f omega / span",
        "dytiscidae/physics/jet.py",
        "dv_dt = -self.volume * self.stroke_fraction * (omega / span)",
        VOLUME_FLOW,
        lambda: -V * Q.of(0.4) * (omega / Q.of(1.2)),
        implicit_unit="`span` is a joint-angle range in radians, so it is "
                      "dimensionless and omega/span has units of 1/s.",
    ),
    # ------------------------------------------------------------- structure
    Check(
        "tube second moment  pi (ro^4 - ri^4) / 4",
        "dytiscidae/physics/structure.py",
        "i = math.pi * (ro**4 - ri**4) / 4.0",
        AREA_MOMENT,
        lambda: Q.of(3.14159) * (Q.of(0.008, M) ** 4 - Q.of(0.0065, M) ** 4) / 4.0,
    ),
    Check(
        "section modulus  Z = I / ro",
        "dytiscidae/physics/structure.py",
        "z = i / ro if ro > 0 else 0.0",
        SECTION_MODULUS,
        lambda: Q.of(1.5e-9, AREA_MOMENT) / Q.of(0.008, M),
    ),
    Check(
        "elliptic root moment  L_semi b 4/(3 pi)",
        "dytiscidae/physics/structure.py",
        "moment = l_semi * semi_span * 4.0 / (3.0 * math.pi)",
        MOMENT,
        lambda: Q.of(180.0, FORCE) * b * Q.of(4.0) / (Q.of(3.0) * Q.of(3.14159)),
    ),
    Check(
        "bending stress  sigma = M / Z",
        "dytiscidae/physics/structure.py",
        "sigma = moment / z",
        PRESSURE,
        lambda: Q.of(70.0, MOMENT) / Q.of(1.9e-7, SECTION_MODULUS),
    ),
    Check(
        "flap angular acceleration  alpha = A (2 pi f)^2",
        "dytiscidae/physics/structure.py",
        "ang_acc = flap_amplitude_rad * omega**2",
        ANG_ACCEL,
        lambda: Q.of(0.7) * Q.of(13.8, ANG_VEL) ** 2,
    ),
    Check(
        "swept-surface root moment  0.5 rho omega^2 Cd int(c r^3 dr)",
        "dytiscidae/physics/structure.py",
        "moment = 0.5 * SEAWATER.rho * omega**2 * drag_coefficient * integral",
        MOMENT,
        lambda: Q.of(0.5) * rho_w * Q.of(5.0, ANG_VEL) ** 2 * Q.of(1.9)
        * Q.of(0.02, M**5),
        note="the integral int(c(r) r^3 dr) carries m.m^3.m = m^5, not m^4; "
             "getting that wrong is exactly what this check is for.",
    ),
    Check(
        "inertial reversal moment  M = I alpha",
        "dytiscidae/physics/structure.py",
        "moment = i_root * ang_acc",
        MOMENT,
        lambda: Q.of(0.9, INERTIA) * (Q.of(0.7) * Q.of(13.8, ANG_VEL) ** 2),
    ),
    Check(
        "cantilever tip deflection  q L^4 / (8 E I)",
        "dytiscidae/physics/structure.py",
        "delta = q * semi_span**4 / (8.0 * material.E * i)",
        M,
        lambda: Q.of(100.0, FORCE / M) * b**4 / (Q.of(8.0) * E * Q.of(1.5e-9, AREA_MOMENT)),
    ),
    Check(
        "hoop stress  p r / t",
        "dytiscidae/physics/structure.py",
        "hoop = p_gauge * radius / max(wall, 1e-6)",
        PRESSURE,
        lambda: Q.of(1.0e5, PRESSURE) * r_hull / t_wall,
    ),
    Check(
        "shell buckling  E/(1-nu^2) (t/r)^3",
        "dytiscidae/physics/structure.py",
        "p_cr = 0.6 * 2.0 * material.E / (1.0 - nu**2) * (wall / max(radius, 1e-6)) ** 3",
        PRESSURE,
        lambda: Q.of(0.6) * Q.of(2.0) * E / (Q.of(1.0) - Q.of(0.09))
        * (t_wall / r_hull) ** 3,
        note="dimensionally sound; the *coefficient* is the (t/D)^3 one applied "
             "to (t/r)^3.  See MATH_AUDIT S-01.",
    ),
    Check(
        "Wagner slam pressure  0.5 rho v^2 (pi/tan beta)^2",
        "dytiscidae/physics/structure.py",
        "return 0.5 * SEAWATER.rho * impact_speed**2 * k",
        PRESSURE,
        lambda: Q.of(0.5) * rho_w * Q.of(8.0, VELOCITY) ** 2 * Q.of(120.0),
    ),
    Check(
        "flat-panel slam bending  0.30 p b^2 / t^2",
        "dytiscidae/physics/structure.py",
        "return 0.30 * p * flat_panel_width**2 / max(wall, 1e-4) ** 2, p",
        PRESSURE,
        lambda: Q.of(0.30) * Q.of(4.0e5, PRESSURE) * Q.of(0.2, M) ** 2 / t_wall**2,
    ),
    Check(
        "net buoyancy  rho g V - m g",
        "dytiscidae/physics/structure.py",
        "return SEAWATER.rho * GRAVITY * v - self.mass * GRAVITY",
        FORCE,
        lambda: rho_w * g * V - mass * g,
    ),
    Check(
        "ballast pump power  dp Q / eta",
        "dytiscidae/physics/structure.py",
        "return dp * flow_m3_s / max(efficiency, 1e-3)",
        POWER,
        lambda: (rho_w * g * depth) * Q_flow / Q.of(0.35),
    ),
    # ---------------------------------------------------------------- energy
    Check(
        "copper loss  (tau / km)^2",
        "dytiscidae/physics/energy.py",
        "p_copper = (tau_m / max(self.km, 1e-9)) ** 2",
        POWER,
        lambda: (torque / km) ** 2,
    ),
    Check(
        "viscous loss  k_visc omega^2",
        "dytiscidae/physics/energy.py",
        "p_visc = self.k_visc * om_m**2",
        POWER,
        lambda: Q.of(2e-5, MOMENT * S) * omega**2,
        implicit_unit="k_visc is documented as 'N.m per rad/s', which is "
                      "N.m.s -- consistent.",
    ),
    Check(
        "iron loss  k_iron omega",
        "dytiscidae/physics/energy.py",
        "p_iron = self.k_iron * om_m",
        POWER,
        lambda: Q.of(0.02 * 240.0 / 1000.0, MOMENT) * omega,
        implicit_unit="k_iron = 0.02 * p_cont / 1000.0 is built from a POWER "
                      "and a bare 1000.  For p_iron to be a power, k_iron must "
                      "be W/(rad/s), so the 1000 is a reference speed of "
                      "1000 rad/s.  Nothing in the code says so.  See "
                      "MATH_AUDIT E-01.",
    ),
    Check(
        "motor constant  km ~ mass^0.75",
        "dytiscidae/physics/energy.py",
        "self.km = 0.05 * (self.mass / 0.1) ** 0.75 * self.spec.efficiency_peak / 0.88",
        MOTOR_CONSTANT,
        lambda: Q.of(0.05, MOTOR_CONSTANT) * Q.of(0.8) ** 0.75 * Q.of(0.88) / Q.of(0.88),
        implicit_unit="0.05 is N.m/sqrt(W) at a reference mass of 0.1 kg; "
                      "0.88 is BLDC_OUTRUNNER.efficiency_peak written as a "
                      "literal, so editing that table entry rescales km for "
                      "every motor class.  See MATH_AUDIT E-02.",
    ),
    Check(
        "stall torque  km sqrt(P) G eta",
        "dytiscidae/physics/energy.py",
        "return self.km * np.sqrt(self.p_cont) * self.gear_ratio * self.eta_gear",
        MOMENT,
        lambda: km * (Q.of(240.0, POWER) ** 0.5) * Q.of(5.0) * Q.of(0.94),
    ),
    # --------------------------------------------------------------- control
    Check(
        "CPG joint command  offset + A sin(2 pi f t + phi)",
        "dytiscidae/control/cpg.py",
        "return p.offset + p.amplitude * np.sin(",
        ONE,
        lambda: Q.of(0.1) + Q.of(0.4) * Q.of(0.5),
        note="joint angles are radians, hence dimensionless.  The CPG "
             "parameter vector therefore mixes radians (amplitude, phase, "
             "offset) with Hz (frequency) in one flat array.  See "
             "MATH_AUDIT C-01.",
    ),
    Check(
        "twist scaling  [1,1,1,0.3,0.3,0.3]",
        "dytiscidae/control/cpg.py",
        "scale = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])",
        VELOCITY,
        lambda: Q.of(1.2, ANG_VEL) * Q.of(0.3, M),
        implicit_unit="the three 0.3s must be a LENGTH in metres for an "
                      "angular rate in rad/s to be commensurable with a linear "
                      "rate in m/s.  Written as a bare number.  Every singular "
                      "value, `reach`, cond(A) and the damping lambda inherit "
                      "it.  See MATH_AUDIT C-02.",
    ),
    Check(
        "damping  lam = 0.01 tr(G)/r",
        "dytiscidae/control/cpg.py",
        "lam = 0.01 * float(np.trace(G)) / max(r, 1) + 1e-12",
        (VELOCITY) ** 2,
        lambda: Q.of(0.01) * Q.of(3.2, VELOCITY**2) / Q.of(4.0),
        implicit_unit="G = A A^T carries (twist per coefficient)^2, so lam "
                      "does too and the ridge is relative.  The additive 1e-12 "
                      "is NOT: it is a bare number added to a dimensional one. "
                      "It only survives because it is negligible except when "
                      "tr(G) is exactly zero, which is the case it exists for. "
                      "See MATH_AUDIT C-03.",
    ),
]


def run(cfg: dict) -> ExperimentResult:
    res = ExperimentResult("dimensional_check", cfg)
    missing_sources: list[str] = []

    for chk in CHECKS:
        path = ROOT / chk.source
        if not path.exists():
            chk.status = "SOURCE MISSING"
            chk.detail = chk.source
            missing_sources.append(chk.name)
            continue
        text = path.read_text()
        if chk.snippet not in text:
            chk.status = "SNIPPET GONE"
            chk.detail = f"{chk.source} no longer contains: {chk.snippet!r}"
            missing_sources.append(chk.name)
            continue
        try:
            q = chk.compute()
        except DimensionError as exc:
            chk.status = "DIMENSION ERROR"
            chk.detail = str(exc)
            continue
        chk.got = str(q.dim)
        chk.value = q.value
        if q.dim == chk.expect:
            chk.status = "ok"
        else:
            chk.status = "MISMATCH"
            chk.detail = f"got [{q.dim}] expected [{chk.expect}]"

    ok = [c for c in CHECKS if c.status == "ok"]
    bad = [c for c in CHECKS if c.status != "ok"]
    implicit = [c for c in CHECKS if c.implicit_unit and "MATH_AUDIT" in c.implicit_unit]

    print(f"\n{len(CHECKS)} formulas checked")
    for chk in CHECKS:
        mark = "ok  " if chk.status == "ok" else chk.status
        print(f"  [{mark}] {chk.name:<52} = {chk.value:.6g} [{chk.got}]")
        if chk.detail:
            print(f"         {chk.detail}")
    print(f"\n  {len(ok)} sound, {len(bad)} failing")
    print(f"  {len([c for c in CHECKS if c.implicit_unit])} carry a literal with "
          f"an undeclared unit, of which {len(implicit)} are ranked findings")

    res.record("n_checks", len(CHECKS))
    res.record("n_sound", len(ok))
    res.record("n_failing", len(bad))
    res.record("n_implicit_unit", len([c for c in CHECKS if c.implicit_unit]))
    res.record("failing", [c.name for c in bad])
    res.record("checks", [
        {"name": c.name, "source": c.source, "status": c.status,
         "dim": c.got, "value": c.value, "implicit_unit": c.implicit_unit,
         "note": c.note, "detail": c.detail}
        for c in CHECKS
    ])
    res.ok = not bad
    return res


if __name__ == "__main__":
    cfg = load_config(Path(__file__).parent / "config.json")
    result = run(cfg)
    result.write(Path(__file__).parent / "results")
    sys.exit(0 if result.ok else 1)
