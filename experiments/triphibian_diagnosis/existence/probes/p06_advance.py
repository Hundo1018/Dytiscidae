#!/usr/bin/env python3
"""Probe: rotor thrust and torque against axial advance ratio J = V_ax/(Omega R),
in air and in water (the repo's own `bemt`), for the controller's inflow term."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from dytiscidae.core.bodyplans import REFERENCE_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.physics.medium import AIR, SEAWATER
from dytiscidae.physics.rotor import bemt
p = build(REFERENCE_PLANS["quad"]())
spec = next(s.rotor for s in p.segments if s.rotor is not None)
for name, fl, om in (("air", AIR, 600.0), ("water", SEAWATER, 40.0)):
    T0, Q0 = bemt(spec, om, 0.0, 0.0, fl.rho, fl.mu)
    row = []
    for J in (-0.2, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        T, Q = bemt(spec, om, J * om * spec.radius, 0.0, fl.rho, fl.mu)
        row.append(f"J={J:+.2f} T/T0={T/T0:+.3f} Q/Q0={Q/Q0:+.3f}")
    print(name, f"om={om} T0={T0:.3f} Q0={Q0:.4f}")
    print("  " + " | ".join(row))
