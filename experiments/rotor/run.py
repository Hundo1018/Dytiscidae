#!/usr/bin/env python3
"""The rotor model against a measured propeller.

UIUC Propeller Data Site, Vol. 1 (Brandt, Deters & Selig), APC 10x4.7 Slow
Flyer: https://m-selig.ae.illinois.edu/props/volume-1/propDB-volume-1.html
(values as fetched in runs/_logs/aero_literature_0923.md item 7).

    static, 4029 RPM:  CT 0.1158  CP 0.0466
    5018 RPM:  J  0.115  0.262  0.408  0.519  0.576
               CT 0.1077 0.0856 0.0597 0.0352 0.0206
               CP 0.0476 0.0442 0.0383 0.0312 0.0268

Geometry: D = 0.254 m, pitch 4.7 in, two blades.  The APC chord distribution
is not modelled; a constant chord is, and its value is swept rather than
chosen, because it is the one free number and the answer should not depend on
picking it.  CT = T / (rho n^2 D^4), CP = P / (rho n^3 D^5), n in rev/s.

    PYTHONPATH=. python experiments/rotor/run.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dytiscidae.physics.medium import AIR  # noqa: E402
from dytiscidae.physics.rotor import RotorSpec, bemt  # noqa: E402

D = 0.254
PITCH = 4.7 * 0.0254
STATIC = (4029, 0.1158, 0.0466)
SWEEP_RPM = 5018
SWEEP = [(0.115, 0.1077, 0.0476), (0.262, 0.0856, 0.0442), (0.408, 0.0597, 0.0383),
         (0.519, 0.0352, 0.0312), (0.576, 0.0206, 0.0268)]


def coeffs(spec, rpm, J):
    n = rpm / 60.0
    V = J * n * D
    T, Q = bemt(spec, 2 * math.pi * n, V, 0.0, AIR.rho, AIR.mu)
    return T / (AIR.rho * n**2 * D**4), Q * 2 * math.pi * n / (AIR.rho * n**3 * D**5)


def main() -> int:
    out = {}
    # Camber is the one number fitted: to static CT alone, on a 0.5% grid.
    # The J sweep and every CP are then out of sample.
    best = min(np.arange(0.0, 0.1001, 0.005), key=lambda f: abs(coeffs(
        RotorSpec(radius=D / 2, pitch=PITCH, blades=2, chord_ratio=0.12, camber=f),
        STATIC[0], 0.0)[0] - STATIC[1]))
    print(f"camber fitted to static CT (c/R 0.12): f/c = {best:.3f}")
    out["camber_fit"] = float(best)
    print(f"{'c/R':>5} {'CT0':>7} {'CP0':>7} | CT(J) model/measured ... | rms CT  rms CP")
    for cr in (0.10, 0.12, 0.14, 0.16):
        spec = RotorSpec(radius=D / 2, pitch=PITCH, blades=2, chord_ratio=cr, camber=best)
        ct0, cp0 = coeffs(spec, STATIC[0], 0.0)
        rows = [(J, *coeffs(spec, SWEEP_RPM, J), ct, cp) for J, ct, cp in SWEEP]
        e_ct = math.sqrt(np.mean([(r[1] - r[3]) ** 2 for r in rows]))
        e_cp = math.sqrt(np.mean([(r[2] - r[4]) ** 2 for r in rows]))
        print(f"{cr:5.2f} {ct0:7.4f} {cp0:7.4f} | "
              + " ".join(f"{r[1]:.3f}/{r[3]:.3f}" for r in rows)
              + f" | {e_ct:.4f} {e_cp:.4f}")
        out[f"{cr:.2f}"] = {"ct0": ct0, "cp0": cp0, "rows": rows, "rms_ct": e_ct, "rms_cp": e_cp}
    print(f"measured static: CT {STATIC[1]}  CP {STATIC[2]}")
    Path(__file__).with_name("results").mkdir(exist_ok=True)
    (Path(__file__).with_name("results") / "result.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
