"""L0: what a variable-throw rocker buys in water and on land, as arithmetic.

One motor turns a crank of radius r continuously; a coupler drives each wing's
rocker at a distance L from the shoulder that a self-locking screw can move.
Stroke = 2 asin(r / L); the crank sees the shoulder torque times r / L.
The water part reads the held-rig map of L1 pass 1b; the land part is
arithmetic.  Caps are L0's (closure.py, L0.md); TYPED = nobody's measurement.

    python experiments/flapper/L0/mechanism.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

G = 9.81
TAU0 = 10.5                      # N m at the shoulders at take-off (L0: 9-12)
MASS = 0.60
SHOULDER_CAP = 1.5               # TYPED: structure takes 1.5 x the take-off torque
THUMB = 0.15                     # TYPED: thumb to shoulder, m
TILT = math.radians(45.0)        # TYPED: stroke plane against the vertical


def water():
    """Per stroke, the best measured net forward force at 0.3 m/s within the
    shoulder cap (L1 pass 1b, results/L1b_water.jsonl).  The scaling rule this
    replaced left out added mass and was wrong by up to 15 times."""
    path = Path(__file__).resolve().parents[1] / "results" / "L1b_water.jsonl"
    rows = [json.loads(line) for line in open(path)]
    cap = SHOULDER_CAP * TAU0
    for deg in sorted({r["Phi_deg"] for r in rows}):
        ok = [r for r in rows if r["Phi_deg"] == deg and r["stroke_torque_peak"] <= cap]
        if not ok:
            yield deg, None
            continue
        yield deg, max(ok, key=lambda r: r["Fx"])


def main():
    print("water, held at 0.3 m/s: best net forward force with the shoulder under "
          f"{SHOULDER_CAP * TAU0:.1f} N m")
    print("stroke  L/r  | Hz    feather  net N  shoulder N m  crank N m   W")
    for deg, r in water():
        if r is None:
            print(f"{deg:5.0f}        | no point under the cap")
            continue
        print(f"{deg:5.0f} {1 / math.sin(math.radians(deg) / 2):5.2f} | {r['f']:4.2f} "
              f"{r['Psi_deg']:7.0f} {r['Fx']:7.2f} {r['stroke_torque_peak']:11.1f} "
              f"{r['crank_torque']:10.1f} {r['aero_power_W']:6.2f}")
    lift = MASS * G * THUMB
    print(f"\ncrutch on land: shoulder torque to lift the body {lift:.2f} N m "
          f"({lift / TAU0:.2f} of take-off)")
    for deg, hz in ((30, 1.0), (40, 1.5), (60, 1.5)):
        stride = THUMB * math.radians(deg) * math.sin(TILT)
        print(f"  stroke {deg} deg at {hz} Hz: stride {stride * 100:.1f} cm, "
              f"{stride * hz:.2f} m/s if the thumb does not slip")


if __name__ == "__main__":
    main()
