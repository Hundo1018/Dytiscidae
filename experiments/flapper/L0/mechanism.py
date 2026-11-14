"""L0: what a variable-throw rocker buys in water and on land, as arithmetic.

One motor turns a crank of radius r continuously; a coupler drives each wing's
rocker at a distance L from the shoulder that a self-locking screw can move.
Stroke = 2 asin(r / L); the crank sees the shoulder torque times r / L.
Reference point and caps are L0's (closure.py, L0.md); TYPED = nobody's
measurement.

    python experiments/flapper/L0/mechanism.py
"""
from __future__ import annotations

import math

G = 9.81
WATER = 833.0                    # rho_water / rho_air at equal stroke
STROKE0 = math.radians(85.0)     # full stroke, air (RoboFalcon, A14 B.5)
F0 = 4.2                         # Hz, standing take-off of the L0 fixture
TAU0 = 10.5                      # N m at the shoulders at take-off (L0: 9-12)
MECH_W0 = 0.6 * 220              # W mechanical at take-off (L0: 196-240 W electrical)
MASS = 0.60
R = 0.70                         # wing length, m
SHOULDER_CAP = 1.5               # TYPED: structure takes 1.5 x the take-off torque
CRANK_CAP = 1.0                  # crank-shaft torque no higher than at take-off
STROUHAL = 0.3                   # TYPED: A11 B uses 0.2-0.4
THUMB = 0.15                     # TYPED: thumb to shoulder, m
TILT = math.radians(45.0)        # TYPED: stroke plane against the vertical


def water(stroke):
    """Highest water frequency at this stroke, and what limits it."""
    gear = math.sin(STROKE0 / 2) / math.sin(stroke / 2)        # crank advantage gained
    cap = min(SHOULDER_CAP, CRANK_CAP * gear)
    f = F0 * (STROKE0 / stroke) * math.sqrt(cap / WATER)
    shoulder = TAU0 * cap
    crank = cap / gear                                        # of its take-off torque
    excursion = 2 * 0.7 * R * math.sin(stroke / 2)            # at 0.7 of the wing
    speed = f * excursion / STROUHAL
    power = MECH_W0 * cap * (stroke * f) / (STROKE0 * F0)
    return f, shoulder, crank, speed, power, "structure" if cap == SHOULDER_CAP else "crank"


def main():
    print("stroke  L/r  | water Hz  shoulder N m  crank torque  swim m/s  W mech  limited by")
    for deg in (85, 60, 40, 25, 15):
        s = math.radians(deg)
        f, sh, cr, u, p, why = water(s)
        print(f"{deg:5d} {1 / math.sin(s / 2):5.2f} | {f:7.2f} {sh:12.1f} {cr:12.2f}x "
              f"{u:9.2f} {p:7.1f}  {why}")
    lift = MASS * G * THUMB
    print(f"\ncrutch on land: shoulder torque to lift the body {lift:.2f} N m "
          f"({lift / TAU0:.2f} of take-off)")
    for deg, hz in ((30, 1.0), (40, 1.5), (60, 1.5)):
        stride = THUMB * math.radians(deg) * math.sin(TILT)
        print(f"  stroke {deg} deg at {hz} Hz: stride {stride * 100:.1f} cm, "
              f"{stride * hz:.2f} m/s if the thumb does not slip")


if __name__ == "__main__":
    main()
