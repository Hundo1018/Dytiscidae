"""L0: two checks on an inflated wing for the reference machine (arithmetic).

1. Buoyancy: displaced volume = flight area x mean thickness, against the weight.
2. Stiffness: a pressurised membrane tube wrinkles at M = pi p r^3 / 2 and
   collapses at about twice that (thin-walled inflated beam); the root bending
   moment it must carry is half the aerodynamic stroke torque of L1a at
   take-off (10.1 N m for both wings), plus inertia.

    python experiments/flapper/L0/inflate.py
"""
import math

G, RHO_W = 9.81, 1000.0
MASS, AREA, LENGTH = 0.60, 0.34, 0.66
M_ROOT = 0.5 * 11.0            # N m a wing: L1a take-off peak drive torque 11.0, both wings


def main():
    W = MASS * G
    print(f"weight {W:.2f} N")
    for t_mm in (2, 5, 10, 20):
        vol = AREA * t_mm / 1000
        print(f"  mean thickness {t_mm:2d} mm: {vol * 1000:.2f} L displaced, "
              f"buoyancy {vol * RHO_W * G:5.1f} N = {vol * RHO_W * G / W:.1f} weights")
    print(f"\nroot moment to carry: {M_ROOT:.1f} N m a wing")
    for bar in (0.5, 1.0, 2.0):
        p = bar * 1e5
        r_w = (2 * M_ROOT / (math.pi * p)) ** (1 / 3)
        vol = math.pi * r_w ** 2 * LENGTH * 2
        print(f"  {bar} bar gauge: tube radius {r_w * 1000:.0f} mm to stay unwrinkled "
              f"(diameter {2 * r_w / 0.257:.0%} of the chord); two such spars "
              f"{vol * 1000:.1f} L, {vol * RHO_W * G / W:.1f} weights of buoyancy")


if __name__ == "__main__":
    main()
