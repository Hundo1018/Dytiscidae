"""Check the GPU lift/drag coefficients against the numpy originals."""
import numpy as np
import fluid_gpu

from dytiscidae.physics.fluid import (
    WING, drag_coefficient, lift_coefficient, skin_friction_cd)


def run_gpu(alpha, re, ar, rf, is_wing):
    n = len(alpha)
    cl = np.zeros(n); cd = np.zeros(n)
    ins = [alpha, re, ar, rf, is_wing]
    desc = np.array([a.ctypes.data for a in ins] +
                    [cl.ctypes.data, cd.ctypes.data, n], dtype=np.int64)
    got = fluid_gpu.coefficients(desc)
    assert got == n, f"kernel reported {got}, expected {n}"
    return cl, cd


def main():
    rng = np.random.default_rng(11)
    n = 200000
    C = np.ascontiguousarray
    # Sweep the whole domain, and deliberately sit on the seams: the stall
    # blend, the laminar/turbulent handover at Re 5e5, the AR and Re clamps,
    # and the +/-1.2*cl_max saturation.
    alpha = C(np.concatenate([
        rng.uniform(-np.pi / 2, np.pi / 2, n - 6),
        np.array([0.0, 11 * np.pi / 180, 37 * np.pi / 180, -np.pi / 2, np.pi / 2, 1e-14]),
    ]))
    re = C(np.concatenate([
        10 ** rng.uniform(0, 8, n - 4),
        np.array([1.0, 10.0, 5e5, 1e8]),
    ]))
    ar = C(np.concatenate([rng.uniform(0.01, 25.0, n - 2), np.array([0.5, 0.0])]))
    rf = C(np.concatenate([rng.uniform(0.0, 1.5, n - 2), np.array([0.0, 0.30])]))
    is_wing = C(np.ones(n, dtype=np.int32))

    cl_ref = lift_coefficient(alpha, re, ar, rf)
    cd_ref = drag_coefficient(alpha, re, ar, cl_ref)
    cl_got, cd_got = run_gpu(alpha, re, ar, rf, is_wing)

    def rel(a, b):
        s = np.maximum(np.abs(a), 1.0)
        return float((np.abs(a - b) / s).max())

    e_cl, e_cd = rel(cl_ref, cl_got), rel(cd_ref, cd_got)
    print(f"{n} samples spanning alpha [-pi/2,pi/2], Re 1..1e8, AR 0..25")
    print(f"  cl max relative error {e_cl:.3e}")
    print(f"  cd max relative error {e_cd:.3e}")

    # skin friction alone, since it carries pow() and the log10 sigmoid
    sf_ref = skin_friction_cd(re)
    print(f"  (skin friction spans {sf_ref.min():.4f}..{sf_ref.max():.4f})")

    # the is_wing gate must zero bluff elements
    bluff = C(np.zeros(n, dtype=np.int32))
    cl_b, cd_b = run_gpu(alpha, re, ar, rf, bluff)
    gated = float(np.abs(cl_b).max() + np.abs(cd_b).max())
    print(f"  bluff elements gated to zero: max |cl|+|cd| = {gated:.3e}")

    ok = e_cl < 1e-12 and e_cd < 1e-12 and gated == 0.0
    print()
    print("GPU coefficient checks passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
