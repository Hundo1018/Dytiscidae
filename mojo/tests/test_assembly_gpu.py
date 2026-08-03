"""Check the GPU force assembly against the numpy original."""
import numpy as np
import assembly_gpu

from dytiscidae.physics.medium import GRAVITY


def reference(q, area, cl, cd, lift_axis, d_hat, f_bluff, c_rot, rho, U,
              omega, s_hat, chord, dr, volb, subf, fza, is_wing,
              lift_scale, cd_scale, water_rho):
    """fluid.py:494-496, 582-588, 660, 676-677, in source order."""
    w = is_wing.astype(bool)
    L = q * area * cl * lift_scale
    D = q * area * cd * cd_scale
    F = L[:, None] * lift_axis + D[:, None] * d_hat
    F = F + f_bluff
    omega_s = np.einsum("ni,ni->n", omega, s_hat)
    f_rot = np.where(w, c_rot * rho * U * omega_s * chord**2 * dr, 0.0)
    F = F + f_rot[:, None] * lift_axis
    F[:, 2] += fza
    f_buoy = water_rho * GRAVITY * volb * subf
    F[:, 2] += f_buoy
    return F, L, D, f_buoy


def run_gpu(ins, scal):
    n = len(ins[0])
    F = np.zeros((n, 3)); L = np.zeros(n); D = np.zeros(n); B = np.zeros(n)
    desc = np.array([a.ctypes.data for a in ins] +
                    [F.ctypes.data, L.ctypes.data, D.ctypes.data,
                     B.ctypes.data, n], dtype=np.int64)
    got = assembly_gpu.assembly(desc, scal)
    assert got == n, f"kernel reported {got}, expected {n}"
    return F, L, D, B


def case(name, n=200000, seed=0, all_wing=None, dry=False):
    rng = np.random.default_rng(seed)
    C = np.ascontiguousarray
    q = C(rng.uniform(0, 5e4, n))
    area = C(rng.uniform(1e-5, 0.2, n))
    if all_wing is None:
        w = (rng.random(n) < 0.5)
    else:
        w = np.full(n, bool(all_wing))
    is_wing = C(w.astype(np.int32))
    # cl/cd are zero on bluff panels, as the coefficient kernel leaves them
    cl = C(np.where(w, rng.uniform(-2, 2, n), 0.0))
    cd = C(np.where(w, rng.uniform(0, 3, n), 0.0))
    la = rng.normal(size=(n, 3)); la /= np.linalg.norm(la, axis=1, keepdims=True); la = C(la)
    dh = rng.normal(size=(n, 3)); dh /= np.linalg.norm(dh, axis=1, keepdims=True); dh = C(dh)
    fb = C(rng.normal(scale=50.0, size=(n, 3)))
    # pitch_axis in [0,1] gives c_rot spanning both signs about 0.75
    c_rot = C(np.pi * (0.75 - rng.uniform(0.0, 1.0, n)))
    rho = C(np.where(rng.random(n) < 0.5, 1.225, 1025.0))
    U = C(rng.uniform(0, 30, n))
    om = C(rng.normal(scale=8.0, size=(n, 3)))
    s = rng.normal(size=(n, 3)); s /= np.linalg.norm(s, axis=1, keepdims=True); s = C(s)
    chord = C(rng.uniform(0.01, 0.6, n))
    dr = C(rng.uniform(0.005, 0.3, n))
    volb = C(rng.uniform(0, 0.05, n))
    subf = C(np.zeros(n) if dry else rng.uniform(0, 1, n))
    # both saturated ends of the submerged ramp
    if not dry:
        subf[:500] = 0.0
        subf[500:1000] = 1.0
    fza = C(rng.uniform(0, 40, n))

    ins = [q, area, cl, cd, la, dh, fb, c_rot, rho, U, om, s, chord, dr,
           volb, subf, fza, is_wing]
    scal = (1.0, 1.0, 1025.0)
    ref = reference(q, area, cl, cd, la, dh, fb, c_rot, rho, U, om, s, chord,
                    dr, volb, subf, fza, is_wing, *scal)
    got = run_gpu(ins, scal)
    errs = []
    for a, b in zip(ref, got):
        sc = max(float(np.abs(a).max()), 1e-12)
        errs.append(float(np.abs(a - b).max()) / sc)
    print(f"{name:26s} " + " ".join(f"{e:10.3e}" for e in errs))
    return max(errs)


def main():
    print(f"{'case':26s} {'F':>10s} {'L':>10s} {'D':>10s} {'f_buoy':>10s}")
    worst = 0.0
    worst = max(worst, case("mixed wing/bluff", seed=1))
    worst = max(worst, case("all wing", seed=2, all_wing=True))
    worst = max(worst, case("all bluff (no Kramer)", seed=3, all_wing=False))
    worst = max(worst, case("fully dry (no buoyancy)", seed=4, dry=True))
    print()
    ok = worst < 1e-12
    print("GPU assembly checks passed" if ok else f"FAILED (worst {worst:.3e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
