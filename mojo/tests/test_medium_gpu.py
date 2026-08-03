"""Check the GPU medium against MediumField itself.

The reference is the real object, not a re-implementation, so a shared
misreading of the source cannot pass.
"""
import numpy as np
import medium_gpu

from dytiscidae.physics.medium import MediumField, SeaState


def run_gpu(pos, hh, med, t):
    n = len(hh)
    rho = np.zeros(n); mu = np.zeros(n); sf = np.zeros(n); uf = np.zeros((n, 3))
    desc = np.array([a.ctypes.data for a in (pos, hh, rho, mu, sf, uf)] + [n],
                    dtype=np.int64)
    s = med.sea_state
    medium_gpu.medium(desc, (
        s.amplitude, s.wavelength, s.period,
        float(np.cos(s.direction)), float(np.sin(s.direction)), t,
        med.air.rho, med.air.mu, med.water.rho, med.water.mu,
        float(med.current[0]), float(med.current[1]), float(med.current[2]),
        float(med.wind[0]), float(med.wind[1]), float(med.wind[2])))
    return rho, mu, sf, uf


def case(name, med, t, n=200000, seed=0, flat=False):
    rng = np.random.default_rng(seed)
    C = np.ascontiguousarray
    pos = C(np.column_stack([
        rng.uniform(-60, 60, n), rng.uniform(-60, 60, n),
        rng.uniform(-14, 6, n)]))
    # Sit exactly on the surface, far above, far below, and at the half-height
    # floor -- the ramp saturates at both ends and the 1e-3 clamp only bites for
    # a panel edge-on to the water.
    pos[:200, 2] = 0.0
    pos[200:400, 2] = 40.0
    pos[400:600, 2] = -40.0
    hh = C(10.0 ** rng.uniform(-6, 0.0, n))
    hh[:100] = 1e-9
    hh[100:200] = 1e-3

    rho, mu, sf, uf = run_gpu(pos, hh, med, t)
    r_rho, r_mu, r_sf = med.properties(pos, hh, t)
    r_uf = med.flow_velocity(pos, t)

    # Two numbers per output, not one.  The tail is dominated by an effect that
    # is real, inherent and physically irrelevant: numpy projects positions onto
    # the wave direction with `xy @ khat`, which goes through BLAS and differs
    # from explicit `x*kx + y*ky` by up to 1.4e-14 (measured).  The submerged
    # fraction then divides that by 2*half_height, and half_height is floored at
    # 1e-3, so a last-bit disagreement is amplified 500x.  Six panels in 200000
    # exceed 1e-13 that way.
    #
    # A bulk quantile catches a real transcription error, which would move every
    # panel; the max catches a gross one without failing on the tail.
    errs, tails = [], []
    for a, b in ((r_rho, rho), (r_mu, mu), (r_sf, sf), (r_uf, uf)):
        sc = max(float(np.abs(a).max()), 1e-12)
        rel = np.abs(a - b) / sc
        errs.append(float(np.percentile(rel, 99.9)))
        tails.append(float(rel.max()))
    print(f"{name:26s} " + " ".join(f"{e:10.3e}" for e in errs)
          + f" | tail {max(tails):8.2e}")
    return max(errs), max(tails)


def main():
    print(f"{'case':26s} {'rho':>10s} {'mu':>10s} {'subf':>10s} {'u_flow':>10s}"
          "   (p99.9 relative)")
    worst = 0.0
    tail = 0.0
    def take(r):
        nonlocal worst, tail
        worst = max(worst, r[0]); tail = max(tail, r[1])
    take(case("waves, t=0", MediumField(), 0.0, seed=1))
    take(case("waves, t=3.7", MediumField(), 3.7, seed=2))
    calm = MediumField(sea_state=SeaState(amplitude=0.0))
    take(case("flat sea (amplitude 0)", calm, 1.5, seed=3))
    windy = MediumField(sea_state=SeaState(amplitude=1.2, wavelength=9.0,
                                           period=3.1, direction=0.8))
    windy.wind = np.array([7.0, -2.0, 0.3])
    windy.current = np.array([-0.4, 0.9, 0.0])
    take(case("wind + current + waves", windy, 2.2, seed=4))

    print()
    ok = worst < 1e-14 and tail < 1e-11
    print(f"bulk (p99.9) {worst:.3e} < 1e-14, tail {tail:.3e} < 1e-11")
    print("GPU medium checks passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
