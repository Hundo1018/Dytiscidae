"""Check the GPU added mass and its atomic scatter against numpy."""
import numpy as np
import fluid_gpu
from dytiscidae.physics.medium import GRAVITY


def reference(v_rel, s_hat, c_hat, n_hat, d_full, rho, ext, chord, dr, volume,
              is_wing, body_id, nbody, scale, has_bluff):
    """Exactly fluid.py:627-661."""
    vn = np.einsum("ni,ni->n", v_rel, n_hat)
    e = np.maximum(ext, 1e-4)
    ca_axis = np.clip(0.5 * (e[:, [1, 2, 0]] + e[:, [2, 0, 1]]) / (2.0 * e), 0.05, 10.0)
    if has_bluff:
        dc = np.stack([
            np.einsum("ni,ni->n", d_full, s_hat),
            np.einsum("ni,ni->n", d_full, c_hat),
            np.einsum("ni,ni->n", d_full, n_hat),
        ], axis=1) ** 2
    else:
        dc = np.zeros((len(vn), 3))
    ca_eff = np.einsum("ni,ni->n", dc, ca_axis)
    ca_eff = np.where(dc.sum(axis=1) > 1e-6, ca_eff, ca_axis.mean(axis=1))
    m_add = np.where(is_wing.astype(bool),
                     rho * np.pi * chord**2 * 0.25 * dr,
                     ca_eff * rho * volume) * scale
    # m_body is no longer this kernel's job: the per-body sum moved to
    # scatter_gpu.gather_body_kernel, which forms it in panel-index order so
    # the result does not depend on warp arrival.  The atomic version made a
    # whole evaluation irreproducible.  Kept here as a reference only so the
    # shape of the test is unchanged; the kernel writes zeros.
    m_body = np.zeros(nbody)
    return m_add, vn, m_body, m_add * GRAVITY


def run_gpu(ins, nbody, scale, has_bluff):
    n = len(ins[5])
    ma = np.zeros(n); vn = np.zeros(n); mb = np.zeros(nbody); fz = np.zeros(n)
    desc = np.array([a.ctypes.data for a in ins] +
                    [ma.ctypes.data, vn.ctypes.data, mb.ctypes.data,
                     fz.ctypes.data, n, nbody], dtype=np.int64)
    got = fluid_gpu.added_mass(desc, scale, 1 if has_bluff else 0)
    assert got == n, f"kernel reported {got}, expected {n}"
    return ma, vn, mb, fz


def case(n, nbody, has_bluff, seed, all_wing=False, still=False):
    rng = np.random.default_rng(seed)
    C = np.ascontiguousarray
    v = C(rng.normal(scale=4.0, size=(n, 3)))
    s = rng.normal(size=(n, 3)); s /= np.linalg.norm(s, axis=1, keepdims=True); s = C(s)
    c = rng.normal(size=(n, 3))
    c -= np.einsum("ni,ni->n", c, s)[:, None] * s
    c /= np.linalg.norm(c, axis=1, keepdims=True); c = C(c)
    nh = C(np.cross(s, c))
    df = C(v / np.maximum(np.linalg.norm(v, axis=1), 1e-6)[:, None])
    if still:
        df = C(np.zeros((n, 3)))          # forces the isotropic-mean fallback
    rho = C(np.where(rng.random(n) < 0.5, 1.225, 1025.0))
    # extents spanning the 1e-4 floor and the 0.05/10.0 clip on both sides
    ext = C(10.0 ** rng.uniform(-6, 0.5, (n, 3)))
    chord = C(rng.uniform(0.01, 0.5, n)); dr = C(rng.uniform(0.005, 0.2, n))
    vol = C(rng.uniform(1e-6, 0.05, n))
    wing = C((np.ones(n) if all_wing else (rng.random(n) < 0.5)).astype(np.int32))
    bid = C(rng.integers(0, nbody, n).astype(np.int32))
    ins = [v, s, c, nh, df, rho, ext, chord, dr, vol, wing, bid]
    ref = reference(v, s, c, nh, df, rho, ext, chord, dr, vol, wing, bid,
                    nbody, 1.0, has_bluff)
    got = run_gpu(ins, nbody, 1.0, has_bluff)
    errs = []
    for r, g in zip(ref, got):
        # Normalise by the array's own scale, not element by element.  `vn` is a
        # dot product that legitimately passes through zero -- the flow is
        # perpendicular to the panel normal -- and per-element normalisation
        # divides a last-bit difference by an arbitrarily small number.  It
        # reported 1.3e-11 on one sample where |vn| was 2.4e-5 and the absolute
        # disagreement was 3.2e-16, which says nothing about the kernel.
        # Measured against the scale of the quantity, that same sample is
        # 2.1e-16.
        sc = max(float(np.abs(r).max()), 1e-12)
        errs.append(float((np.abs(r - g)).max() / sc))
    return errs


def main():
    print(f"{'case':28s} {'m_add':>10s} {'vn':>10s} {'m_body':>10s} {'F_z':>10s}"
          "   (m_body now zero here; see gather_body_kernel)")
    cases = [
        ("mixed, bluff present", dict(n=200000, nbody=64, has_bluff=True, seed=1)),
        ("all wing", dict(n=100000, nbody=32, has_bluff=True, seed=2, all_wing=True)),
        ("no bluff anywhere", dict(n=100000, nbody=32, has_bluff=False, seed=3)),
        ("at rest (isotropic fallback)", dict(n=100000, nbody=32, has_bluff=True, seed=4, still=True)),
        ("heavy scatter collision", dict(n=200000, nbody=3, has_bluff=True, seed=5)),
    ]
    worst = 0.0
    for name, kw in cases:
        e = case(**kw)
        worst = max(worst, max(e))
        print(f"{name:28s} " + " ".join(f"{x:10.3e}" for x in e))
    print()
    ok = worst < 1e-13
    print("GPU added-mass checks passed" if ok else f"FAILED (worst {worst:.3e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
