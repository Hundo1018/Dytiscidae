"""Check the GPU bluff-body drag against the numpy original."""
import numpy as np
import fluid_gpu
from dytiscidae.physics.fluid import CD_CROSSFLOW, skin_friction_cd


def reference(v_rel, s_hat, c_hat, n_hat, rho, mu, ext, cd_bluff, is_wing, cd_scale):
    """Exactly fluid.py:533-578."""
    b = ~is_wing.astype(bool)
    U_full = np.linalg.norm(v_rel, axis=1)
    d_full = v_rel / np.maximum(U_full, 1e-6)[:, None]
    ex, ey, ez = ext[:, 0], ext[:, 1], ext[:, 2]
    v_ax = np.einsum("ni,ni->n", v_rel, s_hat)
    v_cross = v_rel - v_ax[:, None] * s_hat
    u_cross = np.linalg.norm(v_cross, axis=1)
    d_cross = v_cross / np.maximum(u_cross, 1e-9)[:, None]
    wetted = 2.0 * (ex * ey + ey * ez + ex * ez)
    re_b = rho * np.abs(v_ax) * ex / np.maximum(mu, 1e-12)
    f_axial = cd_scale * (
        0.5 * rho * np.abs(v_ax) * v_ax
        * (cd_bluff * ey * ez + skin_friction_cd(re_b) * wetted))
    pc = np.abs(np.einsum("ni,ni->n", d_cross, c_hat))
    pn = np.abs(np.einsum("ni,ni->n", d_cross, n_hat))
    a_side = pc * ex * ez + pn * ex * ey
    f_cross = cd_scale * 0.5 * rho * u_cross**2 * CD_CROSSFLOW * a_side
    F_b = f_axial[:, None] * s_hat + f_cross[:, None] * d_cross
    F = np.where(b[:, None], F_b, 0.0)
    D = np.where(b, np.abs(f_axial) + f_cross, 0.0)
    return F, D, d_full


def run_gpu(ins, cd_scale):
    n = len(ins[4])
    F = np.zeros((n, 3)); D = np.zeros(n); DF = np.zeros((n, 3))
    desc = np.array([a.ctypes.data for a in ins] +
                    [F.ctypes.data, D.ctypes.data, DF.ctypes.data, n], dtype=np.int64)
    got = fluid_gpu.bluff_drag(desc, cd_scale)
    assert got == n, f"kernel reported {got}, expected {n}"
    return F, D, DF


def main():
    rng = np.random.default_rng(23)
    n = 200000
    C = np.ascontiguousarray
    v = C(rng.normal(scale=5.0, size=(n, 3)))
    # Seams: dead still (U_safe and u_cross floors), pure axial flow (u_cross=0,
    # so d_cross is the 1e-9 branch), and pure cross flow (v_ax=0).
    v[:200] *= 1e-12
    s = rng.normal(size=(n, 3)); s /= np.linalg.norm(s, axis=1, keepdims=True); s = C(s)
    v[200:400] = 3.0 * s[200:400]                      # purely axial
    c = rng.normal(size=(n, 3))
    c -= (np.einsum("ni,ni->n", c, s))[:, None] * s
    c /= np.linalg.norm(c, axis=1, keepdims=True); c = C(c)
    nh = C(np.cross(s, c))
    v[400:600] = 4.0 * c[400:600]                      # purely cross
    rho = C(np.where(rng.random(n) < 0.5, 1.225, 1025.0))
    mu = C(np.where(rho > 500, 1.08e-3, 1.81e-5))
    ext = C(np.abs(rng.uniform(1e-3, 0.6, (n, 3))))
    cdb = C(rng.choice([0.20, 0.25, 0.9, 1.1, 0.6], n))
    wing = C((rng.random(n) < 0.5).astype(np.int32))
    cd_scale = 1.0

    ins = [v, s, c, nh, rho, mu, ext, cdb, wing]
    rF, rD, rDF = reference(v, s, c, nh, rho, mu, ext, cdb, wing, cd_scale)
    gF, gD, gDF = run_gpu(ins, cd_scale)

    def rel(a, b):
        sc = np.maximum(np.abs(a), 1.0)
        return float((np.abs(a - b) / sc).max())

    eF, eD, eDF = rel(rF, gF), rel(rD, gD), rel(rDF, gDF)
    print(f"{n} panels, half wing / half bluff, incl. still / pure-axial / pure-cross")
    print(f"  F       max relative error {eF:.3e}")
    print(f"  D       max relative error {eD:.3e}")
    print(f"  d_full  max relative error {eDF:.3e}")
    gated = float(np.abs(gF[wing.astype(bool)]).max() + np.abs(gD[wing.astype(bool)]).max())
    print(f"  wing panels contribute exactly zero: {gated:.3e}")
    # d_full must be written for wings too -- added mass reads it downstream.
    wrote = float(np.abs(gDF[wing.astype(bool)]).max())
    print(f"  d_full still written for wing panels: max |.| = {wrote:.3f}")

    ok = eF < 1e-12 and eD < 1e-12 and eDF < 1e-12 and gated == 0.0 and wrote > 0.1
    print()
    print("GPU bluff-body checks passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
