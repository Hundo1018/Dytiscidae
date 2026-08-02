"""Check the GPU strip theory against the numpy block it replaces."""
import numpy as np
import fluid_gpu

from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import TriphibianEnv


def _cross3(a, b):
    return np.cross(a, b)


def reference(v_rel, s_hat, c_hat, n_hat, omega, rho, mu, chord, camber):
    """Exactly fluid.py:462-487."""
    v_span = np.einsum("ni,ni->n", v_rel, s_hat)[:, None] * s_hat
    v_2d = v_rel - v_span
    U = np.linalg.norm(v_2d, axis=1)
    U_safe = np.maximum(U, 1e-6)
    d_hat = v_2d / U_safe[:, None]
    q = 0.5 * rho * U**2
    re = rho * U * chord / np.maximum(mu, 1e-12)
    cos_a = np.einsum("ni,ni->n", v_2d, c_hat) / U_safe
    sin_a = np.einsum("ni,ni->n", v_2d, n_hat) / U_safe
    alpha = np.arctan2(sin_a, cos_a)
    alpha = np.arctan2(np.sin(alpha), np.abs(np.cos(alpha)) + 1e-12)
    alpha = alpha + 2.0 * camber
    omega_s = np.einsum("ni,ni->n", omega, s_hat)
    reduced_freq = np.abs(omega_s) * chord / (2.0 * U_safe)
    lift_axis = _cross3(s_hat, d_hat)
    lift_axis = lift_axis / np.maximum(
        np.linalg.norm(lift_axis, axis=1, keepdims=True), 1e-12)
    return U, d_hat, q, re, alpha, reduced_freq, lift_axis


def run_gpu(ins):
    n = len(ins[5])
    outs = [np.zeros(n), np.zeros((n, 3)), np.zeros(n), np.zeros(n),
            np.zeros(n), np.zeros(n), np.zeros((n, 3))]
    desc = np.array([a.ctypes.data for a in ins] +
                    [a.ctypes.data for a in outs] + [n], dtype=np.int64)
    got = fluid_gpu.strip_theory(desc)
    assert got == n, f"kernel reported {got}, expected {n}"
    return outs


def main():
    rng = np.random.default_rng(7)
    names = ["U", "d_hat", "q", "re", "alpha", "reduced_freq", "lift_axis"]
    print(f"{'plan':10s} {'panels':>7s} " + " ".join(f"{n:>11s}" for n in names))
    failures = []
    for nm, plan in BODY_PLANS.items():
        env = TriphibianEnv(build(plan()))
        p = env.solver.panels
        n = len(p.body_id)
        C = np.ascontiguousarray
        # Realistic magnitudes, and a few panels forced to near-zero flow so
        # the U_safe and lift_axis guards are actually exercised.
        v_rel = C(rng.normal(scale=4.0, size=(n, 3)))
        v_rel[: max(n // 20, 1)] *= 1e-9
        s = C(rng.normal(size=(n, 3))); s /= np.linalg.norm(s, axis=1, keepdims=True)
        c = C(rng.normal(size=(n, 3))); c /= np.linalg.norm(c, axis=1, keepdims=True)
        nh = C(np.cross(s, c))
        nh /= np.maximum(np.linalg.norm(nh, axis=1, keepdims=True), 1e-12)
        om = C(rng.normal(scale=8.0, size=(n, 3)))
        rho = C(np.where(rng.random(n) < 0.5, 1.225, 1025.0))
        mu = C(np.where(rho > 500, 1.08e-3, 1.81e-5))
        chord = C(np.asarray(p.chord, dtype=np.float64))
        camber = C(np.asarray(p.camber, dtype=np.float64))

        ins = [v_rel, s, c, nh, om, rho, mu, chord, camber]
        ref = reference(*ins)
        got = run_gpu(ins)
        errs = []
        for r, g in zip(ref, got):
            scale = max(float(np.abs(r).max()), 1.0)
            errs.append(float(np.abs(r - g).max()) / scale)
        if max(errs) > 1e-12:
            failures.append(nm)
        print(f"{nm:10s} {n:7d} " + " ".join(f"{e:11.3e}" for e in errs))
    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("all GPU strip-theory checks passed (relative error < 1e-12)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
