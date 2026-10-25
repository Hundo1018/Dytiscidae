"""Device evaluation path (IMPL_1010 A): the guard, and the first fluid kernel.

The device path is optional.  The guard checks run wherever the project runs.
The kernel checks need MuJoCo Warp, warp and a CUDA device, and skip with a
reason otherwise; a skip is never printed as a pass.

Run with:  PYTHONPATH=. .venv/bin/python tests/test_devroll.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES: list[str] = []
SKIPPED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def skip(name: str, reason: str) -> None:
    print(f"  [skip] {name}  -- {reason}")
    SKIPPED.append(name)


def test_the_device_path_reports_why_it_cannot_run() -> None:
    from dytiscidae.envs import devroll

    check("AVAILABLE is a bool", isinstance(devroll.AVAILABLE, bool))
    check("UNAVAILABLE_REASON is a str", isinstance(devroll.UNAVAILABLE_REASON, str))
    ok, reason = devroll.usable()
    check("usable() returns (bool, str)", isinstance(ok, bool) and isinstance(reason, str))
    if not devroll.AVAILABLE:
        check("MJWarp not importable: the reason says so",
              devroll.UNAVAILABLE_REASON == "MJWarp not importable",
              devroll.UNAVAILABLE_REASON)
        check("MJWarp not importable: usable() refuses with that reason",
              ok is False and reason == "MJWarp not importable", reason)
    elif not ok:
        check("MJWarp importable but unusable: reason is 'MJWarp not usable: <why>'",
              reason.startswith("MJWarp not usable: ") and len(reason) > len("MJWarp not usable: "),
              reason)
    else:
        check("usable() passes only with an empty reason", reason == "")


def strip_reference(rel, s, c, n, om, ext, chord, area, dr, cd_bluff, camber, ar, is_wing,
                    rho, mu, lift_scale, cd_scale, c_rot):
    """Quasi-static strip loads in numpy: FluidSolver.apply's lines for the same quantities.

    The Wagner lag, the inflow model, the medium and the added mass are not in it;
    they are later slices of A1a and A1b.
    """
    from dytiscidae.physics.fluid import (CD_CROSSFLOW, drag_coefficient, lift_coefficient,
                                          rossby_lev, skin_friction_cd)

    def dot(a, b):
        return np.einsum("ni,ni->n", a, b)

    v_2d = rel - dot(rel, s)[:, None] * s
    U = np.linalg.norm(v_2d, axis=1)
    Us = np.maximum(U, 1e-6)
    d_hat = v_2d / Us[:, None]
    q = 0.5 * rho * U**2
    re = rho * U * chord / np.maximum(mu, 1e-12)
    cos_a = dot(v_2d, c) / Us
    sin_a = dot(v_2d, n) / Us
    rev = cos_a < 0.0
    alpha = np.arctan2(sin_a, np.where(rev, -cos_a, cos_a) + 1e-12)
    alpha = np.where(rev, -alpha, alpha)
    alpha = alpha + 2.0 * np.where(rev, -camber, camber)
    omega_s = dot(om, s)
    lev = rossby_lev(U, chord, om, s)
    lift_axis = np.cross(s, d_hat)
    lift_axis /= np.maximum(np.linalg.norm(lift_axis, axis=1, keepdims=True), 1e-12)
    is_w = is_wing != 0
    F = np.zeros((len(chord), 3))
    cl = np.where(is_w, lift_coefficient(alpha, re, ar, lev, alpha), 0.0)
    cd = np.where(is_w, drag_coefficient(alpha, re, ar, cl, lev), 0.0)
    L = q * area * cl * lift_scale
    D = q * area * cd * cd_scale
    F += L[:, None] * lift_axis + D[:, None] * d_hat
    b = ~is_w
    ex, ey, ez = ext[:, 0], ext[:, 1], ext[:, 2]
    v_ax = dot(rel, s)
    v_cross = rel - v_ax[:, None] * s
    u_cross = np.linalg.norm(v_cross, axis=1)
    d_cross = v_cross / np.maximum(u_cross, 1e-9)[:, None]
    wetted = 2.0 * (ex * ey + ey * ez + ex * ez)
    re_b = rho * np.abs(v_ax) * ex / np.maximum(mu, 1e-12)
    f_axial = cd_scale * (0.5 * rho * np.abs(v_ax) * v_ax
                          * (cd_bluff * ey * ez + skin_friction_cd(re_b) * wetted))
    pc = np.abs(dot(d_cross, c))
    pn = np.abs(dot(d_cross, n))
    a_side = pc * ex * ez + pn * ex * ey
    f_cross = cd_scale * 0.5 * rho * u_cross**2 * CD_CROSSFLOW * a_side
    F_b = f_axial[:, None] * s + f_cross[:, None] * d_cross
    F += np.where(b[:, None], F_b, 0.0)
    D = D + np.where(b, np.abs(f_axial) + f_cross, 0.0)
    f_rot = np.where(is_w, -c_rot * rho * U * omega_s * chord**2 * dr, 0.0)
    F += f_rot[:, None] * lift_axis
    return {"F": F, "D": D, "cl": cl, "alpha": alpha, "U": U, "lev": lev}


def _random_strips(n: int, seed: int):
    rng = np.random.default_rng(seed)

    def unit(a):
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    s = unit(rng.normal(size=(n, 3)))
    c = unit(rng.normal(size=(n, 3)))
    c = unit(c - np.einsum("ni,ni->n", c, s)[:, None] * s)
    nrm = unit(np.cross(s, c))
    rel = rng.normal(size=(n, 3)) * 3.0
    om = rng.normal(size=(n, 3)) * 10.0
    ext = rng.uniform(0.01, 0.2, size=(n, 3))
    chord = rng.uniform(0.005, 0.05, n)
    area = chord * rng.uniform(0.01, 0.1, n)
    dr = rng.uniform(0.01, 0.1, n)
    cd_bluff = rng.uniform(0.5, 1.2, n)
    camber = rng.uniform(0.0, 0.06, n)
    ar = rng.uniform(1.0, 8.0, n)
    is_wing = (rng.uniform(size=n) < 0.5).astype(int)
    rho = np.full(n, 1.2)
    mu = np.full(n, 1.8e-5)
    return (rel, s, c, nrm, om, ext, chord, area, dr, cd_bluff, camber, ar, is_wing, rho, mu,
            1.0, 1.0, 0.8)


def test_the_strip_kernel_is_the_numpy_strip_loads() -> None:
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("the strip kernel agrees with the numpy strip loads", reason)
        return
    import warp as wp

    from dytiscidae.envs.devroll.fluid_wp import strip_loads
    from dytiscidae.physics.fluid import LEV_ROSSBY

    check("the Rossby bounds the kernel hard-codes are LEV_ROSSBY",
          tuple(LEV_ROSSBY) == (3.0, 8.0), str(LEV_ROSSBY))
    args = _random_strips(300, seed=1)
    ref = strip_reference(*args)
    scale = {k: max(float(np.abs(ref[k]).max()), 1e-30) for k in ref}
    for dtype, tol in ((wp.float64, 1e-9), (wp.float32, 2e-4)):
        name = "float64" if dtype == wp.float64 else "float32"
        out = strip_loads(*args, dtype=dtype)
        worst = {k: float(np.abs(out[k] - ref[k]).max()) / scale[k] for k in ref}
        bad = {k: v for k, v in worst.items() if v > tol}
        worst_k = max(worst, key=worst.get)
        check(f"{name}: every strip output within {tol:g} of its maximum",
              not bad, f"worst {worst_k} at {worst[worst_k]:.2e}")
    rel, s, c = args[0], args[1], args[2]
    v_2d = rel - np.einsum("ni,ni->n", rel, s)[:, None] * s
    cos_a = np.einsum("ni,ni->n", v_2d, c) / np.maximum(np.linalg.norm(v_2d, axis=1), 1e-6)
    near = int(np.sum(np.abs(cos_a) < 1e-6))
    check("no panel sits on the incidence fold (|cos a| < 1e-6) in this draw",
          near == 0, f"{near} of {len(cos_a)}")


def test_the_device_replay_is_not_built_yet() -> None:
    skip("the device fluid replays FluidSolver.apply for 2000 steps (A1a replay)",
         "the Wagner lag, inflow, medium and added mass are A1a and A1b slices, not built")


def main() -> int:
    functions = [test_the_device_path_reports_why_it_cannot_run,
                 test_the_strip_kernel_is_the_numpy_strip_loads,
                 test_the_device_replay_is_not_built_yet]
    for fn in functions:
        print(f"{fn.__name__}")
        fn()
    print("\n" + "=" * 68)
    if SKIPPED:
        print(f"{len(SKIPPED)} SKIPPED: not a defect; the device path is not built yet")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all device-path checks passed" + (f", {len(SKIPPED)} skipped" if SKIPPED else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
