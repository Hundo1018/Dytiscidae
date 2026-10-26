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


def test_the_wagner_lag_is_the_numpy_unsteady_state() -> None:
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("the device Wagner lag and LEV travel agree with UnsteadyState.update", reason)
        return
    import warp as wp

    from dytiscidae.envs.devroll.fluid_wp import WagnerLag
    from dytiscidae.physics.fluid import UnsteadyState

    rng = np.random.default_rng(7)
    n, steps, dt = 60, 400, 0.004
    worst = {"alpha_e": 0.0, "lev": 0.0}
    for dtype, tol in ((wp.float64, 1e-9), (wp.float32, 2e-4)):
        ref = UnsteadyState(n)
        dev = WagnerLag(n, dtype=dtype)
        rng = np.random.default_rng(7)
        rev = np.zeros(n, bool)
        worst = {"alpha_e": 0.0, "lev": 0.0}
        scale = {"alpha_e": 0.0, "lev": 1.0}
        for step in range(steps):
            alpha = rng.uniform(-0.6, 0.6, n)
            flip = rng.uniform(size=n) < 0.02
            rev = rev ^ flip
            U = rng.uniform(0.0, 4.0, n)
            chord = rng.uniform(0.005, 0.05, n)
            om = rng.normal(size=(n, 3)) * 8.0
            s_hat = np.tile([0.0, 1.0, 0.0], (n, 1))
            ae_r, lev_r = ref.update(alpha, rev, U, chord, om, s_hat, dt)
            ae_d, lev_d = dev.update(alpha, rev, U, chord, om, s_hat, dt)
            scale["alpha_e"] = max(scale["alpha_e"], float(np.abs(ae_r).max()))
            worst["alpha_e"] = max(worst["alpha_e"], float(np.abs(ae_d - ae_r).max()))
            worst["lev"] = max(worst["lev"], float(np.abs(lev_d - lev_r).max()))
        name = "float64" if dtype == wp.float64 else "float32"
        rel_ae = worst["alpha_e"] / scale["alpha_e"]
        check(f"{name}: alpha_e over {steps} steps within {tol:g} of its maximum",
              rel_ae <= tol, f"worst {rel_ae:.2e}")
        check(f"{name}: lev over {steps} steps within {tol:g} absolute",
              worst["lev"] <= tol, f"worst {worst['lev']:.2e}")


def added_mass_reference(rel, s, c, n, ext, chord, dr, volume, rho, is_wing, scale):
    """The strip added mass and slam mass of FluidSolver.apply (lines 1316-1374, tensor on)."""
    from dytiscidae.physics.fluid import slam_mass

    vn = np.einsum("ni,ni->n", rel, n)
    e = np.maximum(ext, 1e-4)
    ca_axis = np.clip(0.5 * (e[:, [1, 2, 0]] + e[:, [2, 0, 1]]) / (2.0 * e), 0.05, 10.0)
    d_full = rel / np.maximum(np.linalg.norm(rel, axis=1), 1e-6)[:, None]
    dc = np.stack([np.einsum("ni,ni->n", d_full, s), np.einsum("ni,ni->n", d_full, c),
                   np.einsum("ni,ni->n", d_full, n)], axis=1) ** 2
    moving = dc.sum(axis=1) > 1e-6
    ca_eff = np.einsum("ni,ni->n", dc, ca_axis)
    ca_eff = np.where(moving, ca_eff, ca_axis.mean(axis=1))
    t_wing = np.maximum(ext[:, 2], 1e-5)
    m_axis = (rho * dr)[:, None] * (np.pi * 0.25) * np.stack(
        [np.zeros(len(chord)), t_wing**2, chord**2], axis=1)
    m_wing = np.einsum("ni,ni->n", dc, m_axis)
    m_wing = np.where(moving, m_wing, m_axis.mean(axis=1))
    m_add = np.where(is_wing != 0, m_wing, ca_eff * rho * volume) * scale
    m_slam = slam_mass(m_add, rho, chord, dr, is_wing != 0, scale)
    return {"m_add": m_add, "m_slam": m_slam, "vn": vn}


def test_the_device_medium_and_added_mass_are_the_numpy_ones() -> None:
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("the device medium and added mass agree with MediumField and apply", reason)
        return
    import warp as wp

    from dytiscidae.envs.devroll.fluid_wp import medium_and_added_mass
    from dytiscidae.physics.medium import MediumField

    field = MediumField()
    check("the calm sea is flat (SeaState amplitude 0), which the device medium assumes",
          field.sea_state.amplitude == 0.0)
    rng = np.random.default_rng(3)
    n = 300
    args = _random_strips(n, seed=5)
    rel, s, c, nrm, om, ext, chord, area, dr, cd_bluff, camber, ar, is_wing, rho0, mu0, *_ = args
    pos = rng.uniform(-0.3, 0.3, size=(n, 3))
    pos[:, 2] = rng.uniform(-0.05, 0.05, n)
    half_h = rng.uniform(0.002, 0.03, n)
    wind = rng.normal(size=3) * 2.0
    current = rng.normal(size=3) * 0.5
    volume = ext[:, 0] * ext[:, 1] * ext[:, 2]
    for dtype, tol in ((wp.float64, 1e-9), (wp.float32, 2e-4)):
        name = "float64" if dtype == wp.float64 else "float32"
        out = medium_and_added_mass(pos, half_h, wind, current, rel, s, c, nrm, ext, chord, dr,
                                    volume, is_wing, 1.0, dtype=dtype)
        # The numpy medium takes the same wind and current as the device.
        field.wind = wind
        field.current = current
        rho_ref, mu_ref, subf_ref = field.properties(pos, half_h, 0.0)
        flow_ref = field.flow_velocity(pos, 0.0)
        ref2 = added_mass_reference(rel, s, c, nrm, ext, chord, dr, volume, rho_ref, is_wing, 1.0)
        ref2.update({"rho": rho_ref, "mu": mu_ref, "subf": subf_ref, "flow": flow_ref})
        worst_name, worst = "", 0.0
        for k in ("rho", "mu", "subf", "flow", "m_add", "m_slam", "vn"):
            r = np.asarray(ref2[k])
            scale_k = max(float(np.abs(r).max()), 1e-30)
            rel_err = float(np.abs(out[k] - r).max()) / scale_k
            if rel_err > worst:
                worst_name, worst = k, rel_err
        check(f"{name}: medium, flow and added mass within {tol:g} of their maxima",
              worst <= tol, f"worst {worst_name} at {worst:.2e}")


def test_the_device_assembly_is_the_numpy_finish_bodies() -> None:
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("buoyancy, panel-to-body assembly and finish_bodies agree with apply", reason)
        return
    import warp as wp

    from dytiscidae.envs.devroll.fluid_wp import assemble_bodies
    from dytiscidae.physics.fluid import GRAVITY, finish_bodies
    from dytiscidae.physics.medium import SEAWATER

    rng = np.random.default_rng(11)
    n, nb = 240, 7
    body_id = rng.integers(0, nb, n)
    body_id[:nb] = np.arange(nb)
    pos = rng.uniform(-0.3, 0.3, size=(n, 3))
    xipos = rng.uniform(-0.1, 0.1, size=(nb, 3))
    F = rng.normal(size=(n, 3)) * 5.0
    subf = rng.uniform(0.0, 1.0, n)
    vb = rng.uniform(1e-6, 5e-5, n)
    m_body = rng.uniform(1e-3, 2e-2, nb)

    # numpy reference: apply's buoyancy, accumulation and finish_bodies, verbatim.
    Fr = F.copy()
    Fr[:, 2] += SEAWATER.rho * GRAVITY * vb * subf
    fmag = np.linalg.norm(Fr, axis=1)
    arm = pos - xipos[body_id]
    fb_ref = np.zeros((nb, 6))
    np.add.at(fb_ref[:, :3], body_id, Fr)
    np.add.at(fb_ref[:, 3:], body_id, np.cross(arm, Fr))
    fsum = np.zeros(nb)
    np.add.at(fsum, body_id, fmag)
    total = float(fsum.sum())

    for case, limit, expect_clamp in (("unclamped", 1e9, False),
                                      ("clamped", 0.3 * total, True)):
        fb = fb_ref.copy()
        clamped_ref = finish_bodies(fb, fsum, m_body, limit)
        check(f"{case}: the limiter binds on this draw as designed", clamped_ref == expect_clamp)
        for dtype, tol in ((wp.float64, 1e-9), (wp.float32, 2e-4)):
            name = "float64" if dtype == wp.float64 else "float32"
            out = assemble_bodies(pos, F, body_id, xipos, subf, vb, SEAWATER.rho, m_body,
                                  limit, dtype=dtype)
            scale_f = max(float(np.abs(Fr).max()), 1e-30)
            scale_fb = max(float(np.abs(fb).max()), 1e-30)
            err_F = float(np.abs(out["F"] - Fr).max()) / scale_f
            err_fb = float(np.abs(out["fb"] - fb).max()) / scale_fb
            check(f"{case} {name}: panel force within {tol:g} of its maximum",
                  err_F <= tol, f"{err_F:.2e}")
            check(f"{case} {name}: body wrench within {tol:g} of its maximum",
                  err_fb <= tol, f"{err_fb:.2e}")
            check(f"{case} {name}: limiter flag agrees with finish_bodies",
                  out["clamped"] == clamped_ref)


def test_the_device_entrainment_is_the_numpy_entrainment_reaction() -> None:
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("the entrainment reaction agrees with physics.fluid.entrainment_reaction", reason)
        return
    import warp as wp

    from dytiscidae.envs.devroll.fluid_wp import entrain_bodies
    from dytiscidae.physics.fluid import entrainment_reaction

    rng = np.random.default_rng(23)
    nb = 9
    m_old = rng.uniform(1e-3, 2e-2, nb)
    # Growth on some bodies, shedding on others: the one-sided max must drop the shedding.
    m_new = m_old + rng.normal(0.0, 3e-3, nb)
    lever2 = rng.uniform(1e-4, 1e-2, nb)
    vel = rng.normal(size=(nb, 6)) * 2.0
    fb = rng.normal(size=(nb, 6))
    h = 0.004

    ref = fb.copy()
    entrainment_reaction(ref, m_new, m_old, lever2, vel, h)
    check("the draw has both growth and shedding", bool(np.any(m_new > m_old))
          and bool(np.any(m_new < m_old)))
    for dtype, tol in ((wp.float64, 1e-9), (wp.float32, 2e-4)):
        name = "float64" if dtype == wp.float64 else "float32"
        out = entrain_bodies(fb, m_new, m_old, lever2, vel, h, dtype=dtype)
        scale = max(float(np.abs(ref).max()), 1e-30)
        err = float(np.abs(out - ref).max()) / scale
        check(f"{name}: entrainment wrench within {tol:g} of its maximum",
              err <= tol, f"{err:.2e}")


def test_the_device_replay_is_not_built_yet() -> None:
    skip("the device fluid replays FluidSolver.apply for 2000 steps (A1a replay)",
         "the Wagner lag, inflow, medium and added mass are A1a and A1b slices, not built")


def main() -> int:
    functions = [test_the_device_path_reports_why_it_cannot_run,
                 test_the_strip_kernel_is_the_numpy_strip_loads,
                 test_the_wagner_lag_is_the_numpy_unsteady_state,
                 test_the_device_medium_and_added_mass_are_the_numpy_ones,
                 test_the_device_assembly_is_the_numpy_finish_bodies,
                 test_the_device_entrainment_is_the_numpy_entrainment_reaction,
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
