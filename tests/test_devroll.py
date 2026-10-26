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


def test_the_device_rotors_are_the_rotor_batch() -> None:
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("the device rotors agree with RotorBatch.apply", reason)
        return
    import math

    import mujoco
    import warp as wp

    from dytiscidae.core.bodyplans import REFERENCE_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.devroll.scatter_wp import DeviceRotors
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    from dytiscidae.physics.rotor import RotorBatch, RotorSet, RotorSpec

    rng = np.random.default_rng(13)

    def machine(k0):
        # test_physics' rotor fixture: four different rotors per machine, so a
        # lookup that reads one rotor's table for another's cannot agree.
        env = TriphibianEnv(build(REFERENCE_PLANS["quad"]()), seed=0)
        env.reset(Domain.WATER, randomise=False)
        names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, b)
                 for b in env.rotors.body]
        specs = {nm: RotorSpec(radius=0.08 + 0.03 * ((j + k0) % 4),
                               pitch=0.06 + 0.025 * ((j + 2 * k0) % 3),
                               blades=2 + (j % 2), handed=(1 if j % 2 else -1))
                 for j, nm in enumerate(names)}
        env.rotors = RotorSet(env.model, specs)
        return env

    def place(env, depth_of_first):
        d, m = env.data, env.model
        d.qpos[3:7] = [math.cos(0.2), math.sin(0.2), 0.0, 0.0]
        mujoco.mj_forward(m, d)
        z = d.xipos[env.rotors.body, 2]
        d.qpos[2] += -depth_of_first - z[0]
        d.qvel[:] = rng.normal(0.0, 2.0, m.nv)
        for j, k in enumerate(env.rotors.dof):
            # idle, reversed, forward, forward
            d.qvel[k] = (0.0, -450.0, 700.0, 300.0)[j % 4] * (1.0 + 0.1 * rng.random())
        mujoco.mj_forward(m, d)

    # Two machines laid out as two worlds of one batch.
    envs = [machine(0), machine(1)]
    check("the calm sea is flat (SeaState amplitude 0), which the device medium assumes",
          all(e.medium.sea_state.amplitude == 0.0 for e in envs))
    med = envs[0].medium
    keys = ("f_lin", "f_ang", "dq", "damp", "thrust", "torque")
    refs, inputs, fracs = [], [], []
    for depth in (0.004, -0.5, 0.3, 0.0):
        for e in envs:
            place(e, depth)
        x = {"xmat": [], "omega": [], "cvel": [], "xipos": [], "com": [], "dt": []}
        for e in envs:
            s, d = e.rotors, e.data
            x["xmat"].append(d.xmat[s.body_arr])
            x["omega"].append(d.qvel[s.dof_arr])
            x["cvel"].append(d.cvel[s.body_arr])
            x["xipos"].append(d.xipos[s.body_arr])
            x["com"].append(d.subtree_com[s.root_arr])
            x["dt"].append(np.full(s.n, float(e.model.opt.timestep)))
        inputs.append({k: np.concatenate(v) for k, v in x.items()})
        # The reference: RotorBatch.apply into zeroed targets, read back per rotor.
        saved = []
        for e in envs:
            s = e.rotors
            saved.append(e.model.dof_damping[s.dof_arr].copy())
            e.data.xfrc_applied[:] = 0.0
            e.data.qfrc_applied[:] = 0.0
            e.model.dof_damping[s.dof_arr] = 0.0
        RotorBatch([e.rotors for e in envs]).apply(
            [(i, e.model, e.data, e.medium) for i, e in enumerate(envs)], envs[0].data.time)
        r = {k: [] for k in keys}
        for e, dmp in zip(envs, saved):
            s = e.rotors
            r["f_lin"].append(e.data.xfrc_applied[s.body_arr, :3].copy())
            r["f_ang"].append(e.data.xfrc_applied[s.body_arr, 3:].copy())
            r["dq"].append(e.model.dof_damping[s.dof_arr].copy())
            r["damp"].append(e.data.qfrc_applied[s.dof_arr].copy())
            r["thrust"].append(s.last_thrust.copy())
            r["torque"].append(s.last_torque.copy())
            e.model.dof_damping[s.dof_arr] = dmp
        refs.append({k: np.concatenate(v) for k, v in r.items()})
        fracs += [float(f) for e in envs for f in e.medium.properties(
            e.data.xipos[e.rotors.body], np.full(e.rotors.n, 0.01), e.data.time)[2]]
    fracs = np.array(fracs)
    omegas = np.concatenate([x["omega"] for x in inputs])
    check("the fixture has rotors in air, in water and across the surface",
          bool((fracs == 0).any() and (fracs == 1).any() and ((fracs > 0) & (fracs < 1)).any()),
          f"submerged fractions {np.round(np.unique(fracs), 3).tolist()}")
    check("the fixture has idle, reversed and forward rotors",
          bool((np.abs(omegas) < 1e-6).any() and (omegas < -1e-6).any() and (omegas > 1e-6).any()))
    check("and the rotors make force: the comparison is not of zeros",
          max(float(np.abs(r["thrust"]).max()) for r in refs) > 0.1)
    for dtype, tol in ((wp.float64, 1e-9), (wp.float32, 2e-4)):
        name = "float64" if dtype == wp.float64 else "float32"
        dev = DeviceRotors([e.rotors for e in envs], med.air, med.water, dtype=dtype)
        worst_name, worst = "", 0.0
        for x, ref in zip(inputs, refs):
            out = dev.loads(x["xmat"], x["omega"], x["cvel"], x["xipos"], x["com"], x["dt"],
                            med.wind, med.current)
            for k in keys:
                scale_k = max(float(np.abs(ref[k]).max()), 1e-30)
                err = float(np.abs(out[k] - ref[k]).max()) / scale_k
                if err > worst:
                    worst_name, worst = k, err
        check(f"{name}: rotor force, torque, damping split, thrust and torque within {tol:g} "
              "of their maxima", worst <= tol, f"worst {worst_name} at {worst:.2e}")


def test_the_device_power_budget_is_the_power_budget() -> None:
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("the device power budget agrees with PowerBudget.step", reason)
        return
    import copy

    import warp as wp

    from dytiscidae.envs.devroll.scatter_wp import DevicePower
    from dytiscidae.envs.triphibian import TriphibianEnv

    envs = [TriphibianEnv(build_plan) for build_plan in _power_plans()]
    envs[1].budget.battery.energy_j = 20.0          # goes flat mid-fixture
    e0 = [e.budget.battery.energy_j for e in envs]
    steps = 50
    rng = np.random.default_rng(7)
    draws = []
    for _ in range(steps):
        f, v = [], []
        for e in envs:
            ff = rng.normal(0.0, 3.0, e.model.nu)
            vv = rng.normal(0.0, 20.0, e.model.nu)
            vv[rng.uniform(size=e.model.nu) < 0.2] = 0.0      # shafts at rest
            if e is envs[2]:
                ff *= 12.0                                    # past the pack's rating
            f.append(ff)
            v.append(vv)
        draws.append((f, v))
    ref = [copy.deepcopy(e.budget) for e in envs]
    alive = [True] * len(envs)
    for f, v in draws:
        for i, r in enumerate(ref):
            if alive[i]:
                alive[i] = r.step(np.abs(f[i]), np.abs(v[i]), envs[i].timestep)
    check("the fixture draws real power, overdraws one pack, and has shafts at rest",
          all(r.actuator_j > 1.0 for r in ref) and any(r.battery.overdraw_time > 0 for r in ref)
          and any((vv == 0).any() for _, v in draws for vv in v),
          f"actuator_j {[round(r.actuator_j, 2) for r in ref]}, overdraw "
          f"{[r.battery.overdraw_time for r in ref]}")
    check("one pack goes flat and the others do not", alive == [True, False, True], str(alive))
    want = {"energy_j": [r.battery.energy_j for r in ref], "total_j": [r.total_j for r in ref],
            "actuator_j": [r.actuator_j for r in ref], "avionics_j": [r.avionics_j for r in ref],
            "t": [r.t for r in ref], "max_overload": [r.max_overload for r in ref],
            "peak_draw": [r.battery.peak_draw for r in ref],
            "overdraw_time": [r.battery.overdraw_time for r in ref]}
    want = {k: np.array(v) for k, v in want.items()}
    for dtype, tol in ((wp.float64, 1e-9), (wp.float32, 2e-4)):
        name = "float64" if dtype == wp.float64 else "float32"
        dev = DevicePower(envs, dtype=dtype)
        for f, v in draws:
            dev.step(np.concatenate([ff[:k] for ff, k in zip(f, dev.n)]),
                     np.concatenate([vv[:k] for vv, k in zip(v, dev.n)]))
        got = dev.state()
        worst_name, worst = "", 0.0
        for k, w in want.items():
            err = float(np.abs(got[k] - w).max()) / max(float(np.abs(w).max()), 1e-30)
            if err > worst:
                worst_name, worst = k, err
        check(f"{name}: battery energy, ledger and max_overload within {tol:g} of their maxima",
              worst <= tol, f"worst {worst_name} at {worst:.2e}")
        check(f"{name}: the flat pack leaves the batch as it left the loop",
              [bool(a) for a in got["active"]] == alive, str(list(got["active"])))
        if dtype == wp.float64:
            used_ref = np.array(e0) - want["energy_j"]
            used = np.array(e0) - got["energy_j"]
            err = float(np.abs(used - used_ref).max()) / float(np.abs(used_ref).max())
            check(f"{name}: the energy drawn (not only the energy left) within {tol:g}",
                  err <= tol, f"{err:.2e}")


def _power_plans():
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build

    return [build(BODY_PLANS[k]()) for k in ("beetle", "teal", "ray")]


def test_the_device_servo_command_is_servo_command() -> None:
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("the device servo command agrees with TriphibianEnv.servo_command", reason)
        return
    import warp as wp

    from dytiscidae.envs.triphibian import TriphibianEnv

    rng = np.random.default_rng(29)
    envs = [TriphibianEnv(p) for p in _power_plans()]
    tgt, rate, lead, world, has, ref = [], [], [], [], [], []
    for w, e in enumerate(envs):
        n = len(e.act_names)
        t = rng.uniform(-1.0, 1.0, n)
        # World 0 has a rate, world 1 none, world 2 one of the wrong length.
        r = (rng.normal(0.0, 5.0, n), None, rng.normal(0.0, 5.0, n + 1))[w]
        e.cpg._rate = r
        ref.append(e.servo_command(t))
        h = r is not None and len(r) == n
        tgt.append(t)
        rate.append(r[:n] if h else np.zeros(n))
        lead.append(e.servo_lead[:n])
        world.append(np.full(n, w))
        has.append(int(h))
    ref = np.concatenate(ref)
    lead_all = np.concatenate(lead)
    check("the fixture has a servo lead and a rate to feed forward",
          bool(np.abs(lead[0]).max() > 0) and has == [1, 0, 0], f"has_rate {has}")
    for dtype, tol in ((wp.float64, 1e-9), (wp.float32, 2e-4)):
        from dytiscidae.envs.devroll.scatter_wp import servo_command

        name = "float64" if dtype == wp.float64 else "float32"
        out = servo_command(np.concatenate(tgt), np.concatenate(rate), lead_all,
                            np.concatenate(world), has, dtype=dtype)
        err = float(np.abs(out - ref).max()) / max(float(np.abs(ref).max()), 1e-30)
        check(f"{name}: ctrl within {tol:g} of its maximum", err <= tol, f"{err:.2e}")


def _medusas(k=2):
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    envs = []
    for s in range(k):
        e = TriphibianEnv(build(BODY_PLANS["medusa"]()), seed=s)
        e.reset(Domain.WATER, randomise=False)
        envs.append(e)
    return envs


def test_the_device_jets_are_the_jet_set() -> None:
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("the device jets agree with JetSet.apply", reason)
        return
    import mujoco
    import warp as wp

    from dytiscidae.envs.devroll.scatter_wp import DeviceJets

    rng = np.random.default_rng(31)
    envs = _medusas(2)
    med = envs[0].medium
    check("the fixture carries bells, driven and undriven",
          all(e.jets.n > 0 for e in envs)
          and all(bool((e.jets.joint_id >= 0).any() and (e.jets.joint_id < 0).any())
                  for e in envs), str(envs[0].jets.joint_id.tolist()))
    cases, fracs, qs, states = [], [], [], []
    for z in (-0.5, 0.6, 0.0, -0.05):
        for e in envs:
            d, m = e.data, e.model
            d.qvel[:] = rng.normal(0.0, 6.0, m.nv)
            mujoco.mj_forward(m, d)
            d.qpos[2] += z - d.xpos[1, 2]
            mujoco.mj_forward(m, d)
        ref = []
        for e in envs:
            m, d, js = e.model, e.data, e.jets
            states.append((d.qpos.copy(), d.qvel.copy()))
            base = m.dof_damping.copy()
            js.reset(None)
            d.xfrc_applied[:] = 0.0
            total = js.apply(m, d, e.medium, d.time, e.timestep)
            ref.append({"f": d.xfrc_applied[:, :3].copy(), "c": m.dof_damping - base,
                        "total": total, "pump": js.last_pump_power})
            m.dof_damping[:] = base
            fracs += list(e.medium.properties(
                d.xpos[js.body_id], np.cbrt(js.volume) * 0.5, d.time)[2])
            om = np.array([d.qvel[m.jnt_dofadr[j]] if j >= 0 else 0.0 for j in js.joint_id])
            qs += list(om)
        cases.append(ref)
    fracs, qs = np.array(fracs), np.array(qs)
    check("the fixture has bells in air, in water and across the surface",
          bool((fracs == 0).any() and (fracs == 1).any() and ((fracs > 0) & (fracs < 1)).any()))
    check("the fixture has bells expelling and refilling", bool((qs > 0).any() and (qs < 0).any()))
    check("and the bells make thrust and load the joints: the comparison is not of zeros",
          max(r["total"] for c in cases for r in c) > 1e-3
          and max(r["pump"] for c in cases for r in c) > 1e-6)
    # Replay the same states on the device.
    for dtype, tol in ((wp.float64, 1e-9), (wp.float32, 2e-4)):
        name = "float64" if dtype == wp.float64 else "float32"
        dev = DeviceJets([(e.jets, e.model) for e in envs], med.air, med.water, dtype=dtype)
        worst_name, worst = "", 0.0
        for c_i, ref in enumerate(cases):
            for w, e in enumerate(envs):
                qp, qv = states[c_i * len(envs) + w]
                e.data.qpos[:], e.data.qvel[:] = qp, qv
                mujoco.mj_forward(e.model, e.data)
            x = dev.host_inputs([e.data for e in envs])
            out = dev.loads(x["theta"], x["omega"], x["xmat"], x["xpos"], med.wind, med.current)
            for w, (e, r) in enumerate(zip(envs, ref)):
                a, b = dev.off_np[w], dev.off_np[w + 1]
                f = np.zeros_like(r["f"])
                np.add.at(f, dev.body_id[a:b], out["f"][a:b])
                c = np.zeros(e.model.nv)
                drv = dev.dofadr[a:b] >= 0
                np.add.at(c, dev.dofadr[a:b][drv], out["c_pump"][a:b][drv])
                for k, got, want in (("f", f, r["f"]), ("c_pump", c, r["c"]),
                                     ("total", out["total"][w], r["total"]),
                                     ("pump", out["pump"][w], r["pump"])):
                    err = float(np.abs(np.asarray(got) - want).max()) / max(
                        float(np.abs(want).max()), 1e-30)
                    if err > worst:
                        worst_name, worst = k, err
        check(f"{name}: jet force, pumping load, thrust total and pump power within {tol:g} "
              "of their maxima", worst <= tol, f"worst {worst_name} at {worst:.2e}")


def test_the_device_inflow_is_induced_flow() -> None:
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("the device induced flow agrees with InducedFlow.update", reason)
        return
    import warp as wp

    from dytiscidae.envs.devroll.scatter_wp import DeviceInflow
    from dytiscidae.physics.fluid import InducedFlow

    # Four worlds: a hovering disc, forward flight, a disc with no span, and a
    # disc whose flow runs against its force; forces off and water density on
    # some steps.
    spans = [0.89, 0.35, 0.0, 1.2]
    steps, dt = 160, 0.004 * InducedFlow.UPDATE_EVERY
    rng = np.random.default_rng(41)
    draws = []
    for step in range(steps):
        F = rng.normal(0.0, 1.0, (4, 3)) * np.array([[0.5, 0.5, 6.0]])
        Vf = np.stack([rng.normal(0.0, 0.02, 3),                     # hover
                       np.array([-9.0, 0.0, 0.5]) + rng.normal(0.0, 1.0, 3),
                       rng.normal(0.0, 3.0, 3),
                       -2.0 * F[3] / np.linalg.norm(F[3]) + rng.normal(0.0, 0.3, 3)])
        rho = np.where(rng.uniform(size=4) < 0.3, 1025.0, 1.2)
        if step % 17 == 5:
            F[1] = 0.0                                                # no lift this step
        if step % 23 == 7:
            rho[3] = 0.0
        draws.append((F, Vf, rho))
    refs = [InducedFlow(s) for s in spans]
    want = []
    for F, Vf, rho in draws:
        for i, r in enumerate(refs):
            if r.due():
                r.update(F[i], Vf[i], float(rho[i]), dt)
            r.tick()
        want.append(np.stack([r.w.copy() for r in refs]))
    want = np.array(want)
    scale = float(np.abs(want).max())
    check("the fixture's inflow is not zero, and the spanless disc's is",
          scale > 0.05 and float(np.abs(want[:, 2]).max()) == 0.0, f"max |w| {scale:.3f}")
    for dtype, tol in ((wp.float64, 1e-9), (wp.float32, 2e-4)):
        name = "float64" if dtype == wp.float64 else "float32"
        dev = DeviceInflow(spans, dtype=dtype)
        worst = 0.0
        for (F, Vf, rho), wt in zip(draws, want):
            dev.update(F, Vf, rho, dt)
            dev.tick()
            worst = max(worst, float(np.abs(dev.w - wt).max()) / scale)
        check(f"{name}: inflow over {steps} steps within {tol:g} of its maximum",
              worst <= tol, f"worst {worst:.2e}")


def _strip_positions(e):
    p, d = e.solver.panels, e.data
    R = d.xmat.reshape(-1, 3, 3)[p.body_id]
    return d.xpos[p.body_id] + np.einsum("nij,nj->ni", R, p.pos_local)


def test_the_device_damping_is_implicit_aero_damping() -> None:
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("the device damping split agrees with ImplicitAeroDamping", reason)
        return
    import mujoco
    import warp as wp

    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.devroll.scatter_wp import DeviceDamping
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    from dytiscidae.physics.fluid import WING, ImplicitAeroDamping, strip_damping

    # eel and ray descend through the free surface; a second eel has the split off.
    envs = []
    for k, plan in enumerate(("eel", "ray", "eel")):
        e = TriphibianEnv(build(BODY_PLANS[plan]()), seed=k)
        e.reset(Domain.WATER, randomise=False)
        envs.append(e)
    enabled = [1, 1, 0]
    lift_scale = [1.0, 0.7, 1.0]
    dampers = [ImplicitAeroDamping(e.model, e.medium, e.solver.panels.half_height)
               for e in envs]
    rng = np.random.default_rng(43)
    steps = 48
    z_path = np.linspace(0.25, -0.25, steps) + 0.004 * np.sin(np.arange(steps))
    frames, want = [], []
    for t in range(steps):
        frame, ref = [], []
        for w, (e, dmp) in enumerate(zip(envs, dampers)):
            m, d, pn = e.model, e.data, e.solver.panels
            d.qvel[:] = rng.normal(0.0, 1.5, m.nv)
            mujoco.mj_forward(m, d)
            d.qpos[2] += z_path[t] + 0.03 * w - d.xpos[1, 2]
            mujoco.mj_forward(m, d)
            n = pn.n
            x = {"pos": _strip_positions(e), "q": rng.uniform(0.0, 40.0, n),
                 "area": np.asarray(pn.area, float), "lift": rng.normal(0.0, 2.0, n),
                 "drag": np.abs(rng.normal(0.0, 1.0, n)),
                 "rho": np.where(rng.uniform(size=n) < 0.5, 1025.0, 1.2),
                 "ar": rng.uniform(1.0, 8.0, n), "is_wing": (pn.kind == WING).astype(int),
                 "com": d.subtree_com.copy(), "cdof": d.cdof.copy(), "qvel": d.qvel.copy()}
            frame.append(x)
            if enabled[w]:
                due = dmp.due()
                b = (strip_damping(x["q"], x["area"], x["lift"], x["drag"], x["rho"], x["ar"],
                                   x["is_wing"] == 1, lift_scale[w]) if due else None)
                dmp.apply(d, x["pos"], pn.body_id, b)
            else:
                due = False
                dmp.clear(d)
            ref.append({"dd": m.dof_damping.copy(), "qf": d.qfrc_applied.copy(), "due": due})
            m.dof_damping[:] = dmp.base
        frames.append(frame)
        want.append(ref)
    dues = np.array([[r["due"] for r in ref[:2]] for ref in want])
    off_cadence = [(t, w) for t in range(steps) for w in range(2)
                   if dues[t, w] and t % ImplicitAeroDamping.REFRESH_EVERY != 0]
    check("the descent wets strips between refreshes: B is refreshed off its cadence",
          len(off_cadence) > 0, f"{len(off_cadence)} off-cadence refreshes")
    qf_scale = max(float(np.abs(r["qf"]).max()) for ref in want for r in ref)
    dd_scale = max(float(np.abs(r["dd"]).max()) for ref in want for r in ref)
    check("and the split is not zero: B and its compensation carry load",
          qf_scale > 1e-3 and max(float(np.abs(r["dd"] - dmp.base).max())
                                  for ref in want for r, dmp in zip(ref, dampers)) > 1e-6)
    for dtype, tol in ((wp.float64, 1e-9), (wp.float32, 2e-4)):
        name = "float64" if dtype == wp.float64 else "float32"
        dev = DeviceDamping(dampers, [e.solver.panels.body_id for e in envs], dtype=dtype)
        err = {"dof_damping": 0.0, "qfrc_applied": 0.0}
        due_ok = True
        for frame, ref in zip(frames, want):
            cat = lambda k: np.concatenate([x[k] for x in frame])
            out = dev.step(cat("pos"), cat("q"), cat("area"), cat("lift"), cat("drag"),
                           cat("rho"), cat("ar"), cat("is_wing"), lift_scale, cat("com"),
                           cat("cdof"), cat("qvel"), enabled=enabled)
            due_ok &= [bool(v) for v in out["due"][:2]] == [r["due"] for r in ref[:2]]
            err["dof_damping"] = max(err["dof_damping"], float(np.abs(
                out["dof_damping"] - np.concatenate([r["dd"] for r in ref])).max()) / dd_scale)
            err["qfrc_applied"] = max(err["qfrc_applied"], float(np.abs(
                out["qfrc_applied"] - np.concatenate([r["qf"] for r in ref])).max()) / qf_scale)
        check(f"{name}: B is formed on the same steps (cadence and wet-set staleness)", due_ok)
        worst = max(err, key=err.get)
        check(f"{name}: dof_damping and qfrc_applied over {steps} steps within {tol:g} of "
              "their maxima", err[worst] <= tol, f"worst {worst} at {err[worst]:.2e}")


def test_the_device_added_mass_is_each_worlds_own() -> None:
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("the device added-mass write agrees with FluidSolver.apply per world", reason)
        return
    import mujoco
    import warp as wp

    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.devroll.scatter_wp import added_mass_into_model
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    # Two worlds of one body at different depths, then a ray: each world's mass is its own.
    rng = np.random.default_rng(47)
    envs, ref = [], []
    for k, (plan, z) in enumerate((("eel", -0.4), ("eel", 0.0), ("ray", -0.2))):
        e = TriphibianEnv(build(BODY_PLANS[plan]()), seed=k)
        e.reset(Domain.WATER, randomise=False)
        m, d = e.model, e.data
        d.qvel[:] = rng.normal(0.0, 1.0, m.nv)
        mujoco.mj_forward(m, d)
        d.qpos[2] += z - d.xpos[1, 2]
        mujoco.mj_forward(m, d)
        e.solver.apply(d, d.time)          # the numpy write: body_mass, body_inertia
        sol = e.solver
        ref.append({"mb": m.body_mass - sol._dry_mass, "bm": m.body_mass.copy(),
                    "bi": m.body_inertia.copy(), "dry": sol._dry_mass.copy(),
                    "dry_i": sol._dry_inertia.copy(), "lever2": sol._lever2.copy()})
        envs.append(e)
    check("the two eel worlds carry different added mass",
          float(np.abs(ref[0]["mb"] - ref[1]["mb"]).max()) > 1e-4
          and float(ref[0]["mb"].sum()) > 1e-3)
    boff = np.cumsum([0] + [len(r["bm"]) for r in ref])
    cat = lambda k: np.concatenate([r[k] for r in ref])
    for dtype, tol in ((wp.float64, 1e-9), (wp.float32, 2e-4)):
        name = "float64" if dtype == wp.float64 else "float32"
        bm, bi = added_mass_into_model(cat("dry"), cat("dry_i"), cat("mb"), cat("lever2"), boff,
                                       dtype=dtype)
        e_m = float(np.abs(bm - cat("bm")).max()) / float(np.abs(cat("bm")).max())
        e_i = float(np.abs(bi - cat("bi")).max()) / float(np.abs(cat("bi")).max())
        check(f"{name}: body_mass and body_inertia per world within {tol:g} of their maxima",
              max(e_m, e_i) <= tol, f"mass {e_m:.2e}, inertia {e_i:.2e}")


def test_the_device_replay_is_not_built_yet() -> None:
    skip("the device fluid replays FluidSolver.apply for 2000 steps (A1a replay)",
         "every A1a and A1b slice is parity-tested on its own; the replay that runs them "
         "in sequence inside an MJWarp step needs A2's model and step graph")


def test_the_reason_is_set_when_the_packages_do_not_import() -> None:
    # The guard runs at import, so a GPU machine never reaches its except branch.
    # Force the import to fail in a child process and read the reason it leaves.
    import subprocess
    code = ("import sys; sys.modules['mujoco_warp'] = None; "
            "from dytiscidae.envs import devroll; "
            "print(repr(devroll.UNAVAILABLE_REASON), devroll.AVAILABLE)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[1]), env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    check("a failed import leaves the reason 'MJWarp not importable'",
          out.stdout.strip() == "'MJWarp not importable' False",
          out.stdout.strip() or out.stderr.strip()[-200:])


def main() -> int:
    functions = [test_the_device_path_reports_why_it_cannot_run,
                 test_the_reason_is_set_when_the_packages_do_not_import,
                 test_the_strip_kernel_is_the_numpy_strip_loads,
                 test_the_wagner_lag_is_the_numpy_unsteady_state,
                 test_the_device_medium_and_added_mass_are_the_numpy_ones,
                 test_the_device_assembly_is_the_numpy_finish_bodies,
                 test_the_device_entrainment_is_the_numpy_entrainment_reaction,
                 test_the_device_rotors_are_the_rotor_batch,
                 test_the_device_power_budget_is_the_power_budget,
                 test_the_device_servo_command_is_servo_command,
                 test_the_device_jets_are_the_jet_set,
                 test_the_device_inflow_is_induced_flow,
                 test_the_device_damping_is_implicit_aero_damping,
                 test_the_device_added_mass_is_each_worlds_own,
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
