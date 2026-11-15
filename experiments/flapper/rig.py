"""The held-body rig of the flapper experiments, with every wing joint
prescribed (stage 1: no torque limit).

Roles are read off the built model: a shoulder's stroke hinge (`*_j` on a
body whose parent is the hull), its feather hinge (`*_jf`), and a wrist's
twist hinge (`*_j` on a body whose parent is a wing).  Both sides get the
same angles; the model mirrors the axes.

    stroke  = (Phi/2) sin(wt)
    feather = psi0 - Psi cos(wt)
    twist   = tau0 - T   cos(wt)          (added to the feather, outer panel only)
"""
import math

import numpy as np

G = 9.80665


def run(body: dict, V: float, theta_deg: float, f: float, Phi_deg: float, Psi_deg: float,
        psi0_deg: float, T_deg: float = 0.0, tau0_deg: float = 0.0, *, beats: int = 5,
        skip: int = 2, steps_per_beat: int = 50, medium: str = "air",
        chord_law: tuple | None = None, passive_feather: tuple | None = None,
        cop_law: str | None = None, feather_axis: float = 0.0,
        one_way_flap: dict | None = None, coupling: str = "static",
        legacy_coupling: bool = False, solver_flags: dict | None = None,
        added_mass_ref: bool = False, legacy_added_weight: bool = False) -> dict:
    """chord_law = (k, sign): every wing strip's chord and area are scaled by k
    while the stroke rate has that sign (+1 or -1), blended across the
    reversal, and by k throughout if sign is 0.  Pass 1e's in-stroke shape
    change.  It changes the fluid strips only, not the wing's mass or
    geometry, so a constant k is not the same as a narrower built wing.
    None leaves the wing fixed.

    passive_feather = (k, zeta[, com]): the feather hinges (leading-edge span axis)
    are not driven.  Each is a spring of k N m/rad about psi0 with damping
    zeta of critical for k and the wing's inertia about the hinge; the stroke
    alone is prescribed, as a one-degree-of-freedom drive would give it.  Psi
    is then ignored, and the feather the wing takes is reported.  com, if
    given, moves the wing's centre of mass to com x chord behind the hinge;
    h, if given, moves the hinge axis to h x chord behind the leading edge
    (feather_axis does the same for a prescribed feather).

    cop_law: the solver's `FluidSolver.cop_law` (centre of pressure that moves
    with separation, and pitch damping); None keeps the quarter chord.

    one_way_flap: `flapwing.compile_with_flap`'s keywords (x_h, open_deg, sign,
    k_f, damp_f): each wing's rear part is a hinged flap of its own.

    coupling: how a free joint feels the prescribed motion it rides on.
    "static" (passes 1f-1i) adds M[f, s] a_s with M taken once at the start
    pose; "step" adds (M a)_f from the current pose every step, for every
    prescribed axis.  A flap always uses "step".

    legacy_coupling=True reproduces passes 1f-1h and 1j as they ran: the push
    above was written before `env.step` and the fluid solver, which rewrites
    all of qfrc_applied, wiped it, so the free joint felt no inertial push
    from the stroke at all.  For a free hinge prefer `freerig.run_free`,
    where the stroke is driven by torque and the coupling is the model's own."""
    import dytiscidae.core.phenotype as PH
    PH.AVIONICS_MASS = 0.005
    from body import flapper
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    Phi, Psi, psi0, T, tau0, th = (math.radians(x) for x in
                                   (Phi_deg, Psi_deg, psi0_deg, T_deg, tau0_deg, theta_deg))
    flap = f > 0.0
    dt = min(1e-3, 1.0 / (steps_per_beat * f)) if flap else 1e-3
    p = build(flapper(**body))
    if one_way_flap is not None:
        import dytiscidae.envs.triphibian as TR
        from flapwing import compile_with_flap
        orig = TR.compile_phenotype
        TR.compile_phenotype = lambda p_, **kw: compile_with_flap(p_, **one_way_flap, **kw)
        try:
            env = TriphibianEnv(p, seed=0, timestep=dt)
        finally:
            TR.compile_phenotype = orig
    else:
        env = TriphibianEnv(p, seed=0, timestep=dt)
    dom = Domain.AIR if medium == "air" else Domain.WATER
    env.reset(dom, randomise=False)
    d, m, mj = env.data, env.model, env._mj
    env.solver.cop_law = cop_law
    for k_, v_ in (solver_flags or {}).items():
        setattr(env.solver, k_, v_)
    amref = None
    if added_mass_ref:
        # The plate's potential-flow added-mass load from the prescribed
        # motion (`amref.py`), in place of the solver's entrainment reaction.
        from amref import AddedMassRef
        env.solver.entrainment = False
        amref = AddedMassRef(env.solver, mj)
    q0 = d.qpos[:7].copy()
    q0[:3] = (0.0, 0.0, 30.0 if medium == "air" else -3.0)
    q0[3:7] = (math.cos(-th / 2), 0.0, math.sin(-th / 2), 0.0)
    qadr, vadr = env._act_qadr, env._act_vadr
    names = env.act_names
    hull = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "seg0_hull")
    jid, role = [], []
    for a in names:
        j = int(m.actuator_trnid[mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, a), 0])
        jid.append(j)
        if a.endswith("_f"):
            role.append("feather")
        else:
            bj = int(m.jnt_bodyid[j])
            if "wing" not in mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, bj):
                role.append("held")          # F2: a tail hinge is held at 0, not flapped
            else:
                role.append("stroke" if int(m.body_parentid[bj]) == hull else "twist")
    role = np.array(role)
    pan = env.solver.panels
    from dytiscidae.physics.fluid import WING as KIND_WING
    is_w = pan.kind == KIND_WING
    chord0, area0 = pan.chord.copy(), pan.area.copy()
    w = 2 * math.pi * f
    free = np.zeros(len(names), bool)
    axis = passive_feather[3] if passive_feather is not None and len(passive_feather) > 3 \
        else feather_axis
    if axis:
        # Move each feather hinge's axis axis x chord behind the leading
        # edge, parallel to it; the wing's built shape and pose are unchanged.
        for a, j in enumerate(jid):
            if role[a] != "feather":
                continue
            b = int(m.jnt_bodyid[j])
            sel = (pan.body_id == b) & (pan.kind == KIND_WING)
            cdir = pan.chord_local[sel][0] / np.linalg.norm(pan.chord_local[sel][0])
            if float(pan.pos_local[sel].mean(0) @ cdir) < 0.0:
                cdir = -cdir                       # aft, towards the strips' load points
            m.jnt_pos[j] = m.jnt_pos[j] + axis * float(pan.chord[sel].mean()) * cdir
        mj.mj_forward(m, d)
    if passive_feather is not None:
        k_s, zeta = passive_feather[:2]
        com = passive_feather[2] if len(passive_feather) > 2 else None
        free = role == "feather"
        if com is not None:
            # Move each wing's centre of mass along its chord to com x chord
            # behind the leading-edge hinge (built: 0.25, the strips' load
            # point).  A wing whose spar carries its mass sits near 0.
            for b in sorted({int(x) for x in pan.body_id[pan.kind == KIND_WING]}):
                sel = (pan.body_id == b) & (pan.kind == KIND_WING)
                cdir = pan.chord_local[sel][0]
                cdir = cdir / np.linalg.norm(cdir)
                c_mean = float(pan.chord[sel].mean())
                along = float(m.body_ipos[b] @ cdir)
                m.body_ipos[b] = m.body_ipos[b] + (com * c_mean - along) * cdir
            mj.mj_forward(m, d)
        for a, j in enumerate(jid):
            if not free[a]:
                continue
            aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, names[a])
            m.actuator_gainprm[aid, :] = 0.0
            m.actuator_biasprm[aid, :] = 0.0
            dof = int(m.jnt_dofadr[j])
            b = int(m.jnt_bodyid[j])
            r = d.xipos[b] - d.xanchor[j]
            r = r - (r @ d.xaxis[j]) * d.xaxis[j]
            inertia = float(m.body_mass[b] * (r @ r) + m.dof_armature[dof]) + float(
                d.xaxis[j] @ (d.ximat[b].reshape(3, 3) @ np.diag(m.body_inertia[b])
                              @ d.ximat[b].reshape(3, 3).T) @ d.xaxis[j])
            m.jnt_stiffness[j] = k_s
            m.qpos_spring[int(m.jnt_qposadr[j])] = psi0
            m.dof_damping[dof] = 2.0 * zeta * math.sqrt(k_s * inertia)
            # The solver's implicit damping rewrites dof_damping as its own
            # copy (taken at build) plus the fluid bound every step, which wiped
            # this hinge damping in pass 1f (zeta was 0 there, not 0.3).
            env.solver._damping.base[dof] = m.dof_damping[dof]
        feather_q = [int(m.jnt_qposadr[j]) for a, j in enumerate(jid) if free[a]]
        d.qpos[feather_q] = psi0
        # The rig moves the body and the stroke by overwriting their state each
        # step, but inside a step MuJoCo lets every free dof accelerate, and a
        # wing's pitching moment then spins the body and the stroke instead of
        # the hinge (measured: -1.2 N m on the feather, +85 rad/s2 of it).  A
        # very large armature on the held dofs keeps them where the rig puts
        # them; the inertial push of the prescribed stroke acceleration on the
        # feather, M[f, s] a_s, is then added by hand each step.
        mj.mj_forward(m, d)
        held_dofs = list(range(6)) + [int(m.jnt_dofadr[j]) for a, j in enumerate(jid)
                                      if role[a] == "stroke"]
        couple = []
        for a, j in enumerate(jid):
            if role[a] != "stroke":
                continue
            e = np.zeros(m.nv)
            e[int(m.jnt_dofadr[j])] = 1.0
            col = np.zeros(m.nv)
            mj.mj_mulM(m, d, col, e)
            for a2, j2 in enumerate(jid):
                if free[a2] and int(m.jnt_bodyid[j2]) == int(m.jnt_bodyid[j]):
                    couple.append((a, int(m.jnt_dofadr[j2]), float(col[int(m.jnt_dofadr[j2])])))
        m.dof_armature[held_dofs] = 1e4
    flap_dofs = [int(m.jnt_dofadr[j]) for j in range(m.njnt)
                 if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or "").endswith("_jflap")]
    step_free = flap_dofs + ([int(m.jnt_dofadr[j]) for a, j in enumerate(jid) if free[a]]
                             if coupling == "step" else [])
    if flap_dofs:
        m.dof_armature[list(range(6)) + [int(m.jnt_dofadr[j]) for a, j in enumerate(jid)
                                         if not free[a]]] = 1e4
    dof_of = np.array([int(m.jnt_dofadr[j]) for j in jid])
    flap_q = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
              if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or "").endswith("_jflap")]
    fl = []
    fq = []
    # Each wing's inertia about its feather axis (structure only), for a
    # drive that also turns the feather.
    mj.mj_forward(m, d)
    i_feather = []
    for a, j in enumerate(jid):
        if role[a] != "feather":
            continue
        b = int(m.jnt_bodyid[j])
        r = d.xipos[b] - d.xanchor[j]
        r = r - (r @ d.xaxis[j]) * d.xaxis[j]
        R = d.ximat[b].reshape(3, 3)
        i_feather.append(float(m.body_mass[b] * (r @ r))
                         + float(d.xaxis[j] @ (R @ np.diag(m.body_inertia[b]) @ R.T) @ d.xaxis[j]))
    n_beat = int(round(1.0 / (f * dt))) if flap else int(round(0.2 / dt))
    weight = float(env.solver._dry_mass.sum() * G)
    F, P, Ps, Tq, Tf, ts = [], [], [], [], [], []
    t = 0.0
    for k in range(beats * n_beat):
        tgt, rate = np.zeros(len(names)), np.zeros(len(names))
        s, c = (math.sin(w * t), math.cos(w * t)) if flap else (0.0, 0.0)
        tgt[role == "stroke"], rate[role == "stroke"] = 0.5 * Phi * s, 0.5 * Phi * w * c
        tgt[role == "feather"], rate[role == "feather"] = psi0 - Psi * c, Psi * w * s
        tgt[role == "twist"], rate[role == "twist"] = tau0 - T * c, T * w * s
        if chord_law is not None:
            kc, sign = chord_law[:2]
            # Blended over the stroke reversal (width 0.3 in cos wt, about a
            # fifth of the beat): a step change makes the solver's added-mass
            # backward difference kick the torque by 3x.
            on = 1.0 if sign == 0 else 0.5 * (1.0 + math.tanh(sign * c / 0.3))
            scale = 1.0 - (1.0 - kc) * on
            pan.chord[is_w] = chord0[is_w] * scale
            pan.area[is_w] = area0[is_w] * scale
        d.qpos[:7] = q0
        d.qpos[0] = q0[0] + V * t
        d.qvel[:6] = 0.0
        d.qvel[0] = V
        held = ~free
        d.qpos[np.asarray(qadr)[held]], d.qvel[np.asarray(vadr)[held]] = tgt[held], rate[held]
        push = np.zeros(m.nv)
        if step_free:
            # (M a)_free from the pose of the last step: every prescribed axis's
            # acceleration, so a free joint feels what it rides on.
            acc = np.zeros(m.nv)
            acc[dof_of[role == "stroke"]] = -0.5 * Phi * w * w * s
            if not free.any():
                acc[dof_of[role == "feather"]] = Psi * w * w * c
            acc[dof_of[role == "twist"]] = T * w * w * c
            ma = np.zeros(m.nv)
            mj.mj_mulM(m, d, ma, acc)
            push[step_free] -= ma[step_free]
        elif free.any():
            for a, fdof, mfs in couple:
                push[fdof] -= mfs * (-0.5 * Phi * w * w * s)
        if push.any() and not legacy_coupling:
            # `env.step` unrolled, because the solver rewrites all of
            # qfrc_applied inside it: a push written before the step was wiped
            # (passes 1f-1h and 1j ran with no push at all; see their errata).
            d.ctrl[: len(tgt)] = env.servo_command(tgt)
            d.xfrc_applied[:] = 0.0
            env.solver.apply(d, d.time)
            d.qfrc_applied[:] += push
            mj.mj_step(m, d)
        else:
            env.step(tgt)
        d.qpos[np.asarray(qadr)[held]] = tgt[held]
        t += dt
        if not legacy_added_weight:
            # The solver cancels the weight MuJoCo gives the entrained fluid
            # with an upward force at each body's centre of mass.  It is not a
            # fluid load, and until pass S1 only the net force took it out:
            # the torque and power series carried m_added g times the wing's
            # arm and rate (in water, most of the stroke torque).
            d.xfrc_applied[:, 2] -= (m.body_mass - env.solver._dry_mass) * G
        if amref is not None:
            amref.add(d, t - dt)
        if k >= skip * n_beat:
            Fk = d.xfrc_applied[:, :3].sum(0).copy()
            if legacy_added_weight:
                Fk[2] -= (m.body_mass.sum() - env.solver._dry_mass.sum()) * G
            F.append(Fk)
            # Aerodynamic power: every wing body's fluid wrench about each
            # prescribed axis that carries it, times that axis's rate.
            pw, ps, tq, tf = 0.0, 0.0, 0.0, 0.0   # power, its stroke part; fluid torques
            for a, j in enumerate(jid):
                if rate[a] == 0.0:
                    continue
                b0 = int(m.jnt_bodyid[j])
                tau = 0.0
                for b in range(1, m.nbody):
                    bb = b
                    while bb != 0 and bb != b0:
                        bb = int(m.body_parentid[bb])
                    if bb == b0:
                        tau += float(d.xaxis[j] @ (d.xfrc_applied[b, 3:] + np.cross(
                            d.xipos[b] - d.xanchor[j], d.xfrc_applied[b, :3])))
                pw -= tau * rate[a]
                if role[a] == "stroke":
                    tq += abs(tau)
                    ps -= tau * rate[a]
                elif role[a] == "feather":
                    tf = max(tf, abs(tau))
            P.append(pw)
            Ps.append(ps)
            Tq.append(tq)
            Tf.append(tf)
            ts.append(t - dt)
            if free.any():
                fq.append(float(d.qpos[np.asarray(qadr)[free][0]]))
            if flap_q:
                fl.append(np.abs(d.qpos[flap_q]).copy())
    F = np.array(F).mean(0)
    return {"V": V, "theta_deg": theta_deg, "f": f, "Phi_deg": Phi_deg, "Psi_deg": Psi_deg,
            "psi0_deg": psi0_deg, "T_deg": T_deg, "tau0_deg": tau0_deg, "medium": medium,
            "Fz": float(F[2]), "Fx": float(F[0]), "Fy": float(F[1]), "weight": weight,
            "lift_over_weight": float(F[2] / weight), "aero_power_W": float(np.mean(P)),
            "stroke_torque_mean": float(np.mean(Tq)), "stroke_torque_peak": float(np.max(Tq)),
            # per flap: the widest it opened, and its mean opening over the beat
            "flap_open_max_deg": (np.degrees(np.max(fl, axis=0)).tolist() if fl else None),
            "flap_open_mean_deg": (np.degrees(np.mean(fl, axis=0)).tolist() if fl else None),
            "feather_torque_peak": float(np.max(Tf)), "feather_inertia": i_feather,
            "power_series": [float(x) for x in P],
            "stroke_torque_series": [float(x) for x in Tq],
            "stroke_power_series": [float(x) for x in Ps], "time_series": [float(x) for x in ts],
            "feather_taken_deg": ([math.degrees(min(fq)), math.degrees(max(fq))] if fq else None),
            "mass_kg": float(p.mass), "roles": sorted(set(role.tolist())),
            "finite": bool(np.isfinite(F).all())}


def trim(body: dict, V: float, f: float, Phi: float, Psi: float, T: float = 0.0,
         lo: float = -18.0, hi: float = 0.0, tol: float = 0.03, max_evals: int = 4) -> dict:
    """The feather offset at which mean lift is the weight, by the secant rule."""
    x0, x1 = -8.0, -12.0
    r0, r1 = run(body, V, 0.0, f, Phi, Psi, x0, T), run(body, V, 0.0, f, Phi, Psi, x1, T)
    evals = 2
    best = min((r0, r1), key=lambda r: abs(r["lift_over_weight"] - 1.0))
    while abs(best["lift_over_weight"] - 1.0) > tol and evals < max_evals:
        y0, y1 = r0["lift_over_weight"] - 1.0, r1["lift_over_weight"] - 1.0
        if abs(y1 - y0) < 1e-9:
            break
        x2 = min(max(x1 - y1 * (x1 - x0) / (y1 - y0), lo), hi)
        r2 = run(body, V, 0.0, f, Phi, Psi, x2, T)
        evals += 1
        x0, r0, x1, r1 = x1, r1, x2, r2
        if abs(r2["lift_over_weight"] - 1.0) < abs(best["lift_over_weight"] - 1.0):
            best = r2
    out = dict(best)
    out["trimmed"] = bool(abs(best["lift_over_weight"] - 1.0) <= tol)
    out["evals"] = evals
    return out
