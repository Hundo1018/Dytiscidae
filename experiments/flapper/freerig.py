"""A torque-driven rig whose body can be let go (pass 1k).

`rig.run` moves every wing joint by overwriting its state, so the body never
feels the wings' reaction and the drive torque has to be reconstructed.  Here
the stroke and feather are tracked by a stiff servo (feed-forward plus PD,
applied as joint torque), so:

* the torque and power at each driven axis are the ones actually applied,
  inertia, gravity and any free part (a flap) included;
* the body's reaction is physical, and chosen root axes can be released.

Held root axes carry a very large armature and are reset each step; the
velocity they pick up in a step, times that armature, is the wrench the
holder supplies, which gives the net force and the pitching moment directly.

    stroke  = (Phi/2) sin(wt)
    feather = psi0 - Psi cos(wt)
"""
import math

import numpy as np

G = 9.80665
ARM = 1e4
K85 = math.sin(math.radians(42.5))


def run_free(body: dict, V: float, theta_deg: float, f: float, Phi_deg: float, Psi_deg: float,
             psi0_deg: float, *, free: str = "", hold_beats: int = 2, beats: int = 4,
             mass: float | None = 0.60, medium: str = "air", cop_law: str | None = None,
             feather_axis: float = 0.0, one_way_flap: dict | None = None,
             dt: float = 2.5e-4, wn: float = 600.0, keep_series: bool = False,
             joint_damping: float | None = 0.0, solver_flags: dict | None = None,
             hull_inertia: float | None = None, cg_shift: float = 0.0,
             tail: dict | None = None, pilot: dict | None = None,
             strip_twist: dict | None = None, chain: dict | None = None,
             fluid=None) -> dict:
    """free: root axes released after hold_beats, any of "x", "z", "p" (pitch).
    The measurement runs over the ``beats`` after that, body held or not.
    mass: the machine's all-up mass, made up with ballast at the hull's centre
    of mass (None keeps the built model's).
    joint_damping: N m s/rad on every driven joint.  The built model carries
    0.0365 (a generic bearing figure, TYPED), which costs 46 W at the take-off
    point of pass 1i; 0.0, the default, charges the drive what `rig.run`'s
    aerodynamic power did.  None keeps the built value.
    hull_inertia: kg m2, the hull's two transverse moments of inertia (how far
    the mass is spread along it); None keeps the built hull's, which has the
    ballast as a point at its centre.
    cg_shift: m, the hull's centre of mass moved forward along its axis.
    tail: for a body built with ``stab_length`` (an all-moving tail).  Keys:
    i0_deg (incidence, leading edge up, relative to the hull), kp (rad of tail
    per rad of pitch error), kd (s), ki (1/s, 0), limit_deg (25), rate (rad/s, 8), and
    lag_beats (1: the pitch error and rate are low-passed over this many
    beats before the tail sees them, so it does not chase the flapping),
    ref_deg (the pitch it holds; theta_deg unless given).
    pilot: a height and speed loop over the tail's pitch loop, from release
    on (needs the tail's feedback and pitch released).  Height is held with
    the pitch the tail is asked for, speed with the flapping frequency:
        pitch asked = ref_deg + clip(kh (z0 - z) - kvz vz, +-pitch_lim_deg)
        f = f0 (1 + clip(kf (V - vx), +-f_lim))
    kh deg/m, kvz deg/(m/s), kf 1/(m/s).  z, vz and vx are low-passed over
    lag_beats like the pitch.  The cam offset stays where it was set.
    The tail's servo is rate-limited; its feedback starts at release.
    Optional pilot keys: z_step = (beat, m) and v_step = (beat, m/s), a step
    in the height and in the speed asked for that many beats after release.
    strip_twist: `rig.run`'s (tip_deg, share): the fluid strips turned about
    their span, the feather hinge making the rest; the twist's own drive,
    rate and inertia are not in the loads.
    chain: for a body built with ``wing_panels`` > 1 (pass W7): how the span
    hinges between the panels are held.  The tip's twist against the root is
    the sum of a wing's hinge angles; the reference is
    ``held_deg - tip_deg cos(wt)`` at the tip (the feather's phase and sense),
    shared equally between the hinges.  Keys: mode
      "servo"   every hinge tracks its share (a drive at each rib);
      "tube"    one torque between root and tip tracks the tip's twist, the
                hinges are springs of k N m/rad (a torque tube in a
                torsionally soft wing);
      "passive" springs of k only, nothing driven;
    zeta (0.05): the springs' damping, of critical for k and the panel's own
    inertia about its hinge.  Reported: the twist drive's power and peak
    torque, what the tip actually did, and each hinge's swing.
    fluid: a `medium.Fluid` in place of the medium's own (air, or water when
    ``medium`` is "water"): another density and viscosity, a vacuum.
    The result's ``clamped`` says whether the solver's load limiter ever bound."""
    import dytiscidae.core.phenotype as PH
    PH.AVIONICS_MASS = 0.005
    from body import built
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    from dytiscidae.physics.fluid import WING as KIND_WING

    Phi, Psi, psi0, th = (math.radians(x) for x in (Phi_deg, Psi_deg, psi0_deg, theta_deg))
    w = 2 * math.pi * f
    p = built(body)
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
    env.reset(Domain.AIR if medium == "air" else Domain.WATER, randomise=False)
    d, m, mj = env.data, env.model, env._mj
    if fluid is not None:
        setattr(env.medium, "air" if medium == "air" else "water", fluid)
    env.solver.cop_law = cop_law
    clamped_any = False
    from body import DEFAULTS as _BD
    if dict(_BD, **body)["wing_section"]:
        import sections
        env.solver.section = sections.load(dict(_BD, **body)["wing_section"])
        if not np.any(env.solver.panels.camber != 0.0):
            raise ValueError("wing_section needs a wing_shape with a camber: it marks the upper side")
    for k_, v_ in (solver_flags or {}).items():      # diagnosis only
        setattr(env.solver, k_, v_)
    pan = env.solver.panels
    hull = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "seg0_hull")
    if mass is not None:
        add = mass - float(env.solver._dry_mass.sum())
        env.solver._dry_mass[hull] += add
        m.body_mass[hull] += add
        mj.mj_setConst(m, d)
    total = float(env.solver._dry_mass.sum())

    jid, role, aids = [], [], []
    for a in env.act_names:
        aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, a)
        aids.append(aid)
        m.actuator_gainprm[aid, :] = 0.0
        m.actuator_biasprm[aid, :] = 0.0
        m.actuator_forcelimited[aid] = 0
        m.actuator_ctrllimited[aid] = 0
        j = int(m.actuator_trnid[aid, 0])
        jid.append(j)
        par = int(m.body_parentid[int(m.jnt_bodyid[j])])
        if par != hull and "wing" in mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, par) \
                and "wing" in mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])) \
                and body.get("wing_panels", 1) > 1:
            role.append("twist")
        elif par != hull:
            role.append("tail")
        else:
            role.append("feather" if a.endswith("_f") else "stroke")
    role = np.array(role)
    if hull_inertia is not None:
        long_axis = int(np.argmin(m.body_inertia[hull]))
        for ax in range(3):
            if ax != long_axis:
                m.body_inertia[hull, ax] = hull_inertia
                env.solver._dry_inertia[hull, ax] = hull_inertia
    if cg_shift:
        m.body_ipos[hull, int(np.argmax(np.abs(m.body_ipos[hull])))] += cg_shift
    if hull_inertia is not None or cg_shift:
        mj.mj_setConst(m, d)
    dof = np.array([int(m.jnt_dofadr[j]) for j in jid])
    if joint_damping is not None:
        m.dof_damping[dof] = joint_damping
        env.solver._damping.base[dof] = joint_damping     # the solver owns dof_damping
    qad = np.array([int(m.jnt_qposadr[j]) for j in jid])
    if feather_axis:
        for a, j in enumerate(jid):
            if role[a] not in ("feather", "twist"):
                continue
            b = int(m.jnt_bodyid[j])
            sel = (pan.body_id == b) & (pan.kind == KIND_WING)
            cdir = pan.chord_local[sel][0] / np.linalg.norm(pan.chord_local[sel][0])
            if float(pan.pos_local[sel].mean(0) @ cdir) < 0.0:
                cdir = -cdir
            m.jnt_pos[j] = m.jnt_pos[j] + feather_axis * float(pan.chord[sel].mean()) * cdir

    q0 = d.qpos[:7].copy()
    q0[:3] = (0.0, 0.0, 30.0 if medium == "air" else -3.0)
    q0[3:7] = (math.cos(-th / 2), 0.0, math.sin(-th / 2), 0.0)
    v0 = np.zeros(6)
    v0[0] = V
    # Free-joint dofs: 0-2 linear (world), 3-5 angular (body); pitch is 4.
    rel = {"x": 0, "z": 2, "p": 4}
    released = np.zeros(6, bool)
    m.dof_armature[:6] = ARM

    tl = dict(i0_deg=0.0, kp=0.0, kd=0.0, ki=0.0, limit_deg=25.0, rate=8.0, lag_beats=1.0,
              ref_deg=theta_deg)
    tl.update(tail or {})
    th_ref = math.radians(tl["ref_deg"])
    ap = None
    if pilot is not None:
        ap = dict(kh=3.0, kvz=3.0, pitch_lim_deg=8.0, kf=0.08, f_lim=0.2)
        ap.update(pilot)
    zf, vzf, vxf, w_now, th_ask = 0.0, 0.0, V, w, 0.0
    z_ask, v_ask = 0.0, V
    tail_q = -math.radians(tl["i0_deg"])            # joint angle: leading edge down is positive
    e_f, r_f, e_i = 0.0, 0.0, 0.0

    tw = None
    k_f = 1.0
    if strip_twist is not None:
        from rig import twist_setup, twist_turn
        tw = twist_setup(env, body, [j for a, j in enumerate(jid) if role[a] == "feather"], strip_twist, dt)
        k_f = 1.0 - tw["share"]

    is_tw = role == "twist"
    chn = None
    if is_tw.any():
        chn = dict(mode="servo", tip_deg=0.0, held_deg=0.0, k=1.0, zeta=0.05)
        chn.update(chain or {})
        n_h = int(is_tw.sum()) // 2                 # hinges per wing
        # which wing a hinge is on: walk up to the panel on the hull
        side = []
        for j in np.array(jid)[is_tw]:
            b = int(m.jnt_bodyid[j])
            while int(m.body_parentid[b]) != hull:
                b = int(m.body_parentid[b])
            side.append(b)
        side = np.array(side)
        sides = [np.flatnonzero(is_tw)[side == s_] for s_ in np.unique(side)]
        A_t, H_t = math.radians(chn["tip_deg"]), math.radians(chn["held_deg"])

    def ref(t, ph=None, w=w):
        s, c = (math.sin(w * t), math.cos(w * t)) if ph is None else (math.sin(ph), math.cos(ph))
        q, qd, qdd = np.zeros(len(jid)), np.zeros(len(jid)), np.zeros(len(jid))
        st, fe = role == "stroke", role == "feather"
        q[role == "tail"] = tail_q
        q[st], qd[st], qdd[st] = 0.5 * Phi * s, 0.5 * Phi * w * c, -0.5 * Phi * w * w * s
        q[fe], qd[fe], qdd[fe] = psi0 - k_f * Psi * c, k_f * Psi * w * s, k_f * Psi * w * w * c
        if tw is not None:
            twist_turn(pan, tw, math.degrees(-Psi * c))
        if chn is not None and chn["mode"] == "servo":
            q[is_tw], qd[is_tw], qdd[is_tw] = ((H_t - A_t * c) / n_h, A_t * w * s / n_h,
                                               A_t * w * w * c / n_h)
        return q, qd, qdd

    d.qpos[:7], d.qvel[:6] = q0, v0
    q, qd, _ = ref(0.0)
    d.qpos[qad], d.qvel[dof] = q, qd
    mj.mj_forward(m, d)
    env.solver.apply(d, 0.0)                       # folds the added mass in
    mj.mj_forward(m, d)
    J = np.zeros(len(dof))
    for n_, i in enumerate(dof):                   # the mass matrix's diagonal (3.15 has no qM)
        e, col = np.zeros(m.nv), np.zeros(m.nv)
        e[i] = 1.0
        mj.mj_mulM(m, d, col, e)
        J[n_] = col[i]
    kp, kv = J * wn * wn, 2.0 * J * wn
    # The servo is the model's own position actuator, so the integrator
    # treats its damping implicitly.  An explicit PD (the first version) was
    # stable in air and not in water, where a wing's added mass, and so the
    # inertia the gains were sized on, changes several-fold through a beat.
    if chn is not None and chn["mode"] != "servo":
        # the hinges are the wing's own torsional springs, not drives
        kp[is_tw] = chn["k"]
        kv[is_tw] = 2.0 * chn["zeta"] * np.sqrt(chn["k"] * J[is_tw])
        J_tube = [float(J[ix].sum()) for ix in sides]
    for n_, aid in enumerate(aids):
        m.actuator_gainprm[aid, 0] = kp[n_]
        m.actuator_biasprm[aid, 1], m.actuator_biasprm[aid, 2] = -kp[n_], -kv[n_]

    # the machine's centre of mass from the hull frame's origin, world axes, at the start
    # (not `subtree_com`: the solver inflates body_mass without body_subtreemass)
    dry = env.solver._dry_mass
    com0 = ((dry[:, None] * d.xipos).sum(0) / dry.sum() - d.xpos[hull]).tolist()
    n_beat = int(round(1.0 / (f * dt)))
    n_hold, n_all = hold_beats * n_beat, (hold_beats + beats) * n_beat
    rec = dict(t=[], P=[], Ps=[], tq=[], err=[], hold=[], pos=[], vel=[], pitch=[], tail=[], Pt=[],
               w=[], ask=[], Ptw=[], tqtw=[], tip=[], hinge=[], stroke=[], com_vel=[], angmom=[])
    phase = None
    t = 0.0
    for k in range(n_all):
        if k == n_hold:
            for ch in free:
                released[rel[ch]] = True
                m.dof_armature[rel[ch]] = 0.0
        if k >= n_hold and released[4] and (tl["kp"] or tl["kd"]):
            # pitch, nose up positive, from the hull's attitude; its rate from the gyro
            a_f = dt * f / max(tl["lag_beats"], 1e-6) * 2.0 * math.pi
            if ap is not None:
                zf += a_f * ((d.qpos[2] - q0[2]) - zf)
                vzf += a_f * (d.qvel[2] - vzf)
                vxf += a_f * (d.qvel[0] - vxf)
                lim_p = ap["pitch_lim_deg"]
                kb = (k - n_hold) / n_beat
                z_ask = ap["z_step"][1] if ap.get("z_step") and kb >= ap["z_step"][0] else 0.0
                v_ask = V + (ap["v_step"][1] if ap.get("v_step") and kb >= ap["v_step"][0] else 0.0)
                th_ask = math.radians(min(max(ap["kh"] * (z_ask - zf) - ap["kvz"] * vzf, -lim_p), lim_p))
                w_now = w * (1.0 + min(max(ap["kf"] * (v_ask - vxf), -ap["f_lim"]), ap["f_lim"]))
            e_f += a_f * ((-2.0 * math.atan2(d.qpos[5], d.qpos[3]) - th_ref - th_ask) - e_f)
            r_f += a_f * (-d.qvel[4] - r_f)
            lim = math.radians(tl["limit_deg"])
            if tl["ki"]:                               # integral, held inside the tail's travel
                e_i = min(max(e_i + tl["ki"] * e_f * dt, -lim), lim)
            want = -(math.radians(tl["i0_deg"]) + tl["kp"] * e_f + tl["kd"] * r_f + e_i)
            want = min(max(want, -lim), lim)
            tail_q += min(max(want - tail_q, -tl["rate"] * dt), tl["rate"] * dt)
        if ap is not None and phase is None and k >= n_hold:
            phase = w * t                              # the crank's own angle from release on
        q, qd, qdd = ref(t, phase, w_now)
        d.ctrl[:] = 0.0
        d.ctrl[aids] = q + (kv / kp) * qd          # the reference rate, fed forward
        d.xfrc_applied[:] = 0.0
        env.solver.apply(d, d.time)
        # The solver has just rewritten all of qfrc_applied (its implicit
        # damping's explicit half); the drive is added to that, never set.
        clamped_any |= bool(env.solver.diag.clamped)
        d.qfrc_applied[dof] += J * qdd
        tube = None
        if chn is not None and chn["mode"] == "tube":
            # One torque between the root panel and the tip's: by virtual work
            # it is the same generalised force on every hinge of the wing.
            s_, c_ = (math.sin(phase), math.cos(phase)) if phase is not None else (math.sin(w * t), math.cos(w * t))
            tube = []
            for ix, Jt in zip(sides, J_tube):
                e_ = (H_t - A_t * c_) - float(d.qpos[qad[ix]].sum())
                ed = A_t * w_now * s_ - float(d.qvel[dof[ix]].sum())
                tq_ = Jt * (wn * wn * e_ + 2.0 * wn * ed) / len(ix)
                d.qfrc_applied[dof[ix]] += tq_
                tube.append(tq_)
        vb = d.qvel[:6].copy()
        qv_before = d.qvel[dof].copy()
        mj.mj_step(m, d)
        tau = J * qdd + d.actuator_force[aids]
        p_tw, tq_tw = 0.0, 0.0
        if chn is not None:
            if chn["mode"] == "servo":
                p_tw = float(tau[is_tw] @ qd[is_tw])
                tq_tw = float(np.abs(tau[is_tw]).max())
            elif tube is not None:
                p_tw = float(sum(tq_ * qv_before[ix].sum() for tq_, ix in zip(tube, sides)))
                tq_tw = float(max(abs(x) for x in tube))
            if chn["mode"] != "servo":
                tau = np.where(is_tw, 0.0, tau)        # a spring is not a drive
                tau_plus = p_tw
            else:
                tau_plus = 0.0
        hold = ARM * (d.qvel[:6] - vb) / dt           # wrench the holder did NOT have to: -hold
        t += dt
        if phase is not None:
            phase += w_now * dt
        # put the held root axes back; a released axis keeps what it has
        lin = ~released[:3]
        d.qpos[:3][lin] = (q0[:3] + v0[:3] * t)[lin]
        d.qvel[:3][lin] = v0[:3][lin]
        if not released[4]:
            d.qpos[3:7], d.qvel[3:6] = q0[3:7], 0.0
        else:
            d.qvel[3], d.qvel[5] = 0.0, 0.0
        if k >= n_hold:
            rec["t"].append(t - dt)
            rec["P"].append(float(tau @ qd) + (tau_plus if chn is not None else 0.0))
            if chn is not None:
                rec["Ptw"].append(p_tw)
                rec["tqtw"].append(tq_tw)
                rec["tip"].append(float(d.qpos[qad[sides[0]]].sum()))
                rec["hinge"].append(d.qpos[qad[sides[0]]].copy())
                rec["stroke"].append(float(q[role == "stroke"][0]))
            rec["Ps"].append(float(tau[role == "stroke"] @ qd[role == "stroke"]))
            rec["tq"].append(float(np.abs(tau[role == "stroke"]).sum()))
            trk = ~is_tw if (chn is not None and chn["mode"] != "servo") else np.ones(len(q), bool)
            rec["err"].append(float(np.abs(q - d.qpos[qad])[trk].max()))
            rec["hold"].append(hold)
            rec["pos"].append(d.qpos[:3].copy())
            rec["vel"].append(d.qvel[:3].copy())
            qw, qy = d.qpos[3], d.qpos[5]
            rec["pitch"].append(-2.0 * math.atan2(qy, qw))
            rec["tail"].append(-tail_q)
            rec["w"].append(w_now)
            rec["ask"].append(th_ask)
            rec["Pt"].append(float(tau[role == "tail"] @ d.qvel[dof][role == "tail"]))
            if keep_series:
                # the whole machine's centre-of-mass velocity and its angular
                # momentum about that centre (MuJoCo's own sums over the tree)
                mj.mj_subtreeVel(m, d)
                rec["com_vel"].append(d.subtree_linvel[hull].copy())
                rec["angmom"].append(d.subtree_angmom[hull].copy())
    T = beats / f
    hold = np.array(rec["hold"])
    vel, pos = np.array(rec["vel"]), np.array(rec["pos"])
    P = np.array(rec["P"])
    out = {"V": V, "theta_deg": theta_deg, "f": f, "Phi_deg": Phi_deg, "Psi_deg": Psi_deg,
           "psi0_deg": psi0_deg, "free": free, "medium": medium, "mass_kg": total,
           "finite": bool(np.isfinite(pos).all() and np.isfinite(P).all()),
           "power_W": float(P.mean()), "stroke_power_W": float(np.mean(rec["Ps"])),
           "crank_eq_peak": float((np.abs(P) / np.array(rec["w"])).max() / K85),
           "crank_eq_stroke_only_peak": float((np.abs(rec["Ps"]) / np.array(rec["w"])).max() / K85),
           "crank_eq_min": float((P / np.array(rec["w"])).min() / K85),
           "stroke_torque_peak": float(np.max(rec["tq"])),
           "track_err_deg": float(math.degrees(np.max(rec["err"]))), "clamped": clamped_any}
    # Net force on the machine, gravity included: from the holder where held,
    # from the motion where released.
    net = np.zeros(3)
    for i in range(3):
        net[i] = total * (vel[-1, i] - vel[0, i]) / (rec["t"][-1] - rec["t"][0]) if released[i] \
            else float(hold[:, i].mean())
    out["Fx"], out["Fz"] = float(net[0]), float(net[2] + total * G)
    out["lift_over_weight"] = out["Fz"] / (total * G)
    # The pitching moment about the built centre of mass, from the holder's
    # wrench about the hull frame's origin (gravity acts at the centre of mass).
    F_ng = hold[:, :3] + np.array([0.0, 0.0, total * G])
    c0 = np.array(com0)
    M_cg = hold[:, 4] - (c0[2] * F_ng[:, 0] - c0[0] * F_ng[:, 2]) \
        - np.cross(c0, np.array([0.0, 0.0, -total * G]))[1]
    if chn is not None:
        ptw, tip_s, hg = np.array(rec["Ptw"]), np.degrees(rec["tip"]), np.degrees(rec["hinge"])
        tt = np.array(rec["t"]) - rec["t"][0]
        # the tip's twist as offset - amp cos(wt - lag): the feather's phase is lag 0
        cs, sn = np.cos(w * (tt + rec["t"][0])), np.sin(w * (tt + rec["t"][0]))
        a_c, a_s = -2.0 * float(np.mean((tip_s - tip_s.mean()) * cs)), -2.0 * float(np.mean((tip_s - tip_s.mean()) * sn))
        out.update(twist_mode=chn["mode"], twist_power_W=float(ptw.mean()),
                   twist_power_pos_W=float(np.maximum(ptw, 0.0).mean()),
                   twist_torque_peak=float(np.max(rec["tqtw"])),
                   tip_twist_mean_deg=float(tip_s.mean()), tip_twist_min_deg=float(tip_s.min()),
                   tip_twist_max_deg=float(tip_s.max()), tip_twist_amp_deg=float(math.hypot(a_c, a_s)),
                   tip_twist_lag_deg=float(math.degrees(math.atan2(a_s, a_c))),
                   hinge_swing_deg=[float(x) for x in (hg.max(0) - hg.min(0)) / 2.0],
                   hinge_mean_deg=[float(x) for x in hg.mean(0)])
    if role.tolist().count("tail"):
        out["tail_power_W"] = float(np.mean(np.abs(rec["Pt"])))
        tq_t = np.array(rec["tail"])
        out["tail_min_deg"], out["tail_max_deg"] = (float(math.degrees(tq_t.min())),
                                                     float(math.degrees(tq_t.max())))
    if released[4]:
        pd_ = np.degrees(np.array(rec["pitch"])) - theta_deg
        out["pitch_end_deg"] = float(pd_[-1])
        out["pitch_dev_max_deg"] = float(np.abs(pd_).max())
        nb_ = len(pd_) // n_beat
        bm = pd_[: nb_ * n_beat].reshape(nb_, n_beat)
        out["pitch_beat_mean_deg"] = [float(x) for x in bm.mean(1)]
        out["pitch_swing_deg"] = float((bm.max(1) - bm.min(1)).max() / 2.0)
        out["pitch_swing_late_deg"] = float((bm.max(1) - bm.min(1))[-max(nb_ // 4, 1):].max() / 2.0)
        out["pitch_late_deg"] = float(bm.mean(1)[-10:].mean() + theta_deg)
    else:
        out["pitch_moment_mean"] = float(hold[:, 4].mean())
        out["pitch_moment_peak"] = float(np.abs(hold[:, 4]).max())
        out["pitch_moment_cg_mean"] = float(M_cg.mean())
        out["pitch_moment_cg_swing"] = float(np.abs(M_cg - M_cg.mean()).max())
    if free:
        out["dx"], out["dz"] = float(pos[-1, 0] - pos[0, 0] - V * T), float(pos[-1, 2] - pos[0, 2])
        out["vx_end"], out["vz_end"] = float(vel[-1, 0]), float(vel[-1, 2])
        zz = pos[:n_beat, 2] - np.linspace(pos[0, 2], pos[n_beat - 1, 2], n_beat)
        out["bob_mm"] = float(1e3 * (zz.max() - zz.min()))
        zr = pos[:, 2] - pos[0, 2]
        out["dz_min"], out["dz_max"] = float(zr.min()), float(zr.max())
        out["vx_min"], out["vx_max"] = float(vel[:, 0].min()), float(vel[:, 0].max())
        late = max(len(zr) - 10 * n_beat, 0)
        out["vz_late"] = float((pos[-1, 2] - pos[late, 2]) / max(rec["t"][-1] - rec["t"][late], 1e-9))
        out["power_late_W"] = float(P[late:].mean())
        out["dz_late_min"], out["dz_late_max"] = float(zr[late:].min()), float(zr[late:].max())
        out["vx_late_min"], out["vx_late_max"] = float(vel[late:, 0].min()), float(vel[late:, 0].max())
        if ap is not None:
            ff = np.array(rec["w"]) / (2.0 * math.pi)
            out["f_min"], out["f_max"], out["f_late"] = float(ff.min()), float(ff.max()), float(ff[late:].mean())
            aa = np.degrees(np.array(rec["ask"]))
            out["pitch_ask_min_deg"], out["pitch_ask_max_deg"] = float(aa.min()), float(aa.max())
            out["pitch_ask_late_deg"] = float(aa[late:].mean())
    if keep_series:
        out["series"] = {k: np.asarray(v).tolist() for k, v in rec.items()}
        out["com0"] = com0
    return out
