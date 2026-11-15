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
             joint_damping: float | None = 0.0) -> dict:
    """free: root axes released after hold_beats, any of "x", "z", "p" (pitch).
    The measurement runs over the ``beats`` after that, body held or not.
    mass: the machine's all-up mass, made up with ballast at the hull's centre
    of mass (None keeps the built model's).
    joint_damping: N m s/rad on every driven joint.  The built model carries
    0.0365 (a generic bearing figure, TYPED), which costs 46 W at the take-off
    point of pass 1i; 0.0, the default, charges the drive what `rig.run`'s
    aerodynamic power did.  None keeps the built value."""
    import dytiscidae.core.phenotype as PH
    PH.AVIONICS_MASS = 0.005
    from body import flapper
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    from dytiscidae.physics.fluid import WING as KIND_WING

    Phi, Psi, psi0, th = (math.radians(x) for x in (Phi_deg, Psi_deg, psi0_deg, theta_deg))
    w = 2 * math.pi * f
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
    env.reset(Domain.AIR if medium == "air" else Domain.WATER, randomise=False)
    d, m, mj = env.data, env.model, env._mj
    env.solver.cop_law = cop_law
    pan = env.solver.panels
    hull = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "seg0_hull")
    if mass is not None:
        add = mass - float(env.solver._dry_mass.sum())
        env.solver._dry_mass[hull] += add
        m.body_mass[hull] += add
        mj.mj_setConst(m, d)
    total = float(env.solver._dry_mass.sum())

    jid, role = [], []
    for a in env.act_names:
        aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, a)
        m.actuator_gainprm[aid, :] = 0.0
        m.actuator_biasprm[aid, :] = 0.0
        j = int(m.actuator_trnid[aid, 0])
        jid.append(j)
        role.append("feather" if a.endswith("_f") else "stroke")
    role = np.array(role)
    dof = np.array([int(m.jnt_dofadr[j]) for j in jid])
    if joint_damping is not None:
        m.dof_damping[dof] = joint_damping
        env.solver._damping.base[dof] = joint_damping     # the solver owns dof_damping
    qad = np.array([int(m.jnt_qposadr[j]) for j in jid])
    if feather_axis:
        for a, j in enumerate(jid):
            if role[a] != "feather":
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

    def ref(t):
        s, c = math.sin(w * t), math.cos(w * t)
        q, qd, qdd = np.zeros(len(jid)), np.zeros(len(jid)), np.zeros(len(jid))
        st, fe = role == "stroke", role == "feather"
        q[st], qd[st], qdd[st] = 0.5 * Phi * s, 0.5 * Phi * w * c, -0.5 * Phi * w * w * s
        q[fe], qd[fe], qdd[fe] = psi0 - Psi * c, Psi * w * s, Psi * w * w * c
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

    n_beat = int(round(1.0 / (f * dt)))
    n_hold, n_all = hold_beats * n_beat, (hold_beats + beats) * n_beat
    rec = dict(t=[], P=[], Ps=[], tq=[], err=[], hold=[], pos=[], vel=[], pitch=[])
    t = 0.0
    for k in range(n_all):
        if k == n_hold:
            for ch in free:
                released[rel[ch]] = True
                m.dof_armature[rel[ch]] = 0.0
        q, qd, qdd = ref(t)
        tau = J * qdd + kp * (q - d.qpos[qad]) + kv * (qd - d.qvel[dof])
        d.ctrl[:] = 0.0
        d.xfrc_applied[:] = 0.0
        env.solver.apply(d, d.time)
        # The solver has just rewritten all of qfrc_applied (its implicit
        # damping's explicit half); the drive is added to that, never set.
        d.qfrc_applied[dof] += tau
        vb = d.qvel[:6].copy()
        mj.mj_step(m, d)
        hold = ARM * (d.qvel[:6] - vb) / dt           # wrench the holder did NOT have to: -hold
        t += dt
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
            rec["P"].append(float(tau @ qd))
            rec["Ps"].append(float(tau[role == "stroke"] @ qd[role == "stroke"]))
            rec["tq"].append(float(np.abs(tau[role == "stroke"]).sum()))
            rec["err"].append(float(np.abs(q - d.qpos[qad]).max()))
            rec["hold"].append(hold)
            rec["pos"].append(d.qpos[:3].copy())
            rec["vel"].append(d.qvel[:3].copy())
            qw, qy = d.qpos[3], d.qpos[5]
            rec["pitch"].append(-2.0 * math.atan2(qy, qw))
    T = beats / f
    hold = np.array(rec["hold"])
    vel, pos = np.array(rec["vel"]), np.array(rec["pos"])
    P = np.array(rec["P"])
    out = {"V": V, "theta_deg": theta_deg, "f": f, "Phi_deg": Phi_deg, "Psi_deg": Psi_deg,
           "psi0_deg": psi0_deg, "free": free, "medium": medium, "mass_kg": total,
           "finite": bool(np.isfinite(pos).all() and np.isfinite(P).all()),
           "power_W": float(P.mean()), "stroke_power_W": float(np.mean(rec["Ps"])),
           "crank_eq_peak": float(np.abs(P).max() / w / K85),
           "crank_eq_stroke_only_peak": float(np.abs(rec["Ps"]).max() / w / K85),
           "crank_eq_min": float(P.min() / w / K85),
           "stroke_torque_peak": float(np.max(rec["tq"])),
           "track_err_deg": float(math.degrees(np.max(rec["err"])))}
    # Net force on the machine, gravity included: from the holder where held,
    # from the motion where released.
    net = np.zeros(3)
    for i in range(3):
        net[i] = total * (vel[-1, i] - vel[0, i]) / (rec["t"][-1] - rec["t"][0]) if released[i] \
            else float(hold[:, i].mean())
    out["Fx"], out["Fz"] = float(net[0]), float(net[2] + total * G)
    out["lift_over_weight"] = out["Fz"] / (total * G)
    if released[4]:
        out["pitch_end_deg"] = float(math.degrees(rec["pitch"][-1]) - theta_deg)
    else:
        out["pitch_moment_mean"] = float(hold[:, 4].mean())
        out["pitch_moment_peak"] = float(np.abs(hold[:, 4]).max())
    if free:
        out["dx"], out["dz"] = float(pos[-1, 0] - pos[0, 0] - V * T), float(pos[-1, 2] - pos[0, 2])
        out["vx_end"], out["vz_end"] = float(vel[-1, 0]), float(vel[-1, 2])
        zz = pos[:n_beat, 2] - np.linspace(pos[0, 2], pos[n_beat - 1, 2], n_beat)
        out["bob_mm"] = float(1e3 * (zz.max() - zz.min()))
    if keep_series:
        out["series"] = {k: np.asarray(v).tolist() for k, v in rec.items() if k != "hold"}
    return out
