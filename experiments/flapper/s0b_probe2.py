"""S0b probes B5-B8 (DESIGN_S0b_bold.md, addendum 2026-11-19): the level hover on the sweep stroke.
Probes, not scored measurements.  Every rollout is one row of results/S0b_probes2.jsonl.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/s0b_probe2.py <jobset> [procs]

Body changes against T4 as built (made in `s0b_machine.make`, stated in results/S0b_probes.md):
sweep range +-SWEEP deg, feather range FEATHER deg; the dihedral is a held stroke-hinge angle.
Primary motor: "x4" (the witness curve x4 in torque and speed); "ideal" only as a twin.
A row is ``not_evidence`` if it has a QACC reset, a step on the solver's fluid-force limiter, or a peak hinge
torque above 10x the witness stall.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "S0b_probes2.jsonl"
G = 9.80665
SWEEP = 75.0
FEATHER = (-150.0, 60.0)
BUILD = dict(sweep_range_deg=SWEEP, feather_range_deg=FEATHER)
M_OUT, M_IN = 0.91763805, 1.1590                       # kg: ballast out (b 1.263), ballast in (b 1.000)


def _make(domain, b, motor, dt, z=30.0, x=-30.0, cut=0.0):
    """``cut``: T4c, the inner ``cut`` metres of each main wing absent (`s0b_machine`, root_cut).  b must be
    None with a cut (the keel calibration is T4's)."""
    import s0b_machine as T
    assert not (cut and b), "ballast is calibrated for T4 only"
    return T.make(domain, b=b, dt=dt, motor=motor, pose=dict(x=x, y=0.0, z=z, vx=0.0, vz=0.0), root_cut=cut, **BUILD)


def _pairs(mc):
    """Geom pairs for the clearance read: wing-wing, wing-hull, wing-tail (tail surfaces and boom)."""
    m, mj = mc.m, mc.mj
    name = lambda g: mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
    body = lambda g: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
    wing = [g for g in range(m.ngeom) if body(g) in ("seg1_wing", "seg2_wing", "seg3_wing", "seg4_wing")]
    hull = [g for g in range(m.ngeom) if body(g) == "seg0_hull" and name(g) != "s0b_keel"]
    tail = [g for g in range(m.ngeom) if body(g) in ("seg5_strut", "seg6_wing", "seg7_wing")]
    P = dict(wing_wing=[(a, b) for i, a in enumerate(wing) for b in wing[i + 1:]],
             wing_hull=[(a, b) for a in wing for b in hull], wing_tail=[(a, b) for a in wing for b in tail])
    return P


def clearance(mc, pairs, cur=None):
    """Minimum signed distance (m; negative = interpenetration) per class at the present pose."""
    out = dict(cur or {})
    ft = np.zeros(6)
    for k, pp in pairs.items():
        dmin = min(mc.mj.mj_geomDistance(mc.m, mc.d, a, b, 1.0, ft) for a, b in pp)
        out[k] = min(out.get(k, 9.0), float(dmin))
    return out


def wing_low(mc):
    """Lowest world z of any corner of the four main-wing boxes."""
    m, d = mc.m, mc.d
    lo = 9e9
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) in (2, 3, 4, 5):
            c, Rg, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    lo = min(lo, float((c + Rg @ (np.array([sx, sy, -1]) * s))[2]), float((c + Rg @ (np.array([sx, sy, 1]) * s))[2]))
    return lo


def _wet(mc):
    """(wet main-wing strips, wet strips per wing [FL, FR, HL, HR], all wet panels, sum of the main wings' submerged fraction)."""
    s, p, d = mc.env.solver, mc.env.solver.panels, mc.d
    R = d.xmat.reshape(-1, 3, 3)[p.body_id]
    pos = d.xpos[p.body_id] + np.einsum("nij,nj->ni", R, p.pos_local)
    sub = mc.env.medium.submerged_fraction(pos, p.half_height, d.time)
    per = [int(((sub > 0) & (p.body_id == b)).sum()) for b in (2, 3, 4, 5)]
    main = np.isin(p.body_id, [2, 3, 4, 5])
    return int(sum(per)), per, int((sub > 0).sum()), float(sub[main].sum())


def _flags(row):
    import s0b_machine as T
    over = {r: v > 10.0 * T.WITNESS_STALL[r] for r, v in row["tq_peak"].items()}
    row["tq_over_witness_x1"] = {r: round(v / T.WITNESS_STALL[r], 3) for r, v in row["tq_peak"].items()}
    row["artifact"] = bool(any(over.values()))
    row["not_evidence"] = bool(row["bad_qacc"] or row["clamped_steps"] or row["artifact"])
    return row


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def held(job) -> dict:
    """Hull held level (pose and velocity rewritten every step) in air at 30 m, or on the water at a given
    hull height ``z_hull`` (B6's stroke-through-the-float read).  Mean fluid wrench over whole beats, hull axes
    (x nose, y left, z up = thrust)."""
    import s0b_machine as T
    import s0b_laws as LW
    t_wall = time.time()
    medium, f = job.get("medium", "air"), job["f"]
    P = dict(LW.LEVEL, **job.get("P", {}))
    still = bool(job.get("still"))
    b = job.get("b", None)
    mc = _make("air" if medium == "air" else "water", b, job.get("motor", "x4"), job.get("dt", 1e-4),
               z=30.0 if medium == "air" else job.get("z_hull", 0.0), cut=job.get("cut", 0.0))
    d, s = mc.d, mc.env.solver
    q0 = d.qpos[:7].copy()
    q0[3:7] = (1.0, 0.0, 0.0, 0.0)
    if job.get("pose_quat"):
        q0[3:7] = job["pose_quat"]
    d.qpos[:7] = q0
    d.qvel[:] = 0.0
    w = 2 * math.pi * f
    U = np.array(job.get("U", (0.0, 0.0, 0.0)), dtype=float)     # hull velocity, world (the hull is level: = hull axes)
    qr, _ = LW.targets_level(mc, 0.0, w, P, osc=0.0)
    d.qpos[mc.qad] = qr                                     # start raised: the held pose of the law at osc 0
    if medium != "air":
        s.entrainment = job.get("entrainment", True)
    mc.mj.mj_forward(mc.m, d)
    T.regain(mc)
    s.reset()
    nb = int(round(1.0 / (f * mc.dt)))
    beats, skip = job.get("beats", 7), job.get("skip", 3)
    ramp_beats = job.get("ramp_beats", 2.0)
    F, M = [], []
    tq, om = np.zeros(len(mc.aid)), np.zeros(len(mc.aid))
    clamped = 0
    Epos, E0 = 0.0, None
    pairs = _pairs(mc)
    clr = clearance(mc, pairs)
    clr_rest = dict(clr)
    t = 0.0
    feath, wetmax, wetper, low = [], 0, [0, 0, 0, 0], 9e9
    err = []
    for k in range(beats * nb):
        if k == skip * nb:
            E0 = mc.E_shaft.sum()
            clr = clearance(mc, pairs)                      # the clearance read starts with the measured window
        osc = 0.0 if still else _smooth(t * f / ramp_beats)
        q, qd = LW.targets_level(mc, w * t, w, P, osc=osc)
        d.qpos[:7] = q0
        d.qpos[:3] = q0[:3] + U * t
        d.qvel[:6] = 0.0
        d.qvel[:3] = U
        qv0 = d.qvel[mc.dof].copy()
        mc.step(q, qd * (0.0 if still else 1.0))
        t += mc.dt
        if s.diag.clamped:
            clamped += 1
            s.diag.clamped = False
        if k >= skip * nb:
            Fk, Mk = T.fluid_wrench(mc)
            F.append(Fk)
            M.append(Mk)
            af = d.actuator_force[mc.aid]
            np.maximum(tq, np.abs(af), out=tq)
            np.maximum(om, np.abs(qv0), out=om)
            Epos += float(np.clip(af * qv0, 0, None).sum()) * mc.dt
            if k % 25 == 0:
                clr = clearance(mc, pairs, clr)
                feath.append(float(np.abs(d.qpos[mc.qad][mc.role == "feather"]).max()))
                err.append(d.qpos[mc.qad] - q)
                if medium != "air":
                    wm, per, _, _ = _wet(mc)
                    wetmax = max(wetmax, wm)
                    wetper = [max(a, c) for a, c in zip(wetper, per)]
                    low = min(low, wing_low(mc))
    F, M = np.array(F), np.array(M)
    d.qpos[:7] = q0
    mc.mj.mj_forward(mc.m, d)
    R = d.xmat[mc.hull].reshape(3, 3)
    Fh, Mh = F.mean(0) @ R, M.mean(0) @ R
    nbt = beats - skip
    Fb = (F @ R).reshape(nbt, nb, 3).mean(1)
    roles = sorted(set(mc.role))
    Tm = nbt / f
    err = np.array(err)
    row = dict(job, kind="held", mass=mc.mass, weight=mc.mass * G, F_hull=Fh.tolist(), M_hull=Mh.tolist(),
               thrust_W_out=float(Fh[2] / (M_OUT * G)), thrust_W_in=float(Fh[2] / (M_IN * G)),
               F_hull_beat_sd=Fb.std(0).tolist(), F_hull_peak=np.abs(F @ R).max(0).tolist(),
               shaft_W=float((mc.E_shaft.sum() - E0) / Tm), shaft_pos_W=Epos / Tm,
               tq_peak={r: float(tq[mc.role == r].max()) for r in roles},
               om_peak={r: float(om[mc.role == r].max()) for r in roles},
               track_rms_deg={r: float(np.degrees(np.sqrt((err[:, mc.role == r] ** 2).mean()))) for r in roles},
               at_limit={r: float(mc.at_limit[mc.role == r].mean() / max(mc.n, 1)) for r in roles},
               clearance_min=clr, clearance_rest=clr_rest,
               x2_share=float(np.mean(np.array(feath) > math.radians(30.0))), x2=True,
               clamped_steps=clamped, bad_qacc=int(mc.bad_qacc), finite=bool(np.isfinite(F).all()),
               wall_s=round(time.time() - t_wall, 1))
    if medium != "air":
        row.update(wet_main_max=wetmax, wet_per_wing_max=wetper, wing_low_min=low)
    row["x2"] = bool(row["x2_share"] > 0.5)                 # airspeed is 0 < 6 m/s on a held row
    return _flags(row)


PRINT = ("probe", "id", "f", "mode", "motor", "thrust_W_out", "thrust_W_in", "F_hull", "M_hull", "shaft_pos_W", "tq_peak",
         "clearance_min", "x2_share", "clamped_steps", "bad_qacc", "not_evidence", "wall_s", "wet_main_max", "wing_low_min",
         "summary")


def run(jobs, n):
    rows = []
    with mp.get_context("spawn").Pool(min(n, 4)) as pool:
        for r in pool.imap(_dispatch, jobs, chunksize=1):
            with open(OUT, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
            rows.append(r)

            def rnd(v):
                if isinstance(v, float):
                    return round(v, 3)
                if isinstance(v, list):
                    return np.round(v, 3).tolist()
                if isinstance(v, dict):
                    return {k: rnd(x) for k, x in v.items()}
                return v
            print(json.dumps({k: rnd(r[k]) for k in PRINT if k in r}), flush=True)
    return rows


def _dispatch(job):
    return globals()[job.get("fn", "held")](job)


def D(**kw):
    """Law parameters given in degrees for the angle keys."""
    return {k: (math.radians(v) if k in ("A", "dih", "dih_fh", "sw0", "sw_fh", "sw_lr", "Psi", "psi0", "psi0_lr",
                                         "psi0_fh", "tilt", "tilt_lr", "adv", "Psi_lr", "Psi_fh") else v)
            for k, v in kw.items()}


def jobs_b5a():
    """First look: both flip modes at the centre point, the still machine, and the two obvious clearance layouts."""
    J = [dict(probe="B8", id="still_air", f=4.5, still=True, beats=3, skip=1)]
    for flip in ("vert", "flat"):
        for sf in (0.0, 30.0):
            J.append(dict(probe="B5", id=f"c_{flip}_swfh{sf:.0f}", f=4.5, P=D(flip=flip, A=55, dih=35, Psi=45, sw_fh=sf)))
    J.append(dict(probe="B5", id="c_vert_dihfh20", f=4.5, P=D(flip="vert", A=55, dih=35, Psi=45, dih_fh=20)))
    J.append(dict(probe="B5", id="c_vert_ideal", f=4.5, motor="ideal", P=D(flip="vert", A=55, dih=35, Psi=45)))
    return J


def jobs_b5grid():
    """B5: thrust map, through-vertical flip, both pairs in phase: dihedral x sweep half-amplitude x frequency x mid-stroke pitch."""
    J = []
    for dih in (20, 35, 50):
        for A in (40, 55, 70):
            for f in (2.5, 4.5, 6.0):
                for Psi in (30, 45, 60):
                    J.append(dict(probe="B5", id=f"g_d{dih}_A{A}_f{f}_P{Psi}", f=f, beats=6, skip=3,
                                  P=D(flip="vert", A=A, dih=dih, Psi=Psi)))
    return J


def _att(R):
    """(roll deg: left side up +, pitch deg: nose up +, yaw deg) of the hull."""
    return (math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(math.asin(min(max(R[2, 0], -1.0), 1.0))),
            math.degrees(math.atan2(R[1, 0], R[0, 0])))


def _quat(roll_deg=0.0, pitch_up_deg=0.0):
    """Hull quaternion: pitch (nose up) about y, then roll about the hull's own x."""
    a, r = math.radians(-pitch_up_deg), math.radians(roll_deg)
    qy = np.array([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0])
    qx = np.array([math.cos(r / 2), math.sin(r / 2), 0.0, 0.0])
    w1, x1, y1, z1 = qy
    w2, x2, y2, z2 = qx
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _hull_wet(mc):
    """Submerged fraction of each hull panel, and of the tail's."""
    p, d = mc.env.solver.panels, mc.d
    R = d.xmat.reshape(-1, 3, 3)[p.body_id]
    pos = d.xpos[p.body_id] + np.einsum("nij,nj->ni", R, p.pos_local)
    sub = mc.env.medium.submerged_fraction(pos, p.half_height, d.time)
    return sub[p.body_id == 1].round(3).tolist(), sub[np.isin(p.body_id, [7, 8])].round(3).tolist()


def floatq(job) -> dict:
    """B6: the float.  Free on the water, actuators holding the rest pose of the level law (wings raised to
    the dihedral, flat).  From level at rest (or pushed: ``roll0`` / ``pitch0`` deg about the CoM), T seconds."""
    import s0b_machine as T
    import s0b_laws as LW
    t_wall = time.time()
    P = dict(LW.LEVEL, **job.get("P", {}))
    mc = _make("water", job.get("b"), job.get("motor", "x4"), job.get("dt", 1e-4), z=job.get("z0", 0.02),
               cut=job.get("cut", 0.0))
    d, s = mc.d, mc.env.solver
    s.entrainment = job.get("entrainment", True)
    qr, _ = LW.targets_level(mc, 0.0, 1.0, P, osc=0.0)
    if job.get("wings_flat"):
        qr = np.zeros(len(mc.aid))
    d.qpos[mc.qad] = qr
    d.qpos[3:7] = _quat(job.get("roll0", 0.0), job.get("pitch0", 0.0))
    d.qvel[:] = 0.0
    mc.mj.mj_forward(mc.m, d)
    if "com_z0" in job:
        d.qpos[2] += job["com_z0"] - float(mc.com()[2])
        mc.mj.mj_forward(mc.m, d)
    T.regain(mc)
    s.reset()
    N = int(round(job["T"] / mc.dt))
    tr, clamped = [], 0
    tq = np.zeros(len(mc.aid))
    wetmax = 0
    for k in range(N):
        mc.step(qr)
        np.maximum(tq, np.abs(d.actuator_force[mc.aid]), out=tq)
        if s.diag.clamped:
            clamped += 1
            s.diag.clamped = False
        if k % 500 == 0 or k == N - 1:
            R = d.xmat[mc.hull].reshape(3, 3)
            wm, per, wa, wsum = _wet(mc)
            wetmax = max(wetmax, wm)
            tr.append([round(mc.t, 3)] + [round(float(v), 4) for v in mc.com()] + [round(v, 2) for v in _att(R)]
                      + [wm, round(wing_low(mc), 4), round(float(d.xpos[mc.hull][2]), 4)])
        if mc.bad_qacc or not np.all(np.isfinite(d.qpos)):
            break
    R = d.xmat[mc.hull].reshape(3, 3)
    wm, per, wa, wsum = _wet(mc)
    hw, tw = _hull_wet(mc)
    a = np.array(tr)
    last = a[a[:, 0] >= a[-1, 0] - 1.0]
    roles = sorted(set(mc.role))
    settled = dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), com=mc.com().copy(), att=_att(R), wing_low=wing_low(mc),
                   hull_z=float(d.xpos[mc.hull][2]), wet=per, t=mc.t)
    pushes = {}
    for name, axis, ang, T2 in job.get("push", []):          # a push from the settled state: rotate about the
        d.qpos[:] = settled["qpos"]                          # hull's own x (roll) or y (pitch, + = nose down), at rest
        h = math.radians(ang) / 2
        dq = np.array([math.cos(h), math.sin(h), 0, 0]) if axis == "roll" else np.array([math.cos(h), 0, math.sin(h), 0])
        w1, x1, y1, z1 = d.qpos[3:7]
        w2, x2, y2, z2 = dq
        d.qpos[3:7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                       w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
        d.qvel[:] = 0.0
        mc.mj.mj_forward(mc.m, d)
        d.qpos[:3] += settled["com"] - mc.com()
        mc.mj.mj_forward(mc.m, d)
        s.reset()
        ptr = []
        for k in range(int(round(T2 / mc.dt))):
            mc.step(qr)
            if s.diag.clamped:
                clamped += 1
                s.diag.clamped = False
            if k % 500 == 0:
                Rk = d.xmat[mc.hull].reshape(3, 3)
                ptr.append([round(k * mc.dt, 2)] + [round(v, 2) for v in _att(Rk)] + [round(float(mc.com()[2]), 4), _wet(mc)[0]])
            if mc.bad_qacc:
                break
        pa = np.array(ptr)
        pushes[name] = dict(start=ptr[0][1:4], end=ptr[-1][1:4], roll_range=[float(pa[:, 1].min()), float(pa[:, 1].max())],
                            pitch_range=[float(pa[:, 2].min()), float(pa[:, 2].max())], wet_max=int(pa[:, 5].max()),
                            trace_cols="t roll pitch yaw com_z wet_main", trace=ptr)
    d.qpos[:], d.qvel[:] = settled["qpos"], settled["qvel"]
    mc.mj.mj_forward(mc.m, d)
    R = d.xmat[mc.hull].reshape(3, 3)
    row = dict(job, kind="float", pushes=pushes, mass=mc.mass, b=11.366116705 / (mc.mass * G), t_end=mc.t,
               com_end=mc.com().tolist(), hull_z_end=float(d.xpos[mc.hull][2]), att_end=list(_att(R)),
               att_last1s_range=[float(np.ptp(last[:, 4])), float(np.ptp(last[:, 5]))],
               roll_max=float(np.abs(a[:, 4]).max()), pitch_max=float(np.abs(a[:, 5]).max()),
               wet_main_end=wm, wet_per_wing_end=per, wet_all_end=wa, wet_main_sum_frac=wsum, wet_main_max=wetmax,
               wing_low_end=wing_low(mc), hull_panel_sub=hw, tail_panel_sub=tw, quat_end=d.qpos[3:7].tolist(),
               hull_pos_end=d.qpos[:3].tolist(),
               tq_peak={r: float(tq[mc.role == r].max()) for r in roles}, clamped_steps=clamped,
               bad_qacc=int(mc.bad_qacc), trace_cols="t x y z roll pitch yaw wet_main wing_low hull_z", trace=tr,
               wall_s=round(time.time() - t_wall, 1))
    row["summary"] = dict(hull_z=round(row["hull_z_end"], 4), att=[round(v, 1) for v in row["att_end"]], wet=per,
                          wing_low=round(row["wing_low_end"], 3), roll_max=round(row["roll_max"], 1),
                          pitch_max=round(row["pitch_max"], 1), last1s=[round(v, 2) for v in row["att_last1s_range"]],
                          push={k: dict(start=v["start"], end=v["end"], roll=v["roll_range"], pitch=v["pitch_range"])
                                for k, v in pushes.items()})
    return _flags(row)


def _modes():
    return (("default", True), ("wetting", "wetting"))


def jobs_b6a():
    """The float at rest from level: b x dihedral x mode (6 s)."""
    J = []
    for b in (None, 1.15, 1.08, 1.0):
        for dih in (35, 60):
            for mode, ent in _modes():
                J.append(dict(fn="floatq", probe="B6", id=f"rest_b{b}_d{dih}_{mode}", b=b, mode=mode, entrainment=ent,
                              T=6.0, P=D(dih=dih)))
    for mode, ent in _modes():
        J.append(dict(fn="floatq", probe="B6", id=f"rest_bNone_flat_{mode}", b=None, mode=mode, entrainment=ent, T=6.0,
                      wings_flat=True))
    return J


def fly(job) -> dict:
    """B7: a free lift-off, hold and descent on the level-hover law (`s0b_laws.Hover4`).

    start "float" (on the water, b as given, CoM at ``com_z0``), "beach" (on the ramp at x = ``x0``), "air"
    (at rest at ``z0``, wings raised: a hover test without the surface).  Timeline: settle (wings at the rest
    pose, blended up from flat over ``raise_s``) -> the stroke ramped in over ``ramp_beats`` with the loop on,
    height target rising at ``v_up`` to ``H`` above the start -> ``hold_s`` from the first time within 0.2 m ->
    height target falling at ``v_dn`` until touch-down (contact, or the CoM back at its start height + 0.03) ->
    the stroke ramped out over ``out_s`` -> ``after_s`` more.  ``still``: actuators hold the rest pose throughout."""
    import s0b_machine as T
    import s0b_laws as LW
    t_wall = time.time()
    start, dt = job["start"], job.get("dt", 1e-4)
    still = bool(job.get("still"))
    cfg = job["law"]
    P = dict(LW.LEVEL, **cfg.get("P", {}))
    dom = dict(float="water", beach="land", air="air")[start]
    mc = _make(dom, job.get("b"), job.get("motor", "x4"), dt, z=job.get("z0", 10.0 if start == "air" else 0.02),
               x=job.get("x0", 20.0) if start == "beach" else -30.0, cut=job.get("cut", 0.0))
    d, s = mc.d, mc.env.solver
    over_water = start == "float" or bool(job.get("over_water"))     # "air" starts over the sea at x = -30
    if over_water:
        s.entrainment = job.get("entrainment", True)
    w = 2 * math.pi * cfg["f"]
    # pose0: the dihedral / mean sweep the wings are held at on the surface (deg), blended to the law's own
    # as the stroke ramps in ("osc") or with height over the start ("height": 0 at 0.1 m, 1 at 0.5 m)
    pose0 = {k: math.radians(v) for k, v in (job.get("pose0") or {}).items()}
    P_fly = {k: P[k] for k in pose0}
    q_rest, _ = LW.targets_level(mc, 0.0, w, dict(P, **pose0), osc=0.0)
    q_flat = np.zeros(len(mc.aid))
    q_start = q_rest if (start == "air" or job.get("start_raised")) else q_flat
    d.qpos[mc.qad] = q_start
    d.qvel[:] = 0.0
    if start == "beach":
        from dytiscidae.core.mjcf import BEACH_SLOPE
        d.qpos[3:7] = _quat(0.0, math.degrees(math.atan(BEACH_SLOPE)) if job.get("along_slope", True) else 0.0)
        mc.mj.mj_forward(mc.m, d)
        mc.drop_to_ground(0.003)
    else:
        d.qpos[3:7] = _quat(0.0, 0.0)
    mc.mj.mj_forward(mc.m, d)
    if start == "float" and "com_z0" in job:
        d.qpos[2] += job["com_z0"] - float(mc.com()[2])
        mc.mj.mj_forward(mc.m, d)
    T.regain(mc)
    s.reset()
    I = np.array(T.report(mc)["inertia_hull_diag"])
    settle, raise_s = job.get("settle_s", 1.0 if start != "air" else 0.0), job.get("raise_s", 0.4)
    ramp_beats, H, hold_s = job.get("ramp_beats", 3.0), job.get("H", 2.0), job.get("hold_s", 10.0)
    v_up, v_dn, out_s, after_s = job.get("v_up", 0.8), job.get("v_dn", 0.4), job.get("out_s", 0.4), job.get("after_s", 3.0)
    Tmax = job.get("T", 40.0)
    dwell, lower_s = job.get("dwell_s", 0.0), job.get("lower_s", 2.0)
    con_ph, conw_ph = {}, {}
    law = None
    phase = "settle"
    n = len(mc.aid)
    tq, om = np.zeros(n), np.zeros(n)
    tr, ev = [], {}
    clamped, contact_W, contact_W_wing, Epos = 0, 0.0, 0.0, 0.0
    x2_n = x2_k = 0
    z_start = None
    t_ramp = t_hold = t_out = None
    Ht = 0.0
    hold = dict(z=[], xy=[], up=[])
    wetmax_fly = 0
    q = q_start.copy()
    N = int(round(Tmax / dt))
    wing_geoms = set(g for g in range(mc.m.ngeom) if int(mc.m.geom_bodyid[g]) in (2, 3, 4, 5))
    for k in range(N):
        t = mc.t
        com = mc.com()
        qd = np.zeros(n)
        if phase == "settle":
            r = _smooth((t - (settle - raise_s - dwell)) / raise_s) if raise_s > 0 else 1.0
            q = q_start + (q_rest - q_start) * r
            if t >= settle:
                z_start = float(com[2])
                ev["start"] = dict(t=t, com=com.tolist(), att=list(_att(d.xmat[mc.hull].reshape(3, 3))),
                                   wet=_wet(mc)[1] if over_water else None)
                if still:
                    phase = "still"
                else:
                    phase = "up"
                    t_ramp = t
                    law = LW.Hover4(mc, cfg["f"], cfg["mix"], cfg["chan"], I, cfg["a0"], P=P, H_t=z_start,
                                    **cfg.get("gains", {}))
        if phase in ("up", "hold", "down", "out"):
            osc = _smooth((t - t_ramp) * cfg["f"] / ramp_beats)
            if phase == "up":
                Ht = min(H, v_up * max(t - t_ramp - job.get("climb_delay_s", 0.0), 0.0))
                if com[2] >= z_start + H - 0.2:
                    phase, t_hold = "hold", t
                    ev["at_height"] = dict(t=t)
            elif phase == "hold":
                Ht = H
                if k % 100 == 0:
                    hold["z"].append(float(com[2]))
                    hold["xy"].append(com[:2].tolist())
                    hold["up"].append(float(d.xmat[mc.hull].reshape(3, 3)[2, 2]))
                if t - t_hold >= hold_s:
                    phase = "down"
                    t_dn, H_dn = t, float(com[2]) - z_start
                    ev["descend"] = dict(t=t)
                    if not job.get("descend", True):
                        break
            elif phase == "down":
                Ht = H_dn - v_dn * (t - t_dn)
                touch = (mc.contact_load() > 0.05) if d.ncon else False
                if dom != "land":
                    touch = touch or (com[2] <= job.get("touch_z", 0.03 + (z_start if dom == "water" else -9e9)))
                if dom == "land":
                    Ht = max(Ht, -50.0)
                if touch:
                    phase, t_out = "out", t
                    vb = np.array([b_[2] for b_ in law.buf]) if law.buf else np.zeros((2, 3))
                    ev["touch"] = dict(t=t, vz=float(d.qvel[2]), v=d.qvel[:3].tolist(), com=com.tolist(),
                                       v_beat_mean=((vb[-1] - vb[0]) / max((len(vb) - 1) * 10 * dt, 1e-9)).tolist(),
                                       att=list(_att(d.xmat[mc.hull].reshape(3, 3))))
            if phase == "out":
                osc = osc * (1.0 - _smooth((t - t_out) / out_s))
                if t - t_out >= out_s + after_s:
                    break
            law.H_t = z_start + Ht if phase != "out" else law.H_t
            if pose0:
                pb = job.get("pose_blend", "osc")                 # one rule, or one per key
                bl_h = min(osc, _smooth((float(com[2]) - z_start - 0.1) / 0.4))
                for k_ in pose0:
                    bl = osc if (pb.get(k_, "osc") if isinstance(pb, dict) else pb) == "osc" else bl_h
                    val = pose0[k_] + (P_fly[k_] - pose0[k_]) * bl
                    if k_ in law.base:
                        law.base[k_] = val
                    else:
                        law.P[k_] = val
            law.step()
            q, qd = law.targets(osc=osc)
            qd = qd * osc
            if phase == "out" and t - t_out > out_s and lower_s > 0:      # then the wings are laid flat again
                r = _smooth((t - t_out - out_s) / lower_s)
                q, qd = q + (q_flat - q) * r, qd * 0.0
        qv0 = d.qvel[mc.dof].copy()
        mc.step(q, qd)
        af = d.actuator_force[mc.aid]
        np.maximum(tq, np.abs(af), out=tq)
        np.maximum(om, np.abs(qv0), out=om)
        Epos += float(np.clip(af * qv0, 0, None).sum()) * dt
        if d.ncon:
            cl = mc.contact_load()
            contact_W = max(contact_W, cl)
            con_ph[phase] = max(con_ph.get(phase, 0.0), cl)
            if cl > 0 and any((d.contact[i].geom1 in wing_geoms or d.contact[i].geom2 in wing_geoms) for i in range(d.ncon)):
                contact_W_wing = max(contact_W_wing, cl)
                conw_ph[phase] = max(conw_ph.get(phase, 0.0), cl)
        if s.diag.clamped:
            clamped += 1
            s.diag.clamped = False
        if k % 50 == 0 and phase in ("up", "hold", "down"):
            x2_n += 1
            x2_k += int(np.linalg.norm(d.qvel[:3]) < 6.0 and np.abs(d.qpos[mc.qad][mc.role == "feather"]).max() > math.radians(30.0))
        if k % 200 == 0:
            R = d.xmat[mc.hull].reshape(3, 3)
            wm = _wet(mc)[0] if over_water else 0
            if phase in ("up", "hold", "down") and over_water:
                wetmax_fly = max(wetmax_fly, wm)
            tr.append([round(t, 3)] + [round(float(v), 3) for v in com] + [round(float(d.qvel[2]), 3)]
                      + [round(v, 1) for v in _att(R)] + [round(float(law.out["amp"]), 3) if law else 0.0, wm,
                                                           round(mc.contact_load(), 2) if d.ncon else 0.0, phase]
                      + ([round(float(law.out[c]), 3) for c in law.chan] if law else [0.0, 0.0, 0.0]))
        if not np.all(np.isfinite(d.qpos)) or mc.bad_qacc:
            ev["abort"] = dict(t=t, why="qacc" if mc.bad_qacc else "nan")
            break
        if phase in ("up", "hold", "down") and (d.xmat[mc.hull].reshape(3, 3)[2, 2] < 0.0):
            ev["abort"] = dict(t=t, why="inverted")
            break
    roles = sorted(set(mc.role))
    R = d.xmat[mc.hull].reshape(3, 3)
    hz, hxy = np.array(hold["z"]), np.array(hold["xy"])
    row = dict(job, kind="fly", mass=mc.mass, weight=mc.mass * G, t_end=mc.t, phase_end=phase, events=ev,
               z_start=z_start, com_end=mc.com().tolist(), att_end=list(_att(R)), vel_end=d.qvel[:3].tolist(),
               z_max_rel=float(max(r_[3] for r_ in tr) - (z_start or 0.0)),
               hold_s_done=float(mc.t - t_hold) if (t_hold is not None and phase == "hold") else (hold_s if t_hold is not None else 0.0),
               hold_z_rel=[float(hz.min() - z_start), float(hz.max() - z_start)] if len(hz) else None,
               hold_drift_m=float(np.linalg.norm(hxy - hxy[0], axis=1).max()) if len(hz) else None,
               hold_tilt_max_deg=float(np.degrees(np.arccos(np.clip(min(hold["up"]), -1, 1)))) if len(hz) else None,
               tq_peak={r: float(tq[mc.role == r].max()) for r in roles},
               om_peak={r: float(om[mc.role == r].max()) for r in roles},
               at_limit={r: float(mc.at_limit[mc.role == r].mean() / max(mc.n, 1)) for r in roles},
               contact_peak_W=contact_W, contact_peak_W_wing=contact_W_wing, contact_peak_W_by_phase=con_ph,
               contact_peak_W_wing_by_phase=conw_ph,
               rigid_contact_figure=bool(max([v for k, v in con_ph.items() if k != "settle"] or [0.0]) > 10.0), E_shaft_J=float(mc.E_shaft.sum()),
               E_pos_J=Epos, mean_pos_W=Epos / max(mc.t - (t_ramp or 0.0), 1e-9),
               x2_share=(x2_k / x2_n) if x2_n else 0.0, x2=bool(x2_n and x2_k / x2_n > 0.5),
               law_sat=(law.sat / max(law.n_upd, 1)) if law is not None else None, wet_main_max_flying=wetmax_fly,
               clamped_steps=clamped, bad_qacc=int(mc.bad_qacc), wet_end=_wet(mc)[1] if over_water else None,
               trace_cols="t x y z vz roll pitch yaw amp wet_main contact_W phase u_roll u_pitch u_yaw", trace=tr,
               wall_s=round(time.time() - t_wall, 1))
    row["summary"] = dict(end=phase, t=round(mc.t, 2), zmax=round(row["z_max_rel"], 2), hold=round(row["hold_s_done"], 1),
                          hold_z=[round(v, 2) for v in row["hold_z_rel"]] if row["hold_z_rel"] else None,
                          drift=round(row["hold_drift_m"], 2) if row["hold_drift_m"] is not None else None,
                          tilt=round(row["hold_tilt_max_deg"], 1) if row["hold_tilt_max_deg"] is not None else None,
                          touch=ev.get("touch", {}).get("vz"), att_end=[round(v, 1) for v in row["att_end"]],
                          contact={k: round(v, 1) for k, v in con_ph.items()},
                          wing_contact={k: round(v, 1) for k, v in conw_ph.items()}, sat=row["law_sat"],
                          abort=ev.get("abort"), x2=row["x2_share"], wet_fly=wetmax_fly)
    return _flags(row)


#: the hover point the moment map is taken at: 3 Hz, sweep half-amplitude 40 deg (amp 0.727 of 55), dihedral 35, pitch 45
HOV = dict(flip="vert", A=55, dih=35, Psi=45)
A_HOV = 40.0 / 55.0


def jobs_b5mom():
    """B5: moments per input at the hover point (difference from the base row), and the kinematic variations."""
    base = dict(D(**HOV), amp=A_HOV)
    J = [dict(probe="B5", id="m_base", f=3.0, P=base), dict(probe="B5", id="m_base_b1.0", f=3.0, b=1.0, P=base),
         dict(probe="B8", id="still_air_raised", f=3.0, still=True, beats=3, skip=1, P=base),
         dict(probe="B5", id="m_base_ideal", f=3.0, motor="ideal", P=base)]
    for k, v in (("amp", A_HOV * 1.2), ("amp_fh", 0.2), ("amp_lr", 0.2), ("sw0", 10), ("sw_fh", 10), ("sw_lr", 10),
                 ("psi0", 10), ("psi0_lr", 10), ("psi0_fh", 10), ("tilt", 10), ("tilt_lr", 10), ("dih_fh", 10),
                 ("Psi_lr", 10), ("Psi_fh", 10), ("adv", 15), ("adv", -15), ("flip_k", 2.0)):
        J.append(dict(probe="B5", id=f"m_{k}={v:.3g}", f=3.0, P=dict(base, **D(**{k: v}))))
    J.append(dict(probe="B5", id="m_lr_phase=pi", f=3.0, P=dict(base, lr_phase=math.pi)))
    J.append(dict(probe="B5", id="m_hind_phase=pi/2", f=3.0, P=dict(base, hind_phase=math.pi / 2)))
    J.append(dict(probe="B5", id="m_flat", f=3.0, P=dict(base, flip="flat")))
    return J


PUSH = [("roll+20", "roll", 20.0, 8.0), ("pitch_nose_down20", "pitch", 20.0, 8.0), ("pitch_nose_up20", "pitch", -20.0, 8.0)]


def jobs_b6b():
    """The float settled for 20 s, then pushed 20 deg in roll and in pitch from the settled state (8 s each)."""
    J = []
    for mode, ent in _modes():
        J.append(dict(fn="floatq", probe="B6", id=f"settle_bNone_d35_{mode}", b=None, mode=mode, entrainment=ent,
                      T=20.0, P=D(dih=35), push=PUSH))
        J.append(dict(fn="floatq", probe="B6", id=f"settle_bNone_flat_{mode}", b=None, mode=mode, entrainment=ent,
                      T=8.0, wings_flat=True, push=PUSH))
    return J


#: B5's moment map at the hover point (N m per unit / per rad, hull axes): roll Mx = +4.88 amp_lr, pitch
#: My = -4.10 sw0, yaw Mz = +3.79 psi0_lr, taken at 9.84 N of thrust (3 Hz, amp 0.727 of A 55 deg).
CHAN = ("amp_lr", "sw0", "psi0_lr")
MIX = [[1.0 / 4.88, 0.0, 0.0], [0.0, -1.0 / 4.10, 0.0], [0.0, 0.0, 1.0 / 3.79]]
SW0_TRIM = dict(out=0.419 / 4.10, inn=0.622 / 4.10)          # rad: the base row's My trimmed by the mean sweep


def law_cfg(f=3.0, b=None, gains=None, P=None, a_scale=1.0):
    """The level-hover law at frequency f: the hover amplitude from B5 (thrust ~ (amp f)^2 about the 3 Hz point)."""
    W = (M_OUT if b is None else 11.366116705 / (G * b)) * G
    a0 = A_HOV * (3.0 / f) * math.sqrt(W / 9.84) * a_scale
    Pd = dict(D(**HOV), sw0=SW0_TRIM["out"] if b is None else SW0_TRIM["inn"])
    Pd.update(P or {})
    g = dict(wn=(8.0, 8.0, 4.0), zeta=0.8, u_lim=(0.4, 0.2, 0.3), amp_lim=(0.25 * 3.0 / f, 1.0 * 3.0 / f), k_v=0.15,
             tilt_max=0.25, kp_vz=0.10 * 3.0 / f, ki_vz=0.15 * 3.0 / f)
    g.update(gains or {})
    return dict(f=f, a0=a0, mix=MIX, chan=CHAN, P=Pd, gains=g)


#: T4c (root cut 0.11 m, 0.8903 kg) at dihedral 50, 4.5 Hz, amp 0.6 of A 55 (11.36 N): roll +5.95 amp_lr,
#: pitch -5.09 sw0, yaw +3.17 psi0_lr; base My +0.549.  Scaled to the hover thrust (8.731 N).
CUT = 0.11
M_C = 0.8903


def law_cfg_c(gains=None, P=None, a_scale=1.0):
    k = M_C * G / 11.361
    a0 = 0.6 * k ** (1.0 / 1.75) * a_scale
    mix = [[1.0 / (5.95 * k), 0.0, 0.0], [0.0, -1.0 / (5.09 * k), 0.0], [0.0, 0.0, 1.0 / (3.17 * k)]]
    Pd = dict(D(flip="vert", A=55, dih=50, Psi=45), sw0=0.549 / 5.09)
    Pd.update(P or {})
    g = dict(wn=(20.0, 12.0, 5.0), zeta=0.8, u_lim=(0.4, 0.2, 0.3), amp_lim=(0.2, 0.85), k_v=0.3, k_p=0.3,
             tilt_max=0.25, kp_vz=0.067, ki_vz=0.10)
    g.update(gains or {})
    return dict(f=4.5, a0=a0, mix=mix, chan=CHAN, P=Pd, gains=g)


G_FLY = dict(wn=(20.0, 12.0, 5.0), k_v=0.3, k_p=0.3, u_lim=(0.4, 0.35, 0.3))
AS_T4 = 0.862                                   # hover amplitude at 4.5 Hz from the held row v_base_f4.5
AS_T4C = 0.862 * (M_C / M_OUT) ** (1.0 / 1.75)


def jobs_b7():
    """B7 / B8 rows after tuning (t01-t31, logged as probe B7tune): 10 s hold, descent, 5 s after."""
    J = []
    lawT4 = law_cfg(f=4.5, gains=G_FLY, a_scale=AS_T4)
    lawC = law_cfg(f=4.5, gains=dict(G_FLY, amp_lim=(0.167, 0.8)), a_scale=AS_T4C)
    base = dict(fn="fly", b=None, H=2.0, hold_s=10.0, after_s=5.0, T=45.0)
    beach = dict(base, start="beach", settle_s=1.5, raise_s=0.5, v_dn=0.25, out_s=0.3)
    J.append(dict(beach, probe="B7", id="beach_T4_x4", law=lawT4))
    J.append(dict(beach, probe="B7", id="beach_T4_ideal", law=lawT4, motor="ideal"))
    J.append(dict(beach, probe="B7", id="beach_T4c_x4", law=lawC, cut=CUT))
    J.append(dict(beach, probe="B8", id="still_beach_T4", law=lawT4, still=True, T=8.0))
    for mode, ent in _modes():
        fl = dict(base, start="float", z0=0.0, mode=mode, entrainment=ent, settle_s=6.0, raise_s=3.0, dwell_s=1.0)
        J.append(dict(fl, probe="B7", id=f"float_T4_{mode}", law=lawT4, pose0=dict(sw0=42)))
        J.append(dict(fl, probe="B7", id=f"float_T4c_sw58_{mode}", law=lawC, cut=CUT, pose0=dict(sw0=58, dih=50),
                      pose_blend=dict(sw0="osc", dih="height")))
        J.append(dict(fl, probe="B8", id=f"still_float_T4_{mode}", law=lawT4, pose0=dict(sw0=42), still=True, T=12.0))
        dn = dict(base, start="air", z0=2.0, H=0.0, hold_s=3.0, over_water=True, touch_z=0.05, mode=mode,
                  entrainment=ent, T=25.0)
        J.append(dict(dn, probe="B7", id=f"water_descent_T4_{mode}", law=lawT4))
        J.append(dict(dn, probe="B7", id=f"water_descent_T4c_{mode}", law=lawC, cut=CUT))
    J.append(dict(base, probe="B8", id="still_air_T4", start="air", z0=10.0, law=lawT4, still=True, T=1.5))
    J.append(dict(base, probe="B7", id="air_hover_T4_x4", start="air", z0=10.0, H=0.5, descend=False, law=lawT4, T=14.0))
    return J


def jobs_file():
    return json.loads(Path(sys.argv[3]).read_text())


if __name__ == "__main__":
    what = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    run(globals()["jobs_" + what](), n)
