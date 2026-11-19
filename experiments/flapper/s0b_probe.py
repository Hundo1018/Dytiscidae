"""S0b probes B1-B4 (DESIGN_S0b_bold.md).  Probes, not scored measurements.
Every rollout is one row of results/S0b_probes.jsonl.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/s0b_probe.py b1map|b2|drag|... [jobs]
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
OUT = HERE / "results" / "S0b_probes.jsonl"
G = 9.80665


def _pose_quat(pitch_deg):
    a = math.radians(pitch_deg)
    return np.array([math.cos(-a / 2), 0.0, math.sin(-a / 2), 0.0])


def held(job) -> dict:
    """The hull held (pose and velocity rewritten every step); mean fluid wrench over whole beats."""
    import s0b_machine as T
    import s0b_laws as LW
    t_wall = time.time()
    medium, pitch, f = job["medium"], job.get("pitch_deg", 0.0), job["f"]
    U = np.array(job.get("U", (0.0, 0.0, 0.0)), dtype=float)            # hull velocity, world
    P = dict(LW.ZERO, **job.get("P", {}))
    still = bool(job.get("still"))
    z = 30.0 if medium == "air" else -3.0
    mc = T.make("air" if medium == "air" else "water", b=1.0, dt=job.get("dt", 1e-4), motor=job.get("motor", "ideal"),
                pose=dict(x=-30.0, y=0.0, z=z, vx=0.0, vz=0.0))
    d = mc.d
    q0 = d.qpos[:7].copy()
    q0[3:7] = _pose_quat(pitch)
    d.qpos[:7] = q0
    d.qvel[:] = 0.0
    s = mc.env.solver
    if medium != "air":
        s.entrainment = job.get("entrainment", True)
    mc.mj.mj_forward(mc.m, d)
    T.regain(mc)                                           # gains at this pose, at rest, in this medium
    s.reset()
    w = 2 * math.pi * f
    nb = int(round(1.0 / (f * mc.dt)))
    beats, skip = job.get("beats", 6), job.get("skip", 2)
    F, M, err = [], [], []
    tq, om = np.zeros(len(mc.aid)), np.zeros(len(mc.aid))
    clamped = 0
    ncon = 0
    E0 = None
    Epos = 0.0
    t = 0.0
    for k in range(beats * nb):
        if k == skip * nb:
            E0 = mc.E_shaft.sum()
        q, qd = LW.targets(mc, w * t - math.pi / 2, w, P, still=still)
        ramp = min(1.0, t * f / 0.5)                        # half a beat
        d.qpos[:3] = q0[:3] + U * t
        d.qpos[3:7] = q0[3:7]
        d.qvel[:6] = 0.0
        d.qvel[:3] = U
        qv0 = d.qvel[mc.dof].copy()
        mc.step(q * ramp, qd * ramp)
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
            err.append(d.qpos[mc.qad] - q)
            ncon = max(ncon, int(d.ncon))
    F, M = np.array(F), np.array(M)
    d.qpos[:7] = q0
    mc.mj.mj_forward(mc.m, d)
    R = d.xmat[mc.hull].reshape(3, 3)
    Fh, Mh = F.mean(0) @ R, M.mean(0) @ R
    nbt = beats - skip
    Fb = (F @ R).reshape(nbt, nb, 3).mean(1)
    err = np.array(err)
    roles = sorted(set(mc.role))
    Tm = nbt / f
    row = dict(job, kind="held", mass=mc.mass, weight=mc.mass * G,
               F_hull=Fh.tolist(), M_hull=Mh.tolist(), F_world=F.mean(0).tolist(),
               F_hull_beat_sd=Fb.std(0).tolist(), F_hull_peak=np.abs(F @ R).max(0).tolist(),
               shaft_W=float((mc.E_shaft.sum() - E0) / Tm), shaft_pos_W=Epos / Tm,
               tq_peak={r: float(tq[mc.role == r].max()) for r in roles},
               om_peak={r: float(om[mc.role == r].max()) for r in roles},
               track_rms_deg={r: float(np.degrees(np.sqrt((err[:, mc.role == r] ** 2).mean()))) for r in roles},
               clamped_steps=clamped, ncon_max=ncon, bad_qacc=int(mc.bad_qacc),
               finite=bool(np.isfinite(F).all()), wall_s=round(time.time() - t_wall, 1))
    return row


def run(jobs, n):
    rows = []
    with mp.get_context("spawn").Pool(n) as pool:
        for r in pool.imap(_dispatch, jobs, chunksize=1):
            with open(OUT, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
            rows.append(r)
            print(json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in r.items()
                              if k in ("probe", "id", "f", "F_hull", "M_hull", "shaft_pos_W", "bad_qacc", "mode",
                                       "clamped_steps", "ncon_max", "wall_s", "events", "z_max", "v_max",
                                       "t_end", "com_end", "nose_end", "law_sat", "self_contacts_max", "artifact", "contact_peak_W", "E_shaft_J", "motor", "at_limit",
                                       "tq_peak", "track_rms_deg")}), flush=True)
    return rows


def _dispatch(job):
    return globals()[job.get("fn", "held")](job)


def jobs_b1map():
    J = [dict(probe="B1", id="base", medium="air", pitch_deg=90.0, f=4.5),
         dict(probe="B1", id="base_inphase", medium="air", pitch_deg=90.0, f=4.5, P=dict(hind_phase=0.0)),
         dict(probe="B4", id="still_air", medium="air", pitch_deg=90.0, f=4.5, still=True)]
    for f in (2.5, 3.5, 5.5, 6.0):
        J.append(dict(probe="B1", id=f"f{f}", medium="air", pitch_deg=90.0, f=f))
    for k, v in (("amp", 1.15), ("amp_lr", 0.2), ("amp_fh", 0.2), ("bias", 0.15), ("bias_lr", 0.15), ("bias_fh", 0.15),
                 ("psi0", 0.15), ("psi0_lr", 0.15), ("psi0_fh", 0.15), ("Psi_lr", 0.15), ("Psi_fh", 0.15),
                 ("sw", 0.2), ("sw_lr", 0.2), ("sw_fh", 0.2)):
        J.append(dict(probe="B1", id=f"{k}={v}", medium="air", pitch_deg=90.0, f=4.5, P={k: v}))
    return J


def jobs_b2():
    J = []
    for mode in ("default", "wetting"):
        ent = True if mode == "default" else "wetting"
        base = dict(medium="water", pitch_deg=0.0, entrainment=ent, mode=mode, beats=4, skip=1)
        J.append(dict(base, probe="B4", id="still_water", f=1.0, still=True, beats=2))
        for U in (0.3, 1.0, 2.0):
            J.append(dict(base, probe="B2", id=f"drag_U{U}", f=1.0, still=True, beats=3, U=(U, 0.0, 0.0)))
        for f in (0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0):
            J.append(dict(base, probe="B2", id=f"f{f}", f=f))
        for k, v in (("amp_lr", 0.2), ("amp_fh", 0.2), ("bias", 0.15), ("bias_fh", 0.15), ("psi0_lr", 0.15),
                     ("sw_lr", 0.2), ("sw_fh", 0.2)):
            J.append(dict(base, probe="B2", id=f"{k}={v}", f=1.0, P={k: v}))
    return J


def _wing_wet(mc):
    """Wet strip count on the four main wings, and on the whole machine."""
    s, p, d = mc.env.solver, mc.env.solver.panels, mc.d
    R = d.xmat.reshape(-1, 3, 3)[p.body_id]
    pos = d.xpos[p.body_id] + np.einsum("nij,nj->ni", R, p.pos_local)
    sub = mc.env.medium.submerged_fraction(pos, p.half_height, d.time)
    main = np.isin(p.body_id, [2, 3, 4, 5])
    return int(((sub > 0) & main).sum()), int((sub > 0).sum()), float(pos[:, 2].min())


def free(job) -> dict:
    """A free rollout.  start: "air" (nose up, at rest, z0 over the sea), "water" (nose up, CoM at 3 m depth,
    at rest).  Phases: water stroke (law "open" or "loop") until the CoM reaches the surface; then "stop"
    (every hinge held at 0) or "fly" (water stroke on until no main-wing strip is wet, then the air law)."""
    import s0b_machine as T
    import s0b_laws as LW
    t_wall = time.time()
    start = job["start"]
    still = bool(job.get("still"))
    dt = job.get("dt", 1e-4)
    if start == "air":
        mc = T.make("air", b=1.0, dt=dt, motor=job.get("motor", "ideal"), pose=dict(x=-30.0, y=0.0, z=job.get("z0", 10.0), vx=0.0, vz=0.0))
    else:
        mc = T.make("water", b=1.0, dt=dt, motor=job.get("motor", "ideal"), pose=dict(x=-30.0, y=0.0, z=-3.0, vx=0.0, vz=0.0))
    d, s = mc.d, mc.env.solver
    d.qpos[3:7] = _pose_quat(job.get("pitch_deg", 90.0))
    d.qvel[:] = 0.0
    mc.mj.mj_forward(mc.m, d)
    if start == "water":
        d.qpos[2] += -3.0 - float(mc.com()[2])
        s.entrainment = job.get("entrainment", True)
    mc.mj.mj_forward(mc.m, d)
    T.regain(mc)
    s.reset()
    I = np.array(T.report(mc)["inertia_hull_diag"])
    air = job.get("air") or {}
    wat = job.get("water") or {}

    def mk(cfg, f, H_t):
        return LW.Fly4(mc, f, cfg["mix"], I, cfg["a0"], wn=cfg.get("wn", 6.0), zeta=cfg.get("zeta", 0.7), H_t=H_t,
                       height=cfg.get("height", True), u_lim=cfg.get("u_lim", 0.3), P=cfg.get("P"))
    phase = "fly" if start == "air" else "swim"
    law = None
    if not still:
        law = mk(air, air["f"], float(mc.com()[2])) if start == "air" else (
            mk(wat, wat["f"], None) if wat.get("loop") else None)
    Pw = dict(LW.ZERO, **(wat.get("P") or {}), amp=wat.get("a0", 1.0))
    fw = wat.get("f", 1.0)
    N = int(round(job["T"] / dt))
    tr = []
    clamped = 0
    ev = {}
    zmax, vmax = -1e9, 0.0
    ph = -math.pi / 2
    n = len(mc.aid)
    self_con = 0
    tq, om = np.zeros(n), np.zeros(n)
    contact_W = 0.0
    q_prev, q_from, t_bl, last_phase = np.zeros(n), np.zeros(n), -1e9, phase
    TB = job.get("blend_s", 0.25)                  # every change of phase is blended from the last target
    if law is not None:
        t_bl = 0.0                                 # and the first law is ramped in from the rest pose
    for k in range(N):
        com = mc.com()
        if phase == "swim" and com[2] >= 0.0:
            ev["surface"] = dict(t=mc.t, vz=float(d.qvel[2]), v=d.qvel[:3].tolist(),
                                 nose=d.xmat[mc.hull].reshape(3, 3)[:, 0].tolist(), w=d.qvel[3:6].tolist())
            phase = "stop" if job.get("after", "stop") == "stop" else "breach"
        if phase == "breach" and (k % 10 == 0):
            wm, wa, low = _wing_wet(mc)
            if wm == 0:
                ev["clear"] = dict(t=mc.t, z=float(com[2]), vz=float(d.qvel[2]), low=low,
                                   nose=d.xmat[mc.hull].reshape(3, 3)[:, 0].tolist())
                T.regain(mc)
                law = mk(air, air["f"], air.get("H_t", 2.0))
                phase = "fly"
        if still or phase == "stop":
            q, qd = np.zeros(n), np.zeros(n)
        elif phase in ("swim", "breach"):
            if law is not None:
                law.step()
                q, qd = law.targets()
            else:
                ph += 2 * math.pi * fw * dt
                r = min(1.0, mc.t * fw / 0.5)
                q, qd = LW.targets(mc, ph, 2 * math.pi * fw, Pw)
                q, qd = q * r, qd * r
        else:
            law.step()
            q, qd = law.targets()
        if phase != last_phase:
            q_from, t_bl, last_phase = q_prev.copy(), mc.t, phase
        r = min(max((mc.t - t_bl) / TB, 0.0), 1.0)
        r = r * r * (3.0 - 2.0 * r)
        q, qd = q_from * (1.0 - r) + q * r, qd * r
        q_prev = q
        qv0 = d.qvel[mc.dof].copy()
        mc.step(q, qd)
        np.maximum(tq, np.abs(d.actuator_force[mc.aid]), out=tq)
        np.maximum(om, np.abs(qv0), out=om)
        if d.ncon:
            contact_W = max(contact_W, mc.contact_load())
        if s.diag.clamped:
            clamped += 1
            s.diag.clamped = False
        zmax = max(zmax, float(com[2]))
        vmax = max(vmax, float(np.linalg.norm(d.qvel[:3])))
        if k % 200 == 0:
            Rh = d.xmat[mc.hull].reshape(3, 3)
            wm, wa, low = _wing_wet(mc)
            nc = sum(1 for i in range(d.ncon) if int(mc.m.geom_bodyid[d.contact[i].geom1]) != 0
                     and int(mc.m.geom_bodyid[d.contact[i].geom2]) != 0)
            self_con = max(self_con, nc)
            tr.append([round(mc.t, 3), round(float(com[0]), 3), round(float(com[1]), 3), round(float(com[2]), 3),
                       round(float(d.qvel[2]), 3), round(float(Rh[2, 0]), 3), round(float(np.linalg.norm(d.qvel[3:6])), 2),
                       wm, round(low, 3), phase, round(float(law.out["amp"]), 3) if law is not None else 0.0])
        if not np.all(np.isfinite(d.qpos)) or mc.bad_qacc:
            break
    roles = sorted(set(mc.role))
    Rh = d.xmat[mc.hull].reshape(3, 3)
    row = dict(job, kind="free", mass=mc.mass, t_end=mc.t, events=ev, z_max=zmax, v_max=vmax,
               com_end=mc.com().tolist(), nose_end=Rh[:, 0].tolist(), vel_end=d.qvel[:3].tolist(),
               shaft_W={r: float(mc.E_shaft[mc.role == r].sum() / max(mc.t, 1e-9)) for r in roles},
               tq_peak={r: float(tq[mc.role == r].max()) for r in roles},
               om_peak={r: float(om[mc.role == r].max()) for r in roles},
               artifact=bool(any(tq[mc.role == r].max() > 10.0 * T.WITNESS_STALL[r] for r in roles)),
               contact_peak_W=contact_W, E_shaft_J=float(mc.E_shaft.sum()),
               E_shaft_pos_note="E_shaft is net shaft work (regeneration counted negative)",
               at_limit={r: float(mc.at_limit[mc.role == r].mean() / max(mc.n, 1)) for r in roles},
               law_sat=(law.sat / max(law.n_upd, 1)) if law is not None else None,
               clamped_steps=clamped, self_contacts_max=self_con, bad_qacc=int(mc.bad_qacc),
               trace_cols="t x y z vz nose_z |w| wet_main low phase amp", trace=tr,
               wall_s=round(time.time() - t_wall, 1))
    return row


#: B1's held map at K1 amp 1 (N m per rad or per unit, hull axes): My = 10.7 bias, Mz = -10.3 amp_lr,
#: Mx = -14.4 psi0_lr; thrust 20.94 N at amp 1.  At the hover amplitude the moments scale with thrust.
A0_AIR = math.sqrt(11.366 / 20.94)
K_AIR = A0_AIR ** 2
MIX_AIR = [[0.0, 1.0 / (10.7 * K_AIR), 0.0], [0.0, 0.0, -1.0 / (10.3 * K_AIR)], [-1.0 / (14.4 * K_AIR), 0.0, 0.0]]
AIR = dict(f=4.5, a0=A0_AIR, mix=MIX_AIR)


def jobs_b1free():
    return [dict(fn="free", probe="B1", id=f"free_hover_{mo}", start="air", z0=10.0, T=8.0, motor=mo, air=AIR)
            for mo in ("ideal", "x4")]


def jobs_b4free():
    return [dict(fn="free", probe="B4", id=f"still_breach_{mode}", start="water", T=6.0, still=True, mode=mode,
                 entrainment=True if mode == "default" else "wetting") for mode in ("default", "wetting")]


def jobs_b2b():
    """B2 again at a stroke the witness motor's stall could hold: 30 deg peak to peak (K1 is 85)."""
    J = []
    for mode in ("default", "wetting"):
        ent = True if mode == "default" else "wetting"
        base = dict(probe="B2b", medium="water", pitch_deg=0.0, entrainment=ent, mode=mode, beats=4, skip=1)
        P0 = dict(Phi=math.radians(30.0))
        for f in (0.5, 1.0, 2.0, 3.0, 4.5, 6.0):
            J.append(dict(base, id=f"Phi30_f{f}", f=f, P=P0))
        for k, v in (("amp_lr", 0.2), ("bias", 0.15), ("psi0_lr", 0.15), ("psi0", 0.15), ("sw_lr", 0.2)):
            J.append(dict(base, id=f"Phi30_{k}={v}", f=1.0, P=dict(P0, **{k: v})))
    return J


W30 = dict(Phi=math.radians(30.0))


def jobs_b3a():
    """The unclamped water point (30 deg, 0.5 Hz): its moment map, and the open-loop breach with the wings stopped."""
    J = []
    for mode in ("default", "wetting"):
        ent = True if mode == "default" else "wetting"
        for mo in ("ideal", "x4"):
            J.append(dict(fn="free", probe="B3", id=f"breach_stop_open_{mode}_{mo}", start="water", T=7.0, mode=mode,
                          entrainment=ent, motor=mo, after="stop", water=dict(f=0.5, a0=1.0, P=W30, loop=False)))
        base = dict(probe="B2c", medium="water", pitch_deg=0.0, entrainment=ent, mode=mode, beats=4, skip=1, f=0.5)
        for k, v in (("amp_lr", 0.2), ("bias", 0.15), ("psi0_lr", 0.15)):
            J.append(dict(base, id=f"Phi30_f0.5_{k}={v}", P=dict(W30, **{k: v})))
        J.append(dict(base, id="Phi30_f0.5_x4", P=W30, motor="x4"))
    return J


DIAG = dict(lr_phase=math.pi)


def jobs_b3b():
    """The diagonal stroke (reactions cancel on the hull): held checks, then the open-loop breach, wings stopped."""
    J = [dict(probe="B1", id="diag_air", medium="air", pitch_deg=90.0, f=4.5, P=DIAG),
         dict(probe="B2c", id="Phi30_f0.5_diag", medium="water", pitch_deg=0.0, entrainment=True, mode="default",
              beats=4, skip=1, f=0.5, P=dict(W30, **DIAG))]
    for mode in ("default", "wetting"):
        ent = True if mode == "default" else "wetting"
        for mo in ("ideal", "x4"):
            J.append(dict(fn="free", probe="B3", id=f"breach_stop_diag_{mode}_{mo}", start="water", T=7.0, mode=mode,
                          entrainment=ent, motor=mo, after="stop",
                          water=dict(f=0.5, a0=1.0, P=dict(W30, **DIAG), loop=False)))
    return J


def jobs_b3c():
    """One attempt per mode and motor: the diagonal water stroke kept on through the surface, the air law
    (B1's, untuned) from the moment no main-wing strip is wet."""
    J = []
    for mode in ("default", "wetting"):
        ent = True if mode == "default" else "wetting"
        for mo in ("ideal", "x4"):
            J.append(dict(fn="free", probe="B3", id=f"breach_fly_diag_{mode}_{mo}", start="water", T=9.0, mode=mode,
                          entrainment=ent, motor=mo, after="fly", air=dict(AIR, H_t=2.0),
                          water=dict(f=0.5, a0=1.0, P=dict(W30, **DIAG), loop=False)))
    return J


def jobs_b1rest():
    return [j for j in jobs_b1map() if j["id"].split("=")[0] in ("psi0_lr", "psi0_fh", "Psi_lr", "Psi_fh", "sw", "sw_lr", "sw_fh")]


if __name__ == "__main__":
    what = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run(globals()["jobs_" + what](), n)
