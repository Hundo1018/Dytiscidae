"""B19-B23 (results/S0b_probes.md, part "B19-B23"): surfacing diagnostics and the surfacing recipes on T5b.

`surf(job)`: T5b (`s0b_t5b_machine`, not edited) from rest at depth, wings in the sink pose, fill F_NEUTRAL (+ a left-right
tank mismatch ``tank_eps``), a fill schedule, then (once afloat) wings raised, settle, and the lift-off to 2 m (Hover4,
the B17 law).  Everything the question "why does it capsize" needs is traced every 0.05 s: CoM, attitude, fill, the heights of
the pontoon axis, the hull and the dorsal float's top, the wet fraction of each part group, the fluid force and its moment
about the DRY centre of mass (so the horizontal offset of the effective centre of buoyancy from the CoM, rx = -My/Fz,
ry = Mx/Fz).  Hinge gains re-read at each medium change, InducedFlow state cleared at the start (D2).

fill schedule ``table``: [[z_com, fill], ...] interpolated on the CoM depth (z_com increasing; the fill command is the
interpolated value), rate-limited at ``fill_rate`` /s, optionally ``t_hold`` s of constant fill before the table starts.
    MUJOCO_GL=disable PYTHONPATH=. python experiments/flapper/s0b_t5c_surf.py file <procs> <jobs.json> [out.jsonl]
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
OUT = HERE / "results" / "S0b_probes5.jsonl"
G = 9.80665
import s0b_t5b_probe as Q  # noqa: E402

GAINS = dict(wn=[6.0, 6.0, 3.0], zeta=1.0, u_lim=[0.4, 0.35, 0.3], amp_lim=[0.2, 0.9], k_v=0.5, k_p=0.1,
             tilt_max=0.25, kp_vz=0.067, ki_vz=0.1)          # B17's hover gains (rows b17_*)


def _groups(mc):
    p = mc.env.solver.panels
    bid = np.asarray(p.body_id)
    pos = np.asarray(p.pos_local)
    hull = bid == mc.hull
    g = dict(pont=np.isin(bid, mc.pont), wing=np.isin(bid, [2, 3, 4, 5]),
             dorsal=hull & (pos[:, 2] > 0.1), hullsl=hull & (np.abs(pos[:, 2]) < 0.05),
             strut=hull & (pos[:, 2] < -0.05))
    g["tail"] = ~(g["pont"] | g["wing"] | hull)
    return g


def surf(job) -> dict:
    import s0b_machine as T
    import s0b_laws as LW
    import s0b_probe2 as P2
    import s0b_t5b_machine as TB
    t_wall = time.time()
    dt = job.get("dt", 1e-4)
    depth = job.get("depth", 3.0)
    cfg = Q.law_cfg_b(gains=dict(GAINS, **job.get("gains", {})))
    Pl = dict(LW.LEVEL, **cfg["P"])
    mc = Q._make("water", job.get("fill0", TB.F_NEUTRAL), job.get("motor", "x4"), dt, z=-depth, g=job.get("geom"),
                 mod="s0b_t5b_machine")
    d, s, m = mc.d, mc.env.solver, mc.m
    s.entrainment = job.get("entrainment", True)
    n = len(mc.aid)
    w_h = 2 * math.pi * cfg["f"]
    q_raised, _ = LW.targets_level(mc, 0.0, w_h, Pl, osc=0.0)
    q_sink = q_raised.copy()
    for tag in ("FL", "FR", "HL", "HR"):
        q_sink[mc.ix[(tag, "feather")]] = -0.5 * math.pi
    q_flat = np.zeros(n)                                    # wings level, chord level, at a mean sweep ``flat_sw`` (rad; + = tips toward the nose)
    for tag, L_ in (("FL", 1.0), ("FR", -1.0), ("HL", 1.0), ("HR", -1.0)):
        q_flat[mc.ix[(tag, "sweep")]] = -L_ * job.get("flat_sw", 0.0)
    POSE = dict(raised=q_raised, sink=q_sink, flat=q_flat)
    q_now = POSE[job.get("pose0", "sink")].copy()
    d.qpos[mc.qad] = q_now
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    d.qvel[:] = 0.0
    mc.mj.mj_forward(m, d)
    d.qpos[2] += -depth - float(mc.com()[2])
    mc.mj.mj_forward(m, d)
    T.regain(mc)
    s.reset()
    eps = job.get("tank_eps", 0.0)
    if eps:
        TB.set_fill(mc, mc.fill, eps=eps)
    if job.get("pitch0_deg") or job.get("roll0_deg"):
        d.qpos[3:7] = P2._quat(job.get("roll0_deg", 0.0), job.get("pitch0_deg", 0.0))
        mc.mj.mj_forward(m, d)
    grp = _groups(mc)
    p = s.panels
    table = np.array(job["table"], float)
    fill_rate = job.get("fill_rate", 0.3)
    t_hold = job.get("t_hold", 0.0)
    wing_pose_at = job.get("wing_pose_at")        # [[z_com, "raised"|"sink"], ...]: when the CoM passes z, blend to that pose
    tail_trim = job.get("tail_trim")              # constant tail deflection during the rise
    trace, ev = [], {}
    clamped_total, abort = 0, None
    phase = "rise"
    t_afloat, t_phase = None, 0.0
    t_end = job.get("T", 120.0)
    law = None
    q_from = q_now.copy()
    pose_to, t_pose0, pose_T = None, 0.0, 2.0
    t_dw = None
    vzf = 0.0
    dih_s = dict(L=0.0, R=0.0)
    hold = dict(z=[], xy=[], up=[])
    z_start = H = None
    sub = None
    tq = {r: 0.0 for r in sorted(set(mc.role))}
    x2n = x2k = 0
    k = 0
    q_prev = q_now.copy()
    pl = job.get("pitch_loop")                    # {sw0, kp, kd, lim, rate}: the wings' mean sweep (all four, tips to the nose +) is the pitch actuator while rising
    sw_cmd = float(pl["sw0"]) if pl else job.get("flat_sw", 0.0)
    while True:
        t = mc.t
        com = mc.com()
        R = d.xmat[mc.hull].reshape(3, 3)
        att = P2._att(R)
        # ---- fill
        if k % 10 == 0:
            if phase == "rise":
                cmd = job.get("fill0", TB.F_NEUTRAL) if t < t_hold else float(np.interp(com[2], table[:, 0], table[:, 1]))
                vzl = job.get("vz_limit")                  # an integrator on the climb speed: the fill is raised while the filtered climb rate is over vmax
                if vzl and t >= t_hold:
                    vzf += (float(d.qvel[2]) - vzf) * min(1.0, 10 * dt / vzl.get("tau", 0.5))
                    cmd = min(max(mc.fill + vzl["k"] * (vzf - vzl["vmax"]) * 10 * dt, float(np.interp(com[2], table[:, 0], table[:, 1]))), 0.95)
            else:
                cmd = 0.0
            fr_ = job.get("deflate_rate", fill_rate) if phase == "deflate" else fill_rate
            step = min(max(cmd - mc.fill, -fr_ * 10 * dt), fr_ * 10 * dt)
            if abs(step) > 1e-12:
                TB.set_fill(mc, mc.fill + step, eps=eps)
        # ---- phases
        q = q_prev
        qd = np.zeros(n)
        if phase in ("rise", "deflate"):
            q = q_from
            if wing_pose_at and pose_to is None and com[2] > wing_pose_at[0][0]:
                pose_to, t_pose0 = POSE[wing_pose_at[0][1]], t
                ev["wing_pose_t"] = t
            if pose_to is not None:
                r = P2._smooth((t - t_pose0) / pose_T)
                q = q_from + (pose_to - q_from) * r
            if pl and pose_to is None and com[2] <= job.get("freeze_z", 9.0) and (phase == "rise" or job.get("loop_in_deflate", True)):
                th_up = math.radians(att[1])
                w_up = -float(d.qvel[4])
                if job.get("exit_z") is not None and com[2] > job["exit_z"]:       # near the surface: the sweep goes to a fixed value, no loop
                    tgt = job["sw_exit"]
                else:
                    tgt = min(max(pl["sw0"] + pl["kp"] * th_up + pl["kd"] * w_up, pl["lim"][0]), pl["lim"][1])
                sw_cmd += min(max(tgt - sw_cmd, -pl.get("rate", 0.5) * dt), pl.get("rate", 0.5) * dt)
                q = q.copy()
                rl = job.get("roll_loop")                                      # one-sided dihedral: the raised wing moves its water inboard (+ = left side up)
                if rl:
                    u_ = rl["k"] * math.radians(att[0]) + rl["kd"] * float(d.qvel[3])
                    if rl.get("off_z") is not None and com[2] > rl["off_z"]:
                        u_ = 0.0
                    for k_, tgt_d in (("L", min(max(-u_, 0.0), rl["lim"])), ("R", min(max(u_, 0.0), rl["lim"]))):
                        dih_s[k_] += min(max(tgt_d - dih_s[k_], -rl.get("rate", 1.0) * dt), rl.get("rate", 1.0) * dt)
                for tag, L_ in (("FL", 1.0), ("FR", -1.0), ("HL", 1.0), ("HR", -1.0)):
                    q[mc.ix[(tag, "sweep")]] = -L_ * sw_cmd
                    if rl:
                        q[mc.ix[(tag, "stroke")]] = dih_s["L" if L_ > 0 else "R"]
                    if job.get("dih_exit"):                                       # the wings' tips come out first: dihedral ramp over the last metre
                        z0, z1 = job["dih_z"]
                        q[mc.ix[(tag, "stroke")]] = job["dih_exit"] * min(max((com[2] - z0) / (z1 - z0), 0.0), 1.0)
                q_from = q
            if tail_trim is not None:
                q = q.copy()
                q[mc.tail_ix] = tail_trim
            if phase == "rise" and job.get("deflate"):
                if com[2] > job.get("afloat_z", -0.22) and abs(d.qvel[2]) < job.get("dwell_vz", 0.03) and t > 2.0:
                    t_dw = t if t_dw is None else t_dw
                    if t - t_dw >= job.get("dwell_s", 2.0):
                        phase, t_phase = "deflate", t
                        ev["deflate_t"] = t
                        ev["deflate_att"] = list(att)
                else:
                    t_dw = None
            elif phase == "deflate":
                if mc.fill <= 1e-6 and abs(d.qvel[2]) < 0.05 and t - t_phase >= 1.0:
                    phase, t_phase = "pose", t
                    s._inflow.reset()
                    ev["afloat_t"] = t
                    q_from = q.copy()
                    ev["afloat_att"] = list(att)
            elif (mc.fill <= 1e-6 or job.get("afloat_any_fill")) and com[2] > job.get("afloat_z", -0.15) and abs(d.qvel[2]) < 0.05 and t > 2.0:
                phase, t_phase = "pose", t
                s._inflow.reset()                          # D2: medium change, momentum-theory state cleared
                ev["afloat_t"] = t
                q_from = q.copy()
                ev["afloat_att"] = list(att)
        elif phase == "pose":
            r = P2._smooth((t - t_phase) / 2.0)
            q = q_from + (POSE["raised"] - q_from) * r
            if t - t_phase >= 2.0:
                phase, t_phase = "settle", t
                q_from = POSE["raised"].copy()
        elif phase == "settle":
            q = q_from
            if t - t_phase >= job.get("settle_s", 4.0):
                ev["settled_att"] = list(att)
                ev["settled_z"] = float(com[2])
                if job.get("lift", True):
                    I = np.array(T.report(mc)["inertia_hull_diag"])
                    z_start = float(com[2])
                    T.regain(mc)
                    s._inflow.reset()
                    law = LW.Hover4(mc, cfg["f"], cfg["mix"], cfg["chan"], I, cfg["a0"], P=Pl, H_t=z_start,
                                    **dict(cfg.get("gains", {})))
                    phase, t_phase, sub = "fly", t, "up"
                    H = job.get("H", 2.0)
                else:
                    break
        elif phase == "fly":
            tp = t - t_phase
            osc = P2._smooth(tp * cfg["f"] / 3.0)
            if sub == "up":
                law.H_t = z_start + min(H, 0.8 * tp)
                if com[2] >= z_start + H - 0.2:
                    sub, t_hold_s = "hold", t
                    ev["at_height_s"] = tp
                if tp > 12.0:
                    ev["never_reached"] = True
                    break
            else:
                law.H_t = z_start + H
                if k % 100 == 0:
                    hold["z"].append(float(com[2]))
                    hold["xy"].append(com[:2].tolist())
                    hold["up"].append(float(R[2, 2]))
                if t - t_hold_s >= job.get("hold_s", 5.0):
                    break
            law.step()
            q, qd = law.targets(osc=osc)
            qd = qd * osc
            rb = P2._smooth(tp / 0.3)
            q = q_from + (q - q_from) * rb
            if k % 50 == 0:
                x2n += 1
                x2k += int(np.linalg.norm(d.qvel[:3]) < 6.0 and np.abs(d.qpos[mc.qad][mc.role == "feather"]).max() > math.radians(30.0))
            if R[2, 2] < 0.0:
                abort = dict(t=t, why="inverted", phase=phase)
        # ---- step
        mc.step(q, qd)
        q_prev = q
        k += 1
        af = np.abs(d.actuator_force[mc.aid])
        for r_ in tq:
            tq[r_] = max(tq[r_], float(af[mc.role == r_].max()))
        if s.diag.clamped:
            clamped_total += 1
            s.diag.clamped = False
        if k % job.get("trace_every", 500) == 0:
            pos = d.xpos[p.body_id] + np.einsum("nij,nj->ni", d.xmat.reshape(-1, 3, 3)[p.body_id], p.pos_local)
            sb = mc.env.medium.submerged_fraction(pos, p.half_height, d.time)
            F, M = TB.wrench_about_dry_com(mc)
            Fz = float(F[2]) if abs(F[2]) > 1e-6 else 1e-6
            o = d.xpos[mc.hull]
            Rh = d.xmat[mc.hull].reshape(3, 3)
            pa = float(np.mean([d.xpos[b][2] for b in mc.pont]))
            dor = float((o + Rh @ np.array([0.0, 0.0, job.get("dorsal_top", 0.20 + 0.025)]))[2])
            f_ = lambda g_: round(float(sb[grp[g_]].sum() / max(grp[g_].sum(), 1)), 3)
            trace.append([round(mc.t, 3)] + [round(float(v), 4) for v in com] + [round(float(d.qvel[2]), 3)]
                         + [round(v, 1) for v in P2._att(Rh)] + [round(mc.fill, 4), round(float(o[2]), 4), round(pa, 4), round(dor, 4)]
                         + [f_("pont"), f_("dorsal"), f_("hullsl"), f_("wing"), f_("tail"), int((sb[grp["wing"]] > 0).sum())]
                         + [round(float(F[2]), 3), round(float(M[0]), 4), round(float(M[1]), 4), round(float(-M[1] / Fz), 4), round(float(M[0] / Fz), 4),
                            round(sw_cmd, 4), phase])
        if mc.bad_qacc or not np.all(np.isfinite(d.qpos)):
            abort = dict(t=mc.t, why="qacc", phase=phase)
        if abort or mc.t >= t_end:
            break
    hz, hxy = np.array(hold["z"]), np.array(hold["xy"])
    row = dict(job, kind="surf", t_end=mc.t, abort=abort, events=ev, att_end=list(P2._att(d.xmat[mc.hull].reshape(3, 3))),
               com_end=mc.com().tolist(), fill_end=mc.fill, phase_end=phase, tq_peak=tq, clamped_steps=clamped_total,
               bad_qacc=int(mc.bad_qacc), x2_share=(x2k / x2n) if x2n else None, mass=mc.mass,
               hold_z_rel=[float(hz.min() - z_start), float(hz.max() - z_start)] if len(hz) else None,
               hold_drift_m=float(np.linalg.norm(hxy - hxy[0], axis=1).max()) if len(hz) else None,
               hold_tilt_max_deg=float(np.degrees(np.arccos(np.clip(min(hold["up"]), -1, 1)))) if len(hz) else None,
               trace_cols="t x y z vz roll pitch yaw fill hull_z pont_z dorsal_top_z w_pont w_dorsal w_hull w_wing w_tail n_wet_wing Fz Mx My cob_dx cob_dy sw_cmd phase",
               trace=trace, wall_s=round(time.time() - t_wall, 1), machine="T5B")
    return P2._flags(row)


def _dispatch(job):
    return globals()[job.get("fn", "surf")](job)


def run(jobs, n, out=OUT):
    with mp.get_context("spawn").Pool(min(n, 4)) as pool:
        for r in pool.imap_unordered(_dispatch, jobs, chunksize=1):
            with open(out, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
            print(json.dumps({k: r.get(k) for k in ("id", "abort", "att_end", "events", "t_end", "hold_z_rel", "hold_tilt_max_deg", "not_evidence", "wall_s")}, default=float)[:900], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "file":
        run(json.loads(Path(sys.argv[3]).read_text()), int(sys.argv[2]), Path(sys.argv[4]) if len(sys.argv) > 4 else OUT)
