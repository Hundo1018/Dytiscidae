"""D1-D2 and B14-B17 (results/S0b_probes.md): probes on T5 and T5b.  Rows -> results/S0b_probes4.jsonl.

`mission2` is `s0b_t5_probe.mission` copied (that file is not edited) with: the machine module as a job key
(``mod``: ``s0b_t5_machine`` or ``s0b_t5b_machine``), the start height explicit (``z_float``, no module state),
``yaw0_deg``, ``regain_at`` (phases at which the hinges' PD gains are re-read), a log of torque spikes, and an
optional extended trace (``xtrace``: buoyancy, added mass, vertical fluid force, mass).
    MUJOCO_GL=disable PYTHONPATH=. python experiments/flapper/s0b_t5b_probe.py file <procs> <jobs.json>
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
OUT = HERE / "results" / "S0b_probes4.jsonl"
G = 9.80665
Z_FLOAT = 0.165
from s0b_t5_probe import law_cfg, MAP, HOV, D, _modes, PUSH, SWIM  # noqa: E402,F401  (T5's law and its measured map; not edited)


def _make(domain, b, motor, dt, z=30.0, x=-30.0, g=None, contact_tc=1.0, mod="s0b_t5_machine"):
    """``b`` is the pontoons' fill.  ``mod``: the machine module (T5 or T5b)."""
    import importlib
    TM = importlib.import_module(mod)
    mc = TM.make(domain, motor=motor, g=g, fill=float(b or 0.0), dt=dt, pose=dict(x=x, y=0.0, z=z, vx=0.0, vz=0.0))
    if contact_tc != 1.0:
        mc.m.geom_solref[:, 0] *= contact_tc
    return mc


def mission2(job) -> dict:
    """A scripted sequence over water.  ``start``: "float" (fill 0, wings raised, on the water) or "deep"
    (fill F_NEUTRAL, at rest with the CoM at -``depth``, wings in the sink pose).  ``phases``: a list of dicts,
    run in order (every pose change is blended from the last target over ``blend`` s, smoothstep):

      settle   T                      hold the present pose
      pose     to "raised"|"sink"|"flat", T       move the wings
      fly      H, hold_s, v_up, v_dn, land (True: descend, touch, stroke ramped out, ``after`` s; False: end
               of the phase after the hold, still flying)
      ballast  to <fill> | depth <z of the CoM>, T (or until: "depth" within 0.1 m and slow / "afloat")
               the fill moves at no more than ``fill_rate`` /s; with ``depth`` it is a PD on the CoM depth
               about F_NEUTRAL (and stays on in later phases until another ballast phase)
      swim     dir +1|-1 (along the nose at the phase start, or against it), v (m/s), T, law parameters

    ``still``: the actuators hold the start pose through every phase; the ballast script still runs unless
    ``still_ballast`` is False.  Everything is recorded per phase."""
    import s0b_machine as T
    import importlib
    T5 = importlib.import_module(job.get('mod', 's0b_t5_machine'))   # T5 or T5b: same API (make, set_fill, F_NEUTRAL)
    import s0b_laws as LW
    import s0b_probe2 as P2
    t_wall = time.time()
    dt = job.get("dt", 1e-4)
    still = bool(job.get("still"))
    start = job.get("start", "float")
    cfg = job.get("law") or law_cfg()
    Pl = dict(LW.LEVEL, **cfg.get("P", {}))
    mc = _make("water", 0.0 if start == "float" else T5.F_NEUTRAL, job.get("motor", "x4"), dt,
               z=job.get("z_float", Z_FLOAT) if start == "float" else -job.get("depth", 3.0), g=job.get("geom"),
               mod=job.get("mod", "s0b_t5_machine"))
    d, s, m = mc.d, mc.env.solver, mc.m
    s.entrainment = job.get("entrainment", True)
    n = len(mc.aid)
    w_h = 2 * math.pi * cfg["f"]
    q_raised, _ = LW.targets_level(mc, 0.0, w_h, Pl, osc=0.0)
    q_flat = np.zeros(n)
    q_sink = q_raised.copy()                               # wings raised, chord vertical (leading edge up): the
    for tag in ("FL", "FR", "HL", "HR"):                   # pose of the hover stroke's flip, which clears the
        q_sink[mc.ix[(tag, "feather")]] = -0.5 * math.pi   # pontoons (wings level with the chord vertical do not)
    q_swim = np.zeros(n)                                   # the under-water stroke's mean pose: wings at dihedral ``swim_bias``
    for tag in ("FL", "FR", "HL", "HR"):                   # (rad), flat; clears the pontoons for the whole input range (B15)
        q_swim[mc.ix[(tag, "stroke")]] = job.get("swim_bias", 0.45)
    q_flip = np.zeros(n)                                   # the wings turned over on their spars (feather -180 deg; needs the
    for tag in ("FL", "FR", "HL", "HR"):                   # wider feather range, geom feather_range): trailing edge leads
        q_flip[mc.ix[(tag, "feather")]] = -math.pi
        q_flip[mc.ix[(tag, "stroke")]] = job.get("swim_bias", 0.0)
    POSE = dict(raised=q_raised, flat=q_flat, sink=q_sink, swim=q_swim, flipped=q_flip)
    q_now = POSE[job.get("pose0", "raised" if start == "float" else "sink")].copy()
    d.qpos[mc.qad] = q_now
    d.qpos[3:7] = (math.cos(math.radians(job.get("yaw0_deg", 0.0)) / 2), 0.0, 0.0, math.sin(math.radians(job.get("yaw0_deg", 0.0)) / 2))
    d.qvel[:] = 0.0
    mc.mj.mj_forward(m, d)
    if start == "deep":
        d.qpos[2] += -job.get("depth", 3.0) - float(mc.com()[2])
        mc.mj.mj_forward(m, d)
    T.regain(mc)
    s.reset()
    fill_rate = job.get("fill_rate", 0.1)
    tank_eps = job.get("tank_eps", 0.0)
    roll_trim = job.get("roll_trim")
    trim = dict(e=0.0, i=0.0, peak=0.0)
    if tank_eps and mc.fill:
        T5.set_fill(mc, mc.fill, eps=tank_eps)
    bal = dict(mode="hold", to=mc.fill, z=None, kz=job.get("kz", 0.4), kd=job.get("kzd", 1.5))
    fill_peak_rate, fill_prev = 0.0, mc.fill
    wing_geoms = set(g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in (2, 3, 4, 5))
    float_geoms = set(g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in mc.pont)
    dorsal_geoms = [g for g in range(m.ngeom) if mc.mj.mj_id2name(m, mc.mj.mjtObj.mjOBJ_GEOM, g) == "t5b_dorsal"]
    roles = sorted(set(mc.role))
    tr, phases_out, ev = [], [], {}
    clamped_total = 0
    t_end = job.get("T", 200.0)
    spike_thr, spikes, at_prev = job.get("spike_thr", 40.0), [], np.zeros(len(mc.aid))
    abort = None
    q_prev = q_now.copy()
    _ft = np.zeros(6)

    def att():
        return P2._att(d.xmat[mc.hull].reshape(3, 3))

    for pi, ph in enumerate(job["phases"]):
        kind = ph["kind"]
        t0 = mc.t
        st = dict(i=pi, kind=kind, t0=t0, clamped=0, tq={r: 0.0 for r in roles}, contact=0.0, contact_wing=0.0,
                  contact_float=0.0, contact_other=0.0, wet_main_max=0, com0=mc.com().tolist(), att0=list(att()),
                  fill0=mc.fill)
        q_from = q_prev.copy()
        if kind in job.get("regain_at", ()):
            T.regain(mc)                                   # D2: the hinges' PD gains re-read in the present medium
        if pi in job.get("reset_at", ()):
            s.reset()                                      # D2(b): the solver's carried state cleared at this phase
        for part in (job.get("reset_parts") or {}).get(str(pi), []):      # D2(b): one piece of that state at a time
            if part == "damping":
                s._damping.reset()
            elif part == "unsteady":
                s._unsteady.reset()
            elif part == "inflow":
                s._inflow.reset()
            elif part == "prev":
                s._prev_mbody, s._prev_rho, s._prev_t, s._primed = None, None, None, False
                s._prev_vn[:], s._prev_ma[:] = 0.0, 0.0
        st["kp_sweep"] = float(np.mean(mc.kp[mc.role == "sweep"]))
        blend = ph.get("blend", 1.0)
        law = None
        sub = None
        x2n = x2k = 0
        if kind == "ballast":
            if "depth" in ph:
                bal.update(mode="depth", z=-abs(ph["depth"]), rate=ph.get("fill_rate"))
            else:
                bal.update(mode="to", to=float(ph["to"]), rate=ph.get("fill_rate"))
        if kind == "fly" and not still:
            I = np.array(T.report(mc)["inertia_hull_diag"])
            z_start = float(mc.com()[2])
            law = LW.Hover4(mc, cfg["f"], cfg["mix"], cfg["chan"], I, cfg["a0"], P=Pl, H_t=z_start, **cfg.get("gains", {}))
            sub, t_hold, t_dn, t_out, H_dn = "up", None, None, None, None
            hold = dict(z=[], xy=[], up=[])
            H, hold_s = ph.get("H", 2.0), ph.get("hold_s", 10.0)
            v_up, v_dn = ph.get("v_up", 0.8), ph.get("v_dn", 0.25)
            ramp_beats, out_s, after = ph.get("ramp_beats", 3.0), ph.get("out_s", 0.4), ph.get("after", 3.0)
        if kind == "swim" and not still:
            R0 = d.xmat[mc.hull].reshape(3, 3).copy()
            yaw0 = math.atan2(R0[1, 0], R0[0, 0])
            sdir = float(ph.get("dir", 1.0))
            rev = bool(ph.get("rev"))                               # reverse thrust (the wings flipped on their spars): the machine
            yaw_t = yaw0 + (math.pi if (sdir < 0 and not rev) else 0.0) + math.radians(ph.get("turn_deg", 0.0))   # keeps its heading
            hc = np.array([math.cos(yaw_t), math.sin(yaw_t), 0.0]) * (-1.0 if rev else 1.0)   # and goes backward along it
            h0 = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
            fw = ph.get("f", 0.5)
            Pw = dict(LW.ZERO, Phi=math.radians(ph.get("Phi_deg", 20.0)), lr_phase=math.pi, **ph.get("P", {}))
            Psi0 = math.radians(ph.get("Psi_deg", 50.0))
            loops = {k_: ph[k_] for k_ in ("yaw", "pitch", "roll", "depth") if k_ in ph}
            sw = dict(a=ph.get("a0", 0.15), integ=0.0, ph=-math.pi / 2, vbuf=[], out={v_["ch"]: 0.0 for v_ in loops.values()},
                      ey=0.0)
            z_t = float(mc.com()[2])
            x_s0 = mc.com().copy()
            sw_v = []
        pose_to = POSE[ph["to"]] if kind == "pose" else None
        done = False
        k = 0
        while not done:
            t = mc.t
            tp = t - t0
            com = mc.com()
            qd = np.zeros(n)
            q = q_prev
            # ---- targets
            if still:
                q = q_now
                done = tp >= ph.get("T", ph.get("T_still", 6.0))
            elif kind == "settle":
                q = q_from
                done = tp >= ph["T"]
            elif kind == "pose":
                r = P2._smooth(tp / ph["T"])
                q = q_from + (pose_to - q_from) * r
                done = tp >= ph["T"] + ph.get("wait", 0.0)
            elif kind == "ballast":
                q = q_from
                if "pose" in ph:
                    q = q_from + (POSE[ph["pose"]] - q_from) * P2._smooth(tp / ph.get("pose_T", 3.0))
                u = ph.get("until")
                if u == "depth":
                    done = (abs(com[2] - bal["z"]) < 0.1 and abs(d.qvel[2]) < 0.03 and tp > 2.0) or tp >= ph.get("T", 60.0)
                    if done and tp < ph.get("T", 60.0):
                        st["reached_s"] = tp
                elif u == "near_surface":
                    done = com[2] > ph.get("z_near", -0.25) or tp >= ph.get("T", 60.0)
                elif u == "afloat":
                    done = (mc.fill <= bal["to"] + 1e-6 and com[2] > -0.15 and abs(d.qvel[2]) < 0.05 and tp > 2.0) \
                        or tp >= ph.get("T", 60.0)
                    if done and tp < ph.get("T", 60.0):
                        st["reached_s"] = tp
                else:
                    done = tp >= ph.get("T", 10.0)
            elif kind == "fly":
                osc = P2._smooth(tp * cfg["f"] / ramp_beats)
                if sub == "up":
                    Ht = min(H, v_up * tp)
                    if com[2] >= z_start + H - 0.2:
                        sub, t_hold = "hold", t
                        st["at_height_s"] = tp
                        if ph.get("kick"):                         # a test disturbance (tuning rows only): a step
                            d.qvel[3] += ph["kick"].get("roll_rate", 0.0)       # in roll rate (rad/s) and in
                            d.qvel[1] += ph["kick"].get("vy", 0.0)              # lateral velocity (m/s)
                elif sub == "hold":
                    Ht = H
                    if k % 100 == 0:
                        hold["z"].append(float(com[2]))
                        hold["xy"].append(com[:2].tolist())
                        hold["up"].append(float(d.xmat[mc.hull].reshape(3, 3)[2, 2]))
                    if t - t_hold >= hold_s:
                        if not ph.get("land", True):
                            done = True
                        sub, t_dn, H_dn = "down", t, float(com[2]) - z_start
                elif sub == "down":
                    Ht = H_dn - v_dn * (t - t_dn)
                    if com[2] <= z_start + ph.get("touch_dz", 0.03):
                        sub, t_out = "out", t
                        vb = np.array([b_[2] for b_ in law.buf])
                        st["touch"] = dict(t=t, vz=float(d.qvel[2]), att=list(att()),
                                           v_beat_mean=((vb[-1] - vb[0]) / max((len(vb) - 1) * 10 * dt, 1e-9)).tolist())
                if sub == "out":
                    osc = osc * (1.0 - P2._smooth((t - t_out) / out_s))
                    done = t - t_out >= out_s + after
                else:
                    law.H_t = z_start + Ht
                if sub == "up" and tp > ph.get("T_up_max", 12.0):
                    done = True
                    st["never_reached"] = True
                law.step()
                q, qd = law.targets(osc=osc)
                qd = qd * osc
                rb = P2._smooth(tp / 0.3)                           # from the last target into the law's
                q = q_from + (q - q_from) * rb
                if k % 50 == 0 and sub in ("up", "hold", "down"):
                    x2n += 1
                    x2k += int(np.linalg.norm(d.qvel[:3]) < 6.0
                               and np.abs(d.qpos[mc.qad][mc.role == "feather"]).max() > math.radians(30.0))
                if d.xmat[mc.hull].reshape(3, 3)[2, 2] < 0.0:
                    abort = dict(t=t, why="inverted", phase=pi)
            elif kind == "swim":
                sw["ph"] += 2 * math.pi * fw * dt
                if k % 10 == 0:
                    Rm = d.xmat[mc.hull].reshape(3, 3)
                    yaw = math.atan2(Rm[1, 0], Rm[0, 0])
                    ey = (yaw_t - yaw + math.pi) % (2 * math.pi) - math.pi
                    sw["ey"] = ey
                    pitch_up = math.asin(min(max(Rm[2, 0], -1.0), 1.0))
                    roll = math.atan2(Rm[2, 1], Rm[2, 2])
                    wb = d.qvel[3:6]
                    sw["vbuf"].append(float(d.qvel[:3] @ hc))
                    nbuf = int(round(1.0 / (fw * dt) / 10))
                    if len(sw["vbuf"]) > nbuf:
                        sw["vbuf"].pop(0)
                    vm = float(np.mean(sw["vbuf"]))
                    e_v = ph.get("v", 0.3) - vm
                    facing = max(math.cos(ey), 0.0) ** 2
                    if facing > 0.5:
                        sw["integ"] = min(max(sw["integ"] + e_v * 10 * dt, -3.0), 3.0)
                    a_t = (ph.get("a0", 0.15) + ph.get("kp_v", 0.5) * e_v + ph.get("ki_v", 0.1) * sw["integ"]) * facing
                    if ph.get("a_turn") and facing < 0.25:             # turning: the stroke has to be alive for the yaw inputs to act
                        a_t = ph["a_turn"]
                    a_t = min(max(a_t, ph.get("a_min", 0.0)), ph.get("a_max", 0.6))
                    sw["a"] += min(1.0, 10 * dt / 0.5) * (a_t - sw["a"])
                    pr_ = ph.get("pitch_from_depth")                  # a pitch reference from the depth error (nose up to rise)
                    pitch_ref = 0.0
                    if pr_:
                        pitch_ref = min(max(pr_["k"] * (z_t - float(com[2])) - pr_.get("kd", 0.0) * float(d.qvel[2]), -pr_["lim"]), pr_["lim"])
                    err = dict(yaw=(ey, wb[2]), pitch=(pitch_up - pitch_ref, -wb[1]), roll=(roll, wb[0]),
                               depth=(float(com[2]) - z_t, float(d.qvel[2])))
                    for nm_, L_ in loops.items():
                        e_, r_ = err[nm_]
                        u_ = L_["k"] * e_ + L_.get("kd", 0.0) * r_ * (-1.0 if nm_ == "yaw" else 1.0) \
                            + L_.get("ff", 0.0) * sw["a"] ** 2
                        u_ = min(max(u_, -L_.get("lim", 0.3)), L_.get("lim", 0.3))
                        sw["out"][L_["ch"]] += min(1.0, 10 * dt / L_.get("tau", 0.3)) * (u_ - sw["out"][L_["ch"]])
                r = P2._smooth(tp * fw / 1.0) * (1.0 - P2._smooth((tp - (ph["T"] - 1.0 / fw)) * fw / 1.0))
                Pv = dict(Pw, amp=sw["a"], Psi=Psi0 * sw["a"], **sw["out"])
                if ph.get("yaw_amp_lr_gain") and "Psi_lr" in sw["out"]:
                    Pv["amp_lr"] = ph["yaw_amp_lr_gain"] * sw["out"]["Psi_lr"]     # the left-right stroke difference turns the same way
                if "bias" in sw["out"]:
                    Pv["bias"] = Pw.get("bias", 0.0) + sw["out"]["bias"]       # the pitch input rides on the mean dihedral
                q, qd = LW.targets(mc, sw["ph"], 2 * math.pi * fw, Pv)
                q, qd = q_from + (q - q_from) * r, qd * r
                if k % 100 == 0:
                    sw_v.append([tp, float(d.qvel[:3] @ hc), float((com - x_s0) @ hc), float(com[2]), math.degrees(sw["ey"]),
                                 float((com - x_s0) @ h0)])
                done = tp >= ph["T"]
            # ---- ballast
            if k % 10 == 0 and not (still and not job.get("still_ballast", True)):
                if bal["mode"] == "depth":
                    cmd = T5.F_NEUTRAL + bal["kz"] * (float(com[2]) - bal["z"]) + bal["kd"] * float(d.qvel[2])
                    cmd = min(max(cmd, 0.0), 1.0)
                else:
                    cmd = bal["to"]
                fr = bal.get("rate") or fill_rate
                step = min(max(cmd - mc.fill, -fr * 10 * dt), fr * 10 * dt)
                de = 0.0
                if roll_trim and mc.fill > 0.05:              # roll trim by the two tanks' difference (PI on roll,
                    Rm_ = d.xmat[mc.hull].reshape(3, 3)      # left side up + -> more water left), at the fill rate
                    roll_ = math.atan2(Rm_[2, 1], Rm_[2, 2])
                    trim["i"] = min(max(trim["i"] + roll_trim.get("ki", 0.1) * roll_ * 10 * dt, -0.1), 0.1)
                    want = min(max(roll_trim.get("k", 0.3) * roll_ + roll_trim.get("kd", 0.3) * float(d.qvel[3]) + trim["i"], -0.15), 0.15)
                    de = min(max(want - trim["e"], -fill_rate * 10 * dt), fill_rate * 10 * dt)
                    trim["e"] += de
                    trim["peak"] = max(trim["peak"], abs(trim["e"]))
                if abs(step) > 1e-12 or abs(de) > 1e-12:
                    T5.set_fill(mc, mc.fill + step, eps=tank_eps + trim["e"])
            # ---- step
            mc.step(q, qd)
            q_prev = q
            k += 1
            af = np.abs(d.actuator_force[mc.aid])
            if af.max() > spike_thr and len(spikes) < 40 and (not spikes or mc.t - spikes[-1]["t"] > 0.02):
                n_ = int(af.argmax())
                spikes.append(dict(t=round(mc.t, 4), phase=pi, joint=int(n_), role=str(mc.role[n_]), tau=float(d.actuator_force[mc.aid][n_]),
                                   q=float(d.qpos[mc.qad][n_]), qd=float(d.qvel[mc.dof][n_]), q_target=float(q[n_]),
                                   at_limit=bool(mc.at_limit[n_] > at_prev[n_]), sub=sub if kind == "fly" else None))
            at_prev = mc.at_limit.copy()
            for r_ in roles:
                st["tq"][r_] = max(st["tq"][r_], float(af[mc.role == r_].max()))
            if s.diag.clamped:
                st["clamped"] += 1
                clamped_total += 1
                s.diag.clamped = False
            if d.ncon:
                cl = mc.contact_load()
                if cl > 0:
                    st["contact"] = max(st["contact"], cl)
                    gs = set()
                    for i in range(d.ncon):
                        gs.add(int(d.contact[i].geom1))
                        gs.add(int(d.contact[i].geom2))
                    if gs & wing_geoms:
                        st["contact_wing"] = max(st["contact_wing"], cl)
                    if gs & float_geoms:
                        st["contact_float"] = max(st["contact_float"], cl)
                    if any(int(m.geom_bodyid[g]) != 0 and g not in wing_geoms and g not in float_geoms for g in gs):
                        st["contact_other"] = max(st["contact_other"], cl)
            if k % 200 == 0:
                st["wing_float_min"] = min(st.get("wing_float_min", 9.0),
                                           min(mc.mj.mj_geomDistance(m, d, ga, gb, 1.0, _ft) for ga in wing_geoms for gb in float_geoms))
                if dorsal_geoms:
                    st["wing_dorsal_min"] = min(st.get("wing_dorsal_min", 9.0),
                                                min(mc.mj.mj_geomDistance(m, d, ga, gb, 1.0, _ft) for ga in wing_geoms for gb in dorsal_geoms))
                wm = P2._wet(mc)[0]
                st["wet_main_max"] = max(st["wet_main_max"], wm)
                a_ = att()
                tr.append([round(mc.t, 3)] + [round(float(v), 3) for v in com] + [round(float(d.qvel[2]), 3)]
                          + [round(v, 1) for v in a_] + [round(mc.fill, 4), pi, wm,
                                                          round(mc.contact_load(), 2) if d.ncon else 0.0, clamped_total,
                                                          round(float(law.out["amp"]), 3) if law is not None else
                                                          (round(sw["a"], 3) if (kind == "swim" and not still) else 0.0)]
                          + ([round(float(s.diag.buoyancy), 3), round(float(s.diag.added_mass), 3), round(float(T.fluid_wrench(mc)[0][2]), 3),
                              round(float(mc.mass), 4), round(float(m.dof_damping.sum()), 3), round(float(m.dof_damping[:6].max()), 3)]
                             if job.get("xtrace") else []))
            if mc.bad_qacc or not np.all(np.isfinite(d.qpos)):
                abort = dict(t=mc.t, why="qacc", phase=pi)
            if abort or mc.t >= t_end:
                done = True
        st.update(t1=mc.t, com1=mc.com().tolist(), att1=list(att()), fill1=mc.fill, vel1=d.qvel[:3].tolist(),
                  wet1=P2._wet(mc)[1])
        if kind == "fly" and not still:
            hz, hxy = np.array(hold["z"]), np.array(hold["xy"])
            st.update(sub_end=sub, hold_s_done=(mc.t - t_hold if sub == "hold" else hold_s) if t_hold is not None else 0.0,
                      hold_z_rel=[float(hz.min() - z_start), float(hz.max() - z_start)] if len(hz) else None,
                      hold_drift_m=float(np.linalg.norm(hxy - hxy[0], axis=1).max()) if len(hz) else None,
                      hold_tilt_max_deg=float(np.degrees(np.arccos(np.clip(min(hold["up"]), -1, 1)))) if len(hz) else None,
                      x2_share=(x2k / x2n) if x2n else 0.0, law_sat=law.sat / max(law.n_upd, 1))
        if kind == "swim" and not still:
            a = np.array(sw_v)
            nbw = 1.0 / fw
            last = a[a[:, 0] >= ph["T"] - math.floor(min(ph["T"] * 0.5, 8.0) / nbw) * nbw - 1e-9]
            st.update(dir=sdir, disp_along_cmd=float(a[-1, 2]), disp_along_h0=float(a[-1, 5]),
                      v_mean_along_cmd=float((last[-1, 2] - last[0, 2]) / max(last[-1, 0] - last[0, 0], 1e-9)),
                      v_mean_along_h0=float((last[-1, 5] - last[0, 5]) / max(last[-1, 0] - last[0, 0], 1e-9)),
                      v_window_s=float(last[-1, 0] - last[0, 0]), z_range=[float(a[:, 3].min()), float(a[:, 3].max())],
                      yaw_err_end_deg=float(a[-1, 4]), a_end=sw["a"], out_end=dict(sw["out"]),
                      swim_trace_cols="t v_cmd x_cmd z yaw_err_deg x_h0", swim_trace=a[::5].round(3).tolist())
        rate = abs(mc.fill - st["fill0"]) / max(mc.t - t0, 1e-9)
        fill_peak_rate = max(fill_peak_rate, rate)
        phases_out.append(st)
        if abort or mc.t >= t_end:
            break
    tq = {r: max(p["tq"][r] for p in phases_out) for r in roles}
    fly_ph = [p for p in phases_out if p["kind"] == "fly" and "x2_share" in p]
    R = d.xmat[mc.hull].reshape(3, 3)
    row = dict(job, kind="mission", mass_end=mc.mass, mass_dry=float(mc.t5["m_dry"]), t_end=mc.t, abort=abort, events=ev,
               phases_out=phases_out, com_end=mc.com().tolist(), att_end=list(P2._att(R)), fill_end=mc.fill,
               fill_rate=fill_rate, roll_trim_peak=trim["peak"], tq_peak=tq, clamped_steps=clamped_total, bad_qacc=int(mc.bad_qacc),
               clamped_by_phase=[p["clamped"] for p in phases_out],
               x2_share=float(np.mean([p["x2_share"] for p in fly_ph])) if fly_ph else None,
               contact_peak_W=max(p["contact"] for p in phases_out),
               contact_peak_W_wing=max(p["contact_wing"] for p in phases_out),
               contact_peak_W_float=max(p["contact_float"] for p in phases_out),
               at_limit={r: float(mc.at_limit[mc.role == r].mean() / max(mc.n, 1)) for r in roles},
               trace_cols="t x y z vz roll pitch yaw fill phase wet_main contact_W clamped_cum amp" + (" buoy_N added_mass_kg Fz_fluid_N mass_kg dof_damp_sum dof_damp_hull_max" if job.get("xtrace") else ""),
               trace=tr, spikes=spikes,
               wall_s=round(time.time() - t_wall, 1))
    row["summary"] = [dict(k=p["kind"], t=round(p["t1"] - p["t0"], 1), z=round(p["com1"][2], 2), att=[round(v, 1) for v in p["att1"]],
                           fill=round(p["fill1"], 2), cl=p["clamped"], wet=p["wet_main_max"],
                           **({"hold": p.get("hold_z_rel"), "sub": p.get("sub_end"), "drift": p.get("hold_drift_m")} if p["kind"] == "fly" else {}),
                           **({"v_cmd": round(p["v_mean_along_cmd"], 3), "dx_cmd": round(p["disp_along_cmd"], 2), "v_h0": round(p["v_mean_along_h0"], 3),
                               "ey": round(p["yaw_err_end_deg"], 1), "zr": [round(v, 2) for v in p["z_range"]]} if "v_mean_along_cmd" in p else {}))
                      for p in phases_out] + [dict(abort=abort)]
    return P2._flags(row)


# --------------------------------------------------------------------------------------------- clearance classes
def clear_pairs(mc):
    """Geom pairs of each wing against every other part: wing_pont, wing_strut, wing_dorsal, wing_hull, wing_tail,
    wing_wing (the machine's own geoms do not collide, so this is the only read)."""
    m, mj = mc.m, mc.mj
    bname = lambda g: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
    gname = lambda g: mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
    wing = [g for g in range(m.ngeom) if bname(g) in ("seg1_wing", "seg2_wing", "seg3_wing", "seg4_wing")]
    P = dict(
        wing_pont=[g for g in range(m.ngeom) if bname(g) in ("t5_pont_L", "t5_pont_R")],
        wing_strut=[g for g in range(m.ngeom) if gname(g).startswith("t5_strut")],
        wing_dorsal=[g for g in range(m.ngeom) if gname(g) == "t5b_dorsal"],
        wing_hull=[g for g in range(m.ngeom) if bname(g) == "seg0_hull" and not gname(g).startswith(("t5_strut", "t5b_dorsal"))],
        wing_tail=[g for g in range(m.ngeom) if bname(g) in ("seg5_strut", "seg6_wing", "seg7_wing")])
    out = {k: [(a, b) for a in wing for b in v] for k, v in P.items() if v}
    out["wing_wing"] = [(a, b) for i, a in enumerate(wing) for b in wing[i + 1:]]
    return out


def clear_read(mc, pairs, cur=None):
    ft = np.zeros(6)
    out = dict(cur or {})
    for k, pp in pairs.items():
        dmin = min(mc.mj.mj_geomDistance(mc.m, mc.d, a, b, 1.0, ft) for a, b in pp)
        out[k] = min(out.get(k, 9.0), float(dmin))
    return out


# ---------------------------------------------------------------------------------- held under water, level family
def held_w2(job) -> dict:
    """B15.  Hull held level under water (CoM at ``depth``), fill F_NEUTRAL: the level-hover stroke family
    (`s0b_laws.targets_level`, parameters ``P`` over LEVEL) at ``f`` Hz, ``beats`` beats after ``skip``; the hull
    is moved at ``U`` (drag rows).  Mean wrench (hull axes) less the static buoyancy at the same pose is the
    net of the stroke alone; peak torques; the smallest clearance of every wing against every other part."""
    import importlib
    import s0b_machine as T
    import s0b_laws as LW
    import s0b_probe2 as P2
    t_wall = time.time()
    TM = importlib.import_module(job.get("mod", "s0b_t5_machine"))
    f = job.get("f", 0.5)
    mc = _make("water", job.get("fill", TM.F_NEUTRAL), job.get("motor", "x4"), job.get("dt", 1e-4), z=-job.get("depth", 3.0),
               g=job.get("geom"), mod=job.get("mod", "s0b_t5_machine"))
    d, s = mc.d, mc.env.solver
    s.entrainment = job.get("entrainment", True)
    q0 = d.qpos[:7].copy()
    q0[3:7] = (1.0, 0.0, 0.0, 0.0)
    d.qpos[:7] = q0
    d.qvel[:] = 0.0
    fam = job.get("family", "level")                  # "level": targets_level; "b2": the stroke-hinge stroke (targets)
    P = dict(LW.LEVEL if fam == "level" else LW.ZERO, **job.get("P", {}))
    w = 2 * math.pi * f
    qr = LW.targets_level(mc, 0.0, w, P, osc=0.0)[0] if fam == "level" else np.zeros(len(mc.aid))
    d.qpos[mc.qad] = qr
    mc.mj.mj_forward(mc.m, d)
    T.regain(mc)
    s.reset()
    still = bool(job.get("still"))
    U = np.array(job.get("U", (0.0, 0.0, 0.0)), float)
    nb = int(round(1.0 / (f * mc.dt)))
    beats, skip = job.get("beats", 4), job.get("skip", 1)
    pairs = clear_pairs(mc)
    clr = clear_read(mc, pairs)
    F, M = [], []
    tq = np.zeros(len(mc.aid))
    clamped, t = 0, 0.0
    for k in range(beats * nb):
        osc = 0.0 if still else P2._smooth(t * f / job.get("ramp_beats", 1.0))
        if fam == "level":
            q, qd = LW.targets_level(mc, w * t, w, P, osc=osc)
        else:
            q, qd = LW.targets(mc, w * t - math.pi / 2, w, P, still=still)
            q, qd = q * osc, qd * osc
        d.qpos[:3] = q0[:3] + U * t
        d.qpos[3:7] = q0[3:7]
        d.qvel[:6] = 0.0
        d.qvel[:3] = U
        mc.step(q, qd * (0.0 if still else 1.0))
        t += mc.dt
        if s.diag.clamped:
            clamped += 1
            s.diag.clamped = False
        if k >= skip * nb:
            Fk, Mk = T.fluid_wrench(mc)
            F.append(Fk)
            M.append(Mk)
            np.maximum(tq, np.abs(d.actuator_force[mc.aid]), out=tq)
        if k % 25 == 0:
            clr = clear_read(mc, pairs, clr)
    F, M = np.array(F), np.array(M)
    R = d.xmat[mc.hull].reshape(3, 3)
    Fh, Mh = F.mean(0) @ R, M.mean(0) @ R
    roles = sorted(set(mc.role))
    row = dict(job, kind="held_w2", mass=mc.mass, weight=mc.mass * G, F_hull=Fh.tolist(), M_hull=Mh.tolist(),
               F_net=[float(Fh[0]), float(Fh[1]), float(Fh[2] - mc.mass * G)],
               F_sd=(F @ R).reshape(beats - skip, nb, 3).mean(1).std(0).tolist(),
               tq_peak={r: float(tq[mc.role == r].max()) for r in roles}, clamped_steps=clamped, clearance_min=clr,
               bad_qacc=int(mc.bad_qacc), wall_s=round(time.time() - t_wall, 1))
    return P2._flags(row)


def _dispatch(job):
    mod = job.get("mod", "s0b_t5_machine")
    fn = job.get("fn", "mission2")
    if fn in ("floatq", "fly", "held"):                    # B5-B8's rollouts, with T5's / T5b's builder
        import s0b_probe2 as P2
        g, tc = job.get("geom"), job.get("contact_tc", 1.0)
        P2._make = lambda domain, b, motor, dt, z=30.0, x=-30.0, cut=0.0: _make(domain, b, motor, dt, z=z, x=x, g=g, contact_tc=tc, mod=mod)
        orig = P2._pairs

        def pairs(mc):
            P = orig(mc)
            m = mc.m
            P["wing_float"] = [(a_, b_) for a_ in range(m.ngeom) if int(m.geom_bodyid[a_]) in (2, 3, 4, 5)
                               for b_ in range(m.ngeom) if int(m.geom_bodyid[b_]) in mc.pont]
            return P
        P2._pairs = pairs
        row = getattr(P2, fn)(job)
    else:
        row = globals()[fn](job)
    row["machine"] = mod.replace("s0b_", "").replace("_machine", "").upper()
    return row


def run(jobs, n, out=OUT):
    with mp.get_context("spawn").Pool(min(n, 4)) as pool:
        for r in pool.imap_unordered(_dispatch, jobs, chunksize=1):
            with open(out, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
            print(json.dumps({k: r.get(k) for k in ("id", "abort", "summary", "not_evidence", "wall_s")}, default=float)[:1500], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "file":
        run(json.loads(Path(sys.argv[3]).read_text()), int(sys.argv[2]))


# ------------------------------------------------------------------------------------ T5b's hover law (B14 map)
#: held map of T5b in air at the hover point (ids `b14_m_*`, amp 0.70: 20.03 N, My +1.257; roll +2.179 N m for amp_lr 0.2;
#: sw0 10 deg: -1.657 N m; psi0_lr 10 deg: +1.385 N m), scaled to the weight (20.671 N, thrust ~ amp^1.75).
MAP_B = dict(amp=0.713, thrust=20.742, roll_per_amp_lr=11.2, pitch_per_sw0=-9.67, yaw_per_psi0_lr=7.80, My0=1.28)
GEOM_BDEF = None          # T5b's own default geometry (s0b_t5b_machine.GEOM_B)
FLOAT_Z_HULL_B = 0.2185   # hull origin z at the settled float of T5b (zp -0.23; b14z_float_*)


def law_cfg_b(gains=None, P=None, a_scale=1.0):
    return law_cfg(gains=gains, P=P, a_scale=a_scale, mp_=MAP_B)
