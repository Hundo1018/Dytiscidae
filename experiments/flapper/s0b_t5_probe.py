"""S0b probes B9-B13 (DESIGN_S0b_bold.md, addendum 2) on T5 (`s0b_t5_machine`).  Probes, not scored.
Every rollout is one row of results/S0b_probes3.jsonl.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/s0b_t5_probe.py <jobset> [procs]
    ... s0b_t5_probe.py file <procs> <jobs.json>

The rollouts of B5-B8 (`s0b_probe2.held`, `floatq`, `fly`) are used as they are, with their builder replaced
by T5's in each worker (``b`` in a job is then the pontoons' fill).  New here: `held_w` (the stroke-hinge
stroke of B2 held under water) and `mission` (a scripted sequence over water with the ballast).
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
OUT = HERE / "results" / "S0b_probes3.jsonl"
G = 9.80665
Z_FLOAT = 0.165                                  # hull origin over the water at the float (settles from here)


def _make(domain, b, motor, dt, z=30.0, x=-30.0, cut=0.0, g=None, contact_tc=1.0):
    """``contact_tc``: every geom's contact time constant (solref[0], 0.004 s as built) times this (B13)."""
    import s0b_t5_machine as T5
    mc = T5.make(domain, motor=motor, g=g, fill=float(b or 0.0), dt=dt, pose=dict(x=x, y=0.0, z=z, vx=0.0, vz=0.0))
    if contact_tc != 1.0:
        mc.m.geom_solref[:, 0] *= contact_tc
    return mc


def _patch(job=None):
    import s0b_probe2 as P2
    g = (job or {}).get("geom")
    tc = (job or {}).get("contact_tc", 1.0)
    P2._make = lambda domain, b, motor, dt, z=30.0, x=-30.0, cut=0.0: _make(domain, b, motor, dt, z=z, x=x, g=g, contact_tc=tc)
    if not getattr(P2, "_t5_pairs", False):
        orig = P2._pairs

        def pairs(mc):
            P = orig(mc)
            m, mj = mc.m, mc.mj
            fl = [g_ for g_ in range(m.ngeom) if int(m.geom_bodyid[g_]) in mc.pont]
            wing = [g_ for g_ in range(m.ngeom) if int(m.geom_bodyid[g_]) in (2, 3, 4, 5)]
            P["wing_float"] = [(a, b) for a in wing for b in fl]
            return P
        P2._pairs, P2._t5_pairs = pairs, True
    return P2


def _dispatch(job):
    P2 = _patch(job)
    fn = job.get("fn", "held")
    row = getattr(P2, fn)(job) if fn in ("held", "floatq", "fly") else globals()[fn](job)
    row["machine"] = "T5"
    if fn == "held":
        row["thrust_W"] = float(row["F_hull"][2] / row["weight"])
    return row


PRINT = ("probe", "id", "mode", "motor", "thrust_W", "F_hull", "M_hull", "tq_peak", "clearance_min", "x2_share",
         "clamped_steps", "bad_qacc", "not_evidence", "wall_s", "wet_main_max", "wing_low_min", "summary")


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
                if isinstance(v, (list, tuple)):
                    return [rnd(x) for x in v]
                if isinstance(v, dict):
                    return {k: rnd(x) for k, x in v.items()}
                return v
            print(json.dumps({k: rnd(r[k]) for k in PRINT if k in r}, default=float), flush=True)
    return rows


def _modes():
    return (("default", True), ("wetting", "wetting"))


def D(**kw):
    import s0b_probe2 as P2
    return P2.D(**kw)


# ------------------------------------------------------------------------------------------------ the hover law
#: T5's hover point (B9 held map, ids `m_*`): 4.5 Hz, dihedral 35, pitch 45, A 55 x amp.  Filled from the map.
HOV = dict(flip="vert", A=55, dih=35, Psi=45)
#: measured at amp 0.56 (13.786 N; ids m_base, m_amp_lr=0.2, m_sw0=10, m_psi0_lr=10): roll +7.09 N m per unit
#: amp_lr, pitch -6.21 N m/rad of sw0, yaw +5.17 N m/rad of psi0_lr, base My +0.902 N m; scaled to the weight
#: (13.105 N, thrust ~ amp^1.75).  The first B9 float rows ran with the placeholder trim sw0 = 0.109 rad.
MAP = dict(amp=0.544, thrust=13.105, roll_per_amp_lr=6.74, pitch_per_sw0=-5.90, yaw_per_psi0_lr=4.91, My0=0.857)


def law_cfg(gains=None, P=None, a_scale=1.0, mp_=None):
    mp_ = dict(MAP, **(mp_ or {}))
    mix = [[1.0 / mp_["roll_per_amp_lr"], 0.0, 0.0], [0.0, 1.0 / mp_["pitch_per_sw0"], 0.0],
           [0.0, 0.0, 1.0 / mp_["yaw_per_psi0_lr"]]]
    Pd = dict(D(**HOV), sw0=-mp_["My0"] / mp_["pitch_per_sw0"])
    Pd.update(P or {})
    g = dict(wn=(20.0, 12.0, 5.0), zeta=0.8, u_lim=(0.4, 0.35, 0.3), amp_lim=(0.2, 0.9), k_v=0.3, k_p=0.3,
             tilt_max=0.25, kp_vz=0.067, ki_vz=0.10)
    g.update(gains or {})
    return dict(f=4.5, a0=mp_["amp"] * a_scale, mix=mix, chan=("amp_lr", "sw0", "psi0_lr"), P=Pd, gains=g)


# ------------------------------------------------------------------------------------- held under water (B11)
def held_w(job) -> dict:
    """Hull held level under water (CoM depth ``depth``), pontoons at fill F_NEUTRAL: the stroke-hinge stroke
    of B2 (`s0b_laws.targets`), or wings still and the hull moved at ``U`` (drag).  Mean wrench, hull axes,
    less the static buoyancy read at the same pose."""
    import s0b_machine as T
    import s0b_t5_machine as T5
    import s0b_laws as LW
    import s0b_probe2 as P2
    t_wall = time.time()
    f = job.get("f", 0.5)
    mc = _make("water", job.get("fill", T5.F_NEUTRAL), job.get("motor", "x4"), job.get("dt", 1e-4),
               z=-job.get("depth", 3.0), g=job.get("geom"))
    d, s = mc.d, mc.env.solver
    s.entrainment = job.get("entrainment", True)
    q0 = d.qpos[:7].copy()
    q0[3:7] = (1.0, 0.0, 0.0, 0.0)
    d.qpos[:7] = q0
    d.qvel[:] = 0.0
    mc.mj.mj_forward(mc.m, d)
    T.regain(mc)
    s.reset()
    P = dict(LW.ZERO, **job.get("P", {}))
    still = bool(job.get("still"))
    U = np.array(job.get("U", (0.0, 0.0, 0.0)), float)
    w = 2 * math.pi * f
    nb = int(round(1.0 / (f * mc.dt)))
    beats, skip = job.get("beats", 4), job.get("skip", 1)
    F, M = [], []
    tq = np.zeros(len(mc.aid))
    clamped, t = 0, 0.0
    for k in range(beats * nb):
        q, qd = LW.targets(mc, w * t - math.pi / 2, w, P, still=still)
        ramp = P2._smooth(t * f / 0.5)
        d.qpos[:3] = q0[:3] + U * t
        d.qpos[3:7] = q0[3:7]
        d.qvel[:6] = 0.0
        d.qvel[:3] = U
        mc.step(q * ramp, qd * ramp)
        t += mc.dt
        if s.diag.clamped:
            clamped += 1
            s.diag.clamped = False
        if k >= skip * nb:
            Fk, Mk = T.fluid_wrench(mc)
            F.append(Fk)
            M.append(Mk)
            np.maximum(tq, np.abs(d.actuator_force[mc.aid]), out=tq)
    F, M = np.array(F).mean(0), np.array(M).mean(0)
    roles = sorted(set(mc.role))
    row = dict(job, kind="held_w", mass=mc.mass, weight=mc.mass * G, F_hull=F.tolist(), M_hull=M.tolist(),
               F_net=[float(F[0]), float(F[1]), float(F[2] - mc.mass * G)],
               tq_peak={r: float(tq[mc.role == r].max()) for r in roles}, clamped_steps=clamped,
               bad_qacc=int(mc.bad_qacc), wall_s=round(time.time() - t_wall, 1))
    return P2._flags(row)


# ------------------------------------------------------------------------------------------------- the mission
def mission(job) -> dict:
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
    import s0b_t5_machine as T5
    import s0b_laws as LW
    import s0b_probe2 as P2
    t_wall = time.time()
    dt = job.get("dt", 1e-4)
    still = bool(job.get("still"))
    start = job.get("start", "float")
    cfg = job.get("law") or law_cfg()
    Pl = dict(LW.LEVEL, **cfg.get("P", {}))
    mc = _make("water", 0.0 if start == "float" else T5.F_NEUTRAL, job.get("motor", "x4"), dt,
               z=Z_FLOAT if start == "float" else -job.get("depth", 3.0), g=job.get("geom"))
    d, s, m = mc.d, mc.env.solver, mc.m
    s.entrainment = job.get("entrainment", True)
    n = len(mc.aid)
    w_h = 2 * math.pi * cfg["f"]
    q_raised, _ = LW.targets_level(mc, 0.0, w_h, Pl, osc=0.0)
    q_flat = np.zeros(n)
    q_sink = q_raised.copy()                               # wings raised, chord vertical (leading edge up): the
    for tag in ("FL", "FR", "HL", "HR"):                   # pose of the hover stroke's flip, which clears the
        q_sink[mc.ix[(tag, "feather")]] = -0.5 * math.pi   # pontoons (wings level with the chord vertical do not)
    POSE = dict(raised=q_raised, flat=q_flat, sink=q_sink)
    q_now = POSE[job.get("pose0", "raised" if start == "float" else "sink")].copy()
    d.qpos[mc.qad] = q_now
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
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
    roles = sorted(set(mc.role))
    tr, phases_out, ev = [], [], {}
    clamped_total = 0
    t_end = job.get("T", 200.0)
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
            yaw_t = yaw0 + (math.pi if sdir < 0 else 0.0)          # the commanded heading
            hc = np.array([math.cos(yaw_t), math.sin(yaw_t), 0.0])
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
                    a_t = min(max(a_t, ph.get("a_min", 0.0)), ph.get("a_max", 0.6))
                    sw["a"] += min(1.0, 10 * dt / 0.5) * (a_t - sw["a"])
                    err = dict(yaw=(ey, wb[2]), pitch=(pitch_up, -wb[1]), roll=(roll, wb[0]),
                               depth=(float(com[2]) - z_t, float(d.qvel[2])))
                    for nm_, L_ in loops.items():
                        e_, r_ = err[nm_]
                        u_ = L_["k"] * e_ + L_.get("kd", 0.0) * r_ * (-1.0 if nm_ == "yaw" else 1.0) \
                            + L_.get("ff", 0.0) * sw["a"] ** 2
                        u_ = min(max(u_, -L_.get("lim", 0.3)), L_.get("lim", 0.3))
                        sw["out"][L_["ch"]] += min(1.0, 10 * dt / L_.get("tau", 0.3)) * (u_ - sw["out"][L_["ch"]])
                r = P2._smooth(tp * fw / 1.0) * (1.0 - P2._smooth((tp - (ph["T"] - 1.0 / fw)) * fw / 1.0))
                q, qd = LW.targets(mc, sw["ph"], 2 * math.pi * fw, dict(Pw, amp=sw["a"], Psi=Psi0 * sw["a"], **sw["out"]))
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
                wm = P2._wet(mc)[0]
                st["wet_main_max"] = max(st["wet_main_max"], wm)
                a_ = att()
                tr.append([round(mc.t, 3)] + [round(float(v), 3) for v in com] + [round(float(d.qvel[2]), 3)]
                          + [round(v, 1) for v in a_] + [round(mc.fill, 4), pi, wm,
                                                          round(mc.contact_load(), 2) if d.ncon else 0.0, clamped_total,
                                                          round(float(law.out["amp"]), 3) if law is not None else
                                                          (round(sw["a"], 3) if (kind == "swim" and not still) else 0.0)])
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
               trace_cols="t x y z vz roll pitch yaw fill phase wet_main contact_W clamped_cum amp", trace=tr,
               wall_s=round(time.time() - t_wall, 1))
    row["summary"] = [dict(k=p["kind"], t=round(p["t1"] - p["t0"], 1), z=round(p["com1"][2], 2), att=[round(v, 1) for v in p["att1"]],
                           fill=round(p["fill1"], 2), cl=p["clamped"], wet=p["wet_main_max"],
                           **({"hold": p.get("hold_z_rel"), "sub": p.get("sub_end"), "drift": p.get("hold_drift_m")} if p["kind"] == "fly" else {}),
                           **({"v_cmd": round(p["v_mean_along_cmd"], 3), "dx_cmd": round(p["disp_along_cmd"], 2), "v_h0": round(p["v_mean_along_h0"], 3),
                               "ey": round(p["yaw_err_end_deg"], 1), "zr": [round(v, 2) for v in p["z_range"]]} if "v_mean_along_cmd" in p else {}))
                      for p in phases_out] + [dict(abort=abort)]
    return P2._flags(row)


# ------------------------------------------------------------------------------------------------------ jobs
PUSH = [("roll+20", "roll", 20.0, 8.0), ("pitch_nose_down20", "pitch", 20.0, 8.0), ("pitch_nose_up20", "pitch", -20.0, 8.0)]


def jobs_b9map():
    """B9: the held hover map in air at T5's hover point (for the law), and the still machine."""
    base = dict(D(**HOV), amp=MAP["amp"])
    J = [dict(probe="B9", id="m_base", f=4.5, P=base), dict(probe="B12", id="still_air_raised", f=4.5, still=True, beats=3, skip=1, P=base),
         dict(probe="B9", id="m_base_ideal", f=4.5, motor="ideal", P=base)]
    for k, v in (("amp", MAP["amp"] * 1.2), ("amp_lr", 0.2), ("sw0", 10), ("psi0_lr", 10)):
        J.append(dict(probe="B9", id=f"m_{k}={v:.3g}", f=4.5, P=dict(base, **D(**{k: v}))))
    J.append(dict(probe="B9", id="m_amp_max", f=4.5, P=dict(base, amp=0.9)))
    return J


def jobs_b9float(geom=None, tag=""):
    """B9: the float at rest (wings raised, fill 0), settled 12 s, then pushed 20 deg in roll and pitch."""
    J = []
    for mode, ent in _modes():
        J.append(dict(fn="floatq", probe="B9", id=f"float{tag}_{mode}", b=None, mode=mode, entrainment=ent, T=12.0,
                      z0=Z_FLOAT, P=dict(D(dih=35), sw0=law_cfg()["P"]["sw0"]), push=PUSH, **({"geom": geom} if geom else {})))
    return J


def jobs_b9stroke():
    """B9: the hover stroke held over the float (hull held at the settled float pose), both modes: wet strips,
    limiter steps, the lowest wing point."""
    J = []
    P0 = law_cfg()["P"]
    for mode, ent in _modes():
        for nm, amp in (("hover", MAP["amp"]), ("max", 0.9)):
            J.append(dict(probe="B9", id=f"stroke_on_float_{nm}_{mode}", f=4.5, medium="water", z_hull=FLOAT_Z_HULL,
                          mode=mode, entrainment=ent, P=dict(P0, amp=amp)))
        J.append(dict(probe="B12", id=f"still_on_float_{mode}", f=4.5, medium="water", z_hull=FLOAT_Z_HULL, still=True,
                      beats=3, skip=1, mode=mode, entrainment=ent, P=dict(P0, amp=MAP["amp"])))
    return J


FLOAT_Z_HULL = 0.1626          # hull origin z at the settled float (B9 float rows: 0.1626 default, 0.1634 wetting)


def jobs_file():
    return json.loads(Path(sys.argv[3]).read_text())



# ---------------------------------------------------------------------------------------------- B10 jobs
def _fly_base():
    return dict(fn="fly", b=None, H=2.0, hold_s=10.0, after_s=5.0, T=45.0, start_raised=True, lower_s=0.0, raise_s=0.0)


def jobs_b10(law=None, tag="", motors=("x4", "ideal"), tune=False):
    """B10: lift-off, 2 m, 10 s hold, descent, left for 5 s; beach and float (both modes); B12's still twins.
    ``tune``: the tuning starts (beach 2 m further down the slope; the float dropped from 2 cm higher)."""
    law = law or law_cfg()
    J = []
    for mo in motors:
        J.append(dict(_fly_base(), probe="B10", id=f"beach{tag}_{mo}", start="beach", settle_s=1.5, v_dn=0.25, out_s=0.3,
                      law=law, motor=mo, **(dict(x0=18.0) if tune else {})))
        for mode, ent in _modes():
            J.append(dict(_fly_base(), probe="B10", id=f"float{tag}_{mode}_{mo}", start="float", z0=Z_FLOAT + (0.02 if tune else 0.0),
                          mode=mode, entrainment=ent, settle_s=3.0, v_dn=0.25, out_s=0.3, law=law, motor=mo))
    return J


def jobs_b12fly():
    J = [dict(_fly_base(), probe="B12", id="still_beach", start="beach", settle_s=1.5, law=law_cfg(), still=True, T=8.0)]
    for mode, ent in _modes():
        J.append(dict(_fly_base(), probe="B12", id=f"still_float_{mode}", start="float", z0=Z_FLOAT, mode=mode,
                      entrainment=ent, settle_s=3.0, law=law_cfg(), still=True, T=10.0))
    return J


# ---------------------------------------------------------------------------------------------- B11 jobs
#: the swim law's loops, from the held water map at throttle 0.3 (ids `w_map_a0.3_*`): yaw -5.7 N m per rad of
#: Psi_lr, pitch +1.2..1.6 N m per rad of the stroke bias, thrust's own nose-down moment 0.435 N m at 0.3.
SWIM = dict(v=0.3, a0=0.27, a_min=0.2, a_max=0.6, kp_v=0.5, ki_v=0.1,
            yaw=dict(ch="Psi_lr", k=-0.3, kd=-0.45, lim=0.15),
            pitch=dict(ch="bias", k=1.5, kd=1.0, lim=0.4, ff=-3.9))


def swim_jobs(tag, swim=None, depth=3.0, T=(30.0, 45.0), modes=None, **kw):
    """The same start (at rest at ``depth``, neutral, wings flat), a heading and its antipode."""
    J = []
    for mode, ent in (modes or _modes()):
        for nm, sdir, T_ in (("fwd", 1.0, T[0]), ("anti", -1.0, T[1])):
            J.append(dict(fn="mission", probe="B11", id=f"swim_{nm}{tag}_{mode}", mode=mode, entrainment=ent, start="deep",
                          depth=depth, pose0="flat",
                          phases=[dict(kind="ballast", depth=depth, T=1.0),
                                  dict(kind="swim", dir=sdir, T=T_, **dict(SWIM, **(swim or {})))], **kw))
    return J


if __name__ == "__main__":
    what = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    run(globals()["jobs_" + what](), n)
