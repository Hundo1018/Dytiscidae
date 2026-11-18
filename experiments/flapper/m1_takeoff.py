"""Pass M1: a standing take-off from the beach (PREREG_M1_takeoff.md, frozen e70721f).

    # tuning (rows appended to results/M1_tuning.jsonl)
    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m1_takeoff.py tune ARM X YAW [JSON-gains]
    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m1_takeoff.py auth
    # scored (hash row first, then every row of section 2; at most 4 processes)
    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m1_takeoff.py score

The arms (section 2), the flying law FLY (section 4) and the reads and gates
(sections 1 and 3).  The controller reads the true state.  Nothing here adds a
force: every load on the machine comes from the harness (fluid strips, MuJoCo
contact, the motors' curves).
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m1_lib as L  # noqa: E402

TUNE_OUT = HERE / "results" / "M1_tuning.jsonl"
SCORE_OUT = HERE / "results" / "M1_takeoff.jsonl"

T_C, T_RUN = 1.0, 9.0           # command at 1 s, run to t_c + 9 s
F_BEAT = 4.5                    # every beat mean is over 1/4.5 s
Q_STRIKE = -1.35                # rad, the procedures' stroke target
THREE_CELLS_W = 462.0           # L0: three 21700 cells, continuous (reported, not a gate)

#: the flying law's gains (section 4).  The frozen file carries the tuned values.
GAINS = dict(
    f=4.5,                      # Hz, base frequency
    k_v=0.15,                   # rad of lean per m/s of horizontal velocity
    lean_max=0.3,
    # attitude: direct PD from the error rotation (body frame) and the beat-mean body rates.
    # Signs measured on the free body (tuning rows "auth"): bias + turns about +y; the
    # left-right feather offset + turns about +z (and spins about -x); the left-right
    # amplitude difference + spins about +x, weakly.
    kpy=0.4, kdy=0.10,          # common stroke bias, rad per rad / per rad/s
    kpz=0.4, kdz=0.15,          # left-right feather offset
    kdx=0.10,                   # left-right amplitude difference, spin damping
    kpz_a=0.0, kdz_a=0.0,
    kdx_f=0.0,                  # left-right feather offset on the spin rate
    # ADDED after data (not in the frozen structure; RESULT section 0b): the left-right
    # feather amplitude difference on the back axis.  0 in the pre-registered arms.
    kpP=0.0, kdP=0.0,       # left-right amplitude difference on the back axis (left larger: -z)
    H_t=4.0, v_c=1.5,           # m, m/s
    k_h=1.0,                    # 1/s, height error to climb rate wanted
    a0=0.70, kp_vz=0.10, ki_vz=0.10,   # amplitude: hover guess, PI on climb rate
    tilt_ff=True,
    switch_q=-1.2,              # rad, vault: FLY starts when the stroke passes this
    switch_s=0.25,              # s after t_c, at the latest
    blend_s=0.1,
    p_switch_deg=35.0,          # P: FLY starts at this beat-mean nose elevation
    p_switch_s=3.0,
    lp_beats=0.5,               # the outputs' low-pass time constant, in beats
    tail_kd=0.0,                # tail: pitch damping (rad of tail per rad/s); 0 = not used
)

#: ADDED after data: the arms "V+" and "P+" are V and P with these gains on top (the
#: feather-amplitude difference on the back axis).  Not pre-registered; RESULT section 0b.
PLUS = dict(kpP=1.5, kdP=0.2)

SCORING = [(x, yaw) for x in (15.0, 20.0, 25.0, 30.0) for yaw in (0.0, 180.0)]
TUNING = [(17.0, 0.0), (18.0, 180.0), (27.0, 0.0), (27.0, 180.0), (22.0, 90.0)]
STRESS = [(25.0, 90.0), (25.0, 270.0)]


def clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ======================================================================= law
class Fly:
    """K1 modulated by a loop on beat means of the true state (section 4)."""

    def __init__(self, mc, g, loop=True):
        self.mc, self.g, self.loop = mc, g, loop
        self.nb = int(round(1.0 / (F_BEAT * mc.dt) / 10))       # samples per beat (every 10 steps)
        self.buf = []
        self.out = dict(bias=0.0, amp=1.0 if not loop else g["a0"], amp_lr=0.0, psi0_lr=0.0, tail=0.0, Psi_lr=0.0)
        self.left = np.array([("seg1" in mc.mj.mj_id2name(mc.m, mc.mj.mjtObj.mjOBJ_ACTUATOR, int(a))) for a in mc.aid])
        self.integ = 0.0
        self.t0 = None
        self.k = 0

    def sample(self, H):
        """Every step: every 10th step store the state; update the outputs."""
        self.k += 1
        if self.k % 10:
            return
        mc = self.mc
        nose, left, back = L.axes(mc)
        self.buf.append((nose, left, back, mc.d.qvel[3:6].copy(), mc.com().copy(), H))
        if len(self.buf) > self.nb + 1:
            self.buf.pop(0)
        if not self.loop or len(self.buf) < 3:
            return
        g = self.g
        B = self.buf
        n = np.mean([b[0] for b in B], 0); n /= np.linalg.norm(n)
        l_ = np.mean([b[1] for b in B], 0)
        bk = np.mean([b[2] for b in B], 0)
        R = np.stack([n, l_ / np.linalg.norm(l_), bk / np.linalg.norm(bk)], 1)
        w = np.mean([b[3] for b in B], 0)                         # body-frame rates, beat mean
        T = (len(B) - 1) * 10 * mc.dt
        v = (B[-1][4] - B[0][4]) / T                              # centre-of-mass velocity, beat mean
        Hm = float(np.mean([b[5] for b in B]))
        lean = -g["k_v"] * v[:2]
        ln = np.linalg.norm(lean)
        if ln > g["lean_max"]:
            lean *= g["lean_max"] / ln
        nt = np.array([lean[0], lean[1], 1.0]); nt /= np.linalg.norm(nt)
        e = R.T @ np.cross(n, nt)
        ang = math.asin(clip(np.linalg.norm(np.cross(n, nt)), 0.0, 1.0))
        if n @ nt < 0:
            ang = math.pi - ang
        if np.linalg.norm(e) > 1e-9:                               # error as a rotation vector
            e = e / np.linalg.norm(e) * ang
        tgt = dict(bias=clip(g["kpy"] * e[1] - g["kdy"] * w[1], -0.3, 0.3),
                   psi0_lr=clip(g["kpz"] * e[2] - g["kdz"] * w[2] - g["kdx_f"] * w[0], -0.3, 0.3),
                   amp_lr=clip(-g["kdx"] * w[0] - (g["kpz_a"] * e[2] - g["kdz_a"] * w[2]), -0.3, 0.3))
        vzt = clip(g["k_h"] * (g["H_t"] - Hm), -g["v_c"], g["v_c"])
        err = vzt - v[2]
        dt_s = 10 * mc.dt
        self.integ = clip(self.integ + err * dt_s, -3.0, 3.0)
        cos_t = max(float(n[2]), 0.5)
        a = g["a0"] / math.sqrt(cos_t) if g["tilt_ff"] else g["a0"]
        tgt["Psi_lr"] = clip(-g["kpP"] * e[2] + g["kdP"] * w[2], -0.3, 0.3)
        tgt["amp"] = clip(a + g["kp_vz"] * err + g["ki_vz"] * self.integ, 0.60, 1.15)
        tgt["tail"] = clip(g["tail_kd"] * w[1], -0.5, 0.5)
        al = min(1.0, dt_s / (g["lp_beats"] / g["f"]))                                 # low-pass, half a beat
        for k_, val in tgt.items():
            self.out[k_] += al * (val - self.out[k_])

    def targets(self, t):
        g, o = self.g, self.out
        q, qd = L.k1_targets(self.mc, t - self.t0, amp=o["amp"], bias=o["bias"], f=g["f"],
                             amp_lr=o["amp_lr"], psi0_lr=o["psi0_lr"], phase0=-math.pi / 2)
        q[self.mc.role == "tail"] = o["tail"]
        if o["Psi_lr"]:                     # ADDED after data (M1 result section 0b): feather amplitude, left-right
            w = 2 * math.pi * g["f"]
            ph = w * (t - self.t0) - math.pi / 2
            fe = self.mc.role == "feather"
            s_ = np.where(self.left, 0.5, -0.5) * o["Psi_lr"]
            q[fe] += (-s_ * math.cos(ph))[fe]
            qd[fe] += (s_ * w * math.sin(ph))[fe]
        return q, qd


# ====================================================================== run
def make(entry, J_r, gear, contact=None):
    x, yaw = entry
    mc = L.machine(x=x, yaw_deg=yaw, J_r=J_r, gear=gear)
    if contact:
        m = mc.m
        if "solref_x" in contact:
            m.geom_solref[:, 0] *= contact["solref_x"]
        if "friction" in contact:
            m.geom_friction[:, 0] = contact["friction"]
    return mc


def run(arm, entry, J_r=2e-5, gear=24.0, gains=None, contact=None, trace=False):
    label = arm
    plus = arm.endswith("+")
    arm = arm.rstrip("+")
    g = dict(GAINS, **(PLUS if plus else {}))
    g.update(gains or {})
    mc = make(entry, J_r, gear, contact)
    m, d, mj = mc.m, mc.d, mc.mj
    nA = len(mc.aid)
    st, fe = mc.role == "stroke", mc.role == "feather"
    wing_bodies = [int(m.jnt_bodyid[j]) for j, r in zip(mc.jid, mc.role) if r == "stroke"]
    stroke_j = [int(j) for j, r in zip(mc.jid, mc.role) if r == "stroke"]
    stroke_a = [int(a) for a, r in zip(mc.aid, mc.role) if r == "stroke"]
    n_steps = int(round((T_C + T_RUN) / mc.dt))
    kc = int(round(T_C / mc.dt))
    W = mc.mass * L.G
    loop = arm in ("V", "Vs", "P")
    fly = Fly(mc, g, loop=loop) if arm in ("V", "Vs", "P", "VO") else None
    phase = "hold"
    t_fly = None
    q_proc = np.zeros(nA)
    keys = ("H", "hterr", "load", "nose", "span", "wx", "wy", "wz", "P", "cx", "cy", "cz")
    rec = {k: np.zeros(n_steps) for k in keys}
    lim_steps = np.zeros(n_steps, dtype=bool)
    E_prev = 0.0
    lim_prev = np.zeros(nA)
    peak_wing_N, peak_contact_M, peak_fluid_M, peak_hinge = 0.0, 0.0, 0.0, 0.0
    strike_end = None
    c0z = None
    c4 = np.zeros(6)
    t_wall = time.time()
    for k in range(n_steps):
        t = k * mc.dt
        c = mc.com()
        if k == kc:
            c0z = float(c[2])
        H = float(c[2] - c0z) if c0z is not None else 0.0
        nose, left, back = L.axes(mc)
        # ---------------------------------------------------------- command
        q, qd = np.zeros(nA), np.zeros(nA)
        if k >= kc and arm != "N0":
            tc = t - T_C
            if arm in ("V", "VO", "N2", "Vs", "P"):
                ramp = {"V": 0.0, "VO": 0.0, "N2": 0.0, "Vs": 0.2, "P": 0.5}[arm]
                if phase in ("hold", "proc"):
                    phase = "proc"
                    s = min(tc / ramp, 1.0) if ramp > 0 else 1.0
                    q_proc = np.where(st, Q_STRIKE * s, 0.0)
                    q = q_proc.copy()
                    qs = float(d.qpos[mc.qad[st]].mean())
                    if arm == "P":
                        nb_ = int(round(1 / (F_BEAT * mc.dt)))
                        lo = max(kc, k - nb_)
                        nose_m = float(np.mean(rec["nose"][lo:k])) if k > lo else 0.0
                        go = (tc >= ramp and nose_m >= g["p_switch_deg"]) or tc >= g["p_switch_s"]
                    else:
                        go = qs <= g["switch_q"] or tc >= g["switch_s"]
                    if go:
                        strike_end = t
                        if arm == "N2":
                            phase = "held"
                        else:
                            phase = "fly"
                            t_fly = t
                            fly.t0 = t
                elif phase == "held":
                    q = np.where(st, Q_STRIKE, 0.0)
                if phase == "fly":
                    fly.sample(H)
                    qf, qdf = fly.targets(t)
                    b = min((t - t_fly) / g["blend_s"], 1.0)
                    q = (1 - b) * q_proc + b * qf
                    qd = b * qdf
            elif arm == "N1":
                q, qd = L.k1_targets(mc, tc)
            elif arm == "R0":
                q, qd = L.k1_targets(mc, tc, bias=math.radians(42.5), Phi=math.radians(75.0),
                                     Psi=math.radians(15.0), phase0=-math.pi / 2)
                b = min(tc / 0.1, 1.0)
                q, qd = b * q, b * qd
        mc.step(q, qd)
        # ------------------------------------------------------------ reads
        E = float(mc.E_elec.sum())
        rec["P"][k] = (E - E_prev) / mc.dt
        E_prev = E
        lim_steps[k] = bool(np.any((mc.at_limit - lim_prev)[st] > 0))
        lim_prev = mc.at_limit.copy()
        rec["H"][k] = H
        rec["cx"][k], rec["cy"][k], rec["cz"][k] = c
        rec["load"][k] = mc.contact_load()
        rec["nose"][k] = math.degrees(math.asin(clip(nose[2], -1, 1)))
        rec["span"][k] = math.degrees(math.asin(clip(left[2], -1, 1)))
        rec["wx"][k], rec["wy"][k], rec["wz"][k] = d.qvel[3:6]
        if k >= kc:
            peak_hinge = max(peak_hinge, float(np.abs(d.actuator_force[stroke_a]).max()))
            per_wing = {b: 0.0 for b in wing_bodies}
            for i in range(d.ncon):
                con = d.contact[i]
                b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
                wb = b2 if b2 in per_wing else b1 if b1 in per_wing else None
                if wb is None or (b1 != 0 and b2 != 0):
                    continue
                mj.mj_contactForce(m, d, i, c4)
                fr = con.frame.reshape(3, 3)
                f = fr.T @ c4[:3]
                if wb == b1:
                    f = -f
                per_wing[wb] += c4[0]
                j = stroke_j[wing_bodies.index(wb)]
                Mv = np.cross(con.pos - d.xanchor[j], f)
                peak_contact_M = max(peak_contact_M, float(np.linalg.norm(Mv)))
            peak_wing_N = max(peak_wing_N, max(per_wing.values()) / W)
            if t >= T_C + 2.0:
                for wb, j in zip(wing_bodies, stroke_j):
                    F = d.xfrc_applied[wb]
                    Mv = np.cross(d.xipos[wb] - d.xanchor[j], F[:3]) + F[3:]
                    peak_fluid_M = max(peak_fluid_M, float(np.linalg.norm(Mv)))
    rec["hterr"] = rec["cz"] - mc.env.ground_heights(rec["cx"])
    row = dict(arm=label, entry=list(entry), J_r=J_r, gear=gear, contact=contact,
               gains=g if fly is not None and loop else None, bad_qacc=mc.bad_qacc,
               t_fly=None if t_fly is None else t_fly - T_C,
               strike_end=None if strike_end is None else strike_end - T_C,
               wall_s=round(time.time() - t_wall, 1))
    row.update(gates(rec, mc.dt, kc))
    s = mc.summary()
    Pm = rec["P"][kc:]
    def peak_mean(x, n):
        cs = np.concatenate([[0.0], np.cumsum(x)])
        return float(((cs[n:] - cs[:-n]) / n).max())
    nb = int(round(1 / (F_BEAT * mc.dt)))
    se = int(round(((strike_end if strike_end is not None else T_C + 0.25)) / mc.dt))
    row.update(
        E_elec_J=float(Pm.sum() * mc.dt), P_peak_W=float(Pm.max()), P_peak_01s_W=peak_mean(Pm, 1000),
        P_peak_beat_W=peak_mean(Pm, nb), P_last3s_W=float(Pm[-30000:].mean()),
        P_first2s_W=float(Pm[:20000].mean()), three_cells_W=THREE_CELLS_W,
        elec_W_by_role=s["elec_W"], at_limit_by_role=s["at_limit"],
        stroke_on_limit_strike=float(lim_steps[kc:se].mean()) if se > kc else None,
        peak_wing_contact_W=peak_wing_N, peak_contact_moment_Nm=peak_contact_M,
        peak_fluid_moment_Nm=peak_fluid_M,
        moment_ratio=(peak_contact_M / peak_fluid_M) if peak_fluid_M > 0 else None,
        peak_stroke_hinge_Nm=peak_hinge, end=s["pos"], end_pitch_deg=s["pitch_deg"])
    if trace:
        sl = slice(0, n_steps, 500)
        row["trace"] = {k_: np.round(v[sl], 3).tolist() for k_, v in rec.items()}
    return row


def beat_mean(x, nb):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.empty_like(x)
    idx = np.arange(len(x))
    lo = np.maximum(idx + 1 - nb, 0)
    out[:] = (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)
    return out


def gates(rec, dt, kc):
    nb = int(round(1 / (F_BEAT * dt)))
    n = len(rec["H"])
    t = np.arange(n) * dt
    load = rec["load"]
    touch = np.nonzero(load[kc:] > 1e-3)[0]
    t_off = T_C if len(touch) == 0 else (kc + touch[-1] + 1) * dt
    G1 = t_off <= T_C + 4.0 + 1e-9
    last3 = t >= T_C + 6.0
    H, ht = rec["H"], rec["hterr"]
    G2 = bool(np.all(H[last3] >= 2.0) and np.all(ht[last3] >= 2.0))
    up = np.nonzero((H >= 2.0) & (t >= T_C))[0]
    G3 = bool(len(up) > 0 and np.all(H[up[0]:] >= 1.5))
    nose_b, span_b = beat_mean(rec["nose"], nb), beat_mean(rec["span"], nb)
    w = np.stack([beat_mean(rec[k], nb) for k in ("wx", "wy", "wz")], 1)
    wn = np.linalg.norm(w, axis=1)
    after = t >= t_off
    G4 = bool(np.all(nose_b[after] >= -10.0) and np.all(np.abs(span_b[after]) <= 35.0)
              and np.all(wn[last3] <= 1.5))
    first2 = np.nonzero(H >= 2.0)[0]
    post = t >= T_C
    return dict(t_off=t_off - T_C, G1=bool(G1), G2=G2, G3=G3, G4=G4,
                H_max=float(H[post].max()), H_max_first05=float(H[kc:kc + 5000].max()),
                H_end=float(H[-1]), H_min_last3=float(H[last3].min()), hterr_min_last3=float(ht[last3].min()),
                t_H2=None if len(first2) == 0 else float(t[first2[0]] - T_C),
                nose_min_after=float(nose_b[after].min()) if after.any() else None,
                span_absmax_after=float(np.abs(span_b[after]).max()) if after.any() else None,
                rate_max_last3=float(wn[last3].max()),
                nose_mean_last3=float(nose_b[last3].mean()),
                drift_m=float(math.hypot(rec["cx"][-1] - rec["cx"][kc], rec["cy"][-1] - rec["cy"][kc])),
                load_rest=float(load[kc - 1]))


def passed(r):
    return bool(r["G1"] and r["G2"] and r["G3"] and r["G4"] and r["bad_qacc"] == 0)


# ============================================================ authority
def auth_job(a):
    name, out = a
    g = dict(GAINS)
    mc = L.machine(x=17.0)
    mc.place(z=30.0)
    L.set_attitude(mc, 0.0, 90.0)
    fly = Fly(mc, g, loop=False)
    fly.out.update(amp=g["a0"])
    fly.out.update(out)
    fly.t0 = 0.0
    nb = int(round(1 / (F_BEAT * mc.dt)))
    w = []
    for k in range(6 * nb):
        q, qd = fly.targets(mc.t)
        mc.step(q, qd)
        w.append(mc.d.qvel[3:6].copy())
    w = np.array(w)
    beats = [w[i * nb:(i + 1) * nb].mean(0).tolist() for i in range(6)]
    return dict(kind="auth", arm=name, out=out, w_beats=beats, w_mean_beat1=beats[0],
                w_mean_beat2=beats[1], vz_end=float(mc.d.qvel[2]), bad_qacc=mc.bad_qacc)


def auth():
    jobs = [("base", {}), ("bias+0.2", dict(bias=0.2)), ("amp_lr+0.2", dict(amp_lr=0.2)),
            ("psi0_lr+0.2", dict(psi0_lr=0.2))]
    with mp.Pool(4) as p:
        rows = p.map(auth_job, jobs)
    base = rows[0]
    T = 1 / F_BEAT
    for r in rows:
        dw = np.array(r["w_beats"]) - np.array(base["w_beats"])
        r["alpha_est_body"] = np.polyfit(np.arange(1, 6) * T, dw[1:], 1)[0].tolist()   # slope over beats 2-6
        r["dw_beat1"] = dw[0].tolist()
        with open(TUNE_OUT, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(r["arm"], [round(v, 1) for v in r["alpha_est_body"]], "vz", round(r["vz_end"], 2), r["bad_qacc"])


# ================================================================= hover
def hover_job(a):
    """Tuning aid: released at 30 m nose up and still, FLY from the start with ``gains``;
    ``add`` is added to the loop's outputs from ``t_s`` on.  Logged as kind "hover"."""
    name, gains, add, t_s, T = a
    g = dict(GAINS, **gains)
    mc = L.machine(x=17.0)
    mc.place(z=30.0)
    L.set_attitude(mc, 0.0, 90.0)
    fly = Fly(mc, g, loop=True)
    fly.t0 = 0.0
    c0 = mc.com().copy()
    tr = []
    for k in range(int(T / mc.dt)):
        t = k * mc.dt
        fly.sample(float(mc.com()[2] - c0[2]) + g["H_t"])
        if t >= t_s:
            for k_, v in add.items():
                fly.out[k_] = fly.out[k_] + v if k_ != "_hold" else fly.out[k_]
        q, qd = fly.targets(t)
        if t >= t_s:
            for k_, v in add.items():
                pass
        mc.step(q, qd)
        if t >= t_s:
            for k_, v in add.items():
                fly.out[k_] -= v
        if k % 500 == 0:
            nose, left, back = L.axes(mc)
            c = mc.com()
            tr.append([round(t, 2), round(math.degrees(math.asin(clip(nose[2], -1, 1))), 1),
                       round(math.degrees(math.asin(clip(left[2], -1, 1))), 1)] +
                      np.round(c - c0, 2).tolist() + np.round(mc.d.qvel[3:6], 2).tolist() +
                      [round(fly.out[k_], 3) for k_ in ("bias", "amp", "amp_lr", "psi0_lr")])
    r = dict(kind="hover", arm=name, gains=g, add=add, t_s=t_s, trace=tr, bad_qacc=mc.bad_qacc)
    with open(TUNE_OUT, "a") as fh:
        fh.write(json.dumps(r) + "\n")
    return r


# ================================================================ tune/score
def tune_job(a):
    arm, entry, gains, J_r, gear = a
    r = run(arm, entry, J_r=J_r, gear=gear, gains=gains, trace=True)
    r["kind"] = "tune"
    r["passed"] = passed(r)
    with open(TUNE_OUT, "a") as fh:
        fh.write(json.dumps(r) + "\n")
    return r


def summary_line(r):
    return ("%s %s J%.0e g%d pass=%s G=%d%d%d%d qacc=%d t_off=%.2f Hmax=%.2f Hend=%.2f tH2=%s noseMin=%s span=%s rate=%.2f "
            "nose3=%.0f drift=%.1f Ppk=%.0f P3=%.0f wingN=%.1f" % (
                r["arm"], r["entry"], r["J_r"], r["gear"], passed(r), r["G1"], r["G2"], r["G3"], r["G4"],
                r["bad_qacc"], r["t_off"], r["H_max"], r["H_end"], r["t_H2"],
                None if r["nose_min_after"] is None else round(r["nose_min_after"]),
                None if r["span_absmax_after"] is None else round(r["span_absmax_after"]),
                r["rate_max_last3"], r["nose_mean_last3"], r["drift_m"], r["P_peak_W"], r["P_last3s_W"],
                r["peak_wing_contact_W"]))


def score_rows():
    rows = []
    for arm in ("V", "Vs", "P", "VO", "R0", "N0", "N1", "N2"):
        rows += [(arm, e, 2e-5, 24.0, None) for e in SCORING]
    for J in (5e-6, 1e-6):
        for arm in ("V", "P"):
            rows += [(arm, e, J, 32.0, None) for e in SCORING]
        rows += [("V", e, J, 24.0, None) for e in SCORING]
    rows += [("V", e, 2e-5, 24.0, None) for e in STRESS]
    rows += [("V", (20.0, 0.0), 2e-5, 24.0, c) for c in (dict(solref_x=0.5), dict(solref_x=2.0), dict(friction=0.5))]
    # ADDED after data (RESULT section 0b): every V and P row again with the "+" law
    rows += [(a + "+", e, J, gr, c) for (a, e, J, gr, c) in list(rows) if a in ("V", "P")]
    return rows


def score_job(a):
    arm, entry, J_r, gear, contact = a
    r = run(arm, entry, J_r=J_r, gear=gear, contact=contact, trace=True)
    r["kind"] = "score"
    r["set"] = ("stress" if tuple(entry) in STRESS else "contact" if contact else "scoring")
    r["passed"] = passed(r)
    return r


def score():
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with open(SCORE_OUT, "a") as fh:
        fh.write(json.dumps(dict(kind="freeze", file="m1_takeoff.py", sha256=sha, gains=GAINS,
                                 lib_sha256=hashlib.sha256((HERE / "m1_lib.py").read_bytes()).hexdigest(),
                                 mission_sha256=hashlib.sha256((HERE / "mission.py").read_bytes()).hexdigest())) + "\n")
    jobs = score_rows()
    with mp.Pool(4) as p:
        for r in p.imap_unordered(score_job, jobs):
            with open(SCORE_OUT, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(summary_line(r), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "auth":
        auth()
    elif cmd == "tune":
        # tune ARM,ARM x,yaw;x,yaw JSON [J_r gear]
        arms = sys.argv[2].split(",")
        ents = [tuple(float(v) for v in e.split(",")) for e in sys.argv[3].split(";")]
        gains = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
        J_r = float(sys.argv[5]) if len(sys.argv) > 5 else 2e-5
        gear = float(sys.argv[6]) if len(sys.argv) > 6 else 24.0
        jobs = [(a, e, gains, J_r, gear) for a in arms for e in ents]
        with mp.Pool(min(4, len(jobs))) as p:
            for r in p.imap(tune_job, jobs):
                print(summary_line(r), flush=True)
    elif cmd == "score":
        score()
