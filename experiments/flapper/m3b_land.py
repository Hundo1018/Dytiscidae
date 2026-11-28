"""Pass M3b (PREREG_M3b_land.md): answering both halves of the pair by turning.

Arm T: the M0 machine, turn in place by holding one wing planted while the other
runs the forward hop, then hop forward (no backward mode).  Arm K: the sweep
machine (`m3b_sweep.SweepMachine`, a third hinge per shoulder) with a stepping
gait under M3's steering law (backward = the cycle run in reverse).  U (untimed),
N0 (held at 0), N1 (propped, held) for each machine.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m3b_land.py SPECS.json OUT.jsonl
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
from m3_land import (G, PHI_GOLD, T_BLEND, T_C, T_RISE, T_SETTLE, attitude, contact_read, cycle,  # noqa: E402
                     groups, quat_yaw_pitch, wrap)

TRACE_DIR = HERE / "results" / "M3b_traces"
FEATHER_LIM, TAIL_LIM = 54.0, 33.0
SWEEP_LIM = 32.0                                   # deg: 3 deg inside the sweep hinge's +-35

#: arm T: M3's H box with the plant feather at -20 or less, plus the turn law
T0 = dict(T=1.5, th_s=-10.0, drop=10.0, th_up=10.0, fa=-40.0, fb=0.0, D=0.85, tail=-20.0, kp=1.0,
          h_turn=1.0, tsign=1.0, e_go=30.0, k_h=1.0, e_stop=6.0)
BOX_T = dict(T=(0.6, 3.0), th_s=(-25, -5), drop=(0, 15), th_up=(-5, 15), fa=(-54, -20), fb=(-54, 54),
             D=(0.4, 0.85), tail=(-31, 31), kp=(0, 2), h_turn=(0, 1), tsign=(-1, 1), e_go=(10, 60),
             k_h=(0, 3), e_stop=(3, 12))
#: arm K: the stepping gait on the sweep machine
K0 = dict(T=2.0, D=0.6, th_s=-12.0, th_up=5.0, f_s=0.0, A=20.0, sw0=0.0, tail=-20.0, kr=1.0, kp=0.0,
          off=0.0, e_go=30.0, k=1.0, A_turn=1.0, e_stop=6.0)
BOX_K = dict(T=(0.8, 3.0), D=(0.4, 0.85), th_s=(-25, -5), th_up=(-5, 15), f_s=(-15, 15), A=(5, 30),
             sw0=(-10, 10), tail=(-31, 31), kr=(0, 2), kp=(0, 2), off=(0, 0.5), e_go=(10, 60), k=(0, 2),
             A_turn=(0, 1), e_stop=(3, 12))
#: the stance both machines hold in a turn task once there (M3's frozen S for T)
S_T = dict(th_s=-12.0, f_s=0.0, tail=-20.0, kr=1.0)


def check_box(p, box):
    for k, (lo, hi) in box.items():
        if not (lo - 1e-9 <= p[k] <= hi + 1e-9):
            raise ValueError(f"{k}={p[k]} outside the pre-registered box [{lo}, {hi}]")
    if "off" in p and p["off"] not in (0.0, 0.5):
        raise ValueError("off must be 0 or 0.5")
    if "tsign" in p and p["tsign"] not in (-1.0, 1.0):
        raise ValueError("tsign must be +-1")


def sweep_cycle(s, P):
    """Stepping cycle at phase s: (stroke, sweep) deg and rates per unit phase. +sweep = tip toward the nose."""
    D, w = P["D"], (1.0 - P["D"]) / 3.0
    a, b = P["sw0"] + P["A"], P["sw0"] - P["A"]
    ths, thu = P["th_s"], P["th_up"]
    if s < D:                                   # stance: the planted tip swept back, the body goes forward
        u = s / D
        return ths, a + u * (b - a), 0.0, (b - a) / D
    if s < D + w:                               # lift
        u = (s - D) / w
        return ths + u * (thu - ths), b, (thu - ths) / w, 0.0
    if s < D + 2 * w:                           # sweep forward in the air
        u = (s - D - w) / w
        return thu, b + u * (a - b), 0.0, (a - b) / w
    u = (s - D - 2 * w) / w                     # plant
    return thu + u * (ths - thu), a, (ths - thu) / w, 0.0


class Controller:
    def __init__(self, spec, roles, names):
        self.spec, self.arm, self.kind = spec, spec["arm"], spec["kind"]
        self.mach = spec["machine"]                          # "ref" or "sweep"
        self.P = dict(T0 if self.mach == "ref" else K0, **spec.get("gait", {}))
        if self.arm in ("T", "K", "U", "N1", "KS"):
            check_box(self.P, BOX_T if self.mach == "ref" else BOX_K)
        self.roles, self.names = list(roles), list(names)
        self.left = np.array(["seg1" in n for n in names])
        # +sweep = tip forward; the left joint's positive sense is the other way (m3b_probe build)
        self.ssign = np.array([-1.0 if (r == "sweep" and l) else 1.0 for r, l in zip(roles, self.left)])
        self.mode, self.ref_pitch, self.done = None, None, False

    def stance(self):
        if self.mach == "ref":
            return dict(L=(self.P["th_s"], self.P["fa"], 0.0), R=(self.P["th_s"], self.P["fa"], 0.0), tail=self.P["tail"])
        P = self.P
        return dict(L=(P["th_s"], P["f_s"], P["sw0"] + P["A"]), R=(P["th_s"], P["f_s"], P["sw0"] + P["A"]), tail=P["tail"])

    def hold_pose(self):
        """Where a turn task stops: M3's S for T; the sweep machine's stance with the sweep at its offset."""
        if self.mach == "ref":
            s = S_T
            return dict(L=(s["th_s"], s["f_s"], 0.0), R=(s["th_s"], s["f_s"], 0.0), tail=s["tail"]), s["kr"]
        P = self.P
        return dict(L=(P["th_s"], P["f_s"], P["sw0"]), R=(P["th_s"], P["f_s"], P["sw0"]), tail=P["tail"]), P["kr"]

    def pose(self, ps):
        q = np.zeros(len(self.roles))
        for n, r in enumerate(self.roles):
            side = ps["L"] if self.left[n] else ps["R"]
            q[n] = {"stroke": side[0], "feather": side[1], "sweep": side[2]}.get(r, ps["tail"]) if r != "tail" else ps["tail"]
        return q * self.ssign

    def __call__(self, t, att):
        z = np.zeros(len(self.roles))
        if self.arm == "N0" or t < T_SETTLE:
            return z, z
        st = self.stance() if self.arm != "KS" else self.hold_pose()[0]
        if t < T_C:
            u = (t - T_SETTLE) / T_RISE
            q0 = np.radians(self.pose(st))
            return u * q0, q0 / T_RISE
        yaw, pitch, roll = att
        if self.ref_pitch is None:
            self.ref_pitch = pitch
        if self.arm == "N1":
            return np.radians(self.pose(st)), z
        if self.arm == "KS":                             # stand check: the stance with the roll loop faded in
            hp, kr = self.hold_pose()
            ro = min(1.0, (t - T_C) / T_BLEND) * kr * math.degrees(roll)
            hp = dict(L=(hp["L"][0] + ro, hp["L"][1], hp["L"][2]), R=(hp["R"][0] - ro, hp["R"][1], hp["R"][2]),
                      tail=hp["tail"])
            return np.radians(self.pose(hp)), z
        h = self.spec["heading"]
        e = wrap(h - yaw)
        tg = t - T_C
        P = self.P
        if self.kind == "turn":                          # turn, then stop and hold
            if self.done and abs(e) > math.radians(P["e_stop"] + 10):
                self.done = False
            if not self.done and abs(e) <= math.radians(P["e_stop"]):
                self.done, self.t_done, self.q_done = True, t, None
            if self.done:
                hp, kr = self.hold_pose()
                ro = kr * math.degrees(roll)
                hp = dict(L=(hp["L"][0] + ro, hp["L"][1], hp["L"][2]), R=(hp["R"][0] - ro, hp["R"][1], hp["R"][2]),
                          tail=hp["tail"])
                q1 = self.pose(hp)
                if self.q_done is None:
                    self.q_done = self.last
                u = min(1.0, (t - self.t_done) / T_BLEND)
                q = self.q_done + u * (q1 - self.q_done)
                self.last = q
                return np.radians(q), z
        q, qd = (self.hop if self.mach == "ref" else self.step)(tg, e, pitch, roll)
        if tg < T_BLEND:
            q0 = self.pose(st)
            u = tg / T_BLEND
            q, qd = q0 + u * (q - q0), u * qd + (q - q0) / T_BLEND
        self.last = q
        return np.radians(q), np.radians(qd)

    # ---------------------------------------------------------------- arm T
    def hop(self, tg, e, pitch, roll):
        P, sp = self.P, self.spec
        ego = math.radians(P["e_go"])
        hyst = math.radians(10.0)
        turning = abs(e) > ego + (hyst if self.mode == "walk" else 0.0) if self.kind == "walk" else True
        self.mode = "turn" if turning else "walk"
        hold = P["h_turn"] if turning else min(P["h_turn"], P["k_h"] * abs(e))
        # inner side for a CCW (e > 0) turn: right if tsign > 0
        inner = ("R" if P["tsign"] > 0 else "L") if e > 0 else ("L" if P["tsign"] > 0 else "R")
        ph = sp.get("phase", 0.0)
        sf = (ph + tg / P["T"]) % 1.0
        ss = (ph + PHI_GOLD * tg / P["T"]) % 1.0 if self.arm == "U" else sf
        rs = (PHI_GOLD if self.arm == "U" else 1.0) / P["T"]
        th, _, dth, _ = cycle(ss, P)
        _, f, _, df = cycle(sf, P)
        fo = P["kp"] * math.degrees(self.ref_pitch - pitch)
        out = {}
        for side in ("L", "R"):
            hh = hold if side == inner else 0.0
            ths, fs = P["th_s"], P["fa"]
            out[side] = (ths + (1 - hh) * (th - ths), float(np.clip(fs + (1 - hh) * (f - fs) + fo, -FEATHER_LIM, FEATHER_LIM)),
                         (1 - hh) * dth * rs, (1 - hh) * df / P["T"])
        q = self.pose(dict(L=(out["L"][0], out["L"][1], 0.0), R=(out["R"][0], out["R"][1], 0.0), tail=P["tail"]))
        qd = self.pose(dict(L=(out["L"][2], out["L"][3], 0.0), R=(out["R"][2], out["R"][3], 0.0), tail=0.0))
        return q, qd

    # ---------------------------------------------------------------- arm K
    def step(self, tg, e, pitch, roll):
        P, sp = self.P, self.spec
        ego = math.radians(P["e_go"])
        hyst = math.radians(10.0) if self.mode is not None else 0.0
        e_f, e_b = e, wrap(e - math.pi)
        if self.kind == "walk" and (self.mode == 1 and abs(e_f) <= ego + hyst or self.mode != 1 and abs(e_f) <= ego):
            self.mode, ee = 1, e_f
        elif self.kind == "walk" and (self.mode == -1 and abs(e_b) <= ego + hyst or self.mode != -1 and abs(e_b) <= ego):
            self.mode, ee = -1, e_b
        else:
            self.mode, ee = 0, e_f
        if self.mode == 0:
            gR, gL = P["A_turn"] * math.copysign(1, ee), -P["A_turn"] * math.copysign(1, ee)
        else:
            gR, gL = self.mode + P["k"] * ee, self.mode - P["k"] * ee
        gR, gL = float(np.clip(gR, -1, 1)), float(np.clip(gL, -1, 1))
        ph = sp.get("phase", 0.0)
        sw_s = (ph + tg / P["T"]) % 1.0                                   # the sweep's clock
        st_s = (ph + PHI_GOLD * tg / P["T"]) % 1.0 if self.arm == "U" else sw_s   # the stroke's (U: untimed)
        rate_st = (PHI_GOLD if self.arm == "U" else 1.0) / P["T"]
        out = {}
        for side, g, off in (("L", gL, 0.0), ("R", gR, P["off"])):
            rev = -1.0 if g < 0 else 1.0
            a_w, a_s = (sw_s + off) % 1.0, (st_s + off) % 1.0
            if rev < 0:
                a_w, a_s = (1 - a_w) % 1.0, (1 - a_s) % 1.0
            _, sw, _, dsw = sweep_cycle(a_w, P)
            th, _, dth, _ = sweep_cycle(a_s, P)
            sw = P["sw0"] + abs(g) * (sw - P["sw0"])
            out[side] = (th, sw, rev * dth * rate_st, rev * abs(g) * dsw / P["T"])
        ro = P["kr"] * math.degrees(roll)
        fo = float(np.clip(P["f_s"] + P["kp"] * math.degrees(self.ref_pitch - pitch), -FEATHER_LIM, FEATHER_LIM))
        clip = lambda v: float(np.clip(v, -SWEEP_LIM, SWEEP_LIM))
        q = self.pose(dict(L=(out["L"][0] + ro, fo, clip(out["L"][1])), R=(out["R"][0] - ro, fo, clip(out["R"][1])),
                           tail=P["tail"]))
        qd = self.pose(dict(L=(out["L"][2], 0.0, out["L"][3]), R=(out["R"][2], 0.0, out["R"][3]), tail=0.0))
        return q, qd


def run(spec):
    """spec: id, machine (ref|sweep), arm (T K U N0 N1), kind (walk|turn), pair, yaw0_deg, x, phase,
    heading_deg or turn_deg, window_s, gait{}, J_r (opt)."""
    from mission import Machine
    from m3b_sweep import SweepMachine
    t_wall = time.time()
    make = Machine if spec["machine"] == "ref" else SweepMachine
    mc = make("land", pose=dict(x=spec.get("x", 20.0), z=5))
    yaw0 = math.radians(spec.get("yaw0_deg", 0.0))
    mc.d.qpos[3:7] = quat_yaw_pitch(yaw0, 0.0)
    mc.mj.mj_forward(mc.m, mc.d)
    mc.drop_to_ground()
    sp = dict(spec)
    sp["heading"] = (wrap(yaw0 + math.radians(spec["turn_deg"])) if spec["kind"] == "turn"
                     else math.radians(spec.get("heading_deg", spec.get("yaw0_deg", 0.0))))
    ctl = Controller(sp, mc.role, mc.env.act_names)
    gid, c = groups(mc), np.zeros(6)
    window = float(spec["window_s"])
    n_total, n_c = int(round((T_C + window) / mc.dt)), int(round(T_C / mc.dt))
    every = int(round(0.04 / mc.dt))
    rng_lim = np.array([mc.m.jnt_range[j][1] for j in mc.jid])
    tr = dict(t=[], com=[], att=[], q=[], qpos=[], vert=[], E=[])
    acc = dict(vert=np.zeros(4), steps_body=0, steps_air=0, steps=0, peak_norm=np.zeros(4),
               peak_tau=np.zeros(len(mc.aid)), at_lim=np.zeros(len(mc.aid)), peak_P=0.0, peak_P_all=0.0)
    hash_c = com_c = None
    for k in range(n_total):
        t = k * mc.dt
        if k == n_c:
            h = hashlib.sha256()
            for a in (mc.d.qpos, mc.d.qvel, mc.d.act, mc.E_elec):
                h.update(np.ascontiguousarray(a).tobytes())
            hash_c = h.hexdigest()[:16]
            com_c, E0, lim0 = mc.com().copy(), mc.E_elec.copy(), mc.at_limit.copy()
        att = attitude(mc.d.qpos[3:7])
        q, qd = ctl(t, att)
        e0 = mc.E_elec.sum()
        mc.step(q, qd)
        P_all = (mc.E_elec.sum() - e0) / mc.dt
        acc["peak_P_all"] = max(acc["peak_P_all"], P_all)
        if k >= n_c:
            vert, norm = contact_read(mc, gid, c)
            acc["vert"] += vert
            acc["steps"] += 1
            acc["steps_body"] += int(norm[3] > 1e-6)
            acc["steps_air"] += int(norm.sum() <= 1e-6)
            acc["peak_norm"] = np.maximum(acc["peak_norm"], norm)
            acc["peak_tau"] = np.maximum(acc["peak_tau"], np.abs(mc.d.actuator_force[mc.aid]))
            acc["at_lim"] += np.abs(mc.d.qpos[mc.qad]) >= rng_lim - math.radians(0.5)
            acc["peak_P"] = max(acc["peak_P"], P_all)
        if k % every == 0:
            tr["t"].append(t)
            tr["com"].append(mc.com().tolist())
            tr["att"].append(list(att))
            tr["q"].append(mc.d.qpos[mc.qad].tolist())
            tr["qpos"].append(mc.d.qpos.tolist())
            tr["vert"].append(contact_read(mc, gid, c)[0].tolist())
            tr["E"].append(float(mc.E_elec.sum()))
    roles = list(mc.role)
    by_role = lambda arr, f=max: {r: float(f(arr[[i for i, x in enumerate(roles) if x == r]])) for r in sorted(set(roles))}
    tv, com_t, att_t = np.array(tr["t"]), np.array(tr["com"]), np.array(tr["att"])
    E_t = np.array(tr["E"])
    hv = np.array([math.cos(sp["heading"]), math.sin(sp["heading"])])
    hp = np.array([-hv[1], hv[0]])
    row = dict(spec, mass=mc.mass, bad_qacc=int(mc.bad_qacc), hash_c=hash_c, wall_s=round(time.time() - t_wall, 1))
    win = tv >= T_C
    for W in sorted({12.0, 14.0, window}):                    # the M3 gate windows and the full one
        if W > window + 1e-9:
            continue
        i = int(np.argmin(np.abs(tv - (T_C + W))))
        d = com_t[i, :2] - com_c[:2]
        err = np.degrees(np.abs((att_t[:, 0] - sp["heading"] + np.pi) % (2 * np.pi) - np.pi))
        last = (tv >= T_C + W - 2.0) & (tv <= T_C + W + 1e-9)
        yaw_u = np.unwrap(att_t[:, 0])
        tag = "%g" % W
        row[f"along_{tag}"] = float(d @ hv)
        row[f"across_{tag}"] = float(d @ hp)
        row[f"err_mean_{tag}"] = float(err[last].mean())
        row[f"err_max_{tag}"] = float(err[last].max())
        row[f"dyaw_{tag}"] = float(np.degrees(yaw_u[i] - yaw_u[win][0]))
        row[f"E_{tag}_J"] = float(E_t[i] - E0.sum())
    row["drift_m"] = float(np.linalg.norm(com_t[-1, :2] - com_c[:2]))
    row["dyaw_deg"] = float(np.degrees(np.unwrap(att_t[:, 0])[-1] - np.unwrap(att_t[:, 0])[win][0]))
    pw = np.degrees(att_t[win, 1])
    row.update(pitch_p2p_deg=float(pw.max() - pw.min()),
               share=dict(zip(["wingL", "wingR", "tail", "body"], (acc["vert"] / max(acc["vert"].sum(), 1e-12)).tolist())),
               body_contact_frac=acc["steps_body"] / max(acc["steps"], 1),
               airborne_frac=acc["steps_air"] / max(acc["steps"], 1),
               peak_norm_N=dict(zip(["wingL", "wingR", "tail", "body"], acc["peak_norm"].tolist())),
               peak_tau_Nm=by_role(acc["peak_tau"]),
               at_curve_frac=by_role((mc.at_limit - lim0) / max(acc["steps"], 1)),
               at_range_frac=by_role(acc["at_lim"] / max(acc["steps"], 1)),
               E_window_J=by_role(mc.E_elec - E0, sum), P_window_W=float((mc.E_elec - E0).sum() / window),
               peak_P_window_W=acc["peak_P"], peak_P_run_W=acc["peak_P_all"])
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TRACE_DIR / f"{spec['id']}.npz", **{k: np.array(v) for k, v in tr.items()})
    return row


def main(argv):
    specs = json.load(open(argv[0]))
    out = Path(argv[1])
    with mp.get_context("spawn").Pool(min(4, len(specs))) as pool, open(out, "a") as fh:
        for r in pool.imap_unordered(run, specs, chunksize=1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            W = "%g" % r["window_s"]
            print(r["id"], r["arm"], r["kind"], "along12 %.3f along%s %.3f dyaw %.1f err %.1f air %.2f qacc %d wall %.0f" % (
                r.get("along_12", float("nan")), W, r[f"along_{W}"], r[f"dyaw_{W}"], r[f"err_mean_{W}"], r["airborne_frac"],
                r["bad_qacc"], r["wall_s"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
