"""Pass M3 (PREREG_M3_land.md): standing, walking and turning on the wings.

Arms N0, N1, S, G1, G2, U (untimed), F (heading frozen), J (rotor), each a
run of `mission.Machine("land")` with a controller that reads the true state.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m3_land.py SPECS.json OUT.jsonl [--jobs 4]

SPECS.json is a list of run specs (see `run`); each row is appended to OUT as
it finishes, and a 25 Hz trace goes to results/M3_traces/<id>.npz.
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

G = 9.80665
T_SETTLE, T_RISE, T_BLEND = 1.0, 1.0, 0.3
T_C = T_SETTLE + T_RISE
PHI_GOLD = 1.618
FEATHER_LIM, TAIL_LIM = 54.0, 33.0          # deg: 3 deg inside the hinge (feather), tail mean box 31 + margin
TRACE_DIR = HERE / "results" / "M3_traces"

#: a gait's parameters (deg, s); the prereg's boxes, checked in `check_box`
GAIT0 = dict(T=1.5, th_s=-10.0, drop=10.0, th_up=8.0, fa=-50.0, fb=-10.0, D=0.6,
             tail=-20.0, tail_amp=0.0, tail_ph=0.0, kp=0.0, kr=0.0, e_go=40.0, k=1.0, A_turn=1.0, sgn=1.0)
STAND0 = dict(th_s=-12.0, f_s=0.0, tail=-20.0, kp=0.0, kr=0.0)
BOX_G = dict(T=(0.6, 3.0), th_s=(-25, -5), drop=(0, 15), th_up=(-5, 15), fa=(-54, 54), fb=(-54, 54),
             D=(0.4, 0.85), tail=(-31, 31), tail_amp=(0, 20), tail_ph=(-1e9, 1e9), kp=(0, 2), kr=(0, 2),
             e_go=(20, 60), k=(0, 2), A_turn=(0, 1), sgn=(-1, 1))
BOX_S = dict(th_s=(-20, -6), f_s=(-15, 15), tail=(-31, 31), kp=(0, 2), kr=(0, 2))


def check_box(p: dict, box: dict) -> None:
    for k, (lo, hi) in box.items():
        if not (lo - 1e-9 <= p[k] <= hi + 1e-9):
            raise ValueError(f"{k}={p[k]} outside the pre-registered box [{lo}, {hi}]")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def attitude(q):
    """yaw (body x heading), pitch (nose up), roll (left side up), rad, from the free joint's quaternion."""
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return math.atan2(R[1, 0], R[0, 0]), math.asin(max(-1, min(1, R[2, 0]))), math.asin(max(-1, min(1, R[2, 1])))


def quat_yaw_pitch(yaw, pitch):
    """Body yaw about world z, then nose-up pitch (as mission.place's sign)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(-pitch / 2), math.sin(-pitch / 2)
    # q = qz(yaw) * qy(-pitch)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])


# ------------------------------------------------------------------- gait
def cycle(s: float, P: dict):
    """Forward cycle at phase s in [0, 1): (stroke, feather) deg, and their rates per unit phase."""
    D, w = P["D"], (1.0 - P["D"]) / 3.0
    ths, thd, thu, fa, fb = P["th_s"], P["th_s"] - P["drop"], P["th_up"], P["fa"], P["fb"]
    if s < D:                                   # power: sweep fa -> fb, stroke drops to hold height
        u = s / D
        return ths + u * (thd - ths), fa + u * (fb - fa), (thd - ths) / D, (fb - fa) / D
    if s < D + w:                               # lift
        u = (s - D) / w
        return thd + u * (thu - thd), fb, (thu - thd) / w, 0.0
    if s < D + 2 * w:                           # return the feather in the air
        u = (s - D - w) / w
        return thu, fb + u * (fa - fb), 0.0, (fa - fb) / w
    u = (s - D - 2 * w) / w                     # plant
    return thu + u * (ths - thu), fa, (ths - thu) / w, 0.0


class Controller:
    """One per half; no state crosses halves.  Reads the true state."""

    def __init__(self, spec: dict, roles, names):
        self.spec, self.arm = spec, spec["arm"]
        self.P = dict(GAIT0, **spec.get("gait", {}))
        self.S = dict(STAND0, **spec.get("stand", {}))
        if self.arm in ("G1", "G2", "U", "F", "N1", "H"):
            check_box(self.P, BOX_G)
        if self.arm == "S":
            check_box(self.S, BOX_S)
        self.roles = list(roles)
        self.left = np.array(["seg1" in n for n in names])
        self.right = np.array(["seg2" in n for n in names])
        self.mode = None                        # walking direction latch: +1, -1 or 0 (turn on the spot)
        self.ref_pitch = None
        self.heading = None

    def stance(self):
        if self.arm == "S":
            return self.S["th_s"], self.S["f_s"], self.S["tail"]
        if self.arm == "N0":
            return 0.0, 0.0, 0.0
        return self.P["th_s"], self.P["fa"], self.P["tail"]

    def pose(self, th_l, th_r, f_l, f_r, tail):
        q = np.zeros(len(self.roles))
        for n, r in enumerate(self.roles):
            if r == "stroke":
                q[n] = th_l if self.left[n] else th_r
            elif r == "feather":
                q[n] = f_l if self.left[n] else f_r
            else:
                q[n] = tail
        return q

    def __call__(self, t, state):
        """Targets (rad) and rates (rad/s) for every hinge."""
        th0, f0, ta0 = self.stance()
        if self.arm == "N0" or t < T_SETTLE:
            z = np.zeros(len(self.roles))
            return z, z
        if t < T_C:                              # rise, identical for both halves
            u = (t - T_SETTLE) / T_RISE
            q0 = np.radians(self.pose(th0, th0, f0, f0, ta0))
            return u * q0, q0 / T_RISE
        yaw, pitch, roll = state["att"]
        if self.ref_pitch is None:
            self.ref_pitch = pitch
        if self.arm == "N1":
            q = np.radians(self.pose(th0, th0, f0, f0, ta0))
            return q, np.zeros_like(q)
        if self.arm == "S":                     # loops faded in over T_BLEND (a target step saturates the PD)
            u = min(1.0, (t - T_C) / T_BLEND)
            fo = u * self.S["kp"] * math.degrees(self.ref_pitch - pitch)
            ro = u * self.S["kr"] * math.degrees(roll)
            q = self.pose(th0 + ro, th0 - ro, f0 + fo, f0 + fo, ta0)
            q[[n for n, r in enumerate(self.roles) if r == "feather"]] = np.clip(
                q[[n for n, r in enumerate(self.roles) if r == "feather"]], -FEATHER_LIM, FEATHER_LIM)
            return np.radians(q), np.zeros_like(q)
        return self.gait(t, yaw, pitch, roll)

    def gait(self, t, yaw, pitch, roll):
        P, sp = self.P, self.spec
        tg = t - T_C
        h = sp["heading"] if self.arm != "F" else sp["frozen_heading"]
        # steering: travel direction latch with 10 deg of hysteresis (F2's switch)
        e_fwd, e_bwd = wrap(h - yaw), wrap(h - yaw - math.pi)
        ego = math.radians(P["e_go"])
        hyst = math.radians(10.0) if self.mode is not None else 0.0
        if self.mode == 1 and abs(e_fwd) <= ego + hyst or self.mode != 1 and abs(e_fwd) <= ego:
            self.mode, e = 1, e_fwd
        elif self.mode == -1 and abs(e_bwd) <= ego + hyst or self.mode != -1 and abs(e_bwd) <= ego:
            self.mode, e = -1, e_bwd
        else:
            self.mode, e = 0, e_fwd
        sgn = P["sgn"]
        if self.mode == 0:                      # turn on the spot toward the nearer way
            gR, gL = P["A_turn"] * math.copysign(1, e), -P["A_turn"] * math.copysign(1, e)
        else:
            d = self.mode
            gR, gL = d + P["k"] * e, d - P["k"] * e
        gR, gL = float(np.clip(sgn * gR, -1, 1)), float(np.clip(sgn * gL, -1, 1))
        ph = sp.get("phase", 0.0)
        sf = (ph + tg / P["T"]) % 1.0
        ss = (ph + PHI_GOLD * tg / P["T"]) % 1.0 if self.arm == "U" else sf
        shift = 0.5 if (sp.get("family") == "G2" or self.arm == "G2") else 0.0
        fmid = 0.5 * (P["fa"] + P["fb"])
        out = {}
        for side, g, off in (("L", gL, 0.0), ("R", gR, shift)):
            # sign of g: the cycle run forward or time-reversed (backward); |g| scales the sweep
            rev = -1.0 if g < 0 else 1.0
            a_s, a_f = (ss + off) % 1.0, (sf + off) % 1.0
            if rev < 0:
                a_s, a_f = (1.0 - a_s) % 1.0, (1.0 - a_f) % 1.0
            th, _, dth, _ = cycle(a_s, P)
            _, f, _, df = cycle(a_f, P)
            rate_s = rev * (PHI_GOLD if self.arm == "U" else 1.0) / P["T"]
            out[side] = (th, fmid + abs(g) * (f - fmid), dth * rate_s, abs(g) * df * rev / P["T"])
        fo = P["kp"] * math.degrees(self.ref_pitch - pitch)
        ro = P["kr"] * math.degrees(roll)
        ta = P["tail"] + P["tail_amp"] * math.sin(2 * math.pi * sf + P["tail_ph"])
        dta = P["tail_amp"] * 2 * math.pi / P["T"] * math.cos(2 * math.pi * sf + P["tail_ph"])
        thL, thR = out["L"][0] + ro, out["R"][0] - ro
        fL = float(np.clip(out["L"][1] + fo, -FEATHER_LIM, FEATHER_LIM))
        fR = float(np.clip(out["R"][1] + fo, -FEATHER_LIM, FEATHER_LIM))
        ta = float(np.clip(ta, -TAIL_LIM, TAIL_LIM))
        q = self.pose(thL, thR, fL, fR, ta)
        qd = self.pose(out["L"][2], out["R"][2], out["L"][3], out["R"][3], dta)
        if tg < T_BLEND:                         # blend from the stance pose into the gait
            u = tg / T_BLEND
            th0, f0, ta0 = self.stance()
            q0 = self.pose(th0, th0, f0, f0, ta0)
            q, qd = q0 + u * (q - q0), u * qd + (q - q0) / T_BLEND
        return np.radians(q), np.radians(qd)


# -------------------------------------------------------------------- run
def groups(mc):
    m = mc.m
    names = [mc.mj.mj_id2name(m, mc.mj.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    gid = np.full(m.nbody, -1)
    for b, n in enumerate(names):
        if n in ("seg1_wing",):
            gid[b] = 0
        elif n == "seg2_wing":
            gid[b] = 1
        elif n in ("seg4_wing", "seg5_wing"):
            gid[b] = 2
        elif b > 0:
            gid[b] = 3                           # hull and boom: "body"
    return gid


def contact_read(mc, gid, c):
    """Per group: vertical ground force (N) and normal force magnitude (N) this step."""
    m, d, mj = mc.m, mc.d, mc.mj
    vert, norm = np.zeros(4), np.zeros(4)
    for i in range(d.ncon):
        con = d.contact[i]
        b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
        if (b1 == 0) == (b2 == 0):
            continue
        b = b2 if b1 == 0 else b1
        mj.mj_contactForce(m, d, i, c)
        fn = abs(c[0])
        g = gid[b]
        norm[g] += fn
        vert[g] += abs(c[0] * con.frame[2])
    return vert, norm


def run(spec: dict) -> dict:
    """spec: id, arm (N0 N1 S G1 G2 U F), family (G1/G2 for U, F, N1), kind (stand walk turn tune),
    yaw0_deg, x, phase, heading_deg (walk) or turn_deg (turn), window_s, gait{}, stand{}, J_r (opt)."""
    from mission import Machine
    t_wall = time.time()
    motors = None
    if spec.get("J_r"):
        f = spec["J_r"] / 2e-5
        motors = dict(stroke=dict(J_r=spec["J_r"]), feather=dict(J_r=spec["J_r"]), tail=dict(J_r=2e-6 * f))
    mc = Machine("land", pose=dict(x=spec.get("x", 20.0), z=5), motors=motors)
    yaw0 = math.radians(spec.get("yaw0_deg", 0.0))
    mc.d.qpos[3:7] = quat_yaw_pitch(yaw0, 0.0)
    mc.mj.mj_forward(mc.m, mc.d)
    mc.drop_to_ground()
    sp = dict(spec)
    if spec["kind"] == "turn":
        sp["heading"] = wrap(yaw0 + math.radians(spec["turn_deg"]))
    elif "heading_deg" in spec:
        sp["heading"] = math.radians(spec["heading_deg"])
    if spec["arm"] == "F":
        sp["frozen_heading"] = math.radians(spec["frozen_heading_deg"])
    ctl = Controller(sp, mc.role, mc.env.act_names)
    gid = groups(mc)
    c = np.zeros(6)
    window = float(spec["window_s"])
    n_total = int(round((T_C + window) / mc.dt))
    n_c = int(round(T_C / mc.dt))
    every = int(round(0.04 / mc.dt))
    rng_lim = np.array([mc.m.jnt_range[j][1] for j in mc.jid])
    tr = dict(t=[], com=[], att=[], q=[], qpos=[], vert=[])
    acc = dict(vert=np.zeros(4), steps_body=0, steps_air=0, steps=0, peak_norm=np.zeros(4), peak_vert_total=0.0,
               peak_tau=np.zeros(len(mc.aid)), at_lim_steps=np.zeros(len(mc.aid)), peak_P=0.0, peak_P_all=0.0)
    hash_c = com_c = att_c = None
    E0 = None
    lim0 = None
    for k in range(n_total):
        t = k * mc.dt
        if k == n_c:
            h = hashlib.sha256()
            for a in (mc.d.qpos, mc.d.qvel, mc.d.act, mc.E_elec):
                h.update(np.ascontiguousarray(a).tobytes())
            hash_c = h.hexdigest()[:16]
            com_c, att_c = mc.com().copy(), attitude(mc.d.qpos[3:7])
            E0, lim0 = mc.E_elec.copy(), mc.at_limit.copy()
        att = attitude(mc.d.qpos[3:7])
        q, qd = ctl(t, dict(att=att))
        e_before = mc.E_elec.sum()
        mc.step(q, qd)
        P_all = (mc.E_elec.sum() - e_before) / mc.dt
        acc["peak_P_all"] = max(acc["peak_P_all"], P_all)
        if k >= n_c:
            vert, norm = contact_read(mc, gid, c)
            acc["vert"] += vert
            acc["steps"] += 1
            acc["steps_body"] += int(norm[3] > 1e-6)
            acc["steps_air"] += int(norm.sum() <= 1e-6)
            acc["peak_norm"] = np.maximum(acc["peak_norm"], norm)
            acc["peak_vert_total"] = max(acc["peak_vert_total"], vert.sum())
            acc["peak_tau"] = np.maximum(acc["peak_tau"], np.abs(mc.d.actuator_force[mc.aid]))
            acc["at_lim_steps"] += np.abs(mc.d.qpos[mc.qad]) >= rng_lim - math.radians(0.5)
            acc["peak_P"] = max(acc["peak_P"], P_all)
        if k % every == 0:
            tr["t"].append(t)
            tr["com"].append(mc.com().tolist())
            tr["att"].append(list(att))
            tr["q"].append(mc.d.qpos[mc.qad].tolist())
            tr["qpos"].append(mc.d.qpos.tolist())
            tr["vert"].append(contact_read(mc, gid, c)[0].tolist())
    com_e, att_e = mc.com(), attitude(mc.d.qpos[3:7])
    dxy = (com_e - com_c)[:2]
    roles = list(mc.role)
    by_role = lambda arr, f=max: {r: float(f(arr[[i for i, x in enumerate(roles) if x == r]])) for r in sorted(set(roles))}
    E = mc.E_elec - E0
    tv = np.array(tr["t"])
    yaw_tr = np.unwrap(np.array(tr["att"])[:, 0])
    win = tv >= T_C
    pitch_w = np.degrees(np.array(tr["att"])[win, 1])
    row = dict(spec, bad_qacc=int(mc.bad_qacc), hash_c=hash_c, wall_s=round(time.time() - t_wall, 1),
               dx=float(dxy[0]), dy=float(dxy[1]), dz=float(com_e[2] - com_c[2]),
               yaw_c_deg=math.degrees(att_c[0]), yaw_end_deg=math.degrees(att_e[0]),
               dyaw_deg=float(np.degrees(yaw_tr[-1] - yaw_tr[win][0])),
               pitch_p2p_deg=float(pitch_w.max() - pitch_w.min()), roll_end_deg=math.degrees(att_e[2]),
               share=dict(zip(["wingL", "wingR", "tail", "body"], (acc["vert"] / max(acc["vert"].sum(), 1e-12)).tolist())),
               body_contact_frac=acc["steps_body"] / max(acc["steps"], 1),
               airborne_frac=acc["steps_air"] / max(acc["steps"], 1),
               mean_load_W=float(acc["vert"].sum() / max(acc["steps"], 1) / (mc.mass * G)),
               peak_norm_N=dict(zip(["wingL", "wingR", "tail", "body"], acc["peak_norm"].tolist())),
               peak_vert_total_N=acc["peak_vert_total"],
               peak_tau_Nm=by_role(acc["peak_tau"]),
               at_curve_frac=by_role((mc.at_limit - lim0) / max(acc["steps"], 1)),
               at_range_frac=by_role(acc["at_lim_steps"] / max(acc["steps"], 1)),
               E_window_J=by_role(E, sum), P_window_W=float(E.sum() / window),
               peak_P_window_W=acc["peak_P"], peak_P_run_W=acc["peak_P_all"])
    if "heading" in sp:
        hv = np.array([math.cos(sp["heading"]), math.sin(sp["heading"])])
        row["along"] = float(dxy @ hv)
        row["across"] = float(dxy @ np.array([-hv[1], hv[0]]))
    if spec["kind"] == "turn":
        err = np.degrees(np.array([wrap(a - sp["heading"]) for a in np.array(tr["att"])[:, 0]]))
        last = tv >= tv[-1] - 2.0
        row["err_last2_mean_deg"] = float(np.abs(err[last]).mean())
        row["err_last2_max_deg"] = float(np.abs(err[last]).max())
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TRACE_DIR / f"{spec['id']}.npz", **{k: np.array(v) for k, v in tr.items()})
    return row


def main(argv) -> int:
    specs = json.load(open(argv[0]))
    out = Path(argv[1])
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 4
    jobs = min(jobs, 4)
    with mp.get_context("spawn").Pool(jobs) as pool, open(out, "a") as fh:
        for r in pool.imap_unordered(run, specs, chunksize=1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(r["id"], r["arm"], r["kind"], "along %.3f" % r.get("along", float("nan")),
                  "dyaw %.1f" % r["dyaw_deg"], "pitch_p2p %.1f" % r["pitch_p2p_deg"],
                  "body %.2f" % r["body_contact_frac"], "qacc", r["bad_qacc"], "wall", r["wall_s"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
