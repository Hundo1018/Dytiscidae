"""M4 design probe: C1's cruise with all six axes free (C1 released x, z and
pitch only).  A probe for designing pass M4, not a measurement: every run is
disclosed in M4's pre-registration.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m4_probe.py OUT.jsonl [roll gains...]

The machine of C1 (BODY_S, W6 wing, 0.06 m2 tail, hull transverse inertia
0.012 kg m2) on `mission.Machine`, every hinge on its witness motor; the
tip's feather by the strips turned per side (D3's stand-in, stated in
`mission.enable_strip_twist`).  Laws read the true state.  Pitch, height and
speed: C1's tail PI and pilot, as `freerig` writes them.  Roll: the two tips'
feather offset in opposite senses, ``d = kr * roll + krd * roll rate`` (deg).
Nothing for yaw.
"""
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
I0 = 0.012
TL = dict(i0_deg=0.0, kp=1.0, kd=0.1, ki=2.0, limit_deg=25.0, rate=8.0, lag_beats=1.0)
AP = dict(kh=6.0, kvz=6.0, pitch_lim_deg=12.0, kf=0.08, f_lim=0.2)


def fly(pt, kr, krd, beats=40, V=8.0, z0=30.0, roll0=0.0, sensors=False, seed=0):
    import c1_pilot
    from mission import Machine
    f, Phi_d, Psi_d, psi0_d, _ = pt
    mc = Machine("air", body=c1_pilot.body(), pose=dict(x=-40.0, z=z0, pitch_deg=0.0, vx=V), seed=seed)
    m, d, mj = mc.m, mc.d, mc.mj
    h = mc.hull
    long_axis = int(np.argmin(m.body_inertia[h]))
    for ax in range(3):
        if ax != long_axis:
            m.body_inertia[h, ax] = I0
            mc.env.solver._dry_inertia[h, ax] = I0
    mj.mj_setConst(m, mj.MjData(m))
    if roll0:
        a = math.radians(roll0)
        qr = np.array([math.cos(a / 2), math.sin(a / 2), 0.0, 0.0])
        q = d.qpos[3:7].copy()
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = qr
        d.qpos[3:7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                       w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
        mj.mj_forward(m, d)
    mc.enable_strip_twist(c1_pilot.body())
    S = None
    if sensors:
        from sensors import Sensors
        S = Sensors(mc, seed=seed, init_true=True)
    w0 = 2 * math.pi * f
    Phi, Psi, psi0 = math.radians(Phi_d), math.radians(Psi_d), math.radians(psi0_d)
    st, fe, ta = mc.role == "stroke", mc.role == "feather", mc.role == "tail"
    side = np.array([1.0 if "seg1" in n else -1.0 for n in mc.env.act_names])
    tail_q, e_f, r_f, e_i, zf, vzf, vxf = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, V
    ph, w = 0.0, w0
    n_beat = int(round(1.0 / (f * mc.dt)))
    rec = []
    lim = math.radians(TL["limit_deg"])
    for k in range(beats * n_beat):
        a_f = mc.dt * f / TL["lag_beats"] * 2.0 * math.pi
        if S is not None:
            r_ = S.last
            pitch = math.radians(r_["pitch_deg"]) if r_ else 0.0
            roll = math.radians(r_["roll_deg"]) if r_ else 0.0
            p_rate = -(r_["gyro"][1]) if r_ else 0.0
            r_rate = (r_["gyro"][0]) if r_ else 0.0
            z_now = (r_["height"] if r_ and r_["height"] is not None else 0.0)
        else:
            pitch, roll = math.radians(mc.pitch_deg()), math.radians(mc.roll_deg())
            vel = np.zeros(6)
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_XBODY, h, vel, 1)
            p_rate, r_rate = -vel[1], vel[0]
            z_now = d.qpos[2] - z0
        zf += a_f * (z_now - zf)
        vzf += a_f * (d.qvel[2] - vzf)          # (speeds still true: an air-data sensor is not modelled yet)
        vxf += a_f * (math.hypot(d.qvel[0], d.qvel[1]) - vxf)
        th_ask = math.radians(min(max(AP["kh"] * (0.0 - zf) - AP["kvz"] * vzf, -AP["pitch_lim_deg"]), AP["pitch_lim_deg"]))
        w = w0 * (1.0 + min(max(AP["kf"] * (V - vxf), -AP["f_lim"]), AP["f_lim"]))
        e_f += a_f * ((pitch - th_ask) - e_f)
        r_f += a_f * (p_rate - r_f)
        e_i = min(max(e_i + TL["ki"] * e_f * mc.dt, -lim), lim)
        want = min(max(-(math.radians(TL["i0_deg"]) + TL["kp"] * e_f + TL["kd"] * r_f + e_i), -lim), lim)
        tail_q += min(max(want - tail_q, -TL["rate"] * mc.dt), TL["rate"] * mc.dt)
        s, c = math.sin(ph), math.cos(ph)
        q, qd = np.zeros(len(mc.aid)), np.zeros(len(mc.aid))
        q[st], qd[st] = 0.5 * Phi * s, 0.5 * Phi * w * c
        q[fe] = psi0
        q[ta] = tail_q
        dr = kr * math.degrees(roll) + krd * math.degrees(r_rate)
        tip = math.degrees(-Psi * c)
        mc.set_twist(tip + dr, tip - dr)
        mc.step(q, qd)
        if S is not None:
            S.sample()
        ph += w * mc.dt
        if k % n_beat == n_beat - 1:
            rec.append([mc.t, *d.qpos[:3].tolist(), mc.pitch_deg(), mc.roll_deg(), mc.yaw_deg(),
                        math.hypot(d.qvel[0], d.qvel[1]), math.degrees(tail_q), dr])
        if d.qpos[2] < 0.5 or not np.all(np.isfinite(d.qpos)):
            break
    s_ = mc.summary()
    return dict(point=list(pt[:4]), kr=kr, krd=krd, roll0=roll0, sensors=sensors, beats_flown=len(rec),
                bad_qacc=mc.bad_qacc, elec_W=s_["elec_W"], shaft_W=s_["shaft_W"], at_limit=s_["at_limit"],
                trace=rec)


def job(a):
    pt, kr, krd, roll0, sens = a
    try:
        return fly(pt, kr, krd, roll0=roll0, sensors=sens)
    except Exception as e:
        return dict(point=list(pt[:4]), kr=kr, krd=krd, error=repr(e))


def main() -> int:
    import c1_pilot
    pts = sorted(c1_pilot.points(), key=lambda p: p[4])
    cheap, thrusty = pts[0], [p for p in pts if p[4] > 10.0][0]
    grid = []
    for pt in (cheap, thrusty):
        for kr, krd in ((0.0, 0.0), (1.0, 0.1), (-1.0, -0.1)):
            grid.append((pt, kr, krd, 5.0, False))
    grid.append((thrusty, 1.0, 0.1, 5.0, True))
    with mp.get_context("spawn").Pool(4) as pool, open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in pool.imap_unordered(job, grid):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if "error" in r:
                print(r["point"], r["kr"], r["error"][:200])
                continue
            t = np.array(r["trace"])
            print(r["point"], "kr", r["kr"], "krd", r["krd"], "sens", r["sensors"], "beats", r["beats_flown"],
                  "end z %.1f pitch %.1f roll %.1f yaw %.1f V %.2f" % tuple(t[-1, [3, 4, 5, 6, 7]]),
                  "|roll| max %.1f" % np.abs(t[:, 5]).max(), "elec", {k: round(v, 1) for k, v in r["elec_W"].items()},
                  "qacc", r["bad_qacc"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
