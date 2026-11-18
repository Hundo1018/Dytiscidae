"""M1 probes run before PREREG_M1_takeoff.md was frozen (its section 0 lists them).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m1_probe.py geom|held|gear|side|pushup|attempt

Rows go to results/M1_probe.jsonl.  None of these decides a gate.
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
import m1_lib as L  # noqa: E402

OUT = HERE / "results" / "M1_probe.jsonl"


def emit(row):
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row))


# ------------------------------------------------------------------ P0 geometry
def geom():
    rows = []
    for x, yaw in ((17.0, 0.0), (17.0, 180.0), (17.0, 90.0), (30.0, 0.0)):
        mc = L.machine(x=x, yaw_deg=yaw)
        for _ in range(10000):
            mc.step(np.zeros(len(mc.aid)))
        m, d, mj = mc.m, mc.d, mc.mj
        c = L.com(mc)
        nose, left, back = L.axes(mc)
        R = np.stack([nose, left, back], 1)
        sh = np.array([d.xanchor[j] for j, r in zip(mc.jid, mc.role) if r == "stroke"]).mean(0)
        tail = d.xpos[[b for b in range(m.nbody) if mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b).startswith("seg4")][0]]
        # inertia about the dry CoM with the model's (inertial) masses
        I = np.zeros((3, 3))
        for b in range(1, m.nbody):
            r = d.xipos[b] - c
            Rb = d.ximat[b].reshape(3, 3)
            I += Rb @ np.diag(m.body_inertia[b]) @ Rb.T + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
        Ib = R.T @ I @ R
        # the wing's mass box against its fluid strips (D3's quarter chord)
        pan = mc.env.solver.panels
        b = int(mc.m.jnt_bodyid[mc.jid[list(mc.role).index("feather")]])
        from dytiscidae.physics.fluid import WING as KW
        sel = (pan.body_id == b) & (pan.kind == KW)
        Rw = d.xmat[b].reshape(3, 3)
        strip_c = (d.xpos[b] + (Rw @ pan.pos_local[sel].T).T).mean(0)
        n, bodies = L.world_contacts(mc)
        rows.append(dict(probe="P0 geom", x=x, yaw=yaw, pitch_deg=mc.pitch_deg(), load=mc.contact_load(),
                         bad_qacc=mc.bad_qacc, com_body=(R.T @ (c - d.xpos[mc.hull])).tolist(),
                         shoulder_from_com_body=(R.T @ (sh - c)).tolist(),
                         tail_hinge_from_com_body=(R.T @ (tail - c)).tolist(),
                         com_height_over_ground=float(c[2] - mc.env.ground_height(c[0])),
                         shoulder_height_over_ground=float(sh[2] - mc.env.ground_height(sh[0])),
                         inertia_body_diag=np.diag(Ib).tolist(),
                         wing_massbox_minus_strips_body=(R.T @ (d.xipos[b] - strip_c)).tolist(),
                         dry_mass=float(mc.env.solver._dry_mass.sum()), model_mass=float(m.body_mass.sum()),
                         contacts=n, touching=[mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, k) for k in bodies]))
    for r in rows:
        emit(r)


# ------------------------------------------------------------- P1 held authority
def held_job(a):
    name, pitch, kw, J_r, gear = a
    mc = L.machine(x=17.0, J_r=J_r, gear=gear)
    mc.place(z=30.0)
    L.set_attitude(mc, 0.0, pitch)
    q0 = mc.d.qpos[:7].copy()
    nb = int(round(1 / (L.K1["f"] * mc.dt)))
    acc, alp, err = [], [], []
    t0 = time.time()
    for k in range(6 * nb):
        q, qd = L.k1_targets(mc, mc.t, **kw)
        mc.step(q, qd)
        if k >= 2 * nb:
            acc.append(mc.d.qvel[:3] / mc.dt)
            alp.append(mc.d.qvel[3:6] / mc.dt)
            err.append(float(np.abs(mc.d.qpos[mc.qad] - q)[mc.role != "tail"].max()))
        mc.d.qpos[:7] = q0
        mc.d.qvel[:6] = 0.0
    a_ = np.mean(acc, 0)
    s = mc.summary()
    return dict(probe="P1 held", arm=name, pitch=pitch, kw=kw, J_r=J_r, gear=gear,
                az_over_g_plus1=float(a_[2] / L.G + 1.0), ax_over_g=float(a_[0] / L.G),
                alpha_body=np.mean(alp, 0).tolist(), track_err_deg=math.degrees(max(err)),
                elec_W=s["elec_W"], shaft_W=s["shaft_W"], at_limit=s["at_limit"], bad_qacc=mc.bad_qacc,
                wall_s=time.time() - t0)


def held():
    jobs = [("K1", 90.0, {}, 2e-5, 24.0), ("K1", 45.0, {}, 2e-5, 24.0),
            ("bias+0.2", 90.0, dict(bias=0.2), 2e-5, 24.0),
            ("amp_lr+0.2", 90.0, dict(amp_lr=0.2), 2e-5, 24.0),
            ("psi0_lr+0.2", 90.0, dict(psi0_lr=0.2), 2e-5, 24.0),
            ("psi0+0.2", 90.0, dict(psi0=0.2), 2e-5, 24.0),
            ("amp1.15", 90.0, dict(amp=1.15), 2e-5, 24.0),
            ("f5.2", 90.0, dict(f=5.2), 2e-5, 24.0)]
    with mp.Pool(4) as p:
        for r in p.imap(held_job, jobs):
            emit(r)


def gear():
    jobs = [("K1", 90.0, {}, J, g) for J in (2e-5, 5e-6, 1e-6) for g in (16.0, 24.0, 32.0)]
    with mp.Pool(4) as p:
        for r in p.imap(held_job, jobs):
            emit(r)


# ------------------------------------------------------------------ P2 push-up
def pushup_job(a):
    name, ramp_s = a
    mc = L.machine(x=17.0)
    for _ in range(5000):
        mc.step(np.zeros(len(mc.aid)))
    q_end = -1.35
    trace = []
    T = 1.5
    n = int(T / mc.dt)
    for k in range(n):
        t = k * mc.dt
        s = min(t / ramp_s, 1.0) if ramp_s > 0 else 1.0
        q = np.zeros(len(mc.aid))
        q[mc.role == "stroke"] = q_end * s
        mc.step(q)
        if k % 500 == 0:
            c = L.com(mc)
            nc, bodies = L.world_contacts(mc)
            trace.append([round(t, 3), round(mc.pitch_deg(), 1), round(float(c[2] - mc.env.ground_height(c[0])), 3),
                          round(float(np.degrees(mc.d.qpos[mc.qad[mc.role == "stroke"]].mean())), 1),
                          round(mc.contact_load(), 2), bodies])
    s = mc.summary()
    return dict(probe="P2 pushup", arm=name, ramp_s=ramp_s, trace=trace, at_limit=s["at_limit"],
                elec_W=s["elec_W"], bad_qacc=mc.bad_qacc, max_force_stroke=None)


def pushup():
    with mp.Pool(2) as p:
        for r in p.imap(pushup_job, [("slow", 0.5), ("step", 0.0)]):
            emit(r)


# ------------------------------------------- P3 open-loop attempts (disclosed as data)
def attempt_job(kind):
    mc = L.machine(x=17.0)
    for _ in range(5000):
        mc.step(np.zeros(len(mc.aid)))
    c0 = L.com(mc)
    st = mc.role == "stroke"
    trace = []
    T, start = 3.0, None
    for k in range(int(T / mc.dt)):
        t = k * mc.dt
        q = np.zeros(len(mc.aid))
        qd = np.zeros(len(mc.aid))
        if start is None:
            q[st] = -1.35 * (min(t / 0.5, 1.0) if kind == "pushup" else 1.0)
            ang = float(mc.d.qpos[mc.qad[st]].mean())
            if (kind == "pushup" and (mc.pitch_deg() >= 35.0 or t >= 2.0)) or (kind == "vault" and ang <= -1.2):
                start = t
        if start is not None:
            b = min((t - start) / 0.1, 1.0)
            qk, qdk = L.k1_targets(mc, t - start, phase0=-math.pi / 2)  # begin at the bottom of the stroke
            q = (1 - b) * np.where(st, -1.35, 0.0) + b * qk
            qd = b * qdk
        mc.step(q, qd)
        if k % 1000 == 0:
            c = L.com(mc)
            nc, bodies = L.world_contacts(mc)
            trace.append([round(t, 2), round(mc.pitch_deg(), 1), round(float(c[2] - c0[2]), 3),
                          round(float(c[0] - c0[0]), 3), round(mc.contact_load(), 2), nc])
    s = mc.summary()
    return dict(probe="P3 open-loop attempt", kind=kind, start_s=start, trace=trace, at_limit=s["at_limit"],
                elec_W=s["elec_W"], bad_qacc=mc.bad_qacc)


def attempt():
    with mp.Pool(2) as p:
        for r in p.imap(attempt_job, ["pushup", "vault"]):
            emit(r)


# ------------------------------------------- P1b where the biased stroke's side force comes from
def side_job(a):
    name, kw, vacuum = a
    mc = L.machine(x=17.0)
    mc.place(z=30.0)
    L.set_attitude(mc, 0.0, 90.0)
    if vacuum:
        mc.env.solver.apply = lambda d, t: None
    q0 = mc.d.qpos[:7].copy()
    nb = int(round(1 / (L.K1["f"] * mc.dt)))
    acc, ffl = [], []
    comp = (mc.m.body_mass - mc.env.solver._dry_mass)[:, None] * np.array([0, 0, L.G])
    for k in range(6 * nb):
        q, qd = L.k1_targets(mc, mc.t, **kw)
        mc.step(q, qd)
        if k >= 2 * nb:
            acc.append(mc.d.qvel[:3] / mc.dt)
            ffl.append((mc.d.xfrc_applied[:, :3] - (0 if vacuum else comp)).sum(0))
        mc.d.qpos[:7] = q0
        mc.d.qvel[:6] = 0.0
    W = mc.mass * L.G
    return dict(probe="P1b side force", arm=name, kw=kw, vacuum=vacuum,
                acc_over_g=(np.mean(acc, 0) / L.G).tolist(),
                xfrc_fluid_over_W=(np.mean(ffl, 0) / W).tolist(), bad_qacc=mc.bad_qacc)


def side():
    jobs = [("bias+0.2", dict(bias=0.2), False), ("bias+0.2 vacuum", dict(bias=0.2), True),
            ("bias-0.2", dict(bias=-0.2), False), ("K1", {}, False)]
    with mp.Pool(4) as p:
        for r in p.imap(side_job, jobs):
            emit(r)


if __name__ == "__main__":
    {"geom": geom, "held": held, "gear": gear, "side": side, "pushup": pushup,
     "attempt": attempt}[sys.argv[1]]()
