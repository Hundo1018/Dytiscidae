"""D1, second part: (a) the same kicked hover with the entrainment reaction OFF (entrainment=False), from the
float, in air at 10 m, and in air without a kick; (b) the size of the air added-mass reaction on a held hover:
hull held level in air (30 m), the hover stroke, hull moved at +-0.3 m/s along x / y, the three modes.
Rows -> results/S0b_probes4.jsonl (ids d1b_*).
"""
from __future__ import annotations
import json, math, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "S0b_probes4.jsonl"
MODES = (("default", True), ("wetting", "wetting"), ("off", False))


def held_air(job):
    import s0b_t5_probe as P
    import s0b_machine as T
    import s0b_laws as LW
    import s0b_probe2 as P2
    t0 = time.time()
    cfg = P.law_cfg(gains=dict(wn=(8.0, 8.0, 4.0), zeta=0.8, k_v=0.5, k_p=0.1))
    Pl = dict(LW.LEVEL, **cfg["P"])
    Pl["amp"] = job.get("amp", cfg["a0"])
    f = cfg["f"]
    mc = P._make("water", 0.0, "x4", 1e-4, z=30.0)
    d, s = mc.d, mc.env.solver
    s.entrainment = job["entrainment"]
    q0 = d.qpos[:7].copy(); q0[3:7] = (1.0, 0.0, 0.0, 0.0); d.qpos[:7] = q0; d.qvel[:] = 0.0
    qr, _ = LW.targets_level(mc, 0.0, 2 * math.pi * f, Pl, osc=0.0)
    d.qpos[mc.qad] = qr
    mc.mj.mj_forward(mc.m, d)
    T.regain(mc); s.reset()
    w = 2 * math.pi * f
    nb = int(round(1.0 / (f * mc.dt)))
    beats, skip = 7, 3
    U = np.array(job.get("U", (0, 0, 0)), float)
    F, M, t = [], [], 0.0
    for k in range(beats * nb):
        osc = P2._smooth(t * f / 2.0)
        q, qd = LW.targets_level(mc, w * t, w, Pl, osc=osc)
        d.qpos[:7] = q0; d.qpos[:3] = q0[:3] + U * t; d.qvel[:6] = 0.0; d.qvel[:3] = U
        mc.step(q, qd); t += mc.dt
        if k >= skip * nb:
            Fk, Mk = T.fluid_wrench(mc); F.append(Fk); M.append(Mk)
    F, M = np.array(F), np.array(M)
    R = d.xmat[mc.hull].reshape(3, 3)
    Fb = (F @ R).reshape(beats - skip, nb, 3)
    return dict(job, kind="held_air", F_hull=(F.mean(0) @ R).tolist(), M_hull=(M.mean(0) @ R).tolist(),
                F_hull_sd_in_beat=Fb.std(1).mean(0).tolist(), bad_qacc=int(mc.bad_qacc), wall_s=round(time.time() - t0, 1),
                machine="T5", clamped=int(s.diag.clamped))


def mission_row(job):
    import s0b_t5_probe as P
    P.Z_FLOAT = job.get("z_float", 0.165)
    return P._dispatch(job)


def _run(job):
    return held_air(job) if job["fn"] == "held_air" else mission_row(job)


def jobs():
    import s0b_t5_probe as P
    law = P.law_cfg(gains=dict(wn=(8.0, 8.0, 4.0), zeta=0.8, k_v=0.5, k_p=0.1))
    J = []
    for where, zf, kick in (("air10", 10.165, True), ("float", None, True), ("air10", 10.165, False)):
        fly = dict(kind="fly", H=2.0, hold_s=10.0, land=True, after=5.0)
        if kick:
            fly["kick"] = {"roll_rate": 1.0, "vy": 0.3}
        j = dict(fn="mission", probe="D1", id=f"d1b_{where}_{'kick' if kick else 'nokick'}_off", mode="off", entrainment=False,
                 start="float", law=law, motor="x4", phases=[dict(kind="settle", T=3.0 if where == "float" else 0.0), fly])
        if zf:
            j["z_float"] = zf
        J.append(j)
    for mode, ent in MODES:
        for nm, U in (("U0", (0, 0, 0)), ("Ux+", (0.3, 0, 0)), ("Ux-", (-0.3, 0, 0)), ("Uy+", (0, 0.3, 0)), ("Uy-", (0, -0.3, 0))):
            J.append(dict(fn="held_air", probe="D1", id=f"d1b_held_{mode}_{nm}", mode=mode, entrainment=ent, U=U))
    return J


if __name__ == "__main__":
    with mp.get_context("spawn").Pool(4) as pool:
        for r in pool.imap_unordered(_run, jobs(), chunksize=1):
            with open(OUT, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
            print(r["id"], r.get("abort"), r["wall_s"], flush=True)
