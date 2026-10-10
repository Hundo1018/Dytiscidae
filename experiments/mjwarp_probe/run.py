#!/usr/bin/env python3
"""N9 -- MuJoCo Warp for the same-body work (ROADMAP 2026-10-10 item N9; ARCH51_SPEC §B N9).

Frozen prediction (verbatim): "*Prediction:* the rigid-body part is >= 5x faster
at 24 worlds; the fluid interop is the cost that decides it. *Falsified* if < 2x."

Runs in a SCRATCH venv that never imports the project (the XML is the interface):

    # project venv, writes elite_median_dof.xml / elite_rotor_median_dof.xml
    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/mjwarp_probe/export_mjcf.py experiments/mjwarp_probe [rotor]
    # scratch venv (mujoco 3.15 + mujoco-warp 3.15 + warp-lang 1.18), CPU baseline and MJWarp
    ../mjwarp-venv/bin/python experiments/mjwarp_probe/run.py cpu  <xml> <out.json>
    ../mjwarp-venv/bin/python experiments/mjwarp_probe/run.py warp <xml> <out.json>
    # project venv, CPU baseline with the mujoco the search actually uses (3.11.0)
    .venv/bin/python experiments/mjwarp_probe/run.py cpu <xml> <out.json>   # (no project import)

Modes: gravity off (flight: no contacts, the air-identification shape) and gravity on
(the body falls 19.5 m onto the seabed plane at ~step 500, so contacts are live after).
Every arm uses the same static ctrl (+0.3 rad on every actuator); 3 repeats, medians.
Rigid body only: no fluid forces, no rotor thrust (those are the Mojo kernel's).
"""
import json, statistics, sys, time
import numpy as np
import mujoco

mode, xml, out = sys.argv[1], sys.argv[2], sys.argv[3]
STEPS = 2000
REPS = 3
NJMAX = 512        # MJWarp's default (64 per world) overflows once the body lands: 'nefc overflow' truncates constraints


def load(margin0):
    m = mujoco.MjModel.from_xml_path(xml)
    if margin0:                                   # MJWarp 3.15 rejects geom margin 0.001 on box/mesh pairs
        m.geom_margin[:] = 0.0
    return m


def med(xs):
    return statistics.median(xs)


res = {"mode": mode, "xml": xml, "mujoco": mujoco.__version__, "steps": STEPS, "reps": REPS}


def cpu_rollouts(mjm, n_roll, steps, gravity_on):
    mjm.opt.gravity[2] = -9.80665 if gravity_on else 0.0
    d = mujoco.MjData(mjm)
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        for _w in range(n_roll):
            mujoco.mj_resetData(mjm, d)
            d.ctrl[:] = 0.3
            for _ in range(steps):
                mujoco.mj_step(mjm, d)
        ts.append(time.perf_counter() - t)
    return ts


if mode == "cpu":
    mjm = load(margin0=(len(sys.argv) > 4 and sys.argv[4] == "margin0"))
    res["margin0"] = bool(len(sys.argv) > 4 and sys.argv[4] == "margin0")
    res["nbody"], res["nv"], res["nu"] = int(mjm.nbody), int(mjm.nv), int(mjm.nu)
    for g in (False, True):
        for n_roll, steps in ((24, STEPS), (48, 300)):
            ts = cpu_rollouts(mjm, n_roll, steps, g)
            res[f"cpu_g{int(g)}_{n_roll}x{steps}"] = {"wall_s": ts, "median_s": med(ts),
                                                       "steps_per_s": n_roll * steps / med(ts)}
            print(f"CPU {mujoco.__version__} gravity={g} {n_roll} sequential x {steps}: median {med(ts):.3f} s "
                  f"({n_roll*steps/med(ts):.0f} steps/s)  reps {[round(x,3) for x in ts]}", flush=True)
    # reference trajectory for the agreement check: flight, 300 steps, ctrl 0.3
    mjm.opt.gravity[2] = -9.80665
    d = mujoco.MjData(mjm); d.ctrl[:] = 0.3
    for _ in range(300):
        mujoco.mj_step(mjm, d)
    res["ref_qpos_300"] = d.qpos.tolist()

elif mode == "warp":
    import warp as wp, mujoco_warp as mjw
    res["warp"] = wp.config.version if hasattr(wp.config, "version") else None
    res["device"] = str(wp.get_device())
    mjm = load(margin0=True)
    res["nbody"], res["nv"], res["nu"] = int(mjm.nbody), int(mjm.nv), int(mjm.nu)
    # per-world model fields (the added-mass write of batchroll.py:529-531 needs this)
    m_pw = mjw.put_model(mjm, batch_sizes={"body_mass": 24, "body_inertia": 24})
    res["per_world_body_mass_shape"] = list(m_pw.body_mass.shape)
    res["per_world_body_inertia_shape"] = list(m_pw.body_inertia.shape)
    print("per-world body_mass shape", m_pw.body_mass.shape, "body_inertia", m_pw.body_inertia.shape, flush=True)
    del m_pw

    def make(nworld, gravity_on):
        mjm.opt.gravity[2] = -9.80665 if gravity_on else 0.0
        mjd = mujoco.MjData(mjm)
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=nworld, njmax=NJMAX)
        d.ctrl.assign(np.full((nworld, mjm.nu), 0.3, dtype=np.float32))
        return m, d

    def timed(f, steps):
        wp.synchronize(); t = time.perf_counter(); f(steps); wp.synchronize()
        return time.perf_counter() - t

    # agreement: flight, 300 steps, world 0 vs the CPU reference written by `cpu` mode in this venv
    m, d = make(1, True)
    t = time.perf_counter(); mjw.step(m, d); wp.synchronize(); res["compile_plus_first_step_s"] = time.perf_counter() - t
    for _ in range(299):
        mjw.step(m, d)
    wp.synchronize()
    res["warp_qpos_300"] = d.qpos.numpy()[0].tolist()

    for g in (False, True):
        for nworld in (1, 24, 48, 96):
            for graph in ((False, True) if nworld == 24 else (True,)):   # the un-graphed loop is launch-bound and independent of nworld
                m, d = make(nworld, g)
                mjw.step(m, d); wp.synchronize()                       # warm-up (compile already cached)
                cap = None
                if graph:
                    with wp.ScopedCapture() as cap:
                        mjw.step(m, d)
                def run(n):
                    if graph:
                        for _ in range(n):
                            wp.capture_launch(cap.graph)
                    else:
                        for _ in range(n):
                            mjw.step(m, d)
                ts = []
                for _ in range(REPS):
                    mjw.reset_data(m, d); d.ctrl.assign(np.full((nworld, mjm.nu), 0.3, dtype=np.float32))
                    ts.append(timed(run, STEPS))
                key = f"warp_g{int(g)}_w{nworld}_{'graph' if graph else 'nograph'}"
                sps = nworld * STEPS / med(ts)
                res[key] = {"wall_s": ts, "median_s": med(ts), "world_steps_per_s": sps, "steps_per_s_per_world": STEPS / med(ts)}
                print(f"{key}: {STEPS} steps median {med(ts):.3f} s  -> {sps:.0f} world-steps/s  reps {[round(x,3) for x in ts]}", flush=True)

    # (D) interop proxy, graph-captured step: per step upload ctrl and xfrc_applied, replay the step graph,
    # read back xpos, xmat, xipos, cvel (the arrays BatchedFluid.launch reads on the host).
    for nworld in (24, 48):
        m, d = make(nworld, False)
        mjw.step(m, d); wp.synchronize()
        with wp.ScopedCapture() as cap:
            mjw.step(m, d)
        ctrl = np.full((nworld, mjm.nu), 0.3, dtype=np.float32)
        xfrc = np.zeros((nworld, mjm.nbody, 6), dtype=np.float32)
        def run(n):
            for _ in range(n):
                d.ctrl.assign(ctrl); d.xfrc_applied.assign(xfrc)
                wp.capture_launch(cap.graph)
                _ = (d.xpos.numpy(), d.xmat.numpy(), d.xipos.numpy(), d.cvel.numpy())
        def run2(n):
            for _ in range(n):
                wp.capture_launch(cap.graph)
        n_i = 300
        ts, ts2 = [], []
        for _ in range(REPS):
            mjw.reset_data(m, d); ts.append(timed(run, n_i))
            mjw.reset_data(m, d); ts2.append(timed(run2, n_i))
        res[f"warp_interop_w{nworld}_{n_i}"] = {"wall_s": ts, "median_s": med(ts), "world_steps_per_s": nworld * n_i / med(ts)}
        res[f"warp_nointerop_w{nworld}_{n_i}"] = {"wall_s": ts2, "median_s": med(ts2), "world_steps_per_s": nworld * n_i / med(ts2)}
        print(f"interop (graph) w{nworld} x{n_i}: with host traffic {med(ts):.3f} s, without {med(ts2):.3f} s", flush=True)

json.dump(res, open(out, "w"), indent=1)
print("wrote", out)
