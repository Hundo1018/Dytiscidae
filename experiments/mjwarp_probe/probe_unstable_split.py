# Device "unstable" (bad_qacc) on every draw, 2026-10-26: locate and ablate.
# Builds the worlds as evaluate_tier1_device does (body_worlds -> DeviceSegments -> arm), one
# seed, then steps by hand with DeviceSegments.step split open: everything before mjw.step, a
# hook that restores ONE write to its dry/zero value on the target worlds, then mjw.step and
# the rest of the step unchanged. Per step it reads qacc for each world and records the first
# step with a non-finite qacc and the first with |qacc| > 1e10 (check_record codes both 3).
# At the first bad step it dumps the solver inputs of that world and replays them on host
# MuJoCo (same mjm copy, per-world mass/inertia/damping, qpos/qvel/ctrl/xfrc/qfrc/warmstart,
# mj_forward) as the discriminator: inputs bad vs MJWarp bad.
# No source file is changed. Usage: probe_unstable_split.py <plan> <medium> <seed> <nsteps>
import sys
import numpy as np, warp as wp, mujoco
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.devroll import rollout_wp as R, model_wp as Mw
from dytiscidae.evolution.loop import placement_draw_seed
from tests.test_devroll_agreement import _a4_controller

plan, medium = sys.argv[1], sys.argv[2]
seed, nsteps = int(sys.argv[3]), int(sys.argv[4])
modes = sys.argv[5].split(",") if len(sys.argv) > 5 else [
    "none", "all", "mass", "damping", "xfrc", "qfrc", "ctrl"]
R.repeatable(True)
p = build(BODY_PLANS[plan]())
ctrl, _ = _a4_controller(p, 171)


def build_seg():
    envs, doms, params, pols, bases, index = R.body_worlds(p, ctrl, [seed])
    seg = R.DeviceSegments(envs, seconds=8.0, dtype=wp.float32, policies=pols, bases=bases)
    noise = [placement_draw_seed(seed, 101 + j) for j in range(2 * len(R.MEDIA))]
    seg.arm(doms, params, sampled=[0] * len(envs), seeds=noise)
    return seg, [index[(0, medium, h)] for h in (0, 1)]


def run(mode, dump=False):
    seg, targets = build_seg()
    body, ctl, d, m = seg.body, seg.ctl, seg.body.d, seg.body.m
    K = R.make_kernels(seg.dtype)
    Kb = Mw.make_kernels(seg.dtype)
    dry = {"bm": m.body_mass.numpy().copy(), "bi": m.body_inertia.numpy().copy(),
           "dd": m.dof_damping.numpy().copy()}

    def restore(arr, key=None):
        a = arr.numpy()
        for w in targets:
            a[w] = dry[key][w] if key else 0
        arr.assign(a)

    first_nan = {w: None for w in targets}
    first_big = {w: None for w in targets}
    peak = {w: 0.0 for w in targets}
    for i in range(nsteps):
        seg.decide()
        body.gather()
        body.fluid_and_scatter()
        body.hold(ctl.active)
        if mode in ("mass", "all"):
            restore(m.body_mass, "bm"); restore(m.body_inertia, "bi")
        if mode in ("damping", "all"):
            restore(m.dof_damping, "dd")
        if mode in ("xfrc", "all"):
            restore(d.xfrc_applied)
        if mode in ("qfrc", "all"):
            restore(d.qfrc_applied)
        if mode in ("ctrl", "all"):
            restore(d.ctrl)
        if mode == "subtree":
            # not an ablation: body_subtreemass kept consistent with this step's body_mass
            bmw = m.body_mass.numpy().astype(float)
            sub = bmw.copy()
            par = body.mjm.body_parentid
            for b in range(body.mjm.nbody - 1, 0, -1):
                sub[:, par[b]] += sub[:, b]
            if m.body_subtreemass.shape[0] != seg.W:
                m.body_subtreemass = wp.array(sub.astype(np.float32), dtype=wp.float32)
            else:
                m.body_subtreemass.assign(sub.astype(np.float32))
        alive_before = ctl.active.numpy().copy()
        inp = None
        if dump:
            inp = {k: getattr(src, k).numpy().copy() for src, ks in (
                (d, ("qpos", "qvel", "ctrl", "xfrc_applied", "qfrc_applied", "qacc_warmstart")),
                (m, ("body_mass", "body_inertia", "dof_damping"))) for k in ks}
        Mw._mjw.step(m, d)
        body._charge()
        wp.launch(Kb["count_tick"], dim=seg.W, inputs=[body.count])
        wp.launch(K["count_contacts"], dim=int(d.contact.worldid.shape[0]), inputs=[
            d.nacon, d.contact.worldid, ctl.active, seg.contacts])
        ctl.tick()
        ctl.sense()
        seg.check()
        qacc = d.qacc.numpy()
        newly = []
        for w in targets:
            if not alive_before[w]:
                continue
            row = qacc[w]
            fin = np.isfinite(row).all()
            mx = float(np.abs(row[np.isfinite(row)]).max(initial=0.0))
            peak[w] = max(peak[w], mx)
            if not fin and first_nan[w] is None:
                first_nan[w] = i; newly.append(w)
            if (not fin or mx > 1e10) and first_big[w] is None:
                first_big[w] = i; newly.append(w)
        if dump and newly:
            dump_world(seg, inp, sorted(set(newly))[0], i, qacc)
            dump = False
        wp.launch(K["restore_dead"], dim=seg.W, inputs=[
            body.mjm.nq, body.nv, ctl.active, seg.q0, d.qpos, d.qvel])
        if all(first_big[w] is not None for w in targets):
            break
    fail = seg.fail.numpy()
    for w in targets:
        print(f"RES {plan} {medium} seed={seed} mode={mode:7s} world={w} first_nonfinite="
              f"{first_nan[w]} first_bad(nonfinite|>1e10)={first_big[w]} "
              f"peak_finite_|qacc|={peak[w]:.3g} fail_code={int(fail[w])}", flush=True)


def dump_world(seg, inp, w, i, qacc_dev):
    mjm = seg.body.mjm
    import os
    out = os.environ.get("DUMP_NPZ")
    if out:
        np.savez(out, step=i, world=w, **{k: np.asarray(v[w]) for k, v in inp.items()})
    print(f"DUMP {plan} {medium} world={w} step={i} count={int(seg.body.count.numpy()[w])} "
          f"ndec={int(seg.ctl.ndec.numpy()[w])} due={int(seg.ctl.due.numpy()[w])}", flush=True)
    for k, a in inp.items():
        r = np.asarray(a[w], float)
        print(f"DUMP   {k:15s} finite={np.isfinite(r).all()} min={np.nanmin(r):.4g} "
              f"max={np.nanmax(r):.4g} maxabs={np.nanmax(np.abs(r)):.4g}", flush=True)
    bad = np.where(~np.isfinite(qacc_dev[w]) | (np.abs(qacc_dev[w]) > 1e10))[0]
    print(f"DUMP   device qacc bad dofs={bad.tolist()} vals={qacc_dev[w][bad][:6].tolist()} "
          f"dof_bodyid={mjm.dof_bodyid[bad].tolist()}", flush=True)
    # host discriminator
    import copy
    hm = copy.deepcopy(mjm)
    hm.body_mass[:] = inp["body_mass"][w]
    hm.body_inertia[:] = inp["body_inertia"][w]
    hm.dof_damping[:] = inp["dof_damping"][w]
    hd = mujoco.MjData(hm)
    hd.qpos[:] = inp["qpos"][w]; hd.qvel[:] = inp["qvel"][w]; hd.ctrl[:] = inp["ctrl"][w]
    hd.xfrc_applied[:] = np.asarray(inp["xfrc_applied"][w], float).reshape(-1, 6)
    hd.qfrc_applied[:] = inp["qfrc_applied"][w]
    hd.qacc_warmstart[:] = inp["qacc_warmstart"][w]
    mujoco.mj_forward(hm, hd)
    h = hd.qacc
    hb = np.where(~np.isfinite(h) | (np.abs(h) > 1e10))[0]
    print(f"DUMP   host mj_forward on the same inputs: finite={np.isfinite(h).all()} "
          f"maxabs={np.nanmax(np.abs(h)):.4g} bad dofs={hb.tolist()} "
          f"warnings_badqacc={hd.warning[mujoco.mjtWarning.mjWARN_BADQACC].number}", flush=True)
    # mass-matrix conditioning on the host for those inputs
    M = np.zeros((hm.nv, hm.nv)); mujoco.mj_fullM(hm, hd, M)
    ev = np.linalg.eigvalsh(M)
    print(f"DUMP   host qM eig min={ev.min():.4g} max={ev.max():.4g} cond={ev.max() / max(ev.min(), 1e-300):.3g}",
          flush=True)


for k, mode in enumerate(modes):
    run(mode, dump=(k == 0))
